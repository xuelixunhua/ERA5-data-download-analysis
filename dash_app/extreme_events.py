"""
极端事件与ENSO关联分析模块

功能：
1. 极端事件检测（百分位法、Z-score法、变异系数法）
2. ENSO阶段分类与关联分析
3. 滞后相关性分析
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ExtremeConfig:
    """极端事件检测配置"""
    method: str = 'percentile'  # percentile / zscore / cv

    # 百分位法
    upper_percentile: float = 95.0
    lower_percentile: float = 5.0

    # Z-score法
    zscore_threshold: float = 2.0

    # 变异系数法
    cv_threshold: float = 0.3

    # 基准期
    baseline_start: int = 1995
    baseline_end: int = 2024

    # 是否使用滑动窗口
    use_rolling: bool = False
    rolling_window: int = 10


@dataclass
class ENSOConfig:
    """ENSO分析配置"""
    el_nino_threshold: float = 0.5
    la_nina_threshold: float = -0.5

    # 滞后期
    lag_months: int = 0
    auto_find_best_lag: bool = False
    max_lag_search: int = 12

    # 距平计算
    anomaly_type: str = 'monthly'  # monthly / seasonal / annual


# ============================================================================
# 数据加载层
# ============================================================================

def load_oni_data(file_path: str) -> pd.DataFrame:
    """
    加载ONI数据

    Returns:
    --------
    DataFrame with columns: ['年', '月', 'ONI', 'ANOM']
    """
    df = pd.read_excel(file_path)

    # 重命名列
    df = df.rename(columns={
        'YR': '年',
        'MON': '月',
        'ANOM': 'ONI'  # 使用ANOM列作为ONI指数
    })

    # 只保留需要的列
    df = df[['年', '月', 'ONI']].copy()

    return df


def merge_oni_with_climate(oni_df: pd.DataFrame, climate_df: pd.DataFrame,
                           lag_months: int = 0) -> pd.DataFrame:
    """
    合并ONI和气象数据，支持滞后

    Parameters:
    -----------
    lag_months : int
        滞后月数，正数表示ONI领先气象数据
        例如 lag=3: ONI(2020-01) 对应 Climate(2020-04)
    """
    oni = oni_df.copy()
    climate = climate_df.copy()

    if lag_months != 0:
        # 计算滞后后的年月
        climate['年月偏移'] = pd.to_datetime(
            climate['年'].astype(str) + '-' + climate['月'].astype(str).str.zfill(2)
        ) - pd.DateOffset(months=lag_months)

        climate['年_oni'] = climate['年月偏移'].dt.year
        climate['月_oni'] = climate['年月偏移'].dt.month

        # 合并
        merged = pd.merge(
            climate,
            oni,
            left_on=['年_oni', '月_oni'],
            right_on=['年', '月'],
            suffixes=('', '_oni')
        )

        # 清理临时列
        merged = merged.drop(columns=['年月偏移', '年_oni', '月_oni', '年_oni', '月_oni'])
    else:
        merged = pd.merge(
            climate,
            oni,
            on=['年', '月'],
            how='inner'
        )

    return merged


# ============================================================================
# 极端事件检测层
# ============================================================================

def detect_extreme_percentile(df: pd.DataFrame, variable: str,
                              config: ExtremeConfig) -> pd.DataFrame:
    """
    百分位法检测极端事件
    """
    result = df.copy()

    for province in df['province'].unique():
        df_prov = df[df['province'] == province].copy()

        # 计算基准期数据
        baseline = df_prov[
            (df_prov['年'] >= config.baseline_start) &
            (df_prov['年'] <= config.baseline_end)
            ]

        # 按月份分组计算分位数
        for month in range(1, 13):
            baseline_month = baseline[baseline['月'] == month][variable]

            if len(baseline_month) == 0:
                continue

            upper_threshold = np.percentile(baseline_month, config.upper_percentile)
            lower_threshold = np.percentile(baseline_month, config.lower_percentile)

            # 标记极端事件
            mask = (result['province'] == province) & (result['月'] == month)
            result.loc[mask, 'extreme_high'] = result.loc[mask, variable] > upper_threshold
            result.loc[mask, 'extreme_low'] = result.loc[mask, variable] < lower_threshold
            result.loc[mask, 'upper_threshold'] = upper_threshold
            result.loc[mask, 'lower_threshold'] = lower_threshold

    # 合并标记
    result['is_extreme'] = result['extreme_high'] | result['extreme_low']
    result['extreme_type'] = 'normal'
    result.loc[result['extreme_high'], 'extreme_type'] = 'high'
    result.loc[result['extreme_low'], 'extreme_type'] = 'low'

    return result


def detect_extreme_zscore(df: pd.DataFrame, variable: str,
                          config: ExtremeConfig) -> pd.DataFrame:
    """
    Z-score法检测极端事件
    """
    result = df.copy()

    for province in df['province'].unique():
        df_prov = df[df['province'] == province].copy()

        # 计算基准期数据
        baseline = df_prov[
            (df_prov['年'] >= config.baseline_start) &
            (df_prov['年'] <= config.baseline_end)
            ]

        # 按月份分组计算均值和标准差
        for month in range(1, 13):
            baseline_month = baseline[baseline['月'] == month][variable]

            if len(baseline_month) == 0:
                continue

            mean_val = baseline_month.mean()
            std_val = baseline_month.std()

            if std_val == 0:
                continue

            # 计算Z-score
            mask = (result['province'] == province) & (result['月'] == month)
            result.loc[mask, 'zscore'] = (result.loc[mask, variable] - mean_val) / std_val

    # 标记极端事件
    result['extreme_high'] = result['zscore'] > config.zscore_threshold
    result['extreme_low'] = result['zscore'] < -config.zscore_threshold
    result['is_extreme'] = result['extreme_high'] | result['extreme_low']
    result['extreme_type'] = 'normal'
    result.loc[result['extreme_high'], 'extreme_type'] = 'high'
    result.loc[result['extreme_low'], 'extreme_type'] = 'low'

    return result


def detect_extreme_cv(df: pd.DataFrame, variable: str,
                      config: ExtremeConfig) -> pd.DataFrame:
    """
    变异系数法检测极端事件
    """
    result = df.copy()

    for province in df['province'].unique():
        df_prov = df[df['province'] == province].copy()

        baseline = df_prov[
            (df_prov['年'] >= config.baseline_start) &
            (df_prov['年'] <= config.baseline_end)
            ]

        for month in range(1, 13):
            baseline_month = baseline[baseline['月'] == month][variable]

            if len(baseline_month) == 0:
                continue

            mean_val = baseline_month.mean()
            std_val = baseline_month.std()

            if mean_val == 0:
                continue

            cv = std_val / abs(mean_val)

            # 如果变异系数高，则用更严格的标准
            if cv > config.cv_threshold:
                threshold_multiplier = 1.5
            else:
                threshold_multiplier = 1.0

            upper_threshold = mean_val + threshold_multiplier * std_val
            lower_threshold = mean_val - threshold_multiplier * std_val

            mask = (result['province'] == province) & (result['月'] == month)
            result.loc[mask, 'extreme_high'] = result.loc[mask, variable] > upper_threshold
            result.loc[mask, 'extreme_low'] = result.loc[mask, variable] < lower_threshold
            result.loc[mask, 'cv'] = cv

    result['is_extreme'] = result['extreme_high'] | result['extreme_low']
    result['extreme_type'] = 'normal'
    result.loc[result['extreme_high'], 'extreme_type'] = 'high'
    result.loc[result['extreme_low'], 'extreme_type'] = 'low'

    return result


def detect_extreme_events(df: pd.DataFrame, variable: str,
                          config: ExtremeConfig) -> pd.DataFrame:
    """
    统一接口：根据配置检测极端事件
    """
    if config.method == 'percentile':
        return detect_extreme_percentile(df, variable, config)
    elif config.method == 'zscore':
        return detect_extreme_zscore(df, variable, config)
    elif config.method == 'cv':
        return detect_extreme_cv(df, variable, config)
    else:
        raise ValueError(f"Unknown method: {config.method}")


# ============================================================================
# ENSO分析层
# ============================================================================

def classify_enso_phase(oni_df: pd.DataFrame, config: ENSOConfig) -> pd.DataFrame:
    """
    分类ENSO阶段

    Returns:
    --------
    DataFrame with additional column 'enso_phase': 'El Niño' / 'La Niña' / 'Neutral'
    """
    df = oni_df.copy()

    df['enso_phase'] = 'Neutral'
    df.loc[df['ONI'] >= config.el_nino_threshold, 'enso_phase'] = 'El Niño'
    df.loc[df['ONI'] <= config.la_nina_threshold, 'enso_phase'] = 'La Niña'

    return df


def find_best_lag(oni_df: pd.DataFrame, climate_df: pd.DataFrame,
                  variable: str, province: str, max_lag: int = 12) -> Tuple[int, float]:
    """
    自动寻找最佳滞后期（最大相关系数）

    Returns:
    --------
    (best_lag, best_corr)
    """
    correlations = []

    for lag in range(0, max_lag + 1):
        merged = merge_oni_with_climate(oni_df, climate_df, lag)

        if province != '全部':
            merged = merged[merged['province'] == province]

        if len(merged) < 10:  # 数据太少，跳过
            correlations.append((lag, np.nan))
            continue

        # 计算距平
        merged['anomaly'] = merged.groupby('月')[variable].transform(
            lambda x: x - x.mean()
        )

        # 计算相关系数
        corr = merged['ONI'].corr(merged['anomaly'])
        correlations.append((lag, corr))

    # 找到最大相关系数（绝对值）
    best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)

    return best_lag, best_corr


def calculate_extreme_frequency_by_enso(extreme_df: pd.DataFrame,
                                        oni_df: pd.DataFrame,
                                        config: ENSOConfig) -> pd.DataFrame:
    """
    统计不同ENSO阶段的极端事件频次

    Returns:
    --------
    DataFrame with columns: ['enso_phase', 'extreme_high', 'extreme_low', 'total']
    """
    # 分类ENSO阶段
    oni_classified = classify_enso_phase(oni_df, config)

    # 合并数据
    merged = pd.merge(
        extreme_df,
        oni_classified[['年', '月', 'enso_phase']],
        on=['年', '月'],
        how='inner'
    )

    # 统计（转换为整数）
    freq = merged.groupby('enso_phase').agg({
        'extreme_high': lambda x: int(x.sum()),  # ← 修改这里
        'extreme_low': lambda x: int(x.sum()),   # ← 修改这里
        'is_extreme': lambda x: int(x.sum())     # ← 修改这里
    }).reset_index()

    freq.columns = ['enso_phase', 'extreme_high_count', 'extreme_low_count', 'total_extreme']

    # 计算总数和百分比
    total_by_phase = merged.groupby('enso_phase').size().reset_index(name='total_months')
    freq = pd.merge(freq, total_by_phase, on='enso_phase')

    freq['extreme_high_pct'] = (freq['extreme_high_count'] / freq['total_months'] * 100).round(2)
    freq['extreme_low_pct'] = (freq['extreme_low_count'] / freq['total_months'] * 100).round(2)
    freq['total_extreme_pct'] = (freq['total_extreme'] / freq['total_months'] * 100).round(2)

    return freq


# ============================================================================
# 可视化层
# ============================================================================

def create_extreme_heatmap_single(extreme_df: pd.DataFrame, province: str,
                                  variable: str, var_display_name: str = None) -> go.Figure:
    """
    模式A：单省详细热力图（年×月）
    """
    df_prov = extreme_df[extreme_df['province'] == province].copy()

    # 创建透视表
    pivot = df_prov.pivot_table(
        index='月',
        columns='年',
        values='is_extreme',
        aggfunc='sum'
    ).fillna(0).astype(int)

    # 反转月份顺序（12月在上）
    pivot = pivot.sort_index(ascending=False)

    var_name = var_display_name if var_display_name else variable

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, 'lightgray'], [1, 'red']],
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="是否极端")
    ))

    fig.update_layout(
        title=f'{province} - {var_name} 极端事件分布（年×月）',
        xaxis_title='年份',
        yaxis_title='月份',
        height=500,
        template='plotly_white'
    )

    return fig


def create_extreme_heatmap_national(extreme_df: pd.DataFrame, variable: str,
                                    var_display_name: str = None) -> go.Figure:
    """
    模式B：全国汇总热力图（年×月，值=省份数）
    """
    # 按年月统计有多少省份出现极端
    summary = extreme_df[extreme_df['is_extreme']].groupby(['年', '月']).size().reset_index(name='province_count')

    # 创建完整的年月网格
    years = sorted(extreme_df['年'].unique())
    months = range(1, 13)
    full_grid = pd.DataFrame([(y, m) for y in years for m in months], columns=['年', '月'])

    # 合并
    summary = pd.merge(full_grid, summary, on=['年', '月'], how='left').fillna(0)

    # 透视
    pivot = summary.pivot_table(
        index='月',
        columns='年',
        values='province_count',
        fill_value=0
    ).sort_index(ascending=False)

    var_name = var_display_name if var_display_name else variable

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Reds',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 10},
        colorbar=dict(title="省份数")
    ))

    fig.update_layout(
        title=f'全国 - {var_name} 极端事件省份数（年×月）',
        xaxis_title='年份',
        yaxis_title='月份',
        height=500,
        template='plotly_white'
    )

    return fig


def create_extreme_heatmap_provinces(extreme_df: pd.DataFrame, variable: str,
                                     var_display_name: str = None) -> go.Figure:
    """
    模式C：省份对比热力图（年×省份，值=月数）
    """
    # 按年、省份统计极端月数
    summary = extreme_df[extreme_df['is_extreme']].groupby(['年', 'province']).size().reset_index(name='extreme_months')

    # 创建完整网格
    years = sorted(extreme_df['年'].unique())
    provinces = sorted(extreme_df['province'].unique())
    full_grid = pd.DataFrame([(y, p) for y in years for p in provinces], columns=['年', 'province'])

    summary = pd.merge(full_grid, summary, on=['年', 'province'], how='left').fillna(0)

    # 透视
    pivot = summary.pivot_table(
        index='province',
        columns='年',
        values='extreme_months',
        fill_value=0
    )

    var_name = var_display_name if var_display_name else variable

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 8},
        colorbar=dict(title="极端月数")
    ))

    fig.update_layout(
        title=f'各省 - {var_name} 年度极端月数对比',
        xaxis_title='年份',
        yaxis_title='省份',
        height=800,
        template='plotly_white'
    )

    return fig


def create_enso_scatter(oni_df: pd.DataFrame, climate_df: pd.DataFrame,
                        variable: str, provinces: List[str], config: ENSOConfig,
                        var_display_name: str = None) -> go.Figure:
    """
    ENSO散点图（ONI vs 气象距平）
    """
    # 合并数据
    merged = merge_oni_with_climate(oni_df, climate_df, config.lag_months)

    # 筛选省份
    if '全部' not in provinces:
        merged = merged[merged['province'].isin(provinces)]

    # 计算距平
    if config.anomaly_type == 'monthly':
        merged['anomaly'] = merged.groupby(['province', '月'])[variable].transform(
            lambda x: x - x.mean()
        )
    elif config.anomaly_type == 'seasonal':
        merged['season'] = merged['月'].apply(lambda x: (x % 12 + 3) // 3)
        merged['anomaly'] = merged.groupby(['province', 'season'])[variable].transform(
            lambda x: x - x.mean()
        )
    else:  # annual
        merged['anomaly'] = merged.groupby('province')[variable].transform(
            lambda x: x - x.mean()
        )

    var_name = var_display_name if var_display_name else variable

    fig = go.Figure()

    # 如果是全部省份，用不同颜色
    if '全部' in provinces or len(provinces) > 1:
        for province in merged['province'].unique():
            df_prov = merged[merged['province'] == province]

            # 计算相关系数
            corr = df_prov['ONI'].corr(df_prov['anomaly'])

            fig.add_trace(go.Scatter(
                x=df_prov['ONI'],
                y=df_prov['anomaly'],
                mode='markers',
                name=f'{province} (r={corr:.3f})',
                marker=dict(size=6, opacity=0.6),
                hovertemplate='<b>%{fullData.name}</b><br>ONI: %{x:.2f}<br>距平: %{y:.2f}<extra></extra>'
            ))
    else:
        # 单省份
        corr = merged['ONI'].corr(merged['anomaly'])

        fig.add_trace(go.Scatter(
            x=merged['ONI'],
            y=merged['anomaly'],
            mode='markers',
            marker=dict(size=8, color=merged['ONI'], colorscale='RdBu_r',
                        showscale=True, colorbar=dict(title="ONI")),
            hovertemplate='ONI: %{x:.2f}<br>距平: %{y:.2f}<extra></extra>'
        ))

        # 添加趋势线
        z = np.polyfit(merged['ONI'], merged['anomaly'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(merged['ONI'].min(), merged['ONI'].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name=f'趋势线 (r={corr:.3f})',
            line=dict(color='red', dash='dash', width=2)
        ))

    # 添加ENSO阈值线
    fig.add_vline(x=config.el_nino_threshold, line_dash="dash",
                  line_color="red", opacity=0.5, annotation_text="El Niño")
    fig.add_vline(x=config.la_nina_threshold, line_dash="dash",
                  line_color="blue", opacity=0.5, annotation_text="La Niña")

    lag_text = f" (滞后{config.lag_months}月)" if config.lag_months > 0 else ""

    fig.update_layout(
        title=f'ENSO指数 vs {var_name}距平{lag_text}',
        xaxis_title='ONI指数',
        yaxis_title=f'{var_name}距平',
        template='plotly_white',
        height=600,
        hovermode='closest'
    )

    return fig


def create_enso_timeseries(oni_df: pd.DataFrame, climate_df: pd.DataFrame,
                           variables: List[str], province: str, config: ENSOConfig,
                           var_name_map: dict = None) -> go.Figure:
    """
    ENSO时序对比图（背景化ONI + 前景变量）
    """
    # 分类ENSO阶段
    oni_classified = classify_enso_phase(oni_df, config)

    # 合并数据
    merged = merge_oni_with_climate(oni_classified, climate_df, config.lag_months)

    if province != '全部':
        merged = merged[merged['province'] == province]
    else:
        # 全国平均
        merged = merged.groupby(['年', '月', 'ONI', 'enso_phase']).mean().reset_index()

    # 创建时间列
    merged['date'] = pd.to_datetime(
        merged['年'].astype(str) + '-' + merged['月'].astype(str).str.zfill(2)
    )
    merged = merged.sort_values('date')

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.2, 0.8],
        vertical_spacing=0.05,
        subplot_titles=('ENSO阶段', '气象变量时序'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # ========== 上图：ENSO背景 ==========
    # El Niño区域（红色）
    el_nino_periods = merged[merged['enso_phase'] == 'El Niño']
    if len(el_nino_periods) > 0:
        fig.add_trace(go.Scatter(
            x=el_nino_periods['date'],
            y=el_nino_periods['ONI'],
            fill='tozeroy',
            fillcolor='rgba(255, 100, 100, 0.3)',
            line=dict(color='red', width=0),
            name='El Niño',
            showlegend=True,
            hovertemplate='%{x}<br>ONI: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # La Niña区域（蓝色）
    la_nina_periods = merged[merged['enso_phase'] == 'La Niña']
    if len(la_nina_periods) > 0:
        fig.add_trace(go.Scatter(
            x=la_nina_periods['date'],
            y=la_nina_periods['ONI'],
            fill='tozeroy',
            fillcolor='rgba(100, 100, 255, 0.3)',
            line=dict(color='blue', width=0),
            name='La Niña',
            showlegend=True,
            hovertemplate='%{x}<br>ONI: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # ONI线
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=merged['ONI'],
        mode='lines',
        name='ONI',
        line=dict(color='black', width=1.5),
        showlegend=False
    ), row=1, col=1)

    # ========== 下图：气象变量 ==========
    colors = px.colors.qualitative.Set2

    for idx, variable in enumerate(variables):
        var_display = var_name_map.get(variable, variable) if var_name_map else variable

        # 计算距平
        merged['anomaly'] = merged.groupby('月')[variable].transform(
            lambda x: x - x.mean()
        )

        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=merged['anomaly'],
            mode='lines',
            name=var_display,
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=f'{var_display}<br>%{{x}}<br>距平: %{{y:.2f}}<extra></extra>'
        ), row=2, col=1, secondary_y=False)

    # 布局
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="ONI", row=1, col=1)
    fig.update_yaxes(title_text="距平值", secondary_y=False, row=2, col=1)

    # 添加ENSO阈值线
    fig.add_hline(y=config.el_nino_threshold, line_dash="dash", line_color="red",
                  opacity=0.5, row=1, col=1)
    fig.add_hline(y=config.la_nina_threshold, line_dash="dash", line_color="blue",
                  opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    lag_text = f" (ONI滞后{config.lag_months}月)" if config.lag_months > 0 else ""

    fig.update_layout(
        title=f'ENSO与气象变量时序对比{lag_text}',
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def create_extreme_frequency_table(freq_df: pd.DataFrame) -> go.Figure:
    """
    创建极端事件频次统计表
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>ENSO阶段</b>', '<b>极端高温次数</b>', '<b>占比</b>',
                    '<b>极端低温次数</b>', '<b>占比</b>', '<b>总极端次数</b>', '<b>占比</b>'],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                freq_df['enso_phase'],
                freq_df['extreme_high_count'],
                freq_df['extreme_high_pct'].apply(lambda x: f'{x:.1f}%'),
                freq_df['extreme_low_count'],
                freq_df['extreme_low_pct'].apply(lambda x: f'{x:.1f}%'),
                freq_df['total_extreme'],
                freq_df['total_extreme_pct'].apply(lambda x: f'{x:.1f}%')
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title='不同ENSO阶段的极端事件频次统计',
        height=300
    )

    return fig


# ============================================================================
# 新增：ENSO当前状态 + 均值偏移热力图（供Dash使用）
# ============================================================================

def get_current_enso_status(oni_df: pd.DataFrame) -> dict:
    """
    返回当前ENSO状态信息字典。
    """
    if oni_df.empty:
        return {}
    latest = oni_df.sort_values(['年', '月']).iloc[-1]
    oni_val = latest['ONI']
    phase = '厄尔尼诺' if oni_val >= 0.5 else ('拉尼娜' if oni_val <= -0.5 else '中性')
    phase_color = '#d73027' if phase == '厄尔尼诺' else ('#4393c3' if phase == '拉尼娜' else '#888888')

    # 计算持续月数
    sorted_oni = oni_df.sort_values(['年', '月']).reset_index(drop=True)
    count = 0
    for i in range(len(sorted_oni) - 1, -1, -1):
        row_oni = sorted_oni.loc[i, 'ONI']
        if phase == '厄尔尼诺' and row_oni >= 0.5:
            count += 1
        elif phase == '拉尼娜' and row_oni <= -0.5:
            count += 1
        elif phase == '中性' and -0.5 < row_oni < 0.5:
            count += 1
        else:
            break

    return {
        'year': int(latest['年']),
        'month': int(latest['月']),
        'oni': float(oni_val),
        'phase': phase,
        'color': phase_color,
        'duration': count,
    }


def create_enso_zmean_heatmap(df: pd.DataFrame, oni_df: pd.DataFrame) -> go.Figure:
    """
    ENSO相位下各气候要素系统性偏移热力图（Plotly版，供Dash使用）。
    """
    WIND_COL  = '风速_平均_100米_省内'
    SOLAR_COL = '光辐照_平均_地表_省内'
    TEMP_COL  = '温度_平均_2米_省内'
    WATER_COL = '地表总降水_平均_省内'

    # 计算z分
    df2 = df.copy()
    for col, z_col in [(WIND_COL, 'wind_z'), (SOLAR_COL, 'solar_z'),
                       (TEMP_COL, 'temp_z'), (WATER_COL, 'water_z')]:
        if col not in df2.columns:
            continue
        mm = df2.groupby(['province', '月'])[col].mean().rename('_mm')
        ms = df2.groupby(['province', '月'])[col].std().rename('_ms')
        d  = df2.join(mm, on=['province', '月']).join(ms, on=['province', '月'])
        df2[z_col] = (d[col] - d['_mm']) / d['_ms']

    z_cols = [c for c in ['wind_z', 'solar_z', 'temp_z', 'water_z'] if c in df2.columns]
    if not z_cols or oni_df.empty:
        return go.Figure()

    df_oni = pd.merge(
        df2[['年', '月', 'province'] + z_cols],
        oni_df[['年', '月', 'ONI']], on=['年', '月'], how='inner'
    )
    df_oni['enso_phase'] = '中性'
    df_oni.loc[df_oni['ONI'] >= 0.5,  'enso_phase'] = '厄尔尼诺'
    df_oni.loc[df_oni['ONI'] <= -0.5, 'enso_phase'] = '拉尼娜'

    zmean = df_oni.groupby('enso_phase')[z_cols].mean()
    col_labels = {'wind_z': '风速', 'solar_z': '光照', 'temp_z': '气温', 'water_z': '降水'}
    zmean.columns = [col_labels.get(c, c) for c in zmean.columns]
    zmean_plot = zmean.reindex(['厄尔尼诺', '中性', '拉尼娜'])

    data_mat = zmean_plot.values
    text_mat = [[f'{v:+.3f}' for v in row] for row in data_mat]

    fig = go.Figure(data=go.Heatmap(
        z=data_mat,
        x=list(zmean_plot.columns),
        y=list(zmean_plot.index),
        text=text_mat,
        texttemplate='%{text}',
        textfont=dict(size=13, color='black'),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-0.3,
        zmax=0.3,
        colorbar=dict(title='z分均值', thickness=15),
    ))

    fig.update_layout(
        title=dict(
            text='ENSO相位下各气候要素系统性偏移<br><sup>正=偏高，负=偏低；即使不极端，均值也在移动</sup>',
            font=dict(size=14)
        ),
        height=280,
        margin=dict(l=80, r=60, t=80, b=40),
        xaxis=dict(side='bottom'),
    )
    return fig
