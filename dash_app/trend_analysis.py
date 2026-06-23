import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


# ============================================================================
# 数据聚合函数
# ============================================================================

def aggregate_to_annual(df, variable, agg_method='mean'):
    """
    将月度数据聚合为年度数据

    Parameters:
    -----------
    df : DataFrame
        原始月度数据
    variable : str
        要聚合的变量名
    agg_method : str
        聚合方法：'mean'(均值) 或 'sum'(总和)
    """
    if agg_method == 'mean':
        annual_df = df.groupby(['province', '年'])[variable].mean().reset_index()
    else:
        annual_df = df.groupby(['province', '年'])[variable].sum().reset_index()

    annual_df.columns = ['province', '年', variable]
    return annual_df


def get_seasonal_data(df, variable, season_months, agg_method='mean'):
    """
    提取特定季节的数据

    Parameters:
    -----------
    season_months : list
        季节对应的月份，如 [12, 1, 2] 表示冬季
    agg_method : str
        聚合方法：'mean'(均值) 或 'sum'(总和)
    """
    # 特殊处理跨年季节（冬季：12-2月）
    if 12 in season_months and 1 in season_months:
        # 冬季逻辑：N年的12月 + N+1年的1-2月
        winter_data = []

        for province in df['province'].unique():
            df_prov = df[df['province'] == province].copy()
            years = sorted(df_prov['年'].unique())

            for year in years[:-1]:  # 排除最后一年（因为没有下一年数据）
                # N年的12月
                dec_data = df_prov[(df_prov['年'] == year) & (df_prov['月'] == 12)]
                # N+1年的1-2月
                jan_feb_data = df_prov[(df_prov['年'] == year + 1) & (df_prov['月'].isin([1, 2]))]

                # 合并数据
                season_data = pd.concat([dec_data, jan_feb_data])

                if len(season_data) == 3:  # 确保有完整的3个月数据
                    if agg_method == 'mean':
                        value = season_data[variable].mean()
                    else:
                        value = season_data[variable].sum()

                    winter_data.append({
                        'province': province,
                        '年': year,  # 记录为N年（12月所在年）
                        variable: value
                    })

        return pd.DataFrame(winter_data)

    else:
        # 其他季节：正常处理
        df_season = df[df['月'].isin(season_months)].copy()

        if agg_method == 'mean':
            seasonal_df = df_season.groupby(['province', '年'])[variable].mean().reset_index()
        else:
            seasonal_df = df_season.groupby(['province', '年'])[variable].sum().reset_index()

        return seasonal_df


def get_monthly_specific(df, variable, month):
    """
    提取特定月份的数据（如只看每年1月）
    """
    df_month = df[df['月'] == month].copy()
    return df_month[['province', '年', variable]]


# ============================================================================
# 距平计算
# ============================================================================

def calculate_anomaly(df, variable, baseline_years=None, agg_method='mean'):
    """
    计算距平值（相对于气候基准态的偏差）

    Parameters:
    -----------
    baseline_years : tuple or None
        基准期，如 (1995, 2024)。None 则使用全部年份
    agg_method : str
        聚合方法：'mean'(均值) 或 'sum'(总和)
        注意：对于月度数据，sum通常不适用，但保留选项以保持一致性
    """
    df_result = df.copy()

    for province in df['province'].unique():
        df_prov = df[df['province'] == province].copy()

        # 计算基准态
        if baseline_years:
            baseline_data = df_prov[
                (df_prov['年'] >= baseline_years[0]) &
                (df_prov['年'] <= baseline_years[1])
                ]
        else:
            baseline_data = df_prov

        # 按月份计算基准态均值（距平通常使用均值作为基准）
        if agg_method == 'mean':
            baseline_mean = baseline_data.groupby('月')[variable].mean()
        else:
            # 即使是sum，距平的基准也应该是平均的sum（即每月的平均总量）
            baseline_mean = baseline_data.groupby('月')[variable].sum()

        # 计算距平
        for idx, row in df_prov.iterrows():
            month = row['月']
            anomaly = row[variable] - baseline_mean[month]
            df_result.loc[idx, f'{variable}_anomaly'] = anomaly

    return df_result


# ============================================================================
# 趋势分析可视化
# ============================================================================

def create_annual_trend_plot(df, province, variable, agg_method='mean',
                             compare_provinces=None, show_national=False):
    """
    年度趋势图：显示30年长期趋势

    Parameters:
    -----------
    compare_provinces : list or None
        要对比的省份列表
    show_national : bool
        是否显示全国平均线
    """
    # 聚合数据
    annual_df = aggregate_to_annual(df, variable, agg_method)

    fig = go.Figure()

    # 定义颜色方案
    main_color = 'rgb(0, 100, 250)'
    compare_colors = ['rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                      'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
    national_color = 'rgb(128, 128, 128)'

    # 主省份数据
    df_main = annual_df[annual_df['province'] == province].sort_values('年')

    # 线性回归
    x = df_main['年'].values
    y = df_main[variable].values
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    trend_line = slope * x + intercept

    # 主线条
    fig.add_trace(go.Scatter(
        x=df_main['年'],
        y=df_main[variable],
        mode='lines+markers',
        name=province,
        line=dict(width=3, color=main_color),
        marker=dict(size=8),
        hovertemplate=f'<b>{province}</b><br>年份: %{{x}}<br>值: %{{y:.2f}}<extra></extra>'
    ))

    # 趋势线
    fig.add_trace(go.Scatter(
        x=df_main['年'],
        y=trend_line,
        mode='lines',
        name=f'趋势线 (斜率={slope:.4f}/年)',
        line=dict(dash='dash', color=main_color, width=2),
        hovertemplate=f'趋势值: %{{y:.2f}}<br>R²={r_value ** 2:.3f}<extra></extra>'
    ))

    # 对比省份
    if compare_provinces:
        for idx, comp_prov in enumerate(compare_provinces):
            df_comp = annual_df[annual_df['province'] == comp_prov].sort_values('年')

            if len(df_comp) == 0:
                continue

            color = compare_colors[idx % len(compare_colors)]

            # 线性回归
            x_comp = df_comp['年'].values
            y_comp = df_comp[variable].values
            slope_comp, intercept_comp, r_value_comp, _, _ = scipy_stats.linregress(x_comp, y_comp)
            trend_line_comp = slope_comp * x_comp + intercept_comp

            # 数据线
            fig.add_trace(go.Scatter(
                x=df_comp['年'],
                y=df_comp[variable],
                mode='lines+markers',
                name=comp_prov,
                line=dict(width=2, color=color),
                marker=dict(size=6),
                hovertemplate=f'<b>{comp_prov}</b><br>年份: %{{x}}<br>值: %{{y:.2f}}<extra></extra>'
            ))

            # 趋势线
            fig.add_trace(go.Scatter(
                x=df_comp['年'],
                y=trend_line_comp,
                mode='lines',
                name=f'{comp_prov} 趋势 (斜率={slope_comp:.4f}/年)',
                line=dict(dash='dash', color=color, width=1.5),
                hovertemplate=f'趋势值: %{{y:.2f}}<br>R²={r_value_comp ** 2:.3f}<extra></extra>'
            ))

    # 全国平均
    if show_national:
        df_national = annual_df[annual_df['province'] == '全国平均'].sort_values('年')
        fig.add_trace(go.Scatter(
            x=df_national['年'],
            y=df_national[variable],
            mode='lines',
            name='全国平均',
            line=dict(width=2.5, color=national_color),
            hovertemplate='<b>全国平均</b><br>年份: %{{x}}<br>值: %{{y:.2f}}<extra></extra>'
        ))

    agg_text = '年均值' if agg_method == 'mean' else '年总和'
    fig.update_layout(
        title=f'{province} - {variable} 年度趋势 ({agg_text})',
        xaxis_title='年份',
        yaxis_title=variable,
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    return fig


def create_seasonal_trend_plot(df, province, variable, agg_method='mean',
                               compare_provinces=None, show_national=False):
    """
    季节趋势图：分季节分析30年变化
    """
    seasons = {
        '春季 (3-5月)': [3, 4, 5],
        '夏季 (6-8月)': [6, 7, 8],
        '秋季 (9-11月)': [9, 10, 11],
        '冬季 (12-2月)': [12, 1, 2]
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(seasons.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ['rgb(50, 200, 100)', 'rgb(250, 100, 50)',
              'rgb(250, 200, 50)', 'rgb(100, 150, 250)']

    for idx, (season_name, months) in enumerate(seasons.items()):
        row, col = positions[idx]

        # 主省份 - 添加 agg_method 参数
        seasonal_df = get_seasonal_data(df, variable, months, agg_method)
        df_main = seasonal_df[seasonal_df['province'] == province].sort_values('年')

        if len(df_main) == 0:
            continue

        # 线性回归
        x = df_main['年'].values
        y = df_main[variable].values
        slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
        trend_line = slope * x + intercept

        # 数据线
        fig.add_trace(go.Scatter(
            x=df_main['年'],
            y=df_main[variable],
            mode='lines+markers',
            name=f'{season_name}',
            line=dict(width=2, color=colors[idx]),
            marker=dict(size=6),
            showlegend=(idx == 0),
            hovertemplate=f'{season_name}<br>年份: %{{x}}<br>值: %{{y:.2f}}<extra></extra>'
        ), row=row, col=col)

        # 趋势线
        fig.add_trace(go.Scatter(
            x=df_main['年'],
            y=trend_line,
            mode='lines',
            line=dict(dash='dash', color='red', width=1.5),
            showlegend=False,
            hovertemplate=f'趋势 (斜率={slope:.4f})<extra></extra>'
        ), row=row, col=col)

        # 对比省份
        if compare_provinces:
            for comp_prov in compare_provinces:
                df_comp = seasonal_df[seasonal_df['province'] == comp_prov].sort_values('年')
                fig.add_trace(go.Scatter(
                    x=df_comp['年'],
                    y=df_comp[variable],
                    mode='lines',
                    name=comp_prov,
                    line=dict(width=1.5, dash='dot'),
                    showlegend=(idx == 0),
                    hovertemplate=f'{comp_prov}<br>%{{y:.2f}}<extra></extra>'
                ), row=row, col=col)

        # 全国平均
        if show_national and province != '全国平均':
            df_national = seasonal_df[seasonal_df['province'] == '全国平均'].sort_values('年')
            if len(df_national) > 0:
                fig.add_trace(go.Scatter(
                    x=df_national['年'],
                    y=df_national[variable],
                    mode='lines',
                    name='全国平均',
                    line=dict(width=1.5, dash='dash', color='gray'),
                    showlegend=(idx == 0),
                    hovertemplate='全国平均<br>%{{y:.2f}}<extra></extra>'
                ), row=row, col=col)

    agg_text = '均值' if agg_method == 'mean' else '总和'
    fig.update_layout(
        title=f'{province} - {variable} 季节趋势分析 ({agg_text})',
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="年份")
    fig.update_yaxes(title_text=variable)

    return fig



def create_monthly_specific_trend_plot(df, province, variable, selected_months, agg_method='mean',
                                       compare_provinces=None, show_national=False):
    """
    特定月份趋势图：如只看每年1月、7月的30年变化
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, month in enumerate(selected_months):
        # 主省份
        df_month = get_monthly_specific(df, variable, month)
        df_main = df_month[df_month['province'] == province].sort_values('年')

        if len(df_main) == 0:
            continue

        # 线性回归
        x = df_main['年'].values
        y = df_main[variable].values
        slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
        trend_line = slope * x + intercept

        # 数据线
        fig.add_trace(go.Scatter(
            x=df_main['年'],
            y=df_main[variable],
            mode='lines+markers',
            name=f'{month}月',
            line=dict(width=2.5, color=colors[idx % len(colors)]),
            marker=dict(size=7),
            hovertemplate=f'<b>{month}月</b><br>年份: %{{x}}<br>值: %{{y:.2f}}<extra></extra>'
        ))

        # 趋势线
        fig.add_trace(go.Scatter(
            x=df_main['年'],
            y=trend_line,
            mode='lines',
            name=f'{month}月趋势 ({slope:.4f}/年)',
            line=dict(dash='dash', color=colors[idx % len(colors)], width=1.5),
            hovertemplate=f'{month}月趋势<br>R²={r_value ** 2:.3f}<extra></extra>'
        ))

        # 对比省份
        if compare_provinces:
            for comp_prov in compare_provinces:
                df_comp = df_month[df_month['province'] == comp_prov].sort_values('年')
                fig.add_trace(go.Scatter(
                    x=df_comp['年'],
                    y=df_comp[variable],
                    mode='lines',
                    name=f'{comp_prov} ({month}月)',
                    line=dict(width=1.5, dash='dot', color=colors[idx % len(colors)]),
                    opacity=0.6,
                    hovertemplate=f'{comp_prov} {month}月<br>%{{y:.2f}}<extra></extra>'
                ))

        # 全国平均
        if show_national and province != '全国平均':
            df_national = df_month[df_month['province'] == '全国平均'].sort_values('年')
            if len(df_national) > 0:
                fig.add_trace(go.Scatter(
                    x=df_national['年'],
                    y=df_national[variable],
                    mode='lines',
                    name=f'全国平均 ({month}月)',
                    line=dict(width=1.5, dash='dash', color='gray'),
                    opacity=0.6,
                    hovertemplate=f'全国平均 {month}月<br>%{{y:.2f}}<extra></extra>'
                ))

    fig.update_layout(
        title=f'{province} - {variable} 特定月份趋势对比',
        xaxis_title='年份',
        yaxis_title=variable,
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    return fig


def create_anomaly_plot(df, province, variable, baseline_years=None, agg_method='mean',
                        compare_provinces=None, show_national=False):
    """
    距平分析图：红蓝柱状图显示相对基准态的偏差

    Parameters:
    -----------
    agg_method : str
        聚合方法：'mean'(均值) 或 'sum'(总和)
        注意：距平计算时，基准态和当前值使用相同的聚合方法
    """
    # 计算距平（内部会根据agg_method处理）
    df_anomaly = calculate_anomaly(df, variable, baseline_years, agg_method)

    # 筛选主省份
    df_main = df_anomaly[df_anomaly['province'] == province].copy()
    df_main['年月'] = df_main['年'].astype(str) + '-' + df_main['月'].astype(str).str.zfill(2)
    df_main = df_main.sort_values(['年', '月'])

    anomaly_col = f'{variable}_anomaly'

    # 颜色映射：正值红色，负值蓝色
    colors = ['rgb(220, 50, 50)' if x > 0 else 'rgb(50, 100, 220)'
              for x in df_main[anomaly_col]]

    fig = go.Figure()

    # 主柱状图
    fig.add_trace(go.Bar(
        x=df_main['年月'],
        y=df_main[anomaly_col],
        marker_color=colors,
        name=province,
        hovertemplate='<b>%{x}</b><br>距平: %{y:.2f}<extra></extra>'
    ))

    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # 对比省份（折线叠加）
    if compare_provinces:
        for comp_prov in compare_provinces:
            df_comp = df_anomaly[df_anomaly['province'] == comp_prov].copy()
            df_comp['年月'] = df_comp['年'].astype(str) + '-' + df_comp['月'].astype(str).str.zfill(2)
            df_comp = df_comp.sort_values(['年', '月'])

            fig.add_trace(go.Scatter(
                x=df_comp['年月'],
                y=df_comp[anomaly_col],
                mode='lines',
                name=comp_prov,
                line=dict(width=2),
                hovertemplate=f'{comp_prov}<br>距平: %{{y:.2f}}<extra></extra>'
            ))

    # 全国平均
    if show_national and province != '全国平均':
        df_national = df_anomaly[df_anomaly['province'] == '全国平均'].copy()
        df_national['年月'] = df_national['年'].astype(str) + '-' + df_national['月'].astype(str).str.zfill(2)
        df_national = df_national.sort_values(['年', '月'])

        fig.add_trace(go.Scatter(
            x=df_national['年月'],
            y=df_national[anomaly_col],
            mode='lines',
            name='全国平均',
            line=dict(width=2, dash='dash', color='gray'),
            hovertemplate='全国平均<br>距平: %{{y:.2f}}<extra></extra>'
        ))

    baseline_text = f"{baseline_years[0]}-{baseline_years[1]}" if baseline_years else "全时段"
    agg_text = '均值' if agg_method == 'mean' else '总和'

    fig.update_layout(
        title=f'{province} - {variable} 距平分析 (基准期: {baseline_text}, {agg_text})',
        xaxis_title='年-月',
        yaxis_title=f'{variable} 距平值',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        xaxis=dict(
            tickangle=-45,
            nticks=30,
            rangeslider=dict(visible=True)
        )
    )

    return fig


def create_interannual_variability_plot(df, province, variable):
    """
    年际变化分析：展示逐年波动特征
    """
    annual_df = aggregate_to_annual(df, variable, 'mean')
    df_main = annual_df[annual_df['province'] == province].sort_values('年')

    # 计算年际变化率
    df_main['变化率'] = df_main[variable].pct_change() * 100

    # 计算移动平均（5年）
    df_main['5年移动平均'] = df_main[variable].rolling(window=5, center=True).mean()

    # 创建双Y轴图
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 原始值
    fig.add_trace(go.Scatter(
        x=df_main['年'],
        y=df_main[variable],
        mode='lines+markers',
        name='年度值',
        line=dict(color='lightblue', width=2),
        marker=dict(size=8, color='lightblue')
    ), secondary_y=False)

    # 5年移动平均
    fig.add_trace(go.Scatter(
        x=df_main['年'],
        y=df_main['5年移动平均'],
        mode='lines',
        name='5年移动平均',
        line=dict(color='darkblue', width=3)
    ), secondary_y=False)

    # 年际变化率（柱状图）
    colors_change = ['red' if x > 0 else 'green' for x in df_main['变化率'].fillna(0)]
    fig.add_trace(go.Bar(
        x=df_main['年'],
        y=df_main['变化率'],
        name='年际变化率 (%)',
        marker_color=colors_change,
        opacity=0.6
    ), secondary_y=True)

    fig.update_xaxes(title_text="年份")
    fig.update_yaxes(title_text=variable, secondary_y=False)
    fig.update_yaxes(title_text="变化率 (%)", secondary_y=True)

    fig.update_layout(
        title=f'{province} - {variable} 年际变化分析',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    return fig
