import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from datetime import datetime
from pathlib import Path
from trend_analysis import (
    create_annual_trend_plot,
    create_seasonal_trend_plot,
    create_monthly_specific_trend_plot,
    create_anomaly_plot,
    create_interannual_variability_plot
)
from spatial_analysis import create_cross_sectional_comparison
from spatial_map import create_china_map_heatmap
from extreme_events import (
    ExtremeConfig,
    ENSOConfig,
    load_oni_data,
    detect_extreme_events,
    classify_enso_phase,
    find_best_lag,
    calculate_extreme_frequency_by_enso,
    create_extreme_heatmap_single,
    create_extreme_heatmap_national,
    create_extreme_heatmap_provinces,
    create_enso_scatter,
    create_enso_timeseries,
    create_extreme_frequency_table,
    get_current_enso_status,
    create_enso_zmean_heatmap,
)
from multivariate_analysis import (
    create_correlation_matrix,
    create_scatter_with_regression,
    create_dual_map_comparison,
    var_name_map
)
from resource_stability import (
    calculate_resource_stability,
    create_quadrant_plot,
    create_summary_table,
    export_data_to_excel
)


# ============================================================================
# 数据加载与预处理
# ============================================================================
# 确保导出目录存在
export_dir = str(Path(__file__).parent / '导出数据')
os.makedirs(export_dir, exist_ok=True)
def load_and_explore_data(file_path):
    """加载数据并进行基础探索"""
    df = pd.read_excel(file_path)

    # 基础信息提取
    year_range = sorted(df['年'].unique())
    month_range = sorted(df['月'].unique())
    provinces = sorted(df['province'].unique())

    # 获取数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['年', '月']]

    # 数据完整性
    total_records = len(df)
    expected_records = len(provinces) * len(year_range) * len(month_range)
    missing_count = df.isnull().sum().sum()

    stats = {
        'shape': df.shape,
        'year_range': year_range,
        'year_min': min(year_range),
        'year_max': max(year_range),
        'year_count': len(year_range),
        'month_range': month_range,
        'provinces': provinces,
        'province_count': len(provinces),
        'numeric_columns': numeric_cols,
        'variable_count': len(numeric_cols),
        'total_records': total_records,
        'expected_records': expected_records,
        'missing_count': missing_count,
        'completeness': f"{(total_records / expected_records) * 100:.1f}%"
    }

    return df, stats


def calculate_national_average(df):
    """计算全国平均值（简单平均各省）"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['年', '月']]

    national_df = df.groupby(['年', '月'])[numeric_cols].mean().reset_index()
    national_df['province'] = '全国平均'

    return national_df


# ============================================================================
# Dash 应用初始化
# ============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 加载数据
file_path = str(Path(__file__).parent / 'data' / 'era5历史数据.xlsx')
df, stats = load_and_explore_data(file_path)

# 计算全国平均
national_df = calculate_national_average(df)
df_with_national = pd.concat([df, national_df], ignore_index=True)
# ========== 加载ONI数据 ==========
try:
    oni_file_path = str(Path(__file__).parent / 'data' / 'ONI.xlsx')
    df_oni = load_oni_data(oni_file_path)
    print(f"[OK] ONI数据加载成功: {len(df_oni)} 条记录")

    # 只保留与ERA5重叠的时间段
    df_oni = df_oni[
        (df_oni['年'] >= stats['year_min']) &
        (df_oni['年'] <= stats['year_max'])
        ]
    print(f"[OK] ONI数据筛选后: {len(df_oni)} 条记录 ({stats['year_min']}-{stats['year_max']})")

except Exception as e:
    print(f"[WARN] ONI数据加载失败: {e}")
    df_oni = pd.DataFrame()  # 空数据框
# 变量名称映射
var_name_map = {
    '平均温度(°C)': '平均温度(°C)',
    '最高温度(°C)': '最高温度(°C)',
    '最低温度(°C)': '最低温度(°C)',
    '光辐照_平均_地表_省内': '平均光辐照',
    '光辐照_最大_地表_省内': '最大光辐照',
    '平均风速_10m(m/s)': '平均风速_10m(m/s)',
    '最大风速_10m(m/s)': '最大风速_10m(m/s)',
    '平均风速_100m(m/s)': '平均风速_100m(m/s)',
    '最大风速_100m(m/s)': '最大风速_100m(m/s)',
    '地表总降水_平均_省内': '平均降水(mm)',
    '地表总降水_最大_省内': '最大降水(mm)'
}


# ============================================================================
# 可视化函数
# ============================================================================

def create_seasonal_plot_mode1(df, province, variable, selected_years=None, highlight_year=None):
    """
    模式1: 多年线条图
    每年一条线，展示1-12月的季节变化
        Args:
        highlight_year: 需要加粗的年份，默认为最新年
    """
    df_filtered = df[(df['province'] == province)].copy()

    if selected_years:
        df_filtered = df_filtered[df_filtered['年'].isin(selected_years)]

    # 确定加粗年份
    available_years = sorted(df_filtered['年'].unique())
    if not available_years:
        return go.Figure()  # 没数据就返回空图

    if highlight_year is None:
        highlight_year = available_years[-1]  # 默认最新年

    fig = go.Figure()

    # 按年份分组，每年一条线
    for year in sorted(df_filtered['年'].unique()):
        df_year = df_filtered[df_filtered['年'] == year].sort_values('月')

        # 判断是否是高亮年份
        is_highlight = (year == highlight_year)

        fig.add_trace(go.Scatter(
            x=df_year['月'],
            y=df_year[variable],
            mode='lines+markers',
            name=str(year),
            line=dict(
                width=5 if is_highlight else 2,  # 高亮年份加粗
                color='red' if is_highlight else None  # 可选：高亮年份改颜色
            ),
            marker=dict(
                size=10 if is_highlight else 6  # 高亮年份标记点也变大
            ),
            hovertemplate=f'<b>{year}年</b><br>月份: %{{x}}<br>{var_name_map.get(variable, variable)}: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f'{province} - {var_name_map.get(variable, variable)} 季节变化（逐年对比）',
        xaxis_title='月份',
        yaxis_title=var_name_map.get(variable, variable),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="年份"
        )
    )

    return fig


def create_seasonal_plot_mode2(df, province, variable, show_years=None):
    """
    模式2: 气候包络线
    显示多年平均值、最大最小范围（阴影），可选择显示具体年份
    """
    df_filtered = df[(df['province'] == province)].copy()

    # 计算每月的统计值（跨年份）
    monthly_stats = df_filtered.groupby('月')[variable].agg(['mean', 'min', 'max', 'std']).reset_index()

    fig = go.Figure()

    # 添加最大-最小阴影
    fig.add_trace(go.Scatter(
        x=monthly_stats['月'].tolist() + monthly_stats['月'].tolist()[::-1],
        y=monthly_stats['max'].tolist() + monthly_stats['min'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 250, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='最大-最小范围',
        showlegend=True,
        hoverinfo='skip'
    ))

    # 添加均值线（粗线）
    fig.add_trace(go.Scatter(
        x=monthly_stats['月'],
        y=monthly_stats['mean'],
        mode='lines+markers',
        name='多年平均',
        line=dict(color='rgb(0, 100, 250)', width=4),
        marker=dict(size=8, color='rgb(0, 100, 250)'),
        hovertemplate=f'<b>多年平均</b><br>月份: %{{x}}<br>{var_name_map.get(variable, variable)}: %{{y:.2f}}<extra></extra>'
    ))

    # 如果选择了具体年份，添加年份线条
    if show_years:
        for year in show_years:
            df_year = df_filtered[df_filtered['年'] == year].sort_values('月')
            if len(df_year) > 0:
                fig.add_trace(go.Scatter(
                    x=df_year['月'],
                    y=df_year[variable],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2, dash='dash'),
                    marker=dict(size=5),
                    hovertemplate=f'<b>{year}年</b><br>月份: %{{x}}<br>{var_name_map.get(variable, variable)}: %{{y:.2f}}<extra></extra>'
                ))

    fig.update_layout(
        title=f'{province} - {var_name_map.get(variable, variable)} 气候包络线',
        xaxis_title='月份',
        yaxis_title=var_name_map.get(variable, variable),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


# ============================================================================
# Dash 布局
# ============================================================================

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🌍 中国气候数据分析平台", className="text-center mb-2"),
            html.H5("ERA5 历史数据 (1995-2024)", className="text-center text-muted mb-4")
        ])
    ]),

    dcc.Tabs(id='tabs', value='tab-overview', children=[

        # ===== Tab 1: 数据概览 =====
        dcc.Tab(label='📊 数据概览', value='tab-overview', children=[
            dbc.Container([
                # 关键统计卡片
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("⏰ 时间范围", className="card-title"),
                                html.H3(f"{stats['year_min']} - {stats['year_max']}", className="text-primary"),
                                html.P(f"共 {stats['year_count']} 年 × 12 月", className="text-muted mb-0")
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("🗺️ 省份数量", className="card-title"),
                                html.H3(f"{stats['province_count']}", className="text-success"),
                                html.P(f"总记录: {stats['total_records']:,} 条", className="text-muted mb-0")
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("📈 变量数量", className="card-title"),
                                html.H3(f"{stats['variable_count']}", className="text-info"),
                                html.P("温度/降水/风速/光照", className="text-muted mb-0")
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("✅ 数据完整性", className="card-title"),
                                html.H3(f"{stats['completeness']}", className="text-warning"),
                                html.P(f"缺失值: {stats['missing_count']} 个", className="text-muted mb-0")
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=3),
                ]),

                # 详细信息区域
                dbc.Row([
                    # 左侧：省份列表
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("🗺️ 省份列表", className="mb-0")),
                            dbc.CardBody([
                                html.Div([
                                    dbc.Badge(prov, color="primary", className="me-2 mb-2", pill=True)
                                    for prov in stats['provinces']
                                ], style={'max-height': '400px', 'overflow-y': 'auto'})
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=6),

                    # 右侧：变量列表
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("📊 气候变量列表", className="mb-0")),
                            dbc.CardBody([
                                html.Div([
                                    html.Div([
                                        html.Strong(var_name_map.get(var, var)),
                                        html.Br(),
                                        html.Small(var, className="text-muted")
                                    ], className="mb-3")
                                    for var in stats['numeric_columns']
                                ], style={'max-height': '400px', 'overflow-y': 'auto'})
                            ])
                        ], className="mb-3 shadow-sm")
                    ], width=6),
                ]),

                # 年份和月份信息
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("📅 时间维度详情", className="mb-0")),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("年份列表: "),
                                    html.Span(", ".join(map(str, stats['year_range'])))
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("月份列表: "),
                                    html.Span(", ".join(map(str, stats['month_range'])))
                                ], className="mb-0")
                            ])
                        ], className="shadow-sm")
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),

        # ===== Tab 2: 季节性分析 =====
        dcc.Tab(label='🌡️ 季节性分析', value='tab-seasonal', children=[
            dbc.Container([
                # 第一行：省份和变量选择
                dbc.Row([
                    dbc.Col([
                        html.Label("选择省份:", className="fw-bold"),
                        dcc.Dropdown(
                            id='province-dropdown',
                            options=[{'label': p, 'value': p} for p in ['全国平均'] + stats['provinces']],
                            value='全国平均',
                            clearable=False
                        )
                    ], width=4),

                    dbc.Col([
                        html.Label("选择变量:", className="fw-bold"),
                        dcc.Dropdown(
                            id='variable-dropdown',
                            options=[{'label': var_name_map.get(v, v), 'value': v} for v in stats['numeric_columns']],
                            value='温度_平均_2米_省内',
                            clearable=False
                        )
                    ], width=4),

                    dbc.Col([
                        html.Label("显示模式:", className="fw-bold"),
                        dcc.RadioItems(
                            id='mode-radio',
                            options=[
                                {'label': ' 模式1: 逐年对比', 'value': 'mode1'},
                                {'label': ' 模式2: 气候包络线', 'value': 'mode2'}
                            ],
                            value='mode2',
                            inline=True,
                            labelStyle={'margin-right': '20px'}
                        )
                    ], width=4)
                ], className="mb-3"),

                # 模式1的年份选择器
                dbc.Row([
                    dbc.Col([
                        html.Div(id='mode1-controls', children=[
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("快捷选择:", className="fw-bold mb-2"),
                                            dbc.ButtonGroup([
                                                dbc.Button("全选", id="btn-select-all", color="primary", size="sm",
                                                           outline=True),
                                                dbc.Button("清空", id="btn-clear-all", color="secondary", size="sm",
                                                           outline=True),
                                                dbc.Button("最近5年", id="btn-recent-5", color="info", size="sm",
                                                           outline=True),
                                                dbc.Button("最近10年", id="btn-recent-10", color="info", size="sm",
                                                           outline=True),
                                                dbc.Button("每5年", id="btn-every-5", color="success", size="sm",
                                                           outline=True),
                                            ], className="mb-3")
                                        ], width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("自定义范围:", className="fw-bold mb-2"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='year-start',
                                                        options=[{'label': str(y), 'value': y} for y in
                                                                 stats['year_range']],
                                                        value=stats['year_range'][0],
                                                        placeholder="起始年份"
                                                    )
                                                ], width=5),
                                                dbc.Col([
                                                    html.Div("至", className="text-center",
                                                             style={'line-height': '38px'})
                                                ], width=2),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='year-end',
                                                        options=[{'label': str(y), 'value': y} for y in
                                                                 stats['year_range']],
                                                        value=stats['year_range'][-1],
                                                        placeholder="结束年份"
                                                    )
                                                ], width=5),
                                            ]),
                                            dbc.Button("应用范围", id="btn-apply-range", color="primary", size="sm",
                                                       className="mt-2")
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("已选年份 (可手动调整):", className="fw-bold mt-3 mb-2"),
                                            dcc.Dropdown(
                                                id='year-dropdown-mode1',
                                                options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                                value=list(range(2020, 2025)),
                                                multi=True
                                            )
                                        ])
                                    ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("加粗显示年份:", className="fw-bold mt-3 mb-2"),
                                        dcc.Dropdown(
                                            id='highlight-year-dropdown',
                                            options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                            value=stats['year_max'],  # 默认最新年
                                            clearable=False,
                                            placeholder="选择加粗年份"
                                        )
                                    ])
                                ])
                                ])
                            ], className="shadow-sm")
                        ], style={'display': 'none'})
                    ], width=12)
                ], className="mb-3"),

                # 替换模式2的控制面板（约在第380行）
                html.Div(id='mode2-controls', children=[
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("快捷选择:", className="fw-bold mb-2"),
                                    dbc.ButtonGroup([
                                        dbc.Button("全选", id="btn-select-all-mode2", color="primary", size="sm",
                                                   outline=True),
                                        dbc.Button("清空", id="btn-clear-all-mode2", color="secondary", size="sm",
                                                   outline=True),
                                        dbc.Button("最近5年", id="btn-recent-5-mode2", color="info", size="sm",
                                                   outline=True),
                                        dbc.Button("最近10年", id="btn-recent-10-mode2", color="info", size="sm",
                                                   outline=True),
                                        dbc.Button("每5年", id="btn-every-5-mode2", color="success", size="sm",
                                                   outline=True),
                                    ], className="mb-3")
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("自定义范围:", className="fw-bold mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='year-start-mode2',
                                                options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                                value=stats['year_range'][0],
                                                placeholder="起始年份"
                                            )
                                        ], width=5),
                                        dbc.Col([
                                            html.Div("至", className="text-center", style={'line-height': '38px'})
                                        ], width=2),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='year-end-mode2',
                                                options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                                value=stats['year_range'][-1],
                                                placeholder="结束年份"
                                            )
                                        ], width=5),
                                    ]),
                                    dbc.Button("应用范围", id="btn-apply-range-mode2", color="primary", size="sm",
                                               className="mt-2")
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("已选年份 (叠加显示在气候包络线上):", className="fw-bold mt-3 mb-2"),
                                    dcc.Dropdown(
                                        id='year-dropdown-mode2',
                                        options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                        value=[],
                                        multi=True,
                                        placeholder="可手动调整叠加年份"
                                    )
                                ])
                            ])
                        ])
                    ], className="shadow-sm")
                ]),


                # 图表区域
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='seasonal-plot')
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),

        # ===== Tab 3-6: 空间分布 (美化版) =====
        dcc.Tab(label='🗺️ 空间分布', value='tab-spatial', children=[
            dbc.Container([
                # 子Tab
                dcc.Tabs(id='spatial-subtabs', value='spatial-cross', children=[
                    # ========== 子Tab 1: 横截面对比 ==========
                    dcc.Tab(label='📊 横截面对比', value='spatial-cross', children=[
                        dbc.Container([
                            # 原有的控制面板和图表
                            dbc.Card([
                                dbc.CardHeader("🛠️ 分析参数设置", className="fw-bold bg-light"),
                                dbc.CardBody([
                            # --- 第一行：数据选择 (时间、变量、基准) ---
                            dbc.Row([
                                # 1. 时间选择 (占 3/12)
                                dbc.Col([
                                    html.Label("📅 目标时间:", className="fw-bold small mb-1"),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='spatial-year',
                                                options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                                value=stats['year_max'],
                                                clearable=False,
                                                placeholder="年份"
                                            ), width=7, className="pe-1"  # padding-end-1 减少右边距
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='spatial-month',
                                                options=[{'label': f"{m}月", 'value': m} for m in range(1, 13)],
                                                value=11,
                                                clearable=False,
                                                placeholder="月份"
                                            ), width=5, className="ps-0"  # padding-start-0 减少左边距
                                        )
                                    ], className="g-0")  # g-0 去除内部网格间距
                                ], width=3),

                                # 2. 变量选择 (占 4/12)
                                dbc.Col([
                                    html.Label("📊 分析变量:", className="fw-bold small mb-1"),
                                    dcc.Dropdown(
                                        id='spatial-variable',
                                        options=[{'label': var_name_map.get(v, v), 'value': v}
                                                 for v in stats['numeric_columns']],
                                        value='光辐照_平均_地表_省内',
                                        clearable=False
                                    )
                                ], width=4),

                                # 3. 基准期选择 (占 5/12)
                                dbc.Col([
                                    html.Label("📏 气候基准期 (用于距平计算):", className="fw-bold small mb-1"),
                                    dbc.InputGroup([
                                        dcc.Dropdown(
                                            id='spatial-baseline-start',
                                            options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                            value=max(stats['year_min'], 2015),
                                            clearable=False,
                                            style={'width': '100%'}  # 强制撑满
                                        ),
                                        dbc.InputGroupText("-", className="bg-white border-0"),  # 纯文本连接符
                                        dcc.Dropdown(
                                            id='spatial-baseline-end',
                                            options=[{'label': str(y), 'value': y} for y in stats['year_range']],
                                            value=stats['year_max'],
                                            clearable=False,
                                            style={'width': '100%'}
                                        )
                                    ], className="d-flex flex-nowrap")  # 强制不换行
                                ], width=5),
                            ], className="mb-3"),  # 行间距

                            html.Hr(className="my-3 text-muted"),  # 分割线

                            # --- 第二行：排序控制 ---
                            dbc.Row([
                                # 4. 排序方式 (占 6/12)
                                dbc.Col([
                                    html.Label("📊 排序方式:", className="fw-bold small mb-2 d-block"),  # d-block 独占一行
                                    dbc.RadioItems(
                                        id='spatial-sort-method',
                                        options=[
                                            {'label': ' 🗺️ 地理分区', 'value': 'geo'},
                                            {'label': ' ⬇️ 数值降序', 'value': 'desc'},
                                            {'label': ' ⬆️ 数值升序', 'value': 'asc'},
                                        ],
                                        value='geo',
                                        inline=True,
                                        inputClassName="me-1",  # 按钮和文字的间距
                                        labelClassName="me-3"  # 选项之间的间距
                                    )
                                ], width=6, className="d-flex flex-column justify-content-center"),

                                # 5. 排序依据 (占 4/12)
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col(html.Label("🎯 排序依据:", className="fw-bold small mt-1 text-end"),
                                                width=4),
                                        dbc.Col(
                                            dbc.Select(
                                                id='spatial-sort-target',
                                                options=[
                                                    {'label': '距平 (Climate)', 'value': 'climate'},
                                                    {'label': '同比 (YoY)', 'value': 'yoy'},
                                                    {'label': '环比 (MoM)', 'value': 'mom'},
                                                ],
                                                value='climate',
                                                size="sm"  # 以此区分主要控件
                                            ), width=8
                                        )
                                    ])
                                ], width=4),

                                # 占位 (占 2/12)
                                dbc.Col([], width=2)
                            ], align="center")  # 垂直居中对齐
                        ])
                    ], className="mb-4 shadow-sm"),

                # 图表区域
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            type="default",
                            children=dcc.Graph(id='spatial-plot', style={'height': '900px'})
                        )
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),

                    # ========== 子Tab 2: 地图热力图 ==========
                    dcc.Tab(label='🗺️ 地图热力图', value='spatial-map', children=[
                        dbc.Container([
                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader("🛠️ 地图配置", className="fw-bold bg-light"),
                                dbc.CardBody([
                                    dbc.Row([
                                        # 时间选择
                                        dbc.Col([
                                            html.Label("📅 选择时间:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='map-year',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_max'],
                                                        clearable=False
                                                    )
                                                ], width=6),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='map-month',
                                                        options=[{'label': f'{m}月', 'value': m}
                                                                 for m in range(1, 13)] +
                                                                [{'label': '全年', 'value': 'all'}],
                                                        value=7,  # 默认7月(夏季)
                                                        clearable=False
                                                    )
                                                ], width=6)
                                            ])
                                        ], width=3),

                                        # 变量选择
                                        dbc.Col([
                                            html.Label("📊 选择变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='map-variable',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='温度_平均_2米_省内',
                                                clearable=False
                                            )
                                        ], width=3),

                                        # 显示类型
                                        dbc.Col([
                                            html.Label("🎨 显示类型:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='map-display-type',
                                                options=[
                                                    {'label': '绝对值', 'value': 'absolute'},
                                                    {'label': '距平值', 'value': 'anomaly'},
                                                    {'label': '同比变化', 'value': 'yoy'},
                                                    {'label': '年份段均值', 'value': 'period_mean'}
                                                ],
                                                value='absolute',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 新增：年份段选择器（仅均值模式显示）
                                        dbc.Col([
                                            html.Div(id='map-period-container', children=[
                                                html.Label("📅 年份段:", className="fw-bold"),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='map-period-start',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=1995,
                                                            clearable=False
                                                        )
                                                    ], width=6),
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='map-period-end',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=2024,
                                                            clearable=False
                                                        )
                                                    ], width=6)
                                                ])
                                            ], style={'display': 'none'})
                                        ], width=3),

                                        # 配色方案
                                        dbc.Col([
                                            html.Label("🎨 配色方案:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='map-colorscale',
                                                options=[
                                                    {'label': '红蓝(温度)', 'value': 'RdBu_r'},
                                                    {'label': '蓝色(降水)', 'value': 'Blues'},
                                                    {'label': '黄红(光照)', 'value': 'YlOrRd'},
                                                    {'label': '绿紫(风速)', 'value': 'Viridis'},
                                                    {'label': '等离子', 'value': 'Plasma'}
                                                ],
                                                value='RdBu_r',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 基准期(仅距平模式)
                                        dbc.Col([
                                            html.Div(id='map-baseline-container', children=[
                                                html.Label("📏 基准期:", className="fw-bold"),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='map-baseline-start',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=1995,
                                                            clearable=False
                                                        )
                                                    ], width=6),
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='map-baseline-end',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=2024,
                                                            clearable=False
                                                        )
                                                    ], width=6)
                                                ])
                                            ], style={'display': 'none'})
                                        ], width=2)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 地图区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        type="default",
                                        children=dcc.Graph(id='spatial-map-plot', style={'height': '700px'})
                                    )
                                ])
                            ]),

                            # 统计摘要
                            dbc.Row([
                                dbc.Col([
                                    html.H5("📋 统计摘要", className="mt-4 mb-3"),
                                    html.Div(id='map-summary')
                                ])
                            ])

                        ], fluid=True, className="mt-4")
                    ]),
                    dcc.Tab(label='📊 时间段排序', value='spatial-ranking', children=[
                        dbc.Container([
                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader("🛠️ 分析参数设置", className="fw-bold bg-light"),
                                dbc.CardBody([
                                    # 第一行：时间范围
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("📅 年份范围:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='ranking-year-start',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=2020,
                                                        clearable=False
                                                    )
                                                ], width=6),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='ranking-year-end',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=2024,
                                                        clearable=False
                                                    )
                                                ], width=6)
                                            ])
                                        ], width=3),

                                        dbc.Col([
                                            html.Label("📅 月份选择:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='ranking-months',
                                                options=[{'label': f'{m}月', 'value': m}
                                                         for m in range(1, 13)] +
                                                        [{'label': '全年', 'value': 'all'}],
                                                value='all',
                                                clearable=False
                                            )
                                        ], width=2),

                                        dbc.Col([
                                            html.Label("📊 分析变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='ranking-variable',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='温度_平均_2米_省内',
                                                clearable=False
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Label("🔢 聚合方式:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='ranking-agg-method',
                                                options=[
                                                    {'label': '均值', 'value': 'mean'},
                                                    {'label': '总和（适用降水）', 'value': 'sum'},
                                                    {'label': '最大值', 'value': 'max'},
                                                    {'label': '最小值', 'value': 'min'}
                                                ],
                                                value='mean',
                                                clearable=False
                                            )
                                        ], width=2),

                                        dbc.Col([
                                            html.Label("📏 排序方式:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='ranking-sort-method',
                                                options=[
                                                    {'label': '降序', 'value': 'desc'},
                                                    {'label': '升序', 'value': 'asc'},
                                                    {'label': '地理分区', 'value': 'geo'}
                                                ],
                                                value='desc',
                                                clearable=False
                                            )
                                        ], width=2)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 图表区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        type="default",
                                        children=dcc.Graph(id='ranking-plot', style={'height': '800px'})
                                    )
                                ])
                            ]),

                            # 统计摘要
                            dbc.Row([
                                dbc.Col([
                                    html.H5("📋 统计摘要", className="mt-4 mb-3"),
                                    html.Div(id='ranking-summary')
                                ])
                            ]),

                            # ← 新增：导出按钮
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "📥 导出数据到Excel",
                                        id='btn-export-ranking',
                                        color="success",
                                        className="mt-3"
                                    ),
                                    html.Div(id='export-ranking-status', className="mt-2")
                                ])
                            ])

                        ], fluid=True, className="mt-4")
                    ]),
                    # ========== 新增：象限分析 ==========
                    dcc.Tab(label='📊 象限分析', value='spatial-quadrant', children=[
                                    dbc.Container([
                                        # 控制面板
                                        dbc.Card([
                                            dbc.CardHeader("🛠️ 分析参数设置", className="fw-bold bg-light"),
                                            dbc.CardBody([
                                                dbc.Row([
                                                    # 时间范围
                                                    dbc.Col([
                                                        html.Label("📅 年份范围:", className="fw-bold"),
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dcc.Dropdown(
                                                                    id='quadrant-year-start',
                                                                    options=[{'label': str(y), 'value': y}
                                                                             for y in stats['year_range']],
                                                                    value=2020,
                                                                    clearable=False
                                                                )
                                                            ], width=6),
                                                            dbc.Col([
                                                                dcc.Dropdown(
                                                                    id='quadrant-year-end',
                                                                    options=[{'label': str(y), 'value': y}
                                                                             for y in stats['year_range']],
                                                                    value=2024,
                                                                    clearable=False
                                                                )
                                                            ], width=6)
                                                        ])
                                                    ], width=3),

                                                    # 月份选择
                                                    dbc.Col([
                                                        html.Label("📅 月份选择:", className="fw-bold"),
                                                        dcc.Dropdown(
                                                            id='quadrant-month',
                                                            options=[{'label': f'{m}月', 'value': m}
                                                                     for m in range(1, 13)] +
                                                                    [{'label': '全年', 'value': 'all'}],
                                                            value='all',
                                                            clearable=False
                                                        )
                                                    ], width=2),

                                                    # 变量选择
                                                    dbc.Col([
                                                        html.Label("📊 分析变量:", className="fw-bold"),
                                                        dcc.Dropdown(
                                                            id='quadrant-variable',
                                                            options=[{'label': var_name_map.get(v, v), 'value': v}
                                                                     for v in stats['numeric_columns']],
                                                            value='风速_平均_100米_省内',
                                                            clearable=False
                                                        )
                                                    ], width=3),

                                                    # 高亮省份
                                                    dbc.Col([
                                                        html.Label("🎯 高亮省份:", className="fw-bold"),
                                                        dcc.Dropdown(
                                                            id='quadrant-highlight',
                                                            options=[{'label': p, 'value': p}
                                                                     for p in stats['provinces']],
                                                            value=[],
                                                            multi=True,
                                                            placeholder="可选：高亮特定省份"
                                                        )
                                                    ], width=4)
                                                ])
                                            ])
                                        ], className="mb-4 shadow-sm"),

                                        # 象限图
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Loading(
                                                    type="default",
                                                    children=dcc.Graph(id='quadrant-plot', style={'height': '700px'})
                                                )
                                            ])
                                        ]),

                                        # 统计表格
                                        dbc.Row([
                                            dbc.Col([
                                                html.H5("📋 统计摘要表", className="mt-4 mb-3"),
                                                dcc.Loading(
                                                    type="default",
                                                    children=dcc.Graph(id='quadrant-table')
                                                )
                                            ])
                                        ]),

                                        # 导出按钮
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Button(
                                                    "📥 导出数据到Excel",
                                                    id='btn-export-quadrant',
                                                    color="success",
                                                    className="mt-3"
                                                ),
                                                html.Div(id='export-quadrant-status', className="mt-2")
                                            ])
                                        ])

                                    ], fluid=True, className="mt-4")
                    ])

                ])
            ], fluid=True, className="mt-4")
        ]),

        dcc.Tab(label='📈 趋势分析', value='tab-trend', children=[
            dbc.Container([
                # 控制面板
                dbc.Card([
                    dbc.CardBody([
                        # ========== 新增：数据范围选择 ==========
                        dbc.Row([
                            dbc.Col([
                                html.Label("📅 数据范围:", className="fw-bold"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='trend-year-start',
                                            options=[{'label': str(y), 'value': y}
                                                     for y in stats['year_range']],
                                            value=stats['year_min'],
                                            placeholder="起始年份",
                                            clearable=False
                                        )
                                    ], width=5),
                                    dbc.Col([
                                        html.Div("至", className="text-center",
                                                 style={'line-height': '38px'})
                                    ], width=2),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='trend-year-end',
                                            options=[{'label': str(y), 'value': y}
                                                     for y in stats['year_range']],
                                            value=2024,  # 默认到2024年
                                            placeholder="结束年份",
                                            clearable=False
                                        )
                                    ], width=5)
                                ])
                            ], width=3),

                            dbc.Col([
                                html.Div([
                                    html.Label("快捷选择:", className="fw-bold mb-2"),
                                    dbc.ButtonGroup([
                                        dbc.Button("全部", id="btn-range-all", color="secondary",
                                                   size="sm", outline=True),
                                        dbc.Button("1995-2024", id="btn-range-1995-2024",
                                                   color="primary", size="sm", outline=True),
                                        dbc.Button("最近10年", id="btn-range-recent-10",
                                                   color="info", size="sm", outline=True),
                                        dbc.Button("最近20年", id="btn-range-recent-20",
                                                   color="info", size="sm", outline=True),
                                    ], size="sm")
                                ])
                            ], width=9)
                        ], className="mb-3"),
                        # ========== 新增结束 ==========
                        dbc.Row([
                            # 省份选择
                            dbc.Col([
                                html.Label("选择省份:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='trend-province-dropdown',
                                    options=[{'label': p, 'value': p} for p in ['全国平均'] + stats['provinces']],
                                    value='全国平均',
                                    clearable=False
                                )
                            ], width=3),

                            # 变量选择
                            dbc.Col([
                                html.Label("选择变量:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='trend-variable-dropdown',
                                    options=[{'label': var_name_map.get(v, v), 'value': v}
                                             for v in stats['numeric_columns']],
                                    value='温度_平均_2米_省内',
                                    clearable=False
                                )
                            ], width=3),

                            # 分析类型
                            dbc.Col([
                                html.Label("分析类型:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='trend-analysis-type',
                                    options=[
                                        {'label': '📊 年度趋势', 'value': 'annual'},
                                        {'label': '🍂 季节趋势', 'value': 'seasonal'},
                                        {'label': '📅 月度趋势', 'value': 'monthly'},
                                        {'label': '📉 距平分析', 'value': 'anomaly'},
                                        {'label': '🔄 年际变化', 'value': 'interannual'}
                                    ],
                                    value='annual',
                                    clearable=False
                                )
                            ], width=3),

                            # 聚合方法（所有类型都显示，除了年际变化）
                            dbc.Col([
                                html.Label("聚合方法:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='trend-agg-method',
                                    options=[
                                        {'label': '均值', 'value': 'mean'},
                                        {'label': '总和 (适用于降水)', 'value': 'sum'}
                                    ],
                                    value='mean',
                                    clearable=False
                                )
                            ], width=3)
                        ], className="mb-3"),

                        # 月份选择（仅月度趋势）
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='month-selection-container', children=[
                                    html.Label("选择月份 (月度趋势):", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='trend-month-selection',
                                        options=[{'label': f'{m}月', 'value': m} for m in range(1, 13)],
                                        value=[1, 7],
                                        multi=True
                                    )
                                ], style={'display': 'none'})
                            ], width=6),

                            # 基准期选择（仅距平分析）
                            dbc.Col([
                                html.Div(id='baseline-container', children=[
                                    html.Label("基准期:", className="fw-bold"),
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='baseline-start',
                                                options=[{'label': str(y), 'value': y}
                                                         for y in stats['year_range']],
                                                value=1995,
                                                placeholder="起始年"
                                            )
                                        ], width=5),
                                        dbc.Col(html.Div("至", className="text-center",
                                                         style={'line-height': '38px'}), width=2),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='baseline-end',
                                                options=[{'label': str(y), 'value': y}
                                                         for y in stats['year_range']],
                                                value=2024,
                                                placeholder="结束年"
                                            )
                                        ], width=5)
                                    ])
                                ], style={'display': 'none'})
                            ], width=6)
                        ]),

                        # 对比功能
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    id='trend-show-comparison',
                                    options=[{'label': ' 启用对比功能', 'value': 'show'}],
                                    value=[],
                                    switch=True
                                )
                            ], width=3),

                            dbc.Col([
                                html.Div(id='comparison-controls', children=[
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id='trend-compare-provinces',
                                                options=[{'label': p, 'value': p}
                                                         for p in ['全国平均'] + stats['provinces']],
                                                value=[],
                                                multi=True,
                                                placeholder="选择对比省份"
                                            )
                                        ], width=9),
                                        dbc.Col([
                                            dbc.Checklist(
                                                id='trend-show-national',
                                                options=[{'label': '全国', 'value': 'show'}],
                                                value=[],
                                                inline=True
                                            )
                                        ], width=3)
                                    ])
                                ], style={'display': 'none'})
                            ], width=9)
                        ], className="mt-3")
                    ])
                ], className="mb-4 shadow-sm"),

                # 图表区域
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-trend",
                            type="default",
                            children=dcc.Graph(id='trend-plot', style={'height': '700px'})
                        )
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),

        dcc.Tab(label='⚡ 极端事件', value='tab-extreme', children=[
            dbc.Container([
                # 子Tab
                dcc.Tabs(id='extreme-subtabs', value='extreme-stat', children=[

                    # ========== 子Tab 1: 极端事件统计 ==========
                    dcc.Tab(label='📊 极端事件统计', value='extreme-stat', children=[
                        dbc.Container([
                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader(html.H5("⚙️ 检测配置", className="mb-0")),
                                dbc.CardBody([
                                    dbc.Row([
                                        # 第一行：基本选择
                                        dbc.Col([
                                            html.Label("选择变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='extreme-variable-dropdown',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value=stats['numeric_columns'][0],
                                                clearable=False
                                            )
                                        ], width=4),

                                        dbc.Col([
                                            html.Label("检测方法:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='extreme-method-dropdown',
                                                options=[
                                                    {'label': '百分位法', 'value': 'percentile'},
                                                    {'label': 'Z-score法', 'value': 'zscore'},
                                                    {'label': '变异系数法', 'value': 'cv'}
                                                ],
                                                value='percentile',
                                                clearable=False
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Label("显示模式:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='extreme-display-mode',
                                                options=[
                                                    {'label': '单省详细', 'value': 'single'},
                                                    {'label': '全国汇总', 'value': 'national'},
                                                    {'label': '省份对比', 'value': 'provinces'}
                                                ],
                                                value='single',
                                                clearable=False
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Div(id='extreme-province-selector', children=[
                                                html.Label("选择省份:", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id='extreme-province-dropdown',
                                                    options=[{'label': p, 'value': p}
                                                             for p in sorted(df_with_national['province'].unique())],
                                                    value='全国平均',
                                                    clearable=False
                                                )
                                            ])
                                        ], width=2)
                                    ], className="mb-3"),

                                    # 第二行：方法参数
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(id='percentile-controls', children=[
                                                html.Label("上分位数 (%):", className="fw-bold"),
                                                dcc.Slider(
                                                    id='upper-percentile-slider',
                                                    min=90, max=99, step=1, value=95,
                                                    marks={i: str(i) for i in range(90, 100, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ])
                                        ], width=4),

                                        dbc.Col([
                                            html.Div(id='percentile-controls-lower', children=[
                                                html.Label("下分位数 (%):", className="fw-bold"),
                                                dcc.Slider(
                                                    id='lower-percentile-slider',
                                                    min=1, max=10, step=1, value=5,
                                                    marks={i: str(i) for i in range(1, 11, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ])
                                        ], width=4),

                                        dbc.Col([
                                            html.Div(id='zscore-controls', children=[
                                                html.Label("Z-score阈值:", className="fw-bold"),
                                                dcc.Slider(
                                                    id='zscore-threshold-slider',
                                                    min=1.5, max=3.5, step=0.1, value=2.0,
                                                    marks={i: str(i) for i in [1.5, 2.0, 2.5, 3.0, 3.5]},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], style={'display': 'none'})
                                        ], width=4),

                                        dbc.Col([
                                            html.Div(id='cv-controls', children=[
                                                html.Label("变异系数阈值:", className="fw-bold"),
                                                dcc.Slider(
                                                    id='cv-threshold-slider',
                                                    min=0.1, max=0.5, step=0.05, value=0.3,
                                                    marks={i: f'{i:.1f}' for i in [0.1, 0.2, 0.3, 0.4, 0.5]},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], style={'display': 'none'})
                                        ], width=4)
                                    ], className="mb-3"),

                                    # 第三行：基准期设置
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("基准期起始年:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='baseline-start-dropdown',
                                                options=[{'label': str(y), 'value': y}
                                                         for y in stats['year_range']],
                                                value=stats['year_min'],
                                                clearable=False
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Label("基准期结束年:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='baseline-end-dropdown',
                                                options=[{'label': str(y), 'value': y}
                                                         for y in stats['year_range']],
                                                value=stats['year_max'],
                                                clearable=False
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Label(" ", className="fw-bold"),
                                            dbc.Button(
                                                "🔄 重新检测",
                                                id='extreme-detect-button',
                                                color="primary",
                                                className="w-100"
                                            )
                                        ], width=2)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 图表区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-extreme-heatmap",
                                        type="default",
                                        children=dcc.Graph(id='extreme-heatmap')
                                    )
                                ])
                            ]),

                            # 统计信息
                            dbc.Row([
                                dbc.Col([
                                    html.H5("📋 统计摘要", className="mt-4 mb-3"),
                                    html.Div(id='extreme-summary')
                                ])
                            ])

                        ], fluid=True, className="mt-4")
                    ]),

                    # ========== 子Tab 2: ENSO关联分析 ==========
                    dcc.Tab(label='🌊 ENSO关联', value='enso-analysis', children=[
                        dbc.Container([
                            # 当前ENSO状态卡
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card(id='enso-status-card', className="mb-3")
                                ])
                            ]),
                            # 均值偏移热力图
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='enso-zmean-heatmap')
                                ])
                            ], className="mb-3"),
                            dbc.Card([
                                dbc.CardHeader(html.H5("⚙️ 分析配置", className="mb-0")),
                                dbc.CardBody([
                                    dbc.Row([
                                        # 第一行：基本选择
                                        dbc.Col([
                                            html.Label("分析类型:", className="fw-bold"),
                                            dcc.RadioItems(
                                                id='enso-analysis-type',
                                                options=[
                                                    {'label': ' 散点图（相关性）', 'value': 'scatter'},
                                                    {'label': ' 时序对比', 'value': 'timeseries'}
                                                ],
                                                value='scatter',
                                                inline=True
                                            )
                                        ], width=4),

                                        dbc.Col([
                                            html.Label("选择变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='enso-variable-dropdown',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value=[stats['numeric_columns'][0]],
                                                multi=True
                                            )
                                        ], width=4),

                                        dbc.Col([
                                            html.Label("选择省份:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='enso-province-dropdown',
                                                options=[{'label': '全部省份', 'value': '全部'}] +
                                                        [{'label': p, 'value': p}
                                                         for p in sorted(df_with_national['province'].unique())],
                                                value='全国平均',
                                                multi=False
                                            )
                                        ], width=4)
                                    ], className="mb-3"),

                                    # 第二行：ENSO参数
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("El Niño阈值:", className="fw-bold"),
                                            dcc.Input(
                                                id='el-nino-threshold',
                                                type='number',
                                                value=0.5,
                                                step=0.1,
                                                className="form-control"
                                            )
                                        ], width=2),

                                        dbc.Col([
                                            html.Label("La Niña阈值:", className="fw-bold"),
                                            dcc.Input(
                                                id='la-nina-threshold',
                                                type='number',
                                                value=-0.5,
                                                step=0.1,
                                                className="form-control"
                                            )
                                        ], width=2),

                                        dbc.Col([
                                            html.Label("滞后月数:", className="fw-bold"),
                                            dcc.Slider(
                                                id='enso-lag-slider',
                                                min=0, max=12, step=1, value=0,
                                                marks={i: f'{i}月' for i in range(0, 13, 3)},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ], width=4),

                                        dbc.Col([
                                            html.Label("距平类型:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='anomaly-type-dropdown',
                                                options=[
                                                    {'label': '月度距平', 'value': 'monthly'},
                                                    {'label': '季节距平', 'value': 'seasonal'},
                                                    {'label': '年度距平', 'value': 'annual'}
                                                ],
                                                value='monthly',
                                                clearable=False
                                            )
                                        ], width=2),

                                        dbc.Col([
                                            html.Label(" ", className="fw-bold"),
                                            dbc.Button(
                                                "🔍 自动寻找最佳滞后",
                                                id='auto-lag-button',
                                                color="info",
                                                className="w-100"
                                            )
                                        ], width=2)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 最佳滞后期显示
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='best-lag-display', className="alert alert-info",
                                             style={'display': 'none'})
                                ])
                            ], className="mb-3"),

                            # 图表区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-enso-plot",
                                        type="default",
                                        children=dcc.Graph(id='enso-plot')
                                    )
                                ])
                            ]),

                            # 频次统计表
                            dbc.Row([
                                dbc.Col([
                                    html.H5("📊 ENSO阶段极端事件频次", className="mt-4 mb-3"),
                                    dcc.Loading(
                                        id="loading-enso-frequency",
                                        type="default",
                                        children=dcc.Graph(id='enso-frequency-table')
                                    )
                                ])
                            ])

                        ], fluid=True, className="mt-4")
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),

        # ========== Tab 5: 多变量联动分析 ==========
        dcc.Tab(label='🔗 多变量联动', value='tab-multivar', children=[
            dbc.Container([

                # 子Tab导航
                dcc.Tabs(id='multivar-subtabs', value='multivar-corr', children=[

                    # ========== 子Tab 1: 相关性矩阵 ==========
                    dcc.Tab(label='📊 相关性矩阵', value='multivar-corr', children=[
                        dbc.Container([

                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader("🛠️ 分析参数设置", className="fw-bold bg-light"),
                                dbc.CardBody([
                                    dbc.Row([
                                        # 时间范围
                                        dbc.Col([
                                            html.Label("📅 时间范围:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='corr-year-start',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_min'],
                                                        clearable=False
                                                    )
                                                ], width=6),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='corr-year-end',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_max'],
                                                        clearable=False
                                                    )
                                                ], width=6)
                                            ])
                                        ], width=3),

                                        # 变量选择
                                        dbc.Col([
                                            html.Label("📊 选择变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='corr-variables',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value=stats['numeric_columns'][:5],  # 默认前5个
                                                multi=True,
                                                placeholder="选择至少2个变量"
                                            )
                                        ], width=4),

                                        # ENSO筛选
                                        dbc.Col([
                                            html.Label("🌊 ENSO状态:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='corr-enso-filter',
                                                options=[
                                                    {'label': '全部', 'value': '全部'},
                                                    {'label': '厄尔尼诺', 'value': 'El Niño'},
                                                    {'label': '拉尼娜', 'value': 'La Niña'},
                                                    {'label': '中性', 'value': 'Neutral'}
                                                ],
                                                value='全部',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 相关性方法
                                        dbc.Col([
                                            html.Label("📐 计算方法:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='corr-method',
                                                options=[
                                                    {'label': 'Pearson (线性)', 'value': 'pearson'},
                                                    {'label': 'Spearman (秩相关)', 'value': 'spearman'}
                                                ],
                                                value='pearson',
                                                clearable=False
                                            )
                                        ], width=3)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 图表区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        type="default",
                                        children=dcc.Graph(id='corr-matrix-plot')
                                    )
                                ], width=8),

                                # 说明文字
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("📖 使用说明", className="fw-bold"),
                                        dbc.CardBody([
                                            html.P([
                                                html.Strong("相关系数范围："),
                                                html.Br(),
                                                "• -1 到 1",
                                                html.Br(),
                                                "• 正值：正相关",
                                                html.Br(),
                                                "• 负值：负相关",
                                                html.Br(),
                                                "• 接近0：无相关"
                                            ], className="small mb-3"),

                                            html.P([
                                                html.Strong("显著性标记："),
                                                html.Br(),
                                                "• *** : p < 0.001",
                                                html.Br(),
                                                "• ** : p < 0.01",
                                                html.Br(),
                                                "• * : p < 0.05"
                                            ], className="small mb-3"),

                                            html.P([
                                                html.Strong("颜色说明："),
                                                html.Br(),
                                                "• 红色：正相关",
                                                html.Br(),
                                                "• 蓝色：负相关",
                                                html.Br(),
                                                "• 颜色越深，相关性越强"
                                            ], className="small")
                                        ])
                                    ], className="shadow-sm")
                                ], width=4)
                            ])

                        ], fluid=True, className="mt-4")
                    ]),

                    # ========== 子Tab 2: 双变量散点图 ==========
                    dcc.Tab(label='🎯 双变量散点', value='multivar-scatter', children=[
                        dbc.Container([

                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader("🛠️ 分析参数设置", className="fw-bold bg-light"),
                                dbc.CardBody([
                                    dbc.Row([
                                        # X轴变量
                                        dbc.Col([
                                            html.Label("📊 X轴变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='scatter-x-var',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='温度_平均_2米_省内',
                                                clearable=False
                                            )
                                        ], width=3),

                                        # Y轴变量
                                        dbc.Col([
                                            html.Label("📊 Y轴变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='scatter-y-var',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='地表总降水_平均_省内',
                                                clearable=False
                                            )
                                        ], width=3),

                                        # 时间范围
                                        dbc.Col([
                                            html.Label("📅 时间范围:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='scatter-year-start',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_min'],
                                                        clearable=False
                                                    )
                                                ], width=6),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='scatter-year-end',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_max'],
                                                        clearable=False
                                                    )
                                                ], width=6)
                                            ])
                                        ], width=3),

                                        # ENSO筛选
                                        dbc.Col([
                                            html.Label("🌊 ENSO状态:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='scatter-enso-filter',
                                                options=[
                                                    {'label': '全部', 'value': '全部'},
                                                    {'label': '厄尔尼诺', 'value': 'El Niño'},
                                                    {'label': '拉尼娜', 'value': 'La Niña'},
                                                    {'label': '中性', 'value': 'Neutral'}
                                                ],
                                                value='全部',
                                                clearable=False
                                            )
                                        ], width=3)
                                    ], className="mb-3"),

                                    html.Hr(className="my-3"),

                                    dbc.Row([
                                        # 颜色分组
                                        dbc.Col([
                                            html.Label("🎨 颜色分组:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='scatter-color-by',
                                                options=[
                                                    {'label': '按省份', 'value': 'province'},
                                                    {'label': '按ENSO状态', 'value': 'ENSO_state'},
                                                    {'label': '按季节', 'value': 'season'},
                                                    {'label': '按年份', 'value': '年'}
                                                ],
                                                value='ENSO_state',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 气泡大小
                                        dbc.Col([
                                            html.Label("⚪ 气泡大小:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='scatter-size-var',
                                                options=[{'label': '无', 'value': 'none'}] +
                                                        [{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='none',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 显示选项
                                        dbc.Col([
                                            html.Label("🔧 显示选项:", className="fw-bold"),
                                            dbc.Checklist(
                                                id='scatter-options',
                                                options=[
                                                    {'label': ' 回归线', 'value': 'regression'},
                                                    {'label': ' 置信区间', 'value': 'confidence'},
                                                    {'label': ' 密度等高线', 'value': 'density'}
                                                ],
                                                value=['regression', 'confidence'],
                                                inline=True,
                                                switch=True
                                            )
                                        ], width=5),

                                        dbc.Col([], width=3)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 图表和统计信息
                            dbc.Row([
                                # 散点图
                                dbc.Col([
                                    dcc.Loading(
                                        type="default",
                                        children=dcc.Graph(id='scatter-plot')
                                    )
                                ], width=9),

                                # 统计信息
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("📈 回归统计", className="fw-bold"),
                                        dbc.CardBody([
                                            html.Div(id='scatter-stats', className="small")
                                        ])
                                    ], className="shadow-sm")
                                ], width=3)
                            ])

                        ], fluid=True, className="mt-4")
                    ]),

                    # ========== 子Tab 3: 空间对比 ==========
                    dcc.Tab(label='🗺️ 空间对比', value='multivar-map', children=[
                        dbc.Container([

                            # 控制面板
                            dbc.Card([
                                dbc.CardHeader("🛠️ 地图配置", className="fw-bold bg-light"),
                                dbc.CardBody([
                                    # 第一行：时间和变量选择
                                    dbc.Row([
                                        # 时间选择
                                        dbc.Col([
                                            html.Label("📅 选择时间:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='dual-map-year',
                                                        options=[{'label': str(y), 'value': y}
                                                                 for y in stats['year_range']],
                                                        value=stats['year_max'],
                                                        clearable=False
                                                    )
                                                ], width=6),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id='dual-map-month',
                                                        options=[{'label': f'{m}月', 'value': m}
                                                                 for m in range(1, 13)],
                                                        value=7,
                                                        clearable=False
                                                    )
                                                ], width=6)
                                            ])
                                        ], width=3),

                                        # 左侧地图变量
                                        dbc.Col([
                                            html.Label("📊 左侧地图变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-var1',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='温度_平均_2米_省内',
                                                clearable=False
                                            )
                                        ], width=4),

                                        # 右侧地图变量
                                        dbc.Col([
                                            html.Label("📊 右侧地图变量:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-var2',
                                                options=[{'label': var_name_map.get(v, v), 'value': v}
                                                         for v in stats['numeric_columns']],
                                                value='地表总降水_平均_省内',
                                                clearable=False
                                            )
                                        ], width=4)
                                    ], className="mb-3"),

                                    html.Hr(className="my-3"),

                                    # 第二行：显示类型和配色
                                    dbc.Row([
                                        # 左侧显示类型
                                        dbc.Col([
                                            html.Label("🎨 左侧显示类型:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-display-type1',
                                                options=[
                                                    {'label': '绝对值', 'value': 'absolute'},
                                                    {'label': '距平值', 'value': 'anomaly'},
                                                    {'label': '同比变化', 'value': 'yoy'}
                                                ],
                                                value='absolute',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 左侧配色
                                        dbc.Col([
                                            html.Label("🎨 左侧配色:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-colorscale1',
                                                options=[
                                                    {'label': '红蓝(温度)', 'value': 'RdBu_r'},
                                                    {'label': '蓝色(降水)', 'value': 'Blues'},
                                                    {'label': '黄红(光照)', 'value': 'YlOrRd'},
                                                    {'label': '绿紫(风速)', 'value': 'Viridis'},
                                                    {'label': '等离子', 'value': 'Plasma'}
                                                ],
                                                value='RdBu_r',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 右侧显示类型
                                        dbc.Col([
                                            html.Label("🎨 右侧显示类型:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-display-type2',
                                                options=[
                                                    {'label': '绝对值', 'value': 'absolute'},
                                                    {'label': '距平值', 'value': 'anomaly'},
                                                    {'label': '同比变化', 'value': 'yoy'}
                                                ],
                                                value='absolute',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 右侧配色
                                        dbc.Col([
                                            html.Label("🎨 右侧配色:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-colorscale2',
                                                options=[
                                                    {'label': '红蓝(温度)', 'value': 'RdBu_r'},
                                                    {'label': '蓝色(降水)', 'value': 'Blues'},
                                                    {'label': '黄红(光照)', 'value': 'YlOrRd'},
                                                    {'label': '绿紫(风速)', 'value': 'Viridis'},
                                                    {'label': '等离子', 'value': 'Plasma'}
                                                ],
                                                value='Blues',
                                                clearable=False
                                            )
                                        ], width=2),

                                        # 地图样式
                                        dbc.Col([
                                            html.Label("🗺️ 地图样式:", className="fw-bold"),
                                            dcc.Dropdown(
                                                id='dual-map-style',
                                                options=[
                                                    {'label': '明亮', 'value': 'carto-positron'},
                                                    {'label': '街道', 'value': 'open-street-map'},
                                                    {'label': '白色', 'value': 'white-bg'},
                                                    {'label': '暗黑', 'value': 'carto-darkmatter'}
                                                ],
                                                value='carto-positron',
                                                clearable=False
                                            )
                                        ], width=2)
                                    ], className="mb-3"),

                                    html.Hr(className="my-3"),

                                    # 第三行：基准期设置（仅距平模式显示）
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(id='dual-map-baseline-container', children=[
                                                html.Label("📏 基准期（用于距平计算）:", className="fw-bold"),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='dual-map-baseline-start',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=1995,
                                                            clearable=False
                                                        )
                                                    ], width=5),
                                                    dbc.Col([
                                                        html.Div("至", className="text-center",
                                                                 style={'line-height': '38px'})
                                                    ], width=2),
                                                    dbc.Col([
                                                        dcc.Dropdown(
                                                            id='dual-map-baseline-end',
                                                            options=[{'label': str(y), 'value': y}
                                                                     for y in stats['year_range']],
                                                            value=2024,
                                                            clearable=False
                                                        )
                                                    ], width=5)
                                                ])
                                            ], style={'display': 'none'})
                                        ], width=6)
                                    ])
                                ])
                            ], className="mb-4 shadow-sm"),

                            # 地图区域
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        type="default",
                                        children=dcc.Graph(id='dual-map-plot', style={'height': '700px'})
                                    )
                                ])
                            ]),

                            # 说明文字
                            dbc.Row([
                                dbc.Col([
                                    dbc.Alert([
                                        html.H5("💡 使用提示", className="alert-heading"),
                                        html.Ul([
                                            html.Li(
                                                "左右两张地图显示同一时间点的不同变量，可以直观对比它们的空间分布模式"),
                                            html.Li(
                                                "支持三种显示类型：绝对值、距平值（相对气候均值）、同比变化（相对去年同月）"),
                                            html.Li("可为每张地图独立设置配色方案，更好地突出各自特征"),
                                            html.Li("蒙东和蒙西已自动合并为内蒙古自治区")
                                        ], className="mb-0")
                                    ], color="info", className="mt-3")
                                ])
                            ])

                        ], fluid=True, className="mt-4")
                    ])

                ])

            ], fluid=True, className="mt-4")
        ])

    ])
], fluid=True, className="p-4")


# ============================================================================
# 回调函数
# ============================================================================

@app.callback(
    [Output('mode1-controls', 'style'),
     Output('mode2-controls', 'style')],
    Input('mode-radio', 'value')
)
def toggle_mode_controls(mode):
    """根据模式切换控制面板"""
    if mode == 'mode1':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


# 模式1的快捷按钮回调
@app.callback(
    Output('year-dropdown-mode1', 'value'),
    [Input('btn-select-all', 'n_clicks'),
     Input('btn-clear-all', 'n_clicks'),
     Input('btn-recent-5', 'n_clicks'),
     Input('btn-recent-10', 'n_clicks'),
     Input('btn-every-5', 'n_clicks'),
     Input('btn-apply-range', 'n_clicks')],
    [State('year-start', 'value'),
     State('year-end', 'value')],
    prevent_initial_call=True
)
def update_year_selection_mode1(btn_all, btn_clear, btn_5, btn_10, btn_every5, btn_range,
                                year_start, year_end):
    """处理模式1的快捷按钮"""
    from dash import callback_context

    if not callback_context.triggered:
        return list(range(2020, 2025))

    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-select-all':
        return stats['year_range']
    elif button_id == 'btn-clear-all':
        return []
    elif button_id == 'btn-recent-5':
        return stats['year_range'][-5:]
    elif button_id == 'btn-recent-10':
        return stats['year_range'][-10:]
    elif button_id == 'btn-every-5':
        return [y for y in stats['year_range'] if y % 5 == 0]
    elif button_id == 'btn-apply-range':
        if year_start and year_end:
            return [y for y in stats['year_range'] if year_start <= y <= year_end]

    return list(range(2020, 2025))


# 替换模式2的回调函数（约在第490行）
@app.callback(
    Output('year-dropdown-mode2', 'value'),
    [Input('btn-select-all-mode2', 'n_clicks'),
     Input('btn-clear-all-mode2', 'n_clicks'),
     Input('btn-recent-5-mode2', 'n_clicks'),
     Input('btn-recent-10-mode2', 'n_clicks'),
     Input('btn-every-5-mode2', 'n_clicks'),
     Input('btn-apply-range-mode2', 'n_clicks')],
    [State('year-start-mode2', 'value'),
     State('year-end-mode2', 'value')],
    prevent_initial_call=True
)
def update_year_selection_mode2(btn_all, btn_clear, btn_5, btn_10, btn_every5, btn_range,
                                year_start, year_end):
    """处理模式2的快捷按钮"""
    from dash import callback_context

    if not callback_context.triggered:
        return []

    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-select-all-mode2':
        return stats['year_range']
    elif button_id == 'btn-clear-all-mode2':
        return []
    elif button_id == 'btn-recent-5-mode2':
        return stats['year_range'][-5:]
    elif button_id == 'btn-recent-10-mode2':
        return stats['year_range'][-10:]
    elif button_id == 'btn-every-5-mode2':
        return [y for y in stats['year_range'] if y % 5 == 0]
    elif button_id == 'btn-apply-range-mode2':
        if year_start and year_end:
            return [y for y in stats['year_range'] if year_start <= y <= year_end]

    return []


@app.callback(
    Output('seasonal-plot', 'figure'),
    [Input('province-dropdown', 'value'),
     Input('variable-dropdown', 'value'),
     Input('mode-radio', 'value'),
     Input('year-dropdown-mode1', 'value'),
     Input('year-dropdown-mode2', 'value'),
     Input('highlight-year-dropdown', 'value')]
)
def update_seasonal_plot(province, variable, mode, years_mode1, years_mode2, highlight_year):
    """更新季节性分析图表"""
    if mode == 'mode1':
        return create_seasonal_plot_mode1(
            df_with_national, province, variable,
            years_mode1, highlight_year  # 传入高亮年份
        )
    else:
        return create_seasonal_plot_mode2(df_with_national, province, variable, years_mode2)


# ============================================================================
# 趋势分析回调
# ============================================================================


@app.callback(
    Output('comparison-controls', 'style'),
    Input('trend-show-comparison', 'value')
)
def toggle_comparison_controls(show_comparison):
    """显示/隐藏对比控件"""
    if 'show' in show_comparison:
        return {'display': 'block'}
    return {'display': 'none'}


# ========== 新增：数据范围快捷按钮回调 ==========
@app.callback(
    [Output('trend-year-start', 'value'),
     Output('trend-year-end', 'value')],
    [Input('btn-range-all', 'n_clicks'),
     Input('btn-range-1995-2024', 'n_clicks'),
     Input('btn-range-recent-10', 'n_clicks'),
     Input('btn-range-recent-20', 'n_clicks')],
    prevent_initial_call=True
)
def update_trend_date_range(btn_all, btn_1995_2024, btn_10, btn_20):
    """处理数据范围快捷按钮"""
    from dash import callback_context

    if not callback_context.triggered:
        return stats['year_min'], 2024

    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-range-all':
        return stats['year_min'], stats['year_max']
    elif button_id == 'btn-range-1995-2024':
        return 1995, 2024
    elif button_id == 'btn-range-recent-10':
        return max(stats['year_min'], 2024 - 9), 2024
    elif button_id == 'btn-range-recent-20':
        return max(stats['year_min'], 2024 - 19), 2024

    return stats['year_min'], 2024


# ========== 新增结束 ==========


# ========== 新增：控制月份和基准期显示 ==========
@app.callback(
    [Output('month-selection-container', 'style'),
     Output('baseline-container', 'style')],
    Input('trend-analysis-type', 'value')
)
def toggle_trend_specific_controls(analysis_type):
    """根据分析类型显示/隐藏特定控件"""
    show_month = {'display': 'block'} if analysis_type == 'monthly' else {'display': 'none'}
    show_baseline = {'display': 'block'} if analysis_type == 'anomaly' else {'display': 'none'}

    return show_month, show_baseline


@app.callback(
    [Output('map-baseline-container', 'style'),
     Output('map-period-container', 'style'),
     Output('map-year', 'disabled')],  # ← 均值模式禁用年份选择
    Input('map-display-type', 'value')
)
def toggle_map_controls(display_type):
    show = {'display': 'block'}
    hide = {'display': 'none'}

    if display_type == 'anomaly':
        return show, hide, False
    elif display_type == 'period_mean':
        return hide, show, True  # 均值模式禁用单年选择
    else:
        return hide, hide, False

@app.callback(
    Output('trend-plot', 'figure'),
    [Input('trend-province-dropdown', 'value'),
     Input('trend-variable-dropdown', 'value'),
     Input('trend-analysis-type', 'value'),
     Input('trend-agg-method', 'value'),
     Input('trend-month-selection', 'value'),
     Input('trend-year-start', 'value'),  # 新增
     Input('trend-year-end', 'value'),  # 新增
     Input('baseline-start', 'value'),
     Input('baseline-end', 'value'),
     Input('trend-compare-provinces', 'value'),
     Input('trend-show-national', 'value')]
)
def update_trend_plot(province, variable, analysis_type, agg_method,
                      selected_months, year_start, year_end, baseline_start, baseline_end,
                      compare_provinces, show_national):
    """更新趋势分析图表"""

    df_filtered = df_with_national.copy()
    if year_start and year_end:
        df_filtered = df_filtered[
            (df_filtered['年'] >= year_start) &
            (df_filtered['年'] <= year_end)
            ]

    show_national_bool = 'show' in show_national if show_national else False

    if analysis_type == 'annual':
        return create_annual_trend_plot(
            df_filtered, province, variable, agg_method,
            compare_provinces, show_national_bool
        )

    elif analysis_type == 'seasonal':
        return create_seasonal_trend_plot(
            df_filtered, province, variable, agg_method,
            compare_provinces, show_national_bool
        )

    elif analysis_type == 'monthly':
        if not selected_months:
            selected_months = [1, 7]
        return create_monthly_specific_trend_plot(
            df_filtered, province, variable, selected_months, agg_method,
            compare_provinces, show_national_bool
        )

    elif analysis_type == 'anomaly':
        baseline_years = (baseline_start, baseline_end) if baseline_start and baseline_end else None
        return create_anomaly_plot(
            df_filtered, province, variable, baseline_years, agg_method,
            compare_provinces, show_national_bool
        )

    elif analysis_type == 'interannual':
        # 年际变化分析使用固定的均值方法
        return create_interannual_variability_plot(
            df_filtered, province, variable
        )

    # 默认返回空图
    return go.Figure()


# ============================================================================
# 空间分布回调 (新增)
# ============================================================================

@app.callback(
    Output('spatial-plot', 'figure'),
    Output('spatial-sort-target', 'disabled'),  # 增加一个 Output 控制禁用状态
    [Input('spatial-year', 'value'),
     Input('spatial-month', 'value'),
     Input('spatial-variable', 'value'),
     Input('spatial-baseline-start', 'value'),
     Input('spatial-baseline-end', 'value'),
     Input('spatial-sort-method', 'value'),
     Input('spatial-sort-target', 'value')]  # 新增输入
)
def update_spatial_plot(year, month, variable, baseline_start, baseline_end, sort_method, sort_target):
    """
    更新空间横截面分析图
    """
    # 逻辑：如果选择了“地理排序”，则“排序依据”下拉框应该被禁用，因为没意义
    target_disabled = (sort_method == 'geo')

    if not all([year, month, variable]):
        return go.Figure(), target_disabled

    baseline_years = (baseline_start, baseline_end) if baseline_start and baseline_end else None

    fig = create_cross_sectional_comparison(
        df_with_national,
        target_year=year,
        target_month=month,
        variable=variable,
        baseline_years=baseline_years,
        sort_method=sort_method,
        sort_target=sort_target  # 传入新参数
    )

    return fig, target_disabled


# ========== 地图热力图回调 ==========


@app.callback(
    [Output('spatial-map-plot', 'figure'),
     Output('map-summary', 'children')],
    [Input('map-year', 'value'),
     Input('map-month', 'value'),
     Input('map-variable', 'value'),
     Input('map-display-type', 'value'),
     Input('map-colorscale', 'value'),
     Input('map-baseline-start', 'value'),
     Input('map-baseline-end', 'value'),
     Input('map-period-start', 'value'),
     Input('map-period-end', 'value')]
)
def update_spatial_map(year, month, variable, display_type, colorscale,
                       baseline_start, baseline_end, period_start, period_end):
    """更新地图热力图"""

    # 参数验证
    if not variable:
        return go.Figure(), html.Div()

    # 基准期
    baseline_years = None
    if display_type == 'anomaly' and baseline_start and baseline_end:
        baseline_years = (baseline_start, baseline_end)

    period_years = None
    if display_type == 'period_mean' and period_start and period_end:
        period_years = (period_start, period_end)

    var_display = var_name_map.get(variable, variable)

    # 生成地图
    fig = create_china_map_heatmap(
        df_with_national,
        year, month, variable,
        display_type, baseline_years,
        colorscale, var_display,
        period_years=period_years
    )

    # ========== 统计摘要 ==========
    summary = None  # ← 关键：先初始化

    if display_type == 'period_mean' and period_years:
        # 年份段均值的统计
        start_year, end_year = period_years
        summary_df = df_with_national[
            (df_with_national['年'] >= start_year) &
            (df_with_national['年'] <= end_year) &
            (df_with_national['province'] != '全国平均')
            ].copy()

        # 处理月份筛选
        if month != 'all':
            summary_df = summary_df[summary_df['月'] == month]

        if len(summary_df) > 0:
            # 按省份聚合（地图用，需要合并蒙东蒙西）
            province_mapping = {'蒙东': '内蒙古自治区', '蒙西': '内蒙古自治区'}
            summary_df['province_std'] = summary_df['province'].replace(province_mapping)
            summary_df = summary_df.groupby('province_std').agg({
                variable: 'mean'
            }).reset_index()

            values = summary_df[variable]

            # ← 这里补上 summary 的赋值
            summary = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("最大值", className="text-muted"),
                            html.H4(f"{values.max():.2f}", className="text-danger"),
                            html.P(f"{summary_df.loc[values.idxmax(), 'province_std']}",
                                   className="text-muted small mb-0")
                        ], width=2),
                        dbc.Col([
                            html.H6("最小值", className="text-muted"),
                            html.H4(f"{values.min():.2f}", className="text-info"),
                            html.P(f"{summary_df.loc[values.idxmin(), 'province_std']}",
                                   className="text-muted small mb-0")
                        ], width=2),
                        dbc.Col([
                            html.H6("平均值", className="text-muted"),
                            html.H4(f"{values.mean():.2f}", className="text-primary")
                        ], width=2),
                        dbc.Col([
                            html.H6("标准差", className="text-muted"),
                            html.H4(f"{values.std():.2f}", className="text-warning")
                        ], width=2),
                        dbc.Col([
                            html.H6("极差", className="text-muted"),
                            html.H4(f"{values.max() - values.min():.2f}", className="text-success")
                        ], width=2),
                        dbc.Col([
                            html.H6("变异系数", className="text-muted"),
                            html.H4(f"{(values.std() / values.mean() * 100):.1f}%",
                                    className="text-secondary")
                        ], width=2)
                    ])
                ])
            ], className="shadow-sm")
        else:
            summary = html.P("无数据", className="text-muted")

    else:
        # 原有逻辑（单年单月）
        if month == 'all':
            # 如果不是均值模式但选了"全年"，提示错误
            summary = dbc.Alert(
                "⚠️ 当前模式不支持全年统计，请选择具体月份",
                color="warning"
            )
        else:
            current_df = df_with_national[
                (df_with_national['年'] == year) &
                (df_with_national['月'] == month) &
                (df_with_national['province'] != '全国平均')
                ]

            if len(current_df) > 0:
                values = current_df[variable]

                summary = dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("最大值", className="text-muted"),
                                html.H4(f"{values.max():.2f}", className="text-danger"),
                                html.P(f"{current_df.loc[values.idxmax(), 'province']}",
                                       className="text-muted small mb-0")
                            ], width=2),
                            dbc.Col([
                                html.H6("最小值", className="text-muted"),
                                html.H4(f"{values.min():.2f}", className="text-info"),
                                html.P(f"{current_df.loc[values.idxmin(), 'province']}",
                                       className="text-muted small mb-0")
                            ], width=2),
                            dbc.Col([
                                html.H6("平均值", className="text-muted"),
                                html.H4(f"{values.mean():.2f}", className="text-primary")
                            ], width=2),
                            dbc.Col([
                                html.H6("标准差", className="text-muted"),
                                html.H4(f"{values.std():.2f}", className="text-warning")
                            ], width=2),
                            dbc.Col([
                                html.H6("极差", className="text-muted"),
                                html.H4(f"{values.max() - values.min():.2f}", className="text-success")
                            ], width=2),
                            dbc.Col([
                                html.H6("变异系数", className="text-muted"),
                                html.H4(f"{(values.std() / values.mean() * 100):.1f}%",
                                        className="text-secondary")
                            ], width=2)
                        ])
                    ])
                ], className="shadow-sm")
            else:
                summary = html.P("无数据", className="text-muted")

    # ← 最后统一返回
    return fig, summary


@app.callback(
    [Output('ranking-plot', 'figure'),
     Output('ranking-summary', 'children')],
    [Input('ranking-year-start', 'value'),
     Input('ranking-year-end', 'value'),
     Input('ranking-months', 'value'),
     Input('ranking-variable', 'value'),
     Input('ranking-agg-method', 'value'),
     Input('ranking-sort-method', 'value')]
)
def update_ranking_plot(year_start, year_end, months, variable, agg_method, sort_method):
    """更新时间段排序图"""

    if not all([year_start, year_end, variable]):
        return go.Figure(), html.Div()

    # 数据筛选
    df_filtered = df_with_national[
        (df_with_national['年'] >= year_start) &
        (df_with_national['年'] <= year_end) &
        (df_with_national['province'] != '全国平均')  # ← 排除全国平均
        ].copy()

    if months != 'all':
        df_filtered = df_filtered[df_filtered['月'] == months]

    # ========== 删除省份标准化逻辑 ==========
    # ⚠️ 排序图不需要合并蒙东蒙西，保持原始数据

    # 聚合计算（直接按province分组，不做标准化）
    agg_df = df_filtered.groupby('province').agg({
        variable: agg_method
    }).reset_index()

    # 排序
    if sort_method == 'geo':
        from spatial_analysis import get_geo_sort_key
        agg_df['sort_key'] = agg_df['province'].apply(get_geo_sort_key)
        agg_df = agg_df.sort_values('sort_key')
    else:
        ascending = (sort_method == 'asc')
        agg_df = agg_df.sort_values(variable, ascending=ascending)

    # 绘图
    var_display = var_name_map.get(variable, variable)

    # 颜色映射（根据数值）
    values = agg_df[variable].values
    colors = ['#d62728' if v > values.mean() else '#1f77b4' for v in values]

    fig = go.Figure(go.Bar(
        x=agg_df['province'],
        y=agg_df[variable],
        marker_color=colors,
        text=agg_df[variable].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      f'{var_display}: %{{y:.2f}}<extra></extra>'
    ))

    # 均值线
    mean_val = values.mean()
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"均值: {mean_val:.2f}",
        annotation_position="top right"
    )

    # 标题
    month_str = f"{months}月" if months != 'all' else "全年"
    agg_str = {'mean': '均值', 'sum': '总和', 'max': '最大值', 'min': '最小值'}[agg_method]

    fig.update_layout(
        title=f'{year_start}-{year_end}年{month_str} {var_display} {agg_str}排序',
        xaxis_title='省份',
        yaxis_title=f'{var_display} ({agg_str})',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=800,
        showlegend=False
    )

    # 统计摘要
    summary = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("最大值", className="text-muted"),
                    html.H4(f"{values.max():.2f}", className="text-danger"),
                    html.P(f"{agg_df.loc[agg_df[variable].idxmax(), 'province']}",
                           className="text-muted small mb-0")
                ], width=3),
                dbc.Col([
                    html.H6("最小值", className="text-muted"),
                    html.H4(f"{values.min():.2f}", className="text-info"),
                    html.P(f"{agg_df.loc[agg_df[variable].idxmin(), 'province']}",
                           className="text-muted small mb-0")
                ], width=3),
                dbc.Col([
                    html.H6("平均值", className="text-muted"),
                    html.H4(f"{mean_val:.2f}", className="text-primary")
                ], width=2),
                dbc.Col([
                    html.H6("标准差", className="text-muted"),
                    html.H4(f"{values.std():.2f}", className="text-warning")
                ], width=2),
                dbc.Col([
                    html.H6("极差", className="text-muted"),
                    html.H4(f"{values.max() - values.min():.2f}", className="text-success")
                ], width=2)
            ])
        ])
    ], className="shadow-sm")

    return fig, summary


# ============================================================================
# 象限分析回调
# ============================================================================

@app.callback(
    [Output('quadrant-plot', 'figure'),
     Output('quadrant-table', 'figure')],
    [Input('quadrant-year-start', 'value'),
     Input('quadrant-year-end', 'value'),
     Input('quadrant-month', 'value'),
     Input('quadrant-variable', 'value'),
     Input('quadrant-highlight', 'value')]
)
def update_quadrant_analysis(year_start, year_end, month, variable, highlight_provinces):
    """更新象限分析"""

    if not all([year_start, year_end, variable]):
        return go.Figure(), go.Figure()

    # 计算统计量
    stats_df = calculate_resource_stability(
        df_with_national,
        variable,
        year_start,
        year_end,
        month,
        exclude_national=True,
        deseasonalize=True
    )

    var_display = var_name_map.get(variable, variable)

    # 生成象限图
    fig_quadrant = create_quadrant_plot(
        stats_df,
        variable,
        var_display,
        year_start,
        year_end,
        month,
        highlight_provinces
    )

    # 生成统计表
    fig_table = create_summary_table(stats_df, variable, var_display)

    return fig_quadrant, fig_table


@app.callback(
    Output('export-quadrant-status', 'children'),
    Input('btn-export-quadrant', 'n_clicks'),
    [State('quadrant-year-start', 'value'),
     State('quadrant-year-end', 'value'),
     State('quadrant-month', 'value'),
     State('quadrant-variable', 'value')],
    prevent_initial_call=True
)
def export_quadrant_data(n_clicks, year_start, year_end, month, variable):
    """导出象限分析数据"""

    if not all([year_start, year_end, variable]):
        return dbc.Alert("⚠️ 参数不完整", color="warning")

    try:
        # 计算统计量
        stats_df = calculate_resource_stability(
            df_with_national,
            variable,
            year_start,
            year_end,
            month,
            exclude_national=True,
            deseasonalize=True
        )

        # 生成文件名
        var_name = var_name_map.get(variable, variable).replace('/', '_')
        month_str = f"{month}月" if month != 'all' else "全年"
        filename = f"象限分析_{var_name}_{year_start}-{year_end}年{month_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = str(Path(__file__).parent / '导出数据' / filename)

        # 导出数据
        export_data_to_excel(
            stats_df,
            df_with_national,
            variable,
            year_start,
            year_end,
            month,
            filepath
        )

        return dbc.Alert(f"✓ 数据已导出: {filename}", color="success")

    except Exception as e:
        return dbc.Alert(f"⚠️ 导出失败: {str(e)}", color="danger")


# ============================================================================
# 极端事件分析回调
# ============================================================================

# ========== 控制面板显示/隐藏 ==========

@app.callback(
    [Output('percentile-controls', 'style'),
     Output('percentile-controls-lower', 'style'),
     Output('zscore-controls', 'style'),
     Output('cv-controls', 'style')],
    Input('extreme-method-dropdown', 'value')
)
def toggle_extreme_method_controls(method):
    """根据检测方法显示对应的参数控件"""
    show = {'display': 'block'}
    hide = {'display': 'none'}

    if method == 'percentile':
        return show, show, hide, hide
    elif method == 'zscore':
        return hide, hide, show, hide
    else:  # cv
        return hide, hide, hide, show


@app.callback(
    Output('extreme-province-selector', 'style'),
    Input('extreme-display-mode', 'value')
)
def toggle_province_selector(mode):
    """单省模式才显示省份选择器"""
    if mode == 'single':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('export-ranking-status', 'children'),
    Input('btn-export-ranking', 'n_clicks'),
    [State('ranking-year-start', 'value'),
     State('ranking-year-end', 'value'),
     State('ranking-months', 'value'),
     State('ranking-variable', 'value'),
     State('ranking-agg-method', 'value')],
    prevent_initial_call=True
)
def export_ranking_data(n_clicks, year_start, year_end, months, variable, agg_method):
    """导出时间段排序数据"""

    if not all([year_start, year_end, variable]):
        return dbc.Alert("⚠️ 参数不完整", color="warning")

    try:
        # 数据筛选
        df_filtered = df_with_national[
            (df_with_national['年'] >= year_start) &
            (df_with_national['年'] <= year_end) &
            (df_with_national['province'] != '全国平均')
            ].copy()

        if months != 'all':
            df_filtered = df_filtered[df_filtered['月'] == months]

        # 生成文件名
        var_name = var_name_map.get(variable, variable).replace('/', '_')
        month_str = f"{months}月" if months != 'all' else "全年"
        agg_str = {'mean': '均值', 'sum': '总和', 'max': '最大值', 'min': '最小值'}[agg_method]
        filename = f"时间段排序_{var_name}_{year_start}-{year_end}年{month_str}_{agg_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = str(Path(__file__).parent / '导出数据' / filename)

        # 导出数据（复用 export_data_to_excel 函数）
        # 先计算统计量
        stats_df = df_filtered.groupby('province')[variable].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        stats_df['cv'] = stats_df['std'] / stats_df['mean']

        export_data_to_excel(
            stats_df,
            df_with_national,
            variable,
            year_start,
            year_end,
            months,
            filepath
        )

        return dbc.Alert(f"✓ 数据已导出: {filename}", color="success")

    except Exception as e:
        return dbc.Alert(f"⚠️ 导出失败: {str(e)}", color="danger")


# ========== 极端事件检测与可视化 ==========

@app.callback(
    [Output('extreme-heatmap', 'figure'),
     Output('extreme-summary', 'children')],
    [Input('extreme-detect-button', 'n_clicks')],
    [State('extreme-variable-dropdown', 'value'),
     State('extreme-method-dropdown', 'value'),
     State('extreme-display-mode', 'value'),
     State('extreme-province-dropdown', 'value'),
     State('upper-percentile-slider', 'value'),
     State('lower-percentile-slider', 'value'),
     State('zscore-threshold-slider', 'value'),
     State('cv-threshold-slider', 'value'),
     State('baseline-start-dropdown', 'value'),
     State('baseline-end-dropdown', 'value')]
)
def update_extreme_analysis(n_clicks, variable, method, display_mode, province,
                            upper_pct, lower_pct, zscore_thresh, cv_thresh,
                            baseline_start, baseline_end):
    """更新极端事件分析"""

    # 创建配置
    config = ExtremeConfig(
        method=method,
        upper_percentile=upper_pct,
        lower_percentile=lower_pct,
        zscore_threshold=zscore_thresh,
        cv_threshold=cv_thresh,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )

    # 检测极端事件
    extreme_df = detect_extreme_events(df_with_national, variable, config)

    # 生成图表
    var_display = var_name_map.get(variable, variable)

    if display_mode == 'single':
        fig = create_extreme_heatmap_single(extreme_df, province, variable, var_display)
    elif display_mode == 'national':
        fig = create_extreme_heatmap_national(extreme_df, variable, var_display)
    else:  # provinces
        fig = create_extreme_heatmap_provinces(extreme_df, variable, var_display)

    # 生成统计摘要
    total_extreme = extreme_df['is_extreme'].sum()
    total_high = extreme_df['extreme_high'].sum()
    total_low = extreme_df['extreme_low'].sum()
    total_records = len(extreme_df)
    extreme_pct = (total_extreme / total_records * 100) if total_records > 0 else 0

    summary = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("总记录数", className="text-muted"),
                    html.H4(f"{total_records:,}", className="text-primary")
                ], width=3),
                dbc.Col([
                    html.H6("极端事件总数", className="text-muted"),
                    html.H4(f"{total_extreme:,}", className="text-danger")
                ], width=3),
                dbc.Col([
                    html.H6("极端高值", className="text-muted"),
                    html.H4(f"{total_high:,}", className="text-warning")
                ], width=2),
                dbc.Col([
                    html.H6("极端低值", className="text-muted"),
                    html.H4(f"{total_low:,}", className="text-info")
                ], width=2),
                dbc.Col([
                    html.H6("极端事件占比", className="text-muted"),
                    html.H4(f"{extreme_pct:.2f}%", className="text-success")
                ], width=2)
            ])
        ])
    ], className="shadow-sm")

    return fig, summary


# ============================================================================
# ENSO关联分析回调
# ============================================================================

# ========== 自动寻找最佳滞后期 ==========

@app.callback(
    [Output('enso-lag-slider', 'value'),
     Output('best-lag-display', 'children'),
     Output('best-lag-display', 'style')],
    [Input('auto-lag-button', 'n_clicks')],
    [State('enso-variable-dropdown', 'value'),
     State('enso-province-dropdown', 'value')]
)
def auto_find_best_lag(n_clicks, variables, province):
    """自动寻找最佳滞后期"""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    if not variables or len(variables) == 0:
        return 0, "⚠️ 请先选择变量", {'display': 'block'}

    # 只用第一个变量来寻找最佳滞后
    variable = variables[0] if isinstance(variables, list) else variables

    try:
        best_lag, best_corr = find_best_lag(
            df_oni, df_with_national, variable,
            province if province != '全部' else '全国平均',
            max_lag=12
        )

        message = f"✓ 最佳滞后期: {best_lag}个月 (相关系数: {best_corr:.3f})"

        return best_lag, message, {'display': 'block', 'backgroundColor': '#d1ecf1', 'color': '#0c5460'}

    except Exception as e:
        return 0, f"⚠️ 计算失败: {str(e)}", {'display': 'block', 'backgroundColor': '#f8d7da', 'color': '#721c24'}


# ========== ENSO图表更新 ==========

@app.callback(
    Output('enso-plot', 'figure'),
    [Input('enso-analysis-type', 'value'),
     Input('enso-variable-dropdown', 'value'),
     Input('enso-province-dropdown', 'value'),
     Input('el-nino-threshold', 'value'),
     Input('la-nina-threshold', 'value'),
     Input('enso-lag-slider', 'value'),
     Input('anomaly-type-dropdown', 'value')]
)
def update_enso_plot(analysis_type, variables, province, el_nino_thresh,
                     la_nina_thresh, lag, anomaly_type):
    """更新ENSO分析图表"""

    if not variables or len(variables) == 0:
        # 返回空图
        fig = go.Figure()
        fig.add_annotation(
            text="请选择至少一个变量",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # 创建ENSO配置
    enso_config = ENSOConfig(
        el_nino_threshold=el_nino_thresh,
        la_nina_threshold=la_nina_thresh,
        lag_months=lag,
        anomaly_type=anomaly_type
    )

    # 处理省份选择
    if province == '全部':
        provinces = ['全部']
    else:
        provinces = [province]

    # 生成图表
    if analysis_type == 'scatter':
        # 散点图只用第一个变量
        variable = variables[0] if isinstance(variables, list) else variables
        var_display = var_name_map.get(variable, variable)

        fig = create_enso_scatter(
            df_oni, df_with_national, variable, provinces,
            enso_config, var_display
        )
    else:  # timeseries
        fig = create_enso_timeseries(
            df_oni, df_with_national, variables, province,
            enso_config, var_name_map
        )

    return fig


# ========== ENSO频次统计表 ==========

@app.callback(
    Output('enso-frequency-table', 'figure'),
    [Input('enso-variable-dropdown', 'value'),
     Input('enso-province-dropdown', 'value'),
     Input('el-nino-threshold', 'value'),
     Input('la-nina-threshold', 'value'),
     Input('extreme-method-dropdown', 'value'),
     Input('upper-percentile-slider', 'value'),
     Input('lower-percentile-slider', 'value'),
     Input('zscore-threshold-slider', 'value'),
     Input('cv-threshold-slider', 'value'),
     Input('baseline-start-dropdown', 'value'),
     Input('baseline-end-dropdown', 'value')]
)
def update_enso_frequency_table(variables, province, el_nino_thresh, la_nina_thresh,
                                method, upper_pct, lower_pct, zscore_thresh, cv_thresh,
                                baseline_start, baseline_end):
    """更新ENSO阶段极端事件频次统计"""

    if not variables or len(variables) == 0:
        return go.Figure()

    # 使用第一个变量
    variable = variables[0] if isinstance(variables, list) else variables

    # 极端事件配置
    extreme_config = ExtremeConfig(
        method=method,
        upper_percentile=upper_pct,
        lower_percentile=lower_pct,
        zscore_threshold=zscore_thresh,
        cv_threshold=cv_thresh,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )

    # ENSO配置
    enso_config = ENSOConfig(
        el_nino_threshold=el_nino_thresh,
        la_nina_threshold=la_nina_thresh
    )

    # 检测极端事件
    extreme_df = detect_extreme_events(df_with_national, variable, extreme_config)

    # 筛选省份
    if province != '全部':
        extreme_df = extreme_df[extreme_df['province'] == province]

    # 计算频次
    freq_df = calculate_extreme_frequency_by_enso(extreme_df, df_oni, enso_config)

    # 生成表格
    fig = create_extreme_frequency_table(freq_df)

    return fig


# ========================================
# 回调函数：多变量联动分析
# ========================================

def merge_enso_data(df_main, df_oni):
    """
    按需合并ENSO数据到主DataFrame

    Parameters:
    -----------
    df_main : DataFrame
        主数据（df_with_national）
    df_oni : DataFrame
        ONI数据

    Returns:
    --------
    df_merged : DataFrame
        合并后的数据，包含ENSO_state列
    """
    df_result = df_main.copy()

    if df_oni is not None and len(df_oni) > 0:
        df_oni_temp = df_oni.copy()

        # ENSO分类（简化版）
        enso_config = ENSOConfig()

        def classify_oni(oni_value):
            """简单的ENSO分类"""
            if pd.isna(oni_value):
                return 'Neutral'
            elif oni_value >= enso_config.el_nino_threshold:
                return 'El Niño'
            elif oni_value <= enso_config.la_nina_threshold:
                return 'La Niña'
            else:
                return 'Neutral'

        df_oni_temp['ENSO_state'] = df_oni_temp['ONI'].apply(classify_oni)

        # 合并
        df_result = pd.merge(
            df_result,
            df_oni_temp[['年', '月', 'ONI', 'ENSO_state']],
            on=['年', '月'],
            how='left'
        )
        df_result['ENSO_state'].fillna('Neutral', inplace=True)
    else:
        df_result['ENSO_state'] = 'Neutral'

    return df_result


# ========== 回调 1: 相关性矩阵 ==========

@app.callback(
    Output('corr-matrix-plot', 'figure'),
    [
        Input('corr-year-start', 'value'),
        Input('corr-year-end', 'value'),
        Input('corr-variables', 'value'),
        Input('corr-enso-filter', 'value'),
        Input('corr-method', 'value')
    ]
)
def update_correlation_matrix(year_start, year_end, variables, enso_filter, method):
    """更新相关性矩阵"""

    # 参数验证
    if not variables or len(variables) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 请至少选择2个变量",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig

    if year_start > year_end:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 起始年份不能大于结束年份",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig

        # ========== 新增：按需合并ENSO数据 ==========
    df_for_analysis = df_with_national.copy()

    # ========== 在这里添加（第2498行左右） ==========
    # 按需合并ENSO数据
    df_for_analysis = merge_enso_data(df_with_national, df_oni)
    # ========== 添加结束 ==========

    # 调用函数生成图表
    fig = create_correlation_matrix(
        df=df_for_analysis,  # ← 改这里（原来是 df_merged）
        variables=variables,
        time_range=(year_start, year_end),
        enso_filter=enso_filter if enso_filter != '全部' else None,
        method=method
    )


    return fig


# ========== 回调 2: 双变量散点图 ==========

@app.callback(
    [
        Output('scatter-plot', 'figure'),
        Output('scatter-stats', 'children')
    ],
    [
        Input('scatter-x-var', 'value'),
        Input('scatter-y-var', 'value'),
        Input('scatter-year-start', 'value'),
        Input('scatter-year-end', 'value'),
        Input('scatter-enso-filter', 'value'),
        Input('scatter-color-by', 'value'),
        Input('scatter-size-var', 'value'),
        Input('scatter-options', 'value')
    ]
)
def update_scatter_plot(x_var, y_var, year_start, year_end, enso_filter,
                        color_by, size_var, options):
    """更新散点图"""

    # 参数验证
    if x_var == y_var:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ X轴和Y轴不能是同一个变量",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        stats_text = "请选择不同的变量"
        return fig, stats_text

    if year_start > year_end:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 起始年份不能大于结束年份",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        stats_text = "时间范围错误"
        return fig, stats_text

        # 按需合并ENSO数据
    df_for_analysis = merge_enso_data(df_with_national, df_oni)

    # 解析显示选项
    show_regression = 'regression' in options if options else False
    show_confidence = 'confidence' in options if options else False
    show_density = 'density' in options if options else False

    # 处理气泡大小
    size_var_actual = None if size_var == 'none' else size_var

    # 调用函数生成图表
    fig, stats_text = create_scatter_with_regression(
        df=df_for_analysis,  # ← 使用临时合并的数据
        x_var=x_var,
        y_var=y_var,
        color_by=color_by,
        size_var=size_var_actual,
        time_range=(year_start, year_end),
        enso_filter=enso_filter if enso_filter != '全部' else None,
        show_regression=show_regression,
        show_confidence=show_confidence,
        show_density=show_density
    )

    # 格式化统计文本（转换为HTML）
    stats_html = dcc.Markdown(stats_text)

    return fig, stats_html


# ========== 回调 3: 双地图对比 ==========

# ========== 回调 3: 双地图对比 ==========
@app.callback(
    Output('dual-map-plot', 'figure'),
    [
        Input('dual-map-year', 'value'),
        Input('dual-map-month', 'value'),
        Input('dual-map-var1', 'value'),
        Input('dual-map-var2', 'value'),
        Input('dual-map-display-type1', 'value'),
        Input('dual-map-display-type2', 'value'),
        Input('dual-map-colorscale1', 'value'),
        Input('dual-map-colorscale2', 'value'),
        Input('dual-map-style', 'value'),
        Input('dual-map-baseline-start', 'value'),
        Input('dual-map-baseline-end', 'value')
    ]
)
def update_dual_map(year, month, var1, var2,
                   display_type1, display_type2,
                   colorscale1, colorscale2,
                   mapbox_style,
                   baseline_start, baseline_end):
    """更新双地图对比"""

    # 参数验证
    if var1 == var2:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 两个地图不能显示同一个变量<br>请选择不同的变量",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig

    # 加载GeoJSON
    from spatial_map import load_china_geojson
    geojson = load_china_geojson()

    # 基准期
    baseline_years = None
    if 'anomaly' in [display_type1, display_type2] and baseline_start and baseline_end:
        baseline_years = (baseline_start, baseline_end)

    # 调用函数生成图表
    fig = create_dual_map_comparison(
        df=df_with_national,
        year=year,
        month=month,
        var1=var1,
        var2=var2,
        geojson=geojson,
        display_type1=display_type1,
        display_type2=display_type2,
        baseline_years=baseline_years,
        colorscale1=colorscale1,
        colorscale2=colorscale2,
        mapbox_style=mapbox_style
    )

    return fig


# ========== 控制基准期显示/隐藏 ==========
@app.callback(
    Output('dual-map-baseline-container', 'style'),
    [Input('dual-map-display-type1', 'value'),
     Input('dual-map-display-type2', 'value')]
)
def toggle_dual_map_baseline(type1, type2):
    """任一地图选择距平模式时显示基准期"""
    if 'anomaly' in [type1, type2]:
        return {'display': 'block'}
    return {'display': 'none'}



# ============================================================================
# 新增回调：ENSO当前状态卡 + 均值偏移热力图
# ============================================================================

@app.callback(
    Output('enso-status-card', 'children'),
    Input('enso-analysis-type', 'value')  # 任意触发，页面加载时执行
)
def update_enso_status_card(_):
    if df_oni.empty:
        return dbc.CardBody("ONI数据未加载")
    status = get_current_enso_status(df_oni)
    if not status:
        return dbc.CardBody("无数据")
    color = status['color']
    phase_emoji = '🔴' if status['phase'] == '厄尔尼诺' else ('🔵' if status['phase'] == '拉尼娜' else '⚪')
    return dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5("🌊 当前ENSO状态", className="card-title mb-1"),
                html.H3(
                    f"{phase_emoji} {status['phase']}",
                    style={'color': color, 'fontWeight': 'bold'}
                ),
            ], width=4),
            dbc.Col([
                html.P(f"最新数据：{status['year']}年{status['month']}月", className="mb-1"),
                html.P(f"ONI指数：{status['oni']:+.2f}℃", className="mb-1"),
                html.P(f"持续时长：{status['duration']} 个月", className="mb-0"),
            ], width=4),
            dbc.Col([
                html.Small(
                    "厄尔尼诺(ONI≥+0.5) / 拉尼娜(ONI≤-0.5) / 中性(-0.5~+0.5)",
                    className="text-muted"
                ),
            ], width=4, className="d-flex align-items-center"),
        ])
    ], style={'borderLeft': f'4px solid {color}'})


@app.callback(
    Output('enso-zmean-heatmap', 'figure'),
    Input('enso-analysis-type', 'value')  # 任意触发，页面加载时执行
)
def update_enso_zmean_heatmap(_):
    if df_oni.empty:
        return go.Figure()
    return create_enso_zmean_heatmap(df, df_oni)


# ============================================================================
# 运行应用
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8031, use_reloader=False)
