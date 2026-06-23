"""
资源禀赋与稳定性分析模块
用于分析各省份的资源水平（均值）和稳定性（变异系数）
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_resource_stability(df, variable, year_start, year_end,
                                 month_filter='all', exclude_national=True, deseasonalize=True):
    """
    计算各省份的资源禀赋（均值）和稳定性（变异系数）

    Parameters:
    -----------
    df : DataFrame
        原始数据
    variable : str
        分析变量
    year_start : int
        起始年份
    year_end : int
        结束年份
    month_filter : int or 'all'
        月份筛选（1-12 或 'all'）
    exclude_national : bool
        是否排除全国平均

    Returns:
    --------
    DataFrame
        包含 province, mean, std, cv 列
    """

    # 数据筛选
    df_filtered = df[
        (df['年'] >= year_start) &
        (df['年'] <= year_end)
        ].copy()

    if exclude_national:
        df_filtered = df_filtered[df_filtered['province'] != '全国平均']

    if month_filter != 'all':
        df_filtered = df_filtered[df_filtered['月'] == month_filter]

    # 🔧 按省份计算统计量
    results = []

    for province in df_filtered['province'].unique():
        province_data = df_filtered[df_filtered['province'] == province]

        # 计算均值
        overall_mean = province_data[variable].mean()

        # 🔧 去季节性处理
        if deseasonalize and month_filter == 'all':
            # 计算每个月的平均值
            monthly_means = province_data.groupby('月')[variable].transform('mean')
            # 计算偏差
            anomalies = province_data[variable] - monthly_means
            # 偏差的标准差
            std = anomalies.std()
        else:
            # 直接计算标准差
            std = province_data[variable].std()

        # 🔧 温度转换为开尔文
        if variable in ['温度', '温度_平均_2米_省内', '2m温度']:  # 根据实际变量名调整
            mean_for_cv = overall_mean + 273.15
        else:
            mean_for_cv = overall_mean

        # 计算CV
        cv = std / mean_for_cv if mean_for_cv != 0 else np.nan

        results.append({
            'province': province,
            'mean': overall_mean,
            'std': std,
            'cv': cv,
            'count': len(province_data)
        })

    stats_df = pd.DataFrame(results)

    # 处理异常值
    stats_df['cv'] = stats_df['cv'].replace([np.inf, -np.inf], np.nan)

    return stats_df


def create_quadrant_plot(stats_df, variable, var_display_name,
                         year_start, year_end, month_filter='all',
                         highlight_provinces=None):
    """
    创建资源禀赋 vs 稳定性象限图

    Parameters:
    -----------
    stats_df : DataFrame
        统计数据（来自 calculate_resource_stability）
    variable : str
        变量名
    var_display_name : str
        变量显示名称
    year_start, year_end : int
        时间范围
    month_filter : int or 'all'
        月份
    highlight_provinces : list
        需要高亮的省份

    Returns:
    --------
    plotly.graph_objects.Figure
    """

    # 去除缺失值
    plot_df = stats_df.dropna(subset=['mean', 'cv']).copy()

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 无有效数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig

    # 计算中位数（用于划分象限）
    mean_median = plot_df['mean'].median()
    cv_median = plot_df['cv'].median()

    # 象限分类
    def classify_quadrant(row):
        if row['mean'] >= mean_median and row['cv'] >= cv_median:
            return '第一象限<br>(高资源+高波动)'
        elif row['mean'] >= mean_median and row['cv'] < cv_median:
            return '第四象限<br>(高资源+低波动)'
        elif row['mean'] < mean_median and row['cv'] >= cv_median:
            return '第二象限<br>(低资源+高波动)'
        else:
            return '第三象限<br>(低资源+低波动)'

    plot_df['quadrant'] = plot_df.apply(classify_quadrant, axis=1)

    # 颜色映射
    color_map = {
        '第一象限<br>(高资源+高波动)': '#FF6B6B',  # 红色：带刺的玫瑰
        '第四象限<br>(高资源+低波动)': '#51CF66',  # 绿色：优质资产
        '第二象限<br>(低资源+高波动)': '#868E96',  # 灰色：垃圾股
        '第三象限<br>(低资源+低波动)': '#4DABF7'  # 蓝色：稳定但资源少
    }

    # 创建图表
    fig = go.Figure()

    # 按象限分组绘制
    for quadrant in plot_df['quadrant'].unique():
        df_quad = plot_df[plot_df['quadrant'] == quadrant]

        # 判断是否高亮
        if highlight_provinces:
            df_quad['is_highlight'] = df_quad['province'].isin(highlight_provinces)
        else:
            df_quad['is_highlight'] = False

        # 普通点
        df_normal = df_quad[~df_quad['is_highlight']]
        if len(df_normal) > 0:
            fig.add_trace(go.Scatter(
                x=df_normal['mean'],
                y=df_normal['cv'],
                mode='markers+text',
                name=quadrant,
                marker=dict(
                    size=12,
                    color=color_map.get(quadrant, '#868E96'),
                    line=dict(width=1, color='white')
                ),
                text=df_normal['province'],
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate=(
                        '<b>%{text}</b><br>' +
                        f'{var_display_name}均值: %{{x:.2f}}<br>' +
                        '变异系数: %{y:.3f}<br>' +
                        f'标准差: {df_normal["std"].iloc[0]:.2f}<br>' +
                        '<extra></extra>'
                )
            ))

        # 高亮点
        df_highlight = df_quad[df_quad['is_highlight']]
        if len(df_highlight) > 0:
            fig.add_trace(go.Scatter(
                x=df_highlight['mean'],
                y=df_highlight['cv'],
                mode='markers+text',
                name=f'{quadrant} (高亮)',
                marker=dict(
                    size=18,
                    color=color_map.get(quadrant, '#868E96'),
                    line=dict(width=3, color='black'),
                    symbol='star'
                ),
                text=df_highlight['province'],
                textposition='top center',
                textfont=dict(size=11, color='black'),
                hovertemplate=(
                        '<b>%{text}</b><br>' +
                        f'{var_display_name}均值: %{{x:.2f}}<br>' +
                        '变异系数: %{y:.3f}<br>' +
                        '<extra></extra>'
                ),
                showlegend=False
            ))

    # 添加象限分割线
    fig.add_hline(
        y=cv_median,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"CV中位数: {cv_median:.3f}",
        annotation_position="right"
    )

    fig.add_vline(
        x=mean_median,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"均值中位数: {mean_median:.2f}",
        annotation_position="top"
    )

    # 添加象限标注
    annotations = [
        dict(x=0.95, y=0.95, xref='paper', yref='paper',
             text='<b>第一象限</b><br>带刺的玫瑰<br>(高资源+高波动)',
             showarrow=False, font=dict(size=10, color='#FF6B6B'),
             bgcolor='rgba(255,107,107,0.1)', borderpad=4),

        dict(x=0.95, y=0.05, xref='paper', yref='paper',
             text='<b>第四象限</b><br>优质资产<br>(高资源+低波动)',
             showarrow=False, font=dict(size=10, color='#51CF66'),
             bgcolor='rgba(81,207,102,0.1)', borderpad=4),

        dict(x=0.05, y=0.95, xref='paper', yref='paper',
             text='<b>第二象限</b><br>垃圾股<br>(低资源+高波动)',
             showarrow=False, font=dict(size=10, color='#868E96'),
             bgcolor='rgba(134,142,150,0.1)', borderpad=4),

        dict(x=0.05, y=0.05, xref='paper', yref='paper',
             text='<b>第三象限</b><br>稳定但资源少<br>(低资源+低波动)',
             showarrow=False, font=dict(size=10, color='#4DABF7'),
             bgcolor='rgba(77,171,247,0.1)', borderpad=4),
    ]

    # 标题
    month_str = f"{month_filter}月" if month_filter != 'all' else "全年"
    title_text = f'{year_start}-{year_end}年{month_str} {var_display_name} 资源禀赋 vs 稳定性分析'

    fig.update_layout(
        title=title_text,
        xaxis_title=f'{var_display_name} 均值（资源禀赋）',
        yaxis_title='变异系数 CV（波动性）',
        template='plotly_white',
        height=700,
        hovermode='closest',
        annotations=annotations,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def create_summary_table(stats_df, variable, var_display_name):
    """
    创建统计摘要表格

    Returns:
    --------
    plotly.graph_objects.Figure (Table)
    """

    # 排序：按均值降序
    stats_df = stats_df.sort_values('mean', ascending=False).reset_index(drop=True)

    # 添加排名
    stats_df['rank'] = range(1, len(stats_df) + 1)

    # 象限分类
    mean_median = stats_df['mean'].median()
    cv_median = stats_df['cv'].median()

    def classify_quadrant(row):
        if row['mean'] >= mean_median and row['cv'] >= cv_median:
            return '高资源+高波动'
        elif row['mean'] >= mean_median and row['cv'] < cv_median:
            return '高资源+低波动 ⭐'
        elif row['mean'] < mean_median and row['cv'] >= cv_median:
            return '低资源+高波动 ⚠️'
        else:
            return '低资源+低波动'

    stats_df['category'] = stats_df.apply(classify_quadrant, axis=1)

    # 创建表格
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>排名</b>', '<b>省份</b>', '<b>均值</b>', '<b>标准差</b>',
                    '<b>变异系数</b>', '<b>样本数</b>', '<b>象限分类</b>'],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12)
        ),
        cells=dict(
            values=[
                stats_df['rank'],
                stats_df['province'],
                stats_df['mean'].round(2),
                stats_df['std'].round(2),
                stats_df['cv'].round(3),
                stats_df['count'],
                stats_df['category']
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title=f'{var_display_name} 统计摘要表',
        height=600
    )

    return fig


def export_data_to_excel(stats_df, df_raw, variable, year_start, year_end,
                         month_filter, filename):
    """
    导出数据到Excel（多个sheet）

    Parameters:
    -----------
    stats_df : DataFrame
        统计数据
    df_raw : DataFrame
        原始数据
    variable : str
        变量名
    year_start, year_end : int
        时间范围
    month_filter : int or 'all'
        月份
    filename : str
        文件名
    """

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: 统计摘要
        stats_export = stats_df.copy()
        stats_export = stats_export.sort_values('mean', ascending=False)
        stats_export.to_excel(writer, sheet_name='统计摘要', index=False)

        # Sheet 2: 总数据（筛选后的原始数据）
        df_filtered = df_raw[
            (df_raw['年'] >= year_start) &
            (df_raw['年'] <= year_end) &
            (df_raw['province'] != '全国平均')
            ].copy()

        if month_filter != 'all':
            df_filtered = df_filtered[df_filtered['月'] == month_filter]

        df_filtered.to_excel(writer, sheet_name='总数据', index=False)

        # Sheet 3: 分年数据
        df_by_year = df_filtered.groupby(['province', '年'])[variable].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        df_by_year['cv'] = df_by_year['std'] / df_by_year['mean']
        df_by_year.to_excel(writer, sheet_name='分年数据', index=False)

        # Sheet 4: 分月数据
        df_by_month = df_filtered.groupby(['province', '月'])[variable].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        df_by_month['cv'] = df_by_month['std'] / df_by_month['mean']
        df_by_month.to_excel(writer, sheet_name='分月数据', index=False)

        # Sheet 5: 分年分月数据
        df_by_year_month = df_filtered.groupby(['province', '年', '月'])[variable].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        df_by_year_month['cv'] = df_by_year_month['std'] / df_by_year_month['mean']
        df_by_year_month.to_excel(writer, sheet_name='分年分月数据', index=False)

    print(f"✓ 数据已导出到: {filename}")
