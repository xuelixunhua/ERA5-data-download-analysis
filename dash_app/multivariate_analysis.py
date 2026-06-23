"""
多变量联动分析模块
Multi-variable Interactive Analysis Module

功能：
1. 相关性矩阵热力图
2. 双变量散点图 + 回归分析
3. 双地图空间对比
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr, spearmanr


# ========== 变量名称映射 ==========
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


# ========== 1. 相关性矩阵 ==========

def create_correlation_matrix(df, variables,
                              time_range=None,
                              province_filter=None,
                              enso_filter=None,
                              method='pearson'):
    """
    创建相关性矩阵热力图

    Parameters:
    -----------
    df : DataFrame
        数据
    variables : list
        要分析的变量列表
    time_range : tuple
        (start_year, end_year)，时间范围筛选
    province_filter : list
        省份列表，None表示全部
    enso_filter : str
        'El Niño', 'La Niña', 'Neutral', None表示全部
    method : str
        'pearson' 或 'spearman'

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """

    # ========== 数据筛选 ==========
    filtered_df = df.copy()

    # 时间筛选
    if time_range:
        filtered_df = filtered_df[
            (filtered_df['年'] >= time_range[0]) &
            (filtered_df['年'] <= time_range[1])
        ]

    # 省份筛选
    if province_filter and len(province_filter) > 0:
        filtered_df = filtered_df[filtered_df['province'].isin(province_filter)]

    # ENSO筛选
    if enso_filter and enso_filter != '全部':
        filtered_df = filtered_df[filtered_df['ENSO_state'] == enso_filter]

    # 去除缺失值
    filtered_df = filtered_df[variables].dropna()

    if len(filtered_df) < 10:
        # 数据太少，返回提示
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 数据量不足<br>请调整筛选条件",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig

    # ========== 计算相关系数矩阵 ==========
    n_vars = len(variables)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_values = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                x = filtered_df[var1].values
                y = filtered_df[var2].values

                if method == 'pearson':
                    corr, p_val = pearsonr(x, y)
                else:  # spearman
                    corr, p_val = spearmanr(x, y)

                corr_matrix[i, j] = corr
                p_values[i, j] = p_val

    # ========== 绘制热力图 ==========

    # 变量显示名称
    var_labels = [var_name_map.get(v, v) for v in variables]

    # 创建文本标注（相关系数 + 显著性标记）
    text_matrix = []
    for i in range(n_vars):
        row = []
        for j in range(n_vars):
            corr_val = corr_matrix[i, j]
            p_val = p_values[i, j]

            # 显著性标记
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = ''

            text = f'{corr_val:.2f}{sig}'
            row.append(text)
        text_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=var_labels,
        y=var_labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 11, "color": "black"},
        colorbar=dict(
            title="相关系数",
            thickness=15,
            len=0.7
        ),
        hovertemplate='<b>%{y} vs %{x}</b><br>相关系数: %{z:.3f}<extra></extra>'
    ))

    # 标题信息
    title_parts = [f'变量相关性矩阵 ({method.capitalize()})']
    if time_range:
        title_parts.append(f'{time_range[0]}-{time_range[1]}年')
    if enso_filter and enso_filter != '全部':
        title_parts.append(f'{enso_filter}时期')
    if province_filter and len(province_filter) > 0:
        if len(province_filter) <= 3:
            title_parts.append(f'{", ".join(province_filter)}')
        else:
            title_parts.append(f'{len(province_filter)}个省份')

    title_text = ' | '.join(title_parts)
    title_text += '<br><sub>*p<0.05, **p<0.01, ***p<0.001</sub>'

    fig.update_layout(
        title=title_text,
        xaxis_title="",
        yaxis_title="",
        height=600,
        width=1400,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


# ========== 2. 双变量散点图 + 回归 ==========

def create_scatter_with_regression(df, x_var, y_var,
                                   color_by='province',
                                   size_var=None,
                                   time_range=None,
                                   province_filter=None,
                                   enso_filter=None,
                                   show_regression=True,
                                   show_confidence=True,
                                   show_density=False):
    """
    创建散点图 + 回归线

    Parameters:
    -----------
    df : DataFrame
        数据
    x_var, y_var : str
        X轴和Y轴变量
    color_by : str
        分组依据：'province', 'ENSO_state', 'region', 'season', 'year'
    size_var : str
        用于气泡大小的变量（可选）
    time_range : tuple
        (start_year, end_year)
    province_filter : list
        省份筛选
    enso_filter : str
        ENSO状态筛选
    show_regression : bool
        是否显示回归线
    show_confidence : bool
        是否显示置信区间
    show_density : bool
        是否显示密度等高线

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    stats_text : str (统计信息文本)
    """

    # ========== 数据筛选 ==========
    plot_df = df.copy()

    # 时间筛选
    if time_range:
        plot_df = plot_df[
            (plot_df['年'] >= time_range[0]) &
            (plot_df['年'] <= time_range[1])
        ]

    # 省份筛选
    if province_filter and len(province_filter) > 0:
        plot_df = plot_df[plot_df['province'].isin(province_filter)]

    # ENSO筛选
    if enso_filter and enso_filter != '全部':
        plot_df = plot_df[plot_df['ENSO_state'] == enso_filter]

    # 去除缺失值
    plot_df = plot_df.dropna(subset=[x_var, y_var])

    if len(plot_df) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 数据量不足<br>请调整筛选条件",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="orange")
        )
        return fig, "数据不足"

    # ========== 添加季节信息 ==========
    if color_by == 'season':
        def get_season(month):
            if month in [12, 1, 2]:
                return '冬季'
            elif month in [3, 4, 5]:
                return '春季'
            elif month in [6, 7, 8]:
                return '夏季'
            else:
                return '秋季'

        plot_df['season'] = plot_df['月'].apply(get_season)

    # ========== 创建散点图 ==========

    x_label = var_name_map.get(x_var, x_var)
    y_label = var_name_map.get(y_var, y_var)

    # 根据分组方式设置颜色
    color_discrete_map = None
    if color_by == 'ENSO_state':
        color_discrete_map = {
            'El Niño': '#d62728',      # 红色
            'La Niña': '#1f77b4',      # 蓝色
            'Neutral': '#7f7f7f'       # 灰色
        }
    elif color_by == 'season':
        color_discrete_map = {
            '春季': '#2ca02c',  # 绿色
            '夏季': '#d62728',  # 红色
            '秋季': '#ff7f0e',  # 橙色
            '冬季': '#1f77b4'   # 蓝色
        }

    # 创建散点图
    if size_var:
        fig = px.scatter(
            plot_df,
            x=x_var,
            y=y_var,
            color=color_by,
            size=size_var,
            hover_data=['province', '年', '月', 'ENSO_state'],
            labels={
                x_var: x_label,
                y_var: y_label,
                color_by: color_by,
                size_var: var_name_map.get(size_var, size_var)
            },
            color_discrete_map=color_discrete_map,
            opacity=0.6
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_var,
            y=y_var,
            color=color_by,
            hover_data=['province', '年', '月', 'ENSO_state'],
            labels={
                x_var: x_label,
                y_var: y_label,
                color_by: color_by
            },
            color_discrete_map=color_discrete_map,
            opacity=0.6
        )

    # ========== 回归分析 ==========

    x = plot_df[x_var].values
    y = plot_df[y_var].values

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # 统计信息
    stats_text = f"""
    **回归统计：**
    - 样本量：{len(x)}
    - 相关系数 (R)：{r_value:.4f}
    - 决定系数 (R²)：{r_value**2:.4f}
    - 回归方程：y = {slope:.4f}x + {intercept:.4f}
    - p值：{p_value:.4e}
    - 标准误差：{std_err:.4f}
    """

    # 显著性判断
    if p_value < 0.001:
        stats_text += "\n- **显著性：*** (p < 0.001)"
    elif p_value < 0.01:
        stats_text += "\n- **显著性：** (p < 0.01)"
    elif p_value < 0.05:
        stats_text += "\n- **显著性：* (p < 0.05)"
    else:
        stats_text += "\n- **显著性：不显著 (p ≥ 0.05)**"

    # ========== 添加回归线 ==========

    if show_regression:
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'回归线 (R²={r_value**2:.3f})',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=True
        ))

        # ========== 置信区间 ==========

        if show_confidence:
            # 计算预测标准误差
            y_pred = slope * x + intercept
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(x) - 2)

            # 95%置信区间
            x_mean = x.mean()
            sxx = np.sum((x - x_mean)**2)

            confidence_y = []
            for xi in x_line:
                se = np.sqrt(mse * (1/len(x) + (xi - x_mean)**2 / sxx))
                confidence_y.append(1.96 * se)  # 95% CI

            confidence_y = np.array(confidence_y)

            fig.add_trace(go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([y_line + confidence_y, (y_line - confidence_y)[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.15)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95%置信区间',
                showlegend=True,
                hoverinfo='skip'
            ))

    # ========== 密度等高线（可选） ==========

    if show_density:
        from scipy.stats import gaussian_kde

        # 计算密度
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # 添加等高线
        fig.add_trace(go.Contour(
            x=x,
            y=y,
            z=z,
            colorscale='Greys',
            showscale=False,
            opacity=0.3,
            contours=dict(
                showlabels=False,
                labelfont=dict(size=8, color='white')
            ),
            name='密度分布',
            hoverinfo='skip'
        ))

    # ========== 布局 ==========

    title_parts = [f'{y_label} vs {x_label}']
    if time_range:
        title_parts.append(f'{time_range[0]}-{time_range[1]}年')
    if enso_filter and enso_filter != '全部':
        title_parts.append(f'{enso_filter}时期')

    title_text = ' | '.join(title_parts)

    fig.update_layout(
        title=title_text,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=700,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig, stats_text


# ========== 3. 双地图空间对比 ==========

def create_dual_map_comparison(df, year, month, var1, var2, geojson,
                               display_type1='absolute', display_type2='absolute',
                               baseline_years=None,
                               colorscale1='RdBu_r', colorscale2='Blues',
                               mapbox_style='carto-positron'):
    """
    创建双地图对比（左右并排）
    参考 spatial_map.py 的实现逻辑

    Parameters:
    -----------
    df : DataFrame
        数据
    year, month : int
        时间点
    var1, var2 : str
        两个要对比的变量
    geojson : dict
        中国省份边界GeoJSON
    display_type1, display_type2 : str
        显示类型：'absolute', 'anomaly', 'yoy'
    baseline_years : tuple
        基准期 (start_year, end_year)，用于距平计算
    colorscale1, colorscale2 : str
        配色方案
    mapbox_style : str
        底图样式

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """

    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 缺少GeoJSON文件<br>请检查 geo_plotly.json",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig

    # ========== 数据准备 ==========

    current_df = df[
        (df['年'] == year) &
        (df['月'] == month)
        ].copy()

    if len(current_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ 无数据: {year}年{month}月",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # ========== 检查变量是否存在 ==========
    for var in [var1, var2]:
        if var not in current_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"⚠️ 变量不存在: {var}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="red")
            )
            return fig

    # ========== 省份名称标准化（参考 spatial_map.py） ==========
    province_mapping = {
        '蒙东': '内蒙古自治区',
        '蒙西': '内蒙古自治区',
    }

    current_df['province_std'] = current_df['province'].replace(province_mapping)

    # 聚合数据（合并蒙东蒙西）
    current_df = current_df.groupby('province_std').agg({
        var1: 'mean',
        var2: 'mean'
    }).reset_index()
    current_df.rename(columns={'province_std': 'province'}, inplace=True)

    # ========== 计算显示值的辅助函数 ==========

    def calculate_display_value(df_current, variable, display_type, df_all, target_year, target_month, baseline_years):
        """
        计算显示值（参考 spatial_map.py 的逻辑）

        Returns:
        --------
        df_result : DataFrame (包含 display_value 列)
        value_label : str (显示标签)
        """
        df_result = df_current.copy()

        if display_type == 'absolute':
            # 绝对值
            df_result['display_value'] = df_result[variable]
            value_label = "绝对值"

        elif display_type == 'anomaly':
            # 距平值（相对气候均值）
            if baseline_years:
                baseline_df = df_all[
                    (df_all['年'] >= baseline_years[0]) &
                    (df_all['年'] <= baseline_years[1]) &
                    (df_all['月'] == target_month)
                    ]
            else:
                baseline_df = df_all[df_all['月'] == target_month]

            # 标准化省份名称
            baseline_df = baseline_df.copy()
            baseline_df['province_std'] = baseline_df['province'].replace(province_mapping)

            # 聚合
            climate_mean = baseline_df.groupby('province_std')[variable].mean()

            # 计算距平
            df_result['display_value'] = df_result.apply(
                lambda row: row[variable] - climate_mean.get(row['province'], np.nan),
                axis=1
            )
            value_label = "距平"

        elif display_type == 'yoy':
            # 同比变化（相对去年同月）
            prev_year_df = df_all[
                (df_all['年'] == target_year - 1) &
                (df_all['月'] == target_month)
                ].copy()

            if len(prev_year_df) == 0:
                # 没有去年数据
                df_result['display_value'] = np.nan
                value_label = "同比变化(无去年数据)"
            else:
                # 标准化省份名称
                prev_year_df['province_std'] = prev_year_df['province'].replace(province_mapping)

                # 聚合
                prev_year_df = prev_year_df.groupby('province_std').agg({
                    variable: 'mean'
                }).reset_index()
                prev_year_df.rename(columns={'province_std': 'province'}, inplace=True)

                prev_year_dict = prev_year_df.set_index('province')[variable].to_dict()

                # 计算同比
                df_result['display_value'] = df_result.apply(
                    lambda row: row[variable] - prev_year_dict.get(row['province'], np.nan),
                    axis=1
                )
                value_label = "同比变化"

        else:
            raise ValueError(f"Unknown display_type: {display_type}")

        return df_result, value_label

    # ========== 计算两个变量的显示值 ==========

    df_var1, label1 = calculate_display_value(
        current_df[['province', var1]].copy(),
        var1, display_type1, df, year, month, baseline_years
    )

    df_var2, label2 = calculate_display_value(
        current_df[['province', var2]].copy(),
        var2, display_type2, df, year, month, baseline_years
    )

    # ========== 补充缺失省份（参考 spatial_map.py） ==========
    geojson_provinces = [f['properties']['name'] for f in geojson['features']]

    for province in geojson_provinces:
        if province not in df_var1['province'].values:
            missing_row = pd.DataFrame({
                'province': [province],
                'display_value': [np.nan]
            })
            df_var1 = pd.concat([df_var1, missing_row], ignore_index=True)

        if province not in df_var2['province'].values:
            missing_row = pd.DataFrame({
                'province': [province],
                'display_value': [np.nan]
            })
            df_var2 = pd.concat([df_var2, missing_row], ignore_index=True)

    # ========== 创建子图 ==========

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'{var_name_map.get(var1, var1)} - {label1}',
            f'{var_name_map.get(var2, var2)} - {label2}'
        ),
        specs=[[{'type': 'choroplethmapbox'}, {'type': 'choroplethmapbox'}]],
        horizontal_spacing=0.01
    )

    # ========== 左侧地图（变量1） ==========

    values1 = df_var1['display_value'].dropna()

    if len(values1) > 0:
        # 参考 spatial_map.py 的配色逻辑
        if display_type1 in ['anomaly', 'yoy']:
            # 距平和同比用对称范围
            vmax1 = max(abs(values1.min()), abs(values1.max()))
            vmin1 = -vmax1
            zmid1 = 0
        else:
            # 绝对值用实际范围
            vmin1 = values1.min()
            vmax1 = values1.max()
            zmid1 = None
    else:
        vmin1, vmax1, zmid1 = 0, 1, None

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=df_var1['province'],
            z=df_var1['display_value'],
            featureidkey='properties.name',
            colorscale=colorscale1,
            zmin=vmin1,
            zmax=vmax1,
            zmid=zmid1,
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='white',
            colorbar=dict(
                title=f'{var_name_map.get(var1, var1)}<br>{label1}',
                x=0.46,
                len=0.7,
                thickness=15
            ),
            hovertemplate='<b>%{location}</b><br>' +
                          f'{label1}: %{{z:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # ========== 右侧地图（变量2） ==========

    values2 = df_var2['display_value'].dropna()

    if len(values2) > 0:
        if display_type2 in ['anomaly', 'yoy']:
            vmax2 = max(abs(values2.min()), abs(values2.max()))
            vmin2 = -vmax2
            zmid2 = 0
        else:
            vmin2 = values2.min()
            vmax2 = values2.max()
            zmid2 = None
    else:
        vmin2, vmax2, zmid2 = 0, 1, None

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=df_var2['province'],
            z=df_var2['display_value'],
            featureidkey='properties.name',
            colorscale=colorscale2,
            zmin=vmin2,
            zmax=vmax2,
            zmid=zmid2,
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='white',
            colorbar=dict(
                title=f'{var_name_map.get(var2, var2)}<br>{label2}',
                x=1.01,
                len=0.7,
                thickness=15
            ),
            hovertemplate='<b>%{location}</b><br>' +
                          f'{label2}: %{{z:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # ========== 布局（参考 spatial_map.py） ==========

    fig.update_layout(
        mapbox1=dict(
            style=mapbox_style,
            zoom=3,
            center={"lat": 35.0, "lon": 105.0},
            # 可选：限制地图边界（与 spatial_map.py 一致）
            # bounds={
            #     "west": 73,
            #     "east": 135,
            #     "south": 18,
            #     "north": 54
            # }
        ),
        mapbox2=dict(
            style=mapbox_style,
            zoom=3,
            center={"lat": 35.0, "lon": 105.0},
            # bounds={
            #     "west": 73,
            #     "east": 135,
            #     "south": 18,
            #     "north": 54
            # }
        ),
        height=700,
        title=dict(
            text=f'{year}年{month}月 变量空间分布对比',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        margin={"r": 0, "t": 80, "l": 0, "b": 0}
    )

    return fig

