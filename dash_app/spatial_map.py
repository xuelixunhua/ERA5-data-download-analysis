"""
空间地图可视化模块
用于绘制中国省份热力图
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

_BASE = Path(__file__).parent

def load_china_geojson(file_path=None):
    if file_path is None:
        file_path = str(_BASE / 'data' / 'geo_plotly.json')
    """
    加载中国省份GeoJSON

    如果文件不存在,返回None(后续会提供下载链接)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ GeoJSON文件未找到: {file_path}")
        return None


def create_china_map_heatmap(df, target_year, target_month, variable,
                             display_type='absolute', baseline_years=None,
                             colorscale='RdBu_r', var_display_name=None,
                             period_years=None):
    """
    创建中国省份热力图

    ⚠️ 注意：此函数内部会合并蒙东蒙西为内蒙古自治区（仅用于地图显示）
    """

    geojson = load_china_geojson()
    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ 缺少GeoJSON文件",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig

    # ========== 省份标准化（仅地图用） ==========
    province_mapping = {
        '蒙东': '内蒙古自治区',
        '蒙西': '内蒙古自治区',
    }

    def standardize_provinces(data_df, var_col):  # ← 改这里：增加 var_col 参数
        """
        专门用于地图显示的省份标准化
        ⚠️ 会修改数据，不要用于其他分析

        Parameters:
        -----------
        data_df : DataFrame
            原始数据
        var_col : str
            需要聚合的变量列名

        Returns:
        --------
        DataFrame
            标准化后的数据
        """
        temp = data_df.copy()  # ← 重要：复制数据，不污染原始df
        temp['province_std'] = temp['province'].replace(province_mapping)

        # 按标准化后的省份聚合
        temp = temp.groupby('province_std').agg({var_col: 'mean'}).reset_index()
        temp.rename(columns={'province_std': 'province'}, inplace=True)
        return temp

    # ========== 数据准备 ==========

    if display_type == 'period_mean':
        # 年份段均值
        if not period_years or len(period_years) != 2:
            raise ValueError("period_mean模式需要提供 period_years=(start, end)")

        start_year, end_year = period_years

        # 筛选数据
        period_df = df[
            (df['年'] >= start_year) &
            (df['年'] <= end_year)
            ].copy()

        # 处理月份筛选
        if target_month != 'all':
            period_df = period_df[period_df['月'] == target_month]

        if len(period_df) == 0:
            month_str = f"{target_month}月" if target_month != 'all' else "全年"
            fig = go.Figure()
            fig.add_annotation(
                text=f"无数据: {start_year}-{end_year}年{month_str}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        # 标准化 + 聚合（仅地图用）
        current_df = standardize_provinces(period_df, variable)  # ← 传2个参数
        current_df['display_value'] = current_df[variable]

        # 动态标签
        month_str = f"{target_month}月" if target_month != 'all' else "全年"
        value_label = f"{start_year}-{end_year}年{month_str}均值"

    else:
        # 原有逻辑（绝对值/距平/同比）
        # ⚠️ 这些模式不支持"全年"，需要验证
        if target_month == 'all':
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ 绝对值/距平/同比模式不支持全年<br>请选择具体月份",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="orange")
            )
            return fig

        current_df = df[
            (df['年'] == target_year) &
            (df['月'] == target_month)
            ].copy()

        if len(current_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"无数据: {target_year}年{target_month}月",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        # 标准化（仅地图用）
        current_df = standardize_provinces(current_df, variable)  # ← 传2个参数

        if display_type == 'absolute':
            current_df['display_value'] = current_df[variable]
            value_label = "绝对值"

        elif display_type == 'anomaly':
            # 距平值
            if baseline_years:
                baseline_df = df[
                    (df['年'] >= baseline_years[0]) &
                    (df['年'] <= baseline_years[1]) &
                    (df['月'] == target_month)
                    ]
            else:
                baseline_df = df[df['月'] == target_month]

            # 标准化基准期数据
            baseline_df = standardize_provinces(baseline_df, variable)  # ← 传2个参数
            climate_mean = baseline_df.set_index('province')[variable]

            current_df['display_value'] = current_df.apply(
                lambda row: row[variable] - climate_mean.get(row['province'], np.nan),
                axis=1
            )
            value_label = "距平"

        elif display_type == 'yoy':
            # 同比变化
            prev_year_df = df[
                (df['年'] == target_year - 1) &
                (df['月'] == target_month)
                ].copy()

            prev_year_df = standardize_provinces(prev_year_df, variable)  # ← 传2个参数
            prev_year_dict = prev_year_df.set_index('province')[variable].to_dict()

            current_df['display_value'] = current_df.apply(
                lambda row: row[variable] - prev_year_dict.get(row['province'], np.nan),
                axis=1
            )
            value_label = "同比变化"

    # ========== 绘图 ==========
    var_name = var_display_name if var_display_name else variable

    values = current_df['display_value'].dropna()
    if len(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="无有效数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # 配色范围
    if display_type in ['anomaly', 'yoy']:
        vmax = max(abs(values.min()), abs(values.max()))
        vmin = -vmax
        zmid = 0
    else:
        vmin = values.min()
        vmax = values.max()
        zmid = None

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=current_df['province'],
        z=current_df['display_value'],
        featureidkey='properties.name',
        colorscale=colorscale,
        zmin=vmin, zmax=vmax, zmid=zmid,
        marker_opacity=0.7,
        marker_line_width=1,
        marker_line_color='white',
        colorbar=dict(
            title=f"{var_name}<br>{value_label}",
            thickness=15, len=0.7, x=1.02
        ),
        hovertemplate='<b>%{location}</b><br>' +
                      f'{value_label}: %{{z:.2f}}<extra></extra>'
    ))

    # 标题动态化
    if display_type == 'period_mean':
        month_str = f"{target_month}月" if target_month != 'all' else "全年"
        title_text = f'{period_years[0]}-{period_years[1]}年{month_str} {var_name} 均值分布'
    else:
        title_text = f'{target_year}年{target_month}月 {var_name} {value_label}分布'

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 35.0, "lon": 105.0},
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=700,
        title=title_text
    )

    return fig


def create_dual_map_comparison(df, target_year, target_month, variable,
                               baseline_years=None, var_display_name=None):
    """
    创建双地图对比(绝对值 vs 距平值)
    """
    from plotly.subplots import make_subplots

    # 这个函数可以后续扩展,用于同时显示两张地图
    # 例如:左边显示绝对值,右边显示距平
    pass
