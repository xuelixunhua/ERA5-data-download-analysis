import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# 地理分区排序逻辑
# ============================================================================

# 定义符合中国地理分区的标准排序
# 注意：这里使用了全称，必须与数据中的省份名称完全匹配
PROVINCE_SORT_ORDER = [
    # 汇总
    '全国平均',
    # 西北
    '西藏', '新疆', '宁夏','甘肃', '青海', '陕西',  '蒙西',
    # 东北
    '蒙东','黑龙江','吉林', '辽宁',
# '吉林省',
    # 华北
    '山西', '北京', '河北', '山东',
    # 华中
    '河南', '湖北', '湖南',
    # 华东
    '上海','浙江','江苏', '安徽','江西',
    # 华南
    '广东', '广西', '海南', '福建',
    # 西南
    '四川', '重庆', '贵州', '云南',


]

PROVINCE_ORDER_MAP = {name: i for i, name in enumerate(PROVINCE_SORT_ORDER)}


def get_geo_sort_key(province_name):
    """获取地理排序索引"""
    name = str(province_name).strip()
    if name in PROVINCE_ORDER_MAP:
        return PROVINCE_ORDER_MAP[name]
    for known_name, index in PROVINCE_ORDER_MAP.items():
        if known_name in name or name in known_name:
            return index
    return 9999



# ============================================================================
# 绘图逻辑
# ============================================================================

def create_cross_sectional_comparison(df, target_year, target_month, variable, baseline_years=None,
                                      sort_method='geo', sort_target='climate'):
    """
    创建横截面多维对比图

    Parameters:
    -----------
    sort_method : str
        'geo'  - 地理分区排序 (默认)
        'desc' - 数值降序
        'asc'  - 数值升序
    sort_target : str
        'climate' - 按距平值排序 (默认)
        'yoy'     - 按同比值排序
        'mom'     - 按环比值排序
    """
    # --- 1. 数据准备 ---

    # 目标时间点数据
    current_df = df[(df['年'] == target_year) & (df['月'] == target_month)].set_index('province')[variable]

    if current_df.empty:
        return go.Figure().update_layout(title="无选定时间的当月数据")

    # A. 上月数据 (MoM)
    prev_month_year = target_year if target_month > 1 else target_year - 1
    prev_month_val = target_month - 1 if target_month > 1 else 12
    prev_month_df = df[(df['年'] == prev_month_year) & (df['月'] == prev_month_val)].set_index('province')[variable]

    # B. 去年同月数据 (YoY)
    prev_year_df = df[(df['年'] == target_year - 1) & (df['月'] == target_month)].set_index('province')[variable]

    # C. 多年均值数据 (Climatology)
    if baseline_years:
        base_df = df[(df['年'] >= baseline_years[0]) & (df['年'] <= baseline_years[1])]
        baseline_title_str = f"{baseline_years[0]}-{baseline_years[1]}年均值"
    else:
        base_df = df
        baseline_title_str = "历史全时段均值"

    climate_mean_df = base_df[base_df['月'] == target_month].groupby('province')[variable].mean()

    # --- 2. 计算原始差值 (用于排序) ---
    # 先对齐索引，防止相减时出错
    common_index = current_df.index

    # 原始计算 (未排序)
    delta_climate_raw = current_df - climate_mean_df.reindex(common_index)
    delta_yoy_raw = current_df - prev_year_df.reindex(common_index)
    delta_mom_raw = current_df - prev_month_df.reindex(common_index)

    # --- 3. 确定排序顺序 ---

    all_provinces = current_df.index.tolist()

    if sort_method == 'geo':
        # 地理排序 (忽略 sort_target)
        sorted_provinces = sorted(all_provinces, key=get_geo_sort_key)
    else:
        # 数值排序
        # 1. 确定依据哪个指标排
        if sort_target == 'yoy':
            target_series = delta_yoy_raw
        elif sort_target == 'mom':
            target_series = delta_mom_raw
        else:  # default to climate
            target_series = delta_climate_raw

        # 2. 确定升序还是降序
        ascending = (sort_method == 'asc')

        # 3. 执行排序
        sorted_provinces = target_series.sort_values(ascending=ascending).index.tolist()

    # --- 4. 数据重组 ---
    # 按照确定好的顺序，强制重排所有数据
    current_df = current_df.reindex(sorted_provinces)
    prev_month_df = prev_month_df.reindex(sorted_provinces)
    prev_year_df = prev_year_df.reindex(sorted_provinces)
    climate_mean_df = climate_mean_df.reindex(sorted_provinces)

    # 计算差值 (最终绘图用)
    delta_mom = current_df - prev_month_df
    delta_yoy = current_df - prev_year_df
    delta_climate = current_df - climate_mean_df

    # --- 5. 动态标题生成 ---
    title1 = f"{target_year}年{target_month}月 {variable} 较{baseline_title_str}变动 (距平)"
    title2 = f"{target_year}年{target_month}月 {variable} 较去年同月变动 (同比)"
    title3 = f"{target_year}年{target_month}月 {variable} 较上月变动 (环比)"

    # --- 6. 绘图 ---

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(title1, title2, title3)
    )

    def add_bar_trace(delta_series, row_idx, name):
        colors = ['#d62728' if v >= 0 else '#1f77b4' for v in delta_series.fillna(0)]

        fig.add_trace(go.Bar(
            x=delta_series.index,
            y=delta_series.values,
            name=name,
            marker_color=colors,
            showlegend=False,
            hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:.2f}}<extra></extra>'
        ), row=row_idx, col=1)

        # 零线
        fig.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1, row=row_idx, col=1)

    # 绘制三个子图
    add_bar_trace(delta_climate, 1, "距平")
    add_bar_trace(delta_yoy, 2, "同比")
    add_bar_trace(delta_mom, 3, "环比")

    # --- 7. 布局美化 ---
    fig.update_layout(
        margin=dict(t=40, b=50, l=50, r=20),
        height=900,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_annotations(font_size=16)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=13))

    return fig