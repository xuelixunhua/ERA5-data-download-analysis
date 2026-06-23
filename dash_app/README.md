# Dash 气候可视化看板

这个目录存放 ERA5 月度数据的本地 Dash 看板示例。

它展示的是清洗后的气象数据如何继续变成可交互的分析界面，适合做历史复盘、区域对比和异常观察。

## 运行方式

先在仓库根目录安装依赖：

```bash
pip install -r requirements.txt
```

再进入看板目录运行：

```bash
cd dash_app
python main.py
```

默认端口：

```text
http://127.0.0.1:8031
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | Dash 应用入口，负责布局、数据加载和回调 |
| `trend_analysis.py` | 趋势、距平、年际波动等图表函数 |
| `spatial_analysis.py` | 省份横截面对比 |
| `spatial_map.py` | 中国地图热力图 |
| `extreme_events.py` | 极端事件检测与 ENSO 关联 |
| `multivariate_analysis.py` | 相关性矩阵、散点回归、双地图对比 |
| `resource_stability.py` | 资源均值-稳定性四象限 |
| `data/` | 看板运行所需的最小示例数据 |

## 数据文件

| 文件 | 用途 |
|---|---|
| `data/era5历史数据.xlsx` | 省级月度 ERA5 指标主数据，需要本地自备 |
| `data/ONI.xlsx` | ONI 指数，用于 ENSO 分析，需要本地自备 |
| `data/geo_plotly.json` | 省级地图底图，需要本地自备 |

## 注意事项

- 这里上传的是本地 Dash 看板代码，不是 PowerCurve.cn 网站源码。
- `article_scripts/` 中的 Matplotlib 文章绘图脚本没有放入本仓库。
- 仓库不上传真实气象数据文件，避免公开仓库中的数据合规和授权边界问题。
- 看板会在当前目录下生成 `导出数据/`，该目录已被 `.gitignore` 忽略。
- 如果替换数据，请保持列名与 `main.py` 中的字段映射一致。
