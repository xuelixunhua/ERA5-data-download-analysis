# ERA5 数据下载及分析

这个仓库用于演示如何从 Copernicus Climate Data Store 获取 ERA5 月度再分析数据，并做最基础的数据检查。

它的重点不是复杂建模，而是把“气象数据从哪里来、网页怎么手动下载、代码怎么自动下载”这条路径讲清楚。

## 适合谁

- 想自己拿 ERA5 数据，但第一次打开英文页面不知道从哪里下手的人
- 做电力、风光、水电、负荷或气候专题分析，需要历史气象底座的人
- 希望把一次性网页下载，改成可重复、可维护脚本的人

## 数据源

ERA5 数据来自 Copernicus Climate Data Store：

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means

本示例使用的数据集是：

`ERA5 monthly averaged data on single levels from 1940 to present`

## 使用前准备

1. 注册 Copernicus Climate Data Store 账号。
2. 在个人账号页面配置 CDS API token。
3. 按官方说明创建本机认证文件：

Windows 通常位于：

```text
C:\Users\你的用户名\.cdsapirc
```

macOS / Linux 通常位于：

```text
~/.cdsapirc
```

4. 安装依赖：

```bash
pip install -r requirements.txt
```

## 手动下载流程

第一次建议先手动走一遍网页流程，这样后面看代码会更清楚。

1. 打开 ERA5 月度数据集页面。
2. 进入 `Download` 页签。
3. 在 `Product type` 中选择 `Monthly averaged reanalysis`。
4. 在 `Variable` 中选择需要的变量。
5. 选择年份、月份和时间。
6. 设置空间范围。
7. 提交下载请求，等待生成文件。

如果只做中国区域历史分析，可以先使用下面这个空间范围：

```text
North: 55
West: 70
South: 15
East: 135
```

## 代码下载

下载中国区域 ERA5 月度数据：

```bash
python download_era5_monthly.py --years 2024 --months 1 2 3
```

指定输出文件：

```bash
python download_era5_monthly.py --years 2024 2025 --months 12 1 --output data/era5_china_2024_2025.nc
```

只下载指定变量：

```bash
python download_era5_monthly.py --years 2024 --months 7 8 --variables 2m_temperature total_precipitation
```

默认变量包括：

| 变量名 | 常见用途 |
|---|---|
| `2m_temperature` | 气温、负荷、高温/低温风险 |
| `10m_u_component_of_wind` | 10 米纬向风分量 |
| `10m_v_component_of_wind` | 10 米经向风分量 |
| `100m_u_component_of_wind` | 100 米纬向风分量，常用于风电相关分析 |
| `100m_v_component_of_wind` | 100 米经向风分量，常用于风电相关分析 |
| `surface_solar_radiation_downwards` | 地表向下太阳辐射，常用于光伏资源分析 |
| `total_precipitation` | 总降水，常用于水电、干旱、洪涝分析 |

## 查看下载结果

下载完成后，可以用下面的脚本快速查看 NetCDF 文件里包含哪些变量和维度：

```bash
python inspect_netcdf.py data/era5_china_monthly.nc
```

这个脚本不会做完整清洗，只用于确认文件是否下载成功、变量是否存在。

## 后续可以继续做什么

ERA5 原始文件通常还不能直接用于业务分析。后面通常还需要：

- 单位转换，例如 K 转摄氏度、辐射量转换为日均功率口径
- 风速合成，例如由 u/v 分量合成实际风速
- 空间聚合，例如从格点数据聚合到省级、市级或场站区域
- 时间聚合，例如从小时、日、月统一到分析所需口径

这些属于数据清洗和业务建模部分，可以在后续版本继续补充。
