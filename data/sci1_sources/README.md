# SCI1 数据源工作区

原始卫星/再分析数据不应提交到 git。将本地下载放在此处或外部数据缓存中，
并在 `results/sci1_submission/*/data_manifest.json` 中记录路径/校验和。

## 数据源覆盖

### 海冰密集度 (SIC)
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| NOAA/NSIDC CDR v4 (G02202) | NOAA/NASA NSIDC | 25km, daily, 1978-present | SIC 主数据源 |
| NSIDC-0051 v2 | NASA NSIDC | 25km, daily | SIC 交叉验证 |
| OSI-450 v3 | EUMETSAT OSI SAF | 25km EASE2, daily | 欧洲北极 SIC 验证 |

### 海冰厚度 (SIT)
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| CryoSat-2 L3 | ESA/CPOM/AWI | ~1.5km along-track | 冬季厚度校准 |
| ICESat-2 ATL10 | NASA NSIDC | ~17m footprint | 高分辨率 freeboard 验证 |
| PIOMAS | U. Washington | 1°, daily, 1979-present | 长期厚度气候态 |
| Copernicus Reanalysis | CMEMS | 1/12°, daily, 1991-present | 厚度/漂移联合分布 |
| Copernicus L3 | CMEMS/C3S | satellite L3, Oct-Apr | 冻结季厚度 |

### 海冰漂移 (SID)
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| NSIDC Polar Pathfinder v4 | NASA NSIDC | 25km, daily, 1978-present | 漂移速度/方向主数据源 |
| IFREMER CERSAT | IFREMER | 62.5km, daily | 欧洲北极漂移验证 |

### 海流
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| OSCAR | NASA PO.DAAC | 1/3°, 5-day | 海表流强迫 |
| GlobCurrent | CMEMS | 0.25°, daily | 多源海流产品 |

### 大气强迫
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| ERA5 | ECMWF/C3S | 0.25°, hourly | 风/压/温强迫，SIC 验证 |
| JRA-55 | JMA | 1.25°, 6-hourly | 替代再分析，北极技能评估 |

### 冰图
| 产品 | 权威 | 分辨率 | 用途 |
|------|------|--------|------|
| NIC Sea Ice Analysis | US NIC | polygon-based | 业务冰图验证 |
| MASIE | NSIDC/NIC | 4km, daily | 高分辨率冰范围 |

### 实测校准数据
| 产品 | 权威 | 说明 | 用途 |
|------|------|------|------|
| N-ICE2015 | Norwegian Polar Institute | 2015, Fram Strait, 冰厚/强度 | Lindqvist 模型校准 |
| MOSAiC | AWI | 2019-2020, 中央北极, 全年 | 最全面的北极冰数据集 |
| HSVA Ice Tank | Hamburg Ship Model Basin | 模型尺度冰力试验 | 冰力模型验证 |
| Aalto Ice Tank | Aalto University | 模型尺度冰力试验 | 波罗的海冰力验证 |

### 船舶轨迹
| 产品 | 权威 | 用途 |
|------|------|------|
| MarineTraffic AIS | commercial | 北极船舶航线验证 |
| XueLong2 航迹 | PRIC | 直接验证 DP 场景 |

## 文献校准参数

`data_manifest.json` 现在包含 `literature_calibrations` 字段，列出从公开文献提取的
冰力模型参数 (Lindqvist 1989, ISO 19906, Riska 1997) 和雪龙2号船舶参数。

关键参数：
- 冰抗压强度: 0.5-5.0 MPa (标称 2.0 MPa, Lindqvist 1989)
- 结构系数: 0.3-0.7 (标称 0.45)
- 水线角: 15-45° (标称 30°)
- 雪龙2号排水量: ~14000t, PC7 冰级

## 使用建议

1. **最低要求**: 使用文献校准的合成数据 (literature-calibrated synthetic)，在论文中明确标注
2. **建议**: 下载 NSIDC CDR SIC + PIOMAS SIT 用于至少 3 个场景的冰况校准
3. **理想**: 下载 Copernicus Reanalysis + ERA5 用于完整的冰-海-气耦合场景
4. **验证**: 使用 N-ICE2015/MOSAiC 实测数据校准冰力模型参数

```python
# 下载示例
from arctic_quasi_dp.sci1.data_sources import download_nsidc_sic, download_copernicus_ice, download_era5_arctic

download_nsidc_sic(Path("data/sci1_sources"), start_date="2020-01-01", end_date="2020-12-31")
download_copernicus_ice(Path("data/sci1_sources"))
download_era5_arctic(Path("data/sci1_sources"))
```
