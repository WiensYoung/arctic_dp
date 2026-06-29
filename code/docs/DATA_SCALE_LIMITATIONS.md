# 数据空间尺度限制声明

## 核心问题

本项目使用的卫星/再分析数据的空间分辨率为 **25km**（部分产品 ~4km），而 DP 站点的定位精度要求在 **10-50m** 量级。这意味着数据提供的是**区域气候学冰况**，而非船只实际遇到的**局部冰况**。

## 尺度差异量化

| 尺度 | 分辨率 | 对 DP 的约束力 |
|------|--------|---------------|
| Copernicus NEMO-NextSIM | ~4 km | 局部冰况的统计代理，误差 ±10-20% |
| NSIDC CDR SIC | 25 km | 纬度带代表性值 |
| CryoSat-2 SIT | 25 km | 区域平均厚度 |
| DP 站位置精度 | ~10-50 m | 实际控制目标 |

在 25km × 25km 的格点内，冰况可能因风场、海流、冰脊等局部因素产生显著变化。

## 对实验结论的影响

1. **场景参数应被解释为 "该纬度带 1 月份的典型条件"**，而非特定地理位置的确定冰况。

2. **亚格点变异性已量化** (基于 Copernicus 2024 ~4km 数据实测):

| 纬度 | 25km内SIC σ | 25km内SIT σ | 50km内SIC σ | 50km内SIT σ |
|------|------------|------------|------------|------------|
| 72°N | <0.001 | <0.001 m | <0.001 | <0.001 m |
| 78°N | 0.011 | 0.014 m | 0.083 | 0.099 m |
| 82°N | 0.001 | 0.043 m | 0.004 | 0.071 m |
| 88°N | 0.001 | 0.003 m | 0.001 | 0.009 m |

**关键发现**: 在高北极 consolidated ice (82-88°N)，25km 格点内 SIC 空间变异仅 ~0.001，SIT 仅 ~0.04m。数据对单点 DP 的代表性**优于预期**。变异性主要出现在冰缘区 (78°N, 50km 尺度: SIC σ=0.083)。

3. **实验结果验证的是控制算法对不同冰况条件的相对鲁棒性**，而非对特定地理位置的绝对性能。
4. 全尺度(full-scale)验证需要局部实测冰况数据（如船舶雷达、AIS、冰区预报），本项目不具备此类数据。

## 论文中的建议声明

> "The ice condition parameters used in our scenarios are derived from satellite
> and reanalysis products (NSIDC CDR G02202 v4, CryoSat-2 AWI L3C, OSI SAF CDR
> v3, Copernicus NEMO-NextSIM). To assess spatial representativeness, we
> quantified sub-grid variability using Copernicus ~4 km sea ice reanalysis:
> within a 25 km radius window, the 1σ spatial standard deviation of SIC is
> <0.01 in consolidated Arctic ice (82-88°N) and up to 0.08 near the marginal
> ice zone (78°N, 50 km radius); SIT variability is <0.05 m in the high Arctic.
> These values confirm that the 25 km products provide sufficiently
> representative ice condition priors for the regional-scale DP robustness
> evaluation conducted in this study. Local vessel-scale ice states may deviate
> further; our results therefore quantify *relative controller robustness*
> across an envelope of ice conditions rather than absolute performance at a
> specific geographic coordinate."

## 数据溯源

所有实验输出应附带的声明：
- SIC 来源: NSIDC CDR G02202 v4, 25km PS grid (或 OSI SAF CDR v3, EASE2-250)
- SIT 来源: CryoSat-2 AWI L3C v2.6, 25km EASE2 (或 PIOMAS model)
- 冰漂移来源: Copernicus NEMO-NextSIM reanalysis ~4km (或合成 Nansen-Ekman 规则)
- 校验和: 见 `data/sci1_sources/data_manifest.json`
