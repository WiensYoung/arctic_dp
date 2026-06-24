# SCI一区投稿级实验代码包

本补丁把当前 `arctic_quasi_dp` 项目扩展为"高等级 DP 系统的冰区增强实验平台"。论文叙事建议为：**不替代 DP3 精确定位系统，而是在高等级 DP 架构下提供冰载荷感知、风险约束、推进器退化容错、准定位安全降级与极端冰况 ice-vaning/escape 保护。**

## 新增内容

```text
src/arctic_quasi_dp/sci1/
  controllers.py     # Precision DP / Ice-aware DP / Quasi-DP / Escape / Supervisor
  nmpc_controller.py # 基于 CasADi 的 NMPC 控制器 (NEW)
  thruster.py        # 推进器分配模型: TAM + 伪逆 + 饱和 + 故障注入 (NEW)
  ice_schedule.py    # 时变冰况调度: 线性/阶跃/正弦/随机游走/分段 (NEW)
  sim_loop.py        # 自定义仿真循环: 3-DOF 动力学 + 时变冰况 + 推进器分配 (NEW)
  scenarios.py       # A-F 六组投稿场景 (含非平稳冰况和 MIZ)
  metrics.py         # RMS/P95/P99/CVaR/failure/solver-time + 统计显著性检验
  figures.py         # 顶刊风格图表: 精度-安全、失败率、实时性、尾部风险、消融、统计比较
  data_sources.py    # 权威公开数据源 registry + 下载工具
  runner.py          # Monte Carlo 实验 runner
configs/sci1/sci1_submission.yaml
scripts/run_sci1_experiments.py
data/sci1_sources/README.md
tests/sci1/test_sci1_controllers.py
tests/sci1/test_sci1_integration.py
tests/sci1/test_thruster.py          # 推进器分配测试 (NEW)
tests/sci1/test_ice_schedule.py      # 冰况调度测试 (NEW)
tests/sci1/test_sim_loop.py          # 仿真循环测试 (NEW)
.gitignore
```

## 安装

```bash
# 基础安装 (运行实验)
pip install -e .

# 安装测试依赖
pip install -e ".[test]"

# 安装 NMPC (需要 CasADi)
pip install -e ".[nmpc]"

# 安装全部依赖 (论文实验)
pip install -e ".[paper]"
```

## 快速运行 smoke 实验

```bash
# 方式 1: 使用 console script (推荐)
arctic-sci1 --profile smoke --seeds 2 --controllers pid precision ice_aware full no_cbf no_cvar no_observer no_fallback --no-traces

# 方式 2: 使用 python -m
python -m arctic_quasi_dp.sci1.runner --profile smoke --seeds 2 --controllers pid precision ice_aware full no_cbf no_cvar no_observer no_fallback --no-traces
```

输出目录：

```text
results/sci1_submission/<timestamp>/
  data_manifest.json
  scenario_manifest.json
  per_seed_metrics.csv
  aggregate_metrics_ci95.csv
  statistical_comparisons.csv
  traces/*.csv
  figures/*.png
  figures/*.pdf
```

## 论文正式实验建议

```bash
arctic-sci1 --profile paper --seeds 50 --controllers pid smc precision ice_aware full no_cbf no_cvar no_observer no_fallback
```

建议服务器运行，不建议在笔记本上直接跑全部 `paper` profile。

## 场景矩阵

| 组 | 场景 | 冰况 | 说明 |
|----|------|------|------|
| A_precision | A1, A2 | 无冰 | 开阔水域基线 |
| B_ice_enhancement | B1-B4 | 中-高 | 冰区增强, 含非平稳冰况 (B4) |
| C_fault_tolerance | C1 | 高 | 推进器退化代理 |
| D_safety_degradation | D1 | 极端 | 安全降级与 escape |
| E_realtime | E1 | 中 | 实时性验证 |
| F_miz | F1 | 低-中 | 边缘冰区 (新增) |

## 控制器矩阵

| 控制器 | 类型 | 说明 |
|--------|------|------|
| pid | 基线 | PID 控制器 |
| smc | 基线 | 滑模控制器 |
| precision | 基线 | 精确 PD 控制器 |
| ice_aware | 基线 | 冰感知控制器 (观测器 + CVaR + CBF) |
| full | 提出 | 四模式监督控制器 |
| nmpc | 对比 | 基于 CasADi 的 NMPC 控制器 (需要 casadi) |
| no_cbf | 消融 | 禁用 CBF (所有子控制器) |
| no_cvar | 消融 | 禁用 CVaR (所有子控制器) |
| no_observer | 消融 | 禁用观测器 (所有子控制器) |
| no_fallback | 消融 | 禁用准定位/escape 降级 |

## 关键修复 (v2)

1. **Monte Carlo 种子修复**: 每个 seed 使用不同随机种子 (`cfg.seed = 20260625 + seed`)
2. **消融实验完整传播**: 消融参数传播到 escape 控制器
3. **风险公式统一**: supervisor 和 IceAware 使用相同的标准化冰风险公式
4. **冰力模型升级**: 从纯经验多项式代理升级为 Lindqvist (1989) 简化物理模型
5. **Escape 模式直接恢复**: 允许 ESCAPE 直接回到 ICE_AWARE (不需经过 QUASI_DP)
6. **CBF 半径同步**: 安全区域半径从场景同步到所有控制器
7. **CVaR 种子可控**: 支持外部设置 CVaR 随机种子
8. **failure 阈值修正**: 从 25m 降到 1.5x 安全区域半径
9. **统计显著性检验**: 新增 Wilcoxon rank-sum + Cohen's d
10. **列名容错**: metrics 和 figures 支持多种列名格式
11. **时变冰况仿真**: 自定义仿真循环支持 per-step 冰况更新 (ice_schedule.py + sim_loop.py)
12. **推进器分配模型**: TAM + 伪逆分配 + 饱和裁剪 + 故障注入 (thruster.py)
13. **NMPC 控制器**: CasADi 非线性 MPC，含 CBF 约束和 RK4 离散化 (nmpc_controller.py)

## 权威数据源策略

优先使用：

1. NOAA/NSIDC CDR Passive Microwave Sea Ice Concentration v4：海冰密集度。
2. NSIDC-0051 v2：密集度交叉验证。
3. Copernicus Marine Arctic Ocean Sea Ice Reanalysis：海冰厚度、漂移速度、密集度。
4. Copernicus Marine Sea Ice Thickness Reprocessed L3：冻结季厚度校准。
5. Copernicus Marine High Resolution Arctic Sea Ice Information L4：边缘冰区/作业场景。
6. ERA5 Single Levels：风/浪强迫和海冰覆盖验证。

下载工具:

```python
from arctic_quasi_dp.sci1.data_sources import download_nsidc_sic, download_copernicus_ice
download_nsidc_sic(Path("data/sci1_sources"), start_date="2020-01-01", end_date="2020-12-31")
download_copernicus_ice(Path("data/sci1_sources"))
```

由于这些数据通常需要 Earthdata/Copernicus 账号和较大的 NetCDF 下载，补丁不打包原始数据。若无法获取，应在论文和 `data_manifest.json` 中标注为 `literature-calibrated synthetic`，并用敏感性分析证明结论不依赖单一冰载荷模型。

## 论文图表建议

至少报告：

- 表 1：场景矩阵和数据来源等级。
- 表 2：控制器/baseline/ablation 对照。
- 表 3：Monte Carlo 均值、标准差、95% CI。
- 表 4：统计显著性检验 (p 值、Cohen's d)。
- 图 1：分层 DP 增强架构。
- 图 2：代表场景轨迹 + 安全边界 + 模式切换。
- 图 3：RMS/P95/P99 定位误差箱线或误差条。
- 图 4：精度-安全 Pareto 图。
- 图 5：失位概率/安全边界违反时间。
- 图 6：推力饱和、分配残差、能耗。
- 图 7：P95 求解时间和实时性。
- 图 8：消融实验贡献图。
- 图 9：统计比较图 (p 值 + 效应量)。

## 重要限制

1. **NMPC 需要 CasADi**: `nmpc_controller.py` 需要 `pip install casadi`。未安装时 NMPC 控制器不可用，其他控制器不受影响。
2. **推进器配置为简化值**: `ThrusterConfig.vessel_xuelong2()` 使用简化参数。论文应使用真实船舶的推进器配置。
3. **冰力模型仍为代理**: 虽然升级为 Lindqvist (1989) 简化模型，但仍非精确冰力计算。顶刊建议使用冰水池实验数据校准。
4. **时变冰况需要 --use-sim-loop**: 默认使用父包 simulator (不支持时变冰况)。时变场景自动使用自定义仿真循环。
5. **CasADi/IPOPT 求解时间**: NMPC 的求解时间可能超过 PD 控制器 100 倍以上。paper profile 应测试实时性。
