# Arctic DP 项目代码现状审计报告

**审计日期**: 2026-06-30
**审计范围**: 全部源代码（47 个 `.py` 文件）、16 个 YAML 配置、17 个已下载数据源、50 个测试文件
**审计轮次**: 第三轮（前两轮已于 2026-06-28 和 2026-06-29 完成）

---

## 一、项目概述

Arctic DP（北极动态定位）是一个面向学术论文的仿真实验项目，研究冰载荷扰动下的船舶安全约束控制。核心算法基于 HOCBF（高阶控制屏障函数）+ QP 安全滤波器 + 模式监督器架构，包含 22 个控制器变体、49 个实验场景、17 个卫星/再分析数据源。

| 指标 | 数值 |
|------|------|
| Python 源文件 | 47 个 |
| 代码总量 | ~9,500 行 |
| YAML 配置文件 | 16 个 |
| 测试文件 | 50 个 |
| 实验场景 | 49 个（A-I 组） |
| 控制器变体 | 22 个 |
| 已下载数据源 | 17 个（均有 SHA256 校验和） |
| NetCDF/CSV/DAT 数据文件 | ~75 个 |

---

## 二、Bug 修复历程

经过三轮审计，共修复 **76 个问题**：

| 轮次 | 日期 | 修复数 | 类别 |
|------|------|--------|------|
| 第一轮（初始审计） | 2026-06-28 | 28 | 运行时 5、逻辑 7、架构 2、数据 8、测试 3、其他 3 |
| 第二轮（深度审计） | 2026-06-28 | 10 | LQG 协方差、ADRC 坐标系、风场坐标系、推力饱和检测等 |
| 数据管线增强 | 2026-06-29 | 12 | SHA256 校验和、校准报告重生成、漂移数据确认、精度量化 |
| 第三轮（全面修复） | 2026-06-30 | 26 | Kalman 协方差、HOCBF 约束符号、经度环绕、溯源标志、审稿人敏感项 |

### 代表性修复项

| 严重度 | 问题 | 文件 | 修复内容 |
|--------|------|------|---------|
| 🔴 | Kalman 滤波器协方差 P[1,0] 公式错误 | `sensor_models.py:368` | `(1-K1)*P12` → `P12 - K2*P11` |
| 🔴 | HOCBF 约束在 target_pos=None 时反向 | `hocbf.py:88` | `b_hocbf=1e6` → `b_hocbf=-1e6` |
| 🔴 | 数据溯源标志对合成数据报告 True | `data_bridge.py:364` | 改用 `_drift_source` 判断 |
| 🔴 | LQG 卡尔曼滤波协方差预测公式错误 | `baseline_controllers.py:196` | `A*P*A^T*dt` → `F*P*F^T + Q*dt` |
| 🔴 | NMPC 航向代价 180° 虚假局部极小 | `nmpc_controller.py:243` | `(1-cosΔ)²` → `(1-cosΔ)` |
| 🔴 | ADRC 控制器混合 NED/体坐标系 | `baseline_controllers.py:669` | 统一 NED 帧计算 |
| 🟠 | PD 控制器硬编码质量/惯量 | `controllers.py:285` | 改用 `self._vessel_mass` / `self._vessel_Izz` |
| 🟠 | 经度坐标环绕仅处理单向 | `data_bridge.py`（4处） | 新增 `_adjust_lon_for_grid()` 双向转换 |
| 🟠 | `_extract_point` 2D 索引选错维度 | `data_bridge.py:261` | 增加维度匹配检测 |
| 🟠 | DataSource dataclass 缺 manifest 字段 | `data_sources.py:40-57` | 添加 7 个扩展字段 |
| 🟠 | DOBMPC 偏航力矩缺少横向力臂 | `baseline_controllers.py:419,547` | 添加 `-0.05*B*Fx` 项 |
| 🟡 | 多数据源时间轴不匹配崩溃 | `data_bridge.py:330` | 新增 `_validate_time_alignment()` |
| 🟡 | `_certified_radius` 返回未认证值 | `actuator_feasible_set.py:209` | `max(1.0, lo)` → `lo` |

---

## 三、当前代码质量评估

### 3.1 编译状态

```
全部 47 个 .py 源文件：0 编译错误
```

### 3.2 代码规范

| 检查项 | 结果 |
|--------|------|
| TODO/FIXME/HACK/BUG 标记 | 0 个 |
| 循环导入 | 2 处已知（通过延迟导入规避，有意为之） |
| 全局可变状态 | 2 处（`_RUN_DATA_USAGE`、`_FAIL_FAST_ON_MISSING_DATA`，单进程场景安全） |
| 静默异常吞没 | 8 处（`data_bridge.py` 中 `except Exception`，数据加载降级策略，有意为之） |

### 3.3 代码重复

| 重复点 | 严重度 | 说明 |
|--------|--------|------|
| NMPC/DOBMPC NLP 构建 | 中 | ~80 行重复，共享相同的动力学和目标函数结构 |
| 冰力模型 | 低 | Lindqvist 公式在 4 处出现，但 `ice_force_common.py` 已是规范来源 |
| `_make_state()` 测试辅助函数 | 低 | 3 个测试文件中重复定义 |

### 3.4 函数复杂度

| 文件 | 函数 | 行数 | 建议 |
|------|------|------|------|
| `metrics.py` | `summarize_run()` | ~330 | 可拆分为 5-7 个辅助函数 |
| `runner.py` | `_run_task_worker()` | ~150 | 可接受 |
| `controllers.py` | `ModeSupervisedIceDPController._select_mode()` | ~120 | 可接受 |

---

## 四、实验设计评估

### 4.1 场景覆盖

| 组 | 主题 | 场景数 | 评估 |
|----|------|--------|------|
| A | 开水域 DP 精度 | 5 | 充分 |
| B | 冰况扰动 | 7 | 良好，覆盖非平稳冰况 |
| C | 推进器退化 | 8 | 良好，故障模式多样 |
| D | 安全降级 | 4 | 核心消融，可接受 |
| E | 传感器退化 | 8 | 良好 |
| F | 运行时可行性 | 5 | 充分 |
| G | 冰模型灵敏度 | 7 | 良好 |
| H | 数据驱动回放 | 5 | 含真实 Copernicus 漂移数据 |
| I | 安全滤波器验证 | 5 | 可接受 |

### 4.2 已知实验设计限制

1. **场景时长较短**：Paper profile 默认 300 秒（5 分钟），真实 DP 验证通常 1-3 小时
2. **缺少波浪扰动场景**：A2 场景已标注 "current model not implemented"
3. **缺少多船/编队场景**：仅单船 DP
4. **缺少浅水效应**：北极近岸操作的水深限制未建模
5. **代理尺度**：冰强度为真实值的 1/6000，已在文档中明确声明

### 4.3 统计方法

| 方法 | 实现 | 评估 |
|------|------|------|
| 配对 Wilcoxon signed-rank | ✅ | 正确实现 |
| Cohen's dz 效应量 | ✅ | 含方向符号检查 |
| Bootstrap 置信区间 | ✅ | 5000 次重采样，md5 确定性种子 |
| Holm-Bonferroni 校正 | ✅ | 多重比较校正 |
| 功效分析 | ✅ | 已文档化（30 seeds, d≥0.5, α=0.05, 1-β=0.80） |

---

## 五、数据评估

### 5.1 种类

| 变量 | 数据源数 | 评估 |
|------|---------|------|
| 海冰密集度 (SIC) | 4 | 冗余充分 |
| 海冰厚度 (SIT) | 4 | 包含 CryoSat-2、PIOMAS、Copernicus |
| 冰漂移 | 1（Copernicus ~4km） | 80°N/0°E 实测 0.16 m/s @ 194°，与场景参数吻合 |
| 风场 | 2（ERA5） | 全月数据可用，`DataDrivenWindSchedule` 已接入 |
| 海流 | 1（OSCAR 6 天） | 加载器存在，未完全接入 |
| 原位实测 | 1（N-ICE2015） | ZIP 打包，未解压使用 |

### 5.2 质量

- **SHA256 校验和**：17/17（100%）已下载源
- **校准报告**：基于磁盘实际 Copernicus 2024 数据重新生成
- **废弃产品 ID**：3 个已添加迁移路径

### 5.3 精度

- **空间分辨率**：25km（多数产品）至 ~4km（Copernicus）
- **亚格点变异性**：已量化（`spatial_variability.json`）
  - 高北极 82-88°N：25km 内 SIC σ < 0.01，SIT σ < 0.05m
  - 冰缘区 78°N：50km 内 SIC σ = 0.083，SIT σ = 0.099m

### 5.4 可靠性

- 每个数据文件可独立校验（SHA256）
- 溯源链由 `DataDrivenIceSchedule.provenance` 标记
- `DATA_SCALE_LIMITATIONS.md` 提供了论文声明模板

---

## 六、审稿人敏感项处理

| 问题 | 处理 |
|------|------|
| `robust_mpc` 名不副实 | 文档字符串声明 "NOT a true robust MPC"；显示标签改为 "Conserv. PD" |
| `tube_mpc` 名不副实 | 文档字符串列举缺失的 4 项 Tube MPC 组件；显示标签改为 "Margin PD" |
| "CVaR" 术语不准确 | `METHOD_THEORETICAL_ASSUMPTIONS.md` A8 节建议 "risk-adaptive scaling" |
| 代理尺度参数未文档化 | A6 节含 3 参数对比表（proxy vs full-scale） |
| 缺功效分析 | A9 节提供 30 seeds 的统计依据 |
| 冰强度软因子 | A6 节明确声明，冰力缩放为代理尺度设计 |

---

## 七、已知限制（非 Bug）

以下为经过评估后保留的**设计权衡**，而非需要修复的问题：

1. **`data_bridge.py` 中 8 处 `except Exception`**：数据加载失败不能崩溃实验管线。静默降级到合成回退或返回 None 是有意的容错策略。

2. **`_FAIL_FAST_ON_MISSING_DATA` 和 `_RUN_DATA_USAGE` 模块级全局变量**：当前 runner 为单进程调度，且每次 `run_experiments()` 调用都会 `clear()`。不适用于多线程场景，但当前架构不需要。

3. **`PACKAGED_REPLAY_SOURCES` 计入 `AUTHORITATIVE_SOURCES`**：mock fixture 在 manifest 中已标记为 `packaged_mock_fixture`，可区分。

4. **场景时长 300 秒**：受限于 CasADi IPOPT 求解器计算开销。增加时长需要优化 NLP 求解速度（warm-start、并行化等）。

5. **未接入的 OSCAR 海流数据**：当前仿真未建模海流扰动。`DataDrivenCurrentSchedule` 类已就绪，可在需要时接入。

---

## 八、结论

经过三轮共 76 个修复，项目代码当前处于可提交状态：

- **运行稳定性**：47 个源文件零编译错误，已知的严重/高危 bug 均已修复
- **逻辑正确性**：Kalman 滤波器、HOCBF 约束、坐标系转换、偏航力矩公式等关键数学路径已验证
- **数据完整性**：100% 下载源有 SHA256，校准报告可复现，冰漂移数据确认可用
- **审稿人准备**：过量声明已消解，代理尺度已文档化，统计功效已说明
- **代码清洁度**：0 个 TODO/FIXME/HACK/BUG 遗留标记

项目适合以当前状态提交到 GitHub 并进入论文写作阶段。论文中需如实引用 `METHOD_THEORETICAL_ASSUMPTIONS.md`、`DATA_SCALE_LIMITATIONS.md` 和 `CALIBRATION_REPORT.md` 中的声明。
