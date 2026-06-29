# Arctic-DP 项目全面审计报告

**审计日期**: 2026-06-28
**审计范围**: 全部源代码（`code/src/`, `code/scripts/`, `code/tests/`, `code/configs/`）
**审计维度**: 运行时Bug、逻辑Bug、架构缺陷、模块间耦合、实验设计、前沿对标

---

## 一、运行时Bug（会导致崩溃或错误结果）

### 🔴 严重 (P0)

#### BUG-01: LQG 卡尔曼滤波协方差预测公式错误
**文件**: `baseline_controllers.py:196-197`
```python
x_pred = self._x_hat + dt * (A @ self._x_hat)
P_pred = A @ self._P @ A.T * dt + Q_k * dt
```
**问题**: 离散化协方差预测应为 `P_pred = F @ P @ F.T + Q_d`，其中 `F = I + A*dt`。代码直接用连续时间矩阵 `A` 代替 `F`，缺少单位阵和交叉项，导致卡尔曼增益计算错误，LQG 控制器的状态估计不准确。
**影响**: LQG 基线控制器的状态估计和控制性能不可信，论文对比实验中 LQG 的结果可能偏保守或偏激进。
**修复**: `F = np.eye(6) + A * dt; P_pred = F @ self._P @ F.T + Q_k * dt`

#### BUG-02: NMPC 航向代价函数在180°存在虚假局部极小
**文件**: `nmpc_controller.py:243-245`
```python
heading_err = 1.0 - cs.cos(X[2, k] - param[2])
obj += p.Q_heading * heading_err ** 2
```
**问题**: `(1 - cos(Δ))²` 的导数 `2(1-cosΔ)sinΔ` 在 `Δ=π` 处也为零，即180°航向误差也是局部极小值。NMPC 求解器可能收敛到船体完全朝向反面的解。终端代价（line 296-297）存在同样问题。
**影响**: 在航向偏差较大时，NMPC 可能给出完全错误的航向指令。
**修复**: 使用 `0.5 * (1 - cs.cos(delta))`（不平方）或 `cs.sin(delta/2)**2`，这两种形式在 `Δ=π` 处都是全局最大而非极小。

#### BUG-03: DOBMPC 控制器 `reset()` 未初始化属性导致崩溃
**文件**: `baseline_controllers.py:592`
```python
self._has_observer_estimate = False  # reset() 中
```
**问题**: `_has_observer_estimate` 从未在 `__init__` 中初始化，仅在 `set_observer_estimate()` 中设置。如果在调用 `set_observer_estimate()` 之前调用 `reset()`，会抛出 `AttributeError`。
**影响**: 首次 `reset()` 后才能安全使用 `set_observer_estimate()`；如果 runner 的初始化顺序改变，会崩溃。
**修复**: 在 `__init__` 中添加 `self._has_observer_estimate = False`。

### 🟠 中等 (P1)

#### BUG-04: 航向角单位检测启发式不可靠
**文件**: `sim_loop.py:337,356`
```python
target_psi_rad = math.radians(target_psi) if abs(target_psi) > 2 * math.pi else target_psi
```
**问题**: 阈值 `2*pi ≈ 6.28` 意味着 7.0 弧度（≈401°，绕回≈41°）会被误判为角度值并再次转换。正确阈值应为 `pi`（因为角度值范围通常是 -180~180）。
**影响**: 某些场景的目标航向可能被错误转换。
**修复**: 使用显式单位参数而非启发式检测。

#### BUG-05: SciPy 回退求解器失败时返回不安全控制量
**文件**: `safety_filter.py:202-207`
```python
if result.success and np.all(np.isfinite(result.x)):
    tau_safe = np.asarray(result.x, dtype=np.float64)
else:
    tau_safe = tau0  # 返回裁剪后的 tau_des，无安全保证
    status = "fallback_scipy_last_resort"
```
**问题**: 当 SLSQP 求解失败时，返回 `tau_des` 的裁剪版本，无 HOCBF 约束保证。但外层 `solve()` 仍标记 `qp_success=True`（line 306），误导调用者认为安全约束已满足。
**影响**: 在数值困难的边界情况下，安全滤波器可能返回不安全的控制量而不报告错误。

#### BUG-06: ADRC 控制器混合 NED/体坐标系
**文件**: `baseline_controllers.py:669-677`
```python
pos_err = np.array([target_x - state[0], target_y - state[1]])  # NED
vel = state[3:5]  # 体坐标系
tau = kp * pos_err - kd * vel  # 混合!
```
**问题**: 位置误差在 NED 帧，速度在体帧。当航向非零时，PD 定律的阻尼项作用在错误的分量上。与 PID/SMC 控制器（正确做坐标变换）的对比不公平。
**影响**: ADRC 基线控制器性能偏弱，对比实验中夸大了所提方法的优势。

#### BUG-07: `data_bridge.py` 标量 vy 维度不一致
**文件**: `data_bridge.py:233-234`
```python
vx = np.array([float(vx)])   # 1-d 数组
vy = np.array(float(vy))     # 0-d 数组!
```
**问题**: `np.array(float(vy))` 创建 0 维数组，而 `vx` 是 1 维。虽然后续广播不会崩溃，但速度方向计算结果的维度不对称。
**修复**: `vy = np.array([float(vy)])`

#### BUG-08: `runner._save_metadata` 中 `full_scale_ready` 逻辑反转
**文件**: `runner.py:531`
```python
"full_scale_ready": not _is_full,
```
**问题**: `_is_full` 为 True 时（全尺度实验配置），`full_scale_ready` 反而为 True。语义上应该是"全尺度尚未就绪"才对。实际上代码的意图可能是 `not _is_full` 表示"非全尺度=已就绪"，但变量名 `full_scale_ready` 产生歧义。
**影响**: 清单文件的元数据语义混乱。

#### BUG-09: `logging_config.py` 日志级别 falsy 值 bug
**文件**: `logging_config.py:36`
```python
logger.setLevel(level or logging.INFO)
```
**问题**: `level=0`（即 `logging.DEBUG`）时，`0 or logging.INFO` 返回 `logging.INFO`，忽略调用者的显式请求。
**修复**: `logger.setLevel(level if level is not None else logging.INFO)`

---

## 二、逻辑Bug（不会崩溃但结果不正确）

### 🟠 中等 (P1)

#### LOGIC-01: CVaR 代理中 `control_sat` 对所有样本为常量
**文件**: `controllers.py:371-384`
```python
control_sat = np.linalg.norm(base_tau[:2]) / max_f  # 标量
ice_disturbance = np.abs(draws) / max_f              # n 个随机样本
losses = 0.4 * control_sat + 0.3 * ice_disturbance + 0.3 * violation
```
**问题**: `control_sat` 是标量常量，不随样本变化。CVaR 分位数完全由冰力扰动的尾部驱动，控制饱和度仅作为常数偏移。这违背了"多因子联合风险"的设计意图。
**修复**: 应对控制力也进行随机采样（如添加执行器不确定性）。

#### LOGIC-02: 推力器速率限制与分配器内部状态不同步
**文件**: `sim_loop.py:407-412`
```python
thrusts, alloc_diag = allocator.allocate(tau_cmd, dt=dt)  # 内部已更新方位角
thrusts, rate_diag = _apply_rate_limits(thrusts, prev_thrusts, ...)  # 仅限制推力幅值
```
**问题**: `allocate()` 内部已更新方位角状态，但外部仅对推力幅值做速率限制。方位角的速率限制在分配器内部完成，推力的速率限制在外部完成，两者不对称。分配器内部跟踪的"上一步推力"不反映外部裁剪后的值。

#### LOGIC-03: 推力分配饱和裁剪忽略方向信息
**文件**: `thruster.py:277-335`
**问题**: 当某推进器饱和时，标量超额力被均匀重新分配给其余推进器。但 TAM 将每个推进器映射到 3-DOF 向量 (Fx, Fy, Mz)。重新分配标量超额忽略了饱和推进器的方向贡献，可能产生次优甚至退化的分配。
**改进**: 应固定饱和推进器后重新求解分配问题。

#### LOGIC-04: `RobustMPC` 风险公式与其他控制器不一致
**文件**: `baseline_controllers.py:761`
```python
risk = c * (0.5 * h + v)  # 线性增长，无归一化
```
**对比**: 标准化公式 `c * (h/2.5) * (0.3 + v)` 有内置归一化。RobustMPC 的风险尺度不同，会比预期更激进地收紧约束。
**影响**: RobustMPC 基线在高冰况下的行为与其他控制器不在同一可比尺度上。

#### LOGIC-05: 冰力偏航力矩仅用横向分量
**文件**: `sim_loop.py:454`, `ice_models.py:141`
```python
tau_wind[2] = fy_wind * params.length * _WIND_MOMENT_ARM_FACTOR
mz = lever * force_body_2d[1]  # 仅 sway 分量
```
**问题**: 完整的偏航力矩应为 `Mz = x*Fy - y*Fx`。对于斜向冰载荷，surge 分量 Fx 在非零 y 偏移处也产生力矩。

#### LOGIC-06: `scenarios.py::ice_conditions_at` 角度插值不绕回
**文件**: `scenarios.py:127`
**问题**: 漂移方向在初始值（如350°）和最终值（如10°）之间线性插值，会经过180°（南向），而非走0°/360°的短路径。`ice_schedule.py` 正确使用了 `_wrap_angle_deg`，但 `scenarios.py` 没有。

#### LOGIC-07: `ice_models.py` 文档与默认参数严重不符
**文件**: `ice_models.py:88-89`
```python
# 文档: "crushing_strength: 2.0 MPa (ISO 19906 range: 1-5 MPa)"
# 实际: crushing_strength_mpa: float = 0.0003  # 300 Pa，差 4 个数量级
```
**影响**: 文档严重误导，可能让人误用默认参数进行全尺度计算。

---

## 三、架构与设计缺陷

### ARCH-01: 风险公式在 5+ 处重复实现，无单一来源
三因子风险公式 `0.35*pos_risk + 0.35*ice_risk + 0.30*cvar` 出现在：
- `controllers.py:503` (IceAware)
- `controllers.py:712` (Supervisor)
- `nmpc_controller.py:441`
- `baseline_controllers.py:557` (DOBMPC)
- `risk_estimator.py:105-108`
- `baseline_controllers.py:685` (ADRC 用 `c*h*(1+v)` 又是不同公式)

每处的归一化常数略有不同。修改风险公式需要编辑所有位置。应提取为共享函数。

### ARCH-02: 角度单位（度/弧度）跨模块不一致
- `ice_schedule.py`: 度
- `ice_models.py`: 弧度
- `sensor_models.py`: 度
- `sim_loop.py`: 转换点（line 153: `math.radians()`）
- `scenarios.py`: 度

转换必须在调用点手动完成，是整个代码库最危险的不一致。

### ARCH-03: 双仿真路径（`simulator.py` vs `sim_loop.py`）
`Simulator.run()` 和 `run_simulation()` 是两个独立的仿真循环，特性集不同。`runner.py` 只用 `run_simulation`。`Simulator` 似乎是遗留代码。风险是一个路径的修改不会反映到另一个。

### ARCH-04: 推力器配置定义分散在两处
`ThrusterConfig` 部分定义在 `thruster.py`（`generic_dp_vessel`, `vessel_xuelong2`），部分仅在 `runner.py`（`generic_dp_power_limited`）。独立测试场景更困难。

### ARCH-05: `build_controller` 是 80 行 if/elif 链
**文件**: `runner.py:120`
违反开闭原则。应使用注册表模式（name→constructor 字典）。

### ARCH-06: `_CONTROLLER_CAPABILITIES` 在 `tables.py` 和 `runner.py` 之间隐式同步
如果新增控制器名到 `build_controller` 但忘记更新 `_CONTROLLER_CAPABILITIES`，观测器会静默不创建。

### ARCH-07: 全局可变状态 `_FAIL_FAST_ON_MISSING_DATA`
**文件**: `runner.py:619`
如果 `run_experiments` 在同一进程中被调用两次（如测试），第二次调用继承第一次的状态。应作为参数传递。

### ARCH-08: `simulation` 包依赖 `sci1` 包（反向依赖）
`simulator.py` 从 `sci1.sim_loop` 导入核心动力学函数。通用模块依赖特定模块，方向不合理。

---

## 四、基线控制器实现问题（影响论文公平对比）

| 控制器 | 问题 | 影响 |
|--------|------|------|
| **LQG** | 卡尔曼协方差公式错误 (BUG-01)；H=I 非 proper unobserved-state 模型 | 状态估计不可信 |
| **ADRC** | 混合 NED/体坐标系 (BUG-06) | 性能偏弱，夸大所提方法优势 |
| **DOBMPC** | 观测器从未使用 (dead code)；`reset()` 崩溃风险 (BUG-03) | 不是真正的"DOB-MPC" |
| **RobustMPC** | 风险公式不一致 (LOGIC-04) | 约束收紧程度不同 |
| **NMPC** | 航向代价180°极小 (BUG-02)；小角度代价 ≈ d⁴ 太弱 | 航向跟踪可能失败 |

**关键影响**: 如果论文使用这些基线进行对比实验，结果的公平性和可信度存疑。审稿人可能质疑基线实现是否正确，从而质疑整个实验结论。

---

## 五、实验设计与方法学审查（对标顶会前沿）

### 5.1 ✅ 项目已做好的方面
1. **分层安全架构**: HOCBF + 安全滤波器 + 模式监督器的多层设计符合当前安全控制前沿
2. **统计严谨性**: 配对 Wilcoxon 检验 + Cohen's dz 效应量 + Holm-Bonferroni 校正 + Bootstrap CI
3. **场景覆盖**: 44+ 场景覆盖冰况变化、推力器故障、传感器退化、安全降级
4. **可复现性**: 确定性种子、数据溯源清单、YAML 配置驱动
5. **消融实验设计**: 逐步移除组件验证各模块贡献

### 5.2 ⚠️ 需要完善之处

#### EXP-01: 代理尺度 (proxy-scale) 的物理意义不明确
- 默认冰强度 0.0003 MPa（真实值 1-5 MPa），产生的冰力约 1kN
- 这使得所有控制器的控制裕度非常大，"困难"场景可能实际上很容易
- **建议**: 明确声明这是算法验证而非物理验证；或增加一组归一化到真实冰力尺度的场景

#### EXP-02: 缺少与前沿方法的直接对比
当前基线（PID, SMC, LQG, ADRC）都是经典方法。顶会审稿人可能期望看到：
- **强化学习 DP 控制** (如 SAC/PPO-based DP, Zhao et al. IEEE JOE 2023)
- **自适应神经网络控制** (如 RBF-NN DP, Wang et al. Ocean Eng 2024)
- **分布式 MPC** (如 Li et al. Automatica 2023)
- **Safety-aware RL** (如 Constrained RL with CBF, Cheng et al. L4DC 2023)

#### EXP-03: 缺少计算复杂度分析
- NMPC 和安全滤波器的求解时间仅作为运行时指标记录
- 缺少与线性控制器（PID/SMC）的理论复杂度对比
- 缺少实时性可行性分析（如 DP 系统典型的 1-10 Hz 控制频率要求）

#### EXP-04: 数据驱动回放使用 mock 数据
- H 组场景使用合成 NetCDF 文件，非真实 Copernicus 数据
- `METHOD_HOCBF_LIMITATIONS.md` 已声明此限制
- **建议**: 在论文中明确说明；或提供一个与真实数据的校准对比实验

#### EXP-05: 观测器带宽/收敛性未分析
- `IceLoadObserver` 使用简单 EMA（一阶低通），带宽由 `alpha` 决定
- 未分析在不同冰况变化速率下观测器的跟踪能力
- 未与更先进的观测器（如滑模观测器、高增益观测器、UKF）对比

#### EXP-06: HOCBF 约束的前向不变性未严格证明
- `METHOD_HOCBF_LIMITATIONS.md` 声明不支持形式化安全证明
- 对于顶刊，审稿人可能期望至少提供 Lyapunov 稳定性分析或仿真验证的 CBF margin 统计

#### EXP-07: 多船/编队场景缺失
- 当前仅考虑单船 DP
- Arctic 编队航行是前沿方向（如 icebreaker-convoy DP）

---

## 六、代码质量问题

### 6.1 性能问题
1. **`metrics.py:272`**: `shortest_angle_diff_deg` 在 Python 列表推导中逐元素调用，应向量化
2. **`statistics.py:409`**: 9×10×9 = 810 次 Bootstrap 计算（每次 1000 重采样），CI 管线可能很慢
3. **`ice_schedule.py:180`**: `RandomWalkIce._generate_at` 线性扫描缓存，大 step 值时效率低

### 6.2 文档/注释问题
1. `ice_models.py` 文档与默认参数差 4 个数量级 (LOGIC-07)
2. `BenchmarkIceLoadModel` 文档说 "Default 500 N"，实际默认 100 N
3. `scenarios.py` 场景组文档只列到 G，实际有 H、I 组
4. `data_bridge.py` 文档说 "p10-p90"，实际用 mean±1σ (≈p16-p84)

### 6.3 死代码
1. `qp_solver.py` 整个模块未被使用（功能在 `safety_filter.py` 中重复）
2. `DOBMPCController._observer` 创建但从未调用
3. `units.py` 中 `XUELONG2_LIKE_*` 常量未被引用
4. `runner.py::save_controller_capability_matrix` 函数未被使用

---

## 七、修复优先级建议

### 立即修复（影响实验结果可信度）
1. **BUG-01**: LQG 卡尔曼协方差公式 → 修复 F 矩阵
2. **BUG-02**: NMPC 航向代价 → 去掉平方
3. **BUG-06**: ADRC 坐标系混合 → 统一坐标变换
4. **LOGIC-04**: RobustMPC 风险公式 → 使用标准化公式
5. **ARCH-01**: 风险公式 → 提取为共享函数

### 短期修复（影响代码健壮性）
6. **BUG-03**: DOBMPC `__init__` 初始化 `_has_observer_estimate`
7. **BUG-04**: 航向单位检测 → 显式单位参数
8. **BUG-05**: SciPy 回退 → 标记 `infeasible=True`
9. **BUG-09**: 日志级别 → `is not None` 检查
10. **LOGIC-01**: CVaR 采样 → 对控制力也随机化

### 中期改进（提升论文质量）
11. 增加与 RL/NN 基线的对比
12. 添加计算复杂度分析
13. 观测器带宽分析
14. 代理尺度的物理意义说明
15. HOCBF margin 统计验证

---

## 八、总结

项目整体架构设计合理，分层安全控制思路清晰，实验覆盖面广，统计方法规范。但在**基线控制器实现**上存在多个影响公平对比的 bug（LQG 公式错误、ADRC 坐标系混合、DOBMPC 观测器未使用），**风险公式**在多处不一致，以及**角度单位**跨模块传递容易出错。这些问题如果不修复，可能在审稿中被质疑实验结论的可靠性。

**最高优先级**: 修复 5 个基线控制器 bug + 统一风险公式，确保对比实验的公平性。
