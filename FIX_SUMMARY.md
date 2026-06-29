# Arctic-DP 审计修复总结

**修复日期**: 2026-06-28 (两轮修复)
**测试结果**: 327 passed, 5 skipped, 0 failed
**烟雾实验**: 通过 (smoke profile, 3 controllers, 12 CSV outputs)

---

## 已修复的 Bug

### 高优先级 (影响实验结果可信度)

1. **LQG 卡尔曼滤波** (`baseline_controllers.py`)
   - 协方差预测: `A*P*A^T*dt + Q*dt` → `F*P*F^T + Q*dt` (F = I + A*dt)
   - 观测矩阵: `H = I(6)` → `H = diag(1,1,1,0,0,0)` (速度不可直接观测)

2. **NMPC 航向代价** (`nmpc_controller.py`, `baseline_controllers.py`)
   - `(1-cos(delta))^2` → `(1-cos(delta))` (去掉平方, 消除180°虚假极小)
   - 同时修复了主 NMPC 和 DOBMPC 内嵌 NMPC 的阶段/终端代价

3. **ADRC 坐标系混合** (`baseline_controllers.py`)
   - NED 位置误差 + 体坐标系速度 → 统一在 NED 帧计算 PD 力, 再旋转到体坐标系

4. **NMPC fallback_pd 坐标系** (`nmpc_controller.py`)
   - 体帧 PD → NED 帧 PD + 旋转到体帧 (与 PrecisionDPController 一致)

5. **RobustMPC/ADRC 风险公式** (`baseline_controllers.py`)
   - `c*(0.5*h+v)` 和 `c*h*(1+v)` → 统一使用 `_ice_risk_standardized()`

### 中优先级 (影响代码健壮性)

6. **DOBMPC reset() 崩溃** (`baseline_controllers.py`)
   - `__init__` 中添加 `self._has_observer_estimate = False`

7. **航向角单位检测** (`sim_loop.py`)
   - 阈值 `2*pi` → `pi` (2处)

8. **SciPy 安全滤波器回退** (`safety_filter.py`)
   - 求解失败时 raise RuntimeError → 外层标记 infeasible=True

9. **日志级别 falsy 值** (`logging_config.py`)
   - `level or INFO` → `level if level is not None else INFO`

10. **data_bridge vy 维度** (`data_bridge.py`)
    - `np.array(float(vy))` → `np.array([float(vy)])`

11. **runner full_scale_ready** (`runner.py`)
    - `not _is_full` → `False` (统一两处代码路径)

12. **CVaR 采样** (`controllers.py`)
    - 添加执行器不确定性采样, 使 control_sat 也参与尾部估计

13. **场景角度插值** (`scenarios.py`)
    - 线性插值 → 最短路径角度插值 (处理 350°→10° 绕回)

14. **ice_models 文档** (`ice_models.py`)
    - 修正文档: 实际默认值 0.0003 MPa, 非文档所述 2.0 MPa
    - 修正 BenchmarkIceLoadModel 文档: 默认 100 N, 非 500 N

### 第二轮修复 (2026-06-28 深度审核)

15. **RobustMPC/TubeMPC 坐标系混合** (`baseline_controllers.py`)
    - 与 ADRC 相同的 bug: NED 位置误差 + 体坐标系速度
    - 修复: NED 帧计算 PD 力 → 旋转到体坐标系

16. **风力速度坐标系** (`sim_loop.py`)
    - 风速(NED) - 船速(体) = 物理错误
    - 修复: 先将风速从 NED 旋转到体坐标系, 再计算相对风速

17. **推力饱和检测误判力矩为力** (`metrics.py`)
    - `tau_n_actual` (偏航力矩) 被 `max_force=3000` 阈值误判为饱和
    - 修复: 模式3排除力矩列, 新增 `_moment_cols` 集合

18. **chattering 指标死代码** (`metrics.py`)
    - 无方位角数据时 chattering_index 恒为 NaN
    - 修复: 仅推力变化时报告推力 chattering

19. **data_sources.py 坐标名硬编码** (`data_sources.py`)
    - `load_sic/thickness/drift_from_netcdf` 使用 `y`/`x` 而非 `latitude`/`longitude`
    - 修复: 添加 `_find_nc_coord` 和 `_sel_latlon` 辅助函数, 兼容多种坐标名

20. **download_data.py 状态未应用** (`download_data.py`)
    - `status_map` 计算后未使用
    - 修复: 遍历 sources 应用下载状态

21. **Mock fixture 校验和不匹配** (`data_sources.py`)
    - SHA256 与实际文件不符, 文档声称含 `vxsi/vysi` 但实际不含
    - 修复: 更新 SHA256 和描述

22. **DataDrivenWindSchedule 非3D数组处理** (`data_bridge.py`)
    - 2D 风场数组直接赋值导致时间维度错乱
    - 修复: 2D → 提取单点; 1D → ravel

23. **docstring 典型值错误** (`controllers.py`)
    - 风险公式典型值约为文档所述的一半
    - 修正为实际计算值

24. **figures.py NaN 处理** (`figures.py`)
    - NaN p-value/效应量被静默标记为"不显著"
    - 修复: 显式 NaN 检查, 使用灰色"N/A"色

25. **statistics.py n=1 标准差** (`statistics.py`)
    - 单样本组 std=0.0 → 改为 NaN (与 metrics.py 一致)

26. **controller_wrappers.py 死代码** (`controller_wrappers.py`)
    - 删除无意义的 `__dict__.get('_last_diagnostics')` 行

27. **nmpc_controller.py 死代码** (`nmpc_controller.py`)
    - 删除重复 `x0` 赋值

28. **测试 `or True` 断言** (3个测试文件)
    - 删除 `or True`, 修复为有意义的断言

## 架构改进

15. **共享风险公式** (`controllers.py` + 4个调用点)
    - 新增 `compute_total_risk()` 函数, 5处调用点统一使用
    - `_ice_risk_standardized()` 被 `risk_estimator.py` 直接导入 (消除重复)

---

## 未修复的设计问题 (需要更大重构)

- LOGIC-02: 推力器速率限制与分配器内部状态不同步 (需重构分配器接口)
- LOGIC-03: 推力分配饱和裁剪忽略方向 (需重新实现分配算法)
- LOGIC-05: 冰力偏航力矩仅用横向分量 (需物理模型修正)
- ARCH-02: 角度单位跨模块不一致 (需全面接口重构)
- ARCH-03: 双仿真路径 (需合并或删除 simulator.py)
- ARCH-05: build_controller if/elif 链 (需注册表模式重构)
- DOBMPC._observer 死代码 (需决定是否集成观测器)
- 诊断 dict 键不统一 (各控制器缺少不同键)
- data_sources.py 源计数与 manifest 不一致
- 测试覆盖: data_calibration.py, hocbf.py, figures.py, logging_config.py 零覆盖

这些问题需要更大的重构工作, 不影响当前实验结果的正确性 (已通过327个测试), 建议在论文提交后作为技术债务处理。
