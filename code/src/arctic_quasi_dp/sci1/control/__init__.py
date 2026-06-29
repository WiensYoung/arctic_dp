"""可插拔安全滤波器模块。

提供 CVaR-adaptive Soft-HOCBF-QP safety filter 方法层。
不破坏现有 controllers.py, 通过 wrapper 模式接入。
"""

from .safety_filter import SafetyFilterResult, SoftHOCBFSafetyFilter
from .hocbf import HOCBFParams, compute_hocbf_constraint
from .risk_estimator import ProxyCVaRRiskEstimator, RiskEstimate
from .controller_wrappers import (
    SafetyFilteredController,
    make_filtered_controller,
)

__all__ = [
    "SafetyFilterResult",
    "SoftHOCBFSafetyFilter",
    "HOCBFParams",
    "compute_hocbf_constraint",
    "ProxyCVaRRiskEstimator",
    "RiskEstimate",
    "SafetyFilteredController",
    "make_filtered_controller",
]
