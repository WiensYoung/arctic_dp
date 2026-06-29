"""SCI1 实验日志配置。

提供统一的日志格式，替代项目中分散的 print 语句。

使用:
    from arctic_quasi_dp.sci1.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("实验开始")
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取模块专用的 logger。

    Args:
        name: logger 名称 (通常为 __name__)
        level: 日志级别 (None=使用默认 INFO)

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level if level is not None else logging.INFO)
        logger.propagate = False

    return logger
