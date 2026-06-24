"""Pytest configuration — adds src/ to sys.path for package discovery."""
import sys
from pathlib import Path

_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
