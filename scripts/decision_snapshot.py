"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import decision_snapshot as _implementation

sys.modules[__name__] = _implementation
