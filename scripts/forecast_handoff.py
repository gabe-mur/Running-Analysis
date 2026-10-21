"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import forecast_handoff as _implementation

sys.modules[__name__] = _implementation
