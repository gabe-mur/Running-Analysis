"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import stage_calendar_seeds as _implementation

sys.modules[__name__] = _implementation
