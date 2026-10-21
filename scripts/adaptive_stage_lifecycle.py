"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import adaptive_stage_lifecycle as _implementation

sys.modules[__name__] = _implementation
