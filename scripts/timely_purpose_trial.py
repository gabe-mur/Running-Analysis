"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import timely_purpose_trial as _implementation

sys.modules[__name__] = _implementation
