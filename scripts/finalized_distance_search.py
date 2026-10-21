"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import finalized_distance_search as _implementation

sys.modules[__name__] = _implementation
