"""Compatibility entry point; implementation lives in the installed package."""
import sys
from run_analysis.adaptive import new_model_daily_harness as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
