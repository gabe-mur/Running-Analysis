"""The chosen stack must run when research scripts are not importable."""
from pathlib import Path
import subprocess
import sys


def test_installed_stack_has_no_research_driver_import_dependency(tmp_path):
    code = '''
import sys
from run_analysis.adaptive.adaptive_candidate_pipeline import compile_athlete
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.factory import create_service
assert not any(n == 'scripts' or n.startswith('scripts.') for n in sys.modules)
'''
    subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, check=True)


def test_research_compatibility_alias_shares_identity_and_hooks():
    from scripts import continuous_search_adapter as old
    from run_analysis.adaptive import continuous_search_adapter as new
    from scripts import full_score_refinement as old_refiner
    from run_analysis.adaptive import full_score_refinement as new_refiner
    assert old is new and old_refiner is new_refiner
    with old_refiner.full_score_refinement_policy(budget=17):
        assert new_refiner.REFINEMENT_BUDGET.get()['budget']==17
    assert new_refiner.REFINEMENT_BUDGET.get() is None


def test_packaged_sources_do_not_import_research_modules():
    import ast
    from run_analysis import adaptive
    for path in Path(adaptive.__file__).parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not node.module or not node.module.startswith('scripts'), path
            elif isinstance(node, ast.Import):
                assert not any(n.name=='scripts' or n.name.startswith('scripts.') for n in node.names), path
