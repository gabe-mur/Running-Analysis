"""Reference material stays local; only downstream policy is public."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from run_analysis.race_preparation import PROFILES, reference_review


ROOT = Path(__file__).resolve().parents[1]


def test_reference_material_and_identifying_tooling_are_ignored():
    paths = [
        'data/reference/local.sqlite',
        'output/research/sources/original.pdf',
        'tmp/pdfs/research/extraction.json',
        'docs/local/provenance.md',
        'scripts/training_plan_reference.py',
        'scripts/collect_popular_plan_sources.py',
        'scripts/import_public_training_sessions.py',
        'scripts/import_example_short_race_reference.py',
        'scripts/audit_race_preparation_reference.py',
        'scripts/audit_short_race_closeout.py',
        'tests/test_training_plan_reference.py',
        'tests/test_example_short_race_reference.py',
        'tests/test_race_reference_closeout.py',
        'docs/short_race_reference_review.md',
        'docs/recovery_scheduler_reconciliation.md',
        'docs/adaptive_scheduler_plan.md',
    ]
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
        cwd=ROOT, input='\n'.join(paths) + '\n', text=True, capture_output=True)
    if result.returncode == 128:
        pytest.skip('Git checkout required for ignore-policy validation')
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == set(paths)
    # The standalone policy and its behavioral checks must remain publishable.
    public = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
        cwd=ROOT, input='docs/training_policy_inferences.md\ntests/test_short_race_preparation.py\n',
        text=True, capture_output=True)
    assert public.returncode == 1 and not public.stdout, public.stderr


def test_race_metadata_exposes_only_downstream_inferences():
    public = reference_review()
    assert public['reference_database'] is None
    assert public['reference_review'].startswith('docs/training_policy_inferences.md#')
    assert not public['exact_doses_corpus_fitted']
    assert not public['ultra_guidance_allowed']
    metadata = json.dumps([asdict(profile) for profile in PROFILES.values()] + [public])
    assert 'https://' not in metadata and 'http://' not in metadata
    assert '.sqlite' not in metadata
    document = ROOT / 'docs/training_policy_inferences.md'
    assert document.is_file()
    for profile in PROFILES.values():
        assert profile.sources
        assert all(source.startswith(document.relative_to(ROOT).as_posix() + '#')
                   for source in profile.sources)


def test_privacy_cleanup_keeps_numerical_race_scenarios_unchanged():
    expected = {'5k': (3.1, 10., 7), '10k': (5.5, 14., 7),
                'half_marathon': (10., 25., 10), 'marathon': (20., 40., 21)}
    assert {key: (profile.long_miles, profile.volume_per_7, profile.taper_days)
            for key, profile in PROFILES.items()} == expected
