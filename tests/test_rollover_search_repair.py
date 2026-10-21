from datetime import date
from types import SimpleNamespace

import pytest

from scripts.check_rollover_search_repair import check_empty_rollover


def case(day, **overrides):
    settings=('policy','scales','scenario','exchange','timing','easy_duration_policy',
              'delivery_contract','soft_horizon_taper','continuation_guard_days','volume_linear_coefficient',
              'purpose_continuation','purpose_continuation_phase')
    state=SimpleNamespace(context=SimpleNamespace(decision=day,**{f:'unchanged' for f in settings}),history=(),phase_history=(),
        current=None,responses=(),restrictions=(),contracts=(),resting_hr=60,maximum_hr=190,
        references_available_at='unchanged',stage_key='unchanged',**overrides)
    return dict(state=state,handoffs={'easy':'unchanged'})


def test_empty_rollover_checks_actual_new_plan_not_archived_opening():
    before=case(date(2026,10,12)); after=case(date(2026,10,13))
    assert check_empty_rollover(before,after,(('2026-10-13','easy'),))['opening_is_rest']
    with pytest.raises(ValueError,match='execute/upload'):
        check_empty_rollover(before,after,(('2026-10-12','easy'),))


def test_changed_evidence_or_skipped_day_cannot_be_called_clock_only():
    before=case(date(2026,10,12)); after=case(date(2026,10,13))
    after['state'].history=('new run',)
    with pytest.raises(ValueError,match='retain physical history'):
        check_empty_rollover(before,after,())
    with pytest.raises(ValueError,match='consecutive'):
        check_empty_rollover(before,case(date(2026,10,14)),())


def test_changed_objective_cannot_be_attributed_to_clock_only():
    before=case(date(2026,10,12)); after=case(date(2026,10,13))
    after['state'].context.exchange='new weights'
    with pytest.raises(ValueError,match='objective settings'):
        check_empty_rollover(before,after,())
