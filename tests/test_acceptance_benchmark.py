from contextlib import nullcontext
import json
import sys

import pytest

from scripts import benchmark_candidate_acceptance as benchmark


@pytest.mark.parametrize("order", ["off,on", "on,off"])
def test_benchmark_uses_same_inputs_and_supports_both_orders(monkeypatch, tmp_path, order):
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--order", order, "--output-json", str(output)])
    calls = []

    def build(daily_states, *, expected_target_projector):
        calls.append((daily_states, expected_target_projector([], 21)))
        return None

    monkeypatch.setattr(benchmark.planner, "build_weekly_schedule", build)
    monkeypatch.setattr(benchmark.projection, "build_weekly_schedule", build)
    monkeypatch.setattr(benchmark, "synthetic_history", lambda start: (None, []))
    monkeypatch.setattr(benchmark, "replay_policy", lambda mode: nullcontext())
    monkeypatch.setattr(benchmark, "calendar_neighbor_policy", lambda **kw: nullcontext([{}]))
    monkeypatch.setattr(benchmark, "sessions", lambda plan: [])
    monkeypatch.setattr(benchmark.projection, "simulate_adherence", lambda *a, **kw:
        benchmark.projection.build_weekly_schedule([], expected_target_projector=lambda candidate, n: ((1, 2),) * n))
    benchmark.main()
    result = json.loads(output.read_text())
    assert result["sessions_equal"]
    assert [x["acceptance_enabled"] for x in result["variants"]] == [x == "on" for x in order.split(",")]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert benchmark.projection.build_weekly_schedule is build
