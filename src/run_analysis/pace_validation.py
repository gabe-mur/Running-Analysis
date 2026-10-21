"""Run-blocked checks of within-run pace normalization, not fitness forecasts.

Fit shared slopes on earlier runs only. On each later run, use its first half
to establish that day's speed level and predict its second half. Thus a faster
runner/year cannot count as evidence for a correction. No overlapping windows
are used by the caller. Weather attribution needs a different experiment.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .objective_modeling import _within_run_hr_time_slopes, pace_to_speed_mps, speed_to_pace_min_mile


def validate_pace_normalization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["activity_id"])].append(row)
    runs = sorted(
        (sorted(run, key=lambda r: r["moving_minutes_into_run"]) for run in grouped.values() if len(run) >= 4),
        key=lambda run: (run[0]["start_time_utc"], run[0]["activity_id"]),
    )
    errors = defaultdict(list)
    fold_details = []
    for start_fraction, end_fraction in ((0.5, 0.65), (0.65, 0.8), (0.8, 1.0)):
        start, end = int(len(runs) * start_fraction), int(len(runs) * end_fraction)
        train = [r for run in runs[:start] for r in run]
        if len(train) < 20:
            continue
        speeds = [pace_to_speed_mps(r["moving_pace_min_mile"]) for r in train]
        try:
            hr, time, _ = _within_run_hr_time_slopes(train, speeds)
        except ValueError:
            # A diagnostic must not prevent observed-data scoring when the
            # athlete lacks independent HR/time variation.
            continue
        # HR-only robust regression, with one centered level per training run.
        x, y = [], []
        for run in runs[:start]:
            h = np.array([r["average_hr_bpm"] for r in run], dtype=float)
            s = np.array([pace_to_speed_mps(r["moving_pace_min_mile"]) for r in run])
            x.extend(h - h.mean())
            y.extend(s - s.mean())
        x, y = np.array(x), np.array(y)
        hr_only = float(x @ y / (x @ x)) if x @ x else 0.0
        for _ in range(8):
            residual = y - hr_only * x
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            weights = np.minimum(1.0, 1.345 * max(scale, 1e-9) / np.maximum(np.abs(residual), 1e-12))
            hr_only = float((weights * x) @ y / ((weights * x) @ x))
        grade_hr, grade_time, _ = _within_run_hr_time_slopes(
            train, [s * float(r.get("grade_energy_ratio") or 1) for s, r in zip(speeds, train)]
        )
        models = {"none": (0.0, 0.0, False), "hr": (hr_only, 0.0, False),
                  "hr_time": (hr, time, False), "hr_time_grade": (grade_hr, grade_time, True)}
        fold_details.append({"training_runs": start, "test_runs": end - start,
                             "hr_only_slope": hr_only, "joint_hr_slope": hr, "time_slope": time})
        for run in runs[start:end]:
            anchor, test = run[:len(run)//2], run[len(run)//2:]
            for name, (h, t, grade) in models.items():
                def offset(r):
                    speed = pace_to_speed_mps(r["moving_pace_min_mile"])
                    ratio = float(r.get("grade_energy_ratio") or 1) if grade else 1.0
                    return speed * ratio - h * r["average_hr_bpm"] - t * r["moving_minutes_into_run"]
                level = float(np.median([offset(r) for r in anchor]))
                run_errors = []
                for r in test:
                    ratio = float(r.get("grade_energy_ratio") or 1) if grade else 1.0
                    prediction = (level + h * r["average_hr_bpm"] + t * r["moving_minutes_into_run"]) / ratio
                    run_errors.append(abs(speed_to_pace_min_mile(prediction) - r["moving_pace_min_mile"]) * 60)
                errors[name].append(float(np.mean(run_errors)))
    if not errors:
        return {"status": "insufficient_evidence", "eligible_runs": len(runs)}
    comparison = {}
    for candidate, baseline in (("hr", "none"), ("hr_time", "hr"), ("hr_time_grade", "hr_time")):
        delta = np.array(errors[candidate]) - np.array(errors[baseline])
        rng = np.random.default_rng(20261013)
        bootstrap = np.mean(rng.choice(delta, size=(2000, len(delta)), replace=True), axis=1)
        comparison[candidate] = {"versus": baseline, "mean_error_change_seconds": float(delta.mean()),
                                 "run_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    return {"status": "evaluated", "eligible_runs": len(runs), "held_out_runs": len(errors["none"]),
            "mean_absolute_error_seconds_per_mile": {k: float(np.mean(v)) for k, v in errors.items()},
            "comparisons": comparison, "folds": fold_details,
            "scope": "Earlier-run calibration; first-half anchored second-half prediction. Not a causal physiological validation or a between-run fitness forecast."}
