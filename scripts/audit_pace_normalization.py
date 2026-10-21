"""Read-only audit; does not refit the live model or start the planner."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from run_analysis.config import load_config
from run_analysis.model_windows import load_model_window_sets
from run_analysis.objective_modeling import _shared_parameter_eligible
from run_analysis.pace_validation import validate_pace_normalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", action="store_true", help="Rebuild derived scores in an in-memory database copy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config("config.yaml")
    connection = sqlite3.connect("file:data/run_analysis.sqlite?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    if args.sandbox:
        from run_analysis.objective_modeling import fit_published_reference_model
        from run_analysis.progress import build_progress
        from run_analysis.progress_wip import build_progress_wip
        memory = sqlite3.connect(":memory:")
        memory.row_factory = sqlite3.Row
        connection.backup(memory)
        at = datetime.now(timezone.utc)
        def summary():
            progress = build_progress(memory, 28, as_of=at, config=config)
            view = build_progress_wip(memory, config, Path.cwd(), 28, as_of=at, include_exploratory=False)
            return {"pace_28d_min_mile": progress.current_pace.minutes_per_mile if progress.current_pace else None,
                    "pace_28d_uncertainty": progress.uncertainty_95_min_mile,
                    "confidence": progress.fitness_confidence.value,
                    "scored_runs": memory.execute("SELECT count(*) FROM model_runs WHERE model_name='standardized_pace_at_target_hr'").fetchone()[0],
                    "trend_runs_28d": sum(p.included_in_trend for p in progress.series),
                    "vo2_14d": view["performance"]["vo2"]}
        before = summary()
        with TemporaryDirectory(prefix="pace-validation-") as temporary:
            metadata = fit_published_reference_model(memory, config, Path(temporary)/"model.json")
        report = {"validation": metadata["time_blocked_cross_validation"], "before": before, "after": summary(),
                  "source": "read-only live database; in-memory sandbox rebuild; no planner invoked"}
        memory.close()
    else:
        sets, _ = load_model_window_sets(connection, config, (300,))
        rows = [row for row in sets[300] if _shared_parameter_eligible(row)]
        report = validate_pace_normalization(rows)
    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    connection.close()


if __name__ == "__main__":
    main()
