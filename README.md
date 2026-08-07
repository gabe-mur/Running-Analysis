# Garmin running analysis

This is a local, incremental analysis of Garmin running history. It imports raw
TCX trackpoints, removes traffic-stop time, caches anonymized historical
weather, and estimates pace at a standardized cardiovascular effort.

The primary metric is **prior-anchored standardized pace at 145 bpm**. Grade uses
the Minetti metabolic-cost transform where altitude is available. Heat starts
with a literature prior and is updated only by calendar-local hot/cool matches,
with the personal-data weight and evidence level shown explicitly. The HR/speed
slope is estimated only from run-centered within-run differences.

See [the modeling methodology](docs/modeling_methodology.md) for the published
foundations, raw-window design, and validation rules. Athlete-specific surveys
and validation reports belong in the ignored `docs/local/` directory.

## Current project structure

```text
config.example.yaml         sanitized configuration template
run_overrides.example.csv   sanitized activity-metadata template
start.py                    cross-platform setup and application launcher
src/run_analysis/
  __main__.py               `python -m run_analysis` entry point
  cli.py                    import and inspection commands
  config.py                 YAML configuration
  tcx.py                    namespace-tolerant TCX parser
  models.py                 typed parser records
  db.py                     versioned SQLite schema
  importer.py               incremental import and activity deduplication
  movement.py               interval moving/stopped classification
  segmentation.py           distance segments and smoothed grade metrics
  model_windows.py          overlapping and sensitivity windows from raw intervals
  movement_diagnostics.py   exact removed-time audit for representative runs
  physiology.py             published grade and wet-bulb transforms
  geo.py                    distance and route-bearing calculations
  processing.py             incremental run/quarter-mile metric persistence
  weather.py                cached Open-Meteo retrieval and interpolation
  workload.py               prior-only training load
  training_load.py          time-in-zone session and rolling load
  modeling.py               model utilities and primary scoring entry point
  objective_modeling.py     prior-anchored per-run aerobic-efficiency score
  analytics.py              adjustable time-varying fitness interpretation
  progress.py               fitness/load/consistency/intensity API analysis
  run_feedback.py           raw mile splits and deterministic run feedback
  fitness_state.py          compact state consumed by coaching rules
  recommendation.py         pure, inspectable Python recommendation engine
  recommendation_service.py recommendation persistence/orchestration
  dashboard.py              three-job dashboard composition
  settings_service.py       validated local settings overlay and recalculation
  metadata_service.py       override CSV and SQLite metadata persistence
  project_setup.py          idempotent local file/directory initialization
  web/                      FastAPI routes and static ES-module application
  reporting.py              self-contained local HTML dashboard
  audit.py                  reproducible dataset-quality audit
tests/                      automated unit, integration, and stress tests
scripts/                    developer and isolated stress-test utilities
docs/                       shareable architecture, methodology, and privacy docs
TCX/                        ignored local Garmin exports
uploads/                    ignored local uploads
data/                       ignored SQLite DB and weather cache
output/                     ignored generated reports and diagnostics
```

The SQLite schema already reserves the requested downstream tables while raw
activities, laps, trackpoints, source provenance, and manual overrides have
normalized tables.

## Setup and commands

Python 3.11 or newer is required.

For normal use, setup and startup are one command from the repository root:

```bash
python3 start.py
```

On Windows, use `py start.py`. The launcher creates `.venv`, installs the app,
creates the ignored local configuration and runtime folders when missing, and
starts the web interface at `http://127.0.0.1:8000`. It is idempotent: later
runs preserve local settings and data and simply start the app. Use
`python3 start.py --setup-only` to prepare everything without launching.

The first-run configuration deliberately leaves historical weather and planned
forecast retrieval disabled. Enable either separately in Settings only after
approving the described Open-Meteo transmission.

### Development and individual pipeline commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m run_analysis init
.venv/bin/python -m run_analysis import
.venv/bin/python -m run_analysis process
.venv/bin/python -m run_analysis audit
.venv/bin/python -m run_analysis weather
.venv/bin/python -m run_analysis model
.venv/bin/python -m run_analysis report
.venv/bin/python -m run_analysis all
.venv/bin/python -m run_analysis inspect '<activity-id>'
.venv/bin/python -m run_analysis serve
.venv/bin/pytest
```

Import is incremental: unchanged files are skipped, changed files are replaced
transactionally, and duplicate exports are retained as source provenance while
sharing one canonical activity. Generated state is written under `data/`; TCX
source files are never modified.

Personal activity data, local configuration, overrides, reports, and caches are
excluded from Git. See [the privacy guide](docs/privacy.md) before publishing or
sharing the repository.

`all` runs the incremental pipeline in order. Unchanged TCX files and cached
weather days are reused. If fewer than ten activities contain defensible model
windows, model fitting is reported as deferred and the rest of the pipeline
continues normally.

When explicitly enabled, `weather` sends a coordinate rounded to two decimals,
then deterministically jittered within a configurable 2 km radius, plus required dates to Open-Meteo.
Responses are cached as raw JSON and per-day SQLite records. With explicit user
approval, a no-GPS activity may be explicitly mapped to a confirmed nearby
GPS-recorded run within the configurable 14-day limit. Its anonymized centroid
is labeled as an estimated location throughout the audit data; unmapped
activities are not assigned a guessed location.

The model reads raw trackpoint intervals rather than quarter-mile aggregates.
Quarter-mile segments remain available for diagnostics and the report. Missing
altitude does not exclude an otherwise valid HR/GPS window; that window is
retained without inventing a grade adjustment. Current counts are written to
`output/model_results.json` whenever the model command runs.

Weak GPS also does not discard reliable Garmin cumulative-distance and HR
windows. At or above 80% distance-weighted GPS coverage, the observation is
treated normally. Below that threshold, device distance supplies pace while an
additional uncertainty penalty grows continuously to its configured maximum at
zero GPS. These observations cannot calibrate shared model parameters, and the
fitness trend downweights them automatically through inverse-variance weighting.

The web plan is regenerated as a rolling today-through-six-days schedule. Its
context strip shows the seven completed calendar days ending yesterday. A run
uploaded for today replaces today's prescription with a completed-run card, and
the remaining six days are recalculated from the updated distance, HR load,
workout type, and recovery spacing.

For a no-GPS activity, Edit Run accepts an optional five-digit US ZIP code. The
ZIP is sent to Open-Meteo's geocoding endpoint only when saved, stored locally,
then converted to a rounded and privacy-jittered coordinate for historical
weather. The result remains labeled as an estimated location and receives the
configured uncertainty penalty.

The primary run score uses shared, run-centered HR and time-into-run effects,
then robustly combines all usable overlapping 120-second windows into that
day's performance offset at the configured 145-bpm/20-minute reference. Total
run duration is not a predictor. A strict single-window score remains visible
only as validation.
`output/moving_time_diagnostic.json` records every second removed from eight
representative runs, including exact timestamps, distance, speed, and flags.
## Local Running Coach

Start the local API and application shell with:

```bash
.venv/bin/python -m run_analysis serve
```

Then open `http://127.0.0.1:8000`. The OpenAPI contract is available at
`http://127.0.0.1:8000/docs`.

The app provides Dashboard, Progress, Runs, Next Run, and Settings views.
Upload Run accepts one or several TCX files by file picker or drag/drop and runs
the incremental import → processing → weather → model pipeline. A single upload
opens its run-feedback page.

Recommendations are generated in `src/run_analysis/recommendation.py`, not in the
frontend. The module accepts a serialized `FitnessState`, evaluates named rules
in a fixed order, and returns a complete decision trace. Distance, duration,
intensity-weighted load, recent load, long-run durability, health input, and
data quality stay separate from standardized pace.

The Dashboard distinguishes short-term aerobic efficiency, 90-day training
capacity, and illness/recovery context. Progress charts default to the trailing
year rather than allowing an off year or old lifetime data to dominate the
current headline. Runs tagged illness/recovery still count toward completed
load but are excluded from the aerobic-efficiency aggregate.

Next Run requires a planned date and time. Recovery intervals and rolling load
are projected to that instant. Planned-run forecast retrieval is a separate,
disabled-by-default setting because it sends the privacy-jittered recent-route
centroid and future timestamp to Open-Meteo; historical-weather permission does
not silently enable it.

Garmin VO2-max and race-predictor history is not present in TCX. Dated Garmin
snapshots can be entered on Progress and are stored as a separate corroborating
signal rather than being silently mixed with the app's pace-at-HR estimate.

Settings edited in the app are written to ignored `config.local.yaml`, which is
deep-merged over the ignored local `config.yaml`. The tracked
`config.example.yaml` remains a sanitized starting point, while analysis
commands and the web app continue to see the same local values.
