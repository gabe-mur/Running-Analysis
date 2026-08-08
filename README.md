# Local Running Coach

A private, local app for understanding Garmin running history and planning the
next seven days. It analyzes aerobic efficiency, training load, durability,
individual workouts, weather, and recovery context without an LLM or
subscription.

## What you need

- Python 3.11 or newer.
- Garmin activities exported as **TCX files**. FIT and GPX are not supported.
- At least **10 usable runs** with heart-rate and movement data for the primary
  fitness model. The app can import and display fewer runs, but fitness modeling
  remains deferred until enough evidence exists.

## Start the app

From this folder, run:

```bash
python3 start.py
```

On Windows, use `py start.py`.

The first launch installs everything and prepares the private local files.
Later launches use the same command. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000), then:

1. Update your heart-rate and health settings.
2. Upload your TCX files.
3. Review activity types and health tags, and set an optional goal in the settings. Then generate the plan.

Press `Ctrl+C` to stop. If port 8000 is busy, run
`python3 start.py --port 8001`.

## What it shows

- **Dashboard:** aerobic efficiency, durability, training capacity, recent
  form, load, and the next workout.
- **Progress:** standardized pace-at-heart-rate trends with uncertainty.
- **Runs:** pace, HR, splits/laps, zones, stops, weather, workout scoring, and
  editable metadata.
- **Next seven days:** an explainable schedule based on recent and sustained
  load, workout difficulty, spacing, long-run history, health, weather, and an
  optional validated 5K, 10K, half-marathon, or marathon goal.

The app keeps fitness, session difficulty, and accumulated load separate. A
slow long run is not automatically treated as worse fitness than a short fast
run.

## Privacy

TCX files, settings, health tags, databases, and reports stay on your computer
and are ignored by Git. Historical weather and planned forecasts are separate,
disabled-by-default options. If enabled, the app sends dates or planned times
and rounded, privacy-jittered route centroids to Open-Meteo—not raw routes.

See [Privacy](docs/privacy.md), [Modeling methodology](docs/modeling_methodology.md),
and [Application architecture](docs/application_architecture.md) for details.

This app provides training analysis, not medical diagnosis or clearance to
exercise.
