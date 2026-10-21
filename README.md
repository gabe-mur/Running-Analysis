# Running Coach

A local-first app for understanding your running and planning what comes next.
It combines watch recordings, aerobic performance, training load, recovery and
your goals into a rolling 21-day schedule. Your history stays on your computer.
No Running Coach account or subscription is required.

## Mac app

The packaged **Running Coach.app** includes its own runtime. You do not need
Python, Terminal or a development environment. Version 0.2.0 supports **Apple
Silicon Macs running macOS 14 or later**; it is not an Intel or Windows build.

1. Unzip the application ZIP supplied by the author and move **Running Coach**
   to Applications.
2. Open it and follow Setup. Start by importing your runs or connecting
   Intervals.icu. Without recordings, Setup provides a measured starting run.
3. Confirm your personal details, heart-rate zones and running preferences.
   A GPS-capable watch with recorded heart rate is needed for full analysis;
   a chest-strap heart-rate monitor is recommended.

The app is ad-hoc signed, not Apple-notarized. macOS may require approval under
**System Settings → Privacy & Security → Open Anyway** after the first open
attempt. Only approve an app obtained from a source you trust. No security
settings need to be disabled globally.

### Your data and updates

Settings, recordings and the database live separately from the application at
`~/Library/Application Support/Running Coach/`. Use **Running Coach → Open Data
Folder** to find or back up this folder. Quit the app before copying a backup.
Replacing the app does not replace your history. Keep a backup before updating;
an older app may not support a database updated by a newer version.

The shared application ZIP contains no personal history, API keys, research
databases or downloaded map tiles. The author's separate **Running Coach Clean**
demo uses its own empty data folder and cannot change the main app's history.

The green GitHub **Code → Download ZIP** button downloads source code, not the
ready-built Mac application.

## Using it

- **Dashboard:** current training status, aerobic efficiency and load in context.
- **Progress:** comparable aerobic pace, observed runs, goal distance and pace
  outlook, VO₂ estimates, volume and quality-session share. Forecast ranges are
  estimates, not guarantees.
- **Run Analysis:** individual recordings, splits, heart rate, terrain, weather,
  workout structure and editable activity details.
- **Weekly Plan:** the coming seven days of a rolling 21-day schedule, with
  workout prescriptions, strength sessions, completion tracking and rest-day
  constraints. A full fresh search can take a couple of minutes.
- **Settings:** guided Setup, heart-rate sources, running preferences, training
  goals, weather privacy and the optional Intervals.icu connection.

Upload `.fit`, `.fit.gz`, `.tcx` or supported activity ZIPs. GPX without heart
rate cannot supply the same fitness evidence as a complete recording. Sync and
upload analyze new evidence first; the plan refreshes when Weekly Plan is next
opened or refreshed. Other views then use the updated saved plan.

Runners without recordings can report their established volume, frequency and
regular long run, then record one normal easy run before the full planner starts.
New runners have a separate measured run/walk start and a gradual handoff to the
normal planner, rather than invented training history.

## Run from source

For development or platforms without a packaged build, install Python 3.11+
and run `python3 start.py`, or use the included Mac/Windows launcher. The first
launch installs dependencies and opens a browser; keep that launcher's terminal
open. Source-mode data lives in the checkout, **not** the desktop app-data folder.
Use `--port 8001`, `--no-browser` or `--dev` as needed.

### Build the Mac app

Build on an Apple Silicon Mac with Xcode Command Line Tools installed:

```sh
python3 scripts/prepare_mac_runtime.py --output build/mac-runtime
build/mac-runtime/env/bin/python scripts/build_mac_app.py --runtime-notices build/mac-runtime/python --output build/mac-release
```

The build produces an app, shareable ZIP, checksum and dependency notices. It
copies only explicit runtime resources, retains implementation sources needed
for cache validation, and bundles a CA certificate store for verified HTTPS.
It never copies the development database or credentials into the app.
The runtime preparation checks a pinned publisher SHA-256 before extraction.
The build also rejects native dependencies requiring macOS newer than 14.

## Method and privacy

The planner models completed work, expected workout load and recovery, rather
than treating every workout name as an interchangeable dose. It evaluates
alternative schedules against the same objective and retains reproducible
decision inputs for regression checks. Aerobic comparisons account for recorded
effort, terrain, weather and data quality; they are not laboratory measurements.

Optional Intervals.icu sync contacts that service using locally stored
credentials. Optional weather uses Open-Meteo and an approximate, privacy-blurred
location, not the raw route. Explicit terrain downloads request broad public map
tiles; normal elevation analysis uses local files. There are no analytics or
advertising trackers.

See [Privacy](docs/privacy.md), [Modeling methodology](docs/modeling_methodology.md),
[Application architecture](docs/application_architecture.md) and the anonymous
[training-policy inferences](docs/training_policy_inferences.md). Original
training-plan research materials are private and are not needed at runtime.

## Rights and third-party software

Application code is copyright Gabe Murray; all rights reserved. No open-source
license is granted at this time. See [COPYRIGHT](COPYRIGHT). Bundled dependencies
retain their own licenses, available in **Running Coach → Third-Party Licenses**.

## Important

This is a personal training tool, not a medical device or medical clearance.
Fitness, VO₂, load and forecast values are estimates. The app cannot infer every
effect of sleep, illness, injury, medication, stress or nutrition from a watch.
Use your judgment, seek qualified advice when appropriate, and stop for chest
pain, unusual shortness of breath, faintness or worsening pain.
