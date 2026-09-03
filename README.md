# Local Running Coach

A private, local app for analyzing watch data and planning your training. It
tracks aerobic progress, training load, recovery, and workout history, then uses
that information to recommend what to run next. Analysis happens on your
computer using inspectable rules. No account or subscription is required, and
your run data remains local.

## What you need

- **Python 3.11 or newer.** If you do not have it, get it from
  [python.org/downloads](https://www.python.org/downloads/). On Windows, select
  **"Add python.exe to PATH"** in the installer.
- **Your runs**, as `.tcx` or `.fit` files (`.fit.gz` works too). Garmin
  Connect and Strava both export these.
- Several runs with heart-rate data are needed for a useful fitness trend. The
  app can import and display a smaller history, but will report limited evidence
  until enough comparable runs are available.

## Download it

On this page, click the green **Code** button near the top, then
**Download ZIP**. Unzip it, and you will have a folder called
`Running-Analysis-main`.

Move the folder to a permanent location before using the app. Runs, settings,
and analysis data are stored inside it, so deleting the folder also deletes that
local history.

You can rename or move the folder later. The next launch may take a little
longer while stored paths are updated.

## Open the app

Double-click the launcher for your computer:

| | |
|---|---|
| **Mac** | `1. Open Running Coach - Mac.command` |
| **Windows** | `1. Open Running Coach - Windows.bat` |

Use the launcher for your operating system. The first launch installs the local
environment and may take a minute or two; later launches are faster. The app
opens in your browser automatically. Keep the terminal window open while using
it, because closing that window stops the local server.

### If macOS refuses to open it

macOS blocks files downloaded from the internet until you approve them once.
**Right-click** (or Control-click) the `.command` file, choose **Open**, then
click **Open** in the dialog. You only have to do this the first time.

### Then

1. Drop your run files anywhere on the page, or use **Upload Runs**.
2. Open **Settings → Setup** and confirm your heart-rate numbers. Until you do,
   the app uses defaults that may not match you. The same walkthrough lets you
   set an optional race goal, which changes the plan's priorities.
3. On **Run analysis**, open any run and use **Edit run details** to correct
   the workout type or flag illness or injury. Runs tagged that way still count
   toward your training load but carry less weight in the fitness trend.
4. Record **how you feel** at the top of the **Weekly Plan**. The plan updates
   when this changes or when you upload a run.

### Prefer a terminal?

Run `python3 start.py`. Use `--port 8001` to request a specific port,
`--no-browser` to prevent automatic browser launch, or `--dev` to install test
dependencies.

### Updating to a newer version

Downloading a new ZIP gives you an **empty** app — your history stays behind in
the old folder. To bring it across, copy these from the old folder into the new
one before opening it:

| | |
|---|---|
| `data/` | your database, weather cache, and privacy salt |
| `uploads/` | the original run files you imported |
| `config.local.yaml` | your heart rates, zones, goal, and preferences |
| `run_overrides.csv` | workout-type and health corrections, if present |

Keep the old folder as a backup until you have confirmed the new one works.

## The five screens

- **Dashboard** — a current training status (*building, maintaining,
  rebuilding, recovering, strained, underloaded*, or *not enough data*) with
  the supporting signals, plus aerobic efficiency, durability, training
  capacity, and recent form.
- **Progress** — your pace at a fixed heart rate over time, adjusted for
  weather, hills, and how far into the run you were, with the uncertainty
  shown on the chart. It also includes an estimated VO₂ max, training-intensity
  distribution, and progress toward your goal.
- **Run analysis** — details for each run, including splits, heart-rate zones,
  cadence and stride length, stops, weather, drift, reconstructed interval
  structure, and the next-run recommendation.
- **Weekly plan** — the first seven days of an explainable rolling 21-day plan,
  built from recent and sustained load, workout difficulty, recovery, long-run
  history, how you feel, the forecast, and an optional validated 5K, 10K,
  half-marathon, or marathon goal.
- **Settings** — preferences and a guided **Setup** for the profile information
  used by the analysis and planner.

## How the algorithm works

The app models fitness, training load, and recovery as related but distinct
signals.

1. **Interpret each run.** The importer reconstructs moving time, distance,
   heart-rate zones, terrain, weather, stops, and workout structure. Explicit
   corrections and matched prescriptions take priority. Otherwise a shared
   fallback classifier identifies easy and long runs. Runs tagged for illness,
   injury, or fitness-model exclusion remain in the training history.
2. **Estimate aerobic progress.** Comparable runs estimate pace at the user's
   selected heart rate and reference point in the run. Weather, grade, cardiac
   drift, and data quality affect the estimate and its uncertainty. A robust,
   evidence-weighted trend is calculated from eligible runs using both adjacent
   equal-length periods and the trajectory within the selected period. The UI
   distinguishes likely direction from clear 95% evidence; coaching decisions
   retain the stronger standard. Activities that do not contribute to the trend
   can still appear on the chart as context with their influence clearly labeled.
3. **Measure load and recovery.** Distance, duration, and recorded intensity
   determine session cost. Longer-term training density and short-term recovery
   change continuously with elapsed time. A workout's actual or expected load,
   rather than its name alone, determines its recovery cost. Sleep is not
   modeled because the app does not currently record it.
4. **Build a rolling plan.** The planner compares 21-day schedules containing
   different dates, times, workout types, and distances. It considers target
   mileage, accumulated load, recovery, useful easy volume, long- and
   quality-session timing, weather, protected rest days, and continuity with
   the previous plan. It does not require a fixed number of runs per calendar
   week. The interface displays the first seven days of the plan.
5. **Close the loop.** Only the work actually completed becomes training
   evidence. The app reevaluates the plan as runs are completed, missed, or
   changed, and when rest-day or health information changes. Actual distance,
   duration, and intensity determine how completed work affects later sessions.

With no usable running history, a separate calibration plan begins with a
10–30-minute conversational Zone 2 run or run/walk. Regular mileage, long runs,
and quality sessions are introduced after enough training history is available.
An optional race goal can shape progression, workout emphasis, and tapering;
general-fitness mode also supports gradual progression.

For formulas, evidence thresholds, and limitations, see
[Modeling methodology](docs/modeling_methodology.md).

## Privacy

Run files, settings, health tags, the database, and any reports stay on your
computer and are excluded from Git. There is no account or remote application
server.

Historical weather and forecasts are the only features that use an internet
connection, and both are optional. When enabled, the app sends Open-Meteo a date
or planned time and a rounded, randomly offset approximate location. It does not
send the recorded route. Location-privacy settings are available in Setup.

See [Privacy](docs/privacy.md), [Modeling methodology](docs/modeling_methodology.md),
and [Application architecture](docs/application_architecture.md) for details.

## Disclaimer

This is a personal training-analysis tool, not a medical device and not a
substitute for professional advice.

The fitness trend, VO₂ max estimate, training status, and prescribed workouts
are estimates derived from recorded data. The app cannot account for sleep,
stress, nutrition, medical history, pain, illness, or injury unless the relevant
information is entered, and it does not provide a diagnosis or medical
clearance.

Use your own judgment and seek qualified professional advice when appropriate.
Consult a doctor before starting or substantially changing a training program,
and stop and seek medical attention for chest pain, unusual shortness of breath,
faintness, or pain that worsens while running. The software is provided as-is,
without warranty.
