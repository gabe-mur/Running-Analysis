# Local Running Coach

A private, local app for understanding Garmin running history and planning the
next seven days. It analyzes aerobic efficiency, training load, durability,
individual workouts, weather, and recovery context without an LLM or
subscription.

## What you need

- **Python 3.11 or newer.** If you do not have it, get it from
  [python.org/downloads](https://www.python.org/downloads/) — the large
  download button on that page is the right one. On Windows, tick
  **"Add python.exe to PATH"** on the first screen of the installer.
- **Your runs**, as `.tcx` or `.fit` files (`.fit.gz` works too). Garmin
  Connect and Strava both export these.
- About **10 runs with heart-rate data** before the fitness trend appears. The
  app imports and displays fewer than that; it just will not claim a trend it
  cannot support.

## Open the app

Double-click the launcher for your computer:

| | |
|---|---|
| **Mac** | `1. Open Running Coach - Mac.command` |
| **Windows** | `1. Open Running Coach - Windows.bat` |

Both are numbered `1.` because they are the same step — pick the one for your
computer and ignore the other.

The first launch takes a minute or two while it sets itself up. After that it
starts in a few seconds. **A browser window opens by itself** — you do not need
to type an address. Leave the small black window open while you use the app;
closing it stops the app.

### If macOS refuses to open it

macOS blocks files downloaded from the internet until you approve them once.
**Right-click** (or Control-click) the `.command` file, choose **Open**, then
click **Open** in the dialog. You only have to do this the first time.

### Then

1. Drop your run files anywhere on the page, or use **Upload Runs**.
2. Go to **Settings → Setup** and confirm your heart-rate numbers. Until you
   do, the app is using defaults that may not describe you, and it will say so.
3. Check the **Weekly Plan**.

### Prefer a terminal?

`python3 start.py` does the same thing. `--port 8001` moves it off a busy
port (it also finds a free one by itself), `--no-browser` suppresses the
browser, and `--dev` adds the test dependencies.

## What it shows

- **Dashboard:** aerobic efficiency, durability, training capacity, recent
  form, load, and the next workout.
- **Progress:** standardized pace-at-heart-rate trends with uncertainty.
- **Run analysis:** pace, HR, splits/laps, zones, stops, cadence and stride,
  weather, workout scoring, and editable metadata.
- **Weekly plan:** an explainable seven-day schedule based on recent and sustained
  load, workout difficulty, spacing, long-run history, health, weather, and an
  optional validated 5K, 10K, half-marathon, or marathon goal.

The app keeps fitness, session difficulty, and accumulated load separate. A
slow long run is not automatically treated as worse fitness than a short fast
run.

## Privacy

Run files, settings, health tags, databases, and reports stay on your computer
and are ignored by Git. Historical weather and planned forecasts are separate,
disabled-by-default options. If enabled, the app sends dates or planned times
and rounded, privacy-jittered route centroids to Open-Meteo—not raw routes.

See [Privacy](docs/privacy.md), [Modeling methodology](docs/modeling_methodology.md),
and [Application architecture](docs/application_architecture.md) for details.

## Disclaimer

This is a personal training-analysis tool, not a medical device and not a
substitute for professional advice.

Everything it produces — the fitness trend, the VO₂ max figure, the training
status, and every prescribed workout — is an estimate derived from your own
watch files by fixed arithmetic rules. It has no view of your sleep, stress,
nutrition, medical history, or how you actually feel, and it cannot recognise
pain, illness, or injury. It does not diagnose anything and it is not clearance
to exercise.

Use your own judgement, and a qualified professional's, over anything shown
here. Consult a doctor before starting or changing a training programme, and
stop and seek medical attention for chest pain, unusual shortness of breath,
faintness, or pain that worsens as you run. Following any suggestion in this
app is your decision and your responsibility; it is provided as-is, with no
warranty of any kind.
