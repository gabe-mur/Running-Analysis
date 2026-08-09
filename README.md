# Local Running Coach

A private, local app that reads your own watch files and tells you what your
running is actually doing — whether your aerobic fitness is moving, how hard
you have really been training, and what to run next. Everything is computed on
your computer by fixed, inspectable rules. No account, no subscription, no LLM,
and nothing leaves the machine unless you switch on weather lookups.

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

## Download it

On this page, click the green **Code** button near the top, then
**Download ZIP**. Unzip it, and you will have a folder called
`Running-Analysis-main`.

Put that folder somewhere you intend to keep — your Documents folder is fine,
your Downloads folder is not. The app stores your runs, settings, and analysis
*inside* this folder, so deleting it deletes your history.

Renaming or moving the folder later is safe; the next launch takes an extra
moment while the app repairs its own paths, then carries on with everything
intact.

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
2. Open **Settings → Setup** and confirm your heart-rate numbers. Until you do,
   the app is using defaults that may not describe you, and it will say so.
   The same walkthrough is where you set an optional race goal, which changes
   what the plan prioritises.
3. On **Run analysis**, open any run and use **Edit run details** to correct
   the workout type or flag illness or injury. Runs tagged that way still count
   toward your training load but carry less weight in the fitness trend.
4. Tell the app **how you feel** at the top of the **Weekly Plan**. That, and
   uploading a run, are what cause the plan to rebuild — you never have to
   generate it by hand.

### Prefer a terminal?

`python3 start.py` does the same thing. `--port 8001` moves it off a busy
port (it also finds a free one by itself), `--no-browser` suppresses the
browser, and `--dev` adds the test dependencies.

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

Then delete the old folder once you have confirmed the new one works.

## The five screens

- **Dashboard** — one training status (*building, maintaining, rebuilding,
  recovering, strained, underloaded*, or *not enough data*) with the rules that
  produced it shown in order, plus separate read-outs for aerobic efficiency,
  durability, training capacity, and recent form.
- **Progress** — your pace at a fixed heart rate over time, adjusted for
  weather, hills, and how far into the run you were, with the uncertainty
  drawn rather than hidden. Also an estimated VO₂ max, a verdict on your
  easy/moderate/hard balance, and progress toward your goal.
- **Run analysis** — every run, and for each one: splits, heart-rate zones,
  cadence and stride length, stops, weather, drift, a reconstructed workout
  structure for intervals, and what to run next.
- **Weekly plan** — an explainable seven days built from recent and sustained
  load, workout difficulty, spacing, long-run history, how you feel, the
  forecast, and an optional validated 5K, 10K, half-marathon, or marathon goal.
- **Settings**, with a guided **Setup** inside it for the handful of numbers
  everything else depends on.

## What makes it different

It separates fitness, session difficulty, and accumulated load, so a slow long
run is not mistaken for lost fitness.

It says how much it knows. Every figure carries a confidence level, runs that
cannot support a fair comparison are excluded and say why, and where a
calculation rests on an assumption — an estimated maximum heart rate, a
population constant — it names it.

It will not invent an answer. When the evidence is thin it says so instead of
producing a confident number, and it never turns a free-text note into a
diagnosis.

## Privacy

Run files, settings, health tags, the database, and any reports stay on your
computer and are excluded from Git. There is no account and no server.

Historical weather and forecasts are the only features that reach the internet,
and both are opt-in. When enabled, the app sends a date or planned time and a
rounded, randomly offset approximate location to Open-Meteo. Your route is
never sent, and the blur radius is yours to set in Setup.

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
