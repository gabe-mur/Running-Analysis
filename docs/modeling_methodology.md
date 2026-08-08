# Modeling methodology: published references, moving fitness

## Why the model changed

Fitness is not a fixed nuisance variable. It changes over the same chronology as
season, weather, training volume, and route choice. A global regression on one
runner's history therefore cannot reliably identify a personal heat, hill, wind,
or cardiac-drift coefficient: it can assign genuine fitness change to a seasonal
covariate, or do the reverse.

The primary score no longer selects environmental weights by whichever candidate
best predicts this athlete's historical windows. Each run instead produces its
own pace-at-145 estimate. Adjustable 14-, 28-, 42-, 56-, and 90-day curves are
descriptive robust summaries of those per-run estimates, not fitted latent
fitness states.

## Observation unit and eligibility

The primary score uses overlapping 120-second windows every 60 moving seconds,
built directly from raw trackpoints. Quarter-mile segments remain available
only for familiar run diagnostics. A separate strict benchmark retains one
stable two-minute window near minute 20 as a validation metric, not as the
primary fitness estimate.

Windows are excluded when they:

- occur in the first five minutes, where HR kinetics lag workload;
- occur after 60 minutes;
- fall outside the prespecified 128–166 bpm submaximal range;
- have inadequate HR coverage or lack both usable GPS and Garmin device distance; or
- contain excessive stopped/transition time.

GPS and altitude are independent data channels. Missing altitude no longer
excludes an otherwise usable HR/GPS window. It is retained with no grade
adjustment and identified as grade-unavailable.

GPS coverage is evaluated by distance within each modeling window, not merely
by counting trackpoints. Coverage of at least the configured 80% threshold is
ordinary full evidence. Below that threshold, reliable Garmin cumulative device
distance may supply pace. The added 95% uncertainty grows linearly from zero at
80% coverage to 30 seconds per mile at zero coverage; an estimated weather
location adds another 10 seconds per mile in quadrature. These runs do not
calibrate the shared HR or heat parameters. Their larger uncertainty gives them
less influence in the trend through the same inverse-variance weighting used
for every observation, avoiding a second arbitrary quality multiplier.

When explicitly authorized, historical weather for a zero-GPS run may use an
explicitly confirmed mapping to a nearby GPS activity within 14 days. The
rounded, salted, privacy-jittered centroid, source activity, and temporal offset
are retained in the local audit record. Unmapped runs do not receive a guessed
location. This supports metro-area weather only; it does not invent a route,
bearing, or wind correction.

## Prior-anchored environmental transformations

### Grade

Where at least 70% of a window has stable smoothed elevation coverage, observed
speed is converted to level-equivalent energetic speed with the measured
fifth-order running-cost curve from Minetti et al. (2002). Cost is integrated
over 60 m micrograde windows. Missing grade is not imputed as level terrain.

### Heat and humidity

Hourly temperature and dew point are converted to estimated shade WBGT using
Stull's wet-bulb approximation and `0.7 × wet bulb + 0.3 × air temperature`.
The correction starts with a conservative normal prior of 0.2% performance loss
per degree Celsius WBGT above the 55°F / 45°F-dew-point reference, with a 0.2%
standard deviation. This is a population marathon prior, not a claim about this
runner.

Personal evidence updates the prior only through pairs of runs no more than 56
days apart and at least 3°C apart in WBGT exposure. The comparison uses
grade-corrected raw pace at 145 bpm. Pair estimates are time-weighted, with
leave-one-run-out jackknife uncertainty so hundreds of non-independent pairs do
not masquerade as hundreds of independent athletes. Normal-normal updating
produces the posterior heat coefficient. The report shows the literature prior,
personal likelihood, posterior, personal-data weight, matched runs, and a
Low/Moderate/High evidence label.

Temperature and dew-point contributions are separated with a two-order Shapley
decomposition of the WBGT adjustment. Their contributions sum back to the total
heat adjustment. Direct solar radiation is unavailable, so this is not measured
outdoor WBGT.

### Wind

Route-relative headwind, tailwind, and crosswind remain reported. The primary
score does not apply a wind correction. Pugh established the aerodynamic
relationship, but an objective individual correction also needs defensible drag
area, body mass, air density, and street-level exposure. NYC building effects
make the airport/grid wind especially uncertain. No personal wind coefficient is
learned from the fitness trend.

## HR and time-into-run calibration

There is no universal conversion from 10 bpm to seconds per mile. The personal
HR and time-into-run effects are estimated jointly from differences *within the
same run*:

1. Correct each eligible window with the prior transformations above.
2. Subtract that run's mean HR, mean time, and mean corrected speed from its windows.
3. Fit one robust dataset-level HR/time relationship to those run-centered differences.

Because every run is centered before fitting, its date-specific fitness level
cancels. Improving from one month to another cannot set either shared effect.
Each run's windows are normalized to 145 bpm and minute 20. The remaining
run-specific offset is that day's performance observation. The model then
robustly aggregates every usable window in **speed space**. Continuous weights
favor windows near 145 bpm and the configured 20-moving-minute reference and
downweight stops, within-window HR change, and acceleration/deceleration. A
Huber step further limits isolated GPS/pace outliers. Overlap is accounted for
when calculating effective sample size, which is capped so a long run cannot
manufacture certainty merely by providing correlated windows. Runs far outside
the submaximal HR range are not extrapolated.

The prediction question is: given all usable continuous-running evidence in
this run, what speed does it support at 145 bpm, 20 moving minutes, and the
reference grade/weather conditions? Pace conversion happens only for display.
The strict single-window benchmark is shown alongside it so disagreement can be
inspected.

Minute 20 is a statistical comparison point, not a physiological threshold. It
is late enough to reduce warm-up transients, early enough to limit late-run
fatigue, and directly interpolated by most historical runs. Limited extrapolation
widens uncertainty; activities more than five minutes short of supporting the
reference are unscored. Total duration is never used as an adjustment, and
cardiac drift remains a separate diagnostic.

## Interpretation

The output is a standardized aerobic-efficiency estimate, not VO2 max and not a
causal decomposition. A lower pace-at-145 value is faster. Every scored run has
an auditable chain:

`Raw pace @145 → environmental adjustment → standardized pace @145`

The environmental adjustment is decomposed into grade, temperature, dew point,
wind, and drift. Each part carries its own evidence label and personal-data
weight. Wind and drift currently display as unavailable/zero rather than
silently borrowing a fitted coefficient.

The selectable fitness horizon changes interpretation: 14 days responds quickly
but is noisy; 90 days is stable but slow. The current level, comparison with the
preceding equal-length window, probability of improvement, personal-history
percentile, best sustained period, and evidence density all update together.

## Fitness is a multi-signal interpretation

The trailing pace-at-145 estimate is not treated as a synonym for total
fitness. The application reports three distinct dimensions:

- **Aerobic efficiency:** reference-condition pace at 145 bpm, with short and
  longer-window comparisons.
- **Current condition:** illness/recovery tags, recent response, recovery
  spacing, and rolling training load.
- **Training capacity:** recent distance, frequency, moving time, and longest
  run. More capacity is meaningful progress, but does not mathematically force
  faster pace at a fixed HR.

The headline uses the current 28 days and current versus prior 90-day periods.
Progress charts show at most the trailing 365 days. Older activities remain in
run history rather than silently defining the current baseline. Sparse
comparison periods lower confidence.

Runs tagged `illness`, `illness_recovery`, `injury_affected`, or
`other_abnormal` retain distance, duration, and load but do not vote in the
aerobic-efficiency aggregate. Their observations remain visible, so the app
does not erase or falsify those sessions.

Garmin VO2-max and race-predictor history are not fields in the supplied TCX
files. Dated Garmin snapshots are stored as a separate external signal. They
are never relabeled as this application's own estimate or invisibly merged with
pace-at-HR evidence.

The optional local VO2-max cross-check uses the published George steady-state
jogging equation with profile weight/sex, target HR, and standardized running
speed. Because the source protocol was a controlled, level treadmill jog, its
use on corrected outdoor observations is labeled an adaptation with low
confidence. The displayed ±6.1 mL/kg/min is approximately 1.96 times the
published 3.1 protocol SEE and does not quantify every outdoor adaptation
error. A Jackson age/sex/BMI/activity estimate is shown only as a broad
demographic baseline (published SEE 5.7; approximately ±11.2 at 95%). Neither
is Garmin's proprietary Firstbeat estimate, and the George trend is not an
independent vote because it inherits the same pace/HR signal.

## Planned-run timing and forecast context

Recommendations are evaluated at a user-selected future timestamp. Recovery
spacing and trailing 7/14/28-day load are recalculated at that instant. When
the user separately enables planned-run forecasts, the application may request
hourly Open-Meteo conditions using a rounded, locally salted, privacy-jittered
recent route centroid plus the planned timestamp. This is disabled by default;
historical-weather approval is not treated as forecast approval. Forecast
failure does not prevent local rules from running.
Environmental thresholds are inspectable coaching guardrails, not claims of
individualized medical safety.

## Training load and coaching interpretation

Session difficulty is deliberately not folded into standardized pace. The app
retains distance and moving duration, and calculates an Edwards-style sum of
minutes in recorded Z1–Z5 multiplied by weights 1–5. Above-Z5 time is capped at
the Z5 weight; below-Z1 time contributes easy duration but zero Edwards points.
If less than half of moving time has known HR, the numeric zone load is missing
rather than imputed. Rolling 7/14/28-day distance, time, hard minutes, and load
are computed independently.

The 7-day-to-prior ratio compares current load with the weekly mean of the
preceding 28 days. It is a contextual coaching flag, not a validated injury-risk
threshold. Likewise, the configurable 110% long-run progression factor is a soft
user-requested guardrail, not a safety law.

Recent intensity is shown with the athlete's configured five zones and grouped
for coaching as easy (Z1+Z2), moderate (Z3), and hard (Z4+Z5). The engine flags
moderate-intensity leakage but does not enforce a universal 80/20 quota. The
observational endurance literature uses several incompatible zone systems, so
the interface keeps this athlete's definitions explicit.

## Race-goal guardrails

The optional 5K, 10K, half-marathon, and marathon goals are inspectable planner
inputs, not promises of a finish time. A goal is accepted only after 10 usable,
normal-health running performances exist. Each is projected to the selected
distance with the Riegel 1.06 relationship; the median of the fastest three
projections supplies a robust current-performance guardrail. Marathon
projections receive a conservative 10-minute penalty because recreational
marathon predictions commonly overstate performance even when shorter-distance
predictions are well calibrated. This is deliberately a goal-validation rule,
not a replacement for a race result or a physiological model.

Developing goals require at least 9 weeks for 5K, 12 for 10K, 10 for a half
marathon, and 18 for a marathon. These reflect the scale of established public
beginner/conservative plans. When all 10 runs already support the requested
pace, a shorter race-specific minimum is allowed. Dates more than a year away,
dates too soon, implausible absolute paces, and paces too aggressive for the
evidence and available time are rejected with an earliest date or supported
pace. The allowed improvement margin is a deliberately conservative planning
guardrail of 0.25% per available week, capped at 8%; it is not a promised
adaptation rate. A marathon inside 26 weeks also requires a recent six-mile run.

An accepted goal modifies the transparent workout scores and quality-session
rotation. It never overrides pain, illness, acute-load, recent-workout, weather,
or recovery-spacing guardrails. Goal pace appears as context inside relevant
quality workouts and as the race-day target; controlled effort still takes
precedence over a split.

## References

- Minetti et al. (2002). [Energy cost of walking and running at extreme uphill
  and downhill slopes](https://pubmed.ncbi.nlm.nih.gov/12183501/).
- Pugh (1970). [Oxygen intake in track and treadmill running with observations
  on air resistance](https://pubmed.ncbi.nlm.nih.gov/5532903/).
- Ely et al. (2007). [Impact of weather on marathon-running
  performance](https://pubmed.ncbi.nlm.nih.gov/17473775/).
- Weiss et al. (2022). [Effects of weather parameters on endurance running
  performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC8677617/).
- Stull (2011). [Wet-Bulb Temperature from Relative Humidity and Air
  Temperature](https://doi.org/10.1175/JAMC-D-11-0143.1).
- Seiler & Kjerland (2006). [Quantifying training intensity distribution in
  elite endurance athletes](https://pubmed.ncbi.nlm.nih.gov/16430681/).
- Edwards-style zone weighting as implemented in later training-load research:
  [Absolute and Relative Training Load and Its Relation to Fatigue in
  Football](https://pmc.ncbi.nlm.nih.gov/articles/PMC5459919/).
- ACSM (1998). [Recommended quantity and quality of exercise for developing and
  maintaining cardiorespiratory fitness](https://pubmed.ncbi.nlm.nih.gov/9624661/).
- George et al. (1993), equation and original validation summarized in
  [Submaximal Treadmill Exercise Test to Predict VO2max in Fit Adults](https://www.tandfonline.com/doi/full/10.1080/10913670701294047).
- Jackson et al. (1990). [Prediction of functional aerobic capacity without
  exercise testing](https://pubmed.ncbi.nlm.nih.gov/2287267/).
- Firstbeat. [Automated Fitness Level (VO2max) Estimation with Heart Rate and
  Speed Data](https://assets.firstbeat.com/firstbeat/uploads/2015/10/white_paper_VO2max_11-11-2014.pdf).
- Riegel (1981). [Athletic records and human endurance](https://pubmed.ncbi.nlm.nih.gov/7272663/).
- Vickers & Vertosick (2016). [An empirical study of race times in recreational
  endurance runners](https://doi.org/10.1186/s13102-016-0052-y).
- NHS. [Couch to 5K running plan](https://www.nhs.uk/better-health/get-active/get-running-with-couch-to-5k/).
- New York Road Runners. [5K low-mileage plan](https://webassets.nyrr.org/nyrrsitecoreblob/nyrr/pdf/training-guides/5k-training-plan_low-2020-5k-pr-series_2-rd5.pdf),
  [10K beginner plan](https://webassets.nyrr.org/nyrrsitecoreblob/nyrr/pdf/training-guides/10k_training_plan_beginner_rd41.pdf),
  [half-marathon conservative plan](https://webassets.nyrr.org/nyrrsitecoreblob/nyrr/pdf/training-guides/hm-training-plan_conservative-rd1.pdf), and
  [marathon conservative plan](https://webassets.nyrr.org/nyrrsitecoreblob/nyrr/pdf/training-guides/2024/nyrr-marathon-conservative-training-plan_rd5.pdf).
