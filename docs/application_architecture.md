# Local running coach: backend map and application plan

## Architectural decision

Keep `run_analysis` as the analytical package. Add a FastAPI application that
calls it through service modules and serves a small static ES-module frontend.
The browser renders typed JSON, handles navigation/upload forms, and collects
manual inputs. It does not decide whether a run was easy, whether load is high,
or what workout comes next.

This avoids a Node build pipeline for a single-user local application and does
not require rewriting the TCX/weather/model backend.

```text
TCX upload
   │
   ▼
FastAPI route ──► pipeline service ──► existing importer/process/weather/model
   │                                      │
   │                                      ▼
   │                               SQLite + weather cache
   ▼                                      │
typed response models ◄── query/state/feedback/recommendation services
   │
   ▼
static local frontend (Dashboard / Progress / Runs / Next Run / Settings)
```

## Existing backend inventory

Runtime corpus counts are intentionally omitted from tracked documentation.
They are athlete-specific and can be reproduced locally through the audit and
model commands.

### Existing database/output map

| Existing source | Useful fields | Application job |
|---|---|---|
| `activities` | date, distance, elapsed time, summary HR, GPS/HR/elevation/cadence quality, source | Progress, feedback, upload quality |
| `laps` | recorded time/distance/HR/intensity | Feedback |
| `trackpoints` | raw time, GPS, altitude, distance, HR, cadence, speed | Moving time, mile splits, model windows |
| `activity_metrics` | moving/device/elapsed time and pace, stops, moving HR, zones, eligibility, prior 7/28-day load, standardized result | All three jobs |
| `activity_weather` / `weather_cache` | run-time weather, route-relative wind, cached raw response | Progress, feedback |
| `model_runs` | raw pace @145, atomic adjustments, standardized pace, uncertainty, per-adjustment evidence | Progress, feedback |
| `model_metadata` | HR calibration, heat prior/personal likelihood/posterior, window diagnostics | Progress/method transparency |
| `run_overrides` | inclusion, workout type, illness, notes | Feedback, recommendation |
| `run_analysis.analytics` | adjustable 14/28/42/56/90-day fitness summaries | Progress/dashboard |
| `run_analysis.importer` | incremental import and duplicate detection | Upload |
| `run_analysis.processing` | moving-time classification, zones, metrics | Upload/feedback |
| `run_analysis.weather` | privacy-jittered Open-Meteo retrieval and cache | Upload |
| `run_analysis.objective_modeling` | standardized pace @145 and evidence chain | Progress/feedback |

### What already answers each product job

**Progress Analysis** already has independent per-run raw/standardized pace at
145, uncertainty, atomic environmental contributions, adjustable robust trends,
weather evidence, HR zones, and leakage-safe prior 7/28-day mileage/time.

**Run Feedback** already has moving vs elapsed/device time, traffic-stop
diagnostics, zones, weather, data quality, standardized score, and first/second
half inputs. It lacks mile splits, deterministic assessment language, contextual
comparison selection, and load/difficulty interpretation.

**Run Recommendation** already has prior mileage/time and days since the prior
run/hard run. It lacks a compact current fitness state, intensity-weighted load,
long-run/quality history, health status, fatigue/performance flags, transparent
rules, and persisted recommendation inputs/results.

## Distance and load policy

The existing standardized score is not fully load-aware. Distance currently
affects minimum eligibility, number of model windows, and uncertainty; prior
load is stored but is not used by the primary score. This can let later windows
from a long run look like a fitness regression relative to a short run.

The application will use the following explicit separation:

1. **Comparable fitness observation:** robustly combine all usable overlapping
   windows in speed space. HR/time relevance and transition stability control
   observation weight; effective sample size is capped for overlap. The strict
   fixed-time window is validation only. Do not add an invented seconds-per-mile
   fatigue correction.
2. **Session difficulty:** calculate an inspectable, Edwards-style zone load
   from moving minutes (`Z1×1 + Z2×2 + Z3×3 + Z4×4 + Z5×5`), plus distance,
   duration, hard minutes, and long-run status.
3. **Prior fatigue context:** compute current 7/14/28-day mileage, time, zone
   load, hard minutes, and the 7-day load relative to the prior 28-day weekly
   norm.
4. **Fitness-trend evidence:** illness/injury activities are excluded when
   configured. High-load, long-run-fatigue, low-quality, and noncomparable
   sessions remain visible but receive lower trend confidence rather than an
   arbitrary favorable pace correction.
5. **Coaching decisions:** distance, session load, load ratio, long-run history,
   intensity leakage, response, consistency, and health status directly drive
   the Python recommendation rules.

Thus a fast two-mile run and a slow eight-mile run remain two different
observations: the comparable early/mid-run physiology informs fitness; the full
distance/intensity/durability informs feedback and the next-run decision.

## Missing backend calculations

- approximately one-mile splits built from raw movement intervals;
- session zone load and hard/moderate/easy minutes;
- current 7/14/28/30-day load, longest run, quality count, consistency, gaps;
- comparable-window fitness evidence flags and load-confounded anomaly flags;
- interpretable drift validity classification;
- similar-run selection by workout type, HR, duration, and weather;
- deterministic run assessment and feedback rules;
- compact `FitnessState` contract;
- deterministic recommendation engine and workout templates;
- expanded health/workout metadata and current health-status persistence;
- upload orchestration with stage results;
- editable settings persistence/validation.

## API contracts and routes

All routes are under `/api`; `/` and unknown non-API routes serve the local app.

| Method/path | Contract/purpose |
|---|---|
| `GET /api/health` | database/pipeline readiness |
| `GET /api/dashboard?window_days=28` | fitness, last-run feedback, recommendation cards |
| `GET /api/progress?window_days=28` | chart series, period comparison, load/intensity summaries |
| `GET /api/runs` | sortable/filterable run summaries |
| `GET /api/runs/{id}` | detailed run feedback, splits, zones, context, audit chain |
| `PATCH /api/runs/{id}/metadata` | workout type, health tag, notes, model inclusion |
| `POST /api/uploads` | one/many TCX files; returns per-stage and per-file results |
| `GET /api/fitness-state` | compact structured state used by recommendation rules |
| `POST /api/recommendation` | health status/notes in, transparent recommendation out |
| `GET /api/settings` | editable configuration contract |
| `PATCH /api/settings` | validate/persist settings, then recalculate affected outputs |

## Frontend structure and layout

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Running Coach     Dashboard Progress Runs Next Run Settings  Upload │
├─────────────────────────────────────────────────────────────────────┤
│ DASHBOARD                                                           │
│ ┌ How am I doing? ┐ ┌ Last run ─────────┐ ┌ What next? ─────────┐  │
│ │ pace @145/trend │ │ distance/HR/zones │ │ workout/zones/why   │  │
│ └─────────────────┘ └───────────────────┘ └─────────────────────┘  │
│ load / consistency strip                                            │
├─────────────────────────────────────────────────────────────────────┤
│ PROGRESS: fitness chart + timeframe/raw toggle + period comparison  │
│           fitness | load | consistency | intensity | durability     │
├─────────────────────────────────────────────────────────────────────┤
│ RUNS: filters/table → RUN FEEDBACK detail                            │
│       raw @145 → named adjustments → standardized @145              │
│       mile splits | zones | moving/stopped | context | assessment   │
├─────────────────────────────────────────────────────────────────────┤
│ NEXT RUN: health input → workout prescription → reasons/warnings    │
├─────────────────────────────────────────────────────────────────────┤
│ SETTINGS: physiology | zones | model refs | load/rules | metadata   │
└─────────────────────────────────────────────────────────────────────┘
```

The dark visual system will reuse the current restrained color palette. Faster
fitness is labeled explicitly; uncertainty is shown as a band/interval. Upload
is always available in the top bar, supports drag/drop and file selection, and
opens single-run feedback after successful processing.

## Implementation phases

1. Typed API/data contracts and an app skeleton.
2. Upload pipeline service and upload UI.
3. Run-feedback calculations/rules and page.
4. Progress/load analysis and page.
5. `FitnessState` plus inspectable Python recommendation rules and tests.
6. Next Run page/current-status input.
7. Dashboard composition.
8. Settings, metadata editing, history filters, and polish.

Every phase adds tests before the next phase starts.

## Blind-spot register and confidence policy

The application must never treat an unavailable factor as evidence that the
factor was normal. Each `FitnessState` carries observed/inferred/missing context
and a list of known blind spots. Rules may reduce confidence or choose a safer
workout when important context is absent; they may not invent a numeric pace
correction.

| Factor | What can be known locally | Required behavior |
|---|---|---|
| Session distance/duration | TCX plus calculated movement | Separate comparable fitness observation from full-run load and durability |
| Intensity and cardiac drift | timestamped HR/movement intervals | Require adequate HR coverage; distinguish drift from stops and terrain |
| Prior training load | earlier activities only | Use leakage-safe 7/14/28-day distance, minutes, hard minutes, and zone load |
| Workout intent/type | user tag plus deterministic inference | Show inference and confidence; never call hikes or run/walks regressions |
| Illness/injury/pain | user tag/current-status input only | Exclude or caution explicitly; absence of a tag is not medical clearance |
| Sleep, stress, soreness, nutrition, hydration | not present in TCX | Mark missing; surface as a recommendation limitation rather than guessing |
| External vs optical HR/sensor changes | TCX creator/extensions may be incomplete | Detect observable discontinuities; lower confidence when source is unknown |
| Heat acclimation and direct sun | partially inferable; historical weather is modeled | Use literature prior/personal evidence; disclose shade-WBGT and exposure limits |
| Wind and precipitation exposure | hourly grid weather plus route bearing | Mark unavailable where route/weather resolution is inadequate |
| Terrain and technical surface | grade from GPS elevation; surface usually unknown | Use smoothed physical grade; flag trail/surface as missing unless tagged |
| Altitude/air quality | altitude partly present; air quality absent historically | Do not claim adjustment without reliable data |
| GPS/elevation accuracy | trackpoint coverage and plausibility checks | Propagate data quality to feedback and trend confidence |
| Stops, traffic, GPS gaps | movement classification | Keep elapsed, moving, and stopped time separate |
| Medication/caffeine/menstrual factors | not collected | Never infer; optional notes can inform caution but not diagnosis |
| Cross-training/manual activities | may be missing or mislabeled | State that running load is incomplete; classify suspicious run records |
| Sparse seasonal comparisons | known from matched-run counts | Shrink toward literature prior and label low/moderate/high evidence |
| Fitness changes during long gaps | only observed at recorded runs | Avoid interpolation claims across long gaps; lower trend certainty |

The recommendation trace will record which rules fired, the facts they saw,
and which unavailable factors limited confidence. “Smart” therefore means both
using more relevant evidence and being honest about what the file cannot say.
