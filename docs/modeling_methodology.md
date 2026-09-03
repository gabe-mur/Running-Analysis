# Modeling methodology: published references, moving fitness

## Why the model changed

Fitness is not a fixed nuisance variable. It changes over the same chronology as
season, weather, training volume, and route choice. A global regression on one
runner's history therefore cannot reliably identify a personal heat, hill, wind,
or cardiac-drift coefficient: it can assign genuine fitness change to a seasonal
covariate, or do the reverse.

The primary score no longer selects environmental weights by whichever candidate
best predicts this athlete's historical windows. Each run instead produces its
own pace-at-comparison-HR estimate. Adjustable 14-, 28-, 42-, 56-, 90-,
180-, and 365-day curves are
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
- have inadequate HR coverage or lack both usable GPS and Garmin device distance;
- contain excessive stopped/transition time; or
- overlap the heart-rate recovery that follows a long pause.

The last rule is separate from the stopped-time rule and catches a different
failure. A window *containing* a long stop already fails on stop fraction. The
problem is the window that starts just *after* one: heart rate falls quickly
when an athlete stands still and needs minutes of running to climb back, so
ordinary pace paired with a heart rate that has not caught up reads as unusually
good aerobic efficiency. Any window overlapping the first
`post_pause_suppression_moving_seconds` (180) of moving time after a stop of at
least `post_pause_minimum_stop_seconds` (60) is therefore dropped. Stop time is
summed across consecutive intervals and includes partially stopped `mixed_gap`
intervals, which is what Garmin's auto-pause produces. On this
athlete's history the rule suppresses about 2.6% of otherwise-retained windows
and moves per-run estimates in the conservative direction by a median of about
0.4 s/mi.

Recorded cadence is converted to total steps per minute in exactly one place
(`run_analysis.cadence.cadence_spm`, also exposed as `Trackpoint.cadence_spm`).
Garmin's `RunCadence` extension reports strides per minute for one leg and is
doubled; a plain TCX `Cadence` element already reports total steps and is not.
The raw value and its `cadence_source` are kept in the database so the
conversion stays auditable, and every cadence threshold is named `*_spm` so a
one-sided value can never be compared against a steps-per-minute limit.

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
grade-corrected raw pace at the comparison heart rate. Pair estimates are
time-weighted, with
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
Each run's windows are normalized to the configured comparison heart rate
(`target_hr`, 145 bpm by default) and the configured reference minute (20). No
stored column, model name, result key, or interface string hard-codes that
heart rate; each model run also records the `target_hr_bpm` it was scored at,
so a later change is visible rather than silently relabeling old estimates.
The remaining run-specific offset is that day's performance observation. The
model then robustly aggregates every usable window in **speed space**.
Continuous weights favor windows near the comparison heart rate and the
configured reference minute and downweight stops, within-window HR change, and
acceleration/deceleration. A Huber step further limits isolated GPS/pace
outliers. Overlap is accounted for
when calculating effective sample size, which is capped so a long run cannot
manufacture certainty merely by providing correlated windows. Runs far outside
the submaximal HR range are not extrapolated.

The prediction question is: given all usable continuous-running evidence in
this run, what speed does it support at the comparison heart rate, the
reference minute, and the
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
causal decomposition. A lower pace-at-comparison-HR value is faster. Every
scored run has
an auditable chain:

`Raw pace @target_hr → environmental adjustment → standardized pace @target_hr`

The environmental adjustment is decomposed into grade, temperature, dew point,
wind, and drift. Each part carries its own evidence label and personal-data
weight. Wind and drift currently display as unavailable/zero rather than
silently borrowing a fitted coefficient.

A Huber residual weight is applied to the trailing aggregate on top of the
measurement-uncertainty and health/workout weights. It is a deliberate
trade-off, measured by `scripts/huber_sensitivity.py`: on this athlete's
history a single corrupted run moves the 28-day level by about +2 s/mi with the
layer versus about +13 s/mi without it, while a genuine sustained step of 15 to
45 s/mi is registered within zero to one extra runs. The cost is that the
*reported magnitude* of a change is attenuated to roughly 75-85% of the
unweighted value; direction calls were unchanged across the 14-, 28-, 56-, and
90-day windows. Re-run that script after any change to the weighting.

The selectable fitness horizon changes interpretation: 14 days responds quickly
but is noisy; longer windows are more stable but slower. The current level,
comparison with the preceding equal-length window, within-window trajectory,
probability of improvement, personal-history percentile, best sustained period,
and evidence density all update together. The history range and smoothing
bandwidth are deliberately separate: the chart can show six months of runs
while retaining a rolling 28-day line. Making that line a six-month smoother
would conceal shorter improvement, interruption, and recovery phases.

Direction is evaluated in two complementary ways. The first compares the
weighted mean of the selected period with the preceding equal-length period.
The second fits a measurement-weighted, Huber-robust slope through the selected
period, which can detect gradual movement that two adjacent averages obscure.
If both are directional but disagree, the display reports no clear change
rather than choosing one silently.

Evidence labels make the statistical claim explicit. **Likely** means at least
80% one-sided probability in one direction. **Clear** means the estimated
change excludes zero at the 95% two-sided level. Sparse or stale coverage cannot
earn either label. The planner and training-status logic continue to use only
the clear signal; a likely display result is useful feedback, not permission to
increase training load.

## Training status

The dashboard headline is a classification, not a score. Garmin-style
"Productive / Unproductive" collapses several independent signals into one
number and then cannot explain itself; here the headline names a state, every
rule that produced it is listed with the facts it fired on, and the separate
signals stay visible underneath.

| Status | Meaning |
| --- | --- |
| Building | Load near demonstrated capacity, quality exposure present, no recovery flags, and something pointing upward |
| Maintaining | Steady load and performance, no strong signal either way |
| Rebuilding | Below retained capacity, but the most recent week is climbing back |
| Recovering | Current health check-in, or a health-tagged run not yet followed by three normal runs |
| Strained | Acute week at or above the configured high-load ratio, an unusually costly latest response, or high second-half drift |
| Underloaded | Sustained running below 70% of retained capacity and not climbing |
| Not enough data | Fewer than four activities in 28 days, or no demonstrated capacity |

Rules are evaluated in a fixed precedence order because the states are not
independent: health outranks load, and load outranks progression. An athlete
who is unwell is recovering even when their mileage looks ideal. "A little
tired" is an ordinary training day and deliberately does not trigger recovery.
The evidence gate reads the trailing 28-day activity count rather than
`running_days_28d`, which despite its name spans whatever window the caller
requested.

## Cadence and stride

Cadence feedback is personal and contextual; no population target is ever
prescribed. Two things make it analysis rather than advice.

Speed is exactly the product of turnover and stride length, so logarithms split
any speed change into two additive shares:

`ln(speed₂/speed₁) = ln(cadence₂/cadence₁) + ln(stride₂/stride₁)`

That identity is what lets the app say a pace change came "mostly through
longer stride rather than higher turnover" and mean it arithmetically. Where
one component moves against the speed change its share goes negative and the
other exceeds 100%; that case is described in words rather than shown as a
misleading percentage.

The comparison band is built from this athlete's own past quarter-mile segments
run within 8% of the current pace, reported as a median and a MAD-derived
spread, and requires at least 12 comparable segments. "Unusual" therefore means
unusual for this runner at this pace. Stride length is speed divided by
cadence — a proxy, not a measured ground-contact distance — and is labeled as
such.

## Fitness is a multi-signal interpretation

The trailing pace-at-comparison-HR estimate is not treated as a synonym for total
fitness. The application reports three distinct dimensions:

- **Aerobic efficiency:** reference-condition pace at the comparison heart
  rate, with short and
  longer-window comparisons.
- **Current condition:** illness/recovery tags, recent response, recovery
  spacing, and rolling training load.
- **Training capacity:** retained demonstrated weekly capacity — the best
  completed 28-day block, held in full through a grace period and then decayed,
  compared against the same figure one window earlier. A short illness or trip
  does not immediately redefine what the athlete has shown they can sustain.
- **Training volume:** recent distance and run count for the current period
  versus the preceding one. This is reported separately from capacity because
  a light fortnight lowers volume without lowering capacity. More of either is
  meaningful progress, but neither mathematically forces faster pace at a
  fixed HR.

The dashboard considers a responsive 28-day horizon and a sustained 90-day
horizon, and says when they disagree. The Progress view uses whichever horizon
the athlete selects. Progress charts show at most the trailing 365 days. Older
activities remain in run history rather than silently defining the current
baseline. Sparse comparison periods lower confidence.

Health-tagged runs retain full distance, duration, and load. Their vote in the
aerobic-efficiency aggregate is reduced rather than removed, because a run made
during illness still carries some information about that day's cost:

| Health tag | Trend weight |
| --- | --- |
| `normal` | 1.00 |
| `illness_recovery` | 0.65 |
| `illness` | 0.25 |
| `injury_affected` | 0.25 |
| anything else | 0.50 |

Workout type is a separate multiplier, and this one does reach zero: intervals,
threshold/tempo, race, hikes, and bike activities score 0.00 and therefore do
not vote at all, while run/walk sessions score 0.50. The two multiply, so an
illness-tagged interval session contributes nothing to the trend while an
illness-tagged easy run contributes a quarter vote. Every observation stays
visible in run history and in the coverage table either way, so the app does
not erase or falsify those sessions.

Graph membership is deliberately broader than trend evidence. Every running
activity with usable distance and moving time remains a point on the Progress
chart even when it has zero weight or cannot support a standardized
pace-at-comparison-HR estimate. When no standardized estimate exists, the point
is drawn at its unadjusted full-run pace and labeled as workout context with 0%
influence. It does not enter the robust trend, the VO2 conversion, or the latest
performance-response classification. A manual `include_in_model = false`
choice has the same fitness-evidence effect without deleting the completed
distance, duration, intensity, or recovery load from training history.

Workout classification is resolved once with the same precedence in Run
Analysis, Progress, and Weekly Plan: a manual workout-type override wins, then
the matched prescribed workout, then an athlete-relative fallback. An unlabeled
run at or above the current dynamic long-run threshold is classified as long;
other positive-duration running is classified as easy. The fallback does not
infer a quality label merely from heart-rate zones, because an unexpectedly
hard easy run and a deliberately prescribed quality session have different
planning meaning even though their recorded load is fully counted in both
cases.

## VO2-max estimate

Manual Garmin VO2-max and race-predictor snapshots have been removed. They were
transcribed by hand, arrived at irregular intervals, and were being read as
independent corroboration when they are a proprietary black box the application
cannot inspect or reproduce.

The remaining estimate treats the application's own central measurement as what
it actually is: a submaximal exercise test. Reference-condition speed at a fixed
heart rate is a steady-state workload paired with its heart-rate cost, with
grade, temperature, dew point, wind, and within-run position already removed —
the control a treadmill protocol gets from holding the laboratory constant.

Two steps, both published:

1. The **ACSM running equation** converts the standardized speed to oxygen cost,
   `VO2 = 0.2 x S + 0.9 x S x G + 3.5` for S in m/min. It is validated at or
   above 134 m/min (5 mph); below that no estimate is produced rather than a
   silent extrapolation.
2. **Heart-rate reserve** extrapolates to maximum, using %VO2 reserve = %HRR
   (Swain & Leutholtz, 1997) with VO2rest at 3.5 mL/kg/min. This uses the
   athlete's own resting and maximum heart rates rather than a population
   regression.

George et al. (1993) remains as a second, differently derived estimator. When
the two agree within their combined uncertainty they are pooled by inverse
variance; when they disagree the primary is kept and its interval widened to
span the gap, because averaging two disagreeing equations manufactures a tight
range around a number neither supports.

What makes the figure defensible is the interval, not the point. Uncertainty is
propagated by numeric partial derivatives from the standardized pace's own 95%
interval, the maximum-heart-rate figure, and each equation's published standard
error. Maximum heart rate is usually the largest single contributor, so the
profile records whether it was measured (1 SD 3 bpm) or age-predicted (1 SD
7 bpm), and the interface says outright that measuring it would narrow the
range more than any other single change.

The estimate refuses to produce a number when the standardized pace is below
the ACSM running range, when the comparison heart rate sits outside 25-90% of
heart-rate reserve, or when the required heart rates or profile fields are
missing — each with a specific reason rather than a generic disclaimer.

A Jackson age/sex/BMI/activity estimate is still shown only as a broad
demographic baseline (published SEE 5.7; approximately ±11.2 at 95%), and only
when documented running exceeds 10 mi/week so the activity rating is not
invented. None of this is Garmin's Firstbeat estimate, and none of it is an
independent vote: it is a unit conversion of the same pace-at-HR evidence and
moves with that trend by construction. The interface reflects that — an
"Experimental VO₂ cross-check" inside a collapsed advanced-analysis section,
not a second opinion beside the trend.

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

## Baseline acquisition and established planning

Baseline acquisition is a separate planning path, not a low end of the normal
weekly optimizer. With no usable running history, it prescribes one
conversational Z2/run-walk outing: at least 10 minutes for useful evidence,
continuing only while heart rate and legs remain comfortable, and stopping for
pain, fatigue, elevated heart rate, or at 30 minutes. Sparse history repeats the
observed baseline session until weekly capacity is measurable. It does not
invent a weekly mileage target, long run, or quality workout.

Once capacity is established, that calibration minimum is unavailable to the
weekly allocator. Ordinary easy-session options are anchored to the athlete's
historical easy range or to a shorter standalone prescription already justified
by recovery. Recovery trimming, frequency selection, and long-run preservation
use those established references; they cannot turn normal run opportunities
into ten-minute filler sessions to make a mileage total fit.

The ordinary-easy reference is a continuous-time recency-weighted median of
normal-health easy running, using the same configurable retention half-life as
capacity evidence rather than a hard 28-day cutoff. An easy run intentionally
shortened by the allocator is archived with a `support_easy` planning role. Its
completed distance, duration, intensity, and recovery cost remain fully counted,
but it does not teach the planner that the athlete's ordinary aerobic run has
become that short. Unmatched easy running and completed `ordinary_easy` sessions
continue to update the baseline normally.

The established planner evaluates a continuous 21-day calendar before showing
the first seven days. Dates, workout roles, and distance ranges are compared
together. Final allocated roles—not provisional labels—must preserve long and
quality cadence. Once elapsed recency selects a role, the generic single-run
score cannot silently replace it with easy mileage; exact cumulative recovery
may still make that substitution, and the allocator can shorten a quality dose.
This role binding is deliberate: the coordinator chooses the purpose of each
slot from the whole calendar, while health, the date-specific taper, and exact
recovery are separate eligibility checks. A negative preference score is not
silently treated as a new hard veto.
The finite day-21 edge is not a mileage deadline. A boundary-free load corridor
governs when mileage can be placed, the full-horizon rate governs how much is
funded, and a recency-decayed prior-plan preference prevents normal adherence
from pulling the next workout forward merely because the horizon moved one day.
The full lookahead still evaluates recovery, workout sequence, and overload,
but a 4/4/3-run horizon cannot create terminal "mileage debt" that manufactures
a fifth short run in the visible segment. Frequency candidates are compared
after their actual long, quality, and easy distances have been allocated;
target mileage divided by a guessed number of easy slots is not used as a proxy
for program quality. Provisional load-caution easy ranges later in the horizon
may be resized by the finalized joint model, while the first run's recorded-
recovery cap remains protected. This prevents a calendar of many small caution
runs from becoming the only numerical way to fund the target. Easy mileage
beyond the ordinary range is labeled medium-long only when it forms a
purposeful secondary endurance exposure.

The supplied weekly run-count estimate is not an optimizer target, and no
seven-day reporting slice is rewarded for containing three, four, or any other
count. A user-selected typical rest preference becomes a soft elapsed-hour
cadence reference across the continuous calendar. Time beyond that reference
is priced proportionally, while exact recovery and accumulated density can
justify either a shorter or longer gap. This prevents two complete rest days
from being silently free when the athlete selected a one-rest-day rhythm,
without demanding consecutive dates or imposing a weekly quota.

The adherence simulator uses the same receding horizon and defaults to
regenerating it every simulated day. Seven-day `ProjectionWeek` rows are report
buckets only: streaks and rolling load are calculated from the continuous dated
run history across row boundaries. A seven-day commit interval exists solely as
an explicitly requested fast-test mode and is not production-fidelity evidence
about schedule spacing. Its prescribed-distance column sums the recommendations
actually committed between replans. If a run is missed and the next daily plan
offers a replacement, both are reported as separate attempts; they were never
simultaneous mileage in one live plan.

Its perfect mode commits every prescription at the distance midpoint. The
seeded human mode perturbs only completed behavior: modest skipped, short,
long, or intensity-drifted runs; occasional easy/quality substitutions and
unscheduled easy running; two or three short surprise trips; and zero or one
seven-day vacation. These absences are not disclosed to the planner in advance,
so each daily regeneration must respond to the activity history it would
actually receive. Scenario probabilities are test inputs, not coaching rules.

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
threshold. The configurable 110% single-run progression ceiling is informed by
the Garmin-RUNSAFE cohort's observed increase in overuse-injury rates above a
10% increase over the longest run in the prior 30 days. That observational
association is used as a guardrail, not as a command to increase by 10% or as a
guarantee of safety below it.

There is no conventional five-mile minimum. A long run is labeled as such only
when the athlete-relative progression ceiling can produce a session meaningfully
longer than ordinary easy running. Otherwise the optimizer uses an easy or
quality session rather than manufacturing a long-run role from a global mileage
convention. The progression ceiling governs the prescribed midpoint; the small
published range around that midpoint is route and GPS flexibility.

Long-run targeting is separate from that ceiling. The working target progresses
by a configurable proportion of maintained recent single-run distance (5% by
default); it does not automatically spend the 10% guardrail. Quarter-mile
prescription rounding keeps that proportional target from becoming either a
fixed half-mile build or zero midpoint progress. There is no fixed
percentage-of-week ceiling on this session: the joint allocator first preserves
the recovery-compatible, single-session progression target, then allocates
quality and aerobic support around it. A recency-decayed six-month durability
reference preserves
evidence from older long runs without allowing an old peak to override the
current 30-day single-session guardrail. Completed runs refresh the reference,
so adherence can build the target upward rather than following decay downward.
When the maintained recent distance is below that retained capacity, a general
return-to-capacity rule can use more of the progression headroom, up to the 10%
ceiling. The rate fades back toward the ordinary 5% target as the retained level
is regained; historical progression speed is not learned as a personal safety
claim. The preferred-to-maximum gap is priced as one normalized program-fit
tradeoff, so spare weekly mileage does not make the maximum automatic. It stays
available when recovery, unavailable days, and target shortfall collectively
justify spending that headroom.
Once recovery and workout scoring select a safe, meaningful long run, the
weekly allocator reserves that target before quality structure and easy-mile
distribution; it does not erase the role merely to make all run distances more
equal.

Race goals add a required trajectory rather than merely changing workout
labels. Each profile defines a preparation-level peak weekly volume and long
run to reach before its taper. The planner works backward from the race date
and uses the compound progression required to reach those targets when it is
higher than the general-fitness build. It never raises either trajectory above
the configured weekly planning ceiling or single-session progression ceiling;
if the date would require that, goal validation rejects the date. The marathon
defaults (30 peak weekly miles and a 16-mile long run) are preparation targets,
not claims of medical safety or universal optimality. They are consistent with
observational recreational-marathon evidence associating less than 40 km/week
and a longest endurance run below 25 km with slower performance.

Tapering is evaluated on each calendar date. It suppresses ordinary long and
quality roles only before the race, makes the race itself a required optimizer
candidate, and does not label the rest of a 21-day horizon as taper merely
because a race appears near its beginning. After the race, another taxing
session waits on athlete-relative race load decaying through the existing
short-term density timescale; an easy run can become available sooner. Thus a
5K and marathon do not receive the same fixed post-race ban, and normal role
recurrence resumes once the modeled load supports it.

Recent intensity is shown with the athlete's configured five zones and grouped
for coaching as easy (Z1+Z2), moderate (Z3), and hard (Z4+Z5). The engine flags
moderate-intensity leakage but does not enforce a universal 80/20 quota. The
observational endurance literature uses several incompatible zone systems, so
the interface keeps this athlete's definitions explicit.

## Recovery model

Recovery is modeled as transient athlete-relative session load, not as a fixed
ban on running for 36, 48, or 72 hours. A completed run is compared with the
athlete's trailing-28-day ordinary run using distance, moving time, and recorded
HR-zone load. RPE, available zone fractions, hills/downhills, cardiac drift,
and an unusually costly or strong response make smaller graded adjustments.
Missing HR load does not become zero load; distance and duration retain their
weight and zone fractions provide a limited intensity fallback.

Prospective sessions use the same three-part shape: prescribed distance,
duration estimated at the athlete's recent pace, and the moderate/hard minutes
encoded in the workout structure. Workout names carry no recovery multiplier.
At equal distance, duration, and intensity, an easy run and a labeled long run
therefore have equal recovery cost; after upload, actual duration and HR-zone
load replace the estimates. This also means an easy run that becomes harder
than prescribed cannot remain artificially cheap merely because its label did
not change.

The resulting load decays exponentially by exact elapsed time with a 12-hour
half-life. This half-life is a transparent planning assumption, not a direct
measurement of muscle repair or a claim to reproduce a watch algorithm. The
athlete's own data determine the starting magnitude, so a short easy run clears
much sooner than a long, high-load, high-RPE, or mechanically costly one.
The application does not currently ingest sleep duration or quality. It
therefore does not pretend that a local nighttime interval proves sleep or
assign an unsupported universal conversion between sleeping and waking hours.
Measured sleep can become a separate modifier of this exact-time signal; until
then, athlete-facing guidance continues to disclose sleep as unavailable.

Easy and taxing workouts use different residual-load references. This allows
an easy run while the model would still steer away from another long or quality
session. Candidate timing, workout scoring, continuous spacing, and mileage
allocation all consume this same decaying signal. Calendar spacing remains a
soft cadence preference only; crossing a named number of hours or days never
causes recovery load to disappear.

Pairwise recovery has an explicit zero-cost envelope: a preceding residual and
the proposed session are charged only for the portion of their combined
athlete-relative load above two ordinary-session units. The slower bridge
signal shares that same envelope and contributes only the additional breach
not already priced by immediate recovery. It does not charge every nonzero pair
of runs merely because some exponentially decaying residue still exists.

The whole-program comparison also carries short- and longer-timescale mileage
density forward through the conditional plan. A discretely scheduled run
necessarily creates a pulse above a smooth mileage-rate target, so each curve
allows the exact pulse contributed by one athlete-typical easy run. Only load
that stacks beyond that dynamic session corridor is charged as excess density.
This prevents an ordinary isolated run from being mistaken for overtraining
while still making several individually tolerable runs costly when their
residual loads overlap.

Quality recurrence likewise has no minimum-days prohibition. Its priority
rebuilds continuously as the most recent quality session ages, while recent
14-day dose supplies a fading satisfaction signal. Projected sessions enter
that dose only while their timestamps remain inside the trailing 14 days; an
early-horizon workout cannot suppress quality indefinitely at day 20. A taxing session can move
earlier when recovery load is genuinely clear; a low recent count cannot by
itself manufacture another hard session immediately after the last one.

The final calendar comparison expresses target shortfall, collapsed aerobic
support, and surplus medium-long structure in one athlete-relative program-fit
unit. Recovery-envelope breaches use that same unit. This prevents independent
point multipliers for mileage, support, and schedule shape from silently
changing their relative importance. Continuous recovery overlap, weather,
cadence, and plan-continuity terms remain graded tradeoffs rather than
physiological thresholds.

## Race-goal guardrails

The optional 5K, 10K, half-marathon, and marathon goals are inspectable planner
inputs, not promises of a finish time. At least one ten-minute, normal-health
running performance is required. Up to the latest 10 are projected to the
selected distance with the Riegel 1.06 relationship; the median of the fastest
three available projections supplies the current-performance guardrail.
Limited history proportionally reduces the allowed improvement margin instead
of creating a ten-run eligibility cliff. Marathon
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
guardrail of 0.25% per available week, capped at 8% and scaled by the available
evidence; it is not a promised adaptation rate. Marathon feasibility is checked
with the compound long-run progression needed to reach the preparation target,
not a separate six-mile pass/fail gate.

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
