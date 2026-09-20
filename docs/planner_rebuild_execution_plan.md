# Planner rebuild execution plan

## Objective

Build from the smallest validated planner baseline and recover progression,
safety, and schedule stability without locking a merely compliant prior plan in
place. The release candidate must pass daily replanning over 90 days under both
perfect and imperfect adherence.

The behavioral baseline is commit `4d6dd6d` (planner v104) plus the
baseline-selection fix that filters for eligible activities before selecting
the four most recent baseline samples.

That baseline is intentionally not considered release-ready. It restores
progression, but the 90-day perfect-adherence run produces a seven-day running
streak, a 32.2-mile peak rolling-seven load against a 25.8–27.4-mile target,
and large load sawtoothing.

## Non-goals

- Do not make the prior plan a scoring favorite merely because it is the prior
  plan.
- Do not introduce Sunday–Saturday mileage budgets.
- Do not replace daily replanning with weekly or static-plan tests.
- Do not tune several unrelated weights in one experiment.
- Do not merge the entire dirty v114 layer into the baseline at once.
- Do not use a hard calendar lock to conceal optimizer instability.
- Do not interpret aggregated replacement attempts as one simultaneously live
  plan.

## Working-state protection

The current workspace contains valuable uncommitted planner, analysis,
strength, and UI work. Preserve it before rebuilding.

1. Record `git status --short` and `git diff --stat` in the experiment notes.
2. Do not reset, checkout, stash, or delete the current workspace.
3. Create an isolated worktree or disposable copy at commit `4d6dd6d`.
4. Apply only the baseline-selection fix to that isolated baseline.
5. Produce each phase as a separate patch or commit so it can be accepted or
   rejected independently.
6. Port successful commits back only after the full gate passes.

## Required test artifacts

Every simulation must emit machine-readable JSON in addition to the existing
human-readable report. Store the following for every daily replan:

- decision timestamp and decision start date;
- target trajectory used for scoring;
- selected 21-day dates, workout types, and distance ranges;
- sessions committed before the next refresh;
- opening continuous and short-term load;
- peak projected load;
- objective components for the winning candidate;
- objective components for the translated prior candidate;
- candidate count, full-score count, and elapsed planning time;
- material-evidence reasons that permit a date/type change.

The simulator must return a nonzero exit status when a required invariant
fails. A report that merely prints an unacceptable streak is not a passing
test.

## Gate definitions

### Perfect-adherence gate

Use 91 daily decision windows, exact midpoint completion, normal health, no
invented weather change, and no skipped or unscheduled work.

Required invariants:

1. An established athlete never returns a zero target or baseline-required
   mode solely because recent long, quality, or support runs hide an older
   valid easy baseline.
2. No 21-day plan is empty while the target is positive and the athlete is
   healthy.
3. Capacity at day 91 exceeds opening capacity; no completed compliant block
   causes capacity or target to collapse to zero.
4. Long and quality lanes recur within their configured cadence unless
   recovery evidence explicitly rejects them.
5. Consecutive run dates have no fixed pass/fail limit. Every proposed
   sequence must pass the same athlete-relative recovery/readiness rules,
   taxing-session interaction rules, and continuous-load corridor. Three or
   more consecutive dates remain diagnostic evidence, but fail only when an
   actual rule fails or compliant replanning creates unexplained churn.
6. At every daily boundary, rolling or continuous load stays inside the target
   corridor plus at most one athlete-typical ordinary session of boundary
   phase. A new day entering the horizon cannot repeatedly consume that
   allowance.
7. The 21-day prescribed midpoint funds the integrated target within one
   ordinary session of boundary phase.
8. After excluding a completed/consumed session, a compliant refresh does not
   change dates or workout types in the next four days unless recorded
   evidence changed materially. Distance-only changes are recorded separately.
9. A compliant in-range completion does not create recovery surprise relative
   to the pre-upload projection.

### Imperfect-adherence gates

Run these deterministic scenarios separately before the mixed human profile:

1. **In-range completion:** complete at both the low and high edge. Dates and
   workout types in the next four days must match the expected-compliance
   projection unless another material signal changes.
2. **Below-range easy completion:** permit lower future load or added useful
   work, but do not require the calendar to remain fixed.
3. **Above-range or harder completion:** permit recovery-driven reductions or
   date movement and record the material evidence responsible.
4. **Single hidden miss:** the planner may replan from actual history, but the
   missed mileage must not become a replacement debt that creates a compressed
   streak.
5. **Known forced rest/vacation:** schedule no runs on protected dates and
   re-establish a safe cadence after the block without catch-up compression.
6. **Hidden vacation:** repeated daily recommendations during unknown
   unavailability are counted as attempts, not simultaneous planned mileage.
   Evaluate the first plan after activity resumes separately.
7. **Unscheduled run:** distinguish athlete-created streaks from
   planner-created streaks in all metrics.

After those pass, run the 91-day seeded mixed-human scenario.

## Phase 0 — Make the 90-day workflow executable

### Status — 2026-09-19

Implemented in the protected working tree:

- fail-fast JSON projection reports and nonzero regression exits;
- daily target, load, plan, commitment, timing, and churn evidence;
- winning-candidate and translated-prior objective breakdowns plus candidate
  and full-score counts;
- separate live-plan, committed-prescription, actual, and
  athlete-involved streak accounting;
- unit gates for starvation, readiness, consumed-session churn, fail-fast
  behavior, and JSON serialization.

The isolated v104-plus-baseline-fix projection proved that daily replanning can
assemble September 22–24 as three consecutive prescribed run dates even when
no individual live plan contains more than a two-day future streak. That is a
valuable diagnostic that a static plan misses, but it is not itself a failure:
the sequence must be judged against recovery, readiness, load, and stability.
With the corrected gates, v104 instead fails for an actual stability
violation on the September 22 refresh: after compliant execution and no new
material evidence, its next-four-day plan moves the easy run from September 24
to September 23 and the interval session from September 26 to September 25.

Phase 0 exit criteria are satisfied. Named deterministic scenarios, cadence
gates, pre-upload recovery-surprise evidence, and the focused hidden-vacation
assertion are implemented. Churn coverage distinguishes consumed sessions,
date moves, workout-type changes, distance-only edits, configured window
boundaries, and changes authorized by recorded material evidence. The
corrected v104 proof fails on unexplained compliant-plan churn rather than on
streak length.

### Changes

- Add JSON serialization for replan snapshots and objective components in
  `src/run_analysis/adherence_projection.py`.
- Extend `scripts/simulate_adherence.py` with:
  - `--output-json PATH`;
  - `--fail-on-regression`;
  - `--stop-on-first-failure`;
  - named scenario selection;
  - planner timing and candidate-count reporting.
- Add a gate evaluator that separates:
  - planned streaks;
  - committed prescribed streaks;
  - actual streaks;
  - unscheduled athlete-created streaks.
- Add unit tests for churn-window accounting and gate failures.

### Commands

```bash
.venv/bin/python -m pytest tests/test_adherence_projection.py -q
PYTHONPATH=src .venv/bin/python scripts/simulate_adherence.py \
  --weeks 13 --mode perfect --replan-days 1 \
  --ignore-saved-plan --fail-on-regression --stop-on-first-failure \
  --output-json output/planner-baseline-perfect.json
PYTHONPATH=src .venv/bin/python scripts/simulate_adherence.py \
  --weeks 13 --mode human --seed 20260902 --replan-days 1 \
  --ignore-saved-plan --fail-on-regression --stop-on-first-failure \
  --output-json output/planner-baseline-human.json
```

### Exit criteria

- The existing v104-plus-fix perfect case fails automatically at its first
  actual recovery, readiness, load, or unexplained-stability violation;
  streak length alone is not a violation.
- A synthetic starvation case fails automatically.
- Consumed sessions do not count as churn.
- Hidden-vacation attempts are not presented as one live plan.

Do not begin Phase 1 until the harness can fail for the already-known defects.

## Phase 1 — Establish and freeze the clean baseline

### Status — 2026-09-19

A fresh source tree was extracted from commit `4d6dd6d` at
`/tmp/running-v104-clean.mMvHuY`. Its only planner change is the eligible-first
baseline fallback; a focused regression test was added alongside the three
neighboring baseline-mode tests, and all four pass. No v114 scoring or search
term is present. The corrected fail-fast daily proof from the equivalent
instrumented baseline stops on the September 22 compliant near-term churn, so
the known defect is reproduced without continuing a failed projection through
the remaining 87 decision windows. The exact baseline delta is preserved in
`docs/experiments/v104-eligible-baseline.patch`; it applies cleanly to
`4d6dd6d` in a newly extracted tree. The four observed planner refreshes took
12.99–17.31 seconds (14.58-second median). The reproducibility and failure
record is `docs/experiments/phase1-v104-baseline.json`.

Phase 1 exit criteria are satisfied. This baseline is frozen as the comparison
point for Phase 2; the temporary directories are conveniences, not the source
of truth.

### Changes

- Start from commit `4d6dd6d`.
- Apply only the eligible-baseline fallback fix.
- Add its focused regression test.
- Capture the perfect and imperfect baseline JSON artifacts.
- Record single-replan median and worst-case runtime.

### Exit criteria

- Focused baseline tests pass.
- The 90-day perfect run reproduces progression and the known compliant-plan
  churn failure; consecutive-day patterns remain diagnostic unless an actual
  safety rule also fails.
- No v114 scoring term is present.
- The baseline patch can be reproduced from one documented commit or patch.

## Phase 2 — Port evidence and compliance correctness

### Status — 2026-09-19

A minimal prototype is applied only in the isolated v104 baseline. It contains
adherence-normalized target evidence, upper-edge prescription-aware recovery,
and exact intraday continuous-load impulses. The calendar-day workout-role
logic and its DST-aware regression were already present in v104, so no duplicate
implementation was added.

Eight focused invariants pass, the complete 19-test recovery suite passes, six
related target/time/role tests pass, and the saved-plan replay compatibility
check passes. Runtime remains comparable to Phase 1: the two measured refreshes
took 14.34 and 17.44 seconds.

The evidence invariants do not resolve optimizer churn. The fail-fast daily
replay stops on the September 20 refresh: the September 21 long and September
23 easy sessions remain, but an unanticipated September 24 easy session enters
the next-four-day window after compliant execution. See
`docs/experiments/phase2-v104-gate.json`. Phase 2 is therefore preserved but
not declared complete; its four-day stability exit criterion remains red and
is the concrete transition to the Phase 3 search review.

Port only evidence-handling changes that define what materially changed:

1. Adherence-normalized target distance: any completion inside a prescribed
   range contributes the prescribed midpoint to target evidence; only
   out-of-range deviation changes that evidence.
2. Prescription-aware recovery load: plan against the supported upper edge of
   a compliant range; retain intensity, RPE, terrain, drift, and response as
   independent reasons for higher load.
3. Exact session-time decay across intraday and midnight boundaries.
4. Same-calendar-day refreshes cannot change workout role solely because the
   suggested clock time moved.

### Focused commands

```bash
.venv/bin/python -m pytest tests/test_recovery.py -q
.venv/bin/python -m pytest tests/test_weekly_schedule.py -q -k \
  "adherence_normalized or compliant_upload or exact_time or intraday or same_calendar_day"
```

### Exit criteria

- Low-edge, midpoint, and high-edge compliant completions match the pre-upload
  expected recovery state.
- A materially easier or harder completion remains able to alter the plan.
- The four-day stability gate passes for in-range completion.
- Runtime does not regress materially from the Phase 1 measurement.

## Phase 3 — Port score-neutral search and compute improvements

### Status — 2026-09-19

Code review found that v104 already contains candidate-state and load caches,
joint-cost reuse, dominated-allocation pruning, duplicate-solve avoidance,
diverse beam retention, and equal full scoring for the translated prior and
its bounded neighborhood. The missing rollover safeguard is a natural
far-edge extension when the horizon advances and the candidate count changes.
That addition passes its exactness tests, removes the September 20 churn
failure, and retains Phase 1 runtime (14.11-second median, 17.44-second max in
the three-refresh proof).

Forcing every cadence anchor plus clustered and terminal counterparts through
full distance allocation was rejected: the first refresh exceeded three
minutes and was interrupted inside the allocation DP. Anchor construction and
neighbor generation remain useful tested helpers, but cannot be enabled with
that admission strategy.

Disposable score tracing at the next failure proves the September 21 rewrite
is not a beam omission. Both the stable continuation and the selected calendar
receive the same full objective. The selected ten-run calendar scores 67.01;
the stable ten-run continuation scores 84.35 despite better target fit and
nearly identical finalized recovery. Its deficit is primarily coaching/cadence
and role-shape cost. See
`docs/experiments/phase3-v104-search-diagnosis.json`. Phase 3 therefore
preserves the bounded far-edge candidate but hands the remaining failure to
objective/state-transition analysis rather than wider search.

A later Phase 5 no-evidence boundary exposed one additional, narrower search
case. When the translated prior needs new far-edge dates to match a changed
horizon count, the exact extension was fully scored but its one-day local
neighbors were not. A neighbor scored 25.64 versus the extension's 27.07, but
was discovered only on the following refresh and caused a Sep 27 to Sep 28
move. Fully scoring the same bounded first-four-date neighborhood around the
far-edge extension selects 25.64 on the earlier refresh and retains the exact
same winner and score on the next refresh. This is a score-neutral Phase 3
addition, not a continuity preference. See
`docs/experiments/phase5-v104-neutral-target-search-diagnosis.json`.

Port and validate:

- candidate-state, recommendation-load, and joint-cost reuse;
- exact dominated-assignment pruning;
- avoiding duplicate long/aerobic allocator solves;
- diverse beam retention;
- translated prior plan as an unpreferred candidate;
- bounded prior-plan neighbor generation;
- cadence anchors and their local neighbors reaching full scoring.

These mechanisms may expose a better candidate that the old beam missed, but
they must not give any candidate a score bonus.

### Focused commands

```bash
.venv/bin/python -m pytest tests/test_weekly_schedule.py -q -k \
  "candidate_work_reuse or dominated_allocation or beam_bounds or joint_finalists or incumbent or cadence_anchor"
```

### Exit criteria

- Exactness/search tests pass.
- A wider beam cannot remove a previously evaluated cadence anchor.
- Prior-plan and fresh-plan candidates receive the same objective calculation.
- Single-replan runtime improves or remains within the Phase 1 budget.

## Phase 4 — Add the compression-safety bundle

### Prototype status — 2026-09-19

The isolated v104 prototype passes eight focused recovery, density,
compression, continuation, and incumbent-admission tests. It moves the first
daily churn failure from September 20 to September 22 while improving the
four-refresh median to 12.81 seconds. The remaining change retains September
23 easy and September 25 intervals but adds September 26 easy. That is an
allowed back-to-back rather than a three-session compression breach, but it is
still a churn failure and is not waived.

Preserve this bundle under the Phase 5 experiment, then rerun the same gate.
Do not call Phase 4 complete unless the combined planner passes. See
`docs/experiments/phase4-v104-compression-gate.json`.

Port these as one internally coherent recovery bundle while leaving target
weight at `PROGRAM_FIT_UNIT = 20`:

1. Finalized-distance recovery scoring.
2. Upper-edge recovery pricing for compliant distance ranges.
3. Candidate-independent target-derived bridge density reference.
4. Opening history/plan seam compression cost.
5. Third-compressed-session cost that permits one intentional back-to-back but
   prices the third close session.
6. Short-term density residue across several individually tolerable runs.
7. Soft terminal continuation that prices recovery residue beyond day 21
   without optimizing another calendar.

Do not add stronger target funding, fragmentation penalties, or underload debt
in this phase.

### Test sequence

1. Focused unit tests.
2. Replay the first known v104 compression transition.
3. Run 42 daily replans; this must reach the former week-six four-day streak.
4. If and only if the 42-day gate passes, run the 91-day perfect gate.

### Focused command

```bash
.venv/bin/python -m pytest tests/test_weekly_schedule.py -q -k \
  "projected_recovery_prices_upper_edge or finalized_program_recovery or bridge_density or opening_compression or third_compressed or soft_continuation"
```

### Exit criteria

- No planner-created streak above two days in the perfect fixture.
- Back-to-back running remains available.
- A third close session loses to an otherwise comparable spaced candidate.
- A terminally backloaded candidate costs more than an equivalent earlier
  cadence.
- Long and quality cadence is not starved to satisfy density.

Stop at the first failing daily boundary and compare winner/runner-up objective
components. Do not continue a known-failed 90-day run.

## Phase 5 — Make target projection candidate-independent

### Diagnostic attribution — 2026-09-19

This was an already-identified defect, not a new hypothesis. A disposable
ablation now shows where it is causal: holding the scoring target fixed for
all candidates removes the September 20 long-run date move. It does not remove
all churn; after the compliant September 21 long run, the planner still adds a
September 24 easy run beside the retained September 23 run.

The follow-up full-score comparison also rules out a beam omission for that
remaining boundary. The visible plan is about one ordinary session short of
the target recurring rate, and the seven-day transition-path term makes an
immediate addition cheaper than an otherwise stable later addition. Therefore
Phase 5 must build the neutral stream through a continuation tail beyond day
21. A fixed opening target is only an attribution tool, not the implementation.
See `docs/experiments/phase5-6-boundary-diagnosis.json`.

The first immutable-reference prototype used the prior winning plan as the
next invocation's reference. That was rejected because it remained
self-referential across refreshes: today's winner became tomorrow's target
evidence despite no completed work. The accepted prototype direction uses one
deterministic athlete-relative neutral stream that depends only on observed
history and opening evidence. Candidate calendars and the prior winner cannot
alter it. Focused candidate-independence tests and exact saved-plan replay pass.

The broader daily gate now emits the immediate post-upload replan used by
production. Snapshots identify `scheduled_refresh` versus `post_upload`, the
post-upload decision boundary begins after the completed run, and both the
gate and JSON report separate compliant-upload churn from no-new-evidence
refresh churn. Evidence is scoped to the immediately preceding replan, so a
deviation consumed by the upload replan cannot also excuse a later clock
refresh. Focused exact-compliance, range-edge, material-deviation, unscheduled
run, and event-order tests pass. The next validation is a bounded real-planner
replay before any 42- or 91-day run.

The current v104/v114 projector accepts each candidate's recommendations and
therefore allows candidates to alter the target used to judge themselves.
Replace that interface.

### Design

1. At the start of one planner invocation, construct one deterministic
   reference-compliance activity stream from:
   - observed history;
   - opening target;
   - athlete-typical ordinary session size;
   - a neutral recovery-spaced cadence;
   - expected midpoint compliance.
2. Derive the target trajectory once from that reference stream.
3. Pass the resulting immutable trajectory to every candidate.
4. Candidate dates, roles, and distances never modify their scoring target.
5. Recompute the trajectory on the next real daily refresh using actual
   history. Do not carry missed prescription mileage as debt.

### Required unit invariants

- Sparse and dense candidates receive byte-for-byte identical target
  trajectories within one invocation.
- Removing a candidate run cannot lower that candidate's obligation.
- A missed run changes the next day's target only through actual observed
  history, not through an unpaid prescription ledger.
- A compliant completion matches the trajectory anticipated before upload.
- The reference stream cannot establish a new ordinary-easy baseline before a
  real eligible run is completed.

### Exit criteria

- Phase 4 perfect gates still pass.
- The single-hidden-miss scenario does not create catch-up compression.
- Candidate counts and runtime remain within the recorded budget. If runtime
  expands, profile the exact scoring path before broad simulation.

## Phase 6 — Add boundary-safe funding

### Prototype status — 2026-09-19

A phase-normalized recurring rate plus one ordinary-session boundary allowance
correctly exposes the post-long-run volume gap, but is not safe by itself. With
the existing short transition path it moves the missing session into the next
four days and fails the churn gate. Do not promote this prototype ahead of the
Phase 5 neutral continuation stream.

Only after compression safety and target independence pass, port:

1. Phase-normalized recurring program rate.
2. Hard 21-day funding floor.
3. One ordinary-session boundary-phase allowance.
4. Fragmentation detection for sub-useful established easy runs, initially as
   a diagnostic metric rather than a weighted penalty.

Keep `TARGET_FIT_UNIT` at 20. Do not enable underload debt.

### Required comparisons

- A healthy recurring cadence shifted by one day scores equivalently when its
  only difference is whether the next recurrence lands just inside day 21.
- A plan missing more than one ordinary session of 21-day target load fails
  funding even when its interior cadence looks correct.
- Funding cannot make a compressed candidate beat a safely spaced candidate
  unless the spaced candidate genuinely cannot carry useful sessions.

### Exit criteria

- No starvation.
- No perfect-adherence compression regression.
- No 21-day underfunding larger than the boundary allowance.
- Daily target/load corridor passes without Sunday–Saturday logic.

## Phase 7 — Controlled ablations before weight tuning

Implement real, integration-tested feature switches for:

- soft continuation;
- compression bundle;
- hard funding;
- target weight 20 versus 35;
- any future underload-debt term.

Every switch must be exercised by a test proving that it changes the intended
production objective. Remove or repair the current ineffective
`no-underload-debt` ablation before drawing conclusions from it.

Run the 42-day perfect gate for each single ablation. Run 91 days only for
configurations that pass.

Accept a higher target weight only if it reduces genuine underfunding without:

- increasing maximum planner streak;
- increasing peak load outside the corridor;
- increasing next-four-day date/type churn;
- producing smaller fragmented easy runs;
- materially increasing runtime.

## Phase 8 — Imperfect adherence and stability validation

Run the named scenarios from the imperfect-adherence gate, followed by the
seeded 91-day mixed-human run.

For every refresh, classify the motive for change:

- consumed/completed session;
- material recovery deviation;
- target/capacity change;
- weather change;
- forced rest or health change;
- search-only churn with no material evidence.

Search-only date/type churn inside the next four days is a regression. A
distance change may be accepted when target or recovery changed and the date
pattern remains stable.

### Exit criteria

- Known unavailable dates remain run-free.
- One hidden miss does not generate replacement debt.
- In-range adherence produces no unexplained date/type churn.
- Planner-created streak limits pass independently of actual athlete-created
  streaks.
- Long and quality cadence remains useful after interruptions.

## Phase 9 — Integrate independent features

After the core planner passes all gates, reapply and validate separately:

1. Quality-workout rotation and load-aware workout-type selection.
2. Strength suggestions and leg-day placement.
3. Progress-page and run-analysis fixes.
4. UI and settings changes.

Strength suggestions remain display-only and must not change run optimizer
scores during this phase.

## Phase 10 — Release gate

Run:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/simulate_adherence.py \
  --weeks 13 --mode perfect --replan-days 1 \
  --ignore-saved-plan --fail-on-regression \
  --output-json output/planner-release-perfect.json
PYTHONPATH=src .venv/bin/python scripts/simulate_adherence.py \
  --weeks 13 --mode human --seed 20260902 --replan-days 1 \
  --ignore-saved-plan --fail-on-regression \
  --output-json output/planner-release-human.json
```

Then replay the latest saved production snapshot and confirm deterministic
output.

Do not deploy unless:

- the full unit suite passes;
- both 91-day daily-replan gates pass;
- next-four-day stability passes under perfect adherence;
- no unexplained search-only churn remains;
- runtime is recorded and acceptable;
- the generated live plan is reviewed before server startup.

## Experiment log template

Record one row for every behavioral change:

| Field | Value |
| --- | --- |
| Baseline commit/patch | |
| Hypothesis | |
| Single code change | |
| Expected first affected boundary | |
| Focused tests | |
| 42-day perfect result | |
| 91-day perfect result | |
| Named imperfect scenarios | |
| 91-day human result | |
| Stability metrics, 3/4/7 days | |
| Maximum planner-created streak | |
| Peak load versus corridor | |
| Capacity start/end | |
| Median/worst planner runtime | |
| Decision: keep/revise/revert | |

## Immediate next action

Run a bounded real-planner replay through the corrected event sequence and
inspect the two churn lanes independently. If that passes, run the fail-fast
perfect daily replay only until its first regression; diagnose that transition
before extending to 42 or 91 days. Do not tune scoring weights from aggregate
output.
