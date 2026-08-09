import { api, appSettings, targetHrLabel } from "../api.js";
import { balanceSentiment, DIRECTION, SENTIMENT, comparisonValue, directionValue, trendValue } from "../components.js";
import { fitnessChart, vo2Chart } from "../charts.js";
import { goalMarkup } from "../goal.js";
import { view, loading } from "../dom.js";
import {
  dateLabel, escapeHtml, evidenceLabel, fitnessWindowLabel, number, pace, titleCase,
} from "../format.js";

let progressWindow = 28;
let progressMetric = "standardized";
let progressWindowInitialized = false;


export async function renderProgress() {
  loading("Progress");
  if (!progressWindowInitialized) {
    const settings = await appSettings();
    progressWindow = settings.default_fitness_window;
    progressWindowInitialized = true;
  }
  const [progress, goal] = await Promise.all([
    api(`/api/progress?window_days=${progressWindow}`),
    api("/api/goal-progress").catch(() => null),
  ]);
  const comparison = progress.period_comparison;
  const change = progress.pace_change_seconds_per_mile;
  const changeUncertainty = progress.pace_change_uncertainty_95_seconds_per_mile;
  const trendText = progress.fitness_trend === "improving" ? "Your aerobic efficiency is improving." : progress.fitness_trend === "declining" ? "Your aerobic efficiency has declined recently." : progress.fitness_trend === "stable" ? "Your aerobic efficiency is about the same." : progress.fitness_trend === "uncertain" ? "There is no clear change in aerobic efficiency." : "More comparable runs are needed.";
  const paceChange = Number.isFinite(change) && ["improving", "declining"].includes(progress.fitness_trend)
    ? `${Math.abs(change).toFixed(0)} sec/mi ${change > 0 ? "slower" : "faster"}`
    : Number.isFinite(change) ? "No clear change" : "Not enough data";
  const paceChangeDetail = Number.isFinite(change)
    ? `The estimate is ${change > 0 ? "+" : ""}${change.toFixed(0)} sec/mi${Number.isFinite(changeUncertainty) ? `; the likely range extends about ${changeUncertainty.toFixed(0)} sec/mi either way` : ""}.`
    : "The previous period does not have enough comparable runs.";
  const ratio = progress.current_load.acute_distance_to_capacity_ratio;
  // A week far above demonstrated capacity is a caution, not an achievement,
  // and a week far below it is not either -- only the usual band is neutral.
  // Load is the clearest case where direction and meaning come apart: 145% of
  // capacity is "up" and bad, 55% is "down" and also bad, and 100% is neither.
  const loadDirection = !Number.isFinite(ratio) ? DIRECTION.NONE
    : ratio > 1.05 ? DIRECTION.UP : ratio < 0.95 ? DIRECTION.DOWN : DIRECTION.FLAT;
  const loadSentiment = !Number.isFinite(ratio) ? SENTIMENT.NONE
    : ratio >= 1.3 || ratio < 0.7 ? SENTIMENT.BAD : SENTIMENT.NEUTRAL;
  const previousLongest = comparison.previous.longest_run_miles;
  const localVo2 = progress.local_vo2_estimate;
  const coverageRows = progress.activity_coverage.map((item) => {
    const label = item.score_status === "trend_evidence" ? "Used" : item.score_status === "uncertain_estimate" ? "Used as an estimate" : item.score_status === "reduced_weight" ? "Used with less influence" : item.score_status === "workout_specific" ? "Workout only" : item.score_status === "context_only" ? "Shown only" : item.score_status === "non_running" ? "Not a run" : "Not comparable";
    const quality = item.score_status === "trend_evidence" ? "good" : ["reduced_weight", "uncertain_estimate", "workout_specific"].includes(item.score_status) ? "partial" : "unavailable";
    return `<tr><td><a href="#run/${item.activity_id}">${dateLabel(item.start_time)}</a></td><td>${number(item.distance_miles)} mi</td><td>${titleCase(item.workout_type)}</td><td>${titleCase(item.health_tag)}</td><td>${Number.isFinite(item.standardized_pace_min_mile) ? pace(item.standardized_pace_min_mile) : "—"}</td><td><span class="quality ${quality}">${label}</span></td><td class="coverage-reason">${escapeHtml(item.reason)}</td></tr>`;
  }).join("");
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">Progress</p><h1>${trendText}</h1><p>Compares your pace at the same heart rate, weather, grade, and point in the run.</p></div><div class="headline-pace"><strong>${progress.current_pace?.display ?? "—"}</strong><span>estimated pace at ${targetHrLabel()}</span></div></div>
      ${goalMarkup(goal, "wide")}
      <div class="toolbar"><div class="segmented" aria-label="Fitness time frame">${progress.available_windows.map((days) => `<button type="button" data-window="${days}" class="${days === progressWindow ? "selected" : ""}">${fitnessWindowLabel(days)}</button>`).join("")}</div><div class="segmented"><button type="button" data-metric="standardized" class="${progressMetric === "standardized" ? "selected" : ""}">Adjusted pace</button><button type="button" data-metric="raw" class="${progressMetric === "raw" ? "selected" : ""}">Unadjusted pace</button><button type="button" data-metric="both" class="${progressMetric === "both" ? "selected" : ""}">Both</button></div></div>
      <article class="wide-card chart-card"><div class="card-heading"><div><p class="eyebrow">Last ${fitnessWindowLabel(progressWindow)}</p><h2>${progressMetric === "standardized" ? `Adjusted pace at ${targetHrLabel()}` : progressMetric === "raw" ? `Unadjusted pace at ${targetHrLabel()}` : "Adjusted and unadjusted pace"}</h2></div><span class="quality ${progress.fitness_confidence}">${evidenceLabel(progress.fitness_confidence)}</span></div>${fitnessChart(progress.series, progressMetric, progressMetric !== "raw" ? progress.trend_28d : [], progress.as_of, progressWindow)}<p class="chart-note">Each dot is one run. The line is your 28-day average, which is what the app reasons from. Thin vertical bars show the likely range for each estimate.</p></article>
      <div class="metric-grid progress-metrics">
        <article><span>Aerobic efficiency</span><strong>${trendValue(progress.fitness_trend, paceChange)}</strong><small>${escapeHtml(paceChangeDetail)} ${evidenceLabel(progress.fitness_confidence)}.</small></article>
        <article><span>Last 7 days</span><strong>${comparisonValue(progress.current_load.trailing_7d.distance_miles, progress.current_load.capacity_reference_miles, `${number(progress.current_load.trailing_7d.distance_miles)} mi`, { deadband: 1 })}</strong><small>${number(progress.current_load.trailing_7d.zone_load, 0)} training-load points</small></article>
        <article><span>This week vs usual</span><strong>${Number.isFinite(ratio) ? directionValue(loadDirection, loadSentiment, `${number(ratio * 100, 0)}%`) : "—"}</strong><small>${number(progress.current_load.trailing_7d.distance_miles)} mi this week · about ${number(progress.current_load.capacity_reference_miles)} mi in a typical week</small></article>
        <article><span>Longest recent run</span><strong>${comparisonValue(progress.consistency.longest_run_miles, previousLongest, `${number(progress.consistency.longest_run_miles)} mi`, { deadband: 1 })}</strong><small>${number(progress.consistency.runs_per_week, 1)} runs per week · ${number(previousLongest)} mi previously</small></article>
      </div>
      <div class="two-column">
        <article class="wide-card"><p class="eyebrow">Compared with the previous ${fitnessWindowLabel(progressWindow)}</p><h2>Training volume</h2><div class="comparison-grid"><span>Distance<strong>${comparisonValue(comparison.current.distance_miles, comparison.previous.distance_miles, `${number(comparison.current.distance_miles)} mi`, { deadband: comparison.previous.distance_miles * 0.05 })}</strong><small>Previously ${number(comparison.previous.distance_miles)} mi</small></span><span>Running time<strong>${comparisonValue(comparison.current.moving_minutes, comparison.previous.moving_minutes, `${number(comparison.current.moving_minutes, 0)} min`, { deadband: comparison.previous.moving_minutes * 0.05 })}</strong><small>Previously ${number(comparison.previous.moving_minutes, 0)} min</small></span><span>Training load<strong>${comparisonValue(comparison.current.zone_load, comparison.previous.zone_load, `${number(comparison.current.zone_load, 0)}`, { deadband: comparison.previous.zone_load * 0.05 })}</strong><small>Previously ${number(comparison.previous.zone_load, 0)}</small></span><span>Runs<strong>${comparisonValue(comparison.current.run_count, comparison.previous.run_count, `${comparison.current.run_count}`, { deadband: 1 })}</strong><small>Previously ${comparison.previous.run_count}</small></span></div><p>${escapeHtml(comparison.interpretation)}</p></article>
        <article class="wide-card"><div class="card-heading"><div><p class="eyebrow">Effort mix</p><h2>${number(progress.intensity.easy_percent, 0)}% easy · ${number(progress.intensity.moderate_percent, 0)}% moderate · ${number(progress.intensity.hard_percent, 0)}% hard</h2></div><span class="trend-status ${balanceSentiment(progress.intensity.balance)}">${escapeHtml(progress.intensity.balance_headline)}</span></div><div class="intensity-bar"><b style="width:${progress.intensity.easy_percent ?? 0}%"></b><i style="width:${progress.intensity.moderate_percent ?? 0}%"></i><em style="width:${progress.intensity.hard_percent ?? 0}%"></em></div><p class="balance-detail">${escapeHtml(progress.intensity.balance_detail)}</p><small>${progress.consistency.quality_sessions} hard workouts · ${progress.consistency.running_days} run days</small></article>
      </div>
      <article class="wide-card vo2-card">
        <div class="vo2-layout">
          <div class="vo2-figure">
            <div class="card-heading"><div><p class="eyebrow">Estimated from your own runs</p><h2>VO₂ max</h2></div><span class="quality ${localVo2.confidence}">${evidenceLabel(localVo2.confidence)}</span></div>
            ${Number.isFinite(localVo2.value_ml_kg_min) ? `<p class="vo2-value"><strong>${number(localVo2.value_ml_kg_min, 1)}</strong><span>mL/kg/min</span></p><p class="vo2-range">Likely between ${number(localVo2.value_ml_kg_min - localVo2.uncertainty_95_ml_kg_min, 1)} and ${number(localVo2.value_ml_kg_min + localVo2.uncertainty_95_ml_kg_min, 1)}</p>` : `<p class="vo2-value"><strong>—</strong></p>`}
            <p class="cross-check-caveat">Estimated, not measured. It is calculated from the same pace-and-heart-rate evidence as the trend above, so it moves with that trend rather than confirming it.</p>
            <details><summary>Method and limitations</summary><p>${escapeHtml(localVo2.method)}</p><p>${escapeHtml(localVo2.interpretation)}</p><ul>${(localVo2.limitations ?? []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>${localVo2.demographic_baseline_ml_kg_min ? `<p>Broad demographic baseline for your age, sex, BMI and activity level: ${number(localVo2.demographic_baseline_ml_kg_min, 1)} ± ${number(localVo2.demographic_uncertainty_95_ml_kg_min, 1)} mL/kg/min. That is a population figure, not a measurement of you.</p>` : ""}</details>
          </div>
          <div class="vo2-trend">
            <p class="eyebrow">Last ${fitnessWindowLabel(progressWindow)}</p>
            ${vo2Chart(localVo2.series ?? [])}
            <p class="chart-note">The shaded band is the 95% range. Its width is the honest part of this estimate.</p>
          </div>
        </div>
      </article>
      <article class="wide-card coverage-card"><div class="card-heading"><div><p class="eyebrow">Runs used in this view</p><h2>Last ${fitnessWindowLabel(progressWindow)}</h2></div></div><p>Every activity stays in your history and training load. This table shows whether it also informs the fitness trend.</p><div class="table-scroll"><table><thead><tr><th>Date</th><th>Distance</th><th>Type</th><th>Health</th><th>Adjusted pace</th><th>Fitness trend</th><th>Why</th></tr></thead><tbody>${coverageRows || '<tr><td colspan="7">No activities in this window.</td></tr>'}</tbody></table></div></article>
    </section>`;
  view.querySelectorAll("[data-window]").forEach((button) => button.addEventListener("click", () => { progressWindow = Number(button.dataset.window); renderProgress(); }));
  view.querySelectorAll("[data-metric]").forEach((button) => button.addEventListener("click", () => { progressMetric = button.dataset.metric; renderProgress(); }));
}
