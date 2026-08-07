const view = document.querySelector("#app-view");
const uploadButton = document.querySelector("#upload-button");
const uploadInput = document.querySelector("#upload-input");
const uploadPanel = document.querySelector("#upload-panel");
const uploadClose = document.querySelector("#upload-close");
const uploadSummary = document.querySelector("#upload-summary");
const uploadStages = document.querySelector("#upload-stages");

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

const pace = (value) => {
  if (!Number.isFinite(value)) return "—";
  let minutes = Math.floor(value);
  let seconds = Math.round((value - minutes) * 60);
  if (seconds === 60) { minutes += 1; seconds = 0; }
  return `${minutes}:${String(seconds).padStart(2, "0")}/mi`;
};
const number = (value, digits = 1) => Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
const dateLabel = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "Unknown date";
const calendarDateLabel = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(`${value}T12:00:00`)) : "Unknown date";
const daypartLabel = (value) => {
  if (!value) return null;
  const hour = new Date(value).getHours();
  return hour < 12 ? "Morning" : hour < 17 ? "Afternoon" : "Evening";
};
const titleCase = (value) => String(value ?? "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const durationLabel = (seconds) => {
  if (!Number.isFinite(seconds)) return "—";
  const hours = Math.floor(seconds / 3600); const minutes = Math.floor((seconds % 3600) / 60); const secs = seconds % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${minutes}:${String(secs).padStart(2, "0")}`;
};
const durationSeconds = (value) => {
  if (!value) return null;
  const parts = String(value).split(":").map(Number);
  if (parts.some((part) => !Number.isFinite(part)) || parts.length < 2 || parts.length > 3) return null;
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2];
};
const datetimeLocalValue = (value) => {
  const date = new Date(value);
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `HTTP ${response.status}`);
  return payload;
}

function loading(title = "Loading analysis") {
  view.innerHTML = `<section class="page narrow"><p class="eyebrow">Local analysis</p><h1>${escapeHtml(title)}</h1><div class="loading-line"></div></section>`;
}

async function renderDashboard() {
  loading("Your running data, explained.");
  const dashboard = await api("/api/dashboard");
  const progress = dashboard.progress;
  const interpretation = dashboard.fitness_interpretation;
  const last = dashboard.last_run;
  const recommendation = dashboard.recommendation;
  const weekly = dashboard.weekly_schedule;
  const change = progress.pace_change_seconds_per_mile;
  const nextExtent = weekly
    ? `${number(weekly.projected_distance_range_miles[0])}–${number(weekly.projected_distance_range_miles[1])} mi this week`
    : recommendation.distance_range_miles
    ? `${number(recommendation.distance_range_miles[0])}–${number(recommendation.distance_range_miles[1])} mi`
    : recommendation.duration_range_minutes
      ? `${number(recommendation.duration_range_minutes[0], 0)}–${number(recommendation.duration_range_minutes[1], 0)} min`
      : "No run";
  const lastEasy = last?.run.session_difficulty?.zone_breakdown;
  const easyMinutes = lastEasy ? lastEasy.easy_minutes : 0;
  const knownMinutes = lastEasy ? lastEasy.easy_minutes + lastEasy.moderate_minutes + lastEasy.hard_minutes : 0;
  const signalArrow = (trend) => trend === "improving" ? "↑" : trend === "declining" ? "↓" : ["stable", "uncertain"].includes(trend) ? "↔" : "?";
  const signalRows = interpretation.signals.map((signal) => `<span class="fitness-signal ${signal.trend}"><b>${signalArrow(signal.trend)} ${escapeHtml(signal.label)}: ${escapeHtml(signal.status)}</b><small>${escapeHtml(signal.detail)} · ${titleCase(signal.confidence)} evidence</small></span>`).join("");
  view.innerHTML = `
    <section class="page">
      <div class="hero">
        <p class="eyebrow">Current running state</p>
        <h1>${escapeHtml(interpretation.headline)}</h1>
        <p class="lede">${escapeHtml(interpretation.summary)}</p>
      </div>
      <div class="dashboard-grid">
        <a class="dashboard-card fitness-card" href="#progress"><p class="eyebrow">How am I doing?</p><h2>${progress.current_pace?.display ?? "Not enough data"}</h2><strong>standardized @145 at ${number(progress.reference_within_run_minutes, 0)} minutes</strong><div class="dashboard-stat"><b>${Number.isFinite(change) ? `${change > 0 ? "+" : ""}${change.toFixed(0)} sec/mi` : "—"}</b><span>vs prior ${progress.window_days} days</span></div><div class="dashboard-stat"><b>${number(progress.current_load.trailing_28d.distance_miles)} mi</b><span>last 28 days</span></div><small>${titleCase(progress.fitness_trend)} · ${titleCase(progress.fitness_confidence)} confidence</small></a>
        <a class="dashboard-card" href="${last ? `#run/${last.run.activity_id}` : "#runs"}"><p class="eyebrow">How was my last run?</p><h2>${last ? `${number(last.run.distance_miles)} miles` : "No run yet"}</h2><strong>${last ? `${pace(last.run.moving_pace_min_mile)} · ${number(last.run.average_hr_bpm, 0)} bpm` : "Upload a TCX to begin"}</strong><div class="dashboard-stat"><b>${last ? `${number(knownMinutes ? easyMinutes / knownMinutes * 100 : null, 0)}%` : "—"}</b><span>Z1/Z2 time</span></div><div class="dashboard-stat"><b>${last?.run.fitness_observation?.standardized_pace_at_target_hr.display ?? "—"}</b><span>standardized @145</span></div><small>${escapeHtml(last?.assessment ?? "No feedback available")}</small></a>
        <a class="dashboard-card next-card" href="#next-run"><p class="eyebrow">Your next seven days</p><h2>${weekly ? `${weekly.run_count} planned runs` : escapeHtml(recommendation.title)}</h2><strong>${nextExtent}</strong><div class="dashboard-stat"><b>${weekly ? "By-day timing" : titleCase(recommendation.readiness)}</b><span>${weekly ? "chosen from each forecast" : "readiness"}</span></div><div class="dashboard-stat"><b>${weekly ? weekly.days.filter((day) => ["intervals", "tempo_threshold", "race"].includes(day.recommendation?.workout_type)).length : recommendation.rule_trace.filter((item) => item.fired).length}</b><span>${weekly ? "quality sessions" : "rules fired"}</span></div><small>${escapeHtml(weekly?.summary ?? recommendation.reasons[0] ?? "Open the schedule for its full reasoning.")}</small></a>
      </div>
      <article class="wide-card fitness-context"><p class="eyebrow">Fitness has more than one dimension</p><div class="signal-grid detailed-signals">${signalRows}</div><p>${escapeHtml(interpretation.capacity_summary)}</p>${interpretation.illness_context ? `<p class="context-note">${escapeHtml(interpretation.illness_context)}</p>` : ""}</article>
      <div class="metric-grid dashboard-load"><article><span>7-day distance</span><strong>${number(progress.current_load.trailing_7d.distance_miles)} mi</strong><small>${number(progress.current_load.trailing_7d.moving_minutes, 0)} moving minutes</small></article><article><span>7-day zone load</span><strong>${number(progress.current_load.trailing_7d.zone_load, 0)}</strong><small>${titleCase(progress.current_load.confidence)} HR-load confidence</small></article><article><span>Consistency</span><strong>${number(progress.consistency.runs_per_week, 1)} runs/week</strong><small>${progress.consistency.running_days} running days in 28</small></article><article><span>Longest recent</span><strong>${number(progress.consistency.longest_run_miles)} mi</strong><small>durability, not fitness pace</small></article></div>
    </section>`;
}

let runFilters = { flag: "", sort_by: "date", sort_order: "desc", date_from: "", date_to: "" };
let runPageSize = 100;

async function renderRuns() {
  loading("Runs");
  const query = new URLSearchParams({ limit: String(runPageSize), sort_by: runFilters.sort_by, sort_order: runFilters.sort_order });
  Object.entries(runFilters).forEach(([key, value]) => { if (value && !["sort_by", "sort_order"].includes(key)) query.set(key, value); });
  const runs = await api(`/api/runs?${query}`);
  const rows = runs.map((run) => `
    <a class="run-row" href="#run/${run.activity_id}">
      <span><strong>${dateLabel(run.start_time)}</strong><small>${titleCase(run.workout_type)}</small></span>
      <span><strong>${number(run.distance_miles)} mi</strong><small>${number(run.moving_minutes, 0)} min</small></span>
      <span><strong>${pace(run.moving_pace_min_mile)}</strong><small>${number(run.average_hr_bpm, 0)} avg · ${number(run.maximum_hr_bpm, 0)} max</small></span>
      <span><strong>${run.fitness_observation ? run.fitness_observation.standardized_pace_at_target_hr.display : "—"}</strong><small>standardized @145</small></span>
      <span><strong>${Number.isFinite(run.temperature_f) ? `${number(run.temperature_f, 0)}°F` : "—"}</strong><small>${escapeHtml(run.gps_quality)}</small></span>
      <span><strong>${escapeHtml(run.assessment_label)}</strong><small>${titleCase(run.health_tag)}</small></span>
      <span class="quality ${run.data_quality}">${titleCase(run.data_quality)}</span>
    </a>`).join("");
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">History</p><h1>Runs</h1><p>Showing ${runs.length} activities with fitness and difficulty shown separately.</p></div></div>
      <form id="run-filters" class="filter-bar"><label>Show<select name="flag"><option value="">All activities</option><option value="easy">Easy</option><option value="hard">Hard</option><option value="long">Long</option><option value="illness">Illness / abnormal</option><option value="no_gps">No GPS</option><option value="excluded">Excluded</option></select></label><label>From<input type="date" name="date_from" value="${runFilters.date_from}"></label><label>To<input type="date" name="date_to" value="${runFilters.date_to}"></label><label>Sort<select name="sort_by"><option value="date">Date</option><option value="distance">Distance</option><option value="pace">Moving pace</option><option value="heart_rate">Average HR</option><option value="standardized">Standardized @145</option></select></label><label>Order<select name="sort_order"><option value="desc">Descending</option><option value="asc">Ascending</option></select></label><button type="submit">Apply</button></form>
      <div class="run-table-heading"><span>Run</span><span>Distance</span><span>Raw result</span><span>Fitness evidence</span><span>Conditions</span><span>Assessment</span><span>Quality</span></div>
      <div class="run-table">${rows || '<div class="empty-state">No runs match these filters.</div>'}</div>
      ${runs.length === runPageSize ? '<button id="load-more-runs" class="load-more" type="button">Load 100 more</button>' : ""}
    </section>`;
  const form = view.querySelector("#run-filters");
  form.flag.value = runFilters.flag; form.sort_by.value = runFilters.sort_by; form.sort_order.value = runFilters.sort_order;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runFilters = Object.fromEntries(new FormData(form).entries());
    runPageSize = 100;
    renderRuns();
  });
  view.querySelector("#load-more-runs")?.addEventListener("click", () => { runPageSize += 100; renderRuns(); });
}

let progressWindow = 28;
let progressMetric = "standardized";
let progressWindowInitialized = false;

const fitnessWindowLabel = (days) => ({
  14: "2 weeks", 28: "4 weeks", 42: "6 weeks", 56: "8 weeks",
  90: "3 months", 180: "6 months", 365: "1 year",
}[days] ?? `${days} days`);

function fitnessChart(series, metric, trend7 = [], trend28 = [], domainEnd = null, domainDays = null) {
  const field = metric === "raw" ? "raw_pace_min_mile" : "standardized_pace_min_mile";
  const points = series.filter((item) => Number.isFinite(item[field]));
  if (!points.length) return '<div class="empty-state">No comparable fitness points are available.</div>';
  const timestamps = points.map((item) => new Date(item.start_time).getTime());
  const values = metric === "both"
    ? points.flatMap((item) => [item.standardized_pace_min_mile, item.raw_pace_min_mile]).filter(Number.isFinite)
    : points.map((item) => item[field]);
  const requestedEnd = domainEnd ? new Date(domainEnd).getTime() : null;
  const minX = Number.isFinite(requestedEnd) && Number.isFinite(domainDays) ? requestedEnd - domainDays * 86400000 : Math.min(...timestamps);
  const maxX = Number.isFinite(requestedEnd) && Number.isFinite(domainDays) ? requestedEnd : Math.max(...timestamps);
  const minY = Math.min(...values) - .2; const maxY = Math.max(...values) + .2;
  const x = (value) => 55 + (value - minX) / Math.max(1, maxX - minX) * 810;
  const y = (value) => 25 + (value - minY) / Math.max(.1, maxY - minY) * 250;
  const trendPath = (trend, gapDays) => trend.map((item, index) => {
    const current = new Date(item.as_of).getTime();
    const previous = index ? new Date(trend[index - 1].as_of).getTime() : null;
    const command = !previous || current - previous > gapDays * 86400000 ? "M" : "L";
    return `${command}${x(current).toFixed(1)},${y(item.pace_min_mile).toFixed(1)}`;
  }).join(" ");
  const marks = points.map((item, index) => {
    const cx = x(timestamps[index]); const cy = y(item[field]);
    const uncertainty = metric === "standardized" ? item.uncertainty_95_min_mile : 0;
    const trendWeight = Number.isFinite(item.trend_weight) ? item.trend_weight : 1;
    const estimated = ["device_distance_fallback", "partial_gps_device_distance"].includes(item.measurement_quality) || item.benchmark_quality === "estimated_fixed_time";
    const pointClass = [trendWeight < 1 ? "reduced-evidence" : "", estimated ? "estimated-measurement" : ""].filter(Boolean).join(" ");
    const qualityText = estimated ? ` · ${titleCase(item.measurement_quality)}${item.benchmark_quality === "estimated_fixed_time" ? " · Estimated fixed-time window" : ""}` : "";
    return `${uncertainty ? `<line x1="${cx}" y1="${y(item[field] - uncertainty)}" x2="${cx}" y2="${y(item[field] + uncertainty)}" class="uncertainty-mark"/>` : ""}<a href="#run/${item.activity_id}"><circle cx="${cx}" cy="${cy}" r="4" class="${pointClass}"><title>${dateLabel(item.start_time)} · ${pace(item[field])} · ${number(item.distance_miles)} mi · ${Math.round(trendWeight * 100)}% context weight${qualityText}</title></circle></a>`;
  }).join("");
  const rawMarks = metric === "both" ? points.filter((item) => Number.isFinite(item.raw_pace_min_mile)).map((item) => {
    const cx = x(new Date(item.start_time).getTime()); const cy = y(item.raw_pace_min_mile);
    return `<a href="#run/${item.activity_id}"><rect x="${cx - 3}" y="${cy - 3}" width="6" height="6" class="raw-point"><title>${dateLabel(item.start_time)} · raw ${pace(item.raw_pace_min_mile)}</title></rect></a>`;
  }).join("") : "";
  const dateTick = (value) => new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
  const labels = [minY, (minY + maxY) / 2, maxY].map((value) => `<text x="48" y="${y(value) + 4}" text-anchor="end">${pace(value).replace("/mi", "")}</text><line x1="55" y1="${y(value)}" x2="865" y2="${y(value)}" class="grid-line"/>`).join("") + `<text x="55" y="303" text-anchor="start">${dateTick(minX)}</text><text x="865" y="303" text-anchor="end">${dateTick(maxX)}</text>`;
  const reducedLegend = points.some((item) => Number.isFinite(item.trend_weight) && item.trend_weight < 1) ? '<span><i class="reduced-key"></i>reduced weight</span>' : "";
  const estimatedLegend = points.some((item) => ["device_distance_fallback", "partial_gps_device_distance"].includes(item.measurement_quality) || item.benchmark_quality === "estimated_fixed_time") ? '<span><i class="estimated-key"></i>estimated / higher uncertainty</span>' : "";
  const legend = trend7.length || trend28.length ? `<div class="chart-legend"><span><i class="trend7-key"></i>7-day</span><span><i class="trend28-key"></i>28-day</span><span><i class="point-key"></i>standardized run</span>${reducedLegend}${estimatedLegend}${metric === "both" ? '<span><i class="raw-key"></i>raw run</span>' : ""}</div>` : `<div class="chart-legend"><span><i class="point-key"></i>run</span>${reducedLegend}${estimatedLegend}</div>`;
  return `<div class="chart-wrap"><span class="faster-label">Faster ↑</span>${legend}<svg class="fitness-chart" viewBox="0 0 900 310" role="img" aria-label="Fitness pace over time">${labels}<path d="${trendPath(trend28, 70)}" class="chart-line trend-28"/><path d="${trendPath(trend7, 25)}" class="chart-line trend-7"/>${marks}${rawMarks}</svg></div>`;
}

async function renderProgress() {
  loading("Progress");
  if (!progressWindowInitialized) {
    const settings = await api("/api/settings");
    progressWindow = settings.default_fitness_window;
    progressWindowInitialized = true;
  }
  const progress = await api(`/api/progress?window_days=${progressWindow}`);
  const comparison = progress.period_comparison;
  const change = progress.pace_change_seconds_per_mile;
  const changeUncertainty = progress.pace_change_uncertainty_95_seconds_per_mile;
  const trendText = progress.fitness_trend === "improving" ? "Modeled aerobic efficiency is improving" : progress.fitness_trend === "declining" ? "Modeled aerobic efficiency is down in this window" : "Modeled aerobic efficiency is stable or uncertain";
  const paceChange = Number.isFinite(change) ? `${change > 0 ? "+" : ""}${change.toFixed(0)} sec/mi estimate${Number.isFinite(changeUncertainty) ? ` · ±${changeUncertainty.toFixed(0)} sec uncertainty` : ""}` : "Not enough prior comparable evidence";
  const ratio = progress.current_load.acute_distance_to_capacity_ratio;
  const rawLoadRatio = progress.current_load.acute_to_prior_ratio;
  const external = progress.external_fitness;
  const localVo2 = progress.local_vo2_estimate;
  const externalRows = external.snapshots.slice().reverse().map((item) => `<tr><td>${escapeHtml(item.measured_at)}</td><td>${number(item.vo2_max, 1)}</td><td>${durationLabel(item.predicted_5k_seconds)}</td><td>${escapeHtml(item.source)}</td></tr>`).join("");
  const coverageRows = progress.activity_coverage.map((item) => {
    const label = item.score_status === "trend_evidence" ? "Full weight" : item.score_status === "uncertain_estimate" ? `Estimated · ${Math.round(item.trend_weight * 100)}% context` : item.score_status === "reduced_weight" ? `${Math.round(item.trend_weight * 100)}% context` : item.score_status === "workout_specific" ? "Workout analysis" : item.score_status === "context_only" ? "Context only" : item.score_status === "non_running" ? "Non-running" : "Unscored";
    const quality = item.score_status === "trend_evidence" ? "good" : ["reduced_weight", "uncertain_estimate", "workout_specific"].includes(item.score_status) ? "partial" : "unavailable";
    return `<tr><td><a href="#run/${item.activity_id}">${dateLabel(item.start_time)}</a></td><td>${number(item.distance_miles)} mi</td><td>${titleCase(item.workout_type)}</td><td>${titleCase(item.health_tag)}</td><td>${Number.isFinite(item.standardized_pace_min_mile) ? pace(item.standardized_pace_min_mile) : "—"}</td><td><span class="quality ${quality}">${label}</span></td><td class="coverage-reason">${escapeHtml(item.reason)}</td></tr>`;
  }).join("");
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">Progress analysis</p><h1>${trendText}</h1><p>${escapeHtml(progress.definition)}</p></div><div class="headline-pace"><strong>${progress.current_pace?.display ?? "—"}</strong><span>${progress.uncertainty_95_min_mile ? `±${Math.round(progress.uncertainty_95_min_mile * 60)} sec/mi` : "Unavailable"}</span></div></div>
      <div class="toolbar"><div class="segmented" aria-label="Fitness time frame">${progress.available_windows.map((days) => `<button type="button" data-window="${days}" class="${days === progressWindow ? "selected" : ""}">${fitnessWindowLabel(days)}</button>`).join("")}</div><div class="segmented"><button type="button" data-metric="standardized" class="${progressMetric === "standardized" ? "selected" : ""}">Standardized @145</button><button type="button" data-metric="raw" class="${progressMetric === "raw" ? "selected" : ""}">Raw @145</button><button type="button" data-metric="both" class="${progressMetric === "both" ? "selected" : ""}">Both</button></div></div>
      <article class="wide-card chart-card"><div class="card-heading"><div><p class="eyebrow">Primary fitness estimate · last ${fitnessWindowLabel(progressWindow)}</p><h2>${titleCase(progressMetric)} pace at reference HR</h2></div><span class="quality ${progress.fitness_confidence}">${titleCase(progress.fitness_confidence)} confidence</span></div>${fitnessChart(progress.series, progressMetric, progressMetric !== "raw" ? progress.trend_7d : [], progressMetric !== "raw" ? progress.trend_28d : [], progress.as_of, progressWindow)}<p class="chart-note">Showing only runs from the selected ${fitnessWindowLabel(progressWindow)}. The model robustly combines all usable overlapping two-minute windows in speed space, weighting HR/time relevance and transition stability. Whiskers show run-level uncertainty.</p></article>
      <div class="metric-grid progress-metrics">
        <article><span>Modeled fitness change</span><strong>${paceChange}</strong><small>${titleCase(progress.fitness_trend)} · ${titleCase(progress.fitness_confidence)} confidence</small></article>
        <article><span>7-day running</span><strong>${number(progress.current_load.trailing_7d.distance_miles)} mi</strong><small>${number(progress.current_load.trailing_7d.zone_load, 0)} load points</small></article>
        <article><span>7-day / retained capacity</span><strong>${number(ratio, 2)}×</strong><small>${number(progress.current_load.trailing_7d.distance_miles)} mi vs ${number(progress.current_load.capacity_reference_miles)} demonstrated · raw HR-load/prior ${number(rawLoadRatio, 2)}×</small></article>
        <article><span>Longest run</span><strong>${number(progress.consistency.longest_run_miles)} mi</strong><small>${number(progress.consistency.runs_per_week, 1)} runs/week</small></article>
      </div>
      <div class="two-column">
        <article class="wide-card"><p class="eyebrow">Period comparison</p><h2>Last ${fitnessWindowLabel(progressWindow)} vs preceding ${fitnessWindowLabel(progressWindow)}</h2><div class="comparison-grid"><span>Distance<strong>${number(comparison.current.distance_miles)} / ${number(comparison.previous.distance_miles)} mi</strong></span><span>Moving time<strong>${number(comparison.current.moving_minutes, 0)} / ${number(comparison.previous.moving_minutes, 0)} min</strong></span><span>Zone load<strong>${number(comparison.current.zone_load, 0)} / ${number(comparison.previous.zone_load, 0)}</strong></span><span>Run count<strong>${comparison.current.run_count} / ${comparison.previous.run_count}</strong></span></div><p>${escapeHtml(comparison.interpretation)}</p></article>
        <article class="wide-card"><p class="eyebrow">Intensity distribution</p><h2>${number(progress.intensity.easy_percent, 0)}% easy</h2><div class="intensity-bar"><b style="width:${progress.intensity.easy_percent ?? 0}%"></b><i style="width:${progress.intensity.moderate_percent ?? 0}%"></i><em style="width:${progress.intensity.hard_percent ?? 0}%"></em></div><p>${number(progress.intensity.known_hr_minutes, 0)} known HR minutes · ${number(progress.intensity.missing_hr_minutes, 0)} missing</p><small>${progress.consistency.quality_sessions} quality sessions · ${progress.consistency.running_days} running days</small></article>
      </div>
      <div class="two-column vo2-grid"><article class="wide-card"><div class="card-heading"><div><p class="eyebrow">Local equation cross-check</p><h2>${number(localVo2.value_ml_kg_min, 1)} mL/kg/min</h2></div><span class="quality ${localVo2.confidence}">${titleCase(localVo2.confidence)} confidence</span></div><p>Approximate 95% method range: ${Number.isFinite(localVo2.value_ml_kg_min) ? `${number(localVo2.value_ml_kg_min - localVo2.uncertainty_95_ml_kg_min, 1)}–${number(localVo2.value_ml_kg_min + localVo2.uncertainty_95_ml_kg_min, 1)}` : "unavailable"}</p><p>${escapeHtml(localVo2.interpretation)}</p><small>${escapeHtml(localVo2.method)}. Demographic baseline: ${number(localVo2.demographic_baseline_ml_kg_min, 1)} ± ${number(localVo2.demographic_uncertainty_95_ml_kg_min, 1)}.</small></article><article class="wide-card external-fitness"><div class="card-heading"><div><p class="eyebrow">Independent device estimates</p><h2>Garmin VO₂ max & race prediction</h2></div><span class="quality ${external.confidence}">${titleCase(external.confidence)} confidence</span></div><p>${escapeHtml(external.interpretation)}</p><form id="external-fitness-form" class="inline-metric-form"><label>Date<input required type="date" name="measured_at" value="${new Date().toISOString().slice(0, 10)}"></label><label>Running VO₂ max<input type="number" step="0.1" min="10" max="100" name="vo2_max" placeholder="e.g. 47"></label><label>Predicted 5K<input type="text" name="predicted_5k" placeholder="MM:SS"></label><button type="submit">Save Garmin snapshot</button><span id="external-status"></span></form>${externalRows ? `<div class="table-scroll"><table><thead><tr><th>Date</th><th>VO₂ max</th><th>Predicted 5K</th><th>Source</th></tr></thead><tbody>${externalRows}</tbody></table></div>` : ""}<small>TCX files do not contain Garmin's historical VO₂ max or race-predictor series. Enter occasional snapshots here; they remain visibly separate from this app's own metric.</small></article></div>
      <article class="wide-card coverage-card"><div class="card-heading"><div><p class="eyebrow">Activity coverage</p><h2>Every activity in the last ${fitnessWindowLabel(progressWindow)}</h2></div><span class="quality moderate">Weighted evidence</span></div><p>Illness and recovery runs remain fitness evidence at reduced weight and still count fully toward distance and training load. This table explains every missing or downweighted point.</p><div class="table-scroll"><table><thead><tr><th>Date</th><th>Distance</th><th>Type</th><th>Health</th><th>Standardized</th><th>Trend use</th><th>Reason</th></tr></thead><tbody>${coverageRows || '<tr><td colspan="7">No activities in this window.</td></tr>'}</tbody></table></div></article>
    </section>`;
  view.querySelectorAll("[data-window]").forEach((button) => button.addEventListener("click", () => { progressWindow = Number(button.dataset.window); renderProgress(); }));
  view.querySelectorAll("[data-metric]").forEach((button) => button.addEventListener("click", () => { progressMetric = button.dataset.metric; renderProgress(); }));
  view.querySelector("#external-fitness-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const status = event.currentTarget.querySelector("#external-status");
    const vo2 = values.get("vo2_max");
    const predicted = durationSeconds(values.get("predicted_5k"));
    if (!vo2 && !predicted) { status.textContent = "Enter VO₂ max or a predicted 5K."; return; }
    status.textContent = "Saving…";
    try {
      await api("/api/external-fitness", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ measured_at: values.get("measured_at"), vo2_max: vo2 ? Number(vo2) : null, predicted_5k_seconds: predicted, source: "Garmin" }) });
      await renderProgress();
    } catch (error) { status.textContent = error.message; }
  });
}

function renderAdjustment(observation, workoutType) {
  if (!observation) {
    const quality = ["intervals", "tempo_threshold", "race"].includes(workoutType);
    return `<article class="wide-card"><p class="eyebrow">Fitness observation</p><h2>${quality ? "Kept out of steady aerobic @145" : "Not scoreable"}</h2><p>${quality ? "This quality session still counts fully toward training load and receives workout-specific analysis below. Its changing efforts are not treated as a steady aerobic observation." : "This run remains in load and history, but sensor/weather/comparability requirements did not support a standardized pace."}</p></article>`;
  }
  const sign = observation.environmental_adjustment_min_mile >= 0 ? "+" : "−";
  const seconds = Math.abs(observation.environmental_adjustment_min_mile * 60);
  const contributions = observation.contributions.map((item) => `
    <li><span>${titleCase(item.name)}</span><strong>${item.minutes_per_mile >= 0 ? "+" : "−"}${Math.abs(item.minutes_per_mile * 60).toFixed(0)} sec/mi</strong><small>${titleCase(item.confidence)} · ${escapeHtml(item.evidence)}</small></li>`).join("");
  return `<article class="wide-card audit-card">
    <div class="card-heading"><div><p class="eyebrow">Comparable fitness observation</p><h2>Raw → adjusted → standardized</h2></div><span class="quality ${observation.confidence}">${titleCase(observation.confidence)} confidence</span></div>
    <div class="equation"><span><strong>${observation.raw_pace_at_target_hr.display}</strong><small>raw pace @145</small></span><b>${sign} ${seconds.toFixed(0)} sec/mi</b><span><strong>${observation.standardized_pace_at_target_hr.display}</strong><small>standardized @145</small></span></div>
    <details><summary>Inspect every contribution</summary><ul class="contribution-list">${contributions}</ul></details>
  </article>`;
}

function renderWorkoutAnalysis(analysis) {
  if (!analysis) return "";
  const dimensions = [["Execution", analysis.execution], ["Control", analysis.control], ["Stimulus", analysis.stimulus], ["Recovery", analysis.recovery]];
  const cards = dimensions.map(([name, item]) => `<article class="analysis-dimension"><div class="card-heading"><h3>${name}</h3><span class="quality ${item.confidence}">${escapeHtml(item.status)}</span></div><p>${escapeHtml(item.summary)}</p><ul>${item.metrics.map((metric) => `<li><span>${escapeHtml(metric.name)}</span><strong>${escapeHtml(metric.value)}</strong><small>${escapeHtml(metric.detail)}</small></li>`).join("")}</ul></article>`).join("");
  const intervals = analysis.interval_analysis;
  let intervalTable = "";
  if (intervals) {
    const workNumbers = new Map(); let workNumber = 0; let recoveryNumber = 0;
    intervals.repetitions.forEach((item) => { if (item.kind === "work") workNumbers.set(item.index, ++workNumber); if (item.kind === "recovery") recoveryNumber += 1; });
    recoveryNumber = 0;
    const rows = intervals.repetitions.map((item) => {
      const phase = item.kind === "work" ? `Rep ${workNumbers.get(item.index)}` : item.kind === "recovery" ? `Recovery ${++recoveryNumber}` : titleCase(item.kind);
      const recovery = item.kind === "work" && Number.isFinite(item.recovery_after_seconds) ? durationLabel(Math.round(item.recovery_after_seconds)) : "—";
      const hrRecovery = item.kind === "work" && Number.isFinite(item.recovery_hr_drop_bpm) ? `${number(item.recovery_start_hr_bpm, 0)}→${number(item.recovery_min_hr_bpm, 0)} (−${number(item.recovery_hr_drop_bpm, 0)})` : "—";
      return `<tr class="interval-${item.kind}"><td>${phase}</td><td>${durationLabel(Math.round(item.duration_seconds))}</td><td>${number(item.distance_miles, 2)} mi</td><td>${pace(item.pace_min_mile)}</td><td>${number(item.average_hr_bpm, 0)}</td><td>${number(item.end_hr_bpm, 0)} / ${number(item.maximum_hr_bpm, 0)}</td><td>${number(item.average_cadence_spm, 0)}</td><td>${recovery}</td><td>${hrRecovery}</td></tr>`;
    }).join("");
    intervalTable = `<article class="wide-card interval-card"><div class="card-heading"><div><p class="eyebrow">Reconstructed workout</p><h2>Warmup → work → recovery → cooldown</h2></div><span class="quality ${intervals.confidence}">${intervals.source === "recorded_laps" ? "Recorded Garmin laps" : "Inferred boundaries"}</span></div><p>${escapeHtml(intervals.explanation)}</p>${intervals.available ? `<div class="table-scroll"><table><thead><tr><th>Phase</th><th>Time</th><th>Distance</th><th>Pace</th><th>Avg HR</th><th>End / max HR</th><th>Cadence</th><th>Recovery</th><th>HR recovery</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<p>${escapeHtml(intervals.explanation)}</p>`}</article>`;
  }
  const comparison = analysis.historical_comparison ? `<article class="wide-card"><p class="eyebrow">Like-for-like history</p><h2>${analysis.historical_comparison.available ? `Compared with ${dateLabel(analysis.historical_comparison.date)}` : "No close match yet"}</h2><p>${escapeHtml(analysis.historical_comparison.summary)}</p>${analysis.historical_comparison.metrics.length ? `<ul class="analysis-metrics">${analysis.historical_comparison.metrics.map((item) => `<li><span>${escapeHtml(item.name)}</span><strong>${escapeHtml(item.value)}</strong><small>${escapeHtml(item.detail)}</small></li>`).join("")}</ul>` : ""}</article>` : "";
  return `<section class="workout-analysis"><div class="card-heading"><div><p class="eyebrow">Workout-specific analysis</p><h2>Four answers, no composite score</h2></div><small>${escapeHtml(analysis.definition)}</small></div><div class="analysis-grid">${cards}</div>${intervalTable}${comparison}${analysis.progression_recommendation ? `<article class="wide-card progression"><p class="eyebrow">Actionable next step</p><h2>${escapeHtml(analysis.progression_recommendation)}</h2></article>` : ""}</section>`;
}

async function renderRunDetail(activityId) {
  loading("Run feedback");
  const feedback = await api(`/api/runs/${activityId}`);
  const run = feedback.run;
  const difficulty = run.session_difficulty;
  const zoneEntries = Object.entries(difficulty.zone_breakdown.zone_fractions).filter(([, value]) => value > 0);
  const zoneBars = zoneEntries.map(([zone, fraction]) => `<div class="zone-row"><span>${titleCase(zone)}</span><i><b style="width:${Math.min(100, fraction * 100)}%"></b></i><strong>${Math.round(fraction * 100)}%</strong></div>`).join("");
  const splits = feedback.splits.map((split) => `<tr><td>${split.index}${split.is_partial ? "*" : ""}</td><td>${number(split.distance_miles, 2)} mi</td><td>${pace(split.pace_min_mile)}</td><td>${number(split.average_hr_bpm, 0)} bpm</td><td>${split.elevation_change_feet >= 0 ? "+" : ""}${number(split.elevation_change_feet, 0)} ft</td></tr>`).join("");
  const cautions = feedback.cautions.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const positives = feedback.positives.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const weather = feedback.weather ? `${number(feedback.weather.temperature_f, 0)}°F · dew point ${number(feedback.weather.dewpoint_f, 0)}°F · wind ${number(feedback.weather.wind_speed_mph, 0)} mph` : "Weather unavailable";
  const postalField = run.gps_quality.includes("missing")
    ? `<label>Run ZIP code<input name="postal_code" inputmode="numeric" pattern="[0-9]{5}" maxlength="5" value="${escapeHtml(feedback.metadata.postal_code ?? "")}" placeholder="e.g. 11211"><small>${feedback.metadata.location_label ? `Weather area: ${escapeHtml(feedback.metadata.location_label)}.` : "Optional. Sent to Open-Meteo geocoding, then rounded and privacy-jittered for weather."}</small></label>`
    : "";
  view.innerHTML = `
    <section class="page">
      <a class="back-link" href="#runs">← All runs</a>
      <div class="page-heading run-title"><div><p class="eyebrow">${titleCase(run.workout_type)} · ${titleCase(run.data_quality)} data</p><h1>${number(run.distance_miles)} miles</h1><p>${dateLabel(run.start_time)}</p></div><div class="headline-pace"><strong>${pace(run.moving_pace_min_mile)}</strong><span>${number(run.average_hr_bpm, 0)} bpm average</span></div></div>
      <article class="assessment"><p class="eyebrow">Assessment</p><h2>${escapeHtml(feedback.assessment)}</h2><div class="feedback-columns"><ul class="positive-list">${positives}</ul><ul class="caution-list">${cautions}</ul></div></article>
      <details class="metadata-editor" open><summary>Add effort or edit workout context</summary><form id="metadata-form"><label>Workout type<select name="workout_type">${["easy","recovery","long","tempo_threshold","intervals","race","run_walk","hike","bike","other","unknown"].map((value) => `<option value="${value}" ${feedback.metadata.workout_type === value ? "selected" : ""}>${titleCase(value)}</option>`).join("")}</select></label><label>Health context<select name="health_tag">${["normal","illness","illness_recovery","injury_affected","other_abnormal"].map((value) => `<option value="${value}" ${feedback.metadata.health_tag === value ? "selected" : ""}>${titleCase(value)}</option>`).join("")}</select></label><label>RPE (1–10)<input type="number" min="1" max="10" name="perceived_exertion" value="${feedback.metadata.perceived_exertion ?? ""}" placeholder="Optional"></label><label>Model inclusion<select name="include_in_model"><option value="auto" ${feedback.metadata.include_in_model === null ? "selected" : ""}>Automatic</option><option value="true" ${feedback.metadata.include_in_model === true ? "selected" : ""}>Include</option><option value="false" ${feedback.metadata.include_in_model === false ? "selected" : ""}>Exclude</option></select></label>${postalField}<label class="notes-label">Notes<textarea name="notes" rows="2">${escapeHtml(feedback.metadata.notes)}</textarea></label><button type="submit">Save and recalculate</button><span id="metadata-status"></span></form></details>
      ${renderAdjustment(run.fitness_observation, run.workout_type)}
      ${renderWorkoutAnalysis(feedback.workout_analysis)}
      <div class="metric-grid">
        <article><span>Distance</span><strong>${number(difficulty.distance_miles)} mi</strong><small>${difficulty.is_long_run ? "Long-run context" : "Session volume"}</small></article>
        <article><span>Moving / elapsed</span><strong>${number(difficulty.moving_minutes, 0)} / ${number(difficulty.elapsed_minutes, 0)} min</strong><small>${number(difficulty.stopped_minutes, 1)} stopped</small></article>
        <article><span>Intensity load</span><strong>${number(difficulty.zone_load, 0)}</strong><small>time-in-zone points${difficulty.session_rpe_load == null ? "" : ` · sRPE ${number(difficulty.session_rpe_load, 0)}`}</small></article>
        <article><span>Prior 7-day load</span><strong>${number(feedback.load_context_before_run?.trailing_7d.distance_miles)} mi</strong><small>${number(feedback.load_context_before_run?.trailing_7d.zone_load, 0)} load points</small></article>
      </div>
      <div class="two-column">
        <article class="wide-card"><p class="eyebrow">Heart-rate distribution</p><h2>Zones</h2>${zoneBars || "<p>Heart-rate coverage unavailable.</p>"}</article>
        <article class="wide-card"><p class="eyebrow">Conditions & drift</p><h2>${weather}</h2><p>${feedback.cardiac_drift.valid ? `${number(feedback.cardiac_drift.decoupling_percent)}% decoupling` : "Drift not evaluated"}</p><small>${escapeHtml(feedback.cardiac_drift.reason)}</small></article>
      </div>
      <article class="wide-card"><p class="eyebrow">Raw-interval splits</p><h2>Mile splits</h2><div class="table-scroll"><table><thead><tr><th>Mile</th><th>Distance</th><th>Pace</th><th>HR</th><th>Elevation</th></tr></thead><tbody>${splits || '<tr><td colspan="5">No reliable splits.</td></tr>'}</tbody></table></div></article>
    </section>`;
  view.querySelector("#metadata-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const status = event.currentTarget.querySelector("#metadata-status");
    const inclusion = values.get("include_in_model");
    const payload = {
      workout_type: values.get("workout_type"), health_tag: values.get("health_tag"),
      perceived_exertion: values.get("perceived_exertion") ? Number(values.get("perceived_exertion")) : null,
      include_in_model: inclusion === "auto" ? null : inclusion === "true", notes: values.get("notes"),
    };
    if (values.has("postal_code")) payload.postal_code = String(values.get("postal_code") || "").trim() || null;
    status.textContent = "Recalculating…";
    try {
      await api(`/api/runs/${activityId}/metadata`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      await renderRunDetail(activityId);
    } catch (error) { status.textContent = error.message; }
  });
}

function renderPlaceholder(route) {
  view.innerHTML = `<section class="page narrow"><p class="eyebrow">In progress</p><h1>${titleCase(route)}</h1><p class="lede">The backend contract is ready. This page is being connected in the next implementation phase.</p></section>`;
}

function weeklyScheduleMarkup(schedule) {
  if (!schedule) return `<article class="wide-card empty-state">Choose your current health status, then generate the next seven days.</article>`;
  const dayCards = schedule.days.map((day) => {
    const result = day.recommendation;
    const label = day.planned_at
      ? `${calendarDateLabel(day.date)} · ${daypartLabel(day.planned_at)}`
      : calendarDateLabel(day.date);
    if (day.completed_activities?.length) {
      const completed = day.completed_activities.map((activity) => `<a href="#run/${activity.activity_id}"><strong>${number(activity.distance_miles)} mi · ${titleCase(activity.workout_type)}</strong><small>${titleCase(activity.health_tag)}</small></a>`).join("");
      return `<article class="schedule-day completed-day"><p class="eyebrow">${label} · ${titleCase(day.day_role)}</p><h2>Run completed</h2>${completed}<p>${escapeHtml(day.rationale)}</p></article>`;
    }
    if (!result) return `<article class="schedule-day rest-day"><p class="eyebrow">${label} · Rest / recovery day</p><h2>No scheduled run</h2><p>${escapeHtml(day.rationale)}</p></article>`;
    const extent = result.distance_range_miles
      ? `${number(result.distance_range_miles[0])}–${number(result.distance_range_miles[1])} miles`
      : result.duration_range_minutes
        ? `${number(result.duration_range_minutes[0], 0)}–${number(result.duration_range_minutes[1], 0)} minutes`
        : "No run";
    const weather = result.planned_weather;
    const weatherLine = weather
      ? `${number(weather.temperature_f, 0)}°F · feels like ${number(weather.apparent_temperature_f, 0)}°F · dew point ${number(weather.dewpoint_f, 0)}°F · wind ${number(weather.wind_speed_mph, 0)} mph`
      : "Forecast unavailable; projected recovery and load were still evaluated.";
    const steps = result.structure.map((step) => `<li>${escapeHtml(step.instruction)} <small>${step.target_zones.map(escapeHtml).join(" · ")}</small></li>`).join("");
    const reasons = result.reasons.map(escapeHtml).join(" ");
    const scoring = result.rule_trace.find((item) => item.rule_id === "workout_scoring");
    const scoreText = scoring ? `Easy ${number(scoring.facts.easy_score, 1)} · Long ${number(scoring.facts.long_score, 1)} · Quality ${number(scoring.facts.quality_score, 1)}` : "Guardrail decision";
    const readinessExplanation = result.readiness !== "ready" && result.readiness_reason
      ? `<details><summary>Why ${titleCase(result.readiness)}?</summary><p>${escapeHtml(result.readiness_reason)}</p></details>`
      : "";
    return `<article class="schedule-day ${result.readiness}"><div class="card-heading"><div><p class="eyebrow">${label} · ${titleCase(day.day_role)}</p><h2>${escapeHtml(result.title)}</h2></div><span class="quality ${result.confidence}">${titleCase(result.readiness)}</span></div><strong class="prescription-extent">${extent}</strong><p>${result.target_zones.map(escapeHtml).join(" · ") || "Recovery"}</p><small>${escapeHtml(weatherLine)}</small>${steps ? `<details><summary>Workout structure</summary><ol>${steps}</ol></details>` : ""}${readinessExplanation}<details><summary>Why this day?</summary><p>${reasons}</p><small>${escapeHtml(day.rationale)} ${escapeHtml(scoreText)}</small></details></article>`;
  }).join("");
  const trailingCards = schedule.trailing_days.map((day) => {
    const activities = day.activities.map((activity) => `<a href="#run/${activity.activity_id}"><strong>${number(activity.distance_miles)} mi · ${titleCase(activity.workout_type)}</strong><small>${titleCase(activity.health_tag)}</small></a>`).join("");
    return `<article class="trailing-day ${day.activities.length ? "active-day" : "rest-day"}"><p class="eyebrow">${calendarDateLabel(day.date)}</p><h2>${titleCase(day.day_role)}</h2>${activities || "<small>No recorded activity</small>"}</article>`;
  }).join("");
  const evidence = schedule.target_evidence;
  const referenceText = evidence.capacity_reference_miles > 0
    ? `Demonstrated weekly capacity reference: about ${number(evidence.demonstrated_run_days_per_week, 1)} run days · ${number(schedule.target_distance_range_miles[0])}–${number(schedule.target_distance_range_miles[1])} miles. Evidence: ${number(evidence.recent_7d_miles)} recent · ${number(evidence.chronic_42d_weekly_miles)} chronic · ${number(evidence.best_sustained_28d_weekly_miles)} sustained-peak mi/week.`
    : `Starter placeholder: ${schedule.target_run_count} easy run days · ${number(schedule.target_distance_range_miles[0])}–${number(schedule.target_distance_range_miles[1])} miles. Upload training history before treating this as a personalized target.`;
  return `<article class="wide-card schedule-summary"><div><p class="eyebrow">Tomorrow forward · ${escapeHtml(schedule.start_date)} through ${escapeHtml(schedule.end_date)}</p><h2>${schedule.completed_run_count ? `${schedule.completed_run_count} completed · ` : ""}${schedule.run_count} planned runs · ${number(schedule.projected_distance_range_miles[0])}–${number(schedule.projected_distance_range_miles[1])} planned miles</h2><p>${escapeHtml(schedule.summary)}</p><small>${escapeHtml(referenceText)}</small><details><summary>How was this reference inferred?</summary><p>${escapeHtml(evidence.rationale)}</p></details></div></article><div class="week-grid">${dayCards}</div><article class="wide-card trailing-summary"><p class="eyebrow">Completed context · through today</p><h2>Previous seven days</h2><div class="week-grid trailing-week">${trailingCards}</div></article>`;
}

async function renderNextRun() {
  loading("Week Plan");
  const [status, latest, state] = await Promise.all([
    api("/api/current-status"), api("/api/weekly-schedule/latest"), api("/api/fitness-state"),
  ]);
  const choices = [
    ["normal", "Normal"], ["little_tired", "A little tired"],
    ["sick_or_recovering", "Sick / recovering"], ["pain_or_injury_concern", "Pain / injury concern"],
  ];
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">Leading seven-day schedule</p><h1>Plan the week as a sequence.</h1><p>The Python planner automatically chooses run days and coordinates easy, long, quality, recovery, and rest. Consecutive days are allowed when the sequence is intentional. The plan refreshes when your recorded training or health input changes.</p></div></div>
      <form id="health-form" class="health-form">
        <fieldset><legend>How do you feel now?</legend><div class="health-options">${choices.map(([value, label]) => `<label><input type="radio" name="health_status" value="${value}" ${status.health_status === value ? "checked" : ""}><span>${label}</span></label>`).join("")}</div></fieldset>
        <label class="notes-label">Optional context<textarea name="notes" rows="2" placeholder="Symptoms, unusual fatigue, soreness, or anything the TCX cannot know">${escapeHtml(status.notes)}</textarea></label>
        <button class="primary-button" type="submit">Regenerate next seven days</button>
      </form>
      <div id="recommendation-result">${weeklyScheduleMarkup(latest)}</div>
      <div class="state-strip"><span><b>${number(state.recent_load.trailing_7d.distance_miles)} mi</b>last 7 days</span><span><b>${number(state.recent_load.acute_distance_to_capacity_ratio, 2)}×</b>mileage / retained capacity</span><span><b>${number((state.moderate_fraction_14d ?? 0) * 100, 0)}%</b>unplanned Z3 · ${state.moderate_evidence_runs_14d} eligible run${state.moderate_evidence_runs_14d === 1 ? "" : "s"}</span><span><b>${state.running_days_28d}</b>running days in 28</span></div>
    </section>`;
  view.querySelector("#health-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const button = event.currentTarget.querySelector("button");
    button.disabled = true; button.textContent = "Evaluating rules…";
    try {
      const result = await api("/api/weekly-schedule", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ health_status: form.get("health_status"), notes: form.get("notes") }),
      });
      view.querySelector("#recommendation-result").innerHTML = weeklyScheduleMarkup(result);
    } catch (error) {
      view.querySelector("#recommendation-result").innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    } finally { button.disabled = false; button.textContent = "Regenerate next seven days"; }
  });
}

async function renderSettings() {
  loading("Settings");
  const settings = await api("/api/settings");
  const zoneInputs = Object.entries(settings.zones).map(([name, zone]) => `<div class="zone-input"><strong>${name.toUpperCase()}</strong><input type="number" name="${name}_min" value="${zone.minimum_bpm}"><span>to</span><input type="number" name="${name}_max" value="${zone.maximum_bpm}"><span>bpm</span></div>`).join("");
  const qualitySessionInputs = Object.entries(settings.coaching.quality_sessions).map(([name, enabled]) => `<label><input type="checkbox" name="quality_${name}" ${enabled ? "checked" : ""}>${titleCase(name)}</label>`).join("");
  view.innerHTML = `
    <section class="page settings-page">
      <div class="page-heading"><div><p class="eyebrow">Local configuration</p><h1>Settings</h1><p>Edits are saved to <code>config.local.yaml</code>; the documented base config remains intact.</p></div></div>
      <form id="settings-form"><div class="settings-grid">
        <fieldset><legend>Physiology</legend><label>Maximum HR<input type="number" name="max_hr" value="${settings.max_hr}"></label><label>Resting HR<input type="number" name="resting_hr" value="${settings.resting_hr}"></label><label>Standardized target HR<input type="number" name="target_hr" value="${settings.target_hr}"></label></fieldset>
        <fieldset><legend>VO₂ estimate profile</legend><label>Birth date<input type="date" name="profile_birth_date" value="${settings.profile?.birth_date ?? ""}"></label><label>Equation sex<select name="profile_sex"><option value="male" ${settings.profile?.sex === "male" ? "selected" : ""}>Male</option><option value="female" ${settings.profile?.sex === "female" ? "selected" : ""}>Female</option></select></label><label>Weight lb<input type="number" step="0.1" name="profile_weight_lb" value="${settings.profile?.weight_lb ?? ""}"></label><label>Height in<input type="number" step="0.1" name="profile_height_in" value="${settings.profile?.height_in ?? ""}"></label><p>Used only for published local cross-checks. It does not reconstruct Garmin's proprietary estimate.</p></fieldset>
        <fieldset><legend>Running zones</legend>${zoneInputs}</fieldset>
        <fieldset><legend>Reference conditions and weather privacy</legend><label>Temperature °F<input type="number" step="0.1" name="reference_temperature_f" value="${settings.reference_temperature_f}"></label><label>Dew point °F<input type="number" step="0.1" name="reference_dewpoint_f" value="${settings.reference_dewpoint_f}"></label><label>Wind mph<input type="number" step="0.1" name="reference_wind_mph" value="${settings.reference_wind_mph}"></label><label>Grade %<input type="number" step="0.1" name="reference_grade_percent" value="${settings.reference_grade_percent}"></label><label>Standardized at minute<input type="number" step="0.5" name="reference_within_run_minutes" value="${settings.reference_within_run_minutes}"></label><label>Weather privacy radius km<input type="number" step="0.1" name="weather_privacy_radius_km" value="${settings.weather_privacy_radius_km}"></label><label><input type="checkbox" name="historical_weather_enabled" ${settings.historical_weather_enabled ? "checked" : ""}>Retrieve historical run weather</label><p>Enabling historical retrieval sends run dates and rounded, privacy-jittered route centroids to Open-Meteo.</p><label><input type="checkbox" name="forecast_weather_enabled" ${settings.forecast_weather_enabled ? "checked" : ""}>Retrieve planned-run forecasts</label><p>Enabling forecast retrieval sends the privacy-jittered recent-route centroid and planned timestamp to Open-Meteo.</p></fieldset>
        <fieldset><legend>Moving time</legend>${Object.entries(settings.moving_time).map(([name, value]) => `<label>${titleCase(name)}<input type="number" step="0.01" name="moving_${name}" value="${value}"></label>`).join("")}</fieldset>
        <fieldset><legend>Coaching rules</legend><label>Training goal<select name="training_goal"><option value="general_fitness">General fitness</option></select></label>${Object.entries(settings.coaching).filter(([name]) => !["training_goal", "quality_sessions"].includes(name)).map(([name, value]) => `<label>${titleCase(name)}<input type="number" step="0.01" name="coaching_${name}" value="${value}"></label>`).join("")}</fieldset>
        <fieldset><legend>Quality sessions</legend>${qualitySessionInputs}<p>Disabled types will not be prescribed. At least one type must remain enabled.</p></fieldset>
        <fieldset><legend>Display</legend><label>Default fitness window<select name="default_fitness_window">${settings.available_fitness_windows.map((value) => `<option value="${value}" ${value === settings.default_fitness_window ? "selected" : ""}>${fitnessWindowLabel(value)}</option>`).join("")}</select></label><p>Physiology, zone, reference-condition, or movement changes recalculate affected analysis. Coaching-only changes do not refit fitness.</p></fieldset>
      </div><div class="settings-actions"><button class="primary-button" type="submit">Save settings</button><span id="settings-status"></span></div></form>
    </section>`;
  const form = view.querySelector("#settings-form");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form); const numeric = (name) => Number(data.get(name));
    const payload = {
      max_hr: numeric("max_hr"), resting_hr: numeric("resting_hr"), target_hr: numeric("target_hr"),
      zones: Object.fromEntries(["z1","z2","z3","z4","z5"].map((name) => [name, { minimum_bpm: numeric(`${name}_min`), maximum_bpm: numeric(`${name}_max`) }])),
      reference_temperature_f: numeric("reference_temperature_f"), reference_dewpoint_f: numeric("reference_dewpoint_f"), reference_wind_mph: numeric("reference_wind_mph"), reference_grade_percent: numeric("reference_grade_percent"), reference_within_run_minutes: numeric("reference_within_run_minutes"), weather_privacy_radius_km: numeric("weather_privacy_radius_km"), historical_weather_enabled: data.get("historical_weather_enabled") === "on", forecast_weather_enabled: data.get("forecast_weather_enabled") === "on", default_fitness_window: numeric("default_fitness_window"),
      moving_time: Object.fromEntries(Object.keys(settings.moving_time).map((name) => [name, numeric(`moving_${name}`)])),
      coaching: {
        ...Object.fromEntries(Object.keys(settings.coaching).filter((name) => !["training_goal", "quality_sessions"].includes(name)).map((name) => [name, numeric(`coaching_${name}`)])),
        training_goal: data.get("training_goal"),
        quality_sessions: Object.fromEntries(Object.keys(settings.coaching.quality_sessions).map((name) => [name, data.get(`quality_${name}`) === "on"])),
      },
    };
    const status = form.querySelector("#settings-status"); const button = form.querySelector("button[type=submit]");
    const profileBirthDate = data.get("profile_birth_date");
    const profileWeight = data.get("profile_weight_lb");
    const profileHeight = data.get("profile_height_in");
    const profileHasAnyValue = Boolean(profileBirthDate || profileWeight || profileHeight);
    if (profileHasAnyValue && !(profileBirthDate && profileWeight && profileHeight)) {
      status.textContent = "Complete birth date, weight, and height together, or leave the optional VO₂ profile blank.";
      return;
    }
    if (profileHasAnyValue) {
      payload.profile = { birth_date: profileBirthDate, sex: data.get("profile_sex"), weight_lb: Number(profileWeight), height_in: Number(profileHeight) };
    }
    status.textContent = "Validating and recalculating…"; button.disabled = true;
    try {
      const updated = await api("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      status.textContent = updated.recalculation.length ? updated.recalculation.map((stage) => `${stage.name}: ${stage.status}`).join(" · ") : "Saved. No analytical recalculation needed.";
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  });
}

async function route() {
  const hash = location.hash.slice(1) || "dashboard";
  document.querySelectorAll("nav a").forEach((link) => link.classList.toggle("active", hash === link.dataset.route || hash.startsWith(`${link.dataset.route}/`)));
  try {
    if (hash === "dashboard") await renderDashboard();
    else if (hash === "progress") await renderProgress();
    else if (hash === "runs") await renderRuns();
    else if (hash.startsWith("run/")) await renderRunDetail(Number(hash.split("/")[1]));
    else if (hash === "next-run") await renderNextRun();
    else if (hash === "settings") await renderSettings();
    else renderPlaceholder(hash);
  } catch (error) {
    view.innerHTML = `<section class="page narrow"><p class="eyebrow error-text">Could not load</p><h1>Something needs attention.</h1><p>${escapeHtml(error.message)}</p></section>`;
  }
  view.focus({ preventScroll: true });
}

uploadButton.addEventListener("click", () => uploadInput.click());
uploadClose.addEventListener("click", () => { uploadPanel.hidden = true; });
async function processUploads(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  uploadPanel.hidden = false;
  uploadStages.replaceChildren();
  uploadSummary.textContent = `Processing ${files.length} file${files.length === 1 ? "" : "s"}…`;
  uploadButton.disabled = true;
  try {
    const result = await api("/api/uploads", { method: "POST", body: form });
    result.stages.forEach((stage) => {
      const item = document.createElement("li");
      item.dataset.state = stage.status;
      item.innerHTML = `<strong>${escapeHtml(stage.name)}</strong><span>${escapeHtml(stage.status)}</span>`;
      item.title = stage.detail;
      uploadStages.append(item);
    });
    const accepted = result.files.filter((file) => file.status !== "failed").length;
    uploadSummary.textContent = `${accepted} of ${result.files.length} files accepted.`;
    if (result.primary_activity_id) location.hash = `run/${result.primary_activity_id}`;
    else await route();
  } catch (error) {
    uploadSummary.textContent = `Upload failed: ${error.message}`;
  } finally {
    uploadButton.disabled = false;
    uploadInput.value = "";
  }
}
uploadInput.addEventListener("change", () => processUploads(uploadInput.files));
document.addEventListener("dragover", (event) => {
  if ([...event.dataTransfer.types].includes("Files")) {
    event.preventDefault(); document.body.classList.add("dragging");
  }
});
document.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) document.body.classList.remove("dragging");
});
document.addEventListener("drop", (event) => {
  event.preventDefault(); document.body.classList.remove("dragging"); processUploads(event.dataTransfer.files);
});

window.addEventListener("hashchange", route);
route();
