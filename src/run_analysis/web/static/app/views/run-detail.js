import { api, targetHrLabel } from "../api.js";
import { DIRECTION, SENTIMENT, directionValue } from "../components.js";
import { view, loading } from "../dom.js";
import {
  calendarDateLabel, dataQualityLabel, dateLabel, durationLabel, escapeHtml, evidenceLabel, number,
  pace, titleCase,
} from "../format.js";

function renderAdjustment(observation, workoutType) {
  if (!observation) {
    const quality = ["intervals", "tempo_threshold", "race"].includes(workoutType);
    return `<article class="wide-card"><p class="eyebrow">Aerobic fitness estimate</p><h2>${quality ? "Analyzed as a workout" : "No comparable pace"}</h2><p>${quality ? "Intervals and races count toward training load, but their changing effort is not compared with steady aerobic runs." : "This run still counts toward your mileage and training load. Its sensor data did not support a fair pace-at-HR comparison."}</p></article>`;
  }
  const sign = observation.environmental_adjustment_min_mile >= 0 ? "+" : "−";
  const seconds = Math.abs(observation.environmental_adjustment_min_mile * 60);
  const contributions = observation.contributions.map((item) => `
    <li><span>${titleCase(item.name)}</span><strong>${item.minutes_per_mile >= 0 ? "+" : "−"}${Math.abs(item.minutes_per_mile * 60).toFixed(0)} sec/mi</strong><small>${evidenceLabel(item.confidence)} · ${escapeHtml(item.evidence)}</small></li>`).join("");
  return `<article class="wide-card audit-card">
    <div class="card-heading"><div><p class="eyebrow">Aerobic fitness estimate</p><h2>${observation.standardized_pace_at_target_hr.display} at ${targetHrLabel()}</h2></div><span class="quality ${observation.confidence}">${evidenceLabel(observation.confidence)}</span></div>
    <div class="equation"><span><strong>${observation.raw_pace_at_target_hr.display}</strong><small>before conditions</small></span><b>${sign} ${seconds.toFixed(0)} sec/mi</b><span><strong>${observation.standardized_pace_at_target_hr.display}</strong><small>after conditions</small></span></div>
    <details><summary>See the adjustments</summary><ul class="contribution-list">${contributions}</ul></details>
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
  // Only rendered when the session actually produced advice, which in practice
  // means quality workouts. It sits at the end because it is the conclusion
  // drawn from everything above it, not a separate topic.
  const progression = analysis.progression_recommendation
    ? `<article class="wide-card progression-note"><p class="eyebrow">Before you repeat this</p><p class="progression-text">${escapeHtml(analysis.progression_recommendation)}</p></article>`
    : "";
  return `<section class="workout-analysis"><div class="card-heading"><div><p class="eyebrow">Workout analysis</p><h2>How the workout went</h2></div></div><div class="analysis-grid">${cards}</div>${intervalTable}${comparison}${progression}</section>`;
}

function cadenceCard(cadence) {
  if (!cadence?.available) {
    const why = cadence?.limitations?.[0] ?? "No cadence was recorded for this activity.";
    return `<article class="wide-card"><p class="eyebrow">Turnover and stride</p><h2>Cadence</h2><p>${escapeHtml(why)}</p></article>`;
  }
  const change = cadence.change_spm;
  // Direction only. A cadence that rose is not automatically good news; what
  // it bought in pace is a separate observation below.
  const direction = Math.abs(change ?? 0) < 2 ? DIRECTION.FLAT : change > 0 ? DIRECTION.UP : DIRECTION.DOWN;
  const observations = cadence.observations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const limitations = cadence.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `<article class="wide-card">
    <div class="card-heading"><div><p class="eyebrow">Turnover and stride</p><h2>${number(cadence.average_spm, 0)} steps per minute</h2></div></div>
    <div class="metric-grid cadence-grid">
      <article><span>Average</span><strong>${number(cadence.average_spm, 0)} spm</strong><small>median ${number(cadence.median_spm, 0)}</small></article>
      <article><span>First half → second</span><strong>${directionValue(direction, SENTIMENT.NEUTRAL, `${number(cadence.first_half_spm, 0)} → ${number(cadence.second_half_spm, 0)}`)}</strong><small>${change > 0 ? "+" : ""}${number(change, 1)} spm</small></article>
      <article><span>Stride length</span><strong>${number(cadence.average_stride_length_m, 2)} m</strong><small>${number(cadence.first_half_stride_length_m, 2)} → ${number(cadence.second_half_stride_length_m, 2)} m</small></article>
      <article><span>Your usual here</span><strong>${cadence.comparison.available ? `${number(cadence.comparison.personal_median_spm, 0)} spm` : "—"}</strong><small>${escapeHtml(cadence.comparison.detail)}</small></article>
    </div>
    ${observations ? `<ul class="cadence-observations">${observations}</ul>` : ""}
    <details><summary>How this is calculated</summary><p>Speed is exactly cadence multiplied by stride length, so any pace change divides cleanly between the two. The comparison is against your own past segments run at a similar pace, never a population target.</p><ul>${limitations}</ul></details>
  </article>`;
}


/**
 * What comes next, beside the verdict on what just happened.
 *
 * A finished run raises exactly one question, and it is not about the run: it
 * is what to do tomorrow. That answer already exists in the weekly plan, so
 * this surfaces it here rather than making the athlete go and look, and links
 * through for the rest of the week.
 */
function renderNextRun(weekly) {
  if (!weekly) {
    return `<a class="assessment next-run-panel" href="#next-run"><p class="eyebrow">Your next run</p><h2>No plan yet</h2><p class="next-run-detail">Once a few runs are in, a seven-day plan is built from them.</p><span class="panel-action">Weekly plan →</span></a>`;
  }
  const next = weekly.days.find((day) => day.recommendation);
  const headline = next ? next.recommendation.title : "Nothing scheduled";
  const when = next ? calendarDateLabel(next.date) : "";
  const distance = next?.recommendation.distance_range_miles;
  const duration = next?.recommendation.duration_range_minutes;
  const extent = distance
    ? `${number(distance[0])}–${number(distance[1])} mi`
    : duration
      ? `${number(duration[0], 0)}–${number(duration[1], 0)} min`
      : "";
  const week = `${weekly.run_count} ${weekly.run_count === 1 ? "run" : "runs"} · ${number(weekly.projected_distance_range_miles[0])}–${number(weekly.projected_distance_range_miles[1])} mi`;
  return `<a class="assessment next-run-panel" href="#next-run">
    <p class="eyebrow">Your next run</p>
    <h2>${escapeHtml(headline)}</h2>
    <p class="next-run-detail">${escapeHtml([when, extent].filter(Boolean).join(" · "))}</p>
    <div class="next-run-week"><span>Next seven days</span><strong>${escapeHtml(week)}</strong></div>
    <span class="panel-action">Weekly plan →</span>
  </a>`;
}


export async function renderRunDetail(activityId) {
  loading("Run feedback");
  const [feedback, weekly] = await Promise.all([
    api(`/api/runs/${activityId}`),
    // A missing or not-yet-generated plan must not take the run page down.
    api("/api/weekly-schedule/latest").catch(() => null),
  ]);
  const run = feedback.run;
  const difficulty = run.session_difficulty;
  const zoneEntries = Object.entries(difficulty.zone_breakdown.zone_fractions).filter(([, value]) => value > 0);
  const zoneBars = zoneEntries.map(([zone, fraction]) => `<div class="zone-row"><span>${titleCase(zone)}</span><i><b style="width:${Math.min(100, fraction * 100)}%"></b></i><strong>${Math.round(fraction * 100)}%</strong></div>`).join("");
  const splits = feedback.splits.map((split) => `<tr><td>${split.index}${split.is_partial ? "*" : ""}</td><td>${number(split.distance_miles, 2)} mi</td><td>${pace(split.pace_min_mile)}</td><td>${number(split.average_hr_bpm, 0)} bpm</td><td>${split.average_cadence_spm ? `${number(split.average_cadence_spm, 0)} spm` : "—"}</td><td>${split.stride_length_m ? `${number(split.stride_length_m, 2)} m` : "—"}</td><td>${split.elevation_change_feet >= 0 ? "+" : ""}${number(split.elevation_change_feet, 0)} ft</td></tr>`).join("");
  const cautions = feedback.cautions.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const positives = feedback.positives.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const weather = feedback.weather ? `${number(feedback.weather.temperature_f, 0)}°F · dew point ${number(feedback.weather.dewpoint_f, 0)}°F · wind ${number(feedback.weather.wind_speed_mph, 0)} mph` : "Weather unavailable";
  const postalField = run.gps_quality.includes("missing")
    ? `<label>Run ZIP code<input name="postal_code" inputmode="numeric" pattern="[0-9]{5}" maxlength="5" value="${escapeHtml(feedback.metadata.postal_code ?? "")}" placeholder="e.g. 60601"><small>${feedback.metadata.location_label ? `Weather area: ${escapeHtml(feedback.metadata.location_label)}.` : "Optional. Saving sends the ZIP to Open-Meteo once; weather uses a randomized approximate location."}</small></label>`
    : "";
  view.innerHTML = `
    <section class="page">
      <a class="back-link" href="#runs">← All runs</a>
      <div class="page-heading run-title"><div><p class="eyebrow">${titleCase(run.workout_type)} · ${dataQualityLabel(run.data_quality)} data</p><h1>${number(run.distance_miles)} miles</h1><p>${dateLabel(run.start_time)}</p></div><div class="headline-pace"><strong>${pace(run.moving_pace_min_mile)}</strong><span>${number(run.average_hr_bpm, 0)} bpm average</span></div></div>
      <div class="run-summary-grid">
        <article class="assessment"><p class="eyebrow">How did it go?</p><h2>${escapeHtml(feedback.assessment)}</h2><div class="feedback-columns"><ul class="positive-list">${positives}</ul><ul class="caution-list">${cautions}</ul></div></article>
        ${renderNextRun(weekly)}
      </div>
      <details class="metadata-editor"><summary>Edit run details</summary><form id="metadata-form"><label>Workout type<select name="workout_type">${["easy","recovery","long","tempo_threshold","intervals","race","run_walk","hike","bike","other","unknown"].map((value) => `<option value="${value}" ${feedback.metadata.workout_type === value ? "selected" : ""}>${titleCase(value)}</option>`).join("")}</select></label><label>Health context<select name="health_tag">${["normal","illness","illness_recovery","injury_affected","other_abnormal"].map((value) => `<option value="${value}" ${feedback.metadata.health_tag === value ? "selected" : ""}>${titleCase(value)}</option>`).join("")}</select></label><label>Effort (1–10)<input type="number" min="1" max="10" name="perceived_exertion" value="${feedback.metadata.perceived_exertion ?? ""}" placeholder="Optional"></label><label>Use in fitness trend<select name="include_in_model"><option value="auto" ${feedback.metadata.include_in_model === null ? "selected" : ""}>Decide automatically</option><option value="true" ${feedback.metadata.include_in_model === true ? "selected" : ""}>Use</option><option value="false" ${feedback.metadata.include_in_model === false ? "selected" : ""}>Do not use</option></select></label>${postalField}<label class="notes-label">Notes<textarea name="notes" rows="2">${escapeHtml(feedback.metadata.notes)}</textarea></label><button type="submit">Save changes</button><span id="metadata-status"></span></form></details>
      ${renderAdjustment(run.fitness_observation, run.workout_type)}
      ${renderWorkoutAnalysis(feedback.workout_analysis)}
      <div class="metric-grid">
        <article><span>Distance</span><strong>${number(difficulty.distance_miles)} mi</strong><small>${difficulty.is_long_run ? "Long-run distance" : "Total run distance"}</small></article>
        <article><span>Moving / total time</span><strong>${number(difficulty.moving_minutes, 0)} / ${number(difficulty.elapsed_minutes, 0)} min</strong><small>${number(difficulty.stopped_minutes, 1)} minutes stopped</small></article>
        <article><span>Workout load</span><strong>${number(difficulty.zone_load, 0)}</strong><small>Based on time in each HR zone${difficulty.session_rpe_load == null ? "" : ` · effort load ${number(difficulty.session_rpe_load, 0)}`}</small></article>
        <article><span>Miles in prior 7 days</span><strong>${number(feedback.load_context_before_run?.trailing_7d.distance_miles)} mi</strong><small>${number(feedback.load_context_before_run?.trailing_7d.zone_load, 0)} prior load points</small></article>
      </div>
      <div class="two-column">
        <article class="wide-card"><p class="eyebrow">Heart-rate distribution</p><h2>Zones</h2>${zoneBars || "<p>Heart-rate coverage unavailable.</p>"}</article>
        <article class="wide-card"><p class="eyebrow">Weather and heart-rate drift</p><h2>${weather}</h2><p>${feedback.cardiac_drift.valid ? `${number(feedback.cardiac_drift.decoupling_percent)}% second-half drift` : "Not enough steady running to measure drift"}</p><small>${escapeHtml(feedback.cardiac_drift.reason)}</small></article>
      </div>
      ${cadenceCard(feedback.cadence)}
      <article class="wide-card"><p class="eyebrow">Splits from your track data</p><h2>Mile splits</h2><div class="table-scroll"><table><thead><tr><th>Mile</th><th>Distance</th><th>Pace</th><th>HR</th><th>Cadence</th><th>Stride</th><th>Elevation</th></tr></thead><tbody>${splits || '<tr><td colspan="7">No reliable splits.</td></tr>'}</tbody></table></div></article>
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
