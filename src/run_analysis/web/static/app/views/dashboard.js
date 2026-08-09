import { api, targetHrLabel } from "../api.js";
import { signalChip, statusSentiment } from "../components.js";
import { view, loading } from "../dom.js";
import {
  calendarDateLabel, escapeHtml, evidenceLabel, number, pace, readinessLabel,
  titleCase, trendLabel,
} from "../format.js";

export async function renderDashboard() {
  loading("Your running data, explained.");
  const dashboard = await api("/api/dashboard");
  const progress = dashboard.progress;
  const interpretation = dashboard.fitness_interpretation;
  const last = dashboard.last_run;
  const recommendation = dashboard.recommendation;
  const weekly = dashboard.weekly_schedule;
  const change = progress.pace_change_seconds_per_mile;
  const dashboardChange = ["improving", "declining"].includes(progress.fitness_trend) && Number.isFinite(change)
    ? `${Math.abs(change).toFixed(0)} sec/mi ${change > 0 ? "slower" : "faster"}`
    : Number.isFinite(change) ? "No clear change" : "—";
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
  const signalRows = interpretation.signals.map((signal) => signalChip(signal, evidenceLabel)).join("");
  const status = dashboard.training_status;
  const statusRules = status.rule_trace.map((rule) => `<li class="${rule.fired ? "fired" : ""}"><b>${escapeHtml(rule.description)}</b><small>${Object.entries(rule.facts).map(([key, value]) => `${escapeHtml(titleCase(key))}: ${escapeHtml(String(value ?? "—"))}`).join(" · ")}</small></li>`).join("");
  const nextPlannedDay = weekly?.days.find((day) => day.recommendation);
  const nextPlanLine = nextPlannedDay
    ? `Next: ${calendarDateLabel(nextPlannedDay.date)} — ${nextPlannedDay.recommendation.title}`
    : "No run is currently scheduled.";
  view.innerHTML = `
    <section class="page">
      <div class="hero">
        <p class="eyebrow">Training status</p>
        <h1 class="training-status ${statusSentiment(status.status)}">${escapeHtml(status.label)}</h1>
        <p class="lede status-detail">${escapeHtml(status.detail)}</p>
        <p class="lede">${escapeHtml(interpretation.headline)}</p>
        <p class="lede">${escapeHtml(interpretation.summary)}</p>
      </div>
      <details class="wide-card status-why"><summary><span class="eyebrow">Why this status</span><b>${escapeHtml(status.label)} — ${escapeHtml(evidenceLabel(status.confidence))}</b></summary><p>Rules are checked in order and the first match wins, so health outranks load and load outranks progression.</p><ol class="rule-trace">${statusRules}</ol></details>
      ${dashboard.setup && !dashboard.setup.complete ? `<a class="setup-banner" href="#setup"><div><b>Finish setup — ${dashboard.setup.remaining} ${dashboard.setup.remaining === 1 ? "answer" : "answers"} still on defaults</b><small>${escapeHtml(dashboard.setup.detail)}</small></div><span class="panel-action">Setup →</span></a>` : ""}
      <div class="dashboard-grid">
        <a class="dashboard-card card-progress" href="#progress"><p class="eyebrow">How am I doing?</p><h2>${trendLabel(progress.fitness_trend)}</h2><strong>${progress.current_pace?.display ?? "Not enough data"} at the same HR and conditions</strong><div class="dashboard-stat"><b>${dashboardChange}</b><span>estimated change</span></div><div class="dashboard-stat"><b>${number(progress.current_load.trailing_28d.distance_miles)} mi</b><span>last 28 days</span></div><small>${evidenceLabel(progress.fitness_confidence)}</small><span class="panel-action">Progress →</span></a>
        <a class="dashboard-card card-run" href="${last ? `#run/${last.run.activity_id}` : "#runs"}"><p class="eyebrow">How was my last run?</p><h2>${escapeHtml(last?.assessment ?? "No run yet")}</h2><strong>${last ? `${number(last.run.distance_miles)} mi · ${pace(last.run.moving_pace_min_mile)} · ${number(last.run.average_hr_bpm, 0)} bpm` : "Upload a run file to begin"}</strong><div class="dashboard-stat"><b>${last ? `${number(knownMinutes ? easyMinutes / knownMinutes * 100 : null, 0)}%` : "—"}</b><span>easy HR time</span></div><div class="dashboard-stat"><b>${last?.run.fitness_observation?.standardized_pace_at_target_hr.display ?? "—"}</b><span>adjusted pace at ${targetHrLabel()}</span></div><span class="panel-action">Run analysis →</span></a>
        <a class="dashboard-card card-plan" href="#next-run"><p class="eyebrow">Your next seven days</p><h2>${weekly ? `${weekly.run_count} runs · ${nextExtent}` : escapeHtml(recommendation.title)}</h2><strong>${escapeHtml(nextPlanLine)}</strong><div class="dashboard-stat"><b>${weekly ? weekly.days.filter((day) => ["intervals", "tempo_threshold", "race"].includes(day.recommendation?.workout_type)).length : 0}</b><span>quality sessions</span></div><div class="dashboard-stat"><b>${number(progress.current_load.trailing_7d.distance_miles)} mi</b><span>last 7 days</span></div><span class="panel-action">Weekly plan →</span></a>
      </div>
      <article class="wide-card fitness-context"><p class="eyebrow">Fitness has more than one dimension</p><div class="signal-grid detailed-signals">${signalRows}</div><p>${escapeHtml(interpretation.capacity_summary)}</p>${interpretation.illness_context ? `<p class="context-note">${escapeHtml(interpretation.illness_context)}</p>` : ""}</article>
      <div class="metric-grid dashboard-load"><article><span>Last 7 days</span><strong>${number(progress.current_load.trailing_7d.distance_miles)} mi</strong><small>${number(progress.current_load.trailing_7d.moving_minutes, 0)} running minutes</small></article><article><span>Training load</span><strong>${number(progress.current_load.trailing_7d.zone_load, 0)}</strong><small>${evidenceLabel(progress.current_load.confidence)}</small></article><article><span>Consistency</span><strong>${number(progress.consistency.runs_per_week, 1)} runs/week</strong><small>${progress.consistency.running_days} run days in 28</small></article><article><span>Longest recent run</span><strong>${number(progress.consistency.longest_run_miles)} mi</strong><small>Shows endurance, not speed</small></article></div>
    </section>`;
}
