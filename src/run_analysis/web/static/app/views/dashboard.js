import { api, targetHrLabel } from "../api.js";
import { signalChip, statusSentiment } from "../components.js";
import { fatiguePanel } from "../progress-wip-charts.js";
import { wireRunCharts } from "../chart-interaction.js";
import { loadGuidanceDisplay } from "../load-guidance-display.js";
import { fitnessWindow } from "../fitness-window.js";
import { performanceSummary } from "../progress-insights.js";
import { trainingStatusMessage } from "../training-status-display.js";
import { view, loading } from "../dom.js";
import {
  calendarDateLabel, escapeHtml, evidenceLabel, number, pace,
  titleCase, fitnessWindowLabel,
} from "../format.js";

let requestSequence=0;
export async function renderDashboard({showLoading=true}={}) {
  const sequence=++requestSequence;
  if(showLoading) loading("Your running data, explained.");
  const selectedWindow = await fitnessWindow();
  const [dashboard, overview] = await Promise.all([
    api(`/api/dashboard?fitness_window_days=${selectedWindow}`), api("/api/training-overview").catch(() => null),
  ]);
  if(sequence!==requestSequence || globalThis.location && !['','#dashboard'].includes(location.hash)) return;
  const progress = dashboard.progress;
  const interpretation = dashboard.fitness_interpretation;
  const last = dashboard.last_run;
  const recommendation = dashboard.recommendation;
  const needsPlanRefresh = dashboard.weekly_plan_needs_refresh;
  const weekly = needsPlanRefresh ? null : dashboard.weekly_schedule;
  const planningMode = weekly?.target_evidence?.planning_mode ?? "established";
  const shortHorizon = interpretation.short_term;
  const aerobic = performanceSummary({ ...shortHorizon, fitness_trend: shortHorizon.trend },fitnessWindowLabel(selectedWindow));
  const observedPace=progress.series?.some(p=>p.pace_basis==='observed_comparable');
  const dashboardTrend = aerobic.statusLabel;
  const dashboardChange = aerobic.changeLabel;
  const nextScheduled = weekly?.days.find((day) => day.recommendation)?.recommendation;
  const weeklyDistance = weekly?.visible_7d_scheduled_miles != null
    ? number(weekly.visible_7d_scheduled_miles)
    : weekly?.projected_distance_range_miles?.map(value => number(value)).filter((value, index, values) => index === 0 || value !== values[0]).join("–");
  const nextExtent = planningMode === "baseline_required" && nextScheduled?.duration_range_minutes
    ? `${number(nextScheduled.duration_range_minutes[0], 0)}–${number(nextScheduled.duration_range_minutes[1], 0)} min`
    : weekly
    ? `${weeklyDistance} mi this week`
    : recommendation.distance_target_miles != null
    ? `${number(recommendation.distance_target_miles)} mi`
    : recommendation.distance_range_miles
    ? `${number(recommendation.distance_range_miles[0])}–${number(recommendation.distance_range_miles[1])} mi`
    : recommendation.duration_range_minutes
      ? `${number(recommendation.duration_range_minutes[0], 0)}–${number(recommendation.duration_range_minutes[1], 0)} min`
      : "No run";
  const lastEasy = last?.run.session_difficulty?.zone_breakdown;
  const easyMinutes = lastEasy ? lastEasy.easy_minutes : 0;
  const knownMinutes = lastEasy ? lastEasy.easy_minutes + lastEasy.moderate_minutes + lastEasy.hard_minutes : 0;
  const signalRows = interpretation.signals.map((signal) => signalChip(signal.label === "Aerobic efficiency"
    ? { ...signal, trend: aerobic.direction, status: aerobic.statusLabel }
    : signal, evidenceLabel)).join("");
  const status = dashboard.training_status;
  const statusRules = (status?.rule_trace??[]).map((rule) => `<li class="${rule.fired ? "fired" : ""}"><b>${escapeHtml(rule.description??titleCase(rule.rule_id))}</b><small>${Object.entries(rule.facts??{}).map(([key,value])=>`${escapeHtml(titleCase(key))}: ${escapeHtml(String(value??'—'))}`).join(' · ')}</small></li>`).join('');
  const date = value => new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',timeZone:overview.timezone}).format(new Date(value));
  const report=overview?.current_check_in;
  const reportText=report?.status==='pain_or_injury_concern'?'You reported pain or an injury concern.':report?.status==='sick_or_recovering'?'You reported illness or recovery from illness.':null;
  const reportTime=reportText?new Intl.DateTimeFormat(undefined,{dateStyle:'medium',timeStyle:'short',timeZone:overview.timezone}).format(new Date(report.reported_at)):null;
  const nextPlannedDay = weekly?.days.find((day) => day.recommendation);
  const nextPlanLine = needsPlanRefresh ? 'Open Weekly Plan to update your next run.' : nextPlannedDay
    ? `Next: ${calendarDateLabel(nextPlannedDay.date)} — ${nextPlannedDay.recommendation.title}`
    : planningMode === "baseline_required"
      ? "Schedule a 10–30-minute conversational baseline run."
      : "No run is currently scheduled.";
  const planHeading = needsPlanRefresh ? 'Plan needs updating' : planningMode === "baseline_required"
    ? `Baseline run · ${nextExtent}`
    : planningMode === "baseline_building"
      ? `${weekly.run_count} baseline run`
      : weekly
        ? `${weekly.run_count} runs · ${nextExtent}`
        : escapeHtml(recommendation.title);
  view.innerHTML = `
    <section class="page">
      <div class="hero">
        <p class="eyebrow">Training status</p>
        <h1 class="training-status ${statusSentiment(status?.status)}">${escapeHtml(status?.label??'Status unavailable')}</h1>
        <p class="lede status-detail">${escapeHtml(trainingStatusMessage(status,aerobic))}</p>
        ${reportText?`<p class="load-check-in"><b>Recent check-in</b> ${escapeHtml(reportText)}<small>${escapeHtml(reportTime)} · separate from the load comparison</small></p>`:''}
      </div>
      <details class="wide-card status-why"><summary><span class="eyebrow">Why this status</span><b>${escapeHtml(status?.label??'Unavailable')} — ${escapeHtml(evidenceLabel(status?.confidence))}</b></summary><p>Health, load and progression are checked separately, in that order.</p><ol class="rule-trace">${statusRules}</ol></details>
      ${dashboard.setup && !dashboard.setup.complete ? `<a class="setup-banner" href="#setup"><div><b>Finish setup — ${dashboard.setup.remaining} ${dashboard.setup.remaining === 1 ? "answer" : "answers"} still on defaults</b><small>${escapeHtml(dashboard.setup.detail)}</small></div><span class="panel-action">Setup →</span></a>` : ""}
      <div class="dashboard-grid">
        <a class="dashboard-card card-progress" href="#progress"><p class="eyebrow">How am I doing?</p><h2>${dashboardTrend}</h2><strong>${shortHorizon.current_pace?.display ?? "Not enough data"} ${observedPace?`near ${targetHrLabel()}`:'at the same HR and conditions'}</strong><div class="dashboard-stat"><b>${dashboardChange}</b><span>${aerobic.chosen?escapeHtml(aerobic.basis.toLowerCase()):'estimated change'}</span></div>${aerobic.recentTrajectory?`<p class="recent-trajectory">${escapeHtml(aerobic.recentTrajectory)}</p>`:''}<div class="dashboard-stat"><b>${number(progress.current_load.trailing_28d.distance_miles)} mi</b><span>last 28 days</span></div><small>${fitnessWindowLabel(selectedWindow)} · ${evidenceLabel(shortHorizon.confidence)}</small><span class="panel-action">Progress →</span></a>
        <a class="dashboard-card card-run" href="${last ? `#run/${last.run.activity_id}` : "#runs"}"><p class="eyebrow">How was my last run?</p><h2>${escapeHtml(last?.assessment ?? "No run yet")}</h2><strong>${last ? `${number(last.run.distance_miles)} mi · ${pace(last.run.moving_pace_min_mile)} · ${number(last.run.average_hr_bpm, 0)} bpm` : "Upload a run file to begin"}</strong><div class="dashboard-stat"><b>${last ? `${number(knownMinutes ? easyMinutes / knownMinutes * 100 : null, 0)}%` : "—"}</b><span>easy HR time</span></div><div class="dashboard-stat"><b>${last?.run.fitness_observation?.standardized_pace_at_target_hr.display ?? "—"}</b><span>${last?.run.fitness_observation?.pace_basis==='observed_comparable'?'observed pace near':'adjusted pace at'} ${targetHrLabel()}</span></div><span class="panel-action">Run analysis →</span></a>
        <a class="dashboard-card card-plan" href="#next-run"><p class="eyebrow">Your next seven days</p><h2>${planHeading}</h2><strong>${escapeHtml(nextPlanLine)}</strong><div class="dashboard-stat"><b>${needsPlanRefresh ? '—' : weekly ? weekly.days.filter((day) => ["intervals", "tempo_threshold", "race"].includes(day.recommendation?.workout_type)).length : 0}</b><span>quality sessions</span></div><div class="dashboard-stat"><b>${number(progress.current_load.trailing_7d.distance_miles)} mi</b><span>last 7 days</span></div><span class="panel-action">Weekly plan →</span></a>
      </div>
      <section class="wide-card dashboard-training-load progress-wip" aria-labelledby="dashboard-load-heading">
        <h2 id="dashboard-load-heading">Training load</h2>
        <div class="dashboard-load-layout">
          <div class="load-interpretation">${loadGuidanceDisplay(overview)}</div>
          <div class="load-graph">${overview?fatiguePanel(overview,date,{embedded:true}):'<p class="chart-empty">Load history could not be loaded.</p>'}</div>
        </div>
      </section>
      <article class="wide-card fitness-context"><p class="eyebrow">Fitness has more than one dimension</p><div class="signal-grid detailed-signals">${signalRows}</div>${interpretation.illness_context ? `<p class="context-note">${escapeHtml(interpretation.illness_context)}</p>` : ""}</article>
      <div class="metric-grid dashboard-load"><article><span>Last 7 days</span><strong>${number(progress.current_load.trailing_7d.distance_miles)} mi</strong><small>${number(progress.current_load.trailing_7d.moving_minutes, 0)} running minutes</small></article><article><span>HR-based points</span><strong>${number(progress.current_load.trailing_7d.zone_load, 0)}</strong><small>Separate zone-based total · last 7 days</small></article><article><span>Consistency</span><strong>${number(progress.consistency.runs_per_week, 1)} runs/week</strong><small>${progress.consistency.running_days} run days in 28</small></article><article><span>Longest recent run</span><strong>${number(progress.consistency.longest_run_miles)} mi</strong><small>Shows endurance, not speed</small></article></div>
    </section>`;
  wireRunCharts(view);
}
