import { api } from "../api.js";
import { plannerRequest } from '../planner-request.js';
import { goalMarkup } from "../goal.js";
import { longRunGoalMarkup } from "../planner-goal.js";
import { view } from "../dom.js";
import { advisoryDaypart, checkInFieldsMarkup, runWarningMarkup } from "../plan-status.js";
import { workoutDetailsMarkup } from "../workout-details.js";
import { beginPlannerLoading } from '../planner-loading.js';
import { beginnerTrainingMarkup, beginnerFeedbackPayload } from '../beginner-training.js';
import {
  calendarDateLabel, datetimeLocalValue, escapeHtml, number, pace,
  titleCase,
} from "../format.js";

function workoutTitle(result) {
  // Display-only labels: keep the scored prescription and effort instructions intact.
  const labels = {
    "Controlled intervals": "Intervals",
    "Fartlek by feel": "Fartlek",
    "Short controlled pickups": "Pickups",
    "Long controlled efforts": "Intervals",
    "Continuous threshold run": "Threshold run",
    "Controlled progression run": "Progression run",
    "Hills by terrain": "Hill repeats",
    "Easy run with controlled pickups": "Easy run with pickups",
    "Extended fartlek aerobic session": "Fartlek",
    "Extended aerobic session with short pickups": "Pickups",
    "Extended aerobic session with long efforts": "Intervals",
    "Extended aerobic session with threshold blocks": "Threshold intervals",
    "Extended aerobic progression session": "Progression run",
    "Extended aerobic session with hills": "Hill repeats",
  };
  return labels[result.title] ?? result.title;
}

function durationExtent(range) {
  const format=value=>number(value,Number.isInteger(value)?0:1);
  return `${format(range[0])}${range[0]===range[1]?'':`–${format(range[1])}`} minutes`;
}

function strengthDayLabel(suggestion) {
  const labels = { push: "Push Day", pull: "Pull Day", legs: "Leg Day", full_upper: "Upper Body Day" };
  const legacyTitles = { "Upper-body push": "Push Day", "Upper-body pull": "Pull Day", "Leg strength": "Leg Day", "Upper body": "Upper Body Day" };
  return labels[suggestion.session_type] ?? legacyTitles[suggestion.title] ?? suggestion.title;
}

export function strengthCompletionMarkup(day, todayDate) {
  const session = day.strength_completion?.session ?? day.strength_suggestion;
  if (!session?.session_type) return '';
  const done = Boolean(day.strength_completion);
  const future = day.date > todayDate;
  return `<label class="strength-check${done ? ' is-complete' : ''}"><input type="checkbox" data-strength-date="${escapeHtml(day.date)}" data-strength-type="${escapeHtml(session.session_type)}"${done ? ' checked' : ''}${future ? ' disabled title="Available on the day"' : ''} aria-label="${escapeHtml(`${strengthDayLabel(session)} completed on ${calendarDateLabel(day.date)}`)}"><span>${done ? 'Completed' : 'Mark complete'}</span></label>`;
}

export function updateStrengthCompletion(schedule, date, completion) {
  const update = days => (days ?? []).map(day => day.date === date
    ? {...day, strength_completion: completion} : day);
  return {...schedule, days: update(schedule.days), trailing_days: update(schedule.trailing_days)};
}

function strengthSuggestionMarkup(suggestion, standalone = false, day = null, todayDate = '') {
  if (!suggestion) return "";
  const groups = suggestion.muscle_groups.map(escapeHtml).join(" · ");
  // Omit routine scheduling boilerplate, including in already-cached plans.
  const rationale = suggestion.rationale === "An isolated no-run day can carry your usual upper-body muscle groups together."
    || /^This is the best available no-run day for the \d+-day leg cadence while keeping lower-body work away from long and quality runs when the calendar allows\.$/.test(suggestion.rationale ?? "")
    ? "" : suggestion.rationale;
  return `<div class="strength-suggestion${standalone ? " standalone-strength" : ""}">${standalone ? "" : `<h3>Strength</h3><strong class="strength-label prescription-extent">${escapeHtml(strengthDayLabel(suggestion))}</strong>`}<p class="strength-groups">${groups}</p>${rationale ? `<small>${escapeHtml(rationale)}</small>` : ""}${day ? strengthCompletionMarkup(day, todayDate) : ''}</div>`;
}

export function weeklyScheduleMarkup(schedule, goal, state={}, planner=null, asOf=new Date()) {
  if (!schedule) return `<article class="wide-card empty-state">Choose your current health status, then generate the next seven days.</article>`;
  const evidence = schedule.target_evidence;
  const baselineMode = evidence?.planning_mode && evidence.planning_mode !== "established";
  const baselineRequired = evidence?.planning_mode === "baseline_required";
  const todayDate = new Intl.DateTimeFormat('en-CA',{year:'numeric',month:'2-digit',day:'2-digit'}).format(asOf);
  const dayCards = schedule.days.map((day) => {
    const result = day.recommendation;
    const strength = day.strength_completion?.session ?? day.strength_suggestion;
    const headerContext = [
      day.planned_at ? advisoryDaypart(result) : "",
      day.day_role === "race_event_day" && schedule.planner_status !== "review_required"
        ? "Race" : !result && !day.completed_activities?.length && day.forced_rest ? "Your constraint" : "",
    ].filter(Boolean).join(" · ");
    const dateHeading = `<p class="eyebrow schedule-date"><span>${escapeHtml(calendarDateLabel(day.date))}</span><span class="schedule-daypart"${headerContext ? "" : ' aria-hidden="true"'}>${headerContext ? escapeHtml(headerContext) : "&#160;"}</span></p>`;
    const emptyPrescription = '<span class="schedule-prescription-space" aria-hidden="true"></span>';
    if (!result && day.day_role === 'awaiting_calibration') return `<article class="schedule-day">${dateHeading}<h2>Not planned yet</h2>${emptyPrescription}<div class="schedule-body"><p>${escapeHtml(day.rationale)}</p></div></article>`;
    const restToggle = baselineMode && !result
      ? ""
      : `<button type="button" class="rest-toggle" data-rest-date="${day.date}" data-forced-rest="${day.forced_rest ? "true" : "false"}" aria-pressed="${day.forced_rest ? "true" : "false"}">${day.forced_rest ? "Allow a run" : result ? "Make run rest day" : "Protect as no-run day"}</button>`;
    const strengthSuggestion = schedule.planner_status === "review_required" && !day.strength_completion
      ? "" : strengthSuggestionMarkup(strength, !result, day, todayDate);
    const obsoleteRestRationale = "No run is scheduled between these workouts. Other training is not currently recorded, so only running load is projected.";
    const restRationale = day.rationale && day.rationale !== obsoleteRestRationale
      && day.rationale !== "No run selected by the full projected training program."
      ? `<p>${escapeHtml(day.rationale)}</p>`
      : "";
    if (day.day_role === 'race_event_day' && schedule.planner_status !== 'review_required') {
      return `<article class="schedule-day">${dateHeading}<h2>Race day</h2>${emptyPrescription}<div class="schedule-body">${restRationale}</div></article>`;
    }
    if (day.completed_activities?.length) {
      const completed = day.completed_activities.map((activity) => `<a href="#run/${activity.activity_id}"><strong>${number(activity.distance_miles)} mi · ${titleCase(activity.workout_type)}</strong><small>${titleCase(activity.health_tag)}</small></a>`).join("");
      return `<article class="schedule-day completed-day">${dateHeading}<h2>Run completed</h2>${emptyPrescription}<div class="schedule-body">${completed}<p>${escapeHtml(day.rationale)}</p>${strengthSuggestion}</div></article>`;
    }
    if (!result) return `<article class="schedule-day rest-day ${strengthSuggestion ? "lifting-day" : ""} ${day.forced_rest ? "forced-rest-day" : ""}">${dateHeading}<h2>${schedule.planner_status === "review_required" ? "Planner paused" : strengthSuggestion ? "Strength" : "Rest Day"}</h2>${strengthSuggestion ? `<strong class="strength-label prescription-extent">${escapeHtml(strengthDayLabel(strength))}</strong>` : emptyPrescription}<div class="schedule-body">${strengthSuggestion}${restRationale}${restToggle}</div></article>`;
    const extent = result.distance_target_miles != null
      ? `${number(result.distance_target_miles)} miles`
      : result.distance_range_miles
      ? `${number(result.distance_range_miles[0])}–${number(result.distance_range_miles[1])} miles`
      : result.duration_range_minutes
        ? durationExtent(result.duration_range_minutes)
        : "No run";
    const weather = result.planned_weather;
    const weatherTrace = result.rule_trace.find((item) => item.rule_id === "planned_weather");
    const recoveryTrace = result.rule_trace.find((item) => item.rule_id === "recent_recovery_load");
    const weatherStress = weatherTrace?.facts?.stress_band;
    const alertEvents = weather?.emergency_alerts?.map((alert) => `NWS ${alert.event}`).join(" · ");
    const weatherLine = weather
      ? `${number(weather.temperature_f, 0)}°F · feels like ${number(weather.apparent_temperature_f, 0)}°F · dew point ${number(weather.dewpoint_f, 0)}°F · wind ${number(weather.wind_speed_mph, 0)} mph${weatherStress ? ` · ${titleCase(weatherStress)} training stress` : ""}${alertEvents ? ` · ${escapeHtml(alertEvents)}` : ""}`
      : "No forecast. Recent training is still included in the plan.";
    const easyRecoveryHours = recoveryTrace?.facts?.hours_until_easy;
    const taxingRecoveryHours = recoveryTrace?.facts?.hours_until_taxing;
    const recoveryLine = Number(taxingRecoveryHours) > 0
      ? `<small class="recovery-readout">At this start: ${Number(easyRecoveryHours) > 0 ? `${number(easyRecoveryHours, 0)} hr until normal easy-run recovery` : "easy-run recovery range reached"} · ${number(taxingRecoveryHours, 0)} hr until long/quality recovery</small>`
      : "";
    const workoutDetails = workoutDetailsMarkup(result);
    const warning = runWarningMarkup(result);
    return `<article class="schedule-day scheduled-run-card ${result.readiness}">${dateHeading}<h2>${escapeHtml(workoutTitle(result))}</h2><strong class="prescription-extent">${extent}</strong><div class="schedule-body"><p class="run-description">${result.target_zones.map(escapeHtml).join(" · ") || (result.workout_type === "intervals" ? "Controlled work with easy recoveries" : "Recovery")}</p>${warning.badge}<small>${escapeHtml(weatherLine)}</small>${recoveryLine}${strengthSuggestion}${workoutDetails}${warning.explanation}${restToggle}</div></article>`;
  }).join("");
  const trailingCards = schedule.trailing_days.map((day) => {
    const strength = day.strength_completion?.session ?? day.strength_suggestion;
    const lifting = strength ? `<div class="trailing-strength"><strong>${escapeHtml(strengthDayLabel(strength))}</strong>${strengthSuggestionMarkup(strength, true, day, todayDate)}</div>` : '';
    const heading = `<p class="eyebrow">${calendarDateLabel(day.date)}</p><h2>${!day.activities.length && strength ? 'Strength' : titleCase(day.day_role)}</h2>`;
    if (!day.activities.length) return `<article class="trailing-day rest-day">${heading}${lifting || '<small>No recorded activity</small>'}</article>`;
    const activities = day.activities.map((activity) => `<a class="trailing-day logged-run-card" href="#run/${encodeURIComponent(activity.activity_id)}" aria-label="${escapeHtml(`Run analysis: ${calendarDateLabel(day.date)}, ${number(activity.distance_miles)} miles, ${titleCase(activity.workout_type)}`)}">${heading}<div class="trailing-run-details"><strong>${number(activity.distance_miles)} mi · ${escapeHtml(titleCase(activity.workout_type))}</strong><small>${escapeHtml(titleCase(activity.health_tag))}</small></div></a>`).join("");
    // Doubles keep separate destinations instead of nesting links in one card.
    return day.activities.length === 1 && !lifting ? activities : `<div class="trailing-day-group">${activities}${lifting ? `<article class="trailing-day">${lifting}</article>` : ''}</div>`;
  }).join("");
  const today = schedule.days.find(day=>day.date===todayDate);
  let todayWorkout;
  if(schedule.planner_status==='review_required') todayWorkout=`<h3>Planner paused</h3><p>${escapeHtml(schedule.planner_review_reason??'Review the plan before training.')}</p>`;
  else if(today?.completed_activities?.length) todayWorkout=`<h3>Run completed</h3>${today.completed_activities.map(a=>`<p><a href="#run/${a.activity_id}">${number(a.distance_miles)} miles · ${escapeHtml(titleCase(a.workout_type))}</a></p>`).join('')}`;
  else if(today?.day_role==='race_event_day') todayWorkout='<h3>Race day</h3>';
  else if(today?.recommendation) {
    const r=today.recommendation;
    const extent=r.distance_target_miles!=null?`${number(r.distance_target_miles)} miles`:r.distance_range_miles?`${number(r.distance_range_miles[0])}–${number(r.distance_range_miles[1])} miles`:r.duration_range_minutes?durationExtent(r.duration_range_minutes):'';
    todayWorkout=`<h3>${escapeHtml(workoutTitle(r))}</h3><p class="today-workout-dose">${extent}</p>${workoutDetailsMarkup(r)}${today.strength_suggestion?`<p>${escapeHtml(strengthDayLabel(today.strength_suggestion))}</p>`:''}`;
  } else if(today?.strength_suggestion) todayWorkout=`<h3>${escapeHtml(strengthDayLabel(today.strength_suggestion))}</h3><p>${today.strength_suggestion.muscle_groups.map(escapeHtml).join(' · ')}</p>`;
  else todayWorkout=`<h3>${today?'Rest Day':'No current plan for today'}</h3>`;
  const scheduledMiles=schedule.visible_7d_scheduled_miles??
    ((schedule.projected_distance_range_miles[0]+schedule.projected_distance_range_miles[1])/2);
  const summaryTitle=`${schedule.run_count} planned ${schedule.run_count===1?'run':'runs'}${!baselineMode&&Number.isFinite(scheduledMiles)?` · ${number(scheduledMiles)} miles this week`:''}`;
  const goalPanel=goal&&goal.status!=='no_goal'?goalMarkup(goal,'panel'):longRunGoalMarkup(planner,state,goal);
  const baselineGuide = schedule.beginner_training ? beginnerTrainingMarkup(schedule.beginner_training) : baselineMode ? `<article class="wide-card baseline-guide"><p class="eyebrow">How calibration works</p><h2>${baselineRequired ? "Start with measurement, not a guessed plan" : "One observed session at a time"}</h2><ol><li>${baselineRequired ? "Complete the scheduled 10–30-minute conversational Zone 2 run or run/walk. Stop for pain, fatigue, and/or elevated heart rate and end the run at a maximum of 30 minutes." : "Complete the single easy baseline run. Its duration repeats the aerobic exposure you have actually demonstrated."}</li><li>Upload the result so duration, distance, effort, intensity, and recovery become evidence.</li><li>The app repeats or adjusts observed exposure until a weekly capacity is measurable, then transitions automatically into continuous planning.</li></ol></article>` : "";
  const plannedDays = `<div class="week-grid planned-week">${dayCards}</div>`;
  return `<div class="plan-summary-grid"><article class="wide-card schedule-summary"><div><p class="eyebrow">${calendarDateLabel(schedule.start_date)} through ${calendarDateLabel(schedule.end_date)}</p><h2>${summaryTitle}</h2>${schedule.completed_run_count?`<small>${schedule.completed_run_count} already completed</small>`:''}<div class="today-workout"><p class="eyebrow">Today</p>${todayWorkout}</div></div></article>${goalPanel}</div>${baselineGuide}${plannedDays}<article class="wide-card trailing-summary"><p class="eyebrow">Recent training</p><h2>Previous 7 days</h2><div class="week-grid trailing-week">${trailingCards}</div></article>`;
}

export async function renderNextRun() {
  const progress=beginPlannerLoading(view);
  let status, latest, state, goal, planner;
  try {
    [status, latest, state, goal, planner] = await Promise.all([
      api("/api/current-status"), plannerRequest("/api/weekly-schedule/latest",{headers:progress.headers}), api("/api/fitness-state"),
      api("/api/goal-progress").catch(() => null),
      api("/api/settings").then(settings=>settings.planner).catch(()=>null),
    ]);
    if(!progress.isActive()) return;
  } finally { progress.stop(); }
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">Weekly plan</p><h1>Your next seven days</h1><p>Updates when you upload a run or change how you feel.</p></div></div>
      <form id="health-form" class="health-form">
        ${checkInFieldsMarkup(status)}
        <button class="primary-button" type="submit">Update plan</button>
      </form>
      <div id="recommendation-result">${weeklyScheduleMarkup(latest, goal, state, planner)}</div>
      <div class="state-strip"><span><b>${number(state.recent_load.trailing_7d.distance_miles)} mi</b>last 7 days</span><span><b>${Number.isFinite(state.recent_load.continuous_fatigue_to_capacity_ratio) ? `${number(state.recent_load.continuous_fatigue_to_capacity_ratio * 100, 0)}%` : "—"}</b>continuous fatigue vs capacity</span><span><b>${number((state.moderate_fraction_14d ?? 0) * 100, 0)}%</b>moderate-intensity time</span><span><b>${state.running_days_28d}</b>run days in 28</span></div>
    </section>`;
  view.querySelector('#recommendation-result').addEventListener('change', async event => {
    const input = event.target.closest('[data-strength-date]');
    if (!input) return;
    const checked = input.checked;
    input.disabled = true;
    const label = input.closest('label');
    label.querySelector('.strength-save-error')?.remove();
    try {
      const result = await api('/api/strength-completions', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({date:input.dataset.strengthDate,
          session_type:input.dataset.strengthType, completed:checked}),
      });
      latest = updateStrengthCompletion(latest, result.date, result.strength_completion);
      view.querySelector('#recommendation-result').innerHTML = weeklyScheduleMarkup(latest, goal, state, planner);
    } catch (error) {
      input.checked = !checked;
      label.insertAdjacentHTML('beforeend', `<span class="strength-save-error error-text" role="alert">${escapeHtml(error.message)}</span>`);
    } finally { input.disabled = false; }
  });
  view.querySelector('#recommendation-result').addEventListener('submit', async event => {
    const form=event.target.closest('[data-beginner-feedback]');
    if(!form) return;
    event.preventDefault();
    const buttons=[...form.querySelectorAll('button')];
    const payload=beginnerFeedbackPayload(form,event.submitter);
    buttons.forEach(button=>button.disabled=true);
    const feedbackStatus=form.querySelector('[data-feedback-status]');
    feedbackStatus.textContent='Saving…';
    try {
      await api(`/api/beginner-feedback/${form.dataset.beginnerFeedback}`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      feedbackStatus.textContent='Updating your starter schedule…';
      latest=await plannerRequest('/api/weekly-schedule/latest');
      view.querySelector('#recommendation-result').innerHTML=weeklyScheduleMarkup(latest,goal,state,planner);
    } catch(error) {
      feedbackStatus.textContent=error.message;
      buttons.forEach(button=>button.disabled=false);
    }
  });
  view.querySelector("#recommendation-result").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-rest-date]");
    if (!button) return;
    const makeRest = button.dataset.forcedRest !== "true";
    const resultHost=view.querySelector('#recommendation-result');
    const priorMarkup=resultHost.innerHTML;
    button.parentElement.querySelector(".rest-toggle-error")?.remove();
    button.disabled = true;
    button.textContent = "Replanning…";
    const progress=beginPlannerLoading(resultHost,{title:'Updating your schedule'});
    try {
      const result = await plannerRequest("/api/weekly-schedule/rest-day", {
        method: "POST", headers: { "Content-Type": "application/json",...progress.headers },
        body: JSON.stringify({ date: button.dataset.restDate, is_rest_day: makeRest }),
      });
      if(progress.isActive()) { latest = result; resultHost.innerHTML = weeklyScheduleMarkup(result, goal, state, planner); }
    } catch (error) {
      if(progress.isActive()) resultHost.innerHTML = priorMarkup+`<p class="rest-toggle-error" role="alert">${escapeHtml(error.message)}</p>`;
    } finally {progress.stop();}
  });
  view.querySelector("#health-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const button = formElement.querySelector("button");
    button.disabled = true; button.textContent = "Updating…";
    const resultHost=view.querySelector('#recommendation-result');
    const progress=beginPlannerLoading(resultHost,{title:'Updating your schedule'});
    try {
      const result = await plannerRequest("/api/weekly-schedule", {
        method: "POST", headers: { "Content-Type": "application/json",...progress.headers },
        body: JSON.stringify({ health_status: form.get("health_status") }),
      });
      if(!progress.isActive()) return;
      latest = result;
      resultHost.innerHTML = weeklyScheduleMarkup(result, goal, state, planner);
      const updatedStatus = await api("/api/current-status");
      formElement.querySelector("fieldset").outerHTML = checkInFieldsMarkup(updatedStatus);
    } catch (error) {
      if(progress.isActive()) resultHost.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    } finally {progress.stop();button.disabled = false; button.textContent = "Update plan"; }
  });
}
