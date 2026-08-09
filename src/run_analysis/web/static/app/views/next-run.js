import { api } from "../api.js";
import { DIRECTION, SENTIMENT, directionValue } from "../components.js";
import { goalMarkup } from "../goal.js";
import { view, loading } from "../dom.js";
import {
  calendarDateLabel, datetimeLocalValue, daypartLabel, escapeHtml, number, pace,
  readinessLabel, titleCase,
} from "../format.js";

function weeklyScheduleMarkup(schedule, goal) {
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
    if (!result) return `<article class="schedule-day rest-day"><p class="eyebrow">${label}</p><h2>Rest day</h2><p>Recovery between planned runs.</p></article>`;
    const extent = result.distance_range_miles
      ? `${number(result.distance_range_miles[0])}–${number(result.distance_range_miles[1])} miles`
      : result.duration_range_minutes
        ? `${number(result.duration_range_minutes[0], 0)}–${number(result.duration_range_minutes[1], 0)} minutes`
        : "No run";
    const weather = result.planned_weather;
    const weatherLine = weather
      ? `${number(weather.temperature_f, 0)}°F · feels like ${number(weather.apparent_temperature_f, 0)}°F · dew point ${number(weather.dewpoint_f, 0)}°F · wind ${number(weather.wind_speed_mph, 0)} mph`
      : "No forecast. Recent training is still included in the plan.";
    const steps = result.structure.map((step) => `<li>${escapeHtml(step.instruction)} <small>${step.target_zones.map(escapeHtml).join(" · ")}</small></li>`).join("");
    const reasons = result.reasons.map(escapeHtml).join(" ");
    const readinessExplanation = result.readiness !== "ready" && result.readiness_reason
      ? `<details><summary>Why is this flexible?</summary><p>${escapeHtml(result.readiness_reason)}</p></details>`
      : "";
    return `<article class="schedule-day ${result.readiness}"><div class="card-heading"><div><p class="eyebrow">${label} · ${titleCase(day.day_role)}</p><h2>${escapeHtml(result.title)}</h2></div><span class="quality ${result.confidence}">${readinessLabel(result.readiness)}</span></div><strong class="prescription-extent">${extent}</strong><p>${result.target_zones.map(escapeHtml).join(" · ") || "Recovery"}</p><small>${escapeHtml(weatherLine)}</small>${steps ? `<details><summary>Workout details</summary><ol>${steps}</ol></details>` : ""}${readinessExplanation}<details><summary>Why this workout?</summary><p>${reasons}</p></details></article>`;
  }).join("");
  const trailingCards = schedule.trailing_days.map((day) => {
    const activities = day.activities.map((activity) => `<a href="#run/${activity.activity_id}"><strong>${number(activity.distance_miles)} mi · ${titleCase(activity.workout_type)}</strong><small>${titleCase(activity.health_tag)}</small></a>`).join("");
    return `<article class="trailing-day ${day.activities.length ? "active-day" : "rest-day"}"><p class="eyebrow">${calendarDateLabel(day.date)}</p><h2>${titleCase(day.day_role)}</h2>${activities || "<small>No recorded activity</small>"}</article>`;
  }).join("");
  const evidence = schedule.target_evidence;
  // The planned week against the capacity it was derived from. Above or below
  // is information; only a big overshoot is a caution.
  const plannedMid = (schedule.projected_distance_range_miles[0] + schedule.projected_distance_range_miles[1]) / 2;
  const capacity = evidence.capacity_reference_miles;
  const plannedRatio = capacity > 0 ? plannedMid / capacity : null;
  const planDirection = plannedRatio === null ? DIRECTION.NONE
    : plannedRatio > 1.05 ? DIRECTION.UP : plannedRatio < 0.95 ? DIRECTION.DOWN : DIRECTION.FLAT;
  const planSentiment = plannedRatio === null ? SENTIMENT.NONE
    : plannedRatio >= 1.3 ? SENTIMENT.BAD : SENTIMENT.NEUTRAL;
  const versusCapacity = plannedRatio === null ? ""
    : `<p class="plan-versus">${directionValue(planDirection, planSentiment, `${number(plannedRatio * 100, 0)}% of your ${number(capacity)} mi/week demonstrated capacity`)}</p>`;
  const referenceText = evidence.capacity_reference_miles > 0
    ? `Your recent training supports about ${number(evidence.demonstrated_run_days_per_week, 1)} run days and ${number(schedule.target_distance_range_miles[0])}–${number(schedule.target_distance_range_miles[1])} miles per week.`
    : `This is a starter plan. It will adjust after more runs are uploaded.`;
  return `<div class="plan-summary-grid"><article class="wide-card schedule-summary"><div><p class="eyebrow">${calendarDateLabel(schedule.start_date)} through ${calendarDateLabel(schedule.end_date)}</p><h2>${schedule.completed_run_count ? `${schedule.completed_run_count} completed · ` : ""}${schedule.run_count} planned runs · ${number(schedule.projected_distance_range_miles[0])}–${number(schedule.projected_distance_range_miles[1])} miles</h2><p>${escapeHtml(schedule.summary)}</p>${versusCapacity}<small>${escapeHtml(referenceText)}</small><details><summary>How was this range chosen?</summary><p>${escapeHtml(evidence.rationale)}</p></details></div></article>${goalMarkup(goal, "panel")}</div><div class="week-grid">${dayCards}</div><article class="wide-card trailing-summary"><p class="eyebrow">Recent training</p><h2>Last 7 days</h2><div class="week-grid trailing-week">${trailingCards}</div></article>`;
}

export async function renderNextRun() {
  loading("Weekly plan");
  const [status, latest, state, goal] = await Promise.all([
    api("/api/current-status"), api("/api/weekly-schedule/latest"), api("/api/fitness-state"),
    api("/api/goal-progress").catch(() => null),
  ]);
  const choices = [
    ["normal", "Normal"], ["little_tired", "A little tired"],
    ["sick_or_recovering", "Sick / recovering"], ["pain_or_injury_concern", "Pain / injury concern"],
  ];
  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">Weekly plan</p><h1>Your next seven days</h1><p>Updates when you upload a run or change how you feel.</p></div></div>
      <form id="health-form" class="health-form">
        <fieldset><legend>How do you feel now?</legend><div class="health-options">${choices.map(([value, label]) => `<label><input type="radio" name="health_status" value="${value}" ${status.health_status === value ? "checked" : ""}><span>${label}</span></label>`).join("")}</div></fieldset>
        <button class="primary-button" type="submit">Update plan</button>
      </form>
      <div id="recommendation-result">${weeklyScheduleMarkup(latest, goal)}</div>
      <div class="state-strip"><span><b>${number(state.recent_load.trailing_7d.distance_miles)} mi</b>last 7 days</span><span><b>${Number.isFinite(state.recent_load.acute_distance_to_capacity_ratio) ? `${number(state.recent_load.acute_distance_to_capacity_ratio * 100, 0)}%` : "—"}</b>of usual weekly mileage</span><span><b>${number((state.moderate_fraction_14d ?? 0) * 100, 0)}%</b>moderate-intensity time</span><span><b>${state.running_days_28d}</b>run days in 28</span></div>
    </section>`;
  view.querySelector("#health-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const button = event.currentTarget.querySelector("button");
    button.disabled = true; button.textContent = "Updating…";
    try {
      const result = await api("/api/weekly-schedule", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ health_status: form.get("health_status") }),
      });
      view.querySelector("#recommendation-result").innerHTML = weeklyScheduleMarkup(result, goal);
    } catch (error) {
      view.querySelector("#recommendation-result").innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    } finally { button.disabled = false; button.textContent = "Update plan"; }
  });
}
