import { daypartLabel, escapeHtml, readinessLabel, readinessClass } from "./format.js";

const choices = [
  ["normal", "Normal"], ["little_tired", "A little tired"],
  ["sick_or_recovering", "Sick / recovering"], ["pain_or_injury_concern", "Pain / injury concern"],
];

export function checkInFieldsMarkup(status) {
  // Show the saved selection; only submitting the form records a fresh report.
  return `<fieldset><legend>How do you feel now?</legend><div class="health-options">${choices.map(([value, label]) => `<label><input type="radio" name="health_status" value="${value}" required ${status.health_status === value ? "checked" : ""}><span>${label}</span></label>`).join("")}</div></fieldset>`;
}

export function advisoryDaypart(result) {
  if (result?.time_recommendation_basis === "flexible") return "Flexible time";
  return daypartLabel(result?.planned_for);
}

export function readinessBadgeLabel(result) {
  return result.readiness_label ?? readinessLabel(result.readiness);
}

export function runWarningMarkup(result) {
  // Missing/normal readiness is not a warning, and should not repeat the
  // single check-in control across every run card. Keep real concerns visible.
  if (!["caution", "not_ready"].includes(result.readiness)) return { badge: "", explanation: "" };
  return {
    badge: `<span class="quality ${readinessClass(result.readiness)}" title="${escapeHtml(result.readiness_reason || "")}">${escapeHtml(readinessBadgeLabel(result))}</span>`,
    explanation: result.readiness_reason
      ? `<details><summary>Why this warning?</summary><p>${escapeHtml(result.readiness_reason)}</p></details>` : "",
  };
}
