// Progress toward the stated goal, rendered the same way wherever it appears.
//
// The weekly plan says what to run; this says what it is all for. Both screens
// show the identical read-out because two versions of "how the goal is going"
// would eventually disagree in front of the athlete.

import { SENTIMENT } from "./components.js";
import { escapeHtml, number } from "./format.js";

//: Meaning, not direction. Being ahead of a goal is good news; being behind it
//: is worth attention but is not a failure with months still to run.
const GOAL_SENTIMENT = {
  ahead: SENTIMENT.GOOD,
  on_track: SENTIMENT.GOOD,
  behind: SENTIMENT.BAD,
  insufficient_evidence: SENTIMENT.NONE,
  no_goal: SENTIMENT.NONE,
};

export const goalSentiment = (status) => GOAL_SENTIMENT[status] ?? SENTIMENT.NONE;

const STATUS_LABEL = {
  ahead: "Already there",
  on_track: "On track",
  behind: "Behind the date",
  insufficient_evidence: "Not enough runs yet",
  no_goal: "No goal set",
};

/**
 * @param {object} goal   the /api/goal-progress payload
 * @param {"panel"|"wide"} shape  side panel, or full-width band
 */
export function goalMarkup(goal, shape = "panel") {
  if (!goal) return "";
  const sentiment = goalSentiment(goal.status);
  const paces = goal.goal_pace_min_mile && goal.supported_pace_min_mile
    ? `<div class="goal-paces">
         <span><small>Target</small><strong>${paceText(goal.goal_pace_min_mile)}</strong></span>
         <span><small>Supported now</small><strong>${paceText(goal.supported_pace_min_mile)}</strong></span>
         <span><small>Gap</small><strong>${goal.gap_seconds_per_mile <= 0 ? "None" : `${number(goal.gap_seconds_per_mile, 0)} s/mi`}</strong></span>
       </div>`
    : "";
  const action = shape === "panel"
    ? `<span class="panel-action">Progress →</span>`
    : "";
  const tag = shape === "panel" ? "a" : "div";
  const href = shape === "panel" ? ' href="#progress"' : "";
  return `<${tag} class="goal-panel ${shape}"${href}>
    <div class="goal-heading">
      <p class="eyebrow">Your goal</p>
      <span class="trend-status ${sentiment}">${escapeHtml(STATUS_LABEL[goal.status] ?? "")}</span>
    </div>
    <h2>${escapeHtml(goal.headline)}</h2>
    ${paces}
    <p class="goal-detail">${escapeHtml(goal.detail)}</p>
    ${action}
  </${tag}>`;
}

function paceText(value) {
  const total = Math.round(value * 60);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}/mi`;
}
