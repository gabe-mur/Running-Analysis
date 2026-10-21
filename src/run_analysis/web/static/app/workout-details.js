import { escapeHtml, number } from "./format.js";

const time = minutes => {
  const seconds = Math.round(minutes * 60);
  return seconds % 60 === 0
    ? `${seconds / 60} min`
    : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};
const listItem = (text, zones = [], extra = "") => `<li>${escapeHtml(text)}${zones.length ? ` <small>${zones.map(escapeHtml).join(" · ")}</small>` : ""}${extra}</li>`;
const usesDistance = step => step.prescription_basis === 'distance'
  || (step.prescription_basis == null && Number.isFinite(step.distance_miles)
    && !Number.isFinite(step.duration_minutes));
const measure = step => {
  if (!usesDistance(step) || !Number.isFinite(step.distance_miles) || step.distance_miles <= 0) return time(step.duration_minutes);
  if (step.phase === 'work' || step.phase === 'recovery') {
    const meters = step.distance_miles * 1609.344;
    const quantum = meters < 100 ? 10 : 50;
    return `~${Math.max(quantum, Math.round(meters / quantum) * quantum)} m`;
  }
  return `~${number(step.distance_miles)} miles`;
};

// Only compact the adapter's machine-generated phase list. Existing coached
// instructions (progressions, hills, fartlek, threshold, etc.) stay intact.
const phaseLabels = {
  warmup: "Easy warm-up", cooldown: "Easy cool-down", recovery: "Easy jogging recovery",
  work: "Controlled hard effort; do not chase an instantaneous HR target",
};
const machinePhase = step => Number.isFinite(step.duration_minutes) && step.duration_minutes > 0
  && (step.instruction.startsWith(`${phaseLabels[step.phase]} · `)
    || step.instruction.startsWith("Easy aerobic running · "));

function compactPhases(steps) {
  if (steps.length === 1 && steps[0].instruction.startsWith("Easy aerobic running · ")) {
    return listItem(`Easy aerobic running · ${usesDistance(steps[0]) && Number.isFinite(steps[0].distance_miles) ? `${number(steps[0].distance_miles)} miles` : measure(steps[0])}`, steps[0].target_zones);
  }
  const first = steps[0];
  const last = steps.at(-1);
  const middle = steps.slice(1, -1);
  if (first.phase !== "warmup" || last.phase !== "cooldown" || !middle.length
    || middle.some((step, i) => step.phase !== (i % 2 === 0 ? "work" : "recovery"))
    || middle.at(-1).phase !== "work") return null;
  const work = middle.filter(step => step.phase === "work");
  const recovery = middle.filter(step => step.phase === "recovery");
  const uniformWork = work.every(step => measure(step) === measure(work[0]));
  const uniformRecovery = recovery.every(step => measure(step) === measure(recovery[0]));
  const range = rows => {
    const field = rows.every(step => usesDistance(step) && Number.isFinite(step.distance_miles)) ? 'distance_miles' : 'duration_minutes';
    const ordered = [...rows].sort((a,b) => a[field] - b[field]);
    const low = measure(ordered[0]), high = measure(ordered.at(-1));
    return low === high ? low : `${low}–${high}`;
  };
  let main = uniformWork
    ? `${work.length} × ${measure(work[0])} at controlled hard effort`
    : `${work.length} efforts of ${range(work)} at controlled hard effort`;
  main += recovery.length
    ? `, with ${uniformRecovery ? measure(recovery[0]) : range(recovery)} of easy jogging between efforts.`
    : ".";
  // Never turn uneven issued repetitions into a newly invented uniform workout.
  const timings = [
    uniformWork ? "" : `Efforts: ${work.map(measure).join(" · ")}`,
    uniformRecovery ? "" : `Recoveries: ${recovery.map(measure).join(" · ")}`,
  ].filter(Boolean).map(text => `<small class="workout-timings">${escapeHtml(text)}</small>`).join("");
  const exactTimings = timings
    ? `<details class="workout-exact-timings"><summary>Effort breakdown</summary>${timings}</details>`
    : "";
  return listItem(`Easy warm-up · ${measure(first)}`, first.target_zones)
    + listItem(main, [], exactTimings)
    + listItem(`Easy cool-down · ${measure(last)}`, last.target_zones);
}

export function workoutDetailsMarkup(result) {
  const steps = result.structure ?? [];
  if (!steps.length) return "";
  const compact = steps.every(machinePhase) ? compactPhases(steps) : null;
  const items = compact ?? steps.map(step => {
    // Keep human-written instructions verbatim; only replace generated decimal durations.
    const text = machinePhase(step)
      ? `${step.instruction.split(" · ")[0]} · ${measure(step)}`
      : step.instruction;
    return listItem(text, step.target_zones ?? []);
  }).join("");
  const margin = result.distance_target_miles != null && result.distance_range_miles
    ? `<small class="workout-distance-margin">Distance margin: ${number(result.distance_range_miles[0])}–${number(result.distance_range_miles[1])} miles.</small>` : "";
  const estimate = steps.some(step => machinePhase(step) && usesDistance(step) && Number.isFinite(step.distance_miles)) && result.planning_role === 'quality'
    ? '<small class="workout-distance-margin">Segment distances are approximate.</small>' : '';
  return `<details class="workout-details"><summary>Workout details</summary><ol>${items}</ol>${estimate}${margin}</details>`;
}
