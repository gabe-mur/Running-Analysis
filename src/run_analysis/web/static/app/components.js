// Small shared render pieces. Anything that appears on more than one screen
// lives here so the same idea cannot drift into two different treatments.

import { escapeHtml } from "./format.js";

// Direction and meaning are separate things and must be encoded separately.
//
// The arrow says which way the number moved. The colour says whether that is
// good news. They agree less often than you would expect: a pace figure rises
// when you get slower, a training-load ratio can climb past useful into
// strained, and a longest-run figure that falls is only bad if it was not a
// deliberate down week. Tying colour to the arrow would have the app cheer for
// a bigger number regardless of what the bigger number means.

/** Which way the number moved: "up" | "down" | "flat". */
export const DIRECTION = { UP: "up", DOWN: "down", FLAT: "flat", NONE: "none" };

/** Whether that movement is good news: "good" | "bad" | "neutral" | "none". */
export const SENTIMENT = { GOOD: "good", BAD: "bad", NEUTRAL: "neutral", NONE: "none" };

const ARROW = { up: "↑", down: "↓", flat: "↔", none: "·" };

/** Backend FitnessTrend values already carry meaning, not direction. */
export const trendSentiment = (trend) => ({
  improving: SENTIMENT.GOOD,
  declining: SENTIMENT.BAD,
  stable: SENTIMENT.NEUTRAL,
  uncertain: SENTIMENT.NEUTRAL,
}[trend] ?? SENTIMENT.NONE);

/** ...and for those, the arrow follows the meaning because the metric is
 *  already expressed so that "improving" means the good direction. */
export const trendDirection = (trend) => ({
  improving: DIRECTION.UP,
  declining: DIRECTION.DOWN,
  stable: DIRECTION.FLAT,
  uncertain: DIRECTION.FLAT,
}[trend] ?? DIRECTION.NONE);

export const trendArrow = (trend) => ARROW[trendDirection(trend)];

/**
 * An arrow and its text, with the colour driven by meaning rather than by the
 * arrow. Pass `sentiment` explicitly wherever the two can disagree.
 */
export function directionValue(direction, sentiment, text) {
  return `<span class="trend ${sentiment}"><i class="trend-arrow ${sentiment}">${ARROW[direction] ?? ARROW.none}</i>${escapeHtml(text)}</span>`;
}

/** Shorthand for a backend trend value, where direction and meaning agree. */
export function trendValue(trend, text) {
  return directionValue(trendDirection(trend), trendSentiment(trend), text);
}

/**
 * Compare a number with its previous value: the arrow follows the raw
 * movement, the colour follows `higherIsBetter`.
 */
export function comparisonValue(current, previous, text, { higherIsBetter = true, deadband = 0 } = {}) {
  if (!Number.isFinite(current) || !Number.isFinite(previous)) {
    return directionValue(DIRECTION.NONE, SENTIMENT.NONE, text);
  }
  const change = current - previous;
  if (Math.abs(change) <= deadband) {
    return directionValue(DIRECTION.FLAT, SENTIMENT.NEUTRAL, text);
  }
  const direction = change > 0 ? DIRECTION.UP : DIRECTION.DOWN;
  const good = change > 0 === higherIsBetter;
  return directionValue(direction, good ? SENTIMENT.GOOD : SENTIMENT.BAD, text);
}

/** One row of the dashboard's multi-signal grid. */
export function signalChip(signal, evidenceLabel) {
  const sentiment = trendSentiment(signal.trend);
  return `<span class="fitness-signal ${sentiment}">
    <b><i class="trend-arrow ${sentiment}">${trendArrow(signal.trend)}</i> ${escapeHtml(signal.label)}: <span class="trend-status ${sentiment}">${escapeHtml(signal.status)}</span></b>
    <small>${escapeHtml(signal.detail)} · ${evidenceLabel(signal.confidence)}</small>
  </span>`;
}

//: Training-status states mapped onto the same vocabulary, by meaning. A
//: rebuilding athlete is doing the right thing, so it is not amber.
const STATUS_SENTIMENT = {
  building: SENTIMENT.GOOD,
  maintaining: SENTIMENT.NEUTRAL,
  rebuilding: SENTIMENT.GOOD,
  recovering: SENTIMENT.BAD,
  strained: SENTIMENT.BAD,
  underloaded: SENTIMENT.BAD,
  insufficient_data: SENTIMENT.NONE,
};

export const statusSentiment = (status) => STATUS_SENTIMENT[status] ?? SENTIMENT.NONE;

//: Effort-distribution verdicts on the same three-colour vocabulary. Only the
//: two shapes that call for a change are amber; an all-easy block is a
//: legitimate choice, not a fault, so it stays neutral.
const BALANCE_SENTIMENT = {
  balanced: SENTIMENT.GOOD,
  grey_zone: SENTIMENT.BAD,
  too_hard: SENTIMENT.BAD,
  no_hard_stimulus: SENTIMENT.NEUTRAL,
  nearly_all_easy: SENTIMENT.NEUTRAL,
  insufficient_data: SENTIMENT.NONE,
};

export const balanceSentiment = (balance) => BALANCE_SENTIMENT[balance] ?? SENTIMENT.NONE;
