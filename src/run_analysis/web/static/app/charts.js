// Inline SVG chart rendering. Kept apart from the views so the geometry is
// readable on its own.

import { escapeHtml, dateLabel, number, pace } from "./format.js";

// The 7-day average was dropped deliberately. The individual runs are already
// on the chart, so a short smoother added no information the dots did not
// already carry -- it just drew a confident line through two weeks of noise.
export function fitnessChart(series, metric, trend28 = [], domainEnd = null, domainDays = null) {
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
    const qualityText = estimated ? " · pace estimated from available distance data" : "";
    return `${uncertainty ? `<line x1="${cx}" y1="${y(item[field] - uncertainty)}" x2="${cx}" y2="${y(item[field] + uncertainty)}" class="uncertainty-mark"/>` : ""}<a href="#run/${item.activity_id}"><circle cx="${cx}" cy="${cy}" r="4" class="${pointClass}"><title>${dateLabel(item.start_time)} · ${pace(item[field])} · ${number(item.distance_miles)} mi · ${Math.round(trendWeight * 100)}% influence on the trend${qualityText}</title></circle></a>`;
  }).join("");
  const rawMarks = metric === "both" ? points.filter((item) => Number.isFinite(item.raw_pace_min_mile)).map((item) => {
    const cx = x(new Date(item.start_time).getTime()); const cy = y(item.raw_pace_min_mile);
    return `<a href="#run/${item.activity_id}"><rect x="${cx - 3}" y="${cy - 3}" width="6" height="6" class="raw-point"><title>${dateLabel(item.start_time)} · raw ${pace(item.raw_pace_min_mile)}</title></rect></a>`;
  }).join("") : "";
  const dateTick = (value) => new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
  const labels = [minY, (minY + maxY) / 2, maxY].map((value) => `<text x="48" y="${y(value) + 4}" text-anchor="end">${pace(value).replace("/mi", "")}</text><line x1="55" y1="${y(value)}" x2="865" y2="${y(value)}" class="grid-line"/>`).join("") + `<text x="55" y="303" text-anchor="start">${dateTick(minX)}</text><text x="865" y="303" text-anchor="end">${dateTick(maxX)}</text>`;
  const reducedLegend = points.some((item) => Number.isFinite(item.trend_weight) && item.trend_weight < 1) ? '<span><i class="reduced-key"></i>less comparable</span>' : "";
  const estimatedLegend = points.some((item) => ["device_distance_fallback", "partial_gps_device_distance"].includes(item.measurement_quality) || item.benchmark_quality === "estimated_fixed_time") ? '<span><i class="estimated-key"></i>estimated pace</span>' : "";
  const legend = trend28.length ? `<div class="chart-legend"><span><i class="trend28-key"></i>28-day average</span><span><i class="point-key"></i>adjusted run</span>${reducedLegend}${estimatedLegend}${metric === "both" ? '<span><i class="raw-key"></i>unadjusted run</span>' : ""}</div>` : `<div class="chart-legend"><span><i class="point-key"></i>run</span>${reducedLegend}${estimatedLegend}</div>`;
  return `<div class="chart-wrap"><span class="faster-label">Faster ↑</span>${legend}<svg class="fitness-chart" viewBox="0 0 900 310" role="img" aria-label="Fitness pace over time">${labels}<path d="${trendPath(trend28, 70)}" class="chart-line trend-28"/>${marks}${rawMarks}</svg></div>`;
}

/**
 * A compact area chart for the VO2 estimate over the selected period.
 *
 * The band is the propagated 95% interval, drawn first so the line reads on
 * top of it: the width of that band is the honest part of this estimate and
 * hiding it would oversell the line.
 */
export function vo2Chart(series) {
  // An optional field the server may not send: a missing series is a chart
  // that cannot be drawn, not a page that fails to load. One point is still
  // worth drawing -- a flat line is a real answer, and blanking the panel
  // reads as breakage rather than as "nothing moved".
  if (!Array.isArray(series) || !series.length) {
    return `<p class="chart-empty">No scored runs in this period yet.</p>`;
  }
  // A lone point is stretched into a flat segment purely so it has width to be
  // drawn across. The axis labels below still name the one real date.
  const firstLabel = dateLabel(series[0].as_of);
  const lastLabel = dateLabel(series[series.length - 1].as_of);
  if (series.length === 1) {
    series = [series[0], { ...series[0], as_of: new Date(+new Date(series[0].as_of) + 86400000).toISOString() }];
  }
  const W = 520, H = 210, pad = { l: 40, r: 14, t: 14, b: 26 };
  const times = series.map((point) => +new Date(point.as_of));
  const lows = series.map((point) => point.value_ml_kg_min - point.uncertainty_95_ml_kg_min);
  const highs = series.map((point) => point.value_ml_kg_min + point.uncertainty_95_ml_kg_min);
  const xMin = Math.min(...times), xMax = Math.max(...times);
  const yMin = Math.min(...lows) - 1, yMax = Math.max(...highs) + 1;
  const sx = (t) => pad.l + ((t - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const sy = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);

  // Out along the upper bound, back along the lower one. The return leg must
  // run right-to-left: walk it in the same order and the polygon crosses
  // itself into a bowtie instead of enclosing the interval.
  const band = [
    ...series.map((p, i) => `${i ? "L" : "M"}${sx(times[i]).toFixed(1)},${sy(highs[i]).toFixed(1)}`),
    ...series.map((p, i) => `L${sx(times[i]).toFixed(1)},${sy(lows[i]).toFixed(1)}`).reverse(),
    "Z",
  ].join(" ");
  const line = series
    .map((p, i) => `${i ? "L" : "M"}${sx(times[i]).toFixed(1)},${sy(p.value_ml_kg_min).toFixed(1)}`)
    .join(" ");
  const ticks = [yMin, (yMin + yMax) / 2, yMax]
    .map((v) => `<line class="vo2-grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"></line>`
      + `<text class="vo2-tick" x="${pad.l - 7}" y="${(sy(v) + 4).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`)
    .join("");
  const last = series[series.length - 1];
  return `<svg class="vo2-chart" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Estimated VO2 max from ${firstLabel} to ${lastLabel}">
    ${ticks}
    <path class="vo2-band" d="${band}"></path>
    <path class="vo2-line" d="${line}"></path>
    <circle class="vo2-point" cx="${sx(times[times.length - 1]).toFixed(1)}" cy="${sy(last.value_ml_kg_min).toFixed(1)}" r="4"></circle>
    <text class="vo2-tick" x="${pad.l}" y="${H - 8}">${firstLabel.split(",")[0]}</text>
    <text class="vo2-tick" x="${W - pad.r}" y="${H - 8}" text-anchor="end">${lastLabel.split(",")[0]}</text>
  </svg>`;
}
