// Inline SVG chart rendering. Kept apart from the views so the geometry is
// readable on its own.

import { escapeHtml, dateLabel, number, pace } from "./format.js";
import { runChartAttributes } from './chart-interaction.js';
import { windowedChartSeries } from './chart-window.js';

export function paceChartScale(values) {
  const low=Math.min(...values),high=Math.max(...values);
  const padding=Math.max(.1,(high-low)*.05);
  const required=(high-low+2*padding)/5;
  const step=[.5,1,2,5,10,20,30,60].find(value=>value>=required)??Math.ceil(required/60)*60;
  const min=Math.max(0,Math.floor((low-padding)/step)*step);
  const max=Math.ceil((high+padding)/step)*step;
  return {min,max,ticks:Array.from({length:Math.round((max-min)/step)+1},(_,index)=>min+index*step)};
}

// The 7-day average was dropped deliberately. The individual runs are already
// on the chart, so a short smoother added no information the dots did not
// already carry -- it just drew a confident line through two weeks of noise.
export function fitnessChart(series, metric, trend28 = [], domainEnd = null, domainDays = null, options = {}) {
  const rawTrend=options.rawTrend ?? [];
  const adjustedTrend=options.rawTrend && metric==='raw'?[]:trend28;
  const shownRawTrend=['raw','both'].includes(metric)?rawTrend:[];
  const field = metric === "raw" ? "raw_pace_min_mile" : "standardized_pace_min_mile";
  const plottedPace = (item) => Number.isFinite(item[field])
    ? item[field]
    : item.trend_weight === 0 && Number.isFinite(item.context_pace_min_mile)
      ? item.context_pace_min_mile
      : null;
  const points = series.filter((item) => Number.isFinite(plottedPace(item)));
  if (!points.length) return '<div class="empty-state">No comparable fitness points are available.</div>';
  const timestamps = points.map((item) => new Date(item.start_time).getTime());
  const requestedEnd = domainEnd ? new Date(domainEnd).getTime() : null;
  const minX = Number.isFinite(requestedEnd) && Number.isFinite(domainDays) ? requestedEnd - domainDays * 86400000 : Math.min(...timestamps);
  const maxX = Number.isFinite(requestedEnd) && Number.isFinite(domainDays) ? requestedEnd : Math.max(...timestamps);
  // Adjusted/raw toggles share a domain so switching them cannot imply a change
  // in variability. Scale pace values and curves, never hidden uncertainty.
  // Full uncertainty remains in run inspection rather than stretching the plot.
  const comparableDisplay=Array.isArray(options.rawTrend);
  const scalePoints=comparableDisplay?series:points;
  const values=scalePoints.flatMap(item=>comparableDisplay||metric==='both'
    ? [item.standardized_pace_min_mile,item.raw_pace_min_mile,
        item.trend_weight===0&&!Number.isFinite(item.standardized_pace_min_mile)&&!Number.isFinite(item.raw_pace_min_mile)?item.context_pace_min_mile:null]
    : [plottedPace(item)]).filter(Number.isFinite);
  const clipTrend=trend=>windowedChartSeries(trend,minX,maxX,item=>[item.pace_min_mile],options.trendGapDays??70).groups;
  const adjustedGroups=clipTrend(adjustedTrend),rawGroups=clipTrend(shownRawTrend);
  const scaleGroups=comparableDisplay?[...clipTrend(trend28),...clipTrend(rawTrend)]:[...adjustedGroups,...rawGroups];
  values.push(...scaleGroups.flatMap(group=>group.map(item=>item.values[0])));
  const {min:minY,max:maxY,ticks}=paceChartScale(values);
  const x = (value) => 55 + (value - minX) / Math.max(1, maxX - minX) * 810;
  const y = (value) => 25 + (value - minY) / Math.max(.1, maxY - minY) * 250;
  const trendPath = groups => groups.map(group=>group.map((item,index)=>
    `${index?'L':'M'}${x(item.at).toFixed(1)},${y(item.values[0]).toFixed(1)}`).join(' ')).join(' ');
  const runAttributes=(item,cx,cy)=>options.interactive?` ${runChartAttributes({id:item.activity_id,x:cx,y:cy,title:options.runDateLabel?.(item.start_time)??dateLabel(item.start_time),details:[
    `${number(item.distance_miles)} mi${options.runRoles?.get(item.activity_id)?` · ${options.runRoles.get(item.activity_id)}`:''}`,
    ...(Number.isFinite(item.standardized_pace_min_mile)?[`${options.observedPace?'Observed aerobic pace':'Adjusted pace'}: ${pace(item.standardized_pace_min_mile)}`]:[]),
    ...(options.observedPace&&Number.isFinite(item.observed_average_hr_bpm)?[`Sampled HR: ${number(item.observed_average_hr_bpm,0)} bpm`]:[]),
    ...(!options.observedPace&&Number.isFinite(item.raw_pace_min_mile)?[`Unadjusted pace: ${pace(item.raw_pace_min_mile)}`]:Number.isFinite(item.context_pace_min_mile)?[`Recorded workout pace: ${pace(item.context_pace_min_mile)}`]:[]),
    ...(Number.isFinite(item.standardized_pace_min_mile)&&Number.isFinite(item.uncertainty_95_min_mile)?[`Estimated ${options.observedPace?'observed':'adjusted'} range: ${pace(item.standardized_pace_min_mile-item.uncertainty_95_min_mile)}–${pace(item.standardized_pace_min_mile+item.uncertainty_95_min_mile)}`]:[]),
    ...(item.terrain_quality?.usable===false?['Terrain adjustment not applied']:[]),
    ...(item.terrain_quality?.surface_profile_uncertain?['Surface grades estimated from recorded elevation']:[]),
    item.trend_weight===0?'Workout context only · not in the trend':item.trend_weight<1?'Used with less weight':'Used in the aerobic trend',
  ]})}`:'';
  const marks = points.map((item, index) => {
    const pointPace = plottedPace(item);
    const cx = x(timestamps[index]); const cy = y(pointPace);
    const trendWeight = Number.isFinite(item.trend_weight) ? item.trend_weight : 1;
    const estimated = ["device_distance_fallback", "partial_gps_device_distance"].includes(item.measurement_quality) || item.benchmark_quality === "estimated_fixed_time";
    const pointClass = [trendWeight === 0 ? "context-only" : trendWeight < 1 ? "reduced-evidence" : "", estimated ? "estimated-measurement" : ""].filter(Boolean).join(" ");
    const contextUsesRawPace = !Number.isFinite(item[field]) && Number.isFinite(item.context_pace_min_mile);
    const qualityText = contextUsesRawPace ? " · unadjusted full-run pace shown for context" : estimated ? " · pace estimated from available distance data" : "";
    const influenceText = options.plainLanguage
      ? trendWeight === 0 ? "workout only · not used in the aerobic trend" : trendWeight < 1 ? "used with less weight" : "used in the aerobic trend"
      : trendWeight === 0 ? "workout context only · 0% influence on the aerobic trend" : `${Math.round(trendWeight * 100)}% influence on the trend`;
    const uncertaintyText=Number.isFinite(item.standardized_pace_min_mile)&&Number.isFinite(item.uncertainty_95_min_mile)
      ? ` · estimated ${options.observedPace?'observed':'adjusted'} range ${pace(item.standardized_pace_min_mile-item.uncertainty_95_min_mile)}–${pace(item.standardized_pace_min_mile+item.uncertainty_95_min_mile)}`:'';
    return `<a href="#run/${item.activity_id}"><circle cx="${cx}" cy="${cy}" r="4" class="${pointClass}"${runAttributes(item,cx,cy)}><title>${dateLabel(item.start_time)} · ${pace(pointPace)} · ${number(item.distance_miles)} mi · ${influenceText}${qualityText}${uncertaintyText}</title></circle></a>`;
  }).join("");
  const rawMarks = metric === "both" ? points.filter((item) => Number.isFinite(item.standardized_pace_min_mile) && Number.isFinite(item.raw_pace_min_mile)).map((item) => {
    const cx = x(new Date(item.start_time).getTime()); const cy = y(item.raw_pace_min_mile);
    return `<a href="#run/${item.activity_id}"><rect x="${cx - 3}" y="${cy - 3}" width="6" height="6" class="raw-point"${runAttributes(item,cx,cy)}><title>${dateLabel(item.start_time)} · raw ${pace(item.raw_pace_min_mile)}</title></rect></a>`;
  }).join("") : "";
  const dateTick = (value) => new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
  const labels = ticks.map((value) => `<text x="48" y="${y(value) + 4}" text-anchor="end">${pace(value).replace("/mi", "")}</text><line x1="55" y1="${y(value)}" x2="865" y2="${y(value)}" class="grid-line"/>`).join("") + `<text x="55" y="303" text-anchor="start">${dateTick(minX)}</text><text x="865" y="303" text-anchor="end">${dateTick(maxX)}</text>`;
  const reducedLegend = points.some((item) => Number.isFinite(item.trend_weight) && item.trend_weight > 0 && item.trend_weight < 1) ? '<span><i class="reduced-key"></i>less comparable</span>' : "";
  const contextLegend = points.some((item) => item.trend_weight === 0) ? '<span><i class="context-key"></i>workout context only</span>' : "";
  const estimatedLegend = points.some((item) => ["device_distance_fallback", "partial_gps_device_distance"].includes(item.measurement_quality) || item.benchmark_quality === "estimated_fixed_time") ? '<span><i class="estimated-key"></i>estimated pace</span>' : "";
  const legend = options.observedPace ? `<div class="chart-legend"><span><i class="trend28-key"></i>28-day observed trend</span><span><i class="point-key"></i>aerobic run</span>${contextLegend}${reducedLegend}${estimatedLegend}</div>` : options.rawTrend ? `<div class="chart-legend">${adjustedTrend.length?'<span><i class="trend28-key"></i>Adjusted trend</span>':''}${shownRawTrend.length?'<span><i class="raw-trend-key"></i>Unadjusted trend</span>':''}<span><i class="point-key"></i>${metric==='raw'?'unadjusted':'adjusted'} run</span>${metric==='both'?'<span><i class="raw-key"></i>unadjusted run</span>':''}${contextLegend}${reducedLegend}${estimatedLegend}</div>` : trend28.length ? `<div class="chart-legend"><span><i class="trend28-key"></i>rolling 28-day estimate</span><span><i class="point-key"></i>adjusted run</span>${contextLegend}${reducedLegend}${estimatedLegend}${metric === "both" ? '<span><i class="raw-key"></i>unadjusted run</span>' : ""}</div>` : `<div class="chart-legend"><span><i class="point-key"></i>run</span>${contextLegend}${reducedLegend}${estimatedLegend}</div>`;
  return `<div class="chart-wrap"><span class="faster-label">Faster ↑</span>${legend}<svg class="fitness-chart" viewBox="0 0 900 310" role="img" aria-label="Fitness pace over time"${options.interactive?' data-run-chart data-run-top="25" data-run-bottom="275"':''}>${labels}<path d="${trendPath(adjustedGroups)}" class="chart-line trend-28"/>${shownRawTrend.length?`<path d="${trendPath(rawGroups)}" class="chart-line raw-trend"/>`:''}${marks}${rawMarks}</svg></div>`;
}

/**
 * A compact area chart for the VO2 estimate over the selected period.
 *
 * The band is the propagated 95% interval, drawn first so the line reads on
 * top of it: the width of that band is the honest part of this estimate and
 * hiding it would oversell the line.
 */
export function vo2Chart(series, options = {}) {
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
  if (series.length === 1 && !options.maxGapDays) {
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
  const segments=[[]];
  series.forEach((p,i)=>{
    if(i && options.maxGapDays && times[i]-times[i-1]>options.maxGapDays*86400000) segments.push([]);
    segments.at(-1).push(i);
  });
  const band = segments.map(indices=>[
    ...indices.map((i,j) => `${j ? "L" : "M"}${sx(times[i]).toFixed(1)},${sy(highs[i]).toFixed(1)}`),
    ...indices.map(i => `L${sx(times[i]).toFixed(1)},${sy(lows[i]).toFixed(1)}`).reverse(),
    "Z",
  ].join(" ")).join(" ");
  const line = series
    .map((p, i) => `${i&&(!options.maxGapDays||times[i]-times[i-1]<=options.maxGapDays*86400000) ? "L" : "M"}${sx(times[i]).toFixed(1)},${sy(p.value_ml_kg_min).toFixed(1)}`)
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
