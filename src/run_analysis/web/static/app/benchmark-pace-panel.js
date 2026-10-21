// Direct performance prototype. Older coherent fixtures remain renderable.
import { escapeHtml, number, pace, titleCase } from './format.js';
import { paceChartScale } from './charts.js';
import { windowedChartSeries } from './chart-window.js';
import { runChartAttributes } from './chart-interaction.js';

const usable = value => Number.isFinite(value) && value > 0;
const heartRate = value => usable(value) ? `${number(value, 0)} bpm` : 'the same heart rate';
const visibleRun = (row, hidden) => !hidden.has(row.activity_id) && !row.manually_excluded
  && !['hike','bike'].includes(row.workout_type);
const estimatedRun = row => ['device_distance_fallback','partial_gps_device_distance'].includes(row.measurement_quality);
const reducedReason = row => [row.workout_type==='run_walk'?'run/walk structure':null,
  row.health_tag && row.health_tag!=='normal'?titleCase(row.health_tag):null].filter(Boolean).join(' · ') || 'less comparable context';

function qualitySessionPoints(state,historyDays) {
  const hidden=new Set(state.hidden_activity_ids??[]), end=+new Date(state.as_of),start=end-historyDays*86400000;
  return (state.quality_sessions??[]).filter(row=>visibleRun(row,hidden)
    && usable(row.observed_pace_min_mile)
    && +new Date(row.as_of)>=start && +new Date(row.as_of)<=end);
}

export function benchmarkPaceChart(state, date, historyDays, targetHr, metric='standardized') {
  const compact = globalThis.matchMedia?.('(max-width: 40rem)')?.matches ?? false;
  const width = compact ? 420 : 890, height = compact ? 280 : 300;
  const left = 82, right = width - 28, top = 24, bottom = height - 38;
  const end = +new Date(state.as_of), start = end - historyDays * 86400000;
  if (!Number.isFinite(end) || !Number.isFinite(historyDays) || historyDays <= 0) return '';
  const direct = Array.isArray(state.observations);
  const hidden=new Set(state.hidden_activity_ids??[]);
  const adjusted = windowedChartSeries(state.history ?? [], start, end,
    row => usable(row.pace_min_mile) ? [row.pace_min_mile] : null, direct ? 28 : 14);
  const groups=metric==='raw'?[]:adjusted.groups;
  const rawGroups=metric==='standardized'?[]:windowedChartSeries(state.unadjusted?.history??[],start,end,
    row=>usable(row.pace_min_mile)?[row.pace_min_mile]:null,28).groups;
  const comparison = direct && metric==='standardized' ? state.smoothing_comparison : null;
  const comparisonGroups = comparison ? windowedChartSeries(comparison.history ?? [], start, end,
    row => usable(row.pace_min_mile) ? [row.pace_min_mile] : null, 28).groups : [];
  const observations=direct?state.observations.filter(row=>visibleRun(row,hidden)&&row.trend_weight!==0):[];
  const adjustedPoints=direct?windowedChartSeries(observations,start,end,
    row=>usable(row.pace_min_mile)?[row.pace_min_mile]:null,28).points:adjusted.points;
  const rawPoints=direct?windowedChartSeries(observations,start,end,
    row=>usable(row.observed_pace_min_mile)?[row.observed_pace_min_mile]:null,28).points.map(point=>({...point,raw:true})):[];
  const points=[...(metric==='raw'?[]:adjustedPoints),...(metric==='standardized'?[]:rawPoints)];
  const qualityPoints=qualitySessionPoints(state,historyDays);
  if (!groups.length && !rawGroups.length && !qualityPoints.length) return '<div class="empty-state">Your first comparable runs will start this line.</div>';
  // Reuse the established pace axis: real min/mi, at least 30-second tick
  // spacing. Neither percentage changes nor large VO2 conversion ranges
  // rescale the curve. Full level uncertainty stays in date inspection.
  const scale = paceChartScale([...groups.flatMap(group => group.map(point => point.values[0])),
    ...rawGroups.flatMap(group=>group.map(point=>point.values[0])),
    ...comparisonGroups.flatMap(group => group.map(point => point.values[0])),
    ...points.map(point => point.values[0]),...qualityPoints.map(row=>row.observed_pace_min_mile)]);
  const x = value => left + (value - start) / Math.max(1, end - start) * (right - left);
  const y = value => top + (value - scale.min) / Math.max(.1, scale.max - scale.min) * (bottom - top);
  const axes = scale.ticks.map(value => `<line class="grid-line" x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left - 12}" y="${y(value) + 4}" text-anchor="end">${escapeHtml(pace(value).replace('/mi', ''))}</text>`).join('');
  const lines = groups.map(group => `<path class="benchmark-pace-line" d="${group.map((point, index) => `${index ? 'L' : 'M'}${x(point.at).toFixed(1)},${y(point.values[0]).toFixed(1)}`).join(' ')}"/>`).join('');
  const rawLines=rawGroups.map(group=>`<path class="benchmark-pace-line unadjusted" d="${group.map((point,index)=>`${index?'L':'M'}${x(point.at).toFixed(1)},${y(point.values[0]).toFixed(1)}`).join(' ')}"/>`).join('');
  const comparisonLines = comparisonGroups.map(group => `<path class="benchmark-pace-line comparison14" d="${group.map((point, index) => `${index ? 'L' : 'M'}${x(point.at).toFixed(1)},${y(point.values[0]).toFixed(1)}`).join(' ')}"/>`).join('');
  const qualityMarks=qualityPoints.map(row=>{
    const cx=x(+new Date(row.as_of)),cy=y(row.observed_pace_min_mile);
    const label=row.workout_type==='tempo_threshold'?'Tempo / progression':titleCase(row.workout_type);
    const details=[`${usable(row.distance_miles)?`${number(row.distance_miles)} mi · `:''}${label}`,
      `Recorded pace: ${pace(row.observed_pace_min_mile)}`, 'Quality · not in the aerobic trend'];
    return `<a href="#run/${row.activity_id}"><g ${runChartAttributes({id:`benchmark-quality:${row.activity_id}`,x:cx,y:cy,title:date(row.as_of),details})}><path class="benchmark-quality-point" d="M${cx},${cy-5}l5,5 -5,5 -5,-5Z"/></g></a>`;
  }).join('');
  const marks = points.map(point => {
    const row = point.row, range = row.estimated_range_95_min_mile;
    const latest = point.at === +new Date(state.last_observation);
    const details = direct ? [
      [usable(row.distance_miles)?`${number(row.distance_miles)} mi`:null,
        row.workout_type==='unknown'?'Run':titleCase(row.workout_type)].filter(Boolean).join(' · '),
      point.raw?`Unadjusted pace: ${pace(row.observed_pace_min_mile)}`:`Adjusted pace: ${pace(row.pace_min_mile)}`,
      ...(!point.raw&&usable(row.observed_pace_min_mile)?[`Unadjusted pace: ${pace(row.observed_pace_min_mile)}`]:[]),
      ...(row.trend_weight<1?[`Reduced weight · ${reducedReason(row)}`]:[]),
      ...(estimatedRun(row)?['Limited GPS · device distance used']:[]),
    ] : [`Comparable pace: ${pace(row.pace_min_mile)}`, `At ${heartRate(targetHr)}`,
      `${row.run_count ?? 0} contributing runs`,
      ...(row.evidence === 'provisional' ? ['Early estimate · still gathering evidence'] : []),
      ...(range?.length === 2 && range.every(usable) ? [`Model range: ${pace(range[0])}–${pace(range[1])}`] : []),
      'Recent-performance estimate, not this individual run’s pace'];
    const mark=`<g ${runChartAttributes({ id: `${point.raw?'benchmark-raw':'benchmark'}:${row.activity_id ?? row.as_of}`, x: x(point.at), y: y(point.values[0]), title: date(row.as_of), details })}><circle class="benchmark-pace-point${point.raw?' unadjusted':''}${row.evidence === 'provisional' ? ' provisional' : ''}${row.trend_weight<1?' reduced-evidence':''}${estimatedRun(row)?' estimated-measurement':''}${latest ? ' latest' : ''}" cx="${x(point.at)}" cy="${y(point.values[0])}" r="${latest ? 5 : 4}"/></g>`;
    return row.activity_id!=null?`<a href="#run/${row.activity_id}">${mark}</a>`:mark;
  }).join('');
  return `<p class="benchmark-pace-axis-note"><span class="benchmark-faster">Faster ↑</span><span>${metric==='raw'?'min/mi · recorded pace':metric==='both'?`min/mi · adjusted at ${escapeHtml(heartRate(targetHr))}`:`min/mi · ${escapeHtml(heartRate(targetHr))}`}</span></p>
    <svg class="insight-chart benchmark-pace-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Comparable aerobic pace over time${comparison?'; 14-day and 28-day smoothing comparison':''}; faster pace is higher" data-run-chart data-run-top="${top}" data-run-bottom="${bottom}" data-chart-description="Hover or tap to inspect a performance estimate. Left and right arrows select dates.">
      ${axes}${comparisonLines}${lines}${rawLines}${marks}${qualityMarks}
      <text x="${left}" y="${height - 10}">${escapeHtml(date(new Date(start).toISOString()))}</text><text x="${right}" y="${height - 10}" text-anchor="end">${escapeHtml(date(state.as_of))}</text>
    </svg>`;
}

export function benchmarkPacePanel(state, date, historyDays, targetHr, metric='standardized', {showSummary=true}={}) {
  if (!state) return '';
  const current = metric==='raw'?state.unadjusted?.current:state.current;
  const available = (metric==='raw'?state.unadjusted?.available:state.available) && usable(current?.pace_min_mile);
  const direct = Array.isArray(state.observations);
  const hr = targetHr ?? state.target_hr_bpm ?? state.fitness_level?.target_hr_bpm;
  const early = current?.evidence === 'provisional' || (direct && current?.run_count<3);
  const comparison = direct && metric==='standardized' && usable(state.smoothing_comparison?.current?.pace_min_mile) ? state.smoothing_comparison : null;
  const hidden=new Set(state.hidden_activity_ids??[]);
  const visible=direct?state.observations.filter(row=>visibleRun(row,hidden)&&row.trend_weight!==0):[];
  const hasQuality=qualitySessionPoints(state,historyDays).length>0;
  const legend=direct&&(visible.length||hasQuality)?`<div class="benchmark-evidence-legend"><span><i class="${metric==='raw'?'unadjusted-key':'normal-key'}"></i>${metric==='both'?'Adjusted run':'Aerobic run'}</span>${metric==='both'?'<span><i class="unadjusted-key"></i>Unadjusted run</span>':''}${visible.some(row=>row.trend_weight<1)?'<span><i class="reduced-key"></i>Reduced weight</span>':''}${visible.some(estimatedRun)?'<span><i class="estimated-key"></i>Limited GPS</span>':''}${hasQuality?'<span><i class="quality-key"></i>Quality · recorded pace</span>':''}</div>`:'';
  const paceLegend=`<div class="benchmark-comparison-legend" aria-label="Pace trends">${metric!=='raw'?'<span><i></i>Adjusted trend</span>':''}${metric!=='standardized'?'<span><i class="unadjusted-key"></i>Unadjusted trend</span>':''}</div>`;
  return `<article class="wide-card benchmark-pace-card">
    <div class="card-heading"><div><p class="eyebrow">Comparable pace</p><h2>Aerobic performance</h2></div>${comparison?'<span class="quality">Compare smoothing</span>':''}</div>
    ${available ? `${showSummary?`<div class="benchmark-pace-summary"><strong>${escapeHtml(pace(current.pace_min_mile))}</strong><div><span>${metric==='raw'?'Recorded pace':`At ${escapeHtml(heartRate(hr))}`}</span><small>${metric==='raw'?'Unadjusted trend':comparison?'Current 28-day trend':'Adjusted for conditions'}${early ? ' · Early estimate' : ''}</small></div></div>`:''}
      ${comparison?`<div class="benchmark-comparison-legend" aria-label="Trend smoothing comparison"><span><i class="comparison14-key" aria-hidden="true"></i>14-day · more responsive <b>${escapeHtml(pace(comparison.current.pace_min_mile))}</b></span><span><i class="comparison28-key" aria-hidden="true"></i>28-day · smoother <b>${escapeHtml(pace(current.pace_min_mile))}</b></span></div>`:''}
      <div class="benchmark-chart-key">${paceLegend}${legend}</div>
      ${benchmarkPaceChart(state, date, historyDays, hr, metric)}
      <p class="chart-note">${metric==='both'?'Solid green = adjusted trend · dashed amber = unadjusted trend.':comparison?'Same run dots, different smoothing. Neither line changes your plan.':direct?'Line = performance trend · dots = individual runs.':'Each point summarizes recent running—not a single run.'}${state.days_since_observation > 7 ? ` Last estimate: ${escapeHtml(date(state.last_observation))}.` : ''}</p>
      <details><summary>About this estimate</summary><p>${direct?'The line fits comparable run observations directly, with more influence from nearby dates. Later runs can revise the earlier trend. Long gaps stay blank, and changing the displayed history does not refit the line.':'Estimates your pace at the same heart rate, terrain, weather and stage of a run. Each date uses up to 28 days of available evidence; early estimates use the runs available so far. Changing the displayed history does not change the current value.'}</p><p>A performance benchmark, not a prescribed training pace or measured VO₂max. Fatigue and other unmeasured influences still affect it. ${direct?'Strong or weak runs are not removed simply for being unusual. <a href="#pace-comparison">See individual pace adjustments →</a>':'Inspect a date for run counts and model uncertainty.'}</p>${state.temperature_response?'<p>Temperature is compared with the same mild conditions, using your history to refine a broad starting estimate. No separate humidity correction is fitted.</p>':''}${state.section_screen?`<p>Each score needs at least two separate sections with settled effort and HR, including the running just before and after them. Acceleration, slowing and recovery are not treated as steady aerobic evidence. All runs still count toward training load.</p>${state.section_screen.excluded_runs?.length?`<p>Not enough settled sections in ${state.section_screen.excluded_runs.length} run${state.section_screen.excluded_runs.length===1?'':'s'} shown in this period:</p><ul>${state.section_screen.excluded_runs.map(r=>`<li><a href="#run/${r.activity_id}">${escapeHtml(date(r.start_time))}</a> · changing effort or HR</li>`).join('')}</ul>`:''}`:''}</details>`
      : `<div class="empty-state">Your first comparable runs will establish an initial pace. You don’t need months of history to get started.</div>${hasQuality?`${legend}${benchmarkPaceChart(state,date,historyDays,hr,metric)}`:''}`}
  </article>`;
}
