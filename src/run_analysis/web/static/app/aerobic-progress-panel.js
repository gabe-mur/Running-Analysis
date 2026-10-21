// Additive prototype: existing pace, VO2 and planner presentations stay intact.
import { escapeHtml, number } from './format.js';
import { runChartAttributes } from './chart-interaction.js';
import { windowedChartSeries } from './chart-window.js';

const percent=value=>Number.isFinite(value)?`${value>0?'+':''}${number(value,1)}%`:'—';
const verdict=change=>!change?'More history needed':change.direction==='provisional'?'Early estimate':
  change.inference_method==='bootstrap-likelihood'&&change.evidence_strength!=='clear'?
    Math.abs(change.speed_change_percent)<.05?'No measurable change':change.speed_change_percent>0?'Estimated improvement':'Estimated decline':
  change.evidence_strength==='clear'?(change.direction==='improving'?'Improving':'Declining'):
  change.evidence_strength==='likely'?(change.direction==='improving'?'Likely improving':'Likely declining'):'No clear change';

export function aerobicProgressChart(state,date,historyDays) {
  const compact=globalThis.matchMedia?.('(max-width: 40rem)')?.matches??false;
  const width=compact?420:890,height=compact?280:295,left=58,right=width-28,bottom=height-37;
  const end=+new Date(state.as_of),start=end-historyDays*86400000;
  const {groups,points}=windowedChartSeries(state.history??[],start,end,r=>Number.isFinite(r.change?.speed_change_percent)
    ?[r.change.speed_change_percent,...(r.change.estimated_change_range_95_percent??[])]:null,14);
  const rows=points.map(p=>p.row);
  if(!groups.length) return '<div class="empty-state">More comparable run history is needed to show change between months.</div>';
  const values=[0,...groups.flatMap(g=>g.flatMap(p=>p.values))];
  const extent=Math.max(...values)-Math.min(...values);
  const step=[1,2,5,10,20,50,100].find(v=>v>=extent/4)??Math.ceil(extent/400)*100;
  const low=Math.floor((Math.min(...values)-step*.15)/step)*step;
  const high=Math.ceil((Math.max(...values)+step*.15)/step)*step;
  const x=r=>left+((r.at??+new Date(r.as_of))-start)/Math.max(1,end-start)*(right-left);
  const y=v=>bottom-(v-low)/Math.max(1,high-low)*(bottom-24);
  const line=group=>group.map((r,i)=>`${i?'L':'M'}${x(r).toFixed(1)},${y(r.values[0]).toFixed(1)}`).join(' ');
  const bands=groups.filter(g=>g.every(p=>p.values.length===3)).map(group=>`<path class="aerobic-progress-band" d="${group.map((r,i)=>`${i?'L':'M'}${x(r).toFixed(1)},${y(r.values[2]).toFixed(1)}`).join(' ')} ${[...group].reverse().map(r=>`L${x(r).toFixed(1)},${y(r.values[1]).toFixed(1)}`).join(' ')} Z"/>`).join('');
  const ticks=Array.from({length:Math.round((high-low)/step)+1},(_,i)=>low+i*step);
  const axes=ticks.map(v=>`<line class="${v===0?'aerobic-progress-zero':'grid-line'}" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="48" y="${y(v)+4}" text-anchor="end">${v>0?'+':''}${number(v,0)}%</text>`).join('');
  const marks=rows.map(r=>{
    const range=r.change.estimated_change_range_95_percent;
    return `<g ${runChartAttributes({id:`aerobic:${r.activity_id}`,x:x(r),y:y(r.change.speed_change_percent),title:date(r.as_of),details:[
      `${percent(r.change.speed_change_percent)} vs preceding 28 days`,verdict(r.change),
      ...(range?[`Estimated range: ${percent(range[0])} to ${percent(range[1])}`]:[]),
      `${r.run_count} comparable runs in the latest 28 days`,
      'Same pace/HR evidence as the chart above; not an independent fitness test']})}><circle class="aerobic-progress-point ${r.change.direction==='provisional'?'provisional':''}" cx="${x(r)}" cy="${y(r.change.speed_change_percent)}" r="3.5"/></g>`;
  }).join('');
  return `<p class="aerobic-progress-axis-note">Above zero = faster · Below zero = slower</p><svg class="insight-chart aerobic-progress-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Aerobic performance change versus the preceding 28 days; above zero is faster at comparable effort" data-run-chart data-run-top="24" data-run-bottom="${bottom}" data-chart-description="Hover or tap to inspect monthly aerobic progress estimates. Left and right arrows select dates.">
    ${axes}${bands}${groups.map(g=>`<path class="aerobic-progress-line" d="${line(g)}"/>`).join('')}${marks}
    <text x="${left}" y="${height-9}">${escapeHtml(date(new Date(start).toISOString()))}</text><text x="${right}" y="${height-9}" text-anchor="end">${escapeHtml(date(state.as_of))}</text></svg>`;
}

export function aerobicProgressPanel(state,date,historyDays) {
  if(!state) return '';
  if(!state.available) return `<article class="wide-card aerobic-progress-card"><div class="card-heading"><h2>Aerobic progress</h2><span class="quality">Prototype</span></div><p class="chart-note">${escapeHtml(state.reason??'More comparable runs are needed.')}</p></article>`;
  const change=state.change,longer=state.change_90d;
  const range=change?.estimated_change_range_95_percent;
  return `<article class="wide-card aerobic-progress-card">
    <div class="card-heading"><div><p class="eyebrow">Sustained performance</p><h2>Aerobic progress</h2></div><span class="quality">Prototype</span></div>
    <div class="aerobic-progress-summary"><div><strong class="is-${escapeHtml(change?.direction??'unavailable')}">${percent(change?.speed_change_percent)}</strong><span>${escapeHtml(verdict(change))}<small>Latest 28 days vs preceding 28 days</small></span></div>
      <div class="aerobic-progress-longer"><strong>${percent(longer?.speed_change_percent)}</strong><span>vs level 90 days earlier<small>${escapeHtml(verdict(longer))}</small></span></div></div>
    ${aerobicProgressChart(state,date,historyDays)}
    <p class="chart-note">${state.version==='coherent-aerobic-progress-v1'?'Line = estimated monthly change · confidence is assessed separately.':'Line = monthly performance change · shaded band = estimated range.'}</p>
    <details class="aerobic-progress-method"><summary>About this estimate</summary><p>Uses the same runs as adjusted pace, not a separate fitness test. Each point compares two consecutive 28-day periods. Changing the chart’s date range does not change today’s estimate. The model accounts for shared calibration uncertainty and variation between runs; it does not remove fatigue or every outside influence.</p>
      ${range?`<p>Current estimated change range: ${percent(range[0])} to ${percent(range[1])}. This range is conditional on the model, not a guarantee of physiological fitness change.</p>`:''}
      ${change?.zero_change_included===true?'<p>The estimate points up or down, but the evidence still allows no underlying change.</p>':''}
      <p>Last comparable run: ${escapeHtml(date(state.last_observation))}. ${state.current.run_count} runs inform the latest level.${state.days_since_observation>7?' The evidence is getting older; no new runs does not establish that fitness is unchanged.':''}</p></details>
  </article>`;
}

export function fitnessLevelChart(state,date,historyDays) {
  const compact=globalThis.matchMedia?.('(max-width: 40rem)')?.matches??false;
  const width=compact?420:890,height=compact?280:295,left=58,right=width-28,bottom=height-37;
  const end=+new Date(state.as_of),start=end-historyDays*86400000;
  const valid=r=>r.available&&Number.isFinite(r.value)&&r.estimated_range_95?.length===2&&r.estimated_range_95.every(Number.isFinite);
  const {groups,points}=windowedChartSeries(state.history??[],start,end,r=>valid(r)?[r.value,...r.estimated_range_95]:null,14);
  const rows=points.map(p=>p.row);
  if(!groups.length) return '<div class="empty-state">No comparable fitness estimates in this date range.</div>';
  const values=groups.flatMap(g=>g.flatMap(p=>p.values));
  const extent=Math.max(...values)-Math.min(...values);
  const step=[1,2,5,10,20,50,100].find(v=>v>=extent/4)??Math.ceil(extent/400)*100;
  const low=Math.max(0,Math.floor((Math.min(...values)-step*.15)/step)*step);
  const high=Math.ceil((Math.max(...values)+step*.15)/step)*step;
  const x=r=>left+((r.at??+new Date(r.as_of))-start)/Math.max(1,end-start)*(right-left);
  const y=v=>bottom-(v-low)/Math.max(1,high-low)*(bottom-24);
  const line=g=>g.map((r,i)=>`${i?'L':'M'}${x(r).toFixed(1)},${y(r.values[0]).toFixed(1)}`).join(' ');
  const band=g=>g.map((r,i)=>`${i?'L':'M'}${x(r).toFixed(1)},${y(r.values[2]).toFixed(1)}`).join(' ')+' '+[...g].reverse().map(r=>`L${x(r).toFixed(1)},${y(r.values[1]).toFixed(1)}`).join(' ')+' Z';
  const axes=Array.from({length:Math.round((high-low)/step)+1},(_,i)=>low+i*step).map(v=>`<line class="grid-line" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="48" y="${y(v)+4}" text-anchor="end">${number(v,0)}</text>`).join('');
  const marks=rows.map(r=>`<g ${runChartAttributes({id:`fitness:${r.activity_id}`,x:x(r),y:y(r.value),title:date(r.as_of),details:[
    `Fitness level: ${number(r.value,1)} VO₂-equivalent`,
    `Estimated range: ${number(r.estimated_range_95[0],1)}–${number(r.estimated_range_95[1],1)}`,
    `${r.run_count} comparable runs in the latest 28 days`,
    'Same adjusted pace evidence; not a lab VO₂max measurement']})}><circle class="aerobic-progress-point ${r.evidence==='provisional'?'provisional':''}" cx="${x(r)}" cy="${y(r.value)}" r="3.5"/></g>`).join('');
  return `<p class="aerobic-progress-axis-note">VO₂-equivalent · Higher = stronger aerobic performance</p><svg class="insight-chart fitness-level-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Absolute adjusted aerobic fitness level with estimated uncertainty" data-run-chart data-run-top="24" data-run-bottom="${bottom}" data-chart-description="Hover or tap to inspect fitness level estimates. Left and right arrows select dates.">
    ${axes}${groups.map(g=>`<path class="aerobic-progress-band" d="${band(g)}"/><path class="aerobic-progress-line" d="${line(g)}"/>`).join('')}${marks}
    <text x="${left}" y="${height-9}">${escapeHtml(date(new Date(start).toISOString()))}</text><text x="${right}" y="${height-9}" text-anchor="end">${escapeHtml(date(state.as_of))}</text></svg>`;
}

export function fitnessLevelPanel(state,date,historyDays) {
  if(!state) return '';
  if(!state.available) return `<article class="wide-card fitness-level-card"><div class="card-heading"><h2>Fitness level</h2><span class="quality">Prototype</span></div><p class="chart-note">${escapeHtml(state.reason??'More comparable runs are needed.')}</p></article>`;
  return `<article class="wide-card fitness-level-card">
    <div class="card-heading"><div><p class="eyebrow">Adjusted aerobic performance</p><h2>Fitness level</h2></div><span class="quality">Prototype</span></div>
    <div class="aerobic-progress-summary"><div><strong>${number(state.value,1)}</strong><span>VO₂-equivalent<small>Latest 28 days · not a lab VO₂max</small></span></div></div>
    ${fitnessLevelChart(state,date,historyDays)}
    <p class="chart-note">Line = estimated fitness level · shaded band = uncertainty in the level.</p>
    <details class="aerobic-progress-method"><summary>About this estimate</summary><p>The adjusted pace trend, shown on a fixed VO₂-equivalent scale rather than as a percentage change. Uses the same terrain, weather, heart-rate and within-run duration adjustments as adjusted pace. Each point uses the preceding 28 days; selecting a longer chart does not change the current number.</p>
      <p>This is a performance proxy, not a measured oxygen uptake or an independent fitness test. The conversion assumes typical running economy and a relationship between heart-rate reserve and oxygen-uptake reserve. Shared calibration errors can move the whole curve up or down; the band is not independent noise at each date. It does not isolate fitness from fatigue, illness or every outside influence.</p>
      <p>Uses one consistent conversion of adjusted pace throughout the chart. Last comparable run: ${escapeHtml(date(state.last_observation))}.</p></details>
  </article>`;
}
