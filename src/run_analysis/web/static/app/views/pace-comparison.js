import { api } from '../api.js';
import { view, loading } from '../dom.js';
import { escapeHtml, number, pace, titleCase } from '../format.js';
import { fitnessChart } from '../charts.js';
import { wireRunCharts } from '../chart-interaction.js';

let days=90,sequence=0;
const e=escapeHtml;
const labels={terrain:'Terrain',wind:'Wind',heart_rate:'Heart rate',duration:'Time into run',weather:'Heat + humidity'};
const seconds=value=>{
  if(!Number.isFinite(value))return '—';
  const rounded=Math.round(value),clean=Object.is(rounded,-0)?0:rounded;
  return `${clean>0?'+':''}${number(clean,0)} sec/mi`;
};

const adjustmentDetails=run=>{
  const quality=run.terrain_quality,withheld=quality?.usable===false;
  return `<dl>${Object.entries(labels).map(([key,label])=>`<div><dt>${label}</dt><dd>${key==='terrain'&&withheld?'Not applied':seconds(run.contributions_seconds_per_mile?.[key]??0)}</dd></div>`).join('')}</dl>
    ${withheld?`<p>Terrain adjustment withheld. ${e(quality.reason)}</p>`:quality?.surface_profile_uncertain?'<p>Recorded grades used where the ground map may describe a different surface; this remains an estimate.</p>':quality?.source==='local_dem'?'<p>Terrain from local elevation-map tiles.</p>':''}
    <p>${number(run.terrain_coverage*100,0)}% terrain coverage · ${number(run.weather_coverage*100,0)}% weather coverage</p>
    <p>Estimated range: ${pace(run.estimated_range_95_min_mile[0])}–${pace(run.estimated_range_95_min_mile[1])}</p>`;
};

export function comparisonMarkup(data) {
  const runs=data.runs??[],current=data.trend?.at(-1),recorded=data.recorded_trend?.at(-1);
  const date=value=>new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',timeZone:data.timezone}).format(new Date(value));
  const fullDate=value=>new Intl.DateTimeFormat(undefined,{dateStyle:'medium',timeStyle:'short',timeZone:data.timezone}).format(new Date(value));
  const roles=new Map(runs.map(r=>[r.activity_id,titleCase(r.workout_type)]));
  const breakdown=[...runs].reverse().map(run=>`<tr><td><a href="#run/${run.activity_id}">${e(date(run.start_time))}</a><small>${e(titleCase(run.workout_type))}</small></td><td>${pace(run.observed_pace_min_mile)}</td><td>${pace(run.pace_min_mile)}</td><td><details><summary>${seconds((run.pace_min_mile-run.observed_pace_min_mile)*60)}</summary>${adjustmentDetails(run)}</details></td></tr>`).join('');
  return `<section class="page progress-wip pace-comparison">
    <div class="page-heading"><div><p class="eyebrow"><a href="#progress">Progress ←</a> · Comparison preview</p><h1>Compare your runs</h1><p>Same heart rate. Flatter terrain. Comparable conditions.</p></div></div>
    <div class="toolbar"><div class="segmented" aria-label="Comparison time frame">${data.available_windows.map(value=>`<button type="button" data-comparison-days="${value}" class="${value===data.window_days?'selected':''}" aria-pressed="${value===data.window_days}">${value} days</button>`).join('')}</div></div>
    <article class="wide-card chart-card"><div class="card-heading"><div><p class="eyebrow">28-day trends · same supporting runs</p><h2>Recorded versus comparable pace</h2></div></div>
    <div class="metric-grid progress-metrics wip-running-summary"><article><span>Comparable estimate</span><strong>${pace(current?.pace_min_mile)}</strong></article><article><span>Recorded</span><strong>${pace(recorded?.pace_min_mile)}</strong></article><article><span>Runs in this view</span><strong>${data.trend_runs_in_view}</strong></article></div>
    ${fitnessChart(data.series,'both',data.trend,data.as_of,data.window_days,{rawTrend:data.recorded_trend,interactive:true,plainLanguage:true,runDateLabel:fullDate,runRoles:roles})}
    <p class="chart-note">Green = comparable estimate · amber = recorded pace. Both use the same sections and run weights.</p></article>
    <details class="progress-explanation"><summary>What changes—and what doesn't</summary><p>Estimated at ${number(data.calibration?.target_hr_bpm??runs[0]?.target_hr_bpm,0)} bpm, near minute ${number(data.calibration?.reference_minutes??runs[0]?.reference_minutes??20,0)}, on flat terrain in calm, cool conditions. HR, terrain, heat/humidity, wind and within-run drift are estimated together.</p><p>Missing context widens the estimated range. Fatigue, illness and poor sleep are not erased. Workout-only points do not inform the aerobic trend. All history uses the same current calibration.</p><p>This preview can include more reliable sections than the main Progress chart. It does not replace Progress, VO₂ estimates or planner inputs. These are estimates, not exact measurements.</p></details>
    <article class="wide-card"><div class="card-heading"><h2>What changed for each run?</h2></div><div class="table-scroll"><table><thead><tr><th>Run</th><th>Recorded</th><th>Comparable</th><th>Difference</th></tr></thead><tbody>${breakdown||'<tr><td colspan="4">No comparable sections in this period yet.</td></tr>'}</tbody></table></div><p class="chart-note">Negative seconds mean a faster equivalent pace. Open a difference to see each adjustment.</p></article>
  </section>`;
}

export async function renderPaceComparison() {
  const request=++sequence;
  loading('Comparing your runs');
  const data=await api(`/api/pace-comparability?window_days=${days}`);
  if(request!==sequence||location.hash.slice(1)!=='pace-comparison')return;
  view.innerHTML=comparisonMarkup(data);
  view.querySelectorAll('[data-comparison-days]').forEach(button=>button.addEventListener('click',()=>{
    days=Number(button.dataset.comparisonDays);
    renderPaceComparison().catch(error=>{if(location.hash==='#pace-comparison')view.innerHTML=`<section class="page narrow"><h1>Could not load comparison.</h1><p>${e(error.message)}</p></section>`;});
  }));
  wireRunCharts(view);
}
