// Primary Progress view, promoted from the reviewed WIP. The previous layout
// remains callable for regression/reference checks, but is no longer routed.
import { api, targetHrLabel } from "../api.js";
import { fitnessWindow, selectFitnessWindow } from "../fitness-window.js";
import { DIRECTION, SENTIMENT, comparisonValue, directionValue } from "../components.js";
import { fitnessChart, vo2Chart as baseVo2Chart } from "../charts.js";
import { goalMarkup } from "../goal.js";
import { view, loading } from "../dom.js";
import { escapeHtml, evidenceLabel, fitnessWindowLabel, number, pace, titleCase } from "../format.js";
import { performanceSummary, benchmarkHeadline, loadComparison, vo2EstimateMarkup, vo2MethodMarkup } from "../progress-insights.js";
import { trainingVolumeChart as mileageChart, effortBalance } from "../progress-wip-charts.js";
import { wireRunCharts } from '../chart-interaction.js';
import { aerobicProgressPanel, fitnessLevelPanel } from '../aerobic-progress-panel.js';
import { benchmarkPacePanel } from '../benchmark-pace-panel.js';
import { goalJourneyPanel, wireGoalOutlook } from '../goal-journey.js';

let progressWindow=28, progressMetric="standardized", requestSequence=0;
// The promoted view retains its reviewed three-month default and local controls.
let wipWindow=90, goalOutlook='distance';
const e=escapeHtml;
const n=(v,digits=1)=>Number.isFinite(v)?number(v,digits):"—";
const role=v=>v==="tempo_threshold"?"Progression / tempo":titleCase(v);
const range=q=>q?Math.abs(q.high-q.low)<.01?n(q.low):`${n(q.low)}–${n(q.high)}`:"—";
const vo2Chart=series=>baseVo2Chart(series,{maxGapDays:2});

export function pacePresentation(data) {
  const normalized=data.pace_comparison;
  const hidden=new Set(data.aerobic_performance?.hidden_activity_ids??[]);
  const manuallyExcluded=new Set(data.aerobic_performance?.manually_excluded_activity_ids??[]);
  const visible=run=>!hidden.has(run.activity_id)&&!run.manually_excluded&&!['hike','bike'].includes(run.workout_type);
  if(!normalized) return {progress:{...data.baseline,series:data.baseline.series.filter(visible)},perf:data.performance};
  const estimates=new Map(normalized.runs.map(run=>[run.activity_id,run]));
  const coverage=data.baseline.activity_coverage.map(item=>{
    if(manuallyExcluded.has(item.activity_id)) return {...item,standardized_pace_min_mile:null,
      included_in_trend:false,score_status:'context_only',reason:'Excluded from aerobic scoring by you.'};
    const run=estimates.get(item.activity_id);
    if(!run) return {...item,standardized_pace_min_mile:null,included_in_trend:false,
      score_status:item.score_status==='non_running'?'non_running':'unavailable',
      reason:'No reliable comparable running sections were available.'};
    return {...item,standardized_pace_min_mile:run.pace_min_mile,
      included_in_trend:run.included_in_trend,
      score_status:!run.included_in_trend?'context_only':run.trend_weight<1?'reduced_weight':'trend_evidence',
      reason:!run.included_in_trend?'Workout context only; not used in the aerobic trend.':run.trend_weight<1?'Used with less weight because of the run’s health or run/walk context.':'Used in the aerobic trend. Pace is compared at the same heart rate and conditions.'};
  });
  const current=normalized.performance.current;
  const ids=new Set(normalized.series.map(run=>run.activity_id));
  return {progress:{...data.baseline,
    current_pace:current?{minutes_per_mile:current.pace_min_mile}:null,
    uncertainty_95_min_mile:current?.uncertainty_95_min_mile??null,
    fitness_confidence:normalized.performance.confidence,
    series:[...normalized.series,...data.baseline.series.filter(run=>run.trend_weight===0&&!ids.has(run.activity_id))].filter(visible),
    activity_coverage:coverage},
    perf:{...data.performance,...normalized.performance,
      rolling_28d:normalized.trend,rolling_raw_28d:normalized.recorded_trend}};
}

export const renderProgress=()=>renderProgressView({preview:true});
// Deprecated presentation only: retained for reference, not exposed as a route.
export const renderArchivedProgress=()=>renderProgressView({preview:false});

async function renderProgressView({preview=true}={}) {
  const sequence=++requestSequence;
  const routeName='progress';
  loading('Progress');
  progressWindow=preview?wipWindow:await fitnessWindow();
  const d=await api(`/api/progress-wip?window_days=${progressWindow}&include_exploratory=false${preview?'&include_legacy_aerobic=false&include_goal_pace_outlook=true':''}`);
  if(sequence!==requestSequence || location.hash.slice(1)!==routeName) return;
  const {progress,perf}=pacePresentation(d),summary=performanceSummary(perf,fitnessWindowLabel(progressWindow));
  const normalized=Boolean(d.pace_comparison);
  const observedPace=!normalized&&progress.series.some(p=>p.pace_basis==='observed_comparable');
  if(observedPace) progressMetric='standardized';
  const date=value=>new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',timeZone:d.timezone}).format(new Date(value));
  const fullDate=value=>new Intl.DateTimeFormat(undefined,{dateStyle:'medium',timeZone:d.timezone}).format(new Date(value));
  const goalDate=value=>new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(new Date(value));
  const runDate=value=>new Intl.DateTimeFormat(undefined,{dateStyle:'medium',timeStyle:'short',timeZone:d.timezone}).format(new Date(value));
  const current=d.exposure.current,previous=d.exposure.prior,load=loadComparison(d.load),comparison=progress.period_comparison;
  const hrLoadCurrent=comparison.current.zone_load,hrLoadPrevious=comparison.previous.zone_load;
  const hrLoadDirection=!Number.isFinite(hrLoadCurrent)||!Number.isFinite(hrLoadPrevious)?DIRECTION.NONE:
    Math.abs(hrLoadCurrent-hrLoadPrevious)<=hrLoadPrevious*.05?DIRECTION.FLAT:hrLoadCurrent>hrLoadPrevious?DIRECTION.UP:DIRECTION.DOWN;
  const confidence=progress.fitness_confidence,now=progress.current_pace?.minutes_per_mile,uncertainty=progress.uncertainty_95_min_mile;
  const coverageRows=progress.activity_coverage.map(item=>{
    const used=["trend_evidence","uncertain_estimate","reduced_weight"].includes(item.score_status);
    const label=used?item.score_status==='reduced_weight'?'Used with less weight':'Used':item.score_status==='non_running'?'Not running':'Not used';
    return `<tr><td><a href="#run/${item.activity_id}">${fullDate(item.start_time)}</a></td><td>${n(item.distance_miles)} mi</td><td>${e(role(item.workout_type))}</td><td>${e(titleCase(item.health_tag))}</td><td>${pace(item.standardized_pace_min_mile)}</td><td><span class="quality ${used?'good':'unavailable'}">${label}</span></td><td><details><summary>Why?</summary><p>${e(item.reason)}</p></details></td></tr>`;
  }).join('');
  const goal=d.race_goal.status!=='no_goal'?goalMarkup(d.race_goal,'wide'):'';
  const unifiedFitness=d.aerobic_progress?.fitness_level?.available===true;
  // WIP has one current benchmark, independent of the displayed date range.
  // Keep the legacy cards/headlines on Progress while this layout is reviewed.
  const benchmark=d.aerobic_performance??d.aerobic_progress;
  const benchmarkView=preview&&benchmark?{...benchmark,smoothing_comparison:null}:benchmark;
  const headline=benchmarkHeadline(benchmark,progressMetric);
  // Compact WIP presentation only; shared charts and metric definitions stay.
  const vo2Card=`<article class="wide-card vo2-card"><div class="vo2-layout"><div class="vo2-figure">${vo2EstimateMarkup(perf.vo2,{compact:preview,includeMethod:!preview})}</div><div class="vo2-trend">${preview?'':`<p class="eyebrow">History · last ${fitnessWindowLabel(progressWindow)}</p>`}${vo2Chart(perf.vo2_series)}${preview?'':'<p class="chart-note">Each estimate uses the preceding 14 days. The band shows uncertainty.</p>'}</div></div>${preview?vo2MethodMarkup(perf.vo2,{compact:true}):''}</article>`;
  const qualityPanel=`<aside class="effort-panel">${effortBalance(d,date,{compact:preview})}</aside>`;
  const volumeHeight=preview&&globalThis.matchMedia?.('(min-width: 1101px)')?.matches?420:290;
  const volumeCard=`<article class="wide-card training-history"><div class="card-heading"><div><p class="eyebrow">Last ${fitnessWindowLabel(progressWindow)}</p><h2>Training volume</h2></div></div>
      <div class="training-layout"><div class="training-main">
        <div class="volume-level"><strong>${n(d.exposure.daily.at(-1)?.distance_7d)} <small>mi in the last 7 days</small></strong></div>
        ${mileageChart(d,date,{height:volumeHeight,compact:preview})}
        ${preview?'':`<div class="volume-current"><span>Current 28-day average <b>${n(d.exposure.daily.at(-1)?.distance_28d_weekly)} mi / 7 days</b></span><span>Usual: ${n(progress.current_load.capacity_reference_miles)} mi / 7 days · not a target</span></div>`}
      </div>${preview?'':qualityPanel}</div>
        <details><summary>Totals and load</summary>${preview?`<p>Current 28-day average: ${n(d.exposure.daily.at(-1)?.distance_28d_weekly)} mi / 7 days. Usual: ${n(progress.current_load.capacity_reference_miles)} mi / 7 days—not a target.</p>`:''}<div class="comparison-grid">
          <span>Distance<strong>${comparisonValue(current.distance_miles,previous.distance_miles,`${n(current.distance_miles)} mi`,{deadband:previous.distance_miles*.05})}</strong><small>Previously ${n(previous.distance_miles)} mi</small></span>
          <span>Running time<strong>${comparisonValue(current.moving_minutes,previous.moving_minutes,`${n(current.moving_minutes,0)} min`,{deadband:previous.moving_minutes*.05})}</strong><small>Previously ${n(previous.moving_minutes,0)} min</small></span>
          <span>Training load<strong>${directionValue(hrLoadDirection,SENTIMENT.NEUTRAL,n(hrLoadCurrent,0))}</strong><small>HR-based points · previously ${n(hrLoadPrevious,0)}</small></span>
          <span>Runs<strong>${comparisonValue(current.run_count,previous.run_count,String(current.run_count),{deadband:1})}</strong><small>Previously ${previous.run_count}</small></span>
        </div>
        <p>${e(comparison.interpretation)} HR-based points use the existing Progress accounting; missing HR is not imputed.</p><p>Separate scheduler accounting: ${e(load.label)}. Verified for ${d.load.current.covered_runs} of ${d.load.current.total_runs} runs. Missing runs are not counted as zero.</p><p>Cardiovascular dose: ${range(d.load.current.cardio)}; prior ${range(d.load.prior.cardio)}. Work minutes: ${range(d.load.current.work_minutes)}; prior ${range(d.load.prior.work_minutes)}. These are different units from HR-based points. Bounds describe accounting uncertainty, not a recovery score.</p><p>The usual-volume reference is retained from current Progress. It describes running history, not the planner’s target.</p></details>
      </article>`;
  view.innerHTML=`<section class="page progress-wip${preview?' wip-window-layout':''}">
    ${preview?`<div class="page-heading"><div><p class="eyebrow">Progress</p><h1>${e(headline.headline)}</h1><p>${e(headline.description)}</p>${headline.changeLabel?`<p class="pace-change ${headline.direction==='declining'?'is-declining':''}">${e(headline.changeLabel)} <small>· ${e(headline.changeBasis)}</small></p>`:''}</div><div class="headline-pace"><strong>${pace(headline.value)}</strong><span>${e(headline.valueLabel)}</span><small>${e(headline.basis)}</small></div></div>`:`<div class="page-heading"><div><p class="eyebrow">Progress</p><h1>${e(summary.headline)}</h1><p>${observedPace?'Recorded pace from comparable easy running.':'Pace at the same heart rate, adjusted for conditions.'}</p>${summary.chosen&&['improving','declining'].includes(summary.direction)?`<p class="pace-change">${e(summary.changeLabel)} <small>· ${e(summary.basis.toLowerCase())}</small></p>`:''}${summary.recentTrajectory?`<p class="recent-trajectory">${e(summary.recentTrajectory)}</p>`:''}</div><div class="headline-pace"><strong>${pace(now)}</strong><span>${observedPace?'Near':'At'} ${targetHrLabel()}</span><small>${Number.isFinite(now)?`${fitnessWindowLabel(progressWindow)} estimate${Number.isFinite(uncertainty)?` · range ${pace(now-uncertainty)}–${pace(now+uncertainty)}`:''}`:'Not enough comparable runs'}</small></div></div>`}
    <div class="toolbar"><div class="segmented" aria-label="${preview?'History shown':'Fitness time frame'}">${d.available_windows.map(days=>`<button type="button" data-window="${days}" class="${days===progressWindow?'selected':''}" aria-pressed="${days===progressWindow}">${fitnessWindowLabel(days)}</button>`).join('')}</div>${observedPace&&!preview?'':`<div class="segmented" aria-label="Pace display">${[['standardized',preview?'Adjusted':'Adjusted pace'],['raw',preview?'Unadjusted':'Unadjusted pace'],['both','Both']].map(([metric,label])=>`<button type="button" data-metric="${metric}" class="${metric===progressMetric?'selected':''}" aria-pressed="${metric===progressMetric}">${label}</button>`).join('')}</div>`}</div>
    ${preview?'':`<article class="wide-card chart-card"><div class="card-heading"><div><p class="eyebrow">Last ${fitnessWindowLabel(progressWindow)}</p><h2>${observedPace?`Easy-running pace near ${targetHrLabel()}`:progressMetric==='standardized'?`Adjusted pace at ${targetHrLabel()}`:progressMetric==='raw'?'Unadjusted pace':'Adjusted and unadjusted pace'}</h2></div><span class="quality ${confidence}">${evidenceLabel(confidence)}</span></div>${fitnessChart(progress.series,progressMetric,perf.rolling_28d,d.as_of,progressWindow,{fitUncertainty:true,plainLanguage:true,observedPace,rawTrend:observedPace?null:perf.rolling_raw_28d,interactive:true,runDateLabel:runDate,runRoles:new Map((d.activities??[]).map(a=>[a.activity_id,role(a.role)]))})}<p class="chart-note">${progressMetric==='both'?'Solid green = adjusted trend · dashed amber = unadjusted trend.':'The line shows the 28-day trend.'} Dots are supporting runs; dotted workouts don’t affect the trends.</p></article>`}
    ${benchmarkPacePanel(benchmarkView,fullDate,progressWindow,d.aerobic_performance?.target_hr_bpm??d.aerobic_progress?.fitness_level?.target_hr_bpm,preview?progressMetric:'standardized',{showSummary:!preview})}
    ${preview?'':fitnessLevelPanel(d.aerobic_progress?.fitness_level,fullDate,progressWindow)}
    ${preview?'':aerobicProgressPanel(d.aerobic_progress,fullDate,progressWindow)}
    ${preview?'':`<details class="progress-explanation"><summary>How we know</summary>
      <p>${perf.current?.run_count??0} comparable runs inform the aerobic estimate. Last comparable run: ${perf.last_observation?fullDate(perf.last_observation):'none yet'}. Quality sessions still count toward volume and training load.</p>
      <p>${perf.period_change?`Period comparison: ${n(Math.abs(perf.period_change.pace_change_seconds_per_mile),0)} sec/mi ${perf.period_change.pace_change_seconds_per_mile<0?'faster':'slower'}, with ±${n(perf.period_change.uncertainty_95_seconds_per_mile,0)} sec/mi estimated uncertainty.`:'The previous period has too little comparable data.'}${perf.within_window_trend?` The trend inside this period is ${n(Math.abs(perf.within_window_trend.pace_change_seconds_per_mile),0)} sec/mi ${perf.within_window_trend.pace_change_seconds_per_mile<0?'faster':'slower'} across ${n(perf.current?.coverage_span_days)} observed days.`:''}${summary.conflict?' These describe different timescales: a higher level and a recent slowdown, or the reverse, can both be true.':''}</p>
      <p>${normalized?'Adjusted pace estimates the same heart rate and stage of a run, on flat terrain in calm, mild weather. Unadjusted pace uses the same running sections. Missing data widens the estimated range; fatigue is not removed.':observedPace?'Pace is observed in Z2, weighted toward comparable HR and portions of the run. It is not corrected for heat, hills or fatigue. These can still explain changes; the range does not capture every source of uncertainty.':'Pace uses the stored historical correction model.'} ${unifiedFitness?'Fitness level uses the same adjusted evidence over 28 days, not a separate fitness test.':'The trend uses 28 days; current VO₂ uses 14 days and remains an indirect estimate.'}</p>
      ${normalized?'<p><a class="pace-comparison-link" href="#pace-comparison">See individual pace adjustments →</a></p>':progress.steady_aerobic.current_pace?`<p>Steady-running cross-check: ${pace(progress.steady_aerobic.current_pace.minutes_per_mile)} near ${targetHrLabel()}. This uses the same runs, not an independent fitness test.</p>`:''}
    </details>`}
    ${preview?`<div class="wip-supporting-layout">${volumeCard}<div class="wip-supporting-stack">${vo2Card}<article class="wide-card wip-quality-card">${qualityPanel}</article></div></div>${goalJourneyPanel(d.goal_journey,goalDate,{compact:true,outlook:goalOutlook})}`:`${vo2Card}${volumeCard}`}
    ${preview?'':goal}
    <details class="wide-card coverage-card"><summary><span class="eyebrow">Runs used in this view</span><span class="coverage-summary">${progress.activity_coverage.filter(a=>a.included_in_trend).length} of ${current.run_count} runs in this period inform the aerobic trend</span></summary><div class="table-scroll"><table><thead><tr><th>Date</th><th>Distance</th><th>Type</th><th>Health</th><th>${observedPace?'Observed aerobic pace':'Adjusted pace'}</th><th>Fitness trend</th><th>Why</th></tr></thead><tbody>${coverageRows||'<tr><td colspan="7">No activities in this window.</td></tr>'}</tbody></table></div></details>
  </section>`;
  const refresh=()=>renderProgressView({preview}).catch(error=>{
    if(location.hash.slice(1)===routeName) view.innerHTML=`<section class="page narrow"><h1>Could not load Progress.</h1><p>${e(error.message)}</p></section>`;
  });
  view.querySelectorAll('[data-window]').forEach(b=>b.addEventListener('click',()=>{
    if(preview) wipWindow=Number(b.dataset.window);
    else selectFitnessWindow(Number(b.dataset.window));
    refresh();
  }));
  view.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>{progressMetric=b.dataset.metric;refresh();}));
  if(preview) wireGoalOutlook(view,selected=>{goalOutlook=selected;});
  wireRunCharts(view);
}
