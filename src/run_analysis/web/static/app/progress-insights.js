// Presentation only; preserve the established Progress evidence semantics.
import { escapeHtml, evidenceLabel, number } from './format.js';

export function benchmarkHeadline(state,metric='standardized') {
  const raw=metric==='raw',source=raw?state?.unadjusted:state;
  const current=source?.current,value=current?.pace_min_mile;
  const available=source?.available===true&&Number.isFinite(value)&&value>0;
  const trend=source?.progress_trend;
  const span=trend?.span_days;
  const weeks=Number.isFinite(span)?Math.max(1,Math.round(span/7)):null;
  const description=weeks
    ? `${trend.early_estimate?'Early estimate · ':''}${weeks}-week ${raw?'recorded-pace':'aerobic'} trend · ${raw?'same running sections':'adjusted for conditions'}.`
    : raw?'Recorded pace from the same running sections.':'Pace at the same heart rate, adjusted for conditions.';
  const subject=raw?'Your recorded pace':'Your aerobic pace';
  let headline='Establishing your aerobic trend.';
  if(!available) headline='More comparable runs are needed.';
  else if(state?.days_since_observation>28) headline=raw?'Your last recorded pace estimate.':'Your last aerobic pace estimate.';
  else if(trend?.direction==='improving'||trend?.direction==='declining') {
    headline=`${subject} is ${trend.evidence==='likely'||trend.early_estimate?'likely ':''}${trend.direction==='improving'?'improving':'slowing'}.`;
  } else if(trend?.direction==='maintaining') headline=raw?'Maintaining recorded pace.':'Maintaining aerobic efficiency.';
  const directional=available&&!(state?.days_since_observation>28)&&['improving','declining'].includes(trend?.direction);
  const delta=trend?.pace_change_seconds_per_mile;
  const changeLabel=directional&&Number.isFinite(delta)
    ? `${trend.evidence==='likely'?'Likely ':''}${Math.round(Math.abs(delta))} sec/mi ${trend.direction==='improving'?'faster':'slower'}` : null;
  return {headline,description,value:available?value:null,changeLabel,
    changeBasis:weeks?`over the ${weeks}-week trend`:null,direction:trend?.direction,
    valueLabel:raw?'Recorded pace':Number.isFinite(state?.target_hr_bpm)?`At ${number(state.target_hr_bpm,0)} bpm`:'At the same heart rate',
    basis:available?'Current smoothed pace':'Not enough comparable runs'};
}

export function vo2MethodMarkup(vo2,{compact=false}={}) {
  const e=escapeHtml;
  return `<details><summary>Method and limitations</summary>${compact?'<p>Estimated from recent pace and heart rate—not measured. The chart band shows uncertainty.</p>':''}<p>Uses the existing Progress estimator with a fixed trailing 14-day pace estimate and at least two comparable runs. Changing the history shown does not change the current number. Historical points use only observations available at that date; gaps remain missing.</p><p>${e(vo2.method)}</p><p>${e(vo2.interpretation)}</p>${vo2.limitations.map(v=>`<p>${e(v)}</p>`).join('')}${Number.isFinite(vo2.demographic_baseline_ml_kg_min)?`<p>Broad demographic baseline for your age, sex, BMI and activity level: ${number(vo2.demographic_baseline_ml_kg_min)} ± ${number(vo2.demographic_uncertainty_95_ml_kg_min)} mL/kg/min. That is a population figure, not a measurement of you.</p>`:''}</details>`;
}

export function vo2EstimateMarkup(vo2,{compact=false,includeMethod=true}={}) {
  const e=escapeHtml, available=Number.isFinite(vo2.value_ml_kg_min);
  return `<div class="card-heading"><div><p class="eyebrow">Current · last 14 days</p><h2>VO₂ max</h2></div><span class="quality ${e(vo2.confidence)}">${e(evidenceLabel(vo2.confidence))}</span></div>
    ${available?`<p class="vo2-value"><strong>${number(vo2.value_ml_kg_min)}</strong><span>mL/kg/min</span></p>`:'<p class="vo2-value"><strong>—</strong></p>'}
    ${compact?'':'<p class="cross-check-caveat">Estimated from recent pace and heart rate—not measured.</p>'}
    ${!available?`<p>${e(vo2.interpretation)}</p>`:''}
    ${includeMethod?vo2MethodMarkup(vo2,{compact}):''}`;
}

export function performanceSummary(performance, windowLabel=null) {
  const period=performance.period_change, slope=performance.within_window_trend;
  const directional=value=>["improving","declining"].includes(value?.direction);
  const conflict=directional(period)&&directional(slope)&&period.direction!==slope.direction;
  // Different timescales can legitimately point in opposite directions.
  // Keep the period comparison primary; show the within-period trajectory
  // separately rather than cancelling both into a maintenance claim.
  const chosen=period??(directional(slope)?slope:null);
  const direction=chosen?.direction??performance.fitness_trend??'insufficient_data', likely=chosen?.evidence==='likely';
  const headlines={improving:likely?"You're likely getting faster at the same heart rate.":"You're getting faster at the same heart rate.",declining:likely?"Your pace is likely slower at the same heart rate.":"Your pace is slower at the same heart rate.",stable:"Your pace is about the same.",uncertain:"Maintaining current pace.",insufficient_data:"More comparable runs are needed."};
  const statusLabels={improving:likely?'Likely improving':'Improving',declining:likely?'Likely slowing':'Slowing',stable:'Maintaining current pace',uncertain:'Maintaining current pace',insufficient_data:'Not enough data'};
  const headline=chosen===period&&directional(period)
    ?`${likely?'Likely ':''}${direction==='improving'?(likely?'faster':'Faster'):(likely?'slower':'Slower')} than the previous ${windowLabel??'period'}.`
    :headlines[direction]??headlines.uncertain;
  const recentTrajectory=period&&slope?directional(slope)
    ?`Recent pace is ${slope.evidence==='likely'?'likely ':''}${slope.direction==='improving'?'improving':'slowing'}.`
    :slope.direction==='stable'?'Recent pace is steady.':'Recent pace has no clear direction.':null;
  return {direction,headline,statusLabel:statusLabels[direction]??statusLabels.uncertain,chosen,conflict,recentTrajectory,
    changeLabel:directional(chosen)?`${likely?'Likely ':''}${Math.round(Math.abs(chosen.pace_change_seconds_per_mile))} sec/mi ${direction==='improving'?'faster':'slower'}`:direction==='insufficient_data'?'Not enough data':'No clear change',
    basis:chosen&&chosen===slope?`Trend within ${windowLabel??'this period'}`:`Compared with the previous ${windowLabel??'period'}`};
}

export function qualitySessionLabel(session) {
  if(session.interval?.available) return 'Intervals';
  if(session.continuous?.structure==='progression_finish') return 'Progression';
  if(session.continuous) return 'Sustained effort';
  return null;
}

export function loadComparison(load) {
  const a=load.current,b=load.prior;
  if(!a.complete||!b.complete||!a.cardio||!b.cardio||b.cardio.low<=0||a.accounting_key!==b.accounting_key) return {label:'Not enough load history',direction:'none',low:null,high:null};
  const low=(a.cardio.low/b.cardio.high-1)*100,high=(a.cardio.high/b.cardio.low-1)*100;
  if(low<=0&&high>=0) return {label:'No clear change',direction:'flat',low,high};
  const change=((a.cardio.low+a.cardio.high)/(b.cardio.low+b.cardio.high)-1)*100;
  return {label:`About ${Math.round(Math.abs(change))}% ${change>0?'more':'less'}`,direction:change>0?'up':'down',low,high};
}
