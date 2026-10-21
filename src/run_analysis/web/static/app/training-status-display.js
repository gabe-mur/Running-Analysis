// Surface the existing classifier's evidence. This module does not classify
// training, infer recovery, or substitute the chart's HR-load ratio.
import {escapeHtml, number} from './format.js';

// Plain-language summary of the recorded decision, not a second classifier.
// Aerobic status uses the same selected evidence window as Progress.
export function trainingStatusMessage(summary, aerobic) {
  const likely=aerobic?.chosen?.evidence==='likely';
  const aerobicText={
    improving:likely?'Aerobic efficiency likely improving':'Aerobic efficiency improving',
    declining:likely?'Aerobic efficiency likely declining':'Aerobic efficiency declining',
    stable:'Maintaining aerobic efficiency',
    uncertain:'Maintaining aerobic efficiency',
    insufficient_data:'More data needed to assess aerobic efficiency',
  }[aerobic?.direction]??'More data needed to assess aerobic efficiency';
  const rules=new Map((summary?.rule_trace??[]).map(rule=>[rule.rule_id,rule]));
  const building=rules.get('status_building')?.facts;
  // This is retained demonstrated weekly mileage, not the latest week's
  // total, a planned target, or a measured physiological capacity.
  const sustainedVolumeGrowing=Number.isFinite(building?.capacity_reference_miles)
    &&Number.isFinite(building?.previous_capacity_reference_miles)
    &&building.capacity_reference_miles>building.previous_capacity_reference_miles;
  let reason;
  switch(summary?.status) {
    case 'building': {
      // Quality attendance is a classifier gate, not an observed improvement.
      // Do not repeat the aerobic phrase when it already explains the build.
      reason=sustainedVolumeGrowing?'Sustained volume growing':building?.fitness_trend==='improving'
        ?aerobic?.direction==='improving'?null:'Recent aerobic gains recorded'
        :'Status reason unavailable';
      break;
    }
    case 'maintaining':
      reason=Number.isFinite(building?.sustained_to_capacity_ratio)&&Number.isFinite(building?.near_capacity_ratio)
        &&building.sustained_to_capacity_ratio<building.near_capacity_ratio
        ?'Volume below your established level'
        :Number.isFinite(building?.sustained_to_capacity_ratio)&&Number.isFinite(building?.near_capacity_ratio)
          ?'Volume near your established level':'Status reason unavailable';
      break;
    case 'rebuilding': reason='Load rising after reduced volume'; break;
    case 'underloaded': reason='Sustained volume below your established level'; break;
    case 'recovering': {
      const health=rules.get('status_recovering')?.facts;
      reason=health?.current_health_status==='pain_or_injury_concern'?'Pain or injury concern reported'
        :health?.current_health_status==='sick_or_recovering'?'Illness reported'
          :health?.recent_illness_or_recovery?'Recent health concern':'Recovery concern reported';
      break;
    }
    case 'strained': {
      const strain=rules.get('status_strained')?.facts;
      const high=Number.isFinite(strain?.continuous_fatigue_to_capacity_ratio)&&Number.isFinite(strain?.high_load_ratio)
        &&strain.continuous_fatigue_to_capacity_ratio>=strain.high_load_ratio;
      reason=high&&strain?.costly_response_corroborated?'High accumulated load · Higher effort on the latest run'
        :high?'High accumulated load'
          :strain?.costly_response_corroborated?'Higher effort on the latest run'
            :'Status reason unavailable';
      break;
    }
    case 'insufficient_data': {
      reason='More recent training data needed';
      // Two missing-evidence messages would say the same thing twice.
      if (!aerobic?.direction||aerobic.direction==='insufficient_data') return reason;
      break;
    }
    default: reason='Training status unavailable';
  }
  return [aerobicText,reason].filter(Boolean).join(' · ');
}

export function trainingStatusDisplay(summary, load) {
  if (!summary) return `<p class="eyebrow">Training direction</p><h2 class="training-status none">Status unavailable</h2><p>Your recorded training is still available below.</p>`;
  const rules=new Map((summary.rule_trace??[]).map(r=>[r.rule_id,r]));
  const building=rules.get('status_building')?.facts;
  const strain=rules.get('status_strained')?.facts;
  const capacity=load?.capacity_reference_miles;
  const weekly=load?.trailing_28d?.distance_miles/4;
  const tone=['building','rebuilding'].includes(summary.status)?'good':
    ['strained','recovering'].includes(summary.status)?'bad':summary.status==='insufficient_data'?'none':'neutral';
  const cards=[];
  if (Number.isFinite(weekly)&&Number.isFinite(capacity)&&capacity>0) {
    cards.push(['Sustained running',`${number(weekly)} mi/week`,
      `Last 28 days · demonstrated ${number(capacity)} mi/week`]);
  }
  if (Number.isFinite(building?.quality_sessions_14d)) {
    cards.push(['Quality work',`${number(building.quality_sessions_14d,0)} sessions`,'Last 14 days']);
  }
  // Use the improvement predicate recorded by the classifier, not a different
  // Progress trend or a fresh inference from the chart's line slope.
  if (building) {
    const current=building.capacity_reference_miles,previous=building.previous_capacity_reference_miles;
    const increased=Number.isFinite(current)&&Number.isFinite(previous)&&current>previous;
    cards.push(['Progression evidence',increased?'Capacity increased':building.fitness_trend==='improving'?'Aerobic trend improving':building.improvement_supported===false?'No clear increase':'Not recorded',
      increased?`${number(previous)} → ${number(current)} mi/week`:'Evidence used by the status estimate']);
  }
  const details=summary.status==='building'&&building
    ? 'Sustained running, recent quality work and an improving signal support a build.'
    : summary.detail??'The status estimate has no explanation available.';
  const strainExplanation=strain?`<p>The existing load check ${rules.get('status_strained').fired?'raised':'did not raise'} a strain flag.${Number.isFinite(strain.continuous_fatigue_to_capacity_ratio)?` Its accumulated-load estimate is ${number(strain.continuous_fatigue_to_capacity_ratio,2)}× demonstrated capacity; its flag threshold is ${number(strain.high_load_ratio,2)}×.`:''}${strain.recent_performance_response==='higher_cost_than_recent'?' The latest comparable run took more effort than usual; that alone does not trigger this flag.':''}</p>`:'';
  return `<p class="eyebrow">Training direction · Estimate</p><h2 class="training-status ${tone}">${escapeHtml(summary.label)}</h2>
    <p class="status-detail">${escapeHtml(details)}</p>
    ${cards.length?`<div class="status-evidence-grid">${cards.map(([label,value,context])=>`<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(context)}</small></div>`).join('')}</div>`:''}
    <details class="status-evidence-help"><summary>What drives this status?</summary><p>${escapeHtml(summary.detail??'No explanation recorded.')}</p>${strainExplanation}<p>This is the existing training-status interpretation, not a verified recovery assessment. Its load check uses the earlier distance/time/HR estimate—not the current scheduler's fatigue score or the chart's recent-to-usual ratio.</p></details>`;
}

export function loadAssessmentDisplay(summary,{heading='h1',eyebrow=true}={}) {
  const rules=new Map((summary?.rule_trace??[]).map(r=>[r.rule_id,r]));
  const strain=rules.get('status_strained');
  const insufficient=rules.get('status_insufficient_evidence')?.fired;
  // A health/evidence gate can stop evaluation before strain. Never treat an
  // unevaluated rule as a negative flag, or reuse Building as a load verdict.
  const label=insufficient?'Not enough load evidence':!strain?'Load check unavailable':
    strain.fired?'Elevated strain signal':'No elevated strain flag';
  const detail=!strain?'The existing load check has no current result. The overall training status is shown separately.':
    strain.fired?'The existing check flagged accumulated load or a corroborated higher-cost response.':
    'The existing load check has not flagged excess strain. This is not proof of full recovery.';
  const facts=strain?.facts,ratio=facts?.continuous_fatigue_to_capacity_ratio,threshold=facts?.high_load_ratio;
  const tag=['h1','h2','h3'].includes(heading)?heading:'h3';
  return `${eyebrow?'<p class="eyebrow">Training load · Estimate</p>':''}<${tag} class="training-status ${strain?.fired?'bad':strain?'neutral':'none'}">${label}</${tag}><p class="status-detail">${detail}</p>
    ${strain?`<details class="status-evidence-help"><summary>What drives the load check?</summary>${Number.isFinite(ratio)&&Number.isFinite(threshold)?`<p>The accumulated-load estimate is ${number(ratio*100,0)}% of demonstrated capacity. The existing high-load flag starts at ${number(threshold*100,0)}%.</p>`:''}<p>${facts.costly_response_corroborated?'The latest higher-cost response was corroborated by high within-run drift.':facts.recent_performance_response==='higher_cost_than_recent'?'The latest comparable run took more effort than usual, but that alone does not trigger the strain flag.':'No corroborated higher-cost response triggered the strain flag.'}</p><p>This uses the earlier distance/time/HR estimate, not the current scheduler's fatigue score or the chart's recent-to-usual ratio. It does not require fitness improvement to pass. No flag is not verified recovery.</p></details>`:''}`;
}
