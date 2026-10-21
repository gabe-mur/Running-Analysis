import {escapeHtml,titleCase,pace} from './format.js';
import {paceChartScale} from './charts.js';
import {runChartAttributes} from './chart-interaction.js';

const e=escapeHtml;
const num=x=>Number.isFinite(x)?x.toLocaleString(undefined,{maximumFractionDigits:1}):'—';
const stamp=day=>`${day}T12:00:00Z`;
const roundPace=value=>pace(Math.round(value*12)/12);
const chartWidth=()=>globalThis.matchMedia?.('(max-width: 40rem)')?.matches?420:760;
const domain=data=>{
  const today=data.today??data.as_of.slice(0,10);
  const days=[today,...(data.recorded??[]).map(r=>r.date),...(data.scheduled??[]).map(r=>r.date),
    ...(data.projection?.points??[]).map(r=>r.date),...(data.pace_outlook?.history??[]).map(r=>r.date),
    ...(data.race_date?[data.race_date]:[])];
  let start=Math.min(...days.map(d=>+new Date(stamp(d)))),end=Math.max(...days.map(d=>+new Date(stamp(d))));
  if(start===end){start-=7*86400000;end+=21*86400000;}
  return {start,end,today};
};
const statusCopy={
  supporting_volume:['Building supporting volume','Long-run progression is paused while your easy volume catches up.'],
  progressing:['Progressing toward your goal','Volume and long-run distance can increase with completed training.'],
  long_hold:['Holding the long run','Volume can grow; the long run needs more supporting training before increasing.'],
  holding:['Holding current training','Progression is paused. Your next plan review uses the running you complete.'],
  at_target:['Maintaining your target','The plan has reached its current distance and volume targets.'],
  update_pending:['Plan update pending','Refresh your Weekly Plan to update progression and this projection.'],
  review:['Plan review needed','Check your Weekly Plan before using a growth projection.'],
  taper:['Tapering toward race day','Training is easing back. The build rate no longer applies.'],
  race:['Race day','Training progression is paused for your event.'],
  post_race:['Returning after your race','The plan is allowing a return to training before building again.'],
};

export function goalJourneyChart(data,dateLabel,{compact=false}={}) {
  const historical=data.recorded??[],scheduled=data.scheduled??[],projection=data.projection?.points??[];
  const {start,end,today}=domain(data);
  const width=chartWidth(),height=compact?(width===420?260:220):300,left=width===420?70:64,right=width-24,top=28,bottom=height-48;
  const ymax=Math.ceil(Math.max(data.target_long_miles,data.current_long_miles,
    ...historical.map(r=>r.miles),...scheduled.map(r=>r.miles))*1.15);
  const x=day=>left+(+new Date(stamp(day))-start)/(end-start)*(right-left);
  const y=miles=>bottom-miles/ymax*(bottom-top);
  const line=(rows,field)=>rows.map((r,i)=>`${i?'L':'M'}${x(r.date).toFixed(1)},${y(r[field]).toFixed(1)}`).join(' ');
  const axis=[0,ymax/2,ymax];
  const tickDates=[new Date(start).toISOString().slice(0,10),today,new Date(end).toISOString().slice(0,10)];
  // Keep labels readable when Now is close to the leftmost history date.
  const ticks=[...new Set(tickDates)];
  const axisDate=(day,last)=>new Intl.DateTimeFormat(undefined,{month:'short',
    ...(last&&end-start>120*86400000?{year:'numeric'}:{day:'numeric'}),timeZone:'UTC'}).format(new Date(stamp(day)));
  const targetY=y(data.target_long_miles),nowX=x(today);
  const points=[...historical.map(r=>{
    const at=runChartAttributes({id:r.activity_id,x:x(r.date),y:y(r.miles),
      title:dateLabel(stamp(r.date)),details:[`${num(r.miles)} mi · ${r.role==='long'?'completed long run':'completed run'}`]});
    return `<a href="#run/${r.activity_id}"><circle class="journey-recorded" cx="${x(r.date)}" cy="${y(r.miles)}" r="4.5" ${at}/></a>`;
  }),...scheduled.map((r,i)=>`<circle class="journey-scheduled" cx="${x(r.date)}" cy="${y(r.miles)}" r="4" ${runChartAttributes({
    id:`planned-long-${i}`,x:x(r.date),y:y(r.miles),title:dateLabel(stamp(r.date)),details:[`${num(r.miles)} mi · scheduled long run`]})}/>`)].join('');
  return `<p class="journey-axis-label">Miles per long run</p><div class="journey-chart-wrap"><svg class="insight-chart goal-journey-chart" viewBox="0 0 ${width} ${height}" role="img"
    aria-label="Completed and scheduled long runs with a conditional path toward the long-run target"
    data-run-chart data-run-top="${top}" data-run-bottom="${bottom}">
    <title>Long-run distance toward your goal</title>
    ${projection.length?`<rect class="journey-future" x="${nowX}" y="${top}" width="${right-nowX}" height="${bottom-top}"/>`:''}
    ${axis.map(v=>`<line class="grid" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${num(v)}</text>`).join('')}
    <line class="journey-target" x1="${left}" x2="${right}" y1="${targetY}" y2="${targetY}"/>
    <text class="journey-target-label" x="${right}" y="${targetY-9}" text-anchor="end">${num(data.target_long_miles)} mi target</text>
    <line class="journey-now" x1="${nowX}" x2="${nowX}" y1="${top}" y2="${bottom}"/>
    ${projection.length?`<path class="journey-projection" d="${line(projection,'long_miles')}"/>`:''}
    ${scheduled.length?`<path class="journey-issued-line" d="${line([{date:today,miles:data.current_long_miles},...scheduled],'miles')}"/>`:''}
    ${data.taper_start&&+new Date(stamp(data.taper_start))>=start?`<line class="journey-taper" x1="${x(data.taper_start)}" x2="${x(data.taper_start)}" y1="${top}" y2="${bottom}"/><text x="${Math.min(right,x(data.taper_start)+6)}" y="${bottom-8}">Taper</text>`:''}
    ${points}
    ${ticks.map((d,i)=>`<text class="${d===today?'journey-now-tick':i===0?'journey-start-tick':'journey-end-tick'}" x="${x(d)}" y="${bottom+25}" text-anchor="${i===0?'start':i===ticks.length-1?'end':'middle'}">${d===today?'Now':e(axisDate(d,i===ticks.length-1))}</text>`).join('')}
    </svg></div>`;
}

export function paceOutlookChart(data,dateLabel,{compact=false}={}) {
  const outlook=data.pace_outlook,points=outlook?.points??[],history=outlook?.history??[];
  if(!outlook?.available||points.length<2)return '';
  const {start,end,today}=domain(data),width=chartWidth(),height=compact?(width===420?260:220):270,left=width===420?70:64,right=width-24,top=24,bottom=height-46;
  const last=points.at(-1);
  const hasRange=r=>Number.isFinite(r.low_pace_min_mile)&&Number.isFinite(r.high_pace_min_mile);
  const scale=paceChartScale([...history.map(r=>r.pace_min_mile),...points.map(r=>r.pace_min_mile),
    ...points.filter(hasRange).flatMap(r=>[r.low_pace_min_mile,r.high_pace_min_mile])]);
  const x=d=>left+(+new Date(stamp(d))-start)/(end-start)*(right-left);
  const y=p=>top+(p-scale.min)/(scale.max-scale.min)*(bottom-top);
  const line=(rows,key)=>rows.map((r,i)=>`${i?'L':'M'}${x(r.date).toFixed(1)},${y(r[key]).toFixed(1)}`).join(' ');
  // Use only returned ranges. Do not join missing early horizons to a falsely
  // certain Now, invent a starting floor, or bridge unsupported interior gaps.
  const rangeSegments=[];
  for(const r of points) {
    if(hasRange(r)) {
      if(!rangeSegments.length||!rangeSegments.at(-1).length)rangeSegments.push([r]);
      else rangeSegments.at(-1).push(r);
    } else if(rangeSegments.at(-1)?.length)rangeSegments.push([]);
  }
  const bands=rangeSegments.filter(rows=>rows.length>=2).map(rows=>{
    const band=line(rows,'low_pace_min_mile')+' '+[...rows].reverse().map(r=>`L${x(r.date).toFixed(1)},${y(r.high_pace_min_mile).toFixed(1)}`).join(' ')+' Z';
    return `<path class="outlook-band" d="${band}"/><path class="outlook-bound" d="${line(rows,'low_pace_min_mile')}"/><path class="outlook-bound" d="${line(rows,'high_pace_min_mile')}"/>`;
  }).join('');
  const rangeX=x(last.date);
  const dated=d=>new Intl.DateTimeFormat(undefined,{month:'short',year:'numeric',timeZone:'UTC'}).format(new Date(stamp(d)));
  return `<p class="journey-axis-label"><span class="benchmark-faster">Faster ↑</span><span>min/mi · ${num(outlook.reference_hr_bpm)} bpm</span></p>
    <svg class="insight-chart pace-outlook-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Aerobic pace trend and training-response projection with a rough forecast-error band where estimates are available" data-run-chart data-run-top="${top}" data-run-bottom="${bottom}">
      <title>Aerobic pace outlook</title>
      ${scale.ticks.map(p=>`<line class="grid" x1="${left}" x2="${right}" y1="${y(p)}" y2="${y(p)}"/><text x="${left-10}" y="${y(p)+4}" text-anchor="end">${e(pace(p).replace('/mi',''))}</text>`).join('')}
      ${bands}
      ${hasRange(last)?`<path class="outlook-range-marker" d="M${rangeX},${y(last.low_pace_min_mile)} V${y(last.high_pace_min_mile)} M${rangeX-6},${y(last.low_pace_min_mile)} h12 M${rangeX-6},${y(last.high_pace_min_mile)} h12"/>`:''}
      ${hasRange(last)&&last.high_pace_min_mile-last.low_pace_min_mile>.02?`<text class="outlook-bound-label" x="${rangeX}" y="${y(last.low_pace_min_mile)-8}" text-anchor="end">${e(roundPace(last.low_pace_min_mile))}</text><text class="outlook-bound-label" x="${rangeX}" y="${y(last.high_pace_min_mile)+16}" text-anchor="end">${e(roundPace(last.high_pace_min_mile))}</text>`:''}
      ${history.length?`<path class="outlook-history" d="${line([...history,{date:today,pace_min_mile:outlook.current_pace_min_mile}],'pace_min_mile')}"/>`:''}
      <path class="outlook-line" d="${line(points,'pace_min_mile')}"/>
      <line class="journey-now" x1="${x(today)}" x2="${x(today)}" y1="${top}" y2="${bottom}"/>
      ${points.map((r,i)=>`<g ${runChartAttributes({id:`pace-outlook-${i}`,x:x(r.date),y:y(r.pace_min_mile),title:dateLabel(stamp(r.date)),
        details:[`${i?'Projected':'Current'} pace: ${roundPace(r.pace_min_mile)}`,...(hasRange(r)?[`Rough range: ${roundPace(r.low_pace_min_mile)}–${roundPace(r.high_pace_min_mile)}`]:[]),`At ${num(outlook.reference_hr_bpm)} bpm`]})}><circle class="outlook-point${i===points.length-1?' endpoint':''}" cx="${x(r.date)}" cy="${y(r.pace_min_mile)}" r="${i===points.length-1?4:3}"/></g>`).join('')}
      <text class="journey-start-tick" x="${left}" y="${bottom+26}">${e(dated(new Date(start).toISOString().slice(0,10)))}</text>
      <text class="journey-now-tick" x="${x(today)}" y="${bottom+26}" text-anchor="middle">Now</text>
      <text x="${right}" y="${bottom+26}" text-anchor="end">${e(dated(outlook.date))}</text>
    </svg>`;
}

export function paceOutlookPanel(data,dateLabel,{compact=false}={}) {
  const o=data.pace_outlook;
  if(!o)return compact?'<section class="pace-outlook outlook-pending"><h3>Pace outlook</h3><p>More comparable runs are needed for a pace outlook.</p></section>':'';
  if(!o.available)return `<section class="pace-outlook outlook-pending"><h3>Pace outlook</h3><p>${e(o.reason)}</p></section>`;
  const month=new Intl.DateTimeFormat(undefined,{month:'long',year:'numeric',timeZone:'UTC'}).format(new Date(stamp(o.date)));
  const key='<div class="journey-key"><span><i class="recorded-key"></i>Aerobic trend</span><span><i class="projection-key"></i>Projection</span><span><i class="outlook-range-key"></i>Rough range</span></div>';
  return `<section class="pace-outlook" aria-labelledby="pace-outlook-title"><div class="journey-layout">
    <div class="outlook-summary"><p class="eyebrow" id="pace-outlook-title">Projected aerobic pace</p>
      <p class="outlook-date">${e(month)}${o.date_basis==='taper'?' · before taper':''}</p>
      <strong class="outlook-value${o.expected_pace_min_mile>o.current_pace_min_mile?' slowing':''}">${roundPace(o.expected_pace_min_mile)}</strong>
      <p class="outlook-range">Rough range <strong>${roundPace(o.low_pace_min_mile)}–${roundPace(o.high_pace_min_mile)}</strong></p>
      <p class="outlook-current">Now ${roundPace(o.current_pace_min_mile)} · at ${num(o.reference_hr_bpm)} bpm</p>
      <p class="chart-note">If training stays consistent.${compact?'':' Updates with completed training.'}</p>
      <details><summary>About this outlook</summary>${compact?'<p>Updates with completed training.</p>':''}<p>Uses your training history and a 42-day aerobic trend. The rough range comes from errors predicting actual future runs, with error growth estimated from your history—not a confidence interval. The band begins at the earliest supported forecast horizon; it does not estimate uncertainty today. The central line remains a training-response midpoint, not a separately fitted expected value.</p><p>Assumes consistent training along this progression. Illness or breaks can change the outlook. Aerobic pace at ${num(o.reference_hr_bpm)} bpm—not a prescribed pace or a prediction that you can hold it for ${num(data.target_long_miles)} miles.</p></details>
    </div><div class="journey-visual">${compact?'':key}
      ${paceOutlookChart(data,dateLabel,{compact})}${compact?key:''}
    </div></div></section>`;
}

export function goalJourneyPanel(data,dateLabel=value=>new Date(value).toLocaleDateString(),{compact=false,outlook='distance'}={}) {
  if(!data?.available) return `<article class="wide-card goal-journey"><p class="eyebrow">Your goal</p>
    <h2>Progress toward your goal</h2><p>${e(data?.reason??'Create your Weekly Plan to see goal progression.')}</p>
    <a href="#next-run">Weekly Plan →</a></article>`;
  const race=data.goal_kind!=='general_fitness';
  const [status,detail]=statusCopy[data.status]??['Progression not available','Refresh your Weekly Plan for an updated status.'];
  const recent=data.recent_long_run;
  const finish=data.projection?.target_date;
  const taper=data.projection?.taper;
  const active=['supporting_volume','progressing','long_hold'].includes(data.status);
  const eta=race?(taper?`${num(taper.long_miles)} mi by taper`:finish?'Targets reached before taper':null):
    finish?dateLabel(stamp(finish)):null;
  const month=finish&&!race?new Intl.DateTimeFormat(undefined,{month:'long',year:'numeric',timeZone:'UTC'}).format(new Date(stamp(finish))):null;
  const volumeLabel=data.status==='taper'?'Build paused for taper':active&&data.volume_growth_per_7>0?
    `Up to ${num(data.volume_growth_per_7*100)}% per week`:'Growth on hold';
  const selected=outlook==='pace'?'pace':'distance';
  const key='<div class="journey-key"><span><i class="recorded-key"></i>Completed</span><span><i class="scheduled-key"></i>Scheduled</span><span><i class="projection-key"></i>If growth continues</span></div>';
  return `<article class="wide-card goal-journey${compact?' goal-outlook-window':''}" aria-labelledby="goal-journey-title">
    <div class="card-heading"><div${compact?' class="journey-heading"':''}><p class="eyebrow">Your goal</p>
      ${compact?'<div class="journey-title-row">':''}<h2 id="goal-journey-title">${race?`${e(titleCase(data.goal_kind))} · ${e(dateLabel(stamp(data.race_date)))}`:`Build to a ${num(data.target_long_miles)}-mile weekly long run`}</h2>${compact?'<a class="journey-edit primary-button" href="#settings">Edit goal</a></div>':''}</div>
      ${compact?`<div class="segmented journey-switch" aria-label="Goal outlook">${[['distance','Distance'],['pace','Pace outlook']].map(([key,label])=>`<button type="button" data-goal-outlook="${key}" aria-controls="goal-outlook-${key}" aria-pressed="${selected===key}" class="${selected===key?'selected':''}">${label}</button>`).join('')}</div>`:'<a class="journey-edit" href="#settings">Edit goal →</a>'}</div>
    ${compact?`<div id="goal-outlook-distance" data-goal-panel="distance"${selected==='distance'?'':' hidden'}>`:''}
    <div class="journey-layout"><div class="journey-summary">
      <dl class="journey-distance"><div><dt>Recent long run</dt><dd>${recent?`${num(recent.miles)} <span>mi</span>`:'Not recorded yet'}</dd></div>
      <div><dt>Weekly long-run target</dt><dd>${num(data.target_long_miles)} <span>mi</span></dd></div></dl>
      <div class="journey-status"><h3>${e(status)}</h3><p>${e(detail)}</p></div>
      ${compact?`<p class="journey-growth">${e(volumeLabel)}</p>${eta&&active?`<div class="journey-eta"><span>${race?'At this rate':'Target at this rate'}</span><strong>${e(month??eta)}</strong>${race?`<small>${data.race_projection_meets_target?'Preparation targets fit before taper.':'Below the preparation target at taper.'}</small>`:''}</div>`:''}<details class="journey-details"><summary>Progression details</summary>`:''}
      <dl class="journey-facts"><div><dt>Current volume target</dt><dd>${num(data.volume_per_7)} mi / 7 days</dd></div>
      ${compact?'':`<div><dt>Volume progression</dt><dd>${e(volumeLabel)}</dd></div>`}
      <div><dt>Current long-run prescription</dt><dd>${num(data.current_long_miles)} mi${data.status==='at_target'?' · at target':data.status==='progressing'&&data.long_growth_allowed?' · can increase':' · held'}</dd></div></dl>
      ${data.status==='supporting_volume'?`<p class="journey-milestone">${data.volume_per_7<data.supporting_volume_for_long?
        `Next planning milestone: <strong>${num(data.supporting_volume_for_long)} mi / 7 days</strong>`:
        'Next milestone: complete the supporting volume in your plan.'}</p>`:''}
      ${!compact&&eta&&active?`<div class="journey-eta"><span>${race?'At this rate':'Target at this rate'}</span><strong>${e(month??eta)}</strong>
        ${race?`<small>${data.race_projection_meets_target?'Preparation targets fit before taper.':'Below the preparation target at taper.'}</small>`:''}</div>`:''}
      ${compact?'</details>':''}
    </div><div class="journey-visual">
      ${compact?'':key}${goalJourneyChart(data,dateLabel,{compact})}${compact?key:''}
      <p class="chart-note">${data.projection?.points?.length?'The dotted path assumes continued growth and completed training—not a future schedule.':'Only completed and currently scheduled runs are shown.'}</p>
    </div></div>${compact?`</div><div id="goal-outlook-pace" data-goal-panel="pace"${selected==='pace'?'':' hidden'}>${paceOutlookPanel(data,dateLabel,{compact})}</div>`:paceOutlookPanel(data,dateLabel)}
    ${data.goal_change_pending?'<p class="chart-note">Goal change pending. This shows the issued plan.</p>':''}</article>`;
}

// Both views use the same fetched projection; switching never refreshes a plan
// or refits a forecast. Native buttons retain keyboard and touch support.
export function wireGoalOutlook(root,onChange=()=>{}) {
  root.querySelectorAll('[data-goal-outlook]').forEach(button=>button.addEventListener('click',()=>{
    const selected=button.dataset.goalOutlook;
    root.querySelectorAll('[data-goal-outlook]').forEach(b=>{
      const active=b.dataset.goalOutlook===selected;
      b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));
    });
    root.querySelectorAll('[data-goal-panel]').forEach(panel=>{panel.hidden=panel.dataset.goalPanel!==selected;});
    onChange(selected);
  }));
}
