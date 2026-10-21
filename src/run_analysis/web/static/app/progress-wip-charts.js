// Read-only descriptive charts. No new fitness, readiness or recovery model.
import { escapeHtml, number, pace, titleCase } from './format.js';
import { runChartAttributes } from './chart-interaction.js';

const colors={easy:'#78d8b8',long:'#91bcec',quality:'#e7bc68',short:'#91bcec',long_rep:'#e7bc68',continuous:'#78d8b8'};
const statusColors={above:'#91bcec',within:'#78d8b8',near:'#8796a5',below:'#b5a5d5',unknown:'#35434e',unavailable:'#35434e'};
const path=(rows,x,y,field)=>rows.map((r,i)=>`${i?'L':'M'}${x(r).toFixed(1)},${y(field(r)).toFixed(1)}`).join(' ');
function frame(rows,start,end,min,max,date,{height=210,yLabel=''}={}) {
  const W=720,H=height,left=48,right=705,top=24,bottom=H-32;
  const x=r=>left+(+new Date(r.as_of??r.start_time)-+new Date(start))/Math.max(1,+new Date(end)-+new Date(start))*(right-left);
  const y=v=>bottom-(v-min)/Math.max(.1,max-min)*(bottom-top);
  const grid=[min,(min+max)/2,max].map(v=>`<line class="grid-line" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${number(v,0)}</text>`).join('');
  const axes=`${grid}<text x="${left}" y="14">${escapeHtml(yLabel)}</text><text x="${left}" y="${H-8}">${date(start)}</text><text x="${right}" y="${H-8}" text-anchor="end">${date(end)}</text>`;
  return {W,H,x,y,axes,left,right,bottom};
}
function legend(entries) {return `<div class="insight-legend">${entries.map(([label,color,dashed])=>`<span><i style="--key-color:${color}" class="${dashed==='dotted'?'dotted':dashed?'dashed':''}"></i>${escapeHtml(label)}</span>`).join('')}</div>`;}
function stackPaths(rows,keys,f) {
  return keys.map((key,at)=>{
    const low=r=>keys.slice(0,at).reduce((s,k)=>s+(r[k]??0),0);
    const high=r=>low(r)+(r[key]??0);
    return `<path d="${path(rows,f.x,f.y,high)} ${[...rows].reverse().map(r=>`L${f.x(r).toFixed(1)},${f.y(low(r)).toFixed(1)}`).join(' ')} Z" fill="${colors[key]}" opacity=".48"/>`;
  }).join('');
}
function completeSegments(rows) {
  const groups=[];
  for(const r of rows) {
    if(!r.complete) continue;
    // Adjacent local calendar dates can be 23/25 hours apart at DST changes.
    if(!groups.length||+new Date(r.as_of)-+new Date(groups.at(-1).at(-1).as_of)>26*3600000) groups.push([]);
    groups.at(-1).push(r);
  }
  return groups;
}

// Split the plotted line at crossings of the moving range, not at midnight.
// Both the load and range are linearly interpolated, just like their SVG paths.
function coloredLoadPath(group,f,stateFor,colorFor) {
  const pieces=[];
  const add=(status,points)=>{
    if(pieces.at(-1)?.status===status) pieces.at(-1).points.push(...points.slice(1));
    else pieces.push({status,points});
  };
  if(group.length===1) add(stateFor(group[0])?.status??'unknown',[group[0]]);
  for(let i=1;i<group.length;i++) {
    const a=group[i-1],b=group[i],sa=stateFor(a),sb=stateFor(b);
    const ra=sa?.recommended_range,rb=sb?.recommended_range;
    const cuts=[0,1];
    if(ra&&rb) for(const edge of ['low','high']) {
      const da=a.recent.value-ra[edge],db=b.recent.value-rb[edge];
      if(da*db<0) cuts.push(da/(da-db));
    }
    cuts.sort((a,b)=>a-b);
    const point=t=>({as_of:new Date(+new Date(a.as_of)+t*(+new Date(b.as_of)-+new Date(a.as_of))).toISOString(),
      recent:{value:a.recent.value+t*(b.recent.value-a.recent.value)}});
    for(let j=1;j<cuts.length;j++) {
      const t=(cuts[j-1]+cuts[j])/2,value=point(t).recent.value;
      const low=ra&&rb?ra.low+t*(rb.low-ra.low):null,high=ra&&rb?ra.high+t*(rb.high-ra.high):null;
      const status=low!==null?(value<low?'below':value>high?'above':'within'):
        sa?.status===sb?.status?(sa?.status??'unknown'):'unknown';
      add(status,[point(cuts[j-1]),point(cuts[j])]);
    }
  }
  return pieces.map(({status,points})=>`<path class="load-line load-recent" data-load-status="${status}" style="stroke:${colorFor(status)}" d="${path(points,f.x,f.y,r=>r.recent.value)}"/>`).join('');
}

export function fatiguePanel(d,date,{embedded=false}={}) {
  const display=!!d.load_display,load=d.load_display??d.training_load;
  const guidance=d.load_guidance, guideByTime=new Map((guidance?.daily??[]).map(r=>[r.as_of,r]));
  const stateColor=key=>guidance&&key==='above'?'#e7bc68':statusColors[key]??statusColors.unknown;
  const statuses=guidance?.daily??d.status_history;
  const statusByTime=new Map((statuses??[]).map(r=>[r.as_of,r]));
  if(!load?.daily?.some(r=>r.recent.complete)) return embedded?`<p class="chart-empty">${escapeHtml(load?.reason??'More recorded HR history is needed to show training load.')}</p>`:`<article class="wide-card fatigue-card"><h2>Training load</h2><p class="chart-empty">${escapeHtml(load?.reason??'More recorded HR history is needed to show training load.')}</p></article>`;
  const rows=load.daily;
  const keys=display?['baseline','recent']:['short','baseline','recent'];
  const max=Math.max(1,...rows.flatMap(r=>[...keys.filter(k=>r[k].complete).map(k=>r[k].high),guideByTime.get(r.as_of)?.recommended_range?.high??0]))*1.08;
  const f=frame(rows,rows[0].as_of,d.as_of,0,max,date,{height:250});
  const ticks=[0,max/2,max].map(v=>`<line class="grid-line" x1="${f.left}" x2="${f.right}" y1="${f.y(v)}" y2="${f.y(v)}"/><text x="${f.left-8}" y="${f.y(v)+4}" text-anchor="end">${number(v,0)}</text>`).join('');
  // Each series has its own coverage gate. All use the same unit-gain HR load
  // average: never overlay the differently normalized scheduler strain here.
  const curves=keys.map(key=>completeSegments(rows.map(r=>({...r,complete:r[key].complete}))).map(group=>`${key!=='short'?`<path class="load-band load-band-${key}" d="${path(group,f.x,f.y,r=>r[key].high)} ${[...group].reverse().map(r=>`L${f.x(r).toFixed(1)},${f.y(r[key].low).toFixed(1)}`).join(' ')} Z"/>`:''}${key==='recent'&&statuses?coloredLoadPath(group,f,r=>statusByTime.get(r.as_of),stateColor):`<path class="load-line load-${key}" d="${path(group,f.x,f.y,r=>r[key].value)}"/>`}`).join('')).join('');
  const comparisonZone=guidance?completeSegments(rows.map(r=>({...r,range:guideByTime.get(r.as_of)?.recommended_range,complete:!!guideByTime.get(r.as_of)?.recommended_range}))).map(group=>`<path class="load-recommended-zone" d="${path(group,f.x,f.y,r=>r.range.high)} ${[...group].reverse().map(r=>`L${f.x(r).toFixed(1)},${f.y(r.range.low).toFixed(1)}`).join(' ')} Z"/>`).join(''):display?completeSegments(rows.map(r=>({...r,complete:r.baseline.complete}))).map(group=>`<path class="load-comparison-zone" d="${path(group,f.x,f.y,r=>r.baseline.value*(1+load.equality_tolerance))} ${[...group].reverse().map(r=>`L${f.x(r).toFixed(1)},${f.y(r.baseline.value*(1-load.equality_tolerance)).toFixed(1)}`).join(' ')} Z"/>`).join(''):'';
  const dayKey=new Intl.DateTimeFormat('en-CA',{timeZone:d.timezone,year:'numeric',month:'2-digit',day:'2-digit'});
  const byDay=new Map(rows.map(r=>[dayKey.format(new Date(r.as_of)),r]));
  const stateByDay=new Map((statuses??[]).map(r=>[dayKey.format(new Date(r.as_of)),r]));
  const singlePoint=rows.filter(r=>r.recent.complete).length===1;
  const dailyMarks=statuses?rows.map(r=>{
    const state=stateByDay.get(dayKey.format(new Date(r.as_of)));
    if(!state)return '';
    const x=f.x(r),y=r.recent.complete?f.y(r.recent.value):f.bottom;
    const runs=(d.activities??[]).filter(a=>dayKey.format(new Date(a.start_time))===dayKey.format(new Date(r.as_of)));
    const difference=Number.isFinite(r.ratio)?`${number(Math.abs(r.ratio-1)*100,0)}% ${r.ratio>=1?'higher':'lower'} than your usual load`:state.detail;
    const attrs=runChartAttributes({id:`day:${r.as_of}`,x,y,title:date(r.as_of),details:[state.label,guidance?state.action:display?difference:state.detail,
      ...(r.recent.complete?[`${display?'Recent load':'7-day load'}: ${number(r.recent.value)} HR load/day`]:['Recent load unavailable']),
      ...(r.baseline.complete?[`${display?'Your usual load (6-week trend)':'42-day baseline'}: ${number(r.baseline.value)} HR load/day`]:['Usual load unavailable']),
      ...(state.recommended_range?[`Recommended: ${number(state.recommended_range.low)}–${number(state.recommended_range.high)} HR load/day`]:[]),
      ...(state.boundary_overlap?['Estimate overlaps a range boundary']:[]),
      ...(display&&state.status==='unknown'&&Number.isFinite(r.ratio)?[state.detail]:[]),
      ...(runs.length?[`${runs.length} run${runs.length===1?'':'s'} · ${number(runs.reduce((sum,a)=>sum+a.distance_miles,0))} mi`]:[])]});
    return `<g ${attrs}>${r.recent.complete?`<circle class="fatigue-run load-run ${singlePoint?'':'load-display-marker'}" style="fill:${stateColor(state.status)}" cx="${x}" cy="${y}" r="3"/>`:''}</g>`;
  }).join(''):'';
  const dots=statuses?dailyMarks:(d.activities??[]).map(a=>{
    const r=byDay.get(dayKey.format(new Date(a.start_time)));
    if(!r?.recent.complete)return '';
    const x=f.x(r),y=f.y(r.recent.value);
    return `<g ${runChartAttributes({id:a.activity_id,x,y,title:date(a.start_time),details:[`${number(a.distance_miles)} mi · ${titleCase(a.role)}`,`7-day load: ${number(r.recent.value)} HR load/day`,...(r.baseline.complete?[`42-day baseline: ${number(r.baseline.value)} HR load/day`]:['42-day baseline: not enough history']),...(r.short.complete?[`2-day load: ${number(r.short.value)} HR load/day`]:[])]})}><a href="#run/${a.activity_id}" aria-label="${escapeHtml(date(a.start_time))}: run details"><circle class="fatigue-run load-run" cx="${x}" cy="${y}" r="3"/></a></g>`;
  }).join('');
  const history=`<svg class="insight-chart fatigue-history" viewBox="0 0 ${f.W} ${f.H}" role="img" aria-label="${guidance?'Recent training load colored by its moving recommended range, with six-week training history':display?'Smoothed recent training load colored by comparison with your usual load from a six-week trend':'Seven-day recent training load and 42-day baseline, with a lighter two-day load layer'}; all use the same HR load per day, not percent unrecovered" data-run-chart ${statuses?'data-chart-description="Hover or tap to inspect daily load and status. Left and right arrows select dates."':''} data-run-top="24" data-run-bottom="${f.bottom}">${ticks}${comparisonZone}${curves}${dots}<text x="${f.left}" y="14">HR load / day</text><text x="${f.left}" y="${f.H-8}">${date(rows[0].as_of)}</text><text x="${f.right}" y="${f.H-8}" text-anchor="end">${date(d.as_of)}</text></svg>`;
  const statusLegend=statuses?legend([...new Map(statuses.map(r=>[r.status,r.label])).entries()].map(([key,label])=>[label,stateColor(key)])):'';
  const statusHelp=guidance?'':statuses?`<details class="training-status-method"><summary>About the line colors</summary><p>Your usual load is a reference for comparison, not the amount you should aim to do. Line colors compare recent load with your six-week trend; the faint band marks ±5%, not an optimal or safe-load zone. Uncertain comparisons stay gray.</p></details>`:'';
  const ratio=load.ratio;
  const ratioCaption=Number.isFinite(ratio)?display?`${number(Math.abs(ratio-1)*100,0)}% ${ratio>=1?'higher':'lower'} than your usual load`:`${number(Math.abs(ratio-1)*100,0)}% ${ratio>=1?'above':'below'} your longer-term baseline`:display?'Comparison not yet available':'Longer-term baseline not yet available';
  const lineLegend=guidance?[['Recent load','#78d8b8'],['6-week training history','#91bcec',true],['Recommended range','#78d8b8']]:display?[['Recent load','#78d8b8'],['Your usual load · 6-week trend','#91bcec',true],['Similar load · ±5%','#91bcec']]:[['Recent · 7 days','#78d8b8'],['Baseline · 42 days','#91bcec',true],['Short-term · 2 days','#8796a5',true]];
  const level=guidance?'':`<div class="fatigue-level"><strong>${Number.isFinite(ratio)?`${number(ratio,2)}×`:'—'}</strong><span>${display?'recent load vs your usual load':'recent vs longer-term load'}<small>${escapeHtml(ratioCaption)}</small></span></div>`;
  const note=guidance?'The shaded range adapts to your training history. It is guidance for cardiovascular load—not proof of recovery.':statuses?'Your usual load reflects your six-week training trend—not a target. Line colors show load comparison, not recovery.':'Above the baseline = building load; below = easing off. Neither is automatically good or bad.';
  return `<${embedded?'div':'article'} class="${embedded?'load-chart-pane':'wide-card fatigue-card'}">
    <div class="card-heading"><div><p class="eyebrow">Recorded training · ${statuses?'last 3 months':'history'}</p>${embedded?'':'<h2>Training load</h2>'}</div><span class="quality">${guidance?'Personal load range':display?'Smoothed trend':'Load, not readiness'}</span></div>
    ${level}${history}${legend(lineLegend)}${statusLegend}<p class="chart-note">${note}</p>${statusHelp}</${embedded?'div':'article'}>`;
}

const qualityShareStatus=(value,low,high)=>value<low?'below':value>high?'above':'within';
const qualityShareColor=status=>status==='above'?'#e7bc68':statusColors[status];

// Keep each section of the line in the color of its plotted range, splitting
// exactly at the reference edges rather than coloring a whole day at once.
function coloredQualityPath(group,x,y,low,high) {
  const pieces=[];
  const add=(status,points)=>{
    if(pieces.at(-1)?.status===status) pieces.at(-1).points.push(...points.slice(1));
    else pieces.push({status,points});
  };
  if(group.length===1) add(qualityShareStatus(group[0].quality_percent,low,high),[group[0]]);
  for(let i=1;i<group.length;i++) {
    const a=group[i-1],b=group[i],cuts=[0,1];
    for(const edge of [low,high]) {
      const da=a.quality_percent-edge,db=b.quality_percent-edge;
      if(da*db<0) cuts.push(da/(da-db));
    }
    cuts.sort((a,b)=>a-b);
    const point=t=>({as_of:new Date(+new Date(a.as_of)+t*(+new Date(b.as_of)-+new Date(a.as_of))).toISOString(),
      quality_percent:a.quality_percent+t*(b.quality_percent-a.quality_percent)});
    for(let j=1;j<cuts.length;j++) add(qualityShareStatus(point((cuts[j-1]+cuts[j])/2).quality_percent,low,high),
      [point(cuts[j-1]),point(cuts[j])]);
  }
  return pieces.map(({status,points})=>`<path class="effort-share-line" data-quality-status="${status}" style="stroke:${qualityShareColor(status)}" d="${path(points,x,y,r=>r.quality_percent)}"/>`).join('');
}

export function effortBalance(d,date,{compact=false}={}) {
  const balance=d.effort?.session_balance,current=balance?.current;
  const share=current?.quality_percent;
  const limit=balance?.trailing_runs??20;
  const count=current?.completed_runs??((current?.classified_runs??0)+(current?.unknown_runs??0));
  if(!Number.isFinite(share)) return `<div class="effort-empty"><h3>Quality sessions</h3><p>No classified runs among your last ${limit} runs.${current?.unknown_runs?` ${current.unknown_runs} unclassified run${current.unknown_runs===1?'':'s'} excluded.`:''}</p></div>`;
  const low=balance.reference.low,high=balance.reference.high;
  const status=current.unknown_runs?'Classification incomplete':current.classified_runs<4?'Few runs to compare':share<low?'Lower quality share':share>high?null:'Within typical range';
  const tone=current.unknown_runs||current.classified_runs<4?'uncertain':'neutral';
  const rows=balance.daily.map(r=>({...r,complete:Number.isFinite(r.quality_percent)&&r.unknown_runs===0}));
  const visible=rows.filter(r=>r.complete),max=Math.max(50,Math.ceil(Math.max(0,...visible.map(r=>r.quality_percent),share)/10)*10);
  const W=320,H=compact?150:190,left=38,right=305,top=12,bottom=H-33;
  const start=+new Date(rows[0]?.as_of??d.window_start),end=+new Date(d.as_of);
  const x=r=>left+(+new Date(r.as_of)-start)/Math.max(1,end-start)*(right-left);
  const y=v=>bottom-v/max*(bottom-top);
  const axes=[0,high,max].map(v=>`<line class="grid-line" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-5}" y="${y(v)+4}" text-anchor="end">${v}%</text>`).join('');
  const history=`<svg class="effort-history" viewBox="0 0 ${W} ${H}" role="img" aria-label="Quality sessions as a percentage of classified runs among the last ${limit} completed runs at each date. Shaded reference range ${low} to ${high} percent, not the planner target. Purple below, green within, orange above. Gaps mean incomplete classification."><rect class="effort-reference" x="${left}" y="${y(high)}" width="${right-left}" height="${y(low)-y(high)}"><title>${low}–${high}% reference</title></rect>${axes}${completeSegments(rows).map(group=>coloredQualityPath(group,x,y,low,high)).join('')}${visible.filter((r,i)=>i%7===0||r===rows.at(-1)).map(r=>`<circle style="fill:${qualityShareColor(qualityShareStatus(r.quality_percent,low,high))}" cx="${x(r)}" cy="${y(r.quality_percent)}" r="${r===rows.at(-1)?4:2}"><title>${escapeHtml(date(r.as_of))}: ${r.quality_sessions} of ${r.classified_runs} runs · ${number(r.quality_percent)}% quality</title></circle>`).join('')}<text x="${left}" y="${H-8}">${escapeHtml(date(new Date(start)))}</text><text x="${right}" y="${H-8}" text-anchor="end">${escapeHtml(date(d.as_of))}</text></svg>`;
  return `<div class="card-heading"><div><p class="eyebrow">Last ${count} runs</p><h3>Quality sessions</h3></div></div>
    <div class="effort-share"><strong>${number(share,1)}<small>%</small></strong><span>${current.quality_sessions} of ${current.classified_runs} runs</span></div>
    ${status?`<p class="effort-status ${tone}">${status}</p>`:''}${history}
    ${compact?'':'<p class="effort-context">Follow your Weekly Plan rather than adding runs just to reach this range.</p>'}
    ${current.unknown_runs?`<p class="effort-context">${current.unknown_runs} unclassified run${current.unknown_runs===1?'':'s'} excluded.</p>`:''}
    <details><summary>What counts?</summary><p>Intervals, threshold/progression runs and races count as quality. Easy long runs and brief strides do not; incidental Z3 time alone does not make a run quality.</p><p>Each point uses the last ${limit} completed runs, or all your runs if you have fewer. Rest days do not change the share. The page period only changes the history shown.</p><p>The shaded range is a general reference, not your planner’s target. Run frequency and recovery can put a suitable plan outside it.${compact?' Follow your Weekly Plan rather than adding runs just to reach this range.':''}</p></details>`;
}

export function trainingVolumeChart(d,date,{height=290,compact=false}={}) {
  const rows=d.exposure.daily.map(r=>({...r,...r.distance_by_role}));
  if(!rows.length) return '<p class="chart-empty">No running history in this period.</p>';
  const dayKey=new Intl.DateTimeFormat('en-CA',{timeZone:d.timezone??'UTC',year:'numeric',month:'2-digit',day:'2-digit'});
  const runDate=new Intl.DateTimeFormat(undefined,{timeZone:d.timezone??'UTC',dateStyle:'medium',timeStyle:'short'});
  const daily=new Map();
  for(const a of d.activities??[]) {
    const key=dayKey.format(new Date(a.start_time));
    if(!daily.has(key)) daily.set(key,[]);
    daily.get(key).push(a);
  }
  // Bars use actual recorded runs, never differences of rolling totals. Multiple
  // runs on a date stack, while each keeps its own identity for inspection.
  const bins=rows.map(r=>({...r,runs:daily.get(dayKey.format(new Date(r.as_of)))??[]}));
  const max=Math.max(5,...rows.map(r=>Math.max(r.distance_7d,r.distance_28d_weekly)),...bins.map(r=>r.runs.reduce((s,a)=>s+a.distance_miles,0)))*1.08;
  const f=frame(rows,rows[0].as_of,rows.at(-1).as_of,0,max,date,{height,yLabel:'Miles'});
  const total=path(rows,f.x,f.y,r=>r.distance_7d);
  const width=Math.max(.8,Math.min(11,(f.right-f.left)/bins.length*.65));
  const bars=bins.map(r=>{
    const sums={easy:0,long:0,quality:0};let miles=0;
    const parts=r.runs.map(a=>{
      const bucket=a.role==='long'?'long':['intervals','tempo_threshold','race'].includes(a.role)?'quality':'easy';
      sums[bucket]+=a.distance_miles;
      const base=f.y(miles);miles+=a.distance_miles;const top=f.y(miles),cx=f.x(r);
      const attrs=runChartAttributes({id:a.activity_id,x:cx,y:(top+base)/2,title:runDate.format(new Date(a.start_time)),details:[
        `${number(a.distance_miles)} mi · ${a.role==='tempo_threshold'?'Progression / tempo':titleCase(a.role)}`,
        ...(Number.isFinite(a.moving_minutes)?[`${number(a.moving_minutes,0)} min running`]:[]),
        `7-day total: ${number(r.distance_7d)} mi`,
        `28-day average: ${number(r.distance_28d_weekly)} mi / 7 days`,
      ]});
      return `<g class="volume-run" ${attrs}><a href="#run/${a.activity_id}" aria-label="${escapeHtml(date(a.start_time))}: ${number(a.distance_miles)} miles"><rect x="${cx-width/2}" y="${top}" width="${width}" height="${base-top}" fill="${colors[bucket]}"/></a><circle class="volume-weekly-point" cx="${cx}" cy="${f.y(r.distance_28d_weekly)}" r="4"/></g>`;
    }).join('');
    return `<g><title>${date(r.as_of)}: ${miles?`${number(miles)} miles · easy ${number(sums.easy)}, long ${number(sums.long)}, quality ${number(sums.quality)}`:'no recorded run'}</title>${parts}</g>`;
  }).join('');
  return `${legend([['28-day average',colors.easy],['7-day total','#e6eef5','dotted'],['Easy / other',colors.easy],['Long',colors.long],['Quality',colors.quality]])}<div class="volume-overlay"><svg class="insight-chart volume-trend" viewBox="0 0 ${f.W} ${f.H}" role="img" aria-label="Recorded daily run distances and session types overlaid with a main 28-day weekly average and dotted seven-day mileage reference" data-run-chart data-run-top="24" data-run-bottom="${f.bottom}">${f.axes}<path d="${total}" class="volume-total-line"/>${bars}<path d="${path(rows,f.x,f.y,r=>r.distance_28d_weekly)}" class="long-average"/></svg></div>${compact?'':'<p class="chart-note">Bars = daily miles · lines = rolling volume. Hover or tap to inspect a run.</p>'}`;
}

export function qualityDose(session) {
  const out={short:0,long_rep:0,continuous:0};
  if(session.interval?.available) session.interval.repetitions.filter(r=>r.kind==='work').forEach(r=>out[r.duration_seconds<180?'short':'long_rep']+=r.duration_seconds/60);
  else if(session.continuous) out.continuous=session.continuous.duration_seconds/60;
  return out;
}
export function qualityWorkHistory(d,date) {
  const start=+new Date(d.insights.start),end=+new Date(d.as_of),day=86400000;
  const observed=(d.insights.quality_history??d.quality).map(q=>({...q,...qualityDose(q),known:!!(q.interval?.available||q.continuous)}));
  const rows=Array.from({length:d.insights.days+1},(_,i)=>{
    const at=start+i*day,window=observed.filter(q=>at-28*day<+new Date(q.start_time)&&+new Date(q.start_time)<=at);
    return {as_of:new Date(at).toISOString(),complete:window.every(q=>q.known),...Object.fromEntries(['short','long_rep','continuous'].map(k=>[k,window.reduce((s,r)=>s+r[k],0)]))};
  });
  const latest=rows.at(-1),total=latest.short+latest.long_rep+latest.continuous;
  const displayed=observed.filter(q=>+new Date(q.start_time)>start);
  const covered=displayed.filter(q=>q.known).length;
  const max=Math.max(10,...rows.map(r=>r.short+r.long_rep+r.continuous))*1.08;
  const f=frame(rows,d.insights.start,d.as_of,0,max,date,{yLabel:'Work minutes in the last 28 days'});
  return `<div class="quality-dose-head"><strong>${latest.complete?'':'At least '}${number(total,0)} <small>recorded work minutes</small></strong><span>in the last 28 days · warm-ups and recoveries excluded</span></div>${legend([['Short reps · under 3 min',colors.short],['Longer reps · 3+ min',colors.long_rep],['Continuous efforts',colors.continuous]])}<svg class="insight-chart" viewBox="0 0 ${f.W} ${f.H}" role="img" aria-label="Six months of quality training: trailing 28-day work minutes split into short repetitions, longer repetitions and continuous efforts">${f.axes}${completeSegments(rows).map(s=>stackPaths(s,['short','long_rep','continuous'],f)).join('')}${rows.filter((r,i)=>r.complete&&(i%7===0||i===rows.length-1)).map(r=>`<circle cx="${f.x(r)}" cy="${f.y(r.short+r.long_rep+r.continuous)}" r="2" fill="#dce8ef"><title>${date(r.as_of)}: short reps ${number(r.short)} min; longer reps ${number(r.long_rep)} min; continuous ${number(r.continuous)} min</title></circle>`).join('')}</svg><p class="chart-note">Different colours show different kinds of work—not interchangeable fitness gains.</p><details><summary>What this tells us—and what it doesn’t</summary><p>Work structure is available for ${covered} of ${displayed.length} marked quality sessions in six months. Unresolved sessions leave gaps, not zero work. This shows the stimulus delivered, not proof of adaptation. Longer work can be demanding at a slower pace than short repetitions.</p><p>A quality pace-at-heart-rate trend is not yet estimated. Short repetitions have heart-rate lag; sustained efforts need enough comparable duration, intensity and conditions. The current history is too varied to label a single pooled pace as improvement.</p></details>`;
}


function periodComparison(c,date,{kind='pace',history,domain}={}) {
  const value=v=>Number.isFinite(v)?kind==='pace'?pace(v):`${number(v)}%`:'—';
  const periods=[['Earlier',c.earlier],['Recent',c.recent]];
  const values=periods.flatMap(([,p])=>[p.low,p.high,p.median]).filter(Number.isFinite);
  const spread=values.length?Math.max(...values)-Math.min(...values):0;
  const span=Math.max(kind==='pace'?.3:2,spread*1.3);
  const middle=values.length?(Math.min(...values)+Math.max(...values))/2:0;
  const min=middle-span/2,max=middle+span/2;
  const left=78,right=384,x=v=>left+(v-min)/span*(right-left);
  const similar=Number.isFinite(c.delta)&&Math.abs(c.delta)<(kind==='pace'?1/60:.1);
  const change=!Number.isFinite(c.delta)?null:similar?'About the same':kind==='pace'
    ?`${number(Math.abs(c.delta)*60,0)} sec/mi ${c.delta<0?'faster':'slower'}`
    :`${number(Math.abs(c.delta))} percentage points ${c.delta<0?'less':'more'} drift`;
  const timeChart=history?comparisonHistory(c,history.rows,history.start,history.end,date,{kind,domain}):'';
  return `<div class="matched-periods">${periods.map(([label,p])=>`<div class="matched-period ${label==='Recent'?'recent':''}"><span>${label}<small>${label==='Recent'?'Last 6 weeks':'Earlier in the 6-month history'}</small></span><strong>${value(p.median)}</strong><small>${p.count} ${p.count===1?'session':'sessions'}${p.count?` · ${date(p.first)}${p.first!==p.last?` – ${date(p.last)}`:''}`:''}</small></div>`).join('')}</div>${timeChart|| (values.length?`<svg class="comparison-range" viewBox="0 0 400 132" role="img" aria-label="Earlier and recent ${kind==='pace'?'recorded pace in minutes per mile':'late-run effort drift in percent'}; markers show session medians and bars show the observed range">${[min,middle,max].map(v=>`<line class="grid-line" x1="${x(v)}" x2="${x(v)}" y1="18" y2="85"/><text x="${x(v)}" y="103" text-anchor="middle">${kind==='pace'?pace(v).replace('/mi',''):value(v)}</text>`).join('')}${periods.map(([label,p],i)=>Number.isFinite(p.median)?`<text x="0" y="${34+i*34}">${label}</text>${Number.isFinite(p.low)&&Number.isFinite(p.high)?`<line x1="${x(p.low)}" x2="${x(p.high)}" y1="${30+i*34}" y2="${30+i*34}" stroke="${i?'#78d8b8':'#91bcec'}" stroke-width="5" stroke-linecap="round"/>`:''}<circle cx="${x(p.median)}" cy="${30+i*34}" r="5" fill="${i?'#78d8b8':'#91bcec'}"><title>${label}: ${value(p.median)}; observed range ${value(p.low)}–${value(p.high)}</title></circle>`:'').join('')}<text x="${left}" y="127">${kind==='pace'?'Faster ← · min/mi':'Less drift ←'}</text></svg>`:'' )}${change?`<p class="observed-change"><span>Recorded difference${c.status!=='matched_observations'?' · limited comparison':''}</span><strong>${change}</strong></p>`:''}`;
}

// Join only the backend-selected cohort. No new grouping, fitted trend or
// condition matching is introduced by this visualisation.
export function comparisonHistory(c,rows,start,end,date,{kind='pace',domain}={}) {
  const selected=new Map([...(c.earlier.activity_ids??[]).map(id=>[id,'earlier']),...(c.recent.activity_ids??[]).map(id=>[id,'recent'])]);
  const points=rows.filter(r=>selected.has(r.activity_id)&&Number.isFinite(r.value)&&Number.isFinite(+new Date(r.start_time)));
  if(!points.length||!Number.isFinite(+new Date(start))||!Number.isFinite(+new Date(end))) return '';
  const values=points.map(r=>r.value);
  const lo=domain?.[0]??Math.min(...values),hi=domain?.[1]??Math.max(...values);
  const pad=Math.max(kind==='pace'?.15:1,(hi-lo)*.15),unit=kind==='pace'?.25:5;
  const min=Math.floor((lo-pad)/unit)*unit,max=Math.ceil((hi+pad)/unit)*unit;
  const left=64,right=400,top=32,bottom=172,W=430,H=210;
  const beginning=+new Date(start),ending=+new Date(end),cutoff=ending-(c.recent_days??42)*86400000;
  const x=t=>left+(t-beginning)/Math.max(1,ending-beginning)*(right-left);
  const y=v=>top+(v-min)/(max-min)*(bottom-top); // smaller pace/drift is higher
  const fmt=v=>kind==='pace'?pace(v).replace('/mi',''):`${number(v)}%`;
  const tickStep=Math.max(unit,Math.ceil((max-min)/4/unit)*unit);
  const ticks=Array.from({length:Math.floor((max-min)/tickStep)+1},(_,i)=>min+i*tickStep);
  const medians=['earlier','recent'].map(period=>{
    const p=c[period],begin=period==='earlier'?beginning:cutoff,finish=period==='earlier'?cutoff:ending;
    // A single observation is not a period-level trend.
    return p.count>=2&&Number.isFinite(p.median)?`<line class="period-median ${period}" x1="${x(begin)}" x2="${x(finish)}" y1="${y(p.median)}" y2="${y(p.median)}" stroke="${period==='recent'?colors.easy:colors.long}" stroke-width="2" stroke-dasharray="6 4"><title>${period} median: ${fmt(p.median)} · ${p.count} sessions</title></line>`:'';
  }).join('');
  return `<svg class="comparison-history" viewBox="0 0 ${W} ${H}" role="img" aria-label="Six-month ${kind==='pace'?'work pace':'HR drift'} history for the selected comparison; dots are sessions, dashed lines are period medians"><rect x="${x(cutoff)}" y="${top}" width="${right-x(cutoff)}" height="${bottom-top}" fill="${colors.easy}" opacity=".07"/><text x="${left}" y="16">${kind==='pace'?'Faster ↑ · min/mi':'Less drift ↑'}</text><text x="${right}" y="16" text-anchor="end">Recent</text>${ticks.map(v=>`<line class="grid-line" x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-7}" y="${y(v)+4}" text-anchor="end">${fmt(v)}</text>`).join('')}${kind==='drift'&&min<0&&max>0?`<line class="zero-drift" x1="${left}" x2="${right}" y1="${y(0)}" y2="${y(0)}"/>`:''}${medians}${points.map(r=>`<a href="#run/${r.activity_id}" aria-label="${date(r.start_time)}: ${fmt(r.value)}"><circle cx="${x(+new Date(r.start_time))}" cy="${y(r.value)}" r="4.5" fill="${selected.get(r.activity_id)==='recent'?colors.easy:colors.long}"><title>${date(r.start_time)}: ${fmt(r.value)}</title></circle></a>`).join('')}<text x="${left}" y="198">${date(start)}</text><text x="${x((beginning+ending)/2)}" y="198" text-anchor="middle">${date(new Date((beginning+ending)/2).toISOString())}</text><text x="${right}" y="198" text-anchor="end">${date(end)}</text></svg>`;
}

function repeatPace(q) {
  if(Number.isFinite(q.interval?.mean_work_pace_min_mile)) return q.interval.mean_work_pace_min_mile;
  const work=(q.interval?.repetitions??[]).filter(r=>r.kind==='work'&&Number.isFinite(r.distance_miles)&&r.distance_miles>0);
  const miles=work.reduce((sum,r)=>sum+r.distance_miles,0);
  return miles>0?work.reduce((sum,r)=>sum+r.duration_seconds,0)/60/miles:null;
}

function evidenceCaption(c,kind) {
  if(c.status==='insufficient_history') return 'Not enough matching history to compare yet.';
  const same=Math.abs(c.delta)<(kind==='pace'?1/60:.1);
  const direction=c.delta<0?(kind==='pace'?'faster':'less drift'):(kind==='pace'?'slower':'more drift');
  const finding=same?`Similar recorded ${kind==='pace'?'pace':'drift'}`:`Recorded ${direction}`;
  if(c.status==='matched_observations') return `${finding} in similar conditions. Not proof of a fitness change.`;
  return 'Small sample or different conditions. No fitness-change claim.';
}

export function qualityProgress(d,date) {
  const data=d.insights.quality_comparisons;
  if(!data) return '<p class="chart-empty">Quality comparisons are loading. Refresh after the server update.</p>';
  const primary=data.repeats[0],s=data.sustained;
  const other=data.repeats.slice(1).reduce((sum,g)=>sum+g.total_sessions,0);
  const history=(rows)=>({rows,start:d.insights.start,end:d.as_of});
  const q=d.quality??[];
  const sparse=s.comparison.status==='insufficient_history';
  return `<div class="matched-grid quality-grid ${sparse?'single-comparison':''}"><section class="matched-insight"><p class="eyebrow">Repeat speed</p><h3>${primary?escapeHtml(primary.label):'No repeat history yet'}</h3>${primary?`<p class="comparison-context">Similar reps and recoveries · work pace</p>${periodComparison(primary.comparison,date,{history:history(q.map(r=>({...r,value:repeatPace(r)})))})}<p class="comparison-verdict">${evidenceCaption(primary.comparison,'pace')}</p>`:'<p class="chart-empty">No recorded repetitions to compare yet.</p>'}</section>${!sparse?`<section class="matched-insight"><p class="eyebrow">Sustained faster running</p><h3>Around ${number(s.reference_minutes,0)} minutes at ${number(s.reference_hr,0)} bpm</h3>${periodComparison(s.comparison,date,{history:history(q.map(r=>({...r,value:r.continuous?.pace_min_mile})))})}<p class="comparison-verdict">${evidenceCaption(s.comparison,'pace')}</p></section>`:''}</div>${sparse?`<div class="comparison-pending"><div><span class="eyebrow">Sustained faster running</span><h3>No comparable trend yet</h3></div><p>${s.total_sessions} recorded sustained ${s.total_sessions===1?'session':'sessions'} · not enough similar efforts to compare pace.</p></div>`:''}<p class="chart-note">Dots = sessions · dashed lines = period medians. One session does not establish a trend.</p><details><summary>Comparison method and coverage</summary><p>Recent = last six weeks; earlier = the rest of the six-month history. Progression finishes and steady efforts are kept separate. Different durations or heart rates aren’t evidence of getting faster or slower.</p><p>Repeat pace reuses the detected mean work pace shown in Run Analysis. If that measurement is missing, total work time / total work distance is used. Each session contributes once. Repeats are grouped within 10% of distance, one repetition, and 25% of recovery duration and recovery pace. ${other} other repeat sessions have different structures and aren’t pooled into this comparison.</p><p>Sustained comparisons require the same detected structure, duration within 20% and average HR within 5 bpm. There is no extrapolated pace-at-HR score. Conditions are screened within 10°F temperature, 10°F dew point and 0.5 percentage points of average absolute grade. Missing conditions do not pass the screen. At least two sessions per period must pass before the comparison is called condition-matched. Wind, route differences, intent, fatigue and effort reports are not controlled.</p><p>These are provisional matching rules, not validated physiological thresholds. Only recognized, normal-health quality sessions are included; unrecognized work is not counted as zero. If session-level history is unavailable, bars show the recorded range, not statistical confidence.</p></details>`;
}

export function durabilityProgress(d,date) {
  const profiles=d.insights.endurance_comparisons;
  if(!profiles?.length) return '<p class="chart-empty">No comparable steady-running history yet.</p>';
  const selectedIds=new Set(profiles.flatMap(p=>[...(p.comparison.earlier.activity_ids??[]),...(p.comparison.recent.activity_ids??[])]));
  const rows=(d.durability??[]).map(r=>({...r,value:r.decoupling_percent}));
  const values=rows.filter(r=>selectedIds.has(r.activity_id)&&Number.isFinite(r.value)).map(r=>r.value);
  const domain=values.length?[Math.min(0,...values),Math.max(0,...values)]:undefined;
  return `<p class="comparison-context">How well your pace holds up as a run gets longer. Less heart-rate drift is higher on the chart.</p><div class="matched-grid endurance-grid">${profiles.map(p=>`<section class="matched-insight"><h3>${escapeHtml(p.label)}</h3><p class="comparison-context">Easy running · starts around ${number(p.reference_hr,0)} bpm</p>${periodComparison(p.comparison,date,{kind:'drift',history:{rows,start:d.insights.start,end:d.as_of},domain})}<p class="comparison-verdict">${evidenceCaption(p.comparison,'drift')}</p></section>`).join('')}</div><p class="chart-note">Same scale in both charts · dots = sessions · dashed lines = period medians.</p><details><summary>Comparison method and coverage</summary><p>Recent = last six weeks; earlier = the rest of six months. Sessions must be within 20% of the profile’s duration and 5 bpm of early-half average HR, after the five-minute warm-up exclusion. Duration bands choose profiles but do not exclude runs just across a band boundary. Overlapping duration profiles are not presented as separate evidence.</p><p>Condition-matched comparisons also screen temperature, dew point and average absolute grade, with at least two sessions per period. Otherwise the visible comparison is raw and explicitly limited. Bars show the observed session range, not confidence intervals. Negative drift means efficiency increased, not a recovery score. Clicking through individual runs remains available below.</p><p>The existing stop, finishing-pace and paired-HR checks are retained. Within-run hills, wind, pacing, health and sleep can still affect this measure. It is not a prediction of recovery, a breakdown-duration estimate, or proof of adaptation.</p></details>`;
}
