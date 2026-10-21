import { api } from './api.js';
import { escapeHtml } from './format.js';

const phases={waiting:'Waiting for the planner',history:'Reading your running history',
  weather:'Checking weather and availability',search:'Comparing 21-day schedules',
  refine:'Refining distances and alternatives',publish:'Preparing your plan'};
const stages=[['history','History'],['weather','Weather'],['search','Search'],['publish','Plan']];
// Lightweight visual variation, separate from real search telemetry. Never
// submit these calendars to the planner or count them as evaluated candidates.
export function calendarSketch(iteration=0) {
  let seed=(Math.imul(iteration+1,0x9e3779b9)^0x85ebca6b)>>>0;
  const random=n=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)%n;};
  const days=Array(21).fill('.'),offset=random(7);
  for(let week=0;week<3;week++) {
    days[(offset+week*7)%21]='L';
    days[(offset+week*7+3+random(2))%21]='Q';
  }
  const empty=days.map((role,i)=>role==='.'?i:-1).filter(i=>i>=0);
  for(let i=empty.length-1;i>0;i--) {const j=random(i+1);[empty[i],empty[j]]=[empty[j],empty[i]];}
  let remaining=4+random(6);
  for(const day of empty) {
    if(!remaining) break;
    days[day]='E';
    let streak=0,tooMany=false;
    for(const role of days) {streak=role==='.'?0:streak+1;if(streak>3)tooMany=true;}
    if(tooMany)days[day]='.';else remaining--;
  }
  return days;
}

export function animateCalendar(root,{requestFrame=globalThis.requestAnimationFrame?.bind(globalThis),
  cancelFrame=globalThis.cancelAnimationFrame?.bind(globalThis),
  motion=globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')}={}) {
  const cells=[...root.querySelectorAll('[data-sketch-day]')];
  if(cells.length!==21||!requestFrame)return ()=>{};
  let frame,stopped=false,last=null,iteration=0;
  const paint=()=>calendarSketch(iteration++).forEach((role,i)=>{
    const cell=cells[i],style=`sketch-${role==='.'?'rest':role}`;
    if(cell.className!==style) {cell.className=style;cell.textContent=role==='.'?'·':role;}
  });
  const tick=stamp=>{
    if(stopped||root.isConnected===false)return;
    if(last===null||stamp-last>=120) {paint();last=stamp;}
    frame=requestFrame(tick);
  };
  const change=()=>{
    cancelFrame?.(frame);last=null;
    if(!stopped&&!motion?.matches)frame=requestFrame(tick);
  };
  motion?.addEventListener?.('change',change);
  change();
  return ()=>{stopped=true;cancelFrame?.(frame);motion?.removeEventListener?.('change',change);};
}

export function plannerLoadingMarkup(title='Loading your weekly plan') {
  return `<article class="planner-loader wide-card" data-planner-search aria-busy="true">
    <p class="eyebrow">Weekly planner</p><h2>${escapeHtml(title)}</h2>
    <p class="planner-stage" role="status" data-planner-stage>Contacting the planner…</p>
    <div class="planner-search-sketch" aria-hidden="true"><div class="planner-sketch-frame">${calendarSketch().map((role,i)=>`<i data-sketch-day="${i}" class="sketch-${role==='.'?'rest':role}">${role==='.'?'·':role}</i>`).join('')}</div></div>
    <div class="planner-sketch-key">E Easy · L Long · Q Quality · · Rest</div>
    <div class="planner-progress-track" role="progressbar" aria-label="Estimated planner progress" aria-valuemin="0" aria-valuemax="100" aria-valuetext="Waiting for live progress"><span></span></div>
    <div class="planner-percent"><strong data-planner-percent>Starting…</strong><span data-planner-budget>Based on search allowance—not a time estimate</span></div>
    <div class="planner-steps">${stages.map(([key,label])=>`<span data-planner-step="${key}">${label}</span>`).join('')}</div>
    <div class="planner-work-count"><span data-planner-count>Waiting for live search updates</span><span data-planner-elapsed>0s elapsed</span></div>
  </article>`;
}

export function beginPlannerLoading(host,{title,requestId=globalThis.crypto.randomUUID(),fetcher=api,
  schedule=setTimeout,cancel=clearTimeout,now=()=>performance.now()}={}) {
  // Keep the familiar lightweight loader until actual search work is confirmed.
  // Elapsed time alone cannot distinguish a slow cached read from a rebuild.
  host.innerHTML=`<div data-planner-loader aria-busy="true"><section class="page narrow" role="status"><p class="eyebrow">Local analysis</p><h1>${escapeHtml(title??'Loading your weekly plan')}</h1><div class="loading-line"></div></section></div>`;
  const root=host.querySelector('[data-planner-loader]');
  const started=now(),abort=new AbortController();let stopped=false,timer,searchVisible=false,stopAnimation=()=>{};
  const active=()=>!stopped&&root?.isConnected!==false&&host.querySelector('[data-planner-loader]')===root;
  const stop=()=>{stopped=true;cancel(timer);stopAnimation();abort.abort();};
  const display=data=>{
    if(!active()) return;
    if(!searchVisible) {
      const actualSearch=data.state==='running'&&(
        ['search','refine'].includes(data.phase)||data.phases?.includes('search')
        ||data.budget_total>0||data.model_checks>0);
      if(!actualSearch) return;
      root.innerHTML=plannerLoadingMarkup(title);
      searchVisible=true;
      stopAnimation=animateCalendar(root);
    }
    if(data.state==='complete'||data.state==='failed')stopAnimation();
    const stage=root.querySelector('[data-planner-stage]');
    const count=root.querySelector('[data-planner-count]');
    const elapsed=root.querySelector('[data-planner-elapsed]');
    const bar=root.querySelector('[role="progressbar"]');
    const percentage=root.querySelector('[data-planner-percent]');
    const budget=root.querySelector('[data-planner-budget]');
    const label=data.state==='failed'?'The planner needs attention':data.state==='complete'?'Plan ready':phases[data.phase];
    stage.textContent=data.following_active_job
      ?`Plan rebuild already running · ${label??'Waiting for the planner…'}`
      :label??'Waiting for the planner…';
    bar.setAttribute('aria-valuetext',stage.textContent);
    if(Number.isFinite(data.estimated_percent)) {
      const percent=Math.max(0,Math.min(data.state==='complete'?100:99,Math.floor(data.estimated_percent)));
      percentage.textContent=`${percent}% estimated`;
      bar.setAttribute('aria-valuenow',String(percent));
      bar.classList.add('has-estimate');
      bar.querySelector('span').style.width=`${percent}%`;
    }
    budget.textContent=Number.isFinite(data.budget_total)&&data.budget_total>0
      ?`${data.budget_used.toLocaleString()} / ${data.budget_total.toLocaleString()} main-search checks`
      :'Based on search allowance—not a time estimate';
    // Work count is real completed model invocations, not a total/percentage.
    count.textContent=Number.isFinite(data.model_checks)&&data.model_checks>0
      ?`${data.model_checks.toLocaleString()} full model checks`
      :data.phase==='publish'?'Finishing your plan':data.state==='unknown'?'Waiting for live search updates':'No search checks yet';
    const seconds=Number.isFinite(data.elapsed_seconds)?data.elapsed_seconds:(now()-started)/1000;
    elapsed.textContent=`${Math.floor(seconds)}s elapsed`;
    for(const node of root.querySelectorAll('[data-planner-step]')) {
      const current=node.dataset.plannerStep===data.phase||data.phase==='refine'&&node.dataset.plannerStep==='search';
      node.classList.toggle('is-current',current&&data.state==='running');
      node.classList.toggle('is-done',data.phases?.includes(node.dataset.plannerStep)&&(!current||data.state==='complete'));
    }
  };
  const poll=async()=>{
    if(!active()) {stop();return;}
    try {display(await fetcher(`/api/weekly-schedule/progress/${requestId}`,{signal:abort.signal}));}
    catch {if(active()&&searchVisible) {
      root.querySelector('[data-planner-stage]').textContent='Still waiting for your plan…';
      root.querySelector('[role="progressbar"]').setAttribute('aria-valuetext','Live progress unavailable; still waiting for your plan');
      root.querySelector('[data-planner-count]').textContent='Live progress unavailable';
      root.querySelector('[data-planner-elapsed]').textContent=`${Math.floor((now()-started)/1000)}s elapsed`;
    }}
    if(active()) timer=schedule(poll,1000);
  };
  timer=schedule(poll,150);
  return {requestId,headers:{'X-Planner-Request':requestId},stop,isActive:active};
}
