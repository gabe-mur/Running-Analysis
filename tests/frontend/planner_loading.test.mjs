import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { animateCalendar,beginPlannerLoading,calendarSketch,plannerLoadingMarkup } from '../../src/run_analysis/web/static/app/planner-loading.js';

function harness(fetcher) {
  const classes=()=>({add(){},toggle(){}});
  const nodes=new Map(['[data-planner-stage]','[data-planner-count]','[data-planner-elapsed]',
    '[data-planner-percent]','[data-planner-budget]'].map(key=>[key,{textContent:''}]));
  const bar={attributes:{},classList:classes(),setAttribute(k,v){this.attributes[k]=v;},querySelector:()=>fill};
  const fill={style:{}};nodes.set('[role="progressbar"]',bar);
  const root={innerHTML:'',isConnected:true,querySelector:k=>nodes.get(k),querySelectorAll:()=>[]};
  const host={innerHTML:'',querySelector:()=>root};
  let queued,signal,clearCalls=0;
  const progress=beginPlannerLoading(host,{requestId:'request-123',now:()=>15000,
    fetcher:(path,options)=>{signal=options.signal;return fetcher(path,options);},
    schedule:fn=>{queued=fn;return 1;},cancel:()=>clearCalls++});
  return {host,root,nodes,bar,fill,progress,poll:()=>queued(),signal:()=>signal,clears:()=>clearCalls};
}

test('rapid calendar keeps 21 cells, real progress and a simplified legend',()=>{
  const html=plannerLoadingMarkup('Plan <demo>');
  assert.match(html,/Plan &lt;demo&gt;/);
  assert.doesNotMatch(html,/illustration|planner-math|Training load|Recovery|Progression/);
  assert.equal((html.match(/data-sketch-day=/g)||[]).length,21);
  assert.match(html,/role="status"/);
  assert.match(html,/role="progressbar"/);
  assert.doesNotMatch(html,/aria-valuenow|optimal|calculated score/);
  const css=readFileSync(new URL('../../src/run_analysis/web/static/styles.css',import.meta.url),'utf8');
  assert.match(css,/@media\(prefers-reduced-motion:reduce\)/);
  assert.doesNotMatch(css,/planner-calendars|planner-math/);
});

test('thousands of deterministic sketches retain sensible key spacing and rest',()=>{
  const unique=new Set();
  for(let i=0;i<10000;i++) {
    const days=calendarSketch(i);unique.add(days.join(''));
    assert.equal(days.length,21);
    assert.equal(days.filter(d=>d==='L').length,3);
    assert.equal(days.filter(d=>d==='Q').length,3);
    assert.ok(days.includes('.'));
    assert.ok(days.filter(d=>d==='E').length>=4);
    let streak=0;
    for(let day=0;day<21;day++) {
      const role=days[day];assert.ok(['.','E','L','Q'].includes(role));
      streak=role==='.'?0:streak+1;assert.ok(streak<=3);
      if(['L','Q'].includes(role)&&day<20)assert.ok(!['L','Q'].includes(days[day+1]));
    }
  }
  assert.ok(unique.size>9000,`${unique.size} distinct calendars`);
  assert.deepEqual(calendarSketch(100),calendarSketch(100));
});

test('calendar changes instantly at a capped frame rate and cleans up',()=>{
  const cells=Array.from({length:21},()=>({className:'',textContent:''}));
  const root={isConnected:true,querySelectorAll:()=>cells};
  let pending,id=0,canceled=[],changed,removed=false;
  const motion={matches:false,addEventListener:(event,fn)=>{changed=fn;},removeEventListener:()=>{removed=true;}};
  const stop=animateCalendar(root,{requestFrame:fn=>{pending=fn;return ++id;},cancelFrame:n=>canceled.push(n),motion});
  pending(0);const first=cells.map(c=>c.textContent).join('');
  pending(20);assert.equal(cells.map(c=>c.textContent).join(''),first);
  pending(40);assert.equal(cells.map(c=>c.textContent).join(''),first);
  pending(119);assert.equal(cells.map(c=>c.textContent).join(''),first);
  pending(120);assert.notEqual(cells.map(c=>c.textContent).join(''),first);
  motion.matches=true;changed();assert.ok(canceled.length>0);
  stop();assert.equal(removed,true);
  const before=id;pending(240);assert.equal(id,before);
});

test('reduced motion leaves the calendar static without starting a frame loop',()=>{
  let calls=0;
  const stop=animateCalendar({querySelectorAll:()=>Array(21).fill({})},
    {requestFrame:()=>calls++,cancelFrame:()=>{},motion:{matches:true}});
  assert.equal(calls,0);stop();
});

test('actual budget updates estimated bar, model checks and elapsed time',async()=>{
  let state={state:'running',phase:'search',phases:['waiting','history','weather','search'],
    model_checks:2500,budget_used:2600,budget_total:10000,estimated_percent:29.32,elapsed_seconds:42.7};
  const h=harness(async(path,options)=>{assert.equal(path,'/api/weekly-schedule/progress/request-123');return state;});
  await h.poll();
  assert.equal(h.nodes.get('[data-planner-stage]').textContent,'Comparing 21-day schedules');
  assert.equal(h.nodes.get('[data-planner-percent]').textContent,'29% estimated');
  assert.equal(h.nodes.get('[data-planner-budget]').textContent,'2,600 / 10,000 main-search checks');
  assert.equal(h.nodes.get('[data-planner-count]').textContent,'2,500 full model checks');
  assert.equal(h.nodes.get('[data-planner-elapsed]').textContent,'42s elapsed');
  assert.equal(h.bar.attributes['aria-valuenow'],'29');assert.equal(h.fill.style.width,'29%');
  state={...state,phase:'refine',estimated_percent:92};await h.poll();
  assert.equal(h.nodes.get('[data-planner-stage]').textContent,'Refining distances and alternatives');
  assert.equal(h.fill.style.width,'92%');
  state={...state,state:'running',estimated_percent:100};await h.poll();
  assert.equal(h.fill.style.width,'99%');
  state={...state,state:'complete'};await h.poll();
  assert.equal(h.fill.style.width,'100%');
  h.progress.stop();assert.equal(h.signal().aborted,true);
});

test('unknown/failed polling never fabricates percentages; stops when view leaves',async()=>{
  let fail=false;
  const h=harness(async()=>{if(fail) throw new Error('network');return {state:'unknown'};});
  await h.poll();assert.equal(h.bar.attributes['aria-valuenow'],undefined);
  assert.match(h.host.innerHTML,/loading-line/);
  assert.equal(h.root.innerHTML,'');
  fail=true;await h.poll();
  assert.equal(h.root.innerHTML,'');
  h.root.isConnected=false;await h.poll();
  assert.equal(h.progress.isActive(),false);assert.equal(h.signal().aborted,true);
  assert.ok(h.clears()>0);
});

test('cached reads retain the old simple loader, even when slow or just completed',async()=>{
  let state={state:'running',phase:'history',phases:['waiting','history'],
    model_checks:0,budget_total:null,estimated_percent:3,elapsed_seconds:95};
  const h=harness(async()=>state);
  assert.match(h.host.innerHTML,/Local analysis/);
  assert.match(h.host.innerHTML,/loading-line/);
  assert.doesNotMatch(h.host.innerHTML,/planner-search-sketch|progressbar|main-search checks/);
  for(const phase of ['history','weather','publish']) {
    state={...state,phase,phases:['waiting','history','weather','publish']};
    await h.poll();assert.equal(h.root.innerHTML,'');
  }
  state={...state,state:'complete',estimated_percent:100};
  await h.poll();assert.equal(h.root.innerHTML,'');
});

test('only confirmed running search promotes to rich progress, not completed work',async()=>{
  let state={state:'complete',phase:'publish',phases:['search','publish'],budget_total:100};
  const h=harness(async()=>state);
  await h.poll();assert.equal(h.root.innerHTML,'');
  state={state:'running',phase:'search',phases:['search'],estimated_percent:8};
  await h.poll();assert.match(h.root.innerHTML,/planner-search-sketch/);
  assert.equal(h.nodes.get('[data-planner-percent]').textContent,'8% estimated');
  state={...state,phase:'refine',estimated_percent:92};
  await h.poll();assert.equal(h.nodes.get('[data-planner-percent]').textContent,'92% estimated');
});

test('polling failures after real search starts keep the rich view with honest status',async()=>{
  let fail=false;
  const h=harness(async()=>{if(fail)throw new Error('network');return {state:'running',phase:'search',estimated_percent:25};});
  await h.poll();fail=true;await h.poll();
  assert.match(h.root.innerHTML,/planner-search-sketch/);
  assert.equal(h.nodes.get('[data-planner-count]').textContent,'Live progress unavailable');
  assert.equal(h.nodes.get('[data-planner-percent]').textContent,'25% estimated');
});

test('refresh waiting behind a rebuild displays the active job budget, not zero',async()=>{
  const h=harness(async()=>({state:'running',phase:'search',following_active_job:true,
    model_checks:5000,budget_used:5000,budget_total:10000,estimated_percent:49,elapsed_seconds:90}));
  await h.poll();
  assert.equal(h.nodes.get('[data-planner-stage]').textContent,'Plan rebuild already running · Comparing 21-day schedules');
  assert.equal(h.nodes.get('[data-planner-percent]').textContent,'49% estimated');
  assert.equal(h.nodes.get('[data-planner-budget]').textContent,'5,000 / 10,000 main-search checks');
  assert.equal(h.nodes.get('[data-planner-elapsed]').textContent,'90s elapsed');
});
