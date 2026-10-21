import {test} from 'node:test';
import assert from 'node:assert/strict';
import { selectFitnessWindow } from '../../src/run_analysis/web/static/app/fitness-window.js';

const page={innerHTML:'',querySelectorAll:()=>[]};
globalThis.document={querySelector:()=>page};
const {renderDashboard}=await import('../../src/run_analysis/web/static/app/views/dashboard.js');
const sample=value=>({value,low:value,high:value,complete:true});
const at='2026-10-10T20:00:00Z';
const current={as_of:at,status:'above',label:'More load than usual',detail:'Recent training load is 20% higher than your six-week training trend.'};
const overview={as_of:at,timezone:'UTC',current_status:current,status_history:[current],activities:[],
  load_guidance:{ratio_reference:{low:.8,high:1.3},current:{status:'within',label:'Within recommended load',action:'Continue your planned training.',detail:'Your cardiovascular load is within the recommended range.',recommended_range:{low:24,high:39}},
    daily:[{as_of:at,status:'within',label:'Within recommended load',action:'Continue your planned training.',recommended_range:{low:24,high:39}}]},
  load_display:{ratio:1.2,current:{recent:sample(36)},daily:[{as_of:at,recent:sample(36),baseline:sample(30)}]},
  training_load:{ratio:1.2,daily:[{as_of:at,recent:sample(36),baseline:sample(30),short:sample(40)}]}};
const horizon={trend:'uncertain',confidence:'low',current_pace:null};
const data={progress:{current_pace:null,current_load:{capacity_reference_miles:17.6,trailing_28d:{distance_miles:70},trailing_7d:{distance_miles:18,zone_load:100,moving_minutes:180}},consistency:{runs_per_week:4,running_days:16,longest_run_miles:9}},
  fitness_interpretation:{short_term:horizon,long_term:horizon,signals:[],capacity_summary:'Recorded history',
    headline:'Aerobic efficiency is likely improving.',summary:'The pace-at-heart-rate evidence favors improvement.'},
  last_run:null,recommendation:{title:'Easy run',distance_target_miles:3},weekly_schedule:null,
  training_status:{status:'building',label:'Building',detail:'Near demonstrated capacity with recent quality and increasing capacity.',rule_trace:[
    {rule_id:'status_strained',fired:false,facts:{continuous_fatigue_to_capacity_ratio:1.01,high_load_ratio:1.3}},
    {rule_id:'status_building',fired:true,facts:{quality_sessions_14d:2,capacity_reference_miles:17.6,previous_capacity_reference_miles:16.4,improvement_supported:true,fitness_trend:'uncertain'}},
  ]}};
let unavailable=false;
const requested=[];
globalThis.fetch=async url=>{
  requested.push(url);
  if(url==='/api/training-overview'&&unavailable)throw new Error('Offline');
  return {ok:true,json:async()=>url.startsWith('/api/dashboard')?data:url==='/api/settings'?{default_fitness_window:28}:overview};
};

test('Dashboard restores status-first layout and gives load its own split panel',async()=>{
  await renderDashboard();
  assert.match(page.innerHTML,/<h1 class="training-status good">Building/);
  assert.match(page.innerHTML,/Why this status/);
  assert.match(page.innerHTML,/Fitness has more than one dimension/);
  assert.doesNotMatch(page.innerHTML,/Recorded history/);
  assert.match(page.innerHTML,/<p class="lede status-detail">Maintaining aerobic efficiency · Sustained volume growing<\/p>/);
  assert.doesNotMatch(page.innerHTML,/Near demonstrated capacity with recent quality and increasing capacity|Aerobic efficiency is likely improving/);
  assert.doesNotMatch(page.innerHTML,/The pace-at-heart-rate evidence favors improvement/);
  assert.match(page.innerHTML,/<h3 class="training-status good">Within recommended load/);
  assert.match(page.innerHTML,/Continue your planned training/);
  assert.match(page.innerHTML,/load-recommended-zone/);
  assert.doesNotMatch(page.innerHTML,/No elevated strain flag|More load than usual|Unclear comparison/);
  assert.match(page.innerHTML,/<h2 id="dashboard-load-heading">Training load<\/h2>/);
  assert.match(page.innerHTML,/dashboard-load-layout/);
  assert.match(page.innerHTML,/class="load-interpretation"/);
  assert.match(page.innerHTML,/class="load-graph"/);
  assert.match(page.innerHTML,/class="load-chart-pane"/);
  assert.doesNotMatch(page.innerHTML,/training-direction|Training direction/);
  assert.equal((page.innerHTML.match(/<h1 /g)??[]).length,1);
  assert.ok(page.innerHTML.indexOf('dashboard-grid')<page.innerHTML.indexOf('dashboard-training-load'));
  assert.match(page.innerHTML,/data-load-status="within" style="stroke:#78d8b8"/);
  assert.match(page.innerHTML,/class="fatigue-run load-run " style="fill:#78d8b8"/);
  assert.doesNotMatch(page.innerHTML,/training-status-cell/);
  assert.match(page.innerHTML,/Your next seven days/);
  assert.doesNotMatch(page.innerHTML,/Old conflicting heuristic|Rules are checked in order|NaN|undefined/);
});

test('background refresh replaces stale plan data without showing a loading page',async()=>{
  const saved={schedule:data.weekly_schedule,pending:data.weekly_plan_needs_refresh};
  try {
    data.weekly_plan_needs_refresh=true;await renderDashboard();
    assert.match(page.innerHTML,/Plan needs updating/);
    data.weekly_plan_needs_refresh=false;
    data.weekly_schedule={run_count:4,visible_7d_scheduled_miles:21.2,days:[],projected_distance_range_miles:[21.2,21.2]};
    const refresh=renderDashboard({showLoading:false});
    assert.match(page.innerHTML,/Plan needs updating/); // Keep current view until replacement arrives.
    await refresh;
    assert.match(page.innerHTML,/4 runs · 21\.2 mi this week/);
    assert.doesNotMatch(page.innerHTML,/Plan needs updating|loading-line/);
  } finally {data.weekly_schedule=saved.schedule;data.weekly_plan_needs_refresh=saved.pending;}
});

test('late Dashboard read cannot overwrite another page or its draft',async()=>{
  const priorFetch=globalThis.fetch,priorLocation=globalThis.location;
  globalThis.location={hash:'#dashboard'};
  let finish;
  globalThis.fetch=async url=>url.startsWith('/api/dashboard')
    ? new Promise(resolve=>{finish=resolve;}) : priorFetch(url);
  try {
    const request=renderDashboard({showLoading:false});
    await new Promise(resolve=>setImmediate(resolve));
    globalThis.location.hash='#settings';page.innerHTML='Settings with unsaved edits';
    finish({ok:true,json:async()=>data});await request;
    assert.equal(page.innerHTML,'Settings with unsaved edits');
  } finally {globalThis.fetch=priorFetch;globalThis.location=priorLocation;}
});

test('Dashboard uses the selected Progress window and never substitutes long-term aerobic improvement',async()=>{
  const saved={short:data.fitness_interpretation.short_term,long:data.fitness_interpretation.long_term,signals:data.fitness_interpretation.signals};
  try {
    selectFitnessWindow(42);
    data.fitness_interpretation.short_term={...horizon,window_days:42,current_pace:{display:'10:00/mi'},
      period_change:{direction:'improving',evidence:'clear',pace_change_seconds_per_mile:-43},
      within_window_trend:{direction:'declining',evidence:'clear',pace_change_seconds_per_mile:46}};
    data.fitness_interpretation.long_term={...horizon,trend:'improving',current_pace:{display:'9:00/mi'}};
    data.fitness_interpretation.signals=[{label:'Aerobic efficiency',trend:'improving',status:'Improved',confidence:'low',detail:'Selected window'}];
    await renderDashboard();
    assert.ok(requested.includes('/api/dashboard?fitness_window_days=42'));
    assert.match(page.innerHTML,/<h2>Improving<\/h2>/);
    assert.match(page.innerHTML,/Aerobic efficiency:.*Improving/);
    assert.match(page.innerHTML,/43 sec\/mi faster/);
    assert.match(page.innerHTML,/Recent pace is slowing\./);
    assert.match(page.innerHTML,/10:00\/mi at the same HR/);
    assert.doesNotMatch(page.innerHTML,/9:00\/mi at the same HR|46 sec\/mi slower/);
    assert.match(page.innerHTML,/<h1 class="training-status good">Building/);
  } finally {
    selectFitnessWindow(28);
    Object.assign(data.fitness_interpretation,{short_term:saved.short,long_term:saved.long,signals:saved.signals});
  }
});

test('load chart failure does not hide the independently calculated training status',async()=>{
  unavailable=true;
  try {
    await renderDashboard();
    assert.match(page.innerHTML,/<h1 class="training-status good">Building/);
    assert.match(page.innerHTML,/How am I doing\?/);
    assert.match(page.innerHTML,/Your next seven days/);
    assert.match(page.innerHTML,/Load history could not be loaded/);
    assert.match(page.innerHTML,/Load guidance unavailable/);
    assert.doesNotMatch(page.innerHTML,/Old conflicting heuristic|training-status-cell|NaN|undefined/);
  }finally{unavailable=false;}
});

test('Dashboard uses committed mileage and collapses identical displayed range ends',async()=>{
  const saved=data.weekly_schedule;
  try {
    data.weekly_schedule={run_count:4,days:[],projected_distance_range_miles:[20.2,20.2]};
    await renderDashboard();
    assert.match(page.innerHTML,/<h2>4 runs · 20\.2 mi this week<\/h2>/);
    assert.doesNotMatch(page.innerHTML,/20\.2–20\.2/);
    data.weekly_schedule.visible_7d_scheduled_miles=20.4;
    await renderDashboard();
    assert.match(page.innerHTML,/<h2>4 runs · 20\.4 mi this week<\/h2>/);
    delete data.weekly_schedule.visible_7d_scheduled_miles;
    data.weekly_schedule.projected_distance_range_miles=[20,21];
    await renderDashboard();
    assert.match(page.innerHTML,/<h2>4 runs · 20\.0–21\.0 mi this week<\/h2>/);
  } finally {data.weekly_schedule=saved;}
});

test('adverse check-in remains visible without overriding or coloring load status',async()=>{
  overview.current_check_in={status:'pain_or_injury_concern',reported_at:at};
  try {
    await renderDashboard();
    assert.match(page.innerHTML,/Recent check-in/);
    assert.match(page.innerHTML,/You reported pain or an injury concern/);
    assert.match(page.innerHTML,/<h1 class="training-status good">Building/);
    assert.match(page.innerHTML,/separate from the load comparison/);
  } finally {delete overview.current_check_in;}
});

test('pending plan shows an update link instead of stale mileage, workout or quality count',async()=>{
  const saved=data.weekly_schedule;
  try {
    data.weekly_plan_needs_refresh=true;
    data.weekly_schedule={run_count:4,visible_7d_scheduled_miles:20.4,
      days:[{date:'2026-10-12',recommendation:{title:'Obsolete threshold run',workout_type:'tempo_threshold'}}]};
    await renderDashboard();
    assert.match(page.innerHTML,/<h2>Plan needs updating<\/h2>/);
    assert.match(page.innerHTML,/Open Weekly Plan to update your next run/);
    assert.match(page.innerHTML,/<b>—<\/b><span>quality sessions<\/span>/);
    assert.doesNotMatch(page.innerHTML,/Obsolete threshold run|4 runs · 20\.4/);
  } finally {
    data.weekly_schedule=saved;
    delete data.weekly_plan_needs_refresh;
  }
});

test('Dashboard renders concise status reasons without old copy or quality attendance',async()=>{
  const saved=data.training_status;
  try {
    for(const [status,rule,facts,reason] of [
      ['building','status_building',{capacity_reference_miles:20,previous_capacity_reference_miles:18},'Sustained volume growing'],
      ['maintaining','status_building',{sustained_to_capacity_ratio:1,near_capacity_ratio:.85,quality_sessions_14d:0},'Volume near your established level'],
      ['rebuilding','status_rebuilding',{},'Load rising after reduced volume'],
      ['underloaded','status_underloaded',{},'Sustained volume below your established level'],
      ['recovering','status_recovering',{current_health_status:'sick_or_recovering'},'Illness reported'],
      ['strained','status_strained',{continuous_fatigue_to_capacity_ratio:1.4,high_load_ratio:1.3},'High accumulated load'],
      ['insufficient_data','status_insufficient_evidence',{},'More recent training data needed'],
    ]) {
      data.training_status={status,label:status,detail:'Old verbose status explanation',rule_trace:[{rule_id:rule,fired:true,facts}]};
      await renderDashboard();
      assert.ok(page.innerHTML.includes(`<p class="lede status-detail">Maintaining aerobic efficiency · ${reason}</p>`));
      assert.ok(page.innerHTML.includes(`>${status}</h1>`));
      assert.doesNotMatch(page.innerHTML,/Old verbose status explanation|No quality sessions|quality sessions in 14 days/);
    }
  } finally {data.training_status=saved;}
});
