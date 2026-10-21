import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {vo2EstimateMarkup,vo2MethodMarkup} from '../../src/run_analysis/web/static/app/progress-insights.js';
import {trainingVolumeChart,effortBalance} from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

// Exercise the actual view, not merely the utility functions it could ignore.
const page={innerHTML:'',querySelectorAll:()=>[]};
globalThis.document={querySelector:()=>page};
globalThis.location={hash:'#progress'};
const {renderArchivedProgress,renderProgress,pacePresentation}=await import('../../src/run_analysis/web/static/app/views/progress-wip.js');
const {selectFitnessWindow}=await import('../../src/run_analysis/web/static/app/fitness-window.js');
const exposure={run_count:13,running_days:13,distance_miles:69.4,moving_minutes:731.5,longest_miles:8.45};
const vo2={value_ml_kg_min:52.9,uncertainty_95_ml_kg_min:4.7,confidence:'low',method:'Shared ACSM + George',interpretation:'Two equations agree.',limitations:[]};
const effortPoint={as_of:'2026-10-06T02:00:00Z',easy_percent:80,above_easy_percent:20,known_hr_minutes:600,missing_hr_minutes:0,moderate_minutes:90,hard_minutes:30,hr_coverage:1};
const sessionPoint={as_of:effortPoint.as_of,quality_sessions:3,easy_sessions:10,classified_runs:13,unknown_runs:0,quality_percent:3/13*100};
const data={as_of:'2026-10-06T02:00:00Z',timezone:'America/New_York',window_start:'2026-09-08T02:00:00Z',
  reference_minutes:20,available_windows:[14,28,365],race_goal:{status:'no_goal'},long_goal_miles:null,
  baseline:{current_pace:{minutes_per_mile:10,display:'10:00/mi'},uncertainty_95_min_mile:.5,fitness_confidence:'low',
    activity_coverage:[],series:[],steady_aerobic:{current_pace:null},consistency:{quality_sessions:7},
    current_load:{acute_distance_to_capacity_ratio:.9,capacity_reference_miles:17.6,trailing_7d:{distance_miles:15.8}},
    period_comparison:{current:{zone_load:1416},previous:{zone_load:904},interpretation:'Recorded intensity load.'},
    intensity:{easy_percent:80,moderate_percent:15,hard_percent:5,balance:'balanced',balance_headline:'Mostly easy',balance_detail:'HR-based'}},
  performance:{current:{run_count:5},vo2_input:{pace_min_mile:9},confidence:'low',fitness_trend:'uncertain',
    vo2,vo2_series:[],rolling_28d:[],rolling_raw_28d:[],last_observation:null},
  exposure:{current:exposure,prior:{...exposure,run_count:9},roles:[],daily:[
    {as_of:'2026-09-08T02:00:00Z',distance_by_role:{easy:0,long:0,quality:0},distance_7d:0,distance_28d_weekly:0},
    {as_of:'2026-10-06T02:00:00Z',distance_by_role:{easy:0,long:0,quality:0},distance_7d:15.8,distance_28d_weekly:17.4}]},
  effort:{trailing_days:28,current:effortPoint,daily:[effortPoint],
    session_balance:{trailing_runs:20,current:sessionPoint,daily:[sessionPoint],reference:{low:20,high:25}}},
  load:{current:{covered_runs:0,total_runs:13,complete:false,cardio:null,work_minutes:null},prior:{complete:false}},
  quality:[],durability:[],insights:{activities:[],quality_comparisons:{repeats:[],sustained:{total_sessions:0,comparison:{status:'insufficient_history'}}},endurance_comparisons:[]}};
globalThis.fetch=async path=>({ok:true,json:async()=>path==='/api/settings'?{default_fitness_window:28}:data});

test('goal progression prototype appears only on WIP, leaving Progress intact',async()=>{
  const before=data.goal_journey;
  data.goal_journey={available:false,reason:'Goal progression fixture needs a Weekly Plan.'};
  try {
    globalThis.location.hash='#progress';await renderArchivedProgress();
    assert.doesNotMatch(page.innerHTML,/Goal progression fixture/);
    globalThis.location.hash='#progress';await renderProgress();
    assert.match(page.innerHTML,/Goal progression fixture needs a Weekly Plan/);
  } finally {data.goal_journey=before;globalThis.location.hash='#progress';}
});

test('manual exclusions cannot reappear through the old pace chart context fallback',()=>{
  const excluded={activity_id:999,start_time:data.as_of,trend_weight:0,context_pace_min_mile:40};
  const input={...data,baseline:{...data.baseline,series:[excluded],activity_coverage:[{activity_id:999}]},
    aerobic_performance:{hidden_activity_ids:[999],manually_excluded_activity_ids:[999]}};
  assert.deepEqual(pacePresentation(input).progress.series,[]);
  const normalized={...input,pace_comparison:{runs:[],series:[],performance:{current:null},trend:[],recorded_trend:[]}};
  const result=pacePresentation(normalized);
  assert.deepEqual(result.progress.series,[]);
  assert.equal(result.progress.activity_coverage[0].reason,'Excluded from aerobic scoring by you.');
  assert.equal(input.baseline.series.length,1); // History and source payload retained.
});

test('main Progress uses native headline, load and role counts despite unavailable new ledger',async()=>{
  await renderArchivedProgress();
  assert.match(page.innerHTML,/headline-pace[^]*?<strong>10:00\/mi<\/strong>/);
  assert.doesNotMatch(page.innerHTML,/near minute/);
  assert.doesNotMatch(page.innerHTML,/headline-pace[^]*?<strong>9:00\/mi<\/strong>/);
  assert.match(page.innerHTML,/Training load<strong><span class="trend neutral">[^]*?1416/);
  assert.match(page.innerHTML,/HR-based points · previously 904/);
  assert.match(page.innerHTML,/Separate scheduler accounting: Not enough load history/);
  assert.match(page.innerHTML,/3 of 13 runs/);
  assert.doesNotMatch(page.innerHTML,/<strong>7<\/strong>quality sessions/);
  assert.match(page.innerHTML,/<strong>52.9<\/strong>/);
  assert.match(page.innerHTML,/Shared ACSM \+ George/);
  assert.match(page.innerHTML,/chart-card[^]*?quality low/);
  assert.doesNotMatch(page.innerHTML,/NaN|undefined/);
});

test('Progress shows a period improvement and recent slowdown together without dropping either card',async()=>{
  const saved=data.performance;
  data.performance={...saved,period_change:{direction:'improving',evidence:'clear',pace_change_seconds_per_mile:-34,
    uncertainty_95_seconds_per_mile:20},within_window_trend:{direction:'declining',evidence:'likely',pace_change_seconds_per_mile:8}};
  try {
    await renderArchivedProgress();
    assert.match(page.innerHTML,/<h1>Faster than the previous 4 weeks\.<\/h1>/);
    assert.match(page.innerHTML,/34 sec\/mi faster/);
    assert.match(page.innerHTML,/Recent pace is likely slowing\./);
    assert.match(page.innerHTML,/class="wide-card vo2-card"/);
    assert.doesNotMatch(page.innerHTML,/no single improvement claim|Maintaining current pace/);
  } finally {data.performance=saved;}
});

test('missing original HR load is not filled by unrelated scheduler units',async()=>{
  data.baseline.period_comparison.current.zone_load=null;
  data.baseline.period_comparison.previous.zone_load=null;
  for(const period of ['current','prior']) data.load[period]={complete:true,cardio:{low:80,high:80},work_minutes:{low:20,high:20},accounting_key:'new',covered_runs:13,total_runs:13};
  await renderArchivedProgress();
  assert.match(page.innerHTML,/Training load<strong>[^]*?·<\/i>—<\/span>/);
  assert.match(page.innerHTML,/These are different units from HR-based points/);
  assert.doesNotMatch(page.innerHTML,/NaN|undefined/);
});

test('Progress prioritizes history without the long-run goal or preview header',async()=>{
  data.long_goal_miles=17;
  await renderArchivedProgress();
  assert.doesNotMatch(page.innerHTML,/Your long-run goal|Edit goal|goal-distance|goal-panel|Progress WIP|Preview|Current page →/);
  const vo2Position=page.innerHTML.indexOf('wide-card vo2-card');
  const volumePosition=page.innerHTML.indexOf('wide-card training-history');
  assert.ok(vo2Position>=0&&vo2Position<volumePosition);
  assert.doesNotMatch(page.innerHTML,/wide-card fatigue-card/); // Load/status moved to Dashboard.
  assert.doesNotMatch(page.innerHTML,/wip-running-summary|<span>Time running<\/span>/);
  assert.doesNotMatch(page.innerHTML,/<article><span>Last 7 days<\/span>/);
  assert.match(page.innerHTML,/volume-level"><strong>15.8 <small>mi in the last 7 days<\/small><\/strong><\/div>/);
  assert.match(page.innerHTML,/Bars = daily miles · lines = rolling volume/);
  assert.match(page.innerHTML,/Recorded daily run distances and session types/);
  assert.match(page.innerHTML,/data-run-chart/);
  assert.doesNotMatch(page.innerHTML,/class="insight-chart run-strip"/);
  assert.match(page.innerHTML,/in the last 7 days/);
  assert.match(page.innerHTML,/Usual: 17.6 mi \/ 7 days · not a target/);
  assert.doesNotMatch(page.innerHTML,/<span>Aerobic efficiency<\/span>/);
  assert.doesNotMatch(page.innerHTML,/Time at each effort|Easy vs harder running|HR exposure/);
  assert.match(page.innerHTML,/20–25% reference/);
  assert.match(page.innerHTML,/Follow your Weekly Plan rather than adding runs just to reach this range\./);
  assert.doesNotMatch(page.innerHTML,/Adding easy runs|Aim for mostly easy|Below target|Above target/);
  assert.match(page.innerHTML,/effort-reference/);
  assert.doesNotMatch(page.innerHTML,/Warm-ups and recoveries count at their actual intensity/);
  assert.doesNotMatch(page.innerHTML,/Quality-session performance|Endurance on longer runs|Does pace line up/);
  assert.doesNotMatch(page.innerHTML,/Run feedback and recorded HRV|How the runs felt|Report sources and HRV/);
  assert.doesNotMatch(page.innerHTML,/NaN|undefined/);
});

test('both Progress pages omit redundant summary cards without changing exposure evidence',async()=>{
  const before=structuredClone(data.exposure),route=globalThis.location.hash;
  try {
    for(const [hash,render] of [['#progress',renderArchivedProgress],['#progress',renderProgress]]) {
      globalThis.location.hash=hash;
      await render();
      assert.doesNotMatch(page.innerHTML,/wip-running-summary|<article><span>(Run frequency|Time running|Longest recent run)<\/span>/);
      assert.match(page.innerHTML,/wide-card vo2-card/);
      assert.match(page.innerHTML,/wide-card training-history/);
      assert.match(page.innerHTML,/wide-card coverage-card/);
      assert.deepEqual(data.exposure,before);
    }
  } finally {globalThis.location.hash=route;}
});

test('option-one arrangement and compact presentation are WIP-only, with unchanged metrics',async()=>{
  const route=globalThis.location.hash,before=structuredClone(data);
  const date=value=>new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',timeZone:data.timezone}).format(new Date(value));
  const panels=[vo2EstimateMarkup(data.performance.vo2),trainingVolumeChart(data,date),effortBalance(data,date)];
  try {
    globalThis.location.hash='#progress';await renderArchivedProgress();
    assert.doesNotMatch(page.innerHTML,/wip-window-layout|wip-supporting-layout|wip-quality-card/);
    for(const contents of panels)assert.ok(page.innerHTML.includes(contents));
    globalThis.location.hash='#progress';await renderProgress();
    assert.match(page.innerHTML,/page progress-wip wip-window-layout/);
    assert.match(page.innerHTML,/wip-supporting-layout[^]*?training-history[^]*?wip-supporting-stack[^]*?vo2-card[^]*?wip-quality-card/);
    assert.ok(page.innerHTML.indexOf('wip-supporting-layout')>page.innerHTML.indexOf('benchmark-pace-card'));
    assert.ok(page.innerHTML.indexOf('goal-journey')>page.innerHTML.indexOf('wip-quality-card'));
    const compactPanels=[vo2EstimateMarkup(data.performance.vo2,{compact:true,includeMethod:false}),vo2MethodMarkup(data.performance.vo2,{compact:true}),trainingVolumeChart(data,date,{compact:true}),effortBalance(data,date,{compact:true})];
    for(const contents of compactPanels)assert.ok(page.innerHTML.includes(contents),'WIP uses the shared renderers with compact presentation');
    assert.doesNotMatch(page.innerHTML,/class="cross-check-caveat"|History · last|Bars = daily miles|class="volume-current"/);
    assert.doesNotMatch(page.innerHTML,/Estimated range:|class="vo2-range"/);
    assert.doesNotMatch(page.innerHTML,/Ideal range:/);
    assert.match(page.innerHTML,/vo2-trend[^]*?<\/div><\/div><details><summary>Method and limitations<\/summary>/);
    assert.deepEqual(data,before);
  } finally {globalThis.location.hash=route;}
});

test('simplified Progress does not depend on the removed exploratory datasets',async()=>{
  const previous={insights:data.insights,quality:data.quality,durability:data.durability};
  delete data.insights;delete data.quality;delete data.durability;
  try {
    await renderArchivedProgress();
    assert.match(page.innerHTML,/Training volume|Quality sessions|VO₂ max/);
    assert.doesNotMatch(page.innerHTML,/Quality-session performance|Endurance on longer runs|Run feedback and recorded HRV/);
  } finally {Object.assign(data,previous);}
});

test('missing pace uncertainty does not create an invented or NaN range',async()=>{
  data.baseline.uncertainty_95_min_mile=null;
  await renderArchivedProgress();
  assert.doesNotMatch(page.innerHTML,/NaN|Infinity/);
  assert.match(page.innerHTML,/headline-pace[^]*?<strong>10:00\/mi<\/strong>/);
});

test('reviewed Progress is primary and old WIP links redirect without a duplicate navigation item',()=>{
  const router=readFileSync(new URL('../../src/run_analysis/web/static/app/router.js',import.meta.url),'utf8');
  const index=readFileSync(new URL('../../src/run_analysis/web/static/index.html',import.meta.url),'utf8');
  assert.match(router,/import \{ renderProgress \} from "\.\/views\/progress-wip\.js"/);
  assert.match(router,/hash === "progress"\) await renderProgress\(\)/);
  assert.match(router,/hash === "progress-wip"[^]*?location\.replace\("#progress"\); return;/);
  assert.doesNotMatch(router,/renderArchivedProgress|renderProgressWip/);
  assert.match(index,/data-route="progress" href="#progress">Progress<\/a>/);
  assert.doesNotMatch(index,/data-route="progress-wip"|Progress WIP/);
});

test('WIP has only the new aerobic graph, then the existing lower sections, without window-relative verdicts',async()=>{
  const savedPerformance=data.performance,savedWindows=data.available_windows;
  data.available_windows=[14,28,42,56,90,180,365];
  data.aerobic_performance={available:true,as_of:data.as_of,last_observation:data.as_of,target_hr_bpm:148,
    progress_trend:{direction:'improving',evidence:'likely',span_days:84,early_estimate:false,pace_change_seconds_per_mile:-31.6},
    current:{pace_min_mile:10+8/60,run_count:15},history:[{as_of:data.as_of,pace_min_mile:10+8/60}],
    observations:[{as_of:data.as_of,activity_id:77,pace_min_mile:10,trend_pace_min_mile:10+8/60}],
    smoothing_comparison:{current:{pace_min_mile:11},history:[{as_of:data.as_of,pace_min_mile:11}]},
    quality_sessions:[{as_of:data.as_of,activity_id:88,workout_type:'intervals',observed_pace_min_mile:9}]};
  const source=structuredClone(data.aerobic_performance),handlers={};
  globalThis.location.hash='#progress';
  page.querySelectorAll=selector=>selector==='[data-window]'?[42,56].map(days=>({
    dataset:{window:String(days)},addEventListener:(type,callback)=>handlers[days]=callback})):[];
  try {
    await renderProgress();
    assert.match(page.innerHTML,/data-window="90" class="selected"/);
    assert.equal((page.innerHTML.match(/<strong>10:08\/mi<\/strong>/g)??[]).length,1);
    assert.doesNotMatch(page.innerHTML,/benchmark-pace-summary/);
    for(const days of [42,56]) {
      data.performance={...savedPerformance,period_change:{direction:days===42?'improving':'declining',
        evidence:'likely',pace_change_seconds_per_mile:days===42?-16:26}};
      handlers[days]();await new Promise(resolve=>setImmediate(resolve));
      assert.match(page.innerHTML,/<h1>Your aerobic pace is likely improving\.<\/h1>/);
      assert.match(page.innerHTML,/headline-pace[^]*?<strong>10:08\/mi<\/strong>/);
      assert.match(page.innerHTML,/12-week aerobic trend/);
      assert.match(page.innerHTML,/Likely 32 sec\/mi faster/);
      assert.match(page.innerHTML,/volume-level[^]*?15.8 <small>mi in the last 7 days/);
      assert.match(page.innerHTML,/aria-label="History shown"/);
      assert.match(page.innerHTML,/<strong>10:08\/mi<\/strong>/);
      assert.match(page.innerHTML,/benchmark:77|benchmark-quality:88/);
      assert.equal((page.innerHTML.match(/wide-card benchmark-pace-card/g)??[]).length,1);
      assert.doesNotMatch(page.innerHTML,/wide-card chart-card|fitness-level-card|aerobic-progress-card|recent-trajectory|How we know|Compared with the previous|Faster than|Slower than|Likely faster|Likely slower|Compare smoothing|14-day · more responsive|New view/);
      assert.match(page.innerHTML,/aria-label="Pace display"/);
      for(const section of ['wide-card vo2-card','wide-card training-history','wide-card coverage-card']) {
        assert.ok(page.innerHTML.indexOf(section)>page.innerHTML.indexOf('benchmark-pace-card'));
      }
      assert.doesNotMatch(page.innerHTML,/NaN|undefined|Infinity/);
    }
    // The history buttons must refresh WIP, not abandon it or render the old headline.
    handlers[42](); await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/<h1>Your aerobic pace is likely improving\.<\/h1>/);
    assert.match(page.innerHTML,/data-window="42" class="selected"/);
    assert.deepEqual(data.aerobic_performance,source);
    // Returning to WIP keeps the selected view; original settings stay intact.
    globalThis.location.hash='#progress';await renderArchivedProgress();
    assert.match(page.innerHTML,/data-window="28" class="selected"/);
    globalThis.location.hash='#progress';await renderProgress();
    assert.match(page.innerHTML,/data-window="42" class="selected"/);
  } finally {
    data.performance=savedPerformance;data.available_windows=savedWindows;delete data.aerobic_performance;
    globalThis.location.hash='#progress';selectFitnessWindow(28);page.querySelectorAll=()=>[];
  }
});

test('WIP cannot overwrite another route after its response arrives',async()=>{
  const original=globalThis.fetch;
  let release;
  globalThis.fetch=async()=>({ok:true,json:()=>new Promise(resolve=>{release=()=>resolve(data);})});
  globalThis.location.hash='#progress';
  try {
    const pending=renderProgress();
    await new Promise(resolve=>setImmediate(resolve));
    globalThis.location.hash='#runs';page.innerHTML='Run Analysis';
    release();await pending;
    assert.equal(page.innerHTML,'Run Analysis');
  } finally {globalThis.fetch=original;globalThis.location.hash='#progress';}
});

test('WIP pace controls update the new graph and summary, not the removed pace card',async()=>{
  const point={as_of:data.as_of,activity_id:77,pace_min_mile:10,observed_pace_min_mile:11,
    trend_pace_min_mile:10,unadjusted_trend_pace_min_mile:11};
  data.aerobic_performance={available:true,as_of:data.as_of,last_observation:data.as_of,target_hr_bpm:148,
    progress_trend:{direction:'improving',evidence:'clear',span_days:84,early_estimate:false},
    current:{pace_min_mile:10},history:[point],observations:[point],
    unadjusted:{available:true,progress_trend:{direction:'maintaining',span_days:84},current:{pace_min_mile:11},history:[{as_of:data.as_of,pace_min_mile:11}]}};
  const handlers={};globalThis.location.hash='#progress';
  page.querySelectorAll=selector=>selector==='[data-metric]'?['standardized','raw','both'].map(metric=>({
    dataset:{metric},addEventListener:(type,callback)=>handlers[metric]=callback})):[];
  try {
    await renderProgress();
    assert.match(page.innerHTML,/>Adjusted<\/button>/);
    assert.match(page.innerHTML,/>Unadjusted<\/button>/);
    handlers.raw();await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/<h1>Maintaining recorded pace\.<\/h1>/);
    assert.match(page.innerHTML,/headline-pace[^]*?<strong>11:00\/mi<\/strong>/);
    assert.match(page.innerHTML,/<strong>11:00\/mi<\/strong>/);
    assert.match(page.innerHTML,/data-run-key="benchmark-raw:77"/);
    assert.doesNotMatch(page.innerHTML,/data-run-key="benchmark:77"|wide-card chart-card/);
    handlers.both();await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/<h1>Your aerobic pace is improving\.<\/h1>/);
    assert.match(page.innerHTML,/data-run-key="benchmark-raw:77"/);
    assert.match(page.innerHTML,/data-run-key="benchmark:77"/);
    assert.match(page.innerHTML,/benchmark-pace-line unadjusted/);
    handlers.standardized();await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/<strong>10:00\/mi<\/strong>/);
    assert.doesNotMatch(page.innerHTML,/benchmark-pace-line unadjusted/);
  } finally {delete data.aerobic_performance;globalThis.location.hash='#progress';page.querySelectorAll=()=>[];}
});

test('observed pace view removes duplicate corrected/raw controls and states actual measurement',async()=>{
  data.baseline.series=[{activity_id:1,start_time:'2026-10-05T00:00:00Z',standardized_pace_min_mile:10,
    raw_pace_min_mile:10,uncertainty_95_min_mile:.3,distance_miles:3,trend_weight:1,
    pace_basis:'observed_comparable',observed_average_hr_bpm:146}];
  try {
    await renderArchivedProgress();
    assert.match(page.innerHTML,/Recorded pace from comparable easy running/);
    assert.match(page.innerHTML,/Easy-running pace near/);
    assert.match(page.innerHTML,/Sampled HR: 146 bpm/);
    assert.match(page.innerHTML,/not corrected for heat, hills or fatigue/);
    assert.doesNotMatch(page.innerHTML,/Adjusted pace|Unadjusted pace|Pace display|adjusted for conditions/);
  } finally {data.baseline.series=[];}
});

test('normalized Progress uses matched pace, evidence and coverage while retaining VO2 and load',async()=>{
  const baseline=structuredClone(data.baseline),performance=structuredClone(data.performance);
  const time='2026-10-05T00:00:00Z';
  data.baseline.activity_coverage=[{activity_id:1,start_time:time,distance_miles:4,workout_type:'easy',health_tag:'normal',
    included_in_trend:false,score_status:'unavailable',reason:'Old model had no score'}];
  data.pace_comparison={
    runs:[{activity_id:1,pace_min_mile:9.5,included_in_trend:true,trend_weight:1}],
    series:[{activity_id:1,start_time:time,standardized_pace_min_mile:9.5,raw_pace_min_mile:10,
      uncertainty_95_min_mile:.3,distance_miles:4,trend_weight:1}],
    trend:[{as_of:time,pace_min_mile:9.5}],recorded_trend:[{as_of:time,pace_min_mile:10}],
    performance:{current:{pace_min_mile:9.5,uncertainty_95_min_mile:.3,run_count:1},confidence:'low',
      fitness_trend:'uncertain',last_observation:time}};
  const handlers={};
  page.querySelectorAll=selector=>selector==='[data-metric]'?['raw','both','standardized'].map(metric=>({
    dataset:{metric},addEventListener:(type,callback)=>handlers[metric]=callback})):[];
  try {
    const presentation=pacePresentation(data);
    assert.deepEqual(presentation.perf.vo2,performance.vo2);
    assert.deepEqual(presentation.progress.current_load,baseline.current_load);
    assert.equal(presentation.progress.activity_coverage[0].included_in_trend,true);
    assert.equal(data.baseline.activity_coverage[0].included_in_trend,false);
    await renderArchivedProgress();
    assert.match(page.innerHTML,/headline-pace[^]*?<strong>9:30\/mi<\/strong>/);
    assert.match(page.innerHTML,/Adjusted pace|Unadjusted pace|Pace display/);
    assert.match(page.innerHTML,/<strong>52.9<\/strong>/);
    assert.doesNotMatch(page.innerHTML,/stored historical correction model|Old model had no score/);
    assert.match(page.innerHTML,/1 of 13 runs in this period inform the aerobic trend/);
    handlers.both();await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/Solid green = adjusted trend · dashed amber = unadjusted trend/);
    assert.match(page.innerHTML,/class="chart-line trend-28"/);
    assert.match(page.innerHTML,/class="chart-line raw-trend"/);
    handlers.raw();await new Promise(resolve=>setImmediate(resolve));
    assert.match(page.innerHTML,/<h2>Unadjusted pace<\/h2>/);
    assert.match(page.innerHTML,/class="chart-line raw-trend"/);
    assert.match(page.innerHTML,/<path d="" class="chart-line trend-28"\/>/); // No adjusted curve in raw-only mode.
    handlers.standardized();await new Promise(resolve=>setImmediate(resolve));
  } finally {
    data.baseline=baseline;data.performance=performance;delete data.pace_comparison;
    page.querySelectorAll=()=>[];
  }
});

test('prototype fitness level stays alongside the existing VO2 card, pace and monthly change',async()=>{
  const change={speed_change_percent:4.2,estimated_change_range_95_percent:[-2.2,11.1],direction:'improving',evidence_strength:'likely'};
  data.aerobic_progress={available:true,as_of:data.as_of,last_observation:data.as_of,days_since_observation:0,
    current:{run_count:10,pace_min_mile:10.2},change,change_90d:null,history:[{as_of:data.as_of,activity_id:1,run_count:10,pace_min_mile:10.2,change}],
    fitness_level:{available:true,value:49.7,estimated_range_95:[42.4,57],as_of:data.as_of,last_observation:data.as_of,
      history:[{available:true,value:49.7,estimated_range_95:[42.4,57],as_of:data.as_of,activity_id:1,run_count:10}]}};
  try {
    await renderArchivedProgress();
    assert.match(page.innerHTML,/class="wide-card chart-card"/);
    assert.match(page.innerHTML,/class="wide-card aerobic-progress-card"/);
    assert.match(page.innerHTML,/class="wide-card fitness-level-card"/);
    assert.match(page.innerHTML,/class="wide-card benchmark-pace-card"/);
    assert.match(page.innerHTML,/class="insight-chart benchmark-pace-chart"/);
    assert.match(page.innerHTML,/<strong>49.7<\/strong>/);
    assert.match(page.innerHTML,/class="wide-card vo2-card"/);
    assert.equal((page.innerHTML.match(/class="wide-card vo2-card"/g)??[]).length,1);
    assert.match(page.innerHTML,/Each estimate uses the preceding 14 days/);
    assert.ok(page.innerHTML.indexOf('wide-card chart-card')<page.innerHTML.indexOf('wide-card aerobic-progress-card'));
    assert.match(page.innerHTML,/\+4.2%/);
    assert.doesNotMatch(page.innerHTML,/NaN|undefined|Infinity/);
  } finally {delete data.aerobic_progress;}
});

test('direct performance trend is connected independently without replacing the earlier prototypes',async()=>{
  data.aerobic_performance={available:true,version:'direct-aerobic-performance-v1',as_of:data.as_of,
    last_observation:data.as_of,target_hr_bpm:148,current:{pace_min_mile:9.9,run_count:5},
    history:[{as_of:data.as_of,pace_min_mile:9.9}],
    observations:[{as_of:data.as_of,activity_id:77,pace_min_mile:9.5,trend_pace_min_mile:9.9}]};
  try {
    await renderArchivedProgress();
    assert.match(page.innerHTML,/class="wide-card benchmark-pace-card"/);
    assert.match(page.innerHTML,/data-run-key="benchmark:77"/);
    assert.match(page.innerHTML,/Line = performance trend/);
    assert.match(page.innerHTML,/class="wide-card chart-card"/);
    assert.match(page.innerHTML,/class="wide-card vo2-card"/);
    assert.doesNotMatch(page.innerHTML,/NaN|undefined|Infinity/);
  } finally {delete data.aerobic_performance;}
});
