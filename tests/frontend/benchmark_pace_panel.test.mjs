import { test } from 'node:test';
import assert from 'node:assert/strict';
import { benchmarkPacePanel, benchmarkPaceChart } from '../../src/run_analysis/web/static/app/benchmark-pace-panel.js';

const date = value => value.slice(0, 10);
const rows = [
  { as_of: '2026-10-05T00:00:00Z', activity_id: 1, pace_min_mile: 10.5, run_count: 5, evidence: 'established', estimated_range_95_min_mile: [10, 11] },
  { as_of: '2026-10-14T00:00:00Z', activity_id: 2, pace_min_mile: 10.2, run_count: 7, evidence: 'established', estimated_range_95_min_mile: [9.9, 10.5] },
];
const state = { available: true, as_of: '2026-10-15T00:00:00Z', last_observation: rows[1].as_of, days_since_observation: 1,
  current: rows[1], history: rows, fitness_level: { target_hr_bpm: 148 } };

test('one current benchmark in pace units, no percentage or competing-period summary', () => {
  const html = benchmarkPacePanel(state, date, 28);
  assert.match(html, /Aerobic performance/);
  assert.match(html, /<strong>10:12\/mi<\/strong>/);
  assert.match(html, /At 148 bpm/);
  assert.match(html, /Faster ↑/);
  assert.match(html, /not a prescribed training pace or measured VO₂max/);
  assert.match(html, /data-run-key="benchmark:2"/);
  assert.doesNotMatch(html, /%|90 days earlier|vs preceding|NaN|undefined|Infinity/);
});

test('a faster pace plots higher with readable min:sec axis labels', () => {
  const html = benchmarkPaceChart(state, date, 28, 148);
  const marks = [...html.matchAll(/class="benchmark-pace-point[^\"]*" cx="[\d.]+" cy="([\d.]+)"/g)];
  assert.equal(marks.length, 2);
  assert.ok(Number(marks[1][1]) < Number(marks[0][1]));
  assert.match(html, />10:00<\/text>/);
  assert.match(html, />10:30<\/text>/);
  assert.match(html, /min\/mi · 148 bpm/);
  assert.doesNotMatch(html, /VO₂-equivalent|aerobic-progress-band/);
});

test('full uncertainty is inspectable but does not flatten the pace axis', () => {
  const huge = { ...state, history: rows.map(r => ({ ...r, estimated_range_95_min_mile: [1, 99] })) };
  const axes = html => [...html.matchAll(/<text x="70"[^]*?<\/text>/g)].map(match => match[0]);
  assert.deepEqual(axes(benchmarkPaceChart(huge, date, 28, 148)), axes(benchmarkPaceChart(state, date, 28, 148)));
  assert.match(benchmarkPaceChart(huge, date, 28, 148), /Model range: 1:00\/mi–99:00\/mi/);
});

test('new runners get an initial inspectable point without any prior comparison', () => {
  const first = { ...rows[1], run_count: 1, evidence: 'provisional' };
  const html = benchmarkPacePanel({ ...state, current: first, history: [first] }, date, 28);
  assert.match(html, /Early estimate/);
  assert.equal((html.match(/<circle /g) ?? []).length, 1);
  assert.match(html, /data-run-chart/);
  assert.doesNotMatch(html, /vs |More comparable run history is needed|NaN|undefined/);
});

test('viewports preserve current value and incoming history, without extrapolation or invented marks', () => {
  for (const days of [14, 28, 90, 365]) {
    assert.match(benchmarkPacePanel(state, date, days), /<strong>10:12\/mi<\/strong>/);
  }
  const predecessor = { ...rows[0], as_of: '2026-09-28T00:00:00Z', activity_id: 99 };
  const html = benchmarkPaceChart({ ...state, history: [predecessor, ...rows] }, date, 14, 148);
  assert.match(html, /class="benchmark-pace-line" d="M82\.0,/);
  assert.doesNotMatch(html, /data-run-key="benchmark:99"/);
  assert.equal((html.match(/data-run-key=/g) ?? []).length, 2);
});

test('missing evidence and long breaks are not bridged into invented progress', () => {
  assert.equal(benchmarkPacePanel(null, date, 28), '');
  assert.match(benchmarkPacePanel({ available: false }, date, 28), /first comparable runs/);
  const broken = { ...state, history: [rows[0], { as_of: '2026-10-08T00:00:00Z', pace_min_mile: null }, rows[1]] };
  assert.equal((benchmarkPaceChart(broken, date, 28, 148).match(/class="benchmark-pace-line"/g) ?? []).length, 2);
  const stale = { ...state, days_since_observation: 30 };
  assert.match(benchmarkPacePanel(stale, date, 90), /Last estimate: 2026-10-14/);
});

test('compact layout keeps pace labels and inspection support', () => {
  const previous = globalThis.matchMedia;
  globalThis.matchMedia = () => ({ matches: true });
  try {
    const html = benchmarkPaceChart(state, date, 28, 148);
    assert.match(html, /viewBox="0 0 420 280"/);
    assert.match(html, />10:00<\/text>/);
    assert.doesNotMatch(html, /<text[^>]*>[^<]*\/mi<\/text>/);
    assert.match(html, /data-run-key="benchmark:2"/);
    assert.doesNotMatch(html, /NaN|undefined|Infinity/);
  } finally { globalThis.matchMedia = previous; }
});

test('direct prototype keeps hover focused on run and pace, not internal diagnostics',()=>{
  const direct={...state,version:'direct-aerobic-performance-v1',target_hr_bpm:148,
    observations:rows.map((r,i)=>({...r,pace_min_mile:i?9.5:11,observed_pace_min_mile:11.2,
      trend_pace_min_mile:r.pace_min_mile,sampling_support:{retained_minutes:20,broader_sampling_difference_seconds:-17},
      continuity:{long_stop_count:i?1:0}}))};
  const html=benchmarkPacePanel(direct,date,28);
  assert.match(html,/Line = performance trend · dots = individual runs/);
  assert.match(html,/Adjusted pace: 9:30\/mi/);
  assert.match(html,/Unadjusted pace: 11:12\/mi/);
  assert.doesNotMatch(html,/Broader sampling:|Trend:|long stop recorded|New view/);
  assert.match(html,/Later runs can revise the earlier trend/);
  assert.doesNotMatch(html,/Each point summarizes recent running|Each date uses up to 28|Model range:|aerobic-progress-band|NaN|undefined/);
  assert.equal((html.match(/data-run-key=/g)??[]).length,2);
});

test('settled prototype makes excluded transition evidence inspectable instead of silently dropping history',()=>{
  const direct={...state,observations:rows,section_screen:{excluded_runs:[{activity_id:356,start_time:'2026-04-12T19:35:00Z'}]}};
  const html=benchmarkPacePanel(direct,date,28);
  assert.match(html,/two separate sections with settled effort and HR/);
  assert.match(html,/All runs still count toward training load/);
  assert.match(html,/href="#run\/356"/);
  assert.match(html,/2026-04-12/);
  assert.match(html,/changing effort or HR/);
});

test('detailed adjustments stay behind their link, without hiding run dots',()=>{
  const direct={...state,observations:rows.map(r=>({...r,observed_pace_min_mile:10,
    hr_response:{available:true,delta_seconds_per_mile:20.16}}))};
  const html=benchmarkPacePanel(direct,date,28,148);
  assert.doesNotMatch(html,/HR timing adjustment:/);
  assert.match(html,/Unadjusted pace: 10:00\/mi/);
  assert.match(html,/href="#pace-comparison"/);
  assert.equal((html.match(/data-run-key=/g)??[]).length,2);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('temperature is inspectable separately without a humidity claim or extra headline',()=>{
  const direct={...state,temperature_response:{version:'signed-air-hr-v1'},
    observations:rows.map(r=>({...r,temperature_response:{delta_seconds_per_mile:7.2},
      hr_response:{delta_seconds_per_mile:12.}}))};
  const html=benchmarkPacePanel(direct,date,28,148);
  assert.doesNotMatch(html,/Temperature adjustment:|HR timing adjustment:/);
  assert.match(html,/No separate humidity correction is fitted/);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('smoothing overlay uses one dot set and two distinct labelled curves on a common pace axis',()=>{
  const candidate=rows.map(r=>({...r,pace_min_mile:r.pace_min_mile-.2}));
  const direct={...state,observations:rows.map((r,i)=>({...r,trend_pace_min_mile:r.pace_min_mile,
    comparison_trend_pace_min_mile:candidate[i].pace_min_mile})),
    smoothing_comparison:{bandwidth_days:14,baseline_bandwidth_days:28,current:candidate[1],history:candidate}};
  const html=benchmarkPacePanel(direct,date,28,148);
  assert.match(html,/Compare smoothing/);
  assert.match(html,/14-day · more responsive/);
  assert.match(html,/28-day · smoother/);
  assert.match(html,/Current 28-day trend/);
  assert.doesNotMatch(html,/14-day trend:|28-day trend:/);
  assert.match(html,/class="benchmark-pace-line comparison14"/);
  assert.match(html,/class="benchmark-pace-line"/);
  assert.equal((html.match(/data-run-key=/g)??[]).length,2);
  assert.equal((html.match(/<svg /g)??[]).length,1);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('both smoothing curves retain incoming history and break at missing evidence',()=>{
  const history=[{...rows[0],as_of:'2026-09-28T00:00:00Z'},...rows];
  const direct={...state,observations:rows,smoothing_comparison:{history,current:rows[1]}};
  const html=benchmarkPaceChart(direct,date,14,148);
  assert.match(html,/class="benchmark-pace-line comparison14" d="M82\.0,/);
  assert.equal((html.match(/data-run-key=/g)??[]).length,2);
  const gap={...direct,smoothing_comparison:{...direct.smoothing_comparison,history:[rows[0],
    {as_of:'2026-10-08T00:00:00Z',pace_min_mile:null},rows[1]]}};
  assert.equal((benchmarkPaceChart(gap,date,28,148).match(/class="benchmark-pace-line comparison14"/g)??[]).length,2);
});

test('reduced context and limited GPS are visually distinct and explain the actual reason',()=>{
  const direct={...state,observations:rows.map((r,i)=>({...r,trend_pace_min_mile:r.pace_min_mile,
    trend_weight:i?.25:1,health_tag:i?'injury_affected':'normal',workout_type:i?'run_walk':'easy',
    measurement_quality:i?'full_sensor':'partial_gps_device_distance'}))};
  const html=benchmarkPacePanel(direct,date,28,148);
  assert.match(html,/benchmark-pace-point reduced-evidence/);
  assert.match(html,/benchmark-pace-point estimated-measurement/);
  assert.match(html,/Reduced weight · run\/walk structure · Injury Affected/);
  assert.doesNotMatch(html,/Context weight:/);
  assert.match(html,/Limited GPS/);
  assert.match(html,/href="#run\/2"/);
});

test('manually excluded and non-running dots cannot stretch the aerobic axis',()=>{
  const direct={...state,observations:rows};
  const base=benchmarkPaceChart(direct,date,28,148);
  const extra=[{...rows[0],activity_id:999,pace_min_mile:40,manually_excluded:true},
    {...rows[0],activity_id:998,pace_min_mile:50},
    {...rows[0],activity_id:997,pace_min_mile:60,workout_type:'hike'}];
  const hidden=benchmarkPaceChart({...direct,observations:[...rows,...extra],hidden_activity_ids:[998]},date,28,148);
  assert.equal(hidden,base);
});

test('quality sessions plot their recorded pace on the shared scale without changing trend data',()=>{
  const quality_sessions=[{activity_id:7,as_of:rows[0].as_of,workout_type:'intervals',
    observed_pace_min_mile:9,observed_average_hr_bpm:170,distance_miles:4},
    {activity_id:8,as_of:rows[1].as_of,workout_type:'tempo_threshold',observed_pace_min_mile:11}];
  const direct={...state,observations:rows,quality_sessions};
  const original=structuredClone(direct);
  const html=benchmarkPacePanel(direct,date,28,148);
  assert.match(html,/Quality · recorded pace/);
  assert.match(html,/Quality · not in the aerobic trend/);
  assert.match(html,/href="#run\/7"/);
  assert.match(html,/Recorded pace: 9:00\/mi/);
  assert.match(html,/data-run-key="benchmark-quality:7"/);
  assert.match(html,/>9:00<\/text>/);
  assert.match(html,/>11:00<\/text>/);
  const marks=[...html.matchAll(/class="benchmark-quality-point" d="M[\d.]+,([\d.]+)l/g)];
  assert.equal(marks.length,2);
  assert.ok(Number(marks[0][1])<Number(marks[1][1]));
  assert.equal((html.match(/data-run-key="benchmark:[12]"/g)??[]).length,2);
  assert.equal((html.match(/<svg /g)??[]).length,1);
  assert.deepEqual(direct,original);
  const qualityOnly=benchmarkPacePanel({...direct,available:false,current:null,observations:[],history:[]},date,28,148);
  assert.match(qualityOnly,/benchmark-quality:7/);
  assert.doesNotMatch(qualityOnly,/class="benchmark-pace-line"/);
});

test('hidden, outside-window and invalid quality paces cannot affect marks or axis',()=>{
  const direct={...state,observations:rows,hidden_activity_ids:[7],quality_sessions:[
    {activity_id:7,as_of:rows[0].as_of,observed_pace_min_mile:45},
    {activity_id:8,as_of:rows[0].as_of,observed_pace_min_mile:40,manually_excluded:true},
    {activity_id:9,as_of:'2026-01-01',observed_pace_min_mile:50},
    {activity_id:10,as_of:rows[0].as_of,observed_pace_min_mile:null}]};
  assert.equal(benchmarkPaceChart(direct,date,28,148),benchmarkPaceChart({...direct,quality_sessions:[]},date,28,148));
});

test('adjusted, unadjusted and both use paired dots and distinct fitted lines',()=>{
  const rawRows=rows.map(r=>({...r,pace_min_mile:r.pace_min_mile+1}));
  const direct={...state,observations:rows.map((r,i)=>({...r,observed_pace_min_mile:rawRows[i].pace_min_mile,
    trend_pace_min_mile:r.pace_min_mile,unadjusted_trend_pace_min_mile:rawRows[i].pace_min_mile})),
    unadjusted:{available:true,current:rawRows[1],history:rawRows},
    quality_sessions:[{activity_id:7,as_of:rows[0].as_of,observed_pace_min_mile:9}]};
  const before=structuredClone(direct);
  for(const mode of ['standardized','raw','both']) {
    const html=benchmarkPacePanel(direct,date,28,148,mode);
    assert.equal((html.match(/data-run-key="benchmark-quality:7"/g)??[]).length,1);
    assert.equal((html.match(/data-run-key="benchmark:[12]"/g)??[]).length,mode==='raw'?0:2);
    assert.equal((html.match(/data-run-key="benchmark-raw:[12]"/g)??[]).length,mode==='standardized'?0:2);
    assert.equal((html.match(/class="benchmark-pace-line"/g)??[]).length,mode==='raw'?0:1);
    assert.equal((html.match(/class="benchmark-pace-line unadjusted"/g)??[]).length,mode==='standardized'?0:1);
    assert.doesNotMatch(html,/NaN|undefined|Infinity/);
    if(mode==='raw') {
      assert.match(html,/<strong>11:12\/mi<\/strong>/);
      assert.match(html,/Unadjusted pace: 11:12\/mi/);
      assert.match(html,/Unadjusted trend/);
      assert.doesNotMatch(html,/At 148 bpm|Adjusted pace:/);
    }
    if(mode==='both') {
      assert.match(html,/Solid green = adjusted trend · dashed amber = unadjusted trend/);
      assert.match(html,/Adjusted pace: 10:12\/mi/);
      assert.match(html,/Unadjusted pace: 11:12\/mi/);
    }
  }
  assert.deepEqual(direct,before);
});

test('raw mode never substitutes adjusted evidence for missing recorded pace',()=>{
  const html=benchmarkPacePanel({...state,observations:rows,
    unadjusted:{available:false,current:{available:false},history:[]}},date,28,148,'raw');
  assert.doesNotMatch(html,/data-run-key=|class="benchmark-pace-line"|<strong>10:12/);
});

test('a page-level pace summary can replace the card summary without removing the chart',()=>{
  const html=benchmarkPacePanel(state,date,90,148,'standardized',{showSummary:false});
  assert.doesNotMatch(html,/benchmark-pace-summary|<strong>10:12\/mi<\/strong>/);
  assert.match(html,/benchmark-pace-chart/);
  assert.match(html,/data-run-key="benchmark:2"/);
});
