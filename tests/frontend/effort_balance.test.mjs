import { test } from 'node:test';
import assert from 'node:assert/strict';
import { effortBalance } from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

const date=d=>new Date(d).toISOString().slice(5,10);
const point=(as_of,quality=4,runs=16,unknown=0)=>({as_of,quality_sessions:quality,
  classified_runs:runs,easy_sessions:runs-quality,unknown_runs:unknown,completed_runs:runs+unknown,quality_percent:runs?quality/runs*100:null});
const hrPoint=(as_of,share=11,coverage=1)=>({as_of,above_easy_percent:share,easy_percent:100-share,
  known_hr_minutes:100,missing_hr_minutes:100/coverage-100,hr_coverage:coverage,
  moderate_minutes:share*.6,hard_minutes:share*.4});
const data=(rows,hr=rows.map(r=>hrPoint(r.as_of)))=>({as_of:rows.at(-1)?.as_of??'2026-10-09T12:00:00Z',window_start:'2026-10-01T12:00:00Z',
  effort:{trailing_days:28,current:hr.at(-1),daily:hr,
    session_balance:{trailing_runs:20,current:rows.at(-1),daily:rows,reference:{low:20,high:25}}}});

test('session target is independent of HR share and does not mutate input',()=>{
  const d=data([point('2026-10-08T12:00:00Z'),point('2026-10-09T12:00:00Z')]);
  const before=JSON.stringify(d),html=effortBalance(d,date);
  assert.match(html,/Last 16 runs<\/p>/);
  assert.match(html,/4 of 16 runs/);
  assert.match(html,/Within typical range/);
  assert.match(html,/20–25% reference/);
  assert.doesNotMatch(html,/Ideal range:/);
  assert.match(html,/class="effort-reference"/);
  assert.doesNotMatch(html,/effort-zone-key|Below 20%|Above 25%/);
  assert.match(html,/incidental Z3 time alone does not make a run quality/);
  assert.match(html,/Each point uses the last 20 completed runs/);
  assert.match(html,/Rest days do not change the share/);
  assert.match(html,/general reference, not your planner’s target/);
  assert.match(html,/Follow your Weekly Plan rather than adding runs just to reach this range\./);
  assert.doesNotMatch(html,/not extra runs to fit this chart/);
  assert.doesNotMatch(html,/Adding easy runs|Aim for|Below target|Above target|effort-status above/);
  assert.doesNotMatch(html,/Time at each effort|recorded-intensity|Easy vs harder running|HR exposure/);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
  assert.equal(JSON.stringify(d),before);
});

test('full and partial run windows are labelled honestly without the removed HR-time section',()=>{
  const html=effortBalance(data([point('2026-10-09T12:00:00Z',4,20)]),date);
  assert.match(html,/Last 20 runs<\/p>/);
  assert.doesNotMatch(html,/28 days|trailing 28|Time at each effort/);
  assert.match(effortBalance(data([point('2026-10-09T12:00:00Z',1,3)]),date),/Last 3 runs<\/p>/);
});

test('both reference boundaries are included; higher share has no status label',()=>{
  for(const [quality,runs,label] of [[0,16,'Lower quality share'],[3,16,'Lower quality share'],
    [4,20,'Within typical range'],[4,16,'Within typical range']]) {
    assert.match(effortBalance(data([point('2026-10-09T12:00:00Z',quality,runs)]),date),new RegExp(label));
  }
  const high=effortBalance(data([point('2026-10-09T12:00:00Z',9,10)]),date);
  assert.match(high,/>90%<\/text>/);
  assert.doesNotMatch(high,/Higher quality share|class="effort-status/);
  assert.doesNotMatch(high,/NaN|Infinity|undefined|injury risk/);
});

test('quality line and markers use purple, green and orange with inclusive reference edges',()=>{
  for(const [quality,runs,status,color] of [[3,20,'below','#b5a5d5'],
    [4,20,'within','#78d8b8'],[5,20,'within','#78d8b8'],[6,20,'above','#e7bc68']]) {
    const html=effortBalance(data([point('2026-10-09T12:00:00Z',quality,runs)]),date);
    assert.match(html,new RegExp(`data-quality-status="${status}" style="stroke:${color}"`));
    assert.match(html,new RegExp(`<circle style="fill:${color}"`));
    assert.doesNotMatch(html,/effort-zone-key|Below 20%|Above 25%/);
  }
});

test('line colors split at both reference crossings in either direction',()=>{
  for(const [a,b,expected] of [[3,7,['below','within','above']],[7,3,['above','within','below']]]) {
    const html=effortBalance(data([point('2026-10-08T12:00:00Z',a,20),point('2026-10-09T12:00:00Z',b,20)]),date);
    const paths=[...html.matchAll(/data-quality-status="([^"]+)" style="stroke:[^"]+" d="([^"]+)"/g)];
    assert.deepEqual(paths.map(m=>m[1]),expected);
    assert.equal(paths.length,3);
    const coords=paths.map(m=>[...m[2].matchAll(/[ML]([\d.]+),([\d.]+)/g)].map(p=>[Number(p[1]),Number(p[2])]));
    assert.deepEqual(coords[0].at(-1),coords[1][0]);
    assert.deepEqual(coords[1].at(-1),coords[2][0]);
    assert.deepEqual(coords[1].map(p=>p[1]),a<b?[99,84.5]:[84.5,99]);
  }
});

test('unknown differs from all-easy and incomplete roles do not imply a target verdict',()=>{
  assert.match(effortBalance({},date),/No classified runs/);
  const empty=effortBalance(data([point('2026-10-09T12:00:00Z',0,0,1)]),date);
  assert.match(empty,/No classified runs/);
  assert.doesNotMatch(empty,/Time at each effort|recorded-intensity/);
  const incomplete=effortBalance(data([point('2026-10-09T12:00:00Z',4,16,2)]),date);
  assert.match(incomplete,/Classification incomplete/);
  assert.match(incomplete,/2 unclassified runs excluded/);
  assert.doesNotMatch(incomplete,/effort-status balanced/);
  const few=effortBalance(data([point('2026-10-09T12:00:00Z',1,3)]),date);
  assert.match(few,/Few runs to compare/);
});

test('history gaps retain incomplete classifications and ignore HR coverage',()=>{
  const dates=['2026-10-06T12:00:00Z','2026-10-07T12:00:00Z','2026-10-08T12:00:00Z','2026-10-09T12:00:00Z'];
  const d=data(dates.map((at,i)=>point(at,4,16,i%2)),dates.map((at,i)=>hrPoint(at,11,i%2?.5:1)));
  const html=effortBalance(d,date);
  assert.equal((html.match(/class="effort-share-line"/g)??[]).length,2);
  assert.doesNotMatch(html,/Time at each effort|HR coverage|recorded-intensity/);
  delete d.effort.current;
  delete d.effort.daily;
  assert.equal(effortBalance(d,date),html);
});
