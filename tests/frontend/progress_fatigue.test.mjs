import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fatiguePanel} from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

const date=v=>new Date(v).toISOString().slice(0,10);
const at='2026-10-10T16:00:00Z';
function data() {
  const sample=(value,complete=true)=>({value:complete?value:null,low:complete?value-.1:null,high:complete?value+.1:null,complete});
  return {as_of:at,timezone:'UTC',activities:[{activity_id:5,start_time:at,distance_miles:3,role:'easy'}],
    training_load:{daily:['2026-10-09T12:00:00Z',at].map(as_of=>({as_of,
      recent:sample(36),baseline:sample(30),short:sample(50)})),ratio:1.2,
      hr_coverage:.99,runs:[3,4,5].map((id,i)=>({activity_id:id,start_time:at,
        load:30+i,low:29+i,high:31+i,adjusted_pace:10+i/10,uncertainty_95:.4})),
      comparison:{difference_seconds:15,lower_runs:3,higher_runs:3}},
    fatigue:{daily:[{as_of:'2026-10-09T12:00:00Z',value:.8,low:.7,high:.9,complete:true},
      {as_of:at,value:1.2,low:1.1,high:1.3,complete:true}],usual:1,current:{value:1.2,complete:true},
      memory_days:2,runs:[3,4,5].map((id,i)=>({activity_id:id,start_time:at,
        fatigue:.2+i/10,low:.1+i/10,high:.3+i/10,adjusted_pace:10+i/10,uncertainty_95:.4})),
      comparison:{difference_seconds:15,lower_runs:3,higher_runs:3}}};
}

test('training load preserves the measured curves and interaction without exploratory comparisons',()=>{
  const html=fatiguePanel(data(),date);
  assert.match(html,/<h2>Training load<\/h2>/);
  assert.match(html,/Load, not readiness/);
  assert.match(html,/1.20×/);
  assert.doesNotMatch(html,/Does pace line up|fatigue-scatter|Higher-load runs averaged|Adjusted pace/);
  assert.match(html,/20% above your longer-term baseline/);
  assert.match(html,/load-line load-recent/);
  assert.match(html,/load-line load-baseline/);
  assert.match(html,/load-line load-short/);
  assert.match(html,/HR load \/ day/);
  assert.match(html,/data-run-key="5"/);
  assert.match(html,/href="#run\/5"/);
  assert.doesNotMatch(html,/About training load|exponential averages|time constants|Garmin|EPOC/);
  assert.doesNotMatch(html,/load-target|optimal-zone/);
  assert.doesNotMatch(html,/NaN|undefined|fully recovered|overtraining/);
});

test('missing and partial fatigue evidence stays visibly unavailable, not zero',()=>{
  assert.match(fatiguePanel({},date),/More recorded HR history/);
  const d=data();d.fatigue.current={value:null,complete:false};d.fatigue.usual=null;
  d.fatigue.comparison=null;d.fatigue.runs=[];
  d.training_load.ratio=null;d.training_load.comparison=null;d.training_load.runs=[];
  d.training_load.daily.forEach(r=>{r.baseline={complete:false,value:null,low:null,high:null};});
  const html=fatiguePanel(d,date);
  assert.match(html,/Longer-term baseline not yet available/);
  assert.doesNotMatch(html,/load-line load-baseline/);
  assert.doesNotMatch(html,/Not enough paired evidence|fatigue-performance/);
  assert.doesNotMatch(html,/NaN|undefined|0\.00×/);
});

test('missing recent dates produce gaps, and no zero baseline or fake readiness verdict',()=>{
  const d=data();d.training_load.daily.forEach(r=>{r.recent={complete:false,value:null,low:null,high:null};});
  const html=fatiguePanel(d,date);
  assert.match(html,/More recorded HR history/);
  assert.doesNotMatch(html,/load-line load-recent|NaN|undefined/);
});

test('Dashboard status uses the chart dates, hover metadata and matching colors',()=>{
  const d=data();
  d.load_display={...d.training_load,smoothing_days:7,equality_tolerance:.05,
    daily:d.training_load.daily.map(r=>({...r,ratio:1.2}))};
  d.training_load.ratio=9; // The headline must use the plotted ratio, not raw accounting.
  d.status_history=d.training_load.daily.map((r,i)=>({as_of:r.as_of,
    status:'above',label:'More load than usual',detail:'Recent training load is 20% higher than your six-week training trend.'}));
  const html=fatiguePanel(d,date);
  assert.match(html,/<h2>Training load<\/h2>/);
  assert.doesNotMatch(html,/training-status-cell|load-status-label|aligned strip/);
  assert.match(html,/viewBox="0 0 720 250"/);
  assert.match(html,/data-run-key="day:2026-10-09/);
  assert.match(html,/data-chart-description="Hover or tap to inspect daily load and status/);
  assert.match(html,/data-load-status="above" style="stroke:#91bcec"/);
  assert.match(html,/20% higher than your usual load/);
  assert.match(html,/Recent load: 36.0 HR load\/day/);
  assert.match(html,/Your usual load \(6-week trend\): 30.0 HR load\/day/);
  assert.match(html,/More load than usual/);
  assert.match(html,/Your usual load · 6-week trend/);
  assert.match(html,/a reference for comparison, not the amount you should aim to do/);
  assert.doesNotMatch(html,/Above baseline|Near baseline|Below baseline|longer-term baseline/);
  assert.match(html,/About the line colors/);
  assert.match(html,/load-comparison-zone/);
  assert.match(html,/1.20×/);
  assert.doesNotMatch(html,/9.00×|load-line load-short|Improving pace|Building load|Easing off/);
  assert.doesNotMatch(html,/NaN|undefined|optimal-zone|href="#run/);
});

test('load line changes color exactly where it crosses the moving recommended range',()=>{
  const d=data();
  d.load_display={...d.training_load,daily:d.training_load.daily.map((r,i)=>({...r,
    as_of:i?at:'2026-10-09T16:00:00Z',
    recent:{complete:true,value:i?60:10,low:i?59:9,high:i?61:11}}))};
  d.load_guidance={daily:d.load_display.daily.map((r,i)=>({as_of:r.as_of,
    status:i?'above':'below',label:i?'Above recommended load':'Below recommended load',
    recommended_range:{low:i?30:20,high:i?50:40}}))};
  const html=fatiguePanel(d,date);
  const paths=[...html.matchAll(/data-load-status="(\w+)" style="stroke:([^"]+)" d="([^"]+)"/g)];
  assert.deepEqual(paths.map(p=>[p[1],p[2]]),[
    ['below','#b5a5d5'],['within','#78d8b8'],['above','#e7bc68']]);
  // Crossings are 1/4 and 3/4 of the interval, including a moving band.
  assert.equal(paths[0][3].split(' ').at(-1).slice(1),paths[1][3].split(' ')[0].slice(1));
  assert.equal(paths[1][3].split(' ').at(-1).slice(1),paths[2][3].split(' ')[0].slice(1));
  assert.match(paths[0][3],/L212\.3,/);
  assert.match(paths[1][3],/L540\.8,/);
  assert.match(html,/load-recommended-zone|data-run-key="day:/);
  assert.doesNotMatch(html,/training-status-cell|NaN|undefined/);
});

test('colored load line does not connect across missing days or invent a range',()=>{
  const d=data();
  const start=d.training_load.daily[0];
  d.load_display={...d.training_load,daily:[start,
    {...start,as_of:'2026-10-10T12:00:00Z',recent:{complete:false}},
    {...start,as_of:'2026-10-11T12:00:00Z'}]};
  d.as_of=d.load_display.daily.at(-1).as_of;
  d.load_guidance={daily:d.load_display.daily.map(r=>({as_of:r.as_of,status:'unavailable',label:'Load history incomplete'}))};
  const html=fatiguePanel(d,date);
  const paths=[...html.matchAll(/data-load-status="unavailable" style="stroke:#35434e" d="([^"]+)"/g)];
  assert.equal(paths.length,2);
  assert.ok(paths.every(p=>!p[1].includes('L')));
  assert.doesNotMatch(html,/load-recommended-zone|NaN|undefined|training-status-cell/);
});

test('DST fallback does not split consecutive local dates into a false data gap',()=>{
  const d=data();d.timezone='America/New_York';d.as_of='2026-11-01T12:00:00-05:00';
  d.training_load.daily[0].as_of='2026-10-31T12:00:00-04:00';
  d.training_load.daily[1].as_of=d.as_of;
  const html=fatiguePanel(d,date);
  assert.equal((html.match(/class="load-line load-recent"/g)??[]).length,1);
  assert.equal((html.match(/class="load-line load-baseline"/g)??[]).length,1);
});

test('embedded graph retains interaction without a nested card or duplicate title',()=>{
  const html=fatiguePanel(data(),date,{embedded:true});
  assert.match(html,/class="load-chart-pane"/);
  assert.match(html,/data-run-chart/);
  assert.match(html,/load-line load-recent/);
  assert.doesNotMatch(html,/<article|<h2>Training load/);
  assert.doesNotMatch(fatiguePanel({},date,{embedded:true}),/<article|<h2>/);
});
