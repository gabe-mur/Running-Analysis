import {test} from 'node:test';
import assert from 'node:assert/strict';
import {aerobicProgressPanel,aerobicProgressChart,fitnessLevelPanel,fitnessLevelChart} from '../../src/run_analysis/web/static/app/aerobic-progress-panel.js';

const change={speed_change_percent:4.2,estimated_change_range_95_percent:[-2.2,11.1],direction:'improving',evidence_strength:'likely'};
const state={available:true,as_of:'2026-10-14T16:00:00Z',last_observation:'2026-10-12T16:00:00Z',days_since_observation:2,
  current:{run_count:10},change,change_90d:{...change,speed_change_percent:.9,direction:'stable_or_uncertain',evidence_strength:'inconclusive'},
  history:[{as_of:'2026-10-09T16:00:00Z',activity_id:1,run_count:10,change:{...change,speed_change_percent:3}},
    {as_of:'2026-10-12T16:00:00Z',activity_id:2,run_count:10,change}]};
const date=value=>value.slice(0,10);

test('shows a distinct progress signal with uncertainty and no independent-test claim',()=>{
  const html=aerobicProgressPanel(state,date,28);
  assert.match(html,/Aerobic progress|Prototype/);
  assert.match(html,/\+4.2%/);
  assert.match(html,/Likely improving/);
  assert.match(html,/\+0.9%/);
  assert.match(html,/No clear change/);
  assert.match(html,/Above zero = faster/);
  assert.match(html,/aerobic-progress-band/);
  assert.match(html,/data-run-chart/);
  assert.match(html,/not a separate fitness test/);
  assert.match(html,/-2.2% to \+11.1%/);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('history controls do not relabel the fixed comparison as a selected-window estimate',()=>{
  for(const days of [14,28,90,365]) {
    const html=aerobicProgressPanel(state,date,days);
    assert.match(html,/\+4.2%/);
    assert.match(html,/Latest 28 days vs preceding 28 days/);
    assert.match(html,/vs level 90 days earlier/);
  }
});

test('coherent estimates remain visible without invented historical confidence bands',()=>{
  const point={speed_change_percent:2.8,direction:'stable_or_uncertain',evidence_strength:'inconclusive',
    inference_method:'bootstrap-likelihood',zero_change_included:true};
  const coherent={...state,version:'coherent-aerobic-progress-v1',change:point,
    change_90d:{...point,speed_change_percent:1.},history:state.history.map(r=>({...r,change:point}))};
  const html=aerobicProgressPanel(coherent,date,28);
  assert.match(html,/\+2.8%/);
  assert.match(html,/Estimated improvement/);
  assert.match(html,/evidence still allows no underlying change/);
  assert.match(html,/class="aerobic-progress-line"/);
  assert.doesNotMatch(html,/aerobic-progress-band|conditional_probability|NaN|undefined|Infinity/);
});

test('unavailable or sparse comparative history does not break the rest of Progress',()=>{
  assert.equal(aerobicProgressPanel(null,date,28),'');
  assert.match(aerobicProgressPanel({available:false,reason:'No observations'},date,28),/No observations/);
  const html=aerobicProgressPanel({...state,change:null,change_90d:null,history:[]},date,28);
  assert.match(html,/More history needed/);
  assert.match(html,/More comparable run history/);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('isolated points remain inspectable and gaps do not become continuous progress',()=>{
  const html=aerobicProgressChart({...state,history:[state.history[0],{...state.history[1],as_of:'2026-09-05T16:00:00Z'}]},date,90);
  assert.equal((html.match(/class="aerobic-progress-line"/g)??[]).length,2);
  assert.match(html,/data-run-key="aerobic:1"/);
});

test('declining and provisional estimates are not painted as confident improvement',()=>{
  const html=aerobicProgressPanel({...state,change:{...change,speed_change_percent:-4,direction:'declining'}},date,28);
  assert.match(html,/is-declining/);
  assert.match(html,/Likely declining/);
  const early=aerobicProgressPanel({...state,change:{...change,direction:'provisional'}},date,28);
  assert.match(early,/Early estimate/);
});

const fitness={available:true,value:49.8,estimated_range_95:[41.8,57.8],as_of:state.as_of,last_observation:state.last_observation,
  history:state.history.map((r,i)=>({...r,available:true,value:49+i*.8,estimated_range_95:[41+i*.8,57+i*.8]}))};

test('fitness level is absolute, interactive and separate from monthly change',()=>{
  for(const days of [14,28,90,365]) {
    const html=fitnessLevelPanel(fitness,date,days);
    assert.match(html,/<strong>49.8<\/strong>/);
    assert.match(html,/VO₂-equivalent/);
    assert.match(html,/not a lab VO₂max/);
    assert.match(html,/fitness-level-chart/);
    assert.match(html,/data-run-key="fitness:1"/);
    assert.match(html,/data-run-chart/);
    assert.match(html,/aerobic-progress-band/);
    assert.match(html,/terrain, weather, heart-rate/);
    assert.doesNotMatch(html,/49.8%|NaN|undefined|Infinity/);
  }
});

test('fitness level handles unsupported dates without bridging missing evidence',()=>{
  assert.equal(fitnessLevelPanel(null,date,28),'');
  assert.match(fitnessLevelPanel({available:false,reason:'Unsupported pace'},date,28),/Unsupported pace/);
  const middle={as_of:'2026-10-10T16:00:00Z',available:false};
  const html=fitnessLevelChart({...fitness,history:[...fitness.history,middle]},date,28);
  assert.equal((html.match(/class="aerobic-progress-line"/g)??[]).length,2);
  assert.doesNotMatch(html,/NaN|undefined|Infinity/);
});

test('both performance curves use pre-window history for their incoming segment, not hover targets',()=>{
  const before='2026-09-28T16:00:00Z'; // Two days before the 14-day viewport.
  for(const [render,source,prefix] of [[fitnessLevelChart,fitness,'fitness'],[aerobicProgressChart,state,'aerobic']]) {
    const predecessor={...source.history[0],as_of:before,activity_id:99};
    const withHistory={...source,history:[predecessor,...source.history]};
    const html=render(withHistory,date,14);
    assert.match(html,/<path class="aerobic-progress-line" d="M58\.0,/);
    assert.doesNotMatch(html,new RegExp(`data-run-key="${prefix}:99"`));
    assert.equal((html.match(/data-run-key=/g)??[]).length,2);
    // An unsupported estimate just outside the view must still break the line.
    const broken=render({...withHistory,history:[predecessor,{as_of:'2026-09-29T16:00:00Z',available:false},...source.history]},date,14);
    assert.doesNotMatch(broken,/<path class="aerobic-progress-line" d="M58\.0,/);
    assert.doesNotMatch(html,/NaN|undefined|Infinity/);
  }
});
