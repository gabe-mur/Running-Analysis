import {test} from 'node:test';
import assert from 'node:assert/strict';
import {beginnerTrainingMarkup,beginnerFeedbackPayload} from '../../src/run_analysis/web/static/app/beginner-training.js';
import {baselineInstructions} from '../../src/run_analysis/web/static/app/setup-baseline.js';

test('starter shows a simple check-in and only requests times when missing',()=>{
  const starter={running_minutes:8,summary:'Build easy time.',next_feedback:{activity_id:5,needs_times:false}};
  let html=beginnerTrainingMarkup(starter);
  assert.match(html,/8 minutes of easy running/);
  assert.match(html,/Yes, comfortable/); assert.match(html,/No, too hard/);
  assert.match(html,/#run\/5/);assert.match(html,/full planner takes over/);
  assert.match(html,/<details><summary>Correct running\/walking times/);
  assert.doesNotMatch(html,/step="0.1" required/);
  html=beginnerTrainingMarkup({...starter,next_feedback:{activity_id:5,needs_times:true}});
  assert.match(html,/name="running_minutes"/);assert.match(html,/Longest running stretch/);
  assert.equal(beginnerTrainingMarkup(null),'');
});
test('check-in distinguishes reported run and walk times without inventing a quality session',()=>{
  const fields={running_minutes:{value:'8'},longest_bout_minutes:{value:'1'}};
  const form={querySelector:s=>fields[s.match(/name="([^"]+)/)[1]]};
  assert.deepEqual(beginnerFeedbackPayload(form,{value:'yes'}),{comfortable:true,running_minutes:8,longest_bout_minutes:1});
  assert.deepEqual(beginnerFeedbackPayload({querySelector:()=>null},{value:'no'}),{comfortable:false});
});
test('uploaded starter recording leads to Weekly Plan, not another generic baseline',()=>{
  const state={steps:[{step:'runs',complete:true}]};
  assert.match(baselineInstructions(state,{beginner_start:true}),/#next-run/);
  assert.equal(baselineInstructions(state,{beginner_start:true,planner:{status:'adopted'}}),'');
});
