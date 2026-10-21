import {test} from 'node:test';
import assert from 'node:assert/strict';
import {trainingBackgroundFields,readTrainingBackground} from '../../src/run_analysis/web/static/app/training-background.js';
import {baselineInstructions} from '../../src/run_analysis/web/static/app/setup-baseline.js';

const report={weekly_miles:24,runs_per_week:4,regular_long_miles:8,maintaining_training:true};
const ready={steps:['profile','heart_rate','zones'].map(step=>({step,complete:true}))};
const settings={training_background:report,zones:{z2:{minimum_bpm:141,maximum_bpm:153}}};
test('experienced runner sees current exposure fields distinct from future goals',()=>{
  const html=trainingBackgroundFields(settings);
  assert.match(html,/Weekly volume/);assert.match(html,/Runs per week/);assert.match(html,/Regular long run/);
  assert.match(html,/not your best week or your future goal/);
  assert.match(html,/Record one normal easy run/);
});
test('calibration instructions repeat the starting level instead of a beginner reset',()=>{
  const html=baselineInstructions(ready,settings);
  assert.match(html,/5.3 miles/);assert.match(html,/24 miles per week, 4 runs/);assert.match(html,/8-mile regular long run/);
  assert.doesNotMatch(html,/10–30 minutes/);
  assert.equal(baselineInstructions(ready,{...settings,planner:{status:'adopted'}}),'');
});
test('report serializes separately and absence does not clear an adopted starting report',()=>{
  const values={background_choice:'experienced',background_volume:'24',background_frequency:'4',background_long:'8',
    background_maintaining:'yes',background_last_run:''};
  const form={querySelector:s=>values[s.match(/name="([^"]+)/)[1]]===undefined?null:{value:values[s.match(/name="([^"]+)/)[1]]}};
  assert.deepEqual(readTrainingBackground(form),{...report,last_run_date:null});
  values.background_choice='new';assert.equal(readTrainingBackground(form),null);
  delete values.background_choice;assert.equal(readTrainingBackground(form),undefined);
});
