import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadAssessmentDisplay, trainingStatusDisplay} from '../../src/run_analysis/web/static/app/training-status-display.js';

test('missing status never becomes a recovery or building claim',()=>{
  assert.match(trainingStatusDisplay(null,{}),/Status unavailable/);
});

test('non-building statuses retain the actual explanation and precedence',()=>{
  for(const status of ['recovering','strained','underloaded','insufficient_data','maintaining','rebuilding']) {
    const html=trainingStatusDisplay({status,label:status,detail:'Actual reason <unsafe>',rule_trace:[]},{});
    assert.match(html,/Actual reason &lt;unsafe&gt;/);
    assert.doesNotMatch(html,/NaN|undefined|Capacity increased|fully recovered/);
  }
});

test('improving aerobic evidence can lead without a capacity increase',()=>{
  const html=trainingStatusDisplay({status:'building',label:'Building',detail:'Evidence',rule_trace:[{
    rule_id:'status_building',facts:{fitness_trend:'improving',improvement_supported:true,quality_sessions_14d:1,
      capacity_reference_miles:20,previous_capacity_reference_miles:20}}]},{});
  assert.match(html,/Aerobic trend improving/);
  assert.doesNotMatch(html,/Capacity increased/);
});

test('absent improvement evidence is not fabricated for an older payload',()=>{
  const html=trainingStatusDisplay({status:'building',label:'Building',detail:'Evidence',rule_trace:[{
    rule_id:'status_building',facts:{fitness_trend:'uncertain',quality_sessions_14d:1}}]},{});
  assert.match(html,/Not recorded/);
  assert.doesNotMatch(html,/Capacity increased|Aerobic trend improving/);
});

test('load interpretation does not depend on Building or improvement',()=>{
  const trace=[{rule_id:'status_strained',fired:false,facts:{continuous_fatigue_to_capacity_ratio:1.01,high_load_ratio:1.3}}];
  const build=loadAssessmentDisplay({status:'building',rule_trace:trace});
  assert.equal(build,loadAssessmentDisplay({status:'maintaining',rule_trace:trace}));
  assert.match(build,/No elevated strain flag/);
  assert.match(build,/101% of demonstrated capacity/);
  assert.match(build,/130%/);
  assert.doesNotMatch(build,/>Building<|fully recovered/);
});

test('a skipped load check is unavailable, not a negative strain flag',()=>{
  assert.match(loadAssessmentDisplay({status:'recovering',rule_trace:[]}),/Load check unavailable/);
  assert.match(loadAssessmentDisplay({rule_trace:[{rule_id:'status_insufficient_evidence',fired:true}]}),/Not enough load evidence/);
  assert.doesNotMatch(loadAssessmentDisplay(null),/No elevated strain flag/);
});

test('corroborated costly response can flag strain even below the load threshold',()=>{
  const html=loadAssessmentDisplay({rule_trace:[{rule_id:'status_strained',fired:true,facts:{
    continuous_fatigue_to_capacity_ratio:1.01,high_load_ratio:1.3,costly_response_corroborated:true}}]});
  assert.match(html,/Elevated strain signal/);
  assert.match(html,/corroborated by high within-run drift/);
});
