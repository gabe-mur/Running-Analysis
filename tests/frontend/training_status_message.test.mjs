import {test} from 'node:test';
import assert from 'node:assert/strict';
import {trainingStatusMessage} from '../../src/run_analysis/web/static/app/training-status-display.js';
import {performanceSummary} from '../../src/run_analysis/web/static/app/progress-insights.js';

const summary=(status,facts={},rule=`status_${status}`)=>({status,rule_trace:[{rule_id:rule,fired:true,facts}]});
const steady={direction:'stable'};
const building={capacity_reference_miles:20,previous_capacity_reference_miles:18,fitness_trend:'stable',quality_sessions_14d:2};

test('building explains demonstrated sustained volume growth independently of the aerobic trend',()=>{
  for(const [direction,text] of [['stable','Maintaining aerobic efficiency'],['uncertain','Maintaining aerobic efficiency'],['improving','Aerobic efficiency improving'],['declining','Aerobic efficiency declining'],['insufficient_data','More data needed to assess aerobic efficiency']]) {
    assert.equal(trainingStatusMessage(summary('building',building),{direction}),`${text} · Sustained volume growing`);
  }
});

test('aerobic-only building does not fabricate volume growth',()=>{
  const value=summary('building',{...building,previous_capacity_reference_miles:20,fitness_trend:'improving'});
  assert.equal(trainingStatusMessage(value,{direction:'improving',chosen:{evidence:'likely'}}),
    'Aerobic efficiency likely improving');
  // The selected Progress window can be steady while the classifier records
  // earlier aerobic gains. Both facts must remain visible, not substituted.
  assert.match(trainingStatusMessage(value,steady),/^Maintaining aerobic efficiency · Recent aerobic gains/);
  assert.doesNotMatch(trainingStatusMessage(value,steady),/volume growing/);
});

test('maintaining describes volume without presenting quality attendance as an insight',()=>{
  const facts={sustained_to_capacity_ratio:1,near_capacity_ratio:.85,quality_sessions_14d:1};
  assert.equal(trainingStatusMessage(summary('maintaining',facts,'status_building'),steady),
    'Maintaining aerobic efficiency · Volume near your established level');
  assert.equal(trainingStatusMessage(summary('maintaining',{...facts,quality_sessions_14d:0},'status_building'),steady),
    'Maintaining aerobic efficiency · Volume near your established level');
  assert.match(trainingStatusMessage(summary('maintaining',{...facts,sustained_to_capacity_ratio:.8},'status_building'),steady),/Volume below your established level/);
});

test('health and strain messages preserve their separate recorded causes',()=>{
  for(const [facts,reason] of [
    [{current_health_status:'pain_or_injury_concern'},'Pain or injury concern reported'],
    [{current_health_status:'sick_or_recovering'},'Illness reported'],
    [{current_health_status:'normal',recent_illness_or_recovery:true},'Recent health concern'],
  ]) assert.equal(trainingStatusMessage(summary('recovering',facts),steady),`Maintaining aerobic efficiency · ${reason}`);
  const high={continuous_fatigue_to_capacity_ratio:1.3,high_load_ratio:1.3};
  assert.equal(trainingStatusMessage(summary('strained',high),steady),'Maintaining aerobic efficiency · High accumulated load');
  assert.equal(trainingStatusMessage(summary('strained',{...high,costly_response_corroborated:true}),steady),'Maintaining aerobic efficiency · High accumulated load · Higher effort on the latest run');
  assert.equal(trainingStatusMessage(summary('strained',{...high,continuous_fatigue_to_capacity_ratio:1,costly_response_corroborated:true}),steady),'Maintaining aerobic efficiency · Higher effort on the latest run');
});

test('rebuilding, underloaded and evidence gates get concise distinct reasons',()=>{
  for(const [status,reason] of [
    ['rebuilding','Load rising after reduced volume'],
    ['underloaded','Sustained volume below your established level'],
    ['insufficient_data','More recent training data needed'],
  ]) assert.equal(trainingStatusMessage(summary(status),steady),`Maintaining aerobic efficiency · ${reason}`);
  assert.equal(trainingStatusMessage(null,null),'More data needed to assess aerobic efficiency · Training status unavailable');
  for(const status of ['building','maintaining','strained']) {
    assert.match(trainingStatusMessage(summary(status),steady),/Status reason unavailable/);
    assert.doesNotMatch(trainingStatusMessage(summary(status),steady),/NaN|undefined|fully recovered/);
  }
});

test('missing training evidence gets one short message when aerobic evidence is also missing',()=>{
  const facts={activities_28d:10,minimum_activities_28d:4,capacity_reference_miles:null};
  assert.equal(trainingStatusMessage(summary('insufficient_data',facts,'status_insufficient_evidence'),{direction:'insufficient_data'}),'More recent training data needed');
  assert.equal(trainingStatusMessage(summary('insufficient_data',{...facts,activities_28d:2},'status_insufficient_evidence'),null),'More recent training data needed');
});

test('quality attendance never appears as an improvement reason',()=>{
  for(const count of [0,1,3,undefined]) {
    const value=summary('building',{...building,previous_capacity_reference_miles:20,fitness_trend:'improving',quality_sessions_14d:count});
    const text=trainingStatusMessage(value,{direction:'improving'});
    assert.equal(text,'Aerobic efficiency improving');
    assert.doesNotMatch(text,/quality|session|support|interval|threshold|caused|regular/);
  }
  const missing=summary('building',{fitness_trend:'improving'});
  assert.equal(trainingStatusMessage(missing,{direction:'improving'}),'Aerobic efficiency improving');
  assert.doesNotMatch(trainingStatusMessage(missing,{direction:'improving'}),/quality/);
});

test('all six aerobic phrases combine independently with training reasons',()=>{
  for(const [aerobic,phrase] of [
    [{direction:'improving'},'Aerobic efficiency improving'],
    [{direction:'improving',chosen:{evidence:'likely'}},'Aerobic efficiency likely improving'],
    [{direction:'stable'},'Maintaining aerobic efficiency'],
    [{direction:'declining',chosen:{evidence:'likely'}},'Aerobic efficiency likely declining'],
    [{direction:'declining'},'Aerobic efficiency declining'],
    [{direction:'insufficient_data'},'More data needed to assess aerobic efficiency'],
  ]) assert.equal(trainingStatusMessage(summary('building',building),aerobic),`${phrase} · Sustained volume growing`);
});

test('selected-window conflict and likelihood survive the short message',()=>{
  const trend={direction:'improving',evidence:'clear',probability_faster:.99,pace_change_seconds_per_mile:-10};
  const conflict=performanceSummary({period_change:trend,within_window_trend:{...trend,direction:'declining',probability_faster:.01}});
  assert.match(trainingStatusMessage(summary('building',building),conflict),/^Aerobic efficiency improving/);
  assert.match(trainingStatusMessage(summary('building',building),{direction:'declining',chosen:{evidence:'likely'}}),/^Aerobic efficiency likely declining/);
});
