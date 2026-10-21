import {test} from 'node:test';
import assert from 'node:assert/strict';
import {auditHeadlines} from '../../scripts/audit_progress_headlines.mjs';

test('audit evaluates shipped wording without hiding the conflicting timescales',()=>{
  const perf={period_change:{direction:'improving',evidence:'clear',pace_change_seconds_per_mile:-60},
    within_window_trend:{direction:'declining',evidence:'likely',pace_change_seconds_per_mile:5}};
  const report={version:'fixture',pipeline_replays:[{case_kind:'recent',gain_percent:10,ramp_days:0,
    refitted_calibration_response_percent:10,refitted:{monthly_change:{speed_change_percent:10}},
    pace_headline_baseline:perf,pace_headline_refitted:perf}]};
  const before=structuredClone(report);const result=auditHeadlines(report);
  assert.equal(result.cases[0].injected.headline,'Faster than the previous period.');
  assert.equal(result.cases[0].injected.recentTrajectory,'Recent pace is likely slowing.');
  assert.equal(result.cases[0].injected.conflict,true);
  assert.equal(result.cases[0].period.direction,'improving');
  assert.equal(result.cases[0].trajectory.direction,'declining');
  assert.equal(result.cases[0].recovered_speed_gain_percent,10);
  assert.deepEqual(report,before);
});
