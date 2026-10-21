import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkInFieldsMarkup, advisoryDaypart, readinessBadgeLabel, runWarningMarkup } from '../../src/run_analysis/web/static/app/plan-status.js';

test('saved normal status stays selected without manufacturing a fresh report', () => {
  for (const status of [
    { health_status: 'normal', health_report_current: false },
    { health_status: 'normal', health_report_current: false, health_reported_at: '2026-10-03T12:00:00Z' },
  ]) {
    const markup = checkInFieldsMarkup(status);
    assert.match(markup, /value="normal" required checked/);
    assert.equal(status.health_report_current,false);
    assert.match(markup, /name="health_status"[^>]+required/);
    assert.doesNotMatch(markup,/Last check-in|No check-in recorded/);
  }
});

test('absent or unknown status does not invent a normal selection',()=>{
  for(const status of [{},{health_status:'unknown'}])
    assert.doesNotMatch(checkInFieldsMarkup(status),/required checked/);
});

test('fresh current answer is selected without the added history copy', () => {
  const markup = checkInFieldsMarkup({ health_status: 'little_tired', health_report_current: true,
    health_reported_at: '2026-10-07T12:00:00Z' });
  assert.match(markup, /value="little_tired" required checked/);
  assert.doesNotMatch(markup,/Checked in:|2026-10-07/);
  assert.equal((markup.match(/required checked/g) ?? []).length, 1);
});

test('flexible placeholder is not misrepresented as an evening recommendation', () => {
  assert.equal(advisoryDaypart({ time_recommendation_basis: 'flexible', planned_for: '2026-10-07T18:00:00-04:00' }), 'Flexible time');
  assert.equal(advisoryDaypart({ time_recommendation_basis: 'weather', planned_for: '2026-10-07T07:00:00-04:00' }), advisoryDaypart({ planned_for: '2026-10-07T07:00:00-04:00' }));
  assert.doesNotMatch(advisoryDaypart({ time_recommendation_basis: 'weather', planned_for: '2026-10-07T07:00:00-04:00' }), /suggested/i);
  assert.equal(readinessBadgeLabel({ readiness: 'unknown', readiness_label: 'Check on the day' }), 'Check on the day');
  assert.equal(readinessBadgeLabel({ readiness: 'ready' }), 'As planned');
});

test('routine readiness does not create a reminder or badge on a run card', () => {
  for (const result of [
    { readiness: 'unknown', readiness_label: 'Check in today', readiness_reason: 'No current check-in' },
    { readiness: 'unknown', readiness_label: 'Check on the day', readiness_reason: 'Recheck on that day' },
    { readiness: 'ready', readiness_label: 'As planned', readiness_reason: 'Normal today' },
  ]) assert.deepEqual(runWarningMarkup(result), { badge: '', explanation: '' });
  const form = checkInFieldsMarkup({ health_status: 'normal', health_report_current: false });
  assert.match(form, /How do you feel now\?/);
  assert.doesNotMatch(form, /Please answer|Choose how you feel/);
});

test('actual fatigue and weather warnings retain a visible badge and explanation', () => {
  for (const [readiness, label] of [['caution', 'Feeling tired'], ['caution', 'Weather caution'], ['not_ready', 'Unsafe forecast']]) {
    const markup = runWarningMarkup({ readiness, readiness_label: label, readiness_reason: 'Concern <detail>' });
    assert.ok(markup.badge.includes(label));
    assert.match(markup.explanation, /Why this warning\?/);
    assert.match(markup.explanation, /Concern &lt;detail&gt;/);
    assert.doesNotMatch(markup.explanation, /Concern <detail>/);
  }
});
