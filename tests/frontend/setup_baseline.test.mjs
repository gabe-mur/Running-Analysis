import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ageFromBirthDate, baselineInstructions } from '../../src/run_analysis/web/static/app/setup-baseline.js';

const settings = {
  profile: {max_hr_source: 'estimated'}, setup: {zone_method: 'heart_rate_reserve'},
  zones: {z2: {minimum_bpm: 141, maximum_bpm: 153}},
};
const state = (complete = ['profile', 'heart_rate', 'zones']) => ({steps:
  ['profile', 'heart_rate', 'zones', 'runs'].map(step => ({step, complete: complete.includes(step)}))});

test('no baseline prescription before profile and HR calibration are saved', () => {
  for (const missing of ['profile', 'heart_rate', 'zones']) {
    const html = baselineInstructions(state(['profile', 'heart_rate', 'zones'].filter(x => x !== missing)), settings);
    assert.match(html, /Save your personal details/);
    assert.doesNotMatch(html, /10–30|141–153|Your first baseline run/);
  }
});

test('saved calibration gives conversational effort and the actual numeric Z2 range', () => {
  const html = baselineInstructions(state(), settings);
  assert.match(html, /10–30 minutes at conversational effort/);
  assert.match(html, /Z2: 141–153 bpm/);
  assert.match(html, /age-estimated/);
  assert.match(html, /FIT, TCX or GPX activity files, or a Garmin ZIP/);
  assert.match(html, /running workout with GPS and heart rate enabled/);
  assert.match(html, /Sync/);
  assert.doesNotMatch(baselineInstructions(state(), {...settings, setup: {zone_method: 'device'}}), /age-estimated/);
});

test('existing history does not prescribe another first baseline run', () => {
  assert.equal(baselineInstructions(state(['profile', 'heart_rate', 'zones', 'runs']), settings), '');
});

test('birth date determines age without timezone or future/invalid-date surprises', () => {
  const today = new Date(2026, 9, 11);
  assert.equal(ageFromBirthDate('1995-11-28', today), 30);
  assert.equal(ageFromBirthDate('1995-10-11', today), 31);
  assert.equal(ageFromBirthDate('2027-01-01', today), null);
  assert.equal(ageFromBirthDate('1995-02-30', today), null);
  assert.equal(ageFromBirthDate('', today), null);
});

test('actual setup offers history import first, then profile and zones before a baseline', async () => {
  const source = await readFile(new URL('../../src/run_analysis/web/static/app/views/setup.js', import.meta.url), 'utf8');
  assert.ok(source.indexOf('<h2>About you') < source.indexOf('<h2>Heart-rate zones'));
  assert.doesNotMatch(source, /<h2>Maximum and resting heart rate<\/h2>/);
  assert.match(source, /heartRateFields\(settings/);
  assert.ok(source.indexOf('id="setup-baseline"') > source.indexOf('>Save setup</button>'));
  assert.match(source, /if \(!await runPreview\(\)\)/);
  assert.ok(source.indexOf('id="setup-import"') < source.indexOf('<h2>About you'));
  assert.match(source, /I don’t have recorded runs/);
  assert.match(source, /<form id="setup-form" \$\{chooseHistory \? 'hidden'/);
});
