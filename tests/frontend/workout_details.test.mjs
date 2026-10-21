import { test } from 'node:test';
import assert from 'node:assert/strict';
import { workoutDetailsMarkup } from '../../src/run_analysis/web/static/app/workout-details.js';

const labels = { warmup: 'Easy warm-up', cooldown: 'Easy cool-down', recovery: 'Easy jogging recovery', work: 'Controlled hard effort; do not chase an instantaneous HR target' };
const phase = (kind, seconds) => ({ phase: kind, duration_minutes: seconds / 60,
  instruction: `${labels[kind]} · ${seconds / 60} min`, target_zones: kind === 'work' ? [] : ['Z1', 'Z2'] });

test('uniform phase lists display warm-up, one repetition set and cool-down', () => {
  const result = { structure: [phase('warmup', 600), phase('work', 60), phase('recovery', 90), phase('work', 60), phase('cooldown', 600)] };
  const before = JSON.stringify(result);
  const html = workoutDetailsMarkup(result);
  assert.equal((html.match(/<li>/g) ?? []).length, 3);
  assert.match(html, /2 × 1 min at controlled hard effort, with 1:30 of easy jogging between efforts/);
  assert.doesNotMatch(html, /workout-timings/);
  assert.equal(JSON.stringify(result), before);
});

test('current six-rep prescription keeps every issued duration without decimal-minute clutter', () => {
  const work = [105, 95, 100, 110, 115, 105];
  const rests = [96.693, 97.962, 93.208, 100.334, 101.234];
  const structure = [phase('warmup', 600)];
  work.forEach((s, i) => { structure.push(phase('work', s)); if (i < rests.length) structure.push(phase('recovery', rests[i])); });
  structure.push(phase('cooldown', 600));
  const result = { structure, distance_target_miles: 3.8, distance_range_miles: [3.5, 4] };
  const before = JSON.stringify(result);
  const html = workoutDetailsMarkup(result);
  assert.equal((html.match(/<li>/g) ?? []).length, 3);
  assert.match(html, /6 efforts of 1:35–1:55 at controlled hard effort, with 1:33–1:41 of easy jogging between efforts/);
  assert.match(html, /<summary>Effort breakdown<\/summary>/);
  assert.match(html, /Efforts: 1:45 · 1:35 · 1:40 · 1:50 · 1:55 · 1:45/);
  assert.match(html, /Recoveries: 1:37 · 1:38 · 1:33 · 1:40 · 1:41/);
  assert.match(html, /Distance margin: 3\.5–4\.0 miles/);
  assert.doesNotMatch(html, /1\.61155|1\.58333|1\.66667/);
  assert.equal(JSON.stringify(result), before);
});

test('continuous easy duration is readable and legacy coaching instructions stay intact', () => {
  const easy = workoutDetailsMarkup({ structure: [{ ...phase('warmup', 1468), instruction: 'Easy aerobic running · 24.4667 min' }] });
  assert.match(easy, /Easy aerobic running · 24:28/);
  assert.doesNotMatch(easy, /24\.4667/);
  for (const instruction of ['Run 8 × 1 minute with 90 seconds of easy jogging.', 'Gradually increase through the second half, finishing in Z3.', 'Run 18 minutes at threshold effort.', 'Run 8 × 45 seconds uphill.', 'Add surges using landmarks.']) {
    assert.ok(workoutDetailsMarkup({ structure: [{ instruction, target_zones: [] }] }).includes(instruction));
  }
  assert.match(workoutDetailsMarkup({ structure: [{ instruction: '<script>', target_zones: [] }] }), /&lt;script&gt;/);
  assert.equal(workoutDetailsMarkup({ structure: [] }), '');
});

test('unexpected phase sequences remain in order instead of being falsely grouped', () => {
  const result = { structure: [phase('warmup', 600), phase('work', 60), phase('work', 120), phase('cooldown', 600)] };
  const html = workoutDetailsMarkup(result);
  assert.equal((html.match(/<li>/g) ?? []).length, 4);
  assert.doesNotMatch(html, /×/);
});

test('distance-based display uses metres for efforts and miles for easy volume, without changing timings', () => {
  const meters = (step, distance) => ({ ...step, prescription_basis: 'distance', distance_miles: distance / 1609.344 });
  const result = { planning_role: 'quality', distance_target_miles: 3.8,
    structure: [{ ...phase('warmup', 600), prescription_basis: 'distance', distance_miles: .8 },
      meters(phase('work', 105), 419), meters(phase('recovery', 96.693), 210),
      meters(phase('work', 95), 388), meters(phase('recovery', 97.962), 212),
      meters(phase('work', 110), 409), { ...phase('cooldown', 600), prescription_basis: 'distance', distance_miles: .8 }] };
  const before = JSON.stringify(result);
  const html = workoutDetailsMarkup(result);
  assert.match(html, /Easy warm-up · ~0\.8 miles/);
  assert.match(html, /3 × ~400 m at controlled hard effort, with ~200 m of easy jogging/);
  assert.match(html, /Easy cool-down · ~0\.8 miles/);
  assert.match(html, /Segment distances are approximate/);
  assert.doesNotMatch(html, /10 min|1:45|1:35|Exact timings/);
  assert.equal(JSON.stringify(result), before);
});

test('easy run details use the committed total distance', () => {
  const html = workoutDetailsMarkup({ structure: [{ ...phase('warmup', 1468),
    instruction: 'Easy aerobic running · 24.4667 min', prescription_basis: 'distance', distance_miles: 2.3 }] });
  assert.match(html, /Easy aerobic running · 2\.3 miles/);
  assert.doesNotMatch(html, /24:28|min|approximate/);
});

test('timed work keeps minutes even when distance estimates are available', () => {
  const step = (kind, seconds) => ({ ...phase(kind, seconds), prescription_basis: 'time', distance_miles: .6 });
  const result = { planning_role: 'quality', structure: [step('warmup',600),
    step('work',300),step('recovery',120),step('work',300),step('recovery',120),
    step('work',300),step('cooldown',600)] };
  const html = workoutDetailsMarkup(result);
  assert.match(html, /3 × 5 min at controlled hard effort, with 2 min of easy jogging/);
  assert.doesNotMatch(html, / m |miles|approximate/);
});

test('distance efforts can have explicitly timed recoveries', () => {
  const work = { ...phase('work',105), prescription_basis: 'distance', distance_miles: 400/1609.344 };
  const recovery = { ...phase('recovery',90), prescription_basis: 'time', distance_miles: 200/1609.344 };
  const html = workoutDetailsMarkup({ structure: [phase('warmup',600),work,recovery,work,phase('cooldown',600)] });
  assert.match(html, /2 × ~400 m at controlled hard effort, with 1:30 of easy jogging/);
  assert.doesNotMatch(html, /200 m/);
});

test('unclassified distance estimates cannot silently convert a timed prescription', () => {
  const step = (kind, seconds) => ({ ...phase(kind, seconds), distance_miles: .5 });
  const html = workoutDetailsMarkup({ structure: [step('warmup',600),step('work',300),step('cooldown',600)] });
  assert.match(html, /1 × 5 min/);
  assert.doesNotMatch(html, /~.*miles|~.* m/);
});

test('explicit step structure wins over the broad workout family', () => {
  const effort = { ...phase('work',300), prescription_basis: 'distance', distance_miles: 800/1609.344 };
  const rest = { ...phase('recovery',120), prescription_basis: 'time', distance_miles: .1 };
  const html = workoutDetailsMarkup({ quality_session_type: 'long_intervals',
    structure: [phase('warmup',600),effort,rest,effort,phase('cooldown',600)] });
  assert.match(html, /2 × ~800 m at controlled hard effort, with 2 min of easy jogging/);
  assert.doesNotMatch(html, /× 5 min/);
});

test('explicit mixed block instructions keep distinct effort, units and order', () => {
  const structure = [
    { phase: 'warmup', instruction: '10 min · Warm up easily.', duration_minutes: 10, prescription_basis: 'time' },
    { phase: 'work', instruction: '10 min · At steady effort.', duration_minutes: 10, prescription_basis: 'time' },
    { phase: 'work', instruction: '400 m · At threshold effort. 2 min · Jog easily after each effort.',
      duration_minutes: 4, prescription_basis: 'distance', distance_miles: .4 },
    { phase: 'work', instruction: '200 m · At fast effort, not sprinting.',
      duration_minutes: 1, prescription_basis: 'distance', distance_miles: 200 / 1609.344 },
    { phase: 'cooldown', instruction: '10 min · Cool down easily.', duration_minutes: 10, prescription_basis: 'time' },
  ];
  const result = { structure, quality_session_type: 'mixed_quality' };
  const before = JSON.stringify(result);
  const html = workoutDetailsMarkup(result);
  structure.forEach(step => assert.ok(html.includes(step.instruction)));
  assert.ok(html.indexOf('steady effort') < html.indexOf('threshold effort'));
  assert.ok(html.indexOf('threshold effort') < html.indexOf('fast effort'));
  assert.doesNotMatch(html, /at controlled hard effort/);
  assert.equal(JSON.stringify(result), before);
});

test('repeated threshold commands are not rewritten as continuous or generic intervals', () => {
  const instruction = '3 × 6 min · At threshold effort. 2 min · Jog easily between efforts.';
  const html = workoutDetailsMarkup({ quality_session_type: 'threshold', structure: [
    { phase: 'warmup', instruction: '10 min · Warm up easily.', duration_minutes: 10 },
    { phase: 'work', instruction, duration_minutes: 22, repetitions: 3,
      prescription_basis: 'time', target_zones: ['threshold effort'] },
    { phase: 'cooldown', instruction: '10 min · Cool down easily.', duration_minutes: 10 },
  ] });
  assert.ok(html.includes(instruction));
  assert.equal((html.match(/<li>/g) ?? []).length, 3);
  assert.doesNotMatch(html, /continuous|controlled hard effort/);
});
