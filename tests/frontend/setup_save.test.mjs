import { test } from 'node:test';
import assert from 'node:assert/strict';
import { saveSetupSettings } from '../../src/run_analysis/web/static/app/setup-save.js';

const settings = () => ({ planner: { status: 'adopted', preferences_revision: 'old' },
  setup: { confirmed_steps: ['profile'] } });
const payload = { max_hr: 194, zones: { z2: { minimum_bpm: 141, maximum_bpm: 153 } },
  setup: { confirmed_steps: ['heart_rate', 'zones'], zone_method: 'device' } };
const preferences = { long_run_goal_miles: 17, run_frequency: 'balanced', long_run_weekday: 5 };

test('one setup action saves preferences separately, then confirms progression with calibration', async () => {
  const calls = [], current = settings();
  const result = await saveSetupSettings(current, payload, preferences, async (url, options) => {
    calls.push(JSON.parse(options.body));
    assert.equal(url, '/api/settings');
    assert.equal(options.method, 'PATCH');
    return { planner: { status: 'adopted', preferences_revision: 'new' }, target_hr: 148 };
  });
  assert.deepEqual(calls[0], { planner_preferences: { ...preferences, expected_revision: 'old' } });
  assert.equal(calls[1].planner_preferences, undefined);
  assert.equal(calls[1].max_hr, 194);
  assert.deepEqual(calls[1].zones, payload.zones);
  assert.deepEqual(calls[1].setup.confirmed_steps, ['profile', 'heart_rate', 'zones', 'progression']);
  assert.equal(current.planner.preferences_revision, 'new');
  assert.equal(result.target_hr, 148);
  assert.deepEqual(payload.setup.confirmed_steps, ['heart_rate', 'zones']);
});

test('preference failure cannot mark setup complete or save calibration', async () => {
  let calls = 0;
  await assert.rejects(saveSetupSettings(settings(), payload, preferences, async () => {
    calls++; throw new Error('Stale preference revision');
  }), /Stale preference revision/);
  assert.equal(calls, 1);
});

test('partial failure is explicit and retains revision for retry', async () => {
  let calls = 0;
  const current = settings();
  await assert.rejects(saveSetupSettings(current, payload, preferences, async () => {
    if (++calls === 1) return { planner: { status: 'adopted', preferences_revision: 'new' } };
    throw new Error('Calibration change blocked');
  }), /Running preferences are saved, but setup is not complete: Calibration change blocked/);
  assert.equal(current.planner.preferences_revision, 'new');
  assert.deepEqual(current.setup.confirmed_steps, ['profile']);
});

test('legacy or not-yet-adopted setup never invents or confirms planner preferences', async () => {
  for (const planner of [null, { status: 'review_required' }]) {
    const calls = [];
    await saveSetupSettings({ planner }, payload, preferences, async (_, options) => {
      calls.push(JSON.parse(options.body)); return {};
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].planner_preferences, undefined);
    assert.ok(!calls[0].setup.confirmed_steps.includes('progression'));
  }
});

test('setup activates the goal and only then confirms it, using the current revision', async () => {
  const current = settings(), calls = [];
  const input = {...payload, race_preparation: {distance: 'general_fitness'},
    setup: {...payload.setup, confirmed_steps: ['heart_rate','zones','goal']}};
  const result = await saveSetupSettings(current,input,null,async (url,options) => {
    const body = options?.body ? JSON.parse(options.body) : null;
    calls.push({url,body});
    if (url.endsWith('/status')) return {status:'available',expected_revision:'current-goal'};
    if (url.endsWith('/apply')) return {planner: {status:'adopted',goal_revision:'applied'},detail:'Applied.'};
    return {...current, setup:body.setup,recalculation:[]};
  },{applyTrainingGoal:true});
  assert.ok(!calls[0].body.setup.confirmed_steps.includes('goal'));
  assert.deepEqual(calls[2].body,{goal:input.race_preparation,expected_revision:'current-goal'});
  assert.ok(calls[3].body.setup.confirmed_steps.includes('goal'));
  assert.ok(result.setup.confirmed_steps.includes('goal'));
  assert.equal(result.recalculation.at(-1).status,'complete');
  assert.equal(current.planner.goal_revision,'applied');
});

test('a rejected activation retains saved calibration but does not confirm the goal', async () => {
  const current = settings();
  const input = {...payload,race_preparation:{distance:'marathon'},
    setup:{...payload.setup,confirmed_steps:['profile','goal']}};
  await assert.rejects(saveSetupSettings(current,input,null,async (url,options) => {
    if (url.endsWith('/status')) return {status:'available',expected_revision:'current'};
    if (url.endsWith('/apply')) throw new Error('Preparation is not feasible');
    return {...current,setup:JSON.parse(options.body).setup};
  },{applyTrainingGoal:true}),/Personal details and HR settings are saved, but the training goal could not be applied/);
  assert.ok(current.setup.confirmed_steps.includes('profile'));
  assert.ok(!current.setup.confirmed_steps.includes('goal'));
});
