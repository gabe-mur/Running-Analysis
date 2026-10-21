import { test } from 'node:test';
import assert from 'node:assert/strict';

// Exercise the actual view and submit handler without writing live settings.
const nodes = new Map();
const node = selector => {
  if (!nodes.has(selector)) nodes.set(selector, {
    value: '', textContent: '', disabled: false, hidden: false, label: {}, listeners: {},
    addEventListener(type, handler) { this.listeners[type] = handler; },
    closest() { return this.label; }, reportValidity() { return true; },
  });
  return nodes.get(selector);
};
const form = {
  listeners: {},
  addEventListener(type, handler) { this.listeners[type] = handler; },
  querySelector(selector) { return selector === '[data-race-outcome]' ? null : node(selector); },
  querySelectorAll(selector) {
    if (selector === '[data-method]' || selector === 'input[name="method"]' || selector === '[data-zone-step]') return [];
    return selector === 'button[type=submit]' ? [node('submit-top'), node('submit-bottom')]
      : [node('#settings-status'), node('#settings-status-top')];
  },
};
const page = {
  innerHTML: '',
  querySelector(selector) { return selector === '#settings-form' ? form : node(selector); },
};
globalThis.document = {querySelector: () => page};
const {renderSettings} = await import('../../src/run_analysis/web/static/app/views/settings.js');

const settings = {
  max_hr: 194, resting_hr: 46, target_hr: 148,
  zones: Object.fromEntries(['z1','z2','z3','z4','z5'].map((n,i) => [n, {
    minimum_bpm: 128+i*13, maximum_bpm: 140+i*13,
  }])),
  profile: {birth_date: '1990-01-01', sex: 'male', weight_lb: 150, height_in: 70, max_hr_source: 'measured'},
  reference_temperature_f: 55, reference_dewpoint_f: 45, reference_wind_mph: 3,
  reference_grade_percent: 0, reference_within_run_minutes: 20,
  weather_privacy_radius_km: 5, historical_weather_enabled: true, forecast_weather_enabled: false,
  default_fitness_window: 28, available_fitness_windows: [14,28,365],
  moving_time: {minimum_running_speed_mps: .8, maximum_interval_seconds: 30},
  race_preparation: {distance: 'general_fitness', race_date: null, preparation_mode: 'balanced'},
  coaching: {
    quality_focus: 'variety',
    quality_recency_reference_days: 7,
    quality_sessions: {short_intervals: true, long_intervals: true, threshold: true,
      progression: true, fartlek: false, hill_repeats: false},
    strength_training: {enabled: true, leg_frequency_days: 8, push_pull_frequency_days: 8,
      muscle_groups: {push: ['upper pecs','triceps'], pull: ['lats','biceps'], legs: ['quads','calves']}},
  },
  planner: {status: 'adopted', preferences_revision: 'existing-owner', long_target_miles: 8.9,
    long_goal_miles: 17, run_frequency: 'balanced', minimum_running_minutes: 10,
    penalty_free_running_minutes: 25, demand_memory_days: 2, demand_delivery_exchange: .01,
    frequency_preference_fraction: .02},
};
let calls = [], values;
async function render(planner = settings.planner, setupComplete = true) {
  nodes.clear(); calls = [];
  node('[name="race_distance"]').value = 'general_fitness';
  node('[name="race_mode"]').value = 'balanced';
  globalThis.fetch = async (url, options) => {
    calls.push({url,options});
    const data = url === '/api/settings' ? {...settings, planner, recalculation: []}
      : url === '/api/setup' ? {complete: setupComplete, steps: Array.from({length:8}, (_, i) => ({complete: setupComplete || i > 0}))}
      : url === '/api/connections/intervals' ? {configured: true, available: true}
      : url === '/api/setup/comparison-hr' ? {recommended_bpm: 148, rationale: 'Chosen from recent easy running.'}
      : url === '/api/setup/zone-preview' ? {zones:settings.zones,
        comparison_hr:{recommended_bpm:148,rationale:'Chosen from recent easy running.'},watch_recorded_at:null}
      : {status: 'available', expected_revision: 'existing-goal', race_activation_available: true};
    return {ok: true, json: async () => data};
  };
  await renderSettings();
  for (const key of ['max_hr','resting_hr']) node(`[name="${key}"]`).value = String(settings[key]);
  await new Promise(resolve => setImmediate(resolve));
}

test('everyday settings are grouped in order, technical controls remain in one form', async () => {
  await render();
  const html = page.innerHTML;
  const sections = ['settings-training','settings-heart-rate','settings-connections','settings-display'];
  const offsets = sections.map(id => html.indexOf(`id="${id}"`));
  assert.ok(offsets.every((n,i) => n >= 0 && (!i || n > offsets[i-1])));
  assert.ok(html.indexOf('advanced-settings') > offsets.at(-1));
  assert.equal((html.match(/<form\b/g) || []).length, 1);
  assert.equal((html.match(/<\/form>/g) || []).length, 1);
  const names = Array.from(html.matchAll(/\bname="([^"]+)"/g), m => m[1]).filter(name => name !== 'method');
  assert.equal(new Set(names).size, names.length, 'No duplicated named controls');
  for (const name of ['planner_long_goal','planner_frequency','planner_long_day','race_distance',
    'strength_push','profile_birth_date','weather_privacy_radius_km','reference_temperature_f']) {
    assert.ok(names.includes(name), `Missing control: ${name}`);
  }
  assert.match(html, /<details class="settings-inline-details"><summary>Muscle groups/);
  const weatherPanel = html.match(/<fieldset><legend>Weather<\/legend>([\s\S]*?)<\/fieldset>/)[1];
  assert.match(weatherPanel, /name="weather_privacy_radius_km"/);
  assert.doesNotMatch(weatherPanel, /<details\b|hidden/);
  assert.match(html, /<legend>About you<\/legend>/);
  assert.ok(html.indexOf('<legend>About you') < html.indexOf('<legend>Heart-rate zones'));
  assert.ok(html.indexOf('name="method"') < html.indexOf('name="max_hr"'));
  assert.equal((html.match(/<legend>Heart-rate zones<\/legend>/g) || []).length,1);
  assert.doesNotMatch(html, /<legend>Heart rate<\/legend>|<legend>Running zones<\/legend>/);
  assert.match(html, /name="profile_birth_date"[^>]*required/);
  assert.doesNotMatch(html, /name="target_hr"|Pace comparison|Comparison HR/);
  assert.match(html, /Heart-rate reserve \(HRR\)/);
  assert.doesNotMatch(html, /VO₂ profile <span>Optional/);
  assert.doesNotMatch(html, /Everything is confirmed|evidence migration|handful of settings/);
  assert.match(html, /Review setup/);
  assert.match(html, /You’re set up · 8 of 8 essential settings confirmed/);
  assert.match(html, /Change them at any time\./);
  assert.doesNotMatch(html, /Setup complete/);
  assert.match(html, /<article class="wide-card setup-entry /);
  assert.match(html, /<a class="primary-button" href="#setup">Review setup<\/a>/);
  assert.doesNotMatch(html, /settings-setup/);
  assert.ok(calls.every(c => !c.options?.method ||
    (c.url==='/api/setup/zone-preview' && c.options.method==='POST')),
    'Rendering may only read settings and preview zones, never save or refresh plans');
});

test('focus edits submit the new choice while preserving the complete payload and profile provenance', async () => {
  await render();
  values = new Map();
  const set = (name,value) => values.set(name,String(value));
  set('max_hr',settings.max_hr); set('resting_hr',settings.resting_hr);
  for (const key of ['reference_temperature_f','reference_dewpoint_f','reference_wind_mph',
    'reference_grade_percent','reference_within_run_minutes','weather_privacy_radius_km','default_fitness_window']) {
    set(key,settings[key]);
  }
  for (const [name,zone] of Object.entries(settings.zones)) {
    set(`${name}_min`,zone.minimum_bpm); set(`${name}_max`,zone.maximum_bpm);
  }
  for (const [key,value] of Object.entries(settings.profile)) set(`profile_${key}`,value);
  set('historical_weather_enabled','on');
  set('quality_focus','vo2max');
  for (const [name,enabled] of Object.entries(settings.coaching.quality_sessions)) if (enabled) set(`quality_${name}`,'on');
  const strength = settings.coaching.strength_training;
  set('strength_enabled','on'); set('strength_leg_days',8); set('strength_upper_days',8);
  for (const [name,groups] of Object.entries(strength.muscle_groups)) set(`strength_${name}`,groups.join(', '));
  // Hidden and disabled controls follow the same FormData contract as before.
  const NativeFormData = globalThis.FormData;
  globalThis.FormData = class { get(name) { return values.get(name) ?? null; } };
  try {
    await form.listeners.submit({preventDefault() {}});
  } finally { globalThis.FormData = NativeFormData; }
  const writes = calls.filter(c => c.options?.method && c.url !== '/api/setup/zone-preview');
  assert.equal(writes.length,1);
  assert.equal(writes[0].url,'/api/settings');
  assert.equal(writes[0].options.method,'PATCH');
  const payload = JSON.parse(writes[0].options.body);
  for (const key of ['max_hr','resting_hr','zones','profile','moving_time','coaching',
    'historical_weather_enabled','forecast_weather_enabled','race_preparation',
    'weather_privacy_radius_km','reference_temperature_f','reference_dewpoint_f',
    'reference_wind_mph','reference_grade_percent','reference_within_run_minutes','default_fitness_window']) {
    const expected = key === 'coaching' ? {...settings.coaching, quality_focus: 'vo2max'} : settings[key];
    assert.deepEqual(payload[key],expected,key);
  }
  assert.equal(payload.planner_preferences,undefined, 'Preference save remains separate');
  assert.equal(payload.target_hr,undefined, 'Comparison HR is derived by the server, not a form field');
  assert.equal(payload.api_key,undefined, 'Credential save remains separate');
});

test('legacy controls and incomplete setup remain available without invented planner values', async () => {
  await render(null,false);
  assert.match(page.innerHTML,/1 setup step remaining/);
  assert.match(page.innerHTML,/Finish setup/);
  assert.match(page.innerHTML, /setup-entry pending/);
  assert.match(page.innerHTML, /<a class="primary-button" href="#setup">Finish setup<\/a>/);
  assert.match(page.innerHTML,/name="coaching_quality_recency_reference_days" value="7"/);
  assert.doesNotMatch(page.innerHTML,/name="planner_long_goal"|name="planner_frequency"/);
  assert.doesNotMatch(page.innerHTML,/name="max_hr"[^>]*disabled/);
  assert.doesNotMatch(page.innerHTML,/id="reset-advanced" disabled/);
});
