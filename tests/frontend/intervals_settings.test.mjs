import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bindIntervalsSettings, intervalsSettingsCard } from '../../src/run_analysis/web/static/app/intervals-settings.js';

function form() {
  const nodes = Object.fromEntries(['intervals-api-key','intervals-status','save-intervals-key','check-intervals-connection','remove-intervals-key'].map(id => [id, {
    value: '', disabled: false, textContent: '', listeners: {}, clicks: 0,
    addEventListener(type, callback) { this.listeners[type] = callback; },
    click() { this.clicks++; },
  }]));
  return { nodes, querySelector(selector) { return nodes[selector.slice(1)]; } };
}
const state = configured => ({ configured, available: true, automatic_import: false });
const response = payload => ({ ok: true, json: async () => payload });

test('card masks key and excludes it from the general settings FormData', () => {
  const card = intervalsSettingsCard();
  assert.match(card, /type="password"/);
  assert.doesNotMatch(card, /name=|type="submit"|localStorage|Keychain/);
  assert.match(card, /not encrypted/);
  assert.match(card, /Automatic imports are not enabled/);
});

test('initial connection is usable with no history or key', async () => {
  globalThis.fetch = async () => response(state(false));
  const f = form(); await bindIntervalsSettings(f);
  assert.equal(f.nodes['intervals-api-key'].disabled, false);
  assert.equal(f.nodes['save-intervals-key'].disabled, true);
  assert.equal(f.nodes['check-intervals-connection'].disabled, true);
  f.nodes['intervals-api-key'].value = 'test-secret';
  f.nodes['intervals-api-key'].listeners.input();
  assert.equal(f.nodes['save-intervals-key'].disabled, false);
});

test('save uses separate endpoint, clears key, and enables connection check', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push([url, options]); return response(state(!!options)); };
  const f = form(); await bindIntervalsSettings(f);
  f.nodes['intervals-api-key'].value = 'test-secret';
  await f.nodes['save-intervals-key'].listeners.click();
  assert.equal(f.nodes['intervals-api-key'].value, '');
  assert.equal(f.nodes['check-intervals-connection'].disabled, false);
  assert.equal(calls[1][0], '/api/connections/intervals');
  assert.equal(calls[1][1].method, 'PUT');
  assert.equal(calls[1][1].headers['X-Run-Analysis-Local'], '1');
  assert.deepEqual(JSON.parse(calls[1][1].body), { api_key: 'test-secret' });
  assert.doesNotMatch(f.nodes['intervals-status'].textContent, /test-secret/);
  assert.equal(calls.length, 2);
});

test('empty connection check is success without import', async () => {
  globalThis.fetch = async (url, options) => response(options ? { connected: true, recent_activity_count: 0 } : state(true));
  const f = form(); await bindIntervalsSettings(f);
  await f.nodes['check-intervals-connection'].listeners.click();
  assert.match(f.nodes['intervals-status'].textContent, /^Connected\./);
  assert.match(f.nodes['intervals-status'].textContent, /Nothing imported/);
  assert.equal(f.nodes['check-intervals-connection'].disabled, false);
});

test('Enter in key field cannot submit/recalculate other settings', async () => {
  globalThis.fetch = async () => response(state(false));
  const f = form(); await bindIntervalsSettings(f);
  f.nodes['intervals-api-key'].value = 'test-secret';
  f.nodes['intervals-api-key'].listeners.input();
  let prevented = false;
  f.nodes['intervals-api-key'].listeners.keydown({ key: 'Enter', preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(f.nodes['save-intervals-key'].clicks, 1);
});

test('failed load gives clear error without disrupting general settings', async () => {
  globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) });
  const f = form(); await bindIntervalsSettings(f);
  assert.match(f.nodes['intervals-status'].textContent, /Connection settings unavailable/);
  assert.equal(f.nodes['intervals-api-key'].disabled, true);
});
