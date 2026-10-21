import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bindSync } from '../../src/run_analysis/web/static/app/sync.js';

function controls() {
  const state = { busy: false, refreshes: 0, stages: [], summary: { textContent: '' } };
  const button = { listeners: {}, attrs: {}, addEventListener(name, fn) { this.listeners[name] = fn; }, setAttribute(name, value) { this.attrs[name] = value; } };
  bindSync({ button, summary: state.summary, begin() { if (state.busy) return false; state.busy = true; return true; },
    end() { state.busy = false; }, showStages(stages) { state.stages = stages; }, refresh: async () => { state.refreshes++; } });
  return { state, button, click: () => button.listeners.click() };
}

const response = payload => ({ ok: true, json: async () => payload });

test('icon sync uses protected POST and avoids page refresh for no-op', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push([url, options]); return response({ new_runs: 0, upload: null }); };
  const { state, button, click } = controls();
  await click();
  assert.equal(calls[0][0], '/api/connections/intervals/sync');
  assert.equal(calls[0][1].method, 'POST');
  assert.equal(calls[0][1].headers['X-Run-Analysis-Local'], '1');
  assert.equal(state.summary.textContent, 'Up to date. No new runs.');
  assert.equal(state.refreshes, 0);
  assert.equal(state.busy, false);
  assert.equal(button.attrs['aria-busy'], 'false');
});

test('new or enriched evidence refreshes once and shows pipeline progress', async () => {
  for (const newRuns of [0, 1]) {
    globalThis.fetch = async () => response({ new_runs: newRuns, upload: { stages: [{ name: 'process', status: 'complete' }] } });
    const { state, click } = controls();
    await click();
    assert.equal(state.refreshes, 1);
    assert.equal(state.summary.textContent, newRuns ? 'Synced 1 new run.' : 'Run data updated.');
  }
});

test('partial errors are visible, not falsely up to date', async () => {
  globalThis.fetch = async () => response({ new_runs: 0, errors: ['Original FIT missing.'], remaining_runs: 3 });
  const { state, click } = controls();
  await click();
  assert.match(state.summary.textContent, /Sync incomplete/);
  assert.match(state.summary.textContent, /3 more to check/);
  assert.equal(state.stages[0].detail, 'Original FIT missing.');
});

test('failed parsing and model stages do not get a clean success message', async () => {
  globalThis.fetch = async () => response({ new_runs: 1, upload: {
    files: [{ status: 'failed', error: 'Unable to parse.' }],
    stages: [{ name: 'process', status: 'complete' }, { name: 'model', status: 'failed', detail: 'Unable to fit.' }],
  } });
  const { state, click } = controls();
  await click();
  assert.match(state.summary.textContent, /Some steps need attention/);
  assert.equal(state.stages.at(-1).detail, 'Unable to parse.');
});

test('busy guard prevents overlapping requests and recovers from network error', async () => {
  let reject, requests = 0;
  globalThis.fetch = () => { requests++; return new Promise((_, fail) => { reject = fail; }); };
  const { state, button, click } = controls();
  const first = click();
  assert.equal(button.attrs['aria-busy'], 'true');
  await click();
  assert.equal(requests, 1);
  reject(new Error('Offline.'));
  await first;
  assert.equal(state.busy, false);
  assert.equal(state.summary.textContent, 'Sync failed: Offline.');
});
