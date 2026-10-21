import { test } from 'node:test';
import assert from 'node:assert/strict';
import { refreshAfterTransfer } from '../../src/run_analysis/web/static/app/transfer-refresh.js';

test('sync completing on Weekly Plan does not start a search; Update explicitly does', async () => {
  let refreshes = 0, click;
  const view = { innerHTML: '', querySelector: () => ({ addEventListener: (_, fn) => { click = fn; } }) };
  await refreshAfterTransfer(async () => { refreshes++; }, view, '#next-run');
  assert.equal(refreshes, 0);
  assert.match(view.innerHTML, /Your plan needs updating/);
  assert.match(view.innerHTML, /Update Weekly Plan/);
  await click();
  assert.equal(refreshes, 1);
});

test('other pages refresh their analysis normally after a transfer', async () => {
  let refreshes = 0;
  const view = { innerHTML: 'Existing page' };
  for (const hash of ['#run/42', '#runs', '#dashboard', '#progress']) {
    await refreshAfterTransfer(async () => { refreshes++; }, view, hash);
  }
  assert.equal(refreshes, 4);
  assert.equal(view.innerHTML, 'Existing page');
});

test('unchanged uploads neither invalidate the displayed plan nor start a search', async () => {
  const view = { innerHTML: 'Saved weekly plan' };
  await refreshAfterTransfer(() => assert.fail('No-op transfer refreshed the page'), view, '#next-run', false);
  assert.equal(view.innerHTML, 'Saved weekly plan');
});
