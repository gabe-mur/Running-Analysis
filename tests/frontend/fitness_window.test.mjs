import { test } from 'node:test';
import assert from 'node:assert/strict';
import { primeSettings } from '../../src/run_analysis/web/static/app/api.js';
import { fitnessWindow, selectFitnessWindow } from '../../src/run_analysis/web/static/app/fitness-window.js';

test('both views read the same settings default and explicit session selection', async () => {
  primeSettings({ default_fitness_window: 42 });
  assert.equal(await fitnessWindow(), 42);
  primeSettings({ default_fitness_window: 56 });
  assert.equal(await fitnessWindow(), 56);
  primeSettings({ default_fitness_window: 5 });
  assert.equal(await fitnessWindow(), 28);
  selectFitnessWindow(90);
  assert.equal(await fitnessWindow(), 90);
  primeSettings({ default_fitness_window: 28 });
  assert.equal(await fitnessWindow(), 90);
  assert.throws(() => selectFitnessWindow(5), /Unsupported/);
  assert.equal(await fitnessWindow(), 90);
});
