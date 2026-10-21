import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setupDeviceCard } from '../../src/run_analysis/web/static/app/setup-device.js';
import { intervalsSettingsCard } from '../../src/run_analysis/web/static/app/intervals-settings.js';

test('equipment guidance requires workout HR and GPS but keeps a chest strap optional', () => {
  const html = setupDeviceCard();
  assert.match(html, /GPS-capable watch or fitness tracker/);
  assert.match(html, /heart rate throughout your running workouts/);
  assert.match(html, /chest strap/);
  assert.match(html, /It’s optional/);
  assert.match(html, /Pair it with the watch or app recording your run/);
  assert.match(html, /VO₂max and HRV aren’t required/);
  assert.doesNotMatch(html, /<input|<button/);
});

test('compatibility guidance distinguishes a bridge from verified ingestion', () => {
  const html = setupDeviceCard();
  assert.match(html, /FIT, TCX or GPX activity files/);
  assert.match(html, /Garmin ZIP/);
  assert.match(html, /Sync requires a downloadable original FIT/);
  assert.match(html, /Apple Health bridge/);
  assert.match(html, /verify an Apple Watch file end to end/);
  assert.match(html, /workouts, heart rate and workout routes/);
  assert.doesNotMatch(html, /Strava|export_tcx|Export Original or Export GPX/);
  assert.match(html, /through an iPhone app, not the Intervals.icu website alone/);
  assert.doesNotMatch(html, /HealthFit/);
  assert.match(intervalsSettingsCard(), /downloadable original FIT files/);
  assert.match(intervalsSettingsCard(), /href="#setup">Device connection guide/);
  assert.doesNotMatch(intervalsSettingsCard(), /Import Garmin runs/);
});

test('equipment guidance stays visible before import and the conditional setup form', async () => {
  const source = await readFile(new URL('../../src/run_analysis/web/static/app/views/setup.js', import.meta.url), 'utf8');
  assert.ok(source.indexOf('${setupDeviceCard()}') < source.indexOf('${blocked ? `<article'));
  assert.ok(source.indexOf('${setupDeviceCard()}') < source.indexOf('<form id="setup-form"'));
});
