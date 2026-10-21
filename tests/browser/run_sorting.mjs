// Offline UI check: all requests intercepted, no live planner/data changes.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
const { chromium } = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/', import.meta.url);
const requests = [];
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/runs') {
      requests.push(Object.fromEntries(url.searchParams));
      return route.fulfill({ json: [{ activity_id: 42, start_time: '2026-10-11T10:00:00Z', workout_type: 'easy', distance_miles: 4, moving_minutes: 40, moving_pace_min_mile: 10, average_hr_bpm: 145, maximum_hr_bpm: 155, temperature_f: 60, gps_quality: 'gps_complete', assessment_label: 'Easy run', health_tag: 'normal', data_quality: 'good' }] });
    }
    if (url.pathname.startsWith('/assets/')) return route.fulfill({
      contentType: url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript',
      body: await readFile(new URL(url.pathname.slice(8), assets)),
    });
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><link rel="stylesheet" href="/assets/styles.css"><main id="app-view"></main><script type="module">import {renderRuns} from "/assets/app/views/runs.js"; await renderRuns();</script>' });
    throw new Error(`Unexpected request ${url.pathname}`);
  });
  await page.goto('https://runs.test/');
  await page.locator('[data-sort="date"]').waitFor();
  assert.equal(requests[0].sort_by, 'date');
  assert.equal(requests[0].sort_order, 'desc');
  const row = page.locator('.run-row').first();
  const rowSize = await row.boundingBox();
  const idle = await row.evaluate(el => ({ border: getComputedStyle(el).borderBottomColor, accentOpacity: getComputedStyle(el, '::before').opacity }));
  assert.equal(idle.accentOpacity, '0');
  await row.hover();
  await page.waitForTimeout(180);
  const hover = await row.evaluate(el => ({ background: getComputedStyle(el).backgroundImage, border: getComputedStyle(el).borderBottomColor }));
  assert.match(hover.background, /radial-gradient\(120% 90% at 100% 0(?:px|%)/);
  assert.match(hover.background, /rgba\(96, 124, 178, 0\.3\)/);
  assert.equal(hover.border, idle.border);
  const accent = await row.evaluate(el => {
    const style = getComputedStyle(el, '::before');
    return { background: style.backgroundImage, opacity: style.opacity, top: style.top, height: style.height };
  });
  assert.match(accent.background, /linear-gradient\(90deg, rgb\(111, 143, 196\).*70%\)/);
  assert.equal(accent.opacity, '1');
  assert.equal(accent.top, '0px');
  assert.equal(accent.height, '2px');
  assert.deepEqual(await row.boundingBox(), rowSize);
  await page.screenshot({ path: '/tmp/run-row-corner-hover.png' });
  await row.focus();
  await page.mouse.move(0, 0);
  assert.equal(await row.evaluate(el => getComputedStyle(el).backgroundImage), hover.background);
  assert.equal(await row.evaluate(el => getComputedStyle(el, '::before').opacity), '1');
  const alignment = await page.locator('#run-filters').evaluate(form => ({
    right: form.getBoundingClientRect().right,
    applyRight: form.querySelector('button').getBoundingClientRect().right,
    padding: parseFloat(getComputedStyle(form).paddingRight),
  }));
  assert.ok(Math.abs(alignment.right - alignment.applyRight - alignment.padding - 1) < 1);
  for (const [key, order] of [['distance','desc'], ['pace','asc'], ['standardized','asc'], ['temperature','desc'], ['assessment','asc'], ['data','asc'], ['date','desc']]) {
    await page.locator(`[data-sort="${key}"]`).click();
    await page.locator(`[data-sort="${key}"][aria-pressed="true"]`).waitFor();
    assert.equal(requests.at(-1).sort_by, key);
    assert.equal(requests.at(-1).sort_order, order);
    assert.equal(await page.locator('[name="sort_by"]').inputValue(), key);
  }
  await page.locator('[data-sort="date"]').click();
  await page.locator('[data-sort="date"][aria-label*="ascending"]').waitFor();
  assert.equal(requests.at(-1).sort_order, 'asc');
  await page.locator('[data-sort="distance"]').hover();
  await page.waitForTimeout(180);
  assert.equal(await page.locator('[data-sort="distance"]').evaluate(el => getComputedStyle(el).borderColor), 'rgba(0, 0, 0, 0)');
  assert.equal(await page.locator('[data-sort="distance"]').evaluate(el => getComputedStyle(el).boxShadow), 'none');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[name="sort_by"]').selectOption('assessment');
  assert.equal(await page.locator('[name="sort_order"] option[value="asc"]').textContent(), 'A–Z');
  await page.locator('[name="sort_order"]').selectOption('asc');
  await page.locator('#run-filters button').click();
  await page.locator('[data-sort="assessment"][aria-pressed="true"]').waitFor({ state: 'attached' });
  assert.equal(requests.at(-1).sort_by, 'assessment');
  assert.deepEqual(errors, []);
  console.log('Apply right alignment, seven column toggles, default date order and mobile dropdown sorting passed.');
} finally { await browser.close(); }
