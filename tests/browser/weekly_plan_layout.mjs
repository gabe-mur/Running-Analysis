// Offline visual regression: real view/CSS, mocked API, no planner or server calls.
// Run with NODE_PATH pointing to the bundled Playwright node_modules.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
const { chromium } = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/', import.meta.url);
const strength = (session_type, muscle_groups) => ({ session_type, title: session_type, muscle_groups, rationale: '' });
const run = (title, distance, workout_type = 'easy') => ({
  title, workout_type, distance_target_miles: distance, readiness: 'ready',
  planned_for: '2026-10-09T07:00:00-04:00', time_recommendation_basis: 'weather',
  target_zones: ['Z1', 'Z2'], rule_trace: [], structure: [], reasons: [],
});
const schedule = {
  start_date: '2026-10-08', end_date: '2026-10-14', planner_status: 'ready',
  run_count: 4, target_run_count: 4, projected_distance_range_miles: [18, 19],
  target_distance_range_miles: [18, 19], target_evidence: { planning_mode: 'established', capacity_reference_miles: 18, rationale: 'Fixture.' },
  summary: 'Layout fixture, not a training recommendation.', trailing_days: [], days: [
    { date: '2026-10-08', strength_suggestion: strength('push', ['upper pecs', 'lower pecs', 'side delts', 'triceps', 'core/trunk']) },
    { date: '2026-10-09', recommendation: run('Controlled intervals', 3.8, 'intervals'), planned_at: '2026-10-09T07:00:00-04:00' },
    { date: '2026-10-10', recommendation: run('Easy aerobic run', 2.3), planned_at: '2026-10-10T07:00:00-04:00' },
    { date: '2026-10-11', strength_suggestion: strength('pull', ['lats', 'upper back', 'rear delts', 'biceps']) },
    { date: '2026-10-12', recommendation: run('Easy long run', 8.9), planned_at: '2026-10-12T07:00:00-04:00' },
    { date: '2026-10-13', strength_suggestion: strength('legs', ['quads', 'hamstrings', 'glutes', 'calves']) },
    { date: '2026-10-14', recommendation: run('Easy aerobic run', 3.6), planned_at: '2026-10-14T07:00:00-04:00' },
  ],
};
const responses = {
  '/api/current-status': { health_status: 'normal', health_report_current: true },
  '/api/weekly-schedule/latest': schedule,
  '/api/fitness-state': { recent_load: { trailing_7d: { distance_miles: 18 } }, running_days_28d: 16 },
  '/api/goal-progress': null,
  '/api/settings': { planner: null },
};
// A real warning must not push the run description below the strength routine.
Object.assign(schedule.days[1].recommendation, { readiness: 'caution', readiness_label: 'Weather caution', readiness_reason: 'Fixture heat.' });
schedule.trailing_days = Array.from({ length: 7 }, (_, index) => ({
  date: `2026-10-${String(index + 1).padStart(2, '0')}`,
  day_role: index % 2 ? 'rest_day' : 'run_day',
  activities: index % 2 ? [] : [{ activity_id: 100 + index, distance_miles: 4.2, workout_type: 'easy', health_tag: 'normal' }],
}));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ timezoneId: 'America/New_York' });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', async route => {
    const path = new URL(route.request().url()).pathname;
    if (Object.hasOwn(responses, path)) return route.fulfill({ json: responses[path] });
    if (path.startsWith('/assets/')) return route.fulfill({
      contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript',
      body: await readFile(new URL(path.slice('/assets/'.length), assets)),
    });
    if (path === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><link rel="stylesheet" href="/assets/styles.css"><main id="app-view"></main><script type="module">import {renderNextRun} from "/assets/app/views/next-run.js"; await renderNextRun();</script>' });
    throw new Error(`Unexpected request: ${path}`);
  });
  // A mocked loopback origin provides the secure-context crypto API used by
  // the current loader. The route handler intercepts every request.
  await page.goto('http://127.0.0.1/');
  await page.locator('.planned-week > article').last().waitFor();
  for (const width of [1000, 1280, 1500, 1920]) {
    await page.setViewportSize({ width, height: 1100 });
    const metrics = await page.locator('.planned-week > article').evaluateAll(cards => cards.map(card => {
      const title = card.querySelector(':scope > h2');
      const dose = card.querySelector(':scope > .prescription-extent');
      const description = card.querySelector('.run-description, .standalone-strength .strength-groups');
      const style = getComputedStyle(dose);
      const descriptionStyle = getComputedStyle(description);
      const firstLetter = document.createRange();
      firstLetter.setStart(description.firstChild, 0);
      firstLetter.setEnd(description.firstChild, 1);
      return { titleTop: title.getBoundingClientRect().top, doseTop: dose.getBoundingClientRect().top, label: dose.textContent, font: style.fontSize, weight: style.fontWeight,
        descriptionTop: description.getBoundingClientRect().top, textTop: firstLetter.getBoundingClientRect().top,
        descriptionFont: descriptionStyle.fontSize, descriptionWeight: descriptionStyle.fontWeight, descriptionColor: descriptionStyle.color };
    }));
    assert.ok(Math.max(...metrics.map(m => m.titleTop)) - Math.min(...metrics.map(m => m.titleTop)) < 1, `Title alignment at ${width}`);
    assert.ok(Math.max(...metrics.map(m => m.doseTop)) - Math.min(...metrics.map(m => m.doseTop)) < 1, `Dose alignment at ${width}`);
    assert.equal(new Set(metrics.map(m => m.font)).size, 1);
    assert.equal(new Set(metrics.map(m => m.weight)).size, 1);
    assert.ok(Math.max(...metrics.map(m => m.descriptionTop)) - Math.min(...metrics.map(m => m.descriptionTop)) < 1, `Description alignment at ${width}`);
    assert.ok(Math.max(...metrics.map(m => m.textTop)) - Math.min(...metrics.map(m => m.textTop)) < 1, `Description text alignment at ${width}`);
    assert.equal(new Set(metrics.map(m => m.descriptionFont)).size, 1);
    assert.ok(metrics.every(m => m.descriptionWeight === '400'));
    assert.equal(new Set(metrics.map(m => m.descriptionColor)).size, 1);
    console.log(JSON.stringify({ width, titleTop: metrics[0].titleTop, doseTop: metrics[0].doseTop, descriptionTop: metrics[0].descriptionTop, font: metrics[0].font, labels: metrics.map(m => m.label) }));
    if (width === 1500) await page.locator('.planned-week').screenshot({ path: '/tmp/weekly-plan-alignment.png' });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.locator('.planned-week').evaluate(el => getComputedStyle(el).gridTemplateColumns.split(' ').length), 1);
  await page.setViewportSize({ width: 1500, height: 1100 });
  const runCards = page.locator('.planned-week .scheduled-run-card');
  assert.equal(await runCards.count(), 4);
  assert.equal(await page.locator('.planned-week .rest-day.scheduled-run-card').count(), 0);
  const cardAppearance = el => {
    const style = getComputedStyle(el);
    return { background: style.backgroundImage, border: style.borderColor, shadow: style.boxShadow, transform: style.transform, rule: getComputedStyle(el, '::before').backgroundImage };
  };
  const summaryAppearance = await page.locator('.schedule-summary').evaluate(cardAppearance);
  const runAppearance = await runCards.first().evaluate(cardAppearance);
  assert.equal(runAppearance.background, summaryAppearance.background);
  assert.equal(runAppearance.rule, summaryAppearance.rule);
  await runCards.first().hover();
  await page.waitForTimeout(200);
  assert.deepEqual(await runCards.first().evaluate(cardAppearance), runAppearance, 'Run cards have no hover styling');
  const logged = page.locator('.logged-run-card').first();
  assert.equal(await logged.getAttribute('href'), '#run/100');
  assert.equal(await page.locator('.trailing-week .rest-day a').count(), 0);
  const accent = await logged.evaluate(el => ({
    tint: getComputedStyle(el).getPropertyValue('--card-tint'),
    rule: getComputedStyle(el, '::before').backgroundImage,
  }));
  assert.ok(accent.tint.includes('111,143,196'));
  assert.ok(accent.rule.includes('111, 143, 196'));
  await logged.hover();
  await page.waitForTimeout(200);
  assert.equal(await logged.evaluate(el => getComputedStyle(el).borderColor), 'rgb(111, 143, 196)');
  await logged.focus();
  assert.equal(await logged.evaluate(el => getComputedStyle(el).outlineStyle), 'solid');
  await page.locator('.trailing-summary').screenshot({ path: '/tmp/weekly-plan-logged-runs.png' });
  await logged.click();
  assert.equal(new URL(page.url()).hash, '#run/100');
  assert.deepEqual(errors, []);
  console.log('Desktop/mobile layout, logged-run hover/focus, blue accents and analysis navigation passed.');
} finally {
  await browser.close();
}
