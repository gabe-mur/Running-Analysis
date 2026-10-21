// Actual setup option markup/CSS, offline: no settings saves or planner calls.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { zoneMethodFields } from '../../src/run_analysis/web/static/app/setup-fields.js';
const { chromium } = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/', import.meta.url);
const markup = zoneMethodFields('device');
const css = await readFile(new URL('styles.css', assets), 'utf8');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.setContent(`<!doctype html><style>${css}</style><section class="page narrow setup-page"><form><section class="wide-card setup-block"><p class="eyebrow">Step 2</p><h2>Heart-rate zones</h2>${markup}</section></form></section>`);
  for (const width of [1100, 760, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const offsets = await page.locator('.method-option').evaluateAll(cards => cards.map(card => {
      const input = card.querySelector('input'), title = card.querySelector('b');
      const radio = input.getBoundingClientRect(), heading = title.getBoundingClientRect();
      const lineHeight = parseFloat(getComputedStyle(title).lineHeight);
      return { delta: radio.top + radio.height / 2 - heading.top - lineHeight / 2,
        sameRow: Math.abs(radio.top - heading.top) < lineHeight,
        titleWithinCard: heading.right <= card.getBoundingClientRect().right };
    }));
    assert.ok(offsets.every(o => Math.abs(o.delta) < 1 && o.sameRow && o.titleWithinCard), `Radio/title alignment at ${width}: ${JSON.stringify(offsets)}`);
    assert.equal(await page.locator('input:checked').count(), 1);
  }
  await page.locator('.method-option').last().click();
  assert.equal(await page.locator('input:checked').getAttribute('value'), 'custom');
  await page.locator('input[value="device"]').focus();
  await page.keyboard.press('Space');
  assert.equal(await page.locator('input:checked').getAttribute('value'), 'device');
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.locator('.setup-block').screenshot({ path: '/tmp/setup-radio-alignment.png' });
  console.log('Radio/title alignment at desktop/mobile widths, label click and keyboard selection passed.');
} finally { await browser.close(); }
