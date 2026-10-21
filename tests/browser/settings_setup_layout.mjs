// Actual Settings setup entry and stylesheet; no server writes or planner calls.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
const { chromium } = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/', import.meta.url);
const source = await readFile(new URL('app/views/settings.js', assets), 'utf8');
const template = source.match(/<article class="wide-card setup-entry[\s\S]*?<\/article>/)[0];
const markup = Function('setup', `return \`${template}\`;`);
const css = await readFile(new URL('styles.css', assets), 'utf8');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  for (const complete of [true, false]) {
    await page.setContent(`<!doctype html><style>${css}</style><section class="page settings-page"><form>${markup({complete, total:8, remaining: complete ? 0 : 2})}</form></section>`);
    for (const width of [1100, 760, 390]) {
      await page.setViewportSize({width, height: 600});
      const layout = await page.locator('.setup-entry').evaluate(panel => {
        const p = panel.getBoundingClientRect();
        const b = panel.querySelector('a').getBoundingClientRect();
        return {inPanel: b.left >= p.left && b.right <= p.right && b.bottom <= p.bottom,
          background: getComputedStyle(panel.querySelector('a')).backgroundColor,
          overflow: document.documentElement.scrollWidth > innerWidth};
      });
      assert.ok(layout.inPanel && !layout.overflow, `Setup layout at ${width}: ${JSON.stringify(layout)}`);
      assert.equal(layout.background, 'rgb(23, 61, 52)', 'Restore the filled green button');
      assert.equal(await page.locator('a').getAttribute('href'), '#setup');
      assert.equal(await page.locator('a').textContent(), complete ? 'Review setup' : 'Finish setup');
      assert.equal(await page.locator('h2').textContent(), complete ? 'You’re set up · 8 of 8 essential settings confirmed' : '2 setup steps remaining');
    }
  }
  await page.setViewportSize({width: 1100, height: 600});
  await page.setContent(`<!doctype html><style>${css}</style><section class="page settings-page"><form>${markup({complete:true, total:8, remaining:0})}</form></section>`);
  await page.locator('.setup-entry').screenshot({path:'/tmp/settings-setup-panel.png'});
  console.log('Settings setup panel and green link verified at desktop/mobile widths, for complete/incomplete setup.');
} finally { await browser.close(); }
