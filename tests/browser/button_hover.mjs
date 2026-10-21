// Shared hover/focus styling, using the actual stylesheet without API calls.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
const {chromium} = createRequire(import.meta.url)('playwright');
const css = await readFile(new URL('../../src/run_analysis/web/static/styles.css', import.meta.url), 'utf8');
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage();
  await page.setContent(`<!doctype html><style>${css}
    * {transition:none !important} .test-actions {display:grid;gap:20px;padding:30px;width:400px}
    </style><div class="test-actions">
      <a id="review" class="primary-button" href="#setup">Review setup</a>
      <button id="save" class="primary-button">Save settings</button>
      <button id="plain">Check preparation</button>
      <button id="sync" class="sync-button" aria-label="Sync">↻</button>
      <button id="icon" class="icon-button" aria-label="Close">×</button>
      <button id="rest" class="rest-toggle">Make run rest day</button>
      <div class="forced-rest-day"><button id="protected" class="rest-toggle">Allow run</button></div>
      <div class="segmented"><button id="segment">4 weeks</button><button id="selected" class="selected">8 weeks</button></div>
      <button id="disabled" disabled>Apply training goal</button>
    </div>`);
  const style = id => page.locator(`#${id}`).evaluate(element => {
    const s = getComputedStyle(element), r = element.getBoundingClientRect();
    return {background:s.backgroundColor, color:s.color, border:s.borderTopColor,
      borderWidth:s.borderTopWidth, shadow:s.boxShadow, width:r.width, height:r.height};
  });
  const ids = ['review','save','plain','sync','icon','rest','protected','segment','selected'];
  const resting = Object.fromEntries(await Promise.all(ids.map(async id => [id, await style(id)])));
  await page.locator('#review').hover();
  const reference = await style('review');
  assert.equal(reference.border, 'rgb(121, 213, 179)');
  assert.notEqual(reference.shadow, 'none');
  for (const id of ids) {
    await page.locator(`#${id}`).hover();
    const hovered = await style(id);
    assert.equal(hovered.border, reference.border, `${id}: shared border highlight`);
    assert.equal(hovered.shadow, reference.shadow, `${id}: shared shadow`);
    assert.equal(hovered.borderWidth, '1px');
    for (const property of ['background','color','width','height']) {
      assert.equal(hovered[property], resting[id][property], `${id}: preserve ${property}`);
    }
  }
  const disabled = await style('disabled');
  await page.locator('#disabled').hover();
  assert.deepEqual(await style('disabled'), disabled);
  await page.mouse.move(0,0);
  await page.locator('#review').focus();
  for (const id of ids.slice(1)) {
    await page.keyboard.press('Tab');
    assert.equal(await page.locator(':focus').getAttribute('id'), id);
    const focused = await style(id);
    assert.equal(focused.border, reference.border, `${id}: keyboard highlight`);
    assert.equal(focused.shadow, reference.shadow, `${id}: keyboard shadow`);
  }
  console.log('All button variants share Review Setup hover/focus; resting colors, dimensions and disabled behavior preserved.');
} finally {await browser.close();}
