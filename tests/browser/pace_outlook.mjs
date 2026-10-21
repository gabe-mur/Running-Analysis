// Real read-only WIP route, no settings changes or planner execution.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {chromium}=createRequire(import.meta.url)('playwright');
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8000/#progress');
  await page.locator('[data-goal-outlook="pace"]').click({timeout:120000});
  await page.locator('.pace-outlook-chart').waitFor({timeout:120000});
  assert.match(await page.locator('.outlook-value').innerText(),/\d+:\d{2}/);
  assert.match(await page.locator('.outlook-range').innerText(),/\d+:\d{2}.*\d+:\d{2}/s);
  assert.match(await page.locator('.pace-outlook').textContent(),/42-day aerobic trend/);
  assert.match(await page.locator('.pace-outlook').textContent(),/errors predicting actual future runs/);
  assert.equal(await page.locator('.outlook-range-marker').count(),1);
  assert.equal(await page.locator('.outlook-band').count(),1);
  assert.equal(await page.locator('.outlook-bound').count(),2);
  assert.equal(await page.locator('.outlook-bound-label').count(),2);
  const geometry=await page.locator('.pace-outlook-chart').evaluate(svg=>({
    now:Number(svg.querySelector('.journey-now').getAttribute('x1')),
    endpoint:Number(svg.querySelector('.outlook-point.endpoint').getAttribute('cx')),
    marker:svg.querySelector('.outlook-range-marker').getAttribute('d'),
    bandStart:Number(svg.querySelector('.outlook-band').getAttribute('d').match(/^M([^,]+)/)[1]),
  }));
  assert.ok(geometry.endpoint>geometry.now);
  assert.ok(geometry.bandStart>geometry.now,'Band must not attach to a zero-width Now');
  assert.equal(Number(geometry.marker.match(/^M([^,]+)/)[1]),geometry.endpoint);
  for(const width of [1440,900,390]) {
    await page.setViewportSize({width,height:1100});
    // Re-render at the new breakpoint, like opening the page at that width.
    await page.reload();await page.locator('[data-goal-outlook="pace"]').click({timeout:120000});await page.locator('.pace-outlook-chart').waitFor();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No horizontal overflow at ${width}`);
    await page.locator('.goal-journey').screenshot({path:`/tmp/pace-outlook-${width}.png`});
  }
  await page.setViewportSize({width:1440,height:1100});
  await page.reload();await page.locator('[data-goal-outlook="pace"]').click({timeout:120000});await page.locator('.pace-outlook-chart').waitFor();
  await page.locator('.outlook-point.endpoint').hover();
  await page.locator('.run-chart-tooltip:visible').waitFor({state:'visible'});
  assert.match(await page.locator('.run-chart-tooltip:visible').innerText(),/Projected pace:.*\d+:\d{2}/s);
  assert.deepEqual(errors,[]);
  console.log('WIP pace outlook: supported-horizon band, 1440/900/390 layout and future-date inspection passed.');
} finally {await browser.close();}
