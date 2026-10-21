// Read-only layout/style checks. No planner refresh, settings writes or sync.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {chromium}=createRequire(import.meta.url)('playwright');
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const presentation=()=>page.evaluate(()=>Object.fromEntries([
    '.vo2-value strong','.effort-share strong','.volume-level > strong',
    '.benchmark-pace-card h2','.vo2-card',
  ].map(selector=>{
    const s=getComputedStyle(document.querySelector(selector));
    return [selector,{font:s.fontSize,family:s.fontFamily,weight:s.fontWeight,
      color:s.color,background:s.backgroundImage,border:s.borderTopColor,radius:s.borderRadius}];
  })));
  await page.goto('http://127.0.0.1:8000/#progress');
  await page.locator('.training-history').waitFor({timeout:120000});
  const original=await presentation();
  assert.equal(await page.locator('.wip-supporting-layout').count(),1);
  assert.equal(await page.locator('nav a[data-route="progress"]').count(),1);
  assert.equal(await page.locator('nav a[data-route="progress-wip"]').count(),0);
  assert.equal(await page.locator('.page-heading .eyebrow').innerText(),'PROGRESS');
  await page.goto('http://127.0.0.1:8000/#progress-wip');
  await page.waitForURL('**/#progress');
  await page.locator('.wip-supporting-layout').waitFor({timeout:120000});
  const compact=await presentation();
  for(const selector of Object.keys(original)) {
    assert.deepEqual(compact[selector],original[selector],'Redirect retains the reviewed presentation');
  }
  for(const width of [1440,900,390]) {
    await page.setViewportSize({width,height:1100});
    // Opening at each size also tests chart-label adaptation.
    await page.reload();await page.locator('.wip-supporting-layout').waitFor({timeout:120000});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No overflow at ${width}`);
    const rect=selector=>page.locator(selector).boundingBox();
    const chart=await rect('.benchmark-pace-card'),volume=await rect('.training-history');
    const vo2=await rect('.vo2-card'),quality=await rect('.wip-quality-card'),goal=await rect('.goal-journey');
    assert.ok(chart.y+chart.height<=volume.y+1);
    assert.ok(goal.y>=Math.max(volume.y+volume.height,quality.y+quality.height)-1);
    if(width>1100) {
      assert.match(await page.locator('.volume-trend').getAttribute('viewBox'),/420$/);
      assert.ok(vo2.x>=volume.x+volume.width);
      assert.ok(Math.abs(vo2.y-volume.y)<=1);
      assert.ok(quality.y>=vo2.y+vo2.height);
      assert.ok(Math.abs(volume.y+volume.height-quality.y-quality.height)<=1,'Volume window spans the supporting stack');
      assert.ok(volume.height<950,'Supporting windows no longer require a 1,200px-tall row');
      const volumeBody=await rect('.training-main');
      assert.ok(volumeBody.width>volume.width*.85,'Moving quality must not leave its old empty grid column');
    } else {
      assert.match(await page.locator('.volume-trend').getAttribute('viewBox'),/290$/);
      assert.ok(vo2.y>=volume.y+volume.height);
      assert.ok(Math.abs(vo2.x-volume.x)<=1);
    }
    await page.locator('.wip-supporting-layout').screenshot({path:`/tmp/progress-wip-layout-${width}.png`});
  }
  await page.setViewportSize({width:1440,height:1100});
  await page.reload();await page.locator('.wip-supporting-layout').waitFor({timeout:120000});
  await page.locator('[data-metric="both"]').click();
  await page.locator('.benchmark-pace-line.unadjusted').waitFor();
  assert.equal(await page.locator('.wip-quality-card .effort-history').count(),1);
  assert.equal(await page.locator('.goal-journey-chart').isVisible(),true);
  assert.equal(await page.locator('.pace-outlook-chart').isVisible(),false);
  let requests=0;page.on('request',r=>{if(r.url().includes('/api/'))requests++;});
  await page.locator('[data-goal-outlook="pace"]').click();
  assert.equal(await page.locator('.pace-outlook-chart').isVisible(),true);
  assert.equal(await page.locator('.goal-journey-chart').isVisible(),false);
  assert.equal(await page.locator('[data-goal-outlook="pace"]').getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('.outlook-band').count(),1);
  await page.locator('.goal-journey').screenshot({path:'/tmp/progress-wip-goal-pace.png'});
  await page.locator('[data-goal-outlook="distance"]').focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('.goal-journey-chart').isVisible(),true);
  await page.locator('.goal-journey').screenshot({path:'/tmp/progress-wip-goal-distance.png'});
  assert.equal(requests,0,'Switching outlooks never calls the API or planner');
  assert.deepEqual(errors,[]);
  console.log('Compact WIP: smaller type, original colors/card styling, desktop/mobile geometry, chart controls and local outlook switching passed.');
} finally {await browser.close();}
