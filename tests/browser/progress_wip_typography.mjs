// Read-only audit of the actual rendered scale, including CSS cascade/mobile
// overrides. No planner execution, sync or settings writes.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {chromium}=createRequire(import.meta.url)('playwright');
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const sizes=()=>page.evaluate(()=>{
    const read=selector=>parseFloat(getComputedStyle(document.querySelector(selector)).fontSize);
    return {
      metrics:['.headline-pace strong','.volume-level > strong','.vo2-value strong','.effort-share > strong','.outlook-value'].map(read),
      titles:['.benchmark-pace-card h2','.training-history h2','.vo2-card h2','.effort-panel h3','#goal-journey-title'].map(read),
      secondary:['.journey-distance dd','.journey-eta strong'].map(read),
      labels:['.chart-note','.vo2-value span','.effort-share strong small','.outlook-summary details'].map(read),
      heading:read('.page-heading h1'),
      caption:read('.eyebrow'),
      heights:Object.fromEntries(['.page-heading','.training-history','.vo2-card','.wip-quality-card','.goal-journey'].map(s=>[s,document.querySelector(s).getBoundingClientRect().height])),
    };
  });
  await page.goto('http://127.0.0.1:8000/#progress');
  await page.locator('.vo2-card').waitFor({timeout:120000});
  const original=await page.locator('.vo2-value strong').evaluate(el=>getComputedStyle(el).fontSize);
  assert.equal(original,'28px','Reviewed compact Progress is now primary');
  await page.goto('http://127.0.0.1:8000/#progress-wip');
  await page.waitForURL('**/#progress');
  for(const width of [1440,900,390]) {
    await page.setViewportSize({width,height:1100});await page.reload();
    await page.locator('.goal-outlook-window').waitFor({timeout:120000});
    const result=await sizes();
    assert.deepEqual(result.metrics,[28,28,28,28,28]);
    assert.deepEqual(result.titles,[20,20,20,20,20]);
    assert.deepEqual(result.secondary,[20,20]);
    assert.deepEqual(result.labels,[14,14,14,14]);
    assert.equal(result.caption,12);
    assert.equal(result.heading,Math.min(64,Math.max(40,width*.05)),'Original large headline is retained');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    if(width===1440) {
      assert.ok(result.heights['.training-history']<760,'Supporting row is shorter than the former 794px');
      assert.ok(result.heights['.goal-journey']<480,'Goal plots remove unused vertical space');
    }
    await page.locator('[data-goal-outlook="pace"]').click();
    assert.equal(await page.locator('.outlook-value').isVisible(),true);
    assert.equal(await page.locator('.outlook-value').evaluate(el=>getComputedStyle(el).color),
      await page.locator('.page').evaluate(el=>getComputedStyle(el).color),
      'Projected pace is neutral, not a green improvement claim');
    // Compensated SVG labels remain 12 screen pixels even on narrow plots.
    const chartSizes=await page.locator('svg:visible text').evaluateAll(labels=>labels.map(el=>{
      const m=el.getScreenCTM();return parseFloat(getComputedStyle(el).fontSize)*Math.hypot(m.a,m.b);
    }));
    assert.ok(chartSizes.length>0);
    for(const size of chartSizes)assert.ok(Math.abs(size-12)<.2,`Readable chart label: ${size}`);
    await page.locator('.goal-journey').screenshot({path:`/tmp/progress-wip-type-goal-${width}.png`});
    await page.locator('.wip-supporting-layout').screenshot({path:`/tmp/progress-wip-type-panels-${width}.png`});
    for(const outlook of ['distance','pace']) {
      await page.locator(`[data-goal-outlook="${outlook}"]`).click();
      const rect=selector=>page.locator(selector).boundingBox();
      const title=await rect('#goal-journey-title'),edit=await rect('.journey-edit');
      if(width===1440) {
        assert.ok(edit.x>=title.x+title.width,'Edit goal is to the right of the title');
        assert.ok(Math.abs(edit.y+edit.height/2-title.y-title.height/2)<2,'Title and edit control are vertically aligned');
      } else assert.ok(edit.x>=title.x+title.width||edit.y>=title.y+title.height,'Edit control wraps cleanly when needed');
      if(width===1440) {
        const toggle=await rect('.journey-switch'),eyebrow=await rect('.goal-journey > .card-heading .eyebrow');
        assert.ok(Math.abs(toggle.y-eyebrow.y)<2,'View switch is not pushed down by Edit goal');
      }
      const chart=await rect(`[data-goal-panel="${outlook}"] svg`),key=await rect(`[data-goal-panel="${outlook}"] .journey-key`);
      assert.ok(key.y>=chart.y+chart.height,'Goal keys sit below the graph');
      const svg=page.locator(`[data-goal-panel="${outlook}"] svg`);
      await svg.focus();await page.keyboard.press('End');
      const tooltip=page.locator('[role="tooltip"]:visible');
      assert.equal(await tooltip.evaluate(el=>getComputedStyle(el).fontSize),'14px','Hover/keyboard inspector matches body text');
      await page.keyboard.press('Escape');
      await page.locator('.page details').evaluateAll(ds=>ds.forEach(d=>d.open=true));
      // Inspect EVERY rendered text node, not a shortlist of number selectors.
      const audit=await page.evaluate(()=>{
        const styles=new Map(),families=new Set(),weights=new Set(),unexpected=[];
        const walker=document.createTreeWalker(document.querySelector('.page'),NodeFilter.SHOW_TEXT);
        const headline=parseFloat(getComputedStyle(document.querySelector('h1')).fontSize);
        while(walker.nextNode()) {
          const node=walker.currentNode,el=node.parentElement;
          if(!node.textContent.trim()||el.closest('svg')||!el.checkVisibility())continue;
          const style=getComputedStyle(el),size=parseFloat(style.fontSize);
          families.add(style.fontFamily);weights.add(style.fontWeight);
          styles.set(size,(styles.get(size)??0)+1);
          if(!(el.tagName==='H1'?size===headline:[12,14,20,28].includes(size)))unexpected.push({text:node.textContent.slice(0,50),size});
        }
        return {sizes:[...styles.keys()].sort((a,b)=>a-b),families:[...families],weights:[...weights].sort(),unexpected};
      });
      assert.deepEqual(audit.unexpected,[],`All text follows the type hierarchy at ${width}px (${outlook})`);
      assert.equal(audit.families.length,1,'One font family throughout');
      assert.deepEqual(audit.weights,['400','600','700'],'Regular, emphasis and heading weights only');
      await page.locator('.page details').evaluateAll(ds=>ds.forEach(d=>d.open=false));
      if(width===1440)console.log(JSON.stringify({outlook,audit}));
    }
    assert.equal(await page.locator('.journey-edit').getAttribute('href'),'#settings');
    const editColor=await page.locator('.journey-edit').evaluate(el=>getComputedStyle(el).color);
    await page.locator('.journey-edit').hover();
    assert.notEqual(await page.locator('.journey-edit').evaluate(el=>getComputedStyle(el).color),editColor,'Edit goal has a visible hover response');
    console.log(JSON.stringify({width,...result}));
  }
  assert.deepEqual(errors,[]);
  console.log('Complete WIP audit passed: original headline, every text node in both outlooks/expanded details, neutral forecast, bottom keys and styled Edit goal.');
} finally {await browser.close();}
