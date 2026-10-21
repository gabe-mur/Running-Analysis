// Real components/CSS, no app database or planner requests.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import {fitnessChart, vo2Chart} from '../../src/run_analysis/web/static/app/charts.js';
import {racePreparationFields} from '../../src/run_analysis/web/static/app/race-settings.js';
const {chromium}=createRequire(import.meta.url)('playwright');
const assets=new URL('../../src/run_analysis/web/static/',import.meta.url);
const css=await readFile(new URL('styles.css',assets),'utf8');
const progressCss=await readFile(new URL('progress-wip.css',assets),'utf8');
const series=[0,1,2].map(i=>({activity_id:i+1,start_time:`2026-10-${String(i+1).padStart(2,'0')}T12:00:00Z`,distance_miles:4,
  standardized_pace_min_mile:10-i*.1,raw_pace_min_mile:10.2-i*.1,uncertainty_95_min_mile:.2,trend_weight:1}));
const vo2=series.map((p,i)=>({as_of:p.start_time,value_ml_kg_min:45+i,uncertainty_95_ml_kg_min:2}));
const html=`<!doctype html><style>${css}\n${progressCss}</style><main id="app-view">
  <section class="page settings-page"><div class="settings-grid"><fieldset><legend>Training goal</legend>${racePreparationFields({distance:'general_fitness'})}<label>Reference HR<input type="number" value="148"></label><p>Supporting explanation</p></fieldset></div></section>
  <section class="page"><form class="filter-bar"><label>Show<select><option>All runs</option></select></label><button>Apply</button></form>
    <div class="plan-summary-grid"><article class="wide-card"><h2>4 planned runs · 20.2 miles</h2></article><article class="goal-panel panel"><h2>Training for general fitness</h2></article></div>
  </section><section class="page progress-wip"><article class="wide-card chart-card"><h2>Adjusted pace</h2>${fitnessChart(series,'standardized',[],null,null,{interactive:true})}</article>
  <article class="wide-card"><h2>VO₂ max</h2>${vo2Chart(vo2)}</article></section>
  <section class="page" id="dashboard-spacing"><section class="wide-card dashboard-training-load"><h2>Training load</h2></section><article class="wide-card fitness-context"><p class="eyebrow">Fitness has more than one dimension</p></article></section></main>`;
const browser=await chromium.launch({headless:true});
try {
 const page=await browser.newPage();
 await page.route('https://type.test/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/')return route.fulfill({body:html,contentType:'text/html'});
  assert.ok(path.startsWith('/assets/')&&!path.includes('..'));
  return route.fulfill({body:await readFile(new URL(path.slice('/assets/'.length),assets)),contentType:'text/javascript'});
 });
 await page.goto('https://type.test/');
 await page.evaluate(async()=>{
  const {wireRunCharts}=await import('/assets/app/chart-interaction.js');
  wireRunCharts(document.querySelector('#app-view'));
 });
 const geometry=await page.locator('svg').evaluateAll(nodes=>nodes.map(svg=>({viewBox:svg.getAttribute('viewBox'),paths:[...svg.querySelectorAll('path')].map(p=>p.getAttribute('d'))})));
 for(const width of [1440,1100,760,390,320]){
  await page.setViewportSize({width,height:900});
  await page.waitForFunction(()=>[...document.querySelectorAll('svg text')].every(text=>{
   const m=text.getScreenCTM(); return Math.abs(parseFloat(getComputedStyle(text).fontSize)*Math.hypot(m.a,m.b)-12)<.1;
  }));
  const sizes=await page.locator('button, input, select').evaluateAll(nodes=>nodes.map(n=>getComputedStyle(n).fontSize));
  assert.ok(sizes.every(size=>size==='16px'),`Control sizes at ${width}: ${sizes}`);
  assert.ok((await page.locator('h2').evaluateAll(nodes=>nodes.map(n=>getComputedStyle(n).fontSize))).every(size=>size==='24px'));
  assert.equal(await page.locator('fieldset > p').last().evaluate(n=>getComputedStyle(n).fontSize),'14px');
  const clipped=await page.locator('svg text').evaluateAll(nodes=>nodes.filter(text=>{
   const r=text.getBoundingClientRect(),p=text.closest('article').getBoundingClientRect();
   return r.left<p.left || r.right>p.right || r.top<p.top || r.bottom>p.bottom;
  }).map(text=>text.textContent));
  assert.deepEqual(clipped,[],`Clipped chart labels at ${width}`);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`Page overflow at ${width}`);
  const sectionGap=await page.locator('#dashboard-spacing').evaluate(el=>{
   const load=el.querySelector('.dashboard-training-load').getBoundingClientRect();
   const fitness=el.querySelector('.fitness-context').getBoundingClientRect();
   return fitness.top-load.bottom;
  });
  assert.equal(sectionGap,16,`Dashboard section gap at ${width}`);
 }
 assert.deepEqual(await page.locator('svg').evaluateAll(nodes=>nodes.map(svg=>({viewBox:svg.getAttribute('viewBox'),paths:[...svg.querySelectorAll('path')].map(p=>p.getAttribute('d'))}))),geometry,'Typography must not change chart data or geometry');
 await page.locator('svg[data-run-chart] circle[data-run-key]').first().hover();
 assert.equal(await page.locator('.run-chart-tooltip').first().isVisible(),true);
 assert.ok(await page.locator('.is-nearest-run').count()>0);
 console.log('Typography verified at 1440/1100/760/390/320px; readable unclipped chart labels, unchanged geometry and working run hover.');
}finally{await browser.close();}
