// Actual UI module and styles; no server writes or planner execution.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
const {chromium}=createRequire(import.meta.url)('playwright');
const assets=new URL('../../src/run_analysis/web/static/',import.meta.url);
const data=JSON.parse(execFileSync('.venv/bin/python',['-c',`
from run_analysis.config import load_config,resolve_project_path
from run_analysis.progress import prepare_progress_data
from run_analysis.progress_wip import _canonical_load
from pathlib import Path
from datetime import datetime,timezone
import sqlite3,json
cfg=load_config();p=resolve_project_path(Path.cwd(),cfg['paths']['database'])
with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
 db.row_factory=sqlite3.Row
 prep=prepare_progress_data(db)
 report=_canonical_load(db,Path.cwd(),prep.sessions,datetime.now(timezone.utc),details=prep.details)
 print(json.dumps(report['goal_journey']))
`],{encoding:'utf8'}));
assert.ok(data.available);
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('http://journey.test/**',async route=>{
    const path=new URL(route.request().url()).pathname;
    if(path==='/') return route.fulfill({contentType:'text/html',body:`<!doctype html>
      <link rel="stylesheet" href="/styles.css"><link rel="stylesheet" href="/progress-wip.css">
      <section class="page progress-wip"><div id="preview"></div></section>
      <script type="module">import {goalJourneyPanel} from '/app/goal-journey.js';
      import {wireRunCharts} from '/app/chart-interaction.js';
      window.paint=data=>{const root=document.querySelector('#preview');root.innerHTML=goalJourneyPanel(data,
        value=>new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(new Date(value)));
        wireRunCharts(root);};window.paint(${JSON.stringify(data)});</script>`});
    return route.fulfill({contentType:path.endsWith('.css')?'text/css':'text/javascript',body:await readFile(new URL(path.slice(1),assets),'utf8')});
  });
  await page.goto('http://journey.test/');
  await page.locator('.goal-journey-chart').waitFor();
  for(const width of [1440,900,390]) {
    await page.setViewportSize({width,height:1100});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No overflow at ${width}`);
    assert.ok(await page.locator('.goal-journey').isVisible());
    await page.locator('.goal-journey').screenshot({path:`/tmp/goal-journey-${width}.png`});
  }
  await page.setViewportSize({width:1440,height:1000});
  const mark=page.locator('.journey-recorded').last();
  if(await mark.count()) {await mark.hover();await page.locator('.run-chart-tooltip').waitFor({state:'visible'});}
  assert.deepEqual(errors,[]);
  console.log('Goal prototype: real read-only goal data, desktop/mobile layout and run inspection passed.');
} finally {await browser.close();}
