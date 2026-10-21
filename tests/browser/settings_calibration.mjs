// Actual Settings view and CSS, mocked APIs only; no live settings/planner writes.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
const {chromium} = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/',import.meta.url);
let settings = {
  max_hr:194,resting_hr:46,target_hr:148,planner:null,
  profile:{birth_date:'1990-01-01',sex:'male',weight_lb:150,height_in:70,max_hr_source:'measured'},
  setup:{zone_method:'device',confirmed_steps:['profile','heart_rate','zones']},
  zones:Object.fromEntries(['z1','z2','z3','z4','z5'].map((name,i)=>[name,{minimum_bpm:128+i*13,maximum_bpm:140+i*13}])),
  race_preparation:{distance:'general_fitness',preparation_mode:'balanced',race_date:null},
  reference_temperature_f:55,reference_dewpoint_f:45,reference_wind_mph:3,reference_grade_percent:0,
  reference_within_run_minutes:20,weather_privacy_radius_km:2,historical_weather_enabled:false,
  forecast_weather_enabled:false,default_fitness_window:28,available_fitness_windows:[14,28,90],
  moving_time:{minimum_running_speed_mps:.8,maximum_interval_seconds:30},
  coaching:{quality_focus:'variety',quality_sessions:{short_intervals:true,long_intervals:true,
    threshold:true,progression:true,fartlek:true,hill_repeats:false},
    strength_training:{enabled:true,leg_frequency_days:8,push_pull_frequency_days:8,
      muscle_groups:{push:['triceps'],pull:['lats'],legs:['quads']}}},
};
const requests=[],errors=[];
let recordedWatch = false;
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1100,height:900}});
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('http://settings.test/**',async route=>{
    const path=new URL(route.request().url()).pathname;
    if(path==='/')return route.fulfill({contentType:'text/html',body:
      '<link rel="stylesheet" href="/styles.css"><div id="app-view"></div><script type="module">import {renderSettings} from "/app/views/settings.js"; await renderSettings();</script>'});
    if(!path.startsWith('/api/'))return route.fulfill({contentType:path.endsWith('.css')?'text/css':'text/javascript',
      body:await readFile(new URL(path.slice(1),assets),'utf8')});
    const body=route.request().postDataJSON();
    requests.push({path,method:route.request().method(),body});
    let result;
    if(path==='/api/settings') {
      if(body)settings={...settings,...body,recalculation:[]};
      result=settings;
    } else if(path==='/api/setup')result={complete:false,steps:[{complete:false}]};
    else if(path==='/api/connections/intervals')result={configured:false,available:true};
    else if(path==='/api/setup/comparison-hr')result={recommended_bpm:148,rationale:'Chosen from recent easy running.'};
    else if(path==='/api/settings/race-preparation/status')result={status:'unavailable'};
    else if(path==='/api/setup/zone-preview') {
      // Older previews finish after the newer request to catch stale writes.
      await new Promise(resolve=>setTimeout(resolve,body.resting_hr===46?200:25));
      const lower=body.resting_hr===45?140:141;
      const zones=body.boundaries??Object.fromEntries(['z1','z2','z3','z4','z5'].map((n,i)=>
        [n,{minimum_bpm:lower-13+i*13,maximum_bpm:lower-1+i*13}]));
      result={zones,comparison_hr:{recommended_bpm:zones.z2.minimum_bpm+6,rationale:'Updated comparison.'}};
      if(recordedWatch && body.method==='device')result={
        watch_recorded_at:'2026-10-10T14:06:02Z',
        max_hr:194, resting_hr:45,
        zones:Object.fromEntries(['z1','z2','z3','z4','z5'].map((n,i)=>
          [n,{minimum_bpm:127+i*13,maximum_bpm:139+i*13}])),
        comparison_hr:{recommended_bpm:146,rationale:'Updated comparison.'}};
    } else throw new Error(`Unexpected API: ${path}`);
    return route.fulfill({json:result});
  });
  await page.goto('http://settings.test/');
  await page.getByRole('heading',{name:'Heart rate & profile'}).waitFor();
  await page.locator('[data-zone-edit-status]').getByText('Read-only: matched to your watch.',{exact:false}).waitFor();
  assert.equal(await page.locator('[name=target_hr]').count(),0);
  assert.equal(await page.locator('[name=profile_birth_date]').getAttribute('required'),'');
  assert.ok(await page.locator('legend').evaluateAll(nodes=>nodes.findIndex(n=>n.textContent==='About you')<nodes.findIndex(n=>n.textContent==='Heart-rate zones')));
  const hrPanel=page.locator('fieldset').filter({has:page.locator('legend',{hasText:'Heart-rate zones'})});
  assert.equal(await hrPanel.count(),1);
  assert.equal(await hrPanel.locator('[name=max_hr]').count(),1);
  assert.equal(await hrPanel.locator('[name=resting_hr]').count(),1);
  for(const width of [1100,760,390]) {
    await page.setViewportSize({width,height:900});
    const offsets=await page.locator('.method-option').evaluateAll(cards=>cards.map(card=>{
      const input=card.querySelector('input').getBoundingClientRect(),title=card.querySelector('b'),rect=title.getBoundingClientRect();
      return input.top+input.height/2-rect.top-parseFloat(getComputedStyle(title).lineHeight)/2;
    }));
    assert.ok(offsets.every(value=>Math.abs(value)<1),`Radio alignment at ${width}: ${offsets}`);
    const methodBox=await hrPanel.locator('.method-options').boundingBox();
    const inputBox=await hrPanel.locator('.heart-rate-inputs').boundingBox();
    assert.ok(inputBox.y >= methodBox.y+methodBox.height,`Method/watch comes before HR inputs at ${width}`);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No overflow at ${width}`);
  }
  await page.setViewportSize({width:1100,height:900});
  await page.locator('[name=method][value=heart_rate_reserve]').check();
  assert.match(await page.locator('[data-zone-edit-status]').textContent(),/Calculated from maximum and resting HR/);
  const increaseZone=page.getByRole('button',{name:'Increase Z2 minimum'});
  assert.equal(await increaseZone.isDisabled(),true);
  await increaseZone.hover();
  assert.equal(await increaseZone.evaluate(button=>getComputedStyle(button).cursor),'default');
  await page.locator('[name=resting_hr]').fill('45');
  await page.getByRole('button',{name:'Save settings',exact:true}).first().click();
  await page.waitForFunction(()=>document.querySelector('#settings-status-top').textContent==='Saved.');
  const saved=requests.filter(r=>r.path==='/api/settings'&&r.method==='PATCH').at(-1).body;
  assert.equal(saved.resting_hr,45);
  assert.equal(saved.target_hr,undefined);
  assert.equal(saved.zones.z2.minimum_bpm,140);
  assert.equal(saved.setup.comparison_hr_mode,'automatic');
  assert.equal(saved.setup.zone_method,'heart_rate_reserve');
  assert.equal(saved.profile.max_hr_source,'measured');
  await page.waitForTimeout(250);
  assert.equal(await page.locator('[name=z2_min]').inputValue(),'140');
  await page.locator('#settings-heart-rate').evaluate(node=>node.parentElement.scrollIntoView());
  await page.screenshot({path:'/tmp/settings-calibration.png',fullPage:true});
  await hrPanel.screenshot({path:'/tmp/settings-heart-rate-panel.png'});
  settings.planner={status:'adopted',preferences_revision:'current',run_frequency:'balanced',long_target_miles:9,
    long_goal_miles:17,minimum_running_minutes:10,penalty_free_running_minutes:25,demand_memory_days:2,
    demand_delivery_exchange:.01,frequency_preference_fraction:.02};
  await page.reload();
  await page.locator('[name=max_hr]').waitFor();
  assert.equal(await page.locator('[name=max_hr]').isDisabled(),false);
  recordedWatch=true;settings.setup.zone_method='device';
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('[name=z2_min]')?.readOnly===true);
  await page.waitForFunction(()=>document.querySelector('[name=max_hr_source]')?.value==='watch');
  assert.equal(await page.locator('[name=z2_min]').inputValue(),'140');
  assert.equal(await page.locator('[name=z2_min]').getAttribute('readonly'),'');
  assert.equal(await page.getByRole('button',{name:'Increase Z2 minimum'}).isDisabled(),true);
  assert.match(await page.locator('[data-zone-edit-status]').textContent(),/Read-only: matched to your watch/);
  await increaseZone.hover();
  assert.equal(await increaseZone.evaluate(button=>getComputedStyle(button).cursor),'default');
  const lockedValue=await page.locator('[name=z2_min]').inputValue();
  await increaseZone.click({force:true});
  assert.equal(await page.locator('[name=z2_min]').inputValue(),lockedValue);
  assert.equal(await page.locator('[name=max_hr]').getAttribute('readonly'),'');
  assert.equal(await page.locator('[name=resting_hr]').inputValue(),'45');
  assert.equal(await page.locator('[name=max_hr_source]').isVisible(),false);
  assert.equal(await page.locator('#settings-estimate-max').isVisible(),false);
  await hrPanel.screenshot({path:'/tmp/settings-watch-hr-panel.png'});
  await page.getByRole('button',{name:'Save settings',exact:true}).first().click();
  await page.waitForFunction(()=>document.querySelector('#settings-status-top').textContent==='Saved.');
  const watchSave=requests.filter(r=>r.path==='/api/settings'&&r.method==='PATCH').at(-1).body;
  assert.equal(watchSave.zones.z2.minimum_bpm,140);
  assert.equal(watchSave.max_hr,194);
  assert.equal(watchSave.resting_hr,45);
  assert.equal(watchSave.profile.max_hr_source,'watch');
  assert.equal(watchSave.target_hr,undefined);
  await page.locator('[name=method][value=custom]').check();
  assert.equal(await page.locator('[data-zone-edit-status]').isVisible(),false);
  assert.equal(await page.locator('[name=max_hr]').getAttribute('readonly'),null);
  assert.equal(await page.locator('#settings-estimate-max').isVisible(),true);
  assert.equal(await page.locator('[name=z2_min]').getAttribute('readonly'),null);
  assert.equal(await page.getByRole('button',{name:'Increase Z2 minimum'}).isDisabled(),false);
  await increaseZone.hover();
  assert.equal(await increaseZone.evaluate(button=>getComputedStyle(button).cursor),'pointer');
  await increaseZone.click();
  assert.equal(Number(await page.locator('[name=z2_min]').inputValue()),Number(lockedValue)+1);
  await page.locator('[name=max_hr]').fill('190');
  await page.locator('[name=resting_hr]').fill('48');
  await page.locator('[name=max_hr_source]').selectOption('measured');
  await page.locator('[name=z2_min]').fill('142');
  await page.locator('[name=method][value=device]').check();
  await page.waitForFunction(()=>document.querySelector('[name=resting_hr]').value==='45');
  await page.locator('[name=method][value=custom]').check();
  assert.equal(await page.locator('[name=max_hr]').inputValue(),'190');
  assert.equal(await page.locator('[name=resting_hr]').inputValue(),'48');
  assert.equal(await page.locator('[name=max_hr_source]').inputValue(),'measured');
  assert.equal(await page.locator('[name=z2_min]').inputValue(),'142');
  assert.deepEqual(errors,[]);
  assert.ok(requests.every(r=>! /dashboard|weekly|refresh|recommendation/.test(r.path)));
  console.log('Settings layout, stale previews, full watch HR bundle, custom draft preservation and automatic coordinate passed. No live API calls.');
} finally {await browser.close();}
