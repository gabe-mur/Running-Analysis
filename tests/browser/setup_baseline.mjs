// Real setup view, mocked local APIs only. Never touches the live planner.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
const { chromium } = createRequire(import.meta.url)('playwright');
const assets = new URL('../../src/run_analysis/web/static/', import.meta.url);
let settings = {
  max_hr: 190, resting_hr: 55, target_hr: 145, profile: null, planner: null,
  setup: {confirmed_steps: [], zone_method: 'device'},
  race_preparation: {distance: 'general_fitness'},
  historical_weather_enabled: false, forecast_weather_enabled: false, weather_privacy_radius_km: 2,
  zones: Object.fromEntries(['z1','z2','z3','z4','z5'].map((name, i) =>
    [name, {minimum_bpm: 115+i*15, maximum_bpm: 129+i*15}])),
};
let hasHistory = false;
const setupState = () => ({complete: false, steps:
  ['profile','heart_rate','zones','goal','weather','runs'].map(step => ({
    step, title: step, detail: '', blocking: step === 'runs',
    complete: step === 'runs' ? hasHistory : settings.setup.confirmed_steps.includes(step),
  }))});
const requests = [], errors = [];
let previewDelay = 0, failSave = false;
const browser = await chromium.launch({headless: true});
try {
  const page = await browser.newPage({viewport: {width: 1100, height: 900}});
  page.on('pageerror', error => errors.push(error.message));
  await page.route('http://setup.test/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname;
    if (path === '/') return route.fulfill({contentType: 'text/html', body:
      '<link rel="stylesheet" href="/styles.css"><div id="app-view"></div><script type="module">import {renderSetup} from "/app/views/setup.js"; await renderSetup();</script>'});
    if (!path.startsWith('/api/')) return route.fulfill({contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript',
      body: await readFile(new URL(path.slice(1), assets), 'utf8')});
    requests.push({path, method: route.request().method(), body: route.request().postDataJSON()});
    let result;
    if (path === '/api/setup') result = setupState();
    else if (path === '/api/settings' && route.request().method() === 'GET') result = settings;
    else if (path === '/api/settings') {
      if (failSave) return route.fulfill({status: 422, json: {detail: 'Test save rejected'}});
      const input = route.request().postDataJSON();
      settings = {...settings, ...input, setup: {...settings.setup,...input.setup}, recalculation: []};
      settings.target_hr = Math.floor((settings.zones.z2.minimum_bpm + settings.zones.z2.maximum_bpm)/2);
      result = settings;
    } else if (path === '/api/setup/max-hr') result = {estimated_max_hr: 187};
    else if (path === '/api/setup/zone-preview') {
      const input = route.request().postDataJSON();
      if (input.method === 'device') return route.fulfill({status:422,
        json:{detail:'Import a FIT run with recorded watch HR settings, or choose another heart-rate method.'}});
      await new Promise(resolve => setTimeout(resolve, previewDelay));
      result = {zones: input.boundaries ?? Object.fromEntries(['z1','z2','z3','z4','z5'].map((name, i) =>
        [name, {minimum_bpm: 110+i*15, maximum_bpm: 124+i*15}])),
        comparison_hr: {recommended_bpm: null, rationale: 'Waiting for easy-running history.'}};
    } else if (path === '/api/settings/race-preparation/status') result = settings.planner
      ? {status:'available',expected_revision:'goal-current'} : {status: 'unavailable'};
    else if (path === '/api/settings/race-preparation/apply') {
      assert.equal(route.request().postDataJSON().expected_revision,'goal-current');
      result={planner:settings.planner,detail:'Goal applied.'};
    }
    else throw new Error(`Unexpected API call: ${path}`);
    return route.fulfill({json: result});
  });
  await page.goto('http://setup.test/');
  await page.getByRole('heading', {name: 'Add your runs first', exact: true}).waitFor();
  assert.equal(await page.locator('#setup-form').isVisible(), false);
  assert.match(await page.locator('#setup-import').textContent(), /FIT, TCX or GPX/);
  assert.match(await page.locator('#setup-import').textContent(), /Sync/);
  await page.getByRole('button', {name: 'I don’t have recorded runs', exact: true}).click();
  await page.getByRole('heading', {name: 'About you', exact: true}).waitFor();
  assert.match(await page.locator('[data-zone-edit-status]').textContent(),/Read-only: matched to your watch/);
  assert.equal(await page.locator('[name=target_hr]').count(),0);
  assert.equal(await page.getByRole('heading',{name:'Comparison heart rate',exact:true}).count(),0);
  const hrPanel=page.locator('.setup-block').filter({has:page.getByRole('heading',{name:'Heart-rate zones',exact:true})});
  assert.equal(await hrPanel.count(),1);
  assert.equal(await hrPanel.locator('[name=max_hr]').count(),1);
  assert.equal(await hrPanel.locator('[name=resting_hr]').count(),1);
  assert.equal(await page.getByRole('heading', {name: 'Your first baseline run', exact: true}).count(), 0);
  assert.equal(await page.locator('.zone-calibration-row').count(),5);
  assert.equal(await page.locator('.zone-table').count(),0);
  await page.locator('[name=method][value=custom]').check();
  assert.equal(await page.locator('[data-zone-edit-status]').isVisible(),false);
  await page.getByRole('button',{name:'Increase Z1 minimum',exact:true}).click();
  assert.equal(await page.locator('[name=z1_min]').inputValue(),'116');
  await page.getByRole('button',{name:'Decrease Z1 minimum',exact:true}).click();
  assert.equal(await page.locator('[name=z1_min]').inputValue(),'115');
  for (const width of [1100,760,390]) {
    await page.setViewportSize({width,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No overflow at ${width}`);
    const placements=await page.locator('.zone-number-field').evaluateAll(fields=>fields.map(field=>{
      const input=field.querySelector('input').getBoundingClientRect(),buttons=field.querySelector('.zone-steppers').getBoundingClientRect();
      return {external:buttons.left>=input.right, readable:input.width>=55};
    }));
    assert.ok(placements.every(p=>p.external&&p.readable),`External steppers and readable numbers at ${width}: ${JSON.stringify(placements)}`);
    if(width===1100) {
      const rows=await page.locator('.zone-calibration-row').evaluateAll(rows=>rows.map(row=>{
        const label=row.querySelector('.zone-current').getBoundingClientRect(),input=row.querySelector('.zone-pair').getBoundingClientRect();
        return input.left>label.right&&Math.abs(input.top+input.height/2-label.top-label.height/2)<1;
      }));
      assert.ok(rows.every(Boolean),'Range and controls align on the same row');
      await page.locator('.zone-calibration-list').screenshot({path:'/tmp/setup-combined-zones.png'});
    }
  }
  await page.setViewportSize({width:1100,height:900});
  await page.locator('[name=profile_birth_date]').fill('1995-11-28');
  await page.locator('[name=profile_sex]').selectOption('male');
  await page.locator('[name=profile_weight_lb]').fill('150');
  await page.locator('[name=profile_height_in]').fill('70');
  await page.locator('#estimate-max-hr').click();
  await page.waitForFunction(() => document.querySelector('[name=max_hr]').value === '187');
  await page.locator('[name=method][value=heart_rate_reserve]').check();
  assert.match(await page.locator('[data-zone-edit-status]').textContent(),/Calculated from maximum and resting HR/);
  assert.equal(await page.getByRole('button',{name:'Increase Z1 minimum',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Increase Z1 minimum',exact:true}).hover();
  assert.equal(await page.getByRole('button',{name:'Increase Z1 minimum',exact:true}).evaluate(button=>getComputedStyle(button).cursor),'default');
  await page.waitForFunction(() => document.querySelector('[data-zone="z2"] [data-zone-current]').textContent.includes('125–139 bpm'));
  // A failed write cannot expose baseline guidance from unsaved calibration.
  failSave = true;
  await page.getByRole('button', {name: 'Save setup', exact: true}).click();
  await page.getByText('Test save rejected', {exact: true}).waitFor();
  assert.equal(await page.getByRole('heading', {name: 'Your first baseline run', exact: true}).count(), 0);
  failSave = false;
  previewDelay = 100;
  await page.locator('[name=resting_hr]').fill('50');
  await page.getByRole('button', {name: 'Save setup', exact: true}).click();
  await page.getByRole('heading', {name: 'Your first baseline run', exact: true}).waitFor();
  assert.match(await page.locator('#setup-baseline').textContent(), /Z2: 125–139 bpm/);
  assert.match(await page.locator('#setup-baseline').textContent(), /age-estimated/);
  await page.setViewportSize({width:1100,height:900});
  await hrPanel.screenshot({path:'/tmp/setup-heart-rate-panel.png'});
  const saved = requests.filter(r => r.path === '/api/settings' && r.method === 'PATCH').at(-1).body;
  assert.equal(saved.resting_hr, 50);
  assert.deepEqual(saved.zones.z2, {minimum_bpm: 125, maximum_bpm: 139});
  assert.equal(saved.target_hr, undefined);
  assert.equal(settings.target_hr,132);
  assert.equal(saved.profile.birth_date, '1995-11-28');
  assert.ok(saved.setup.confirmed_steps.includes('profile'));
  // Experienced runners enter current exposure separately from future goals.
  await page.locator('[name=background_choice]').selectOption('experienced');
  await page.locator('[name=background_volume]').fill('24');
  await page.locator('[name=background_frequency]').fill('4');
  await page.locator('[name=background_long]').fill('8');
  await page.getByRole('button',{name:'Save setup',exact:true}).click();
  await page.getByRole('heading',{name:'Your first recorded easy run',exact:true}).waitFor();
  assert.match(await page.locator('#setup-baseline').textContent(),/5.3 miles/);
  assert.match(await page.locator('#setup-baseline').textContent(),/24 miles per week, 4 runs/);
  assert.equal(settings.training_background.runs_per_week,4);
  for(const width of [1100,390]) {
    await page.setViewportSize({width,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Starting-level layout at ${width}`);
  }
  await page.setViewportSize({width:1100,height:900});
  await page.locator('#training-background').screenshot({path:'/tmp/setup-reported-training.png'});
  settings.training_background=null;
  // Importers skip the no-history branch and continue directly with setup.
  hasHistory = true;
  await page.reload();
  await page.getByRole('heading', {name: 'About you', exact: true}).waitFor();
  assert.equal(await page.locator('#setup-form').isVisible(), true);
  assert.equal(await page.locator('#setup-import').count(), 0);
  assert.equal(await page.locator('#setup-no-history').count(), 0);
  assert.equal(await page.getByRole('heading', {name: 'Your first baseline run', exact: true}).count(), 0);
  settings.planner={status:'adopted',preferences_revision:'preferences-current',long_target_miles:9,
    long_goal_miles:17,run_frequency:'balanced',frequency_lower_per_7:3,frequency_upper_per_7:5};
  settings.training_background={weekly_miles:24,runs_per_week:4,regular_long_miles:8,maintaining_training:true};
  await page.reload();
  await page.getByRole('heading',{name:'Running progression',exact:true}).waitFor();
  assert.equal(await page.locator('[name=background_choice]').count(),0,'An adopted starting level cannot be reset from Setup');
  const preferencesPanel=page.locator('.setup-block').filter({has:page.getByRole('heading',{name:'Running progression',exact:true})});
  assert.equal(await preferencesPanel.locator('[name=planner_frequency]').count(),1);
  assert.equal(await preferencesPanel.locator('[name=planner_long_day]').count(),1);
  assert.equal(await preferencesPanel.locator('[name=planner_long_day]').inputValue(),'');
  assert.equal(await preferencesPanel.locator('[name=planner_long_day] option').count(),8);
  await preferencesPanel.locator('[name=planner_long_day]').selectOption('5');
  await preferencesPanel.locator('[name=planner_frequency]').selectOption('fewer');
  for(const width of [1100,390]) {
    await page.setViewportSize({width,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Preferences do not overflow at ${width}`);
  }
  await page.setViewportSize({width:1100,height:900});
  await preferencesPanel.screenshot({path:'/tmp/setup-running-preferences.png'});
  assert.equal(await page.locator('#estimate-max-hr').isDisabled(),false);
  assert.equal(await page.getByRole('button',{name:'Apply training goal',exact:true}).isVisible(),false);
  await page.getByRole('button',{name:'Save setup',exact:true}).click();
  await page.getByText('Setup saved.',{exact:true}).waitFor();
  const preferencesRequest=requests.findLast(r=>r.path==='/api/settings'&&r.body?.planner_preferences)?.body.planner_preferences;
  assert.equal(preferencesRequest.long_run_weekday,5);
  assert.equal(preferencesRequest.run_frequency,'fewer');
  assert.equal(preferencesRequest.expected_revision,'preferences-current');
  assert.equal(requests.filter(r=>r.path==='/api/settings'&&r.method==='PATCH').at(-1).body.training_background,undefined);
  assert.ok(requests.some(r=>r.path==='/api/settings/race-preparation/apply'));
  assert.ok(settings.setup.confirmed_steps.includes('goal'));
  assert.deepEqual(errors, []);
  assert.ok(requests.every(r => !/dashboard|weekly|refresh|recommendation/.test(r.path)));
  console.log('Setup ordering, age estimate, pending zone preview, failed/successful save and numeric baseline guidance passed. No live API calls.');
} finally { await browser.close(); }
