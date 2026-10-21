// Fully intercepted API: no live planner, uploads, settings or check-in writes.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {chromium}=createRequire(import.meta.url)('playwright');
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],writes=[];
  page.on('pageerror',e=>errors.push(e.message));
  let checked=false;
  const plan=()=>({planner_status:'ready',start_date:'2026-10-22',end_date:'2026-10-28',
    run_count:3,target_run_count:3,projected_distance_range_miles:[0,0],target_distance_range_miles:[0,0],
    target_evidence:{planning_mode:'baseline_building'},trailing_days:[],
    beginner_training:{status:'building',running_minutes:8,summary:'Build comfortable running time, with a rest day between outings.',
      next_feedback:checked?null:{activity_id:5,needs_times:true}},
    days:Array.from({length:7},(_,i)=>({date:`2026-10-${22+i}`,day_role:i%2?'rest_recovery_day':'easy',rationale:'',
      recommendation:i%2?null:{title:'Easy run/walk',workout_type:'run_walk',planning_role:'easy',
        readiness:'unknown',duration_range_minutes:[28.5,28.5],target_zones:['Z1','Z2'],rule_trace:[],
        structure:[{instruction:'5 min · Walk easily.'},{instruction:'8 × 1 min of easy running. Walk 1:30 between running stretches.'},{instruction:'5 min · Walk easily.'}]}}))});
  await page.route('**/api/**',async route=>{
    const url=new URL(route.request().url());
    let body={};
    if(url.pathname==='/api/settings')body={target_hr:145,planner:{status:'introductory'}};
    else if(url.pathname==='/api/current-status')body={health_status:'normal',health_report_current:true};
    else if(url.pathname==='/api/fitness-state')body={recent_load:{trailing_7d:{distance_miles:3}},running_days_28d:1};
    else if(url.pathname==='/api/goal-progress')body=null;
    else if(url.pathname==='/api/weekly-schedule/jobs')body={job_id:'mock-starter',state:'complete',result:plan()};
    else if(url.pathname==='/api/beginner-feedback/5'){
      writes.push(route.request().postDataJSON());checked=true;body={saved:true};
    }
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
  });
  await page.goto('http://127.0.0.1:8000/#next-run');
  await page.locator('.beginner-training').waitFor();
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1100});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.ok(await page.getByRole('button',{name:'Yes, comfortable'}).isVisible());
    assert.ok(await page.getByRole('button',{name:'No, too hard'}).isVisible());
    assert.equal(await page.locator('.baseline-guide').count(),0);
    assert.equal(await page.getByText('One observed session at a time',{exact:true}).count(),0);
    await page.locator('.beginner-training').screenshot({path:`/tmp/beginner-start-${width}.png`});
  }
  await page.locator('[name="running_minutes"]').fill('8');
  await page.locator('[name="longest_bout_minutes"]').fill('1');
  await page.getByRole('button',{name:'Yes, comfortable'}).click();
  await page.locator('[data-beginner-feedback]').waitFor({state:'detached'});
  assert.deepEqual(writes,[{comfortable:true,running_minutes:8,longest_bout_minutes:1}]);
  assert.deepEqual(errors,[]);
  console.log('Starter layout and check-in passed at 1440px and 390px; all API requests intercepted.');
} finally {await browser.close();}
