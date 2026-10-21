import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPlannerRequest} from '../../src/run_analysis/web/static/app/planner-request.js';

test('short submission/polls preserve telemetry ID and deliver the exact result',async()=>{
  const calls=[],waits=[],plan={planner_status:'ready',days:[{date:'2026-10-20'}]};let notified=0;
  const replies=[{job_id:'job',state:'queued'},{job_id:'job',state:'running'},
    {job_id:'job',state:'complete',result:plan}];
  const request=createPlannerRequest({fetcher:async(...args)=>{calls.push(args);return replies.shift();},
    pause:async ms=>waits.push(ms),announce:()=>notified++});
  assert.equal(await request('/api/weekly-schedule/latest',{headers:{'X-Planner-Request':'telemetry'}}),plan);
  assert.deepEqual(JSON.parse(calls[0][1].body),{request_id:'telemetry',operation:'refresh'});
  assert.deepEqual(calls.map(c=>c[0]),['/api/weekly-schedule/jobs',
    '/api/weekly-schedule/jobs/job','/api/weekly-schedule/jobs/job']);
  assert.deepEqual(waits,[750]);assert.equal(notified,1);
});

test('edits use their original fields; accepted jobs do not depend on page abort',async()=>{
  const calls=[],abort=new AbortController();abort.abort();let notified=0;
  const request=createPlannerRequest({fetcher:async(path,options)=>{
    calls.push([path,options]);return {job_id:'shared',state:'complete',result:{planner_status:'ready'}};
  },uuid:()=> 'id',announce:()=>notified++});
  await request('/api/weekly-schedule',{body:JSON.stringify({health_status:'tired'}),signal:abort.signal});
  await request('/api/weekly-schedule/rest-day',{body:JSON.stringify({date:'2026-10-20',is_rest_day:true})});
  assert.deepEqual(JSON.parse(calls[0][1].body),{request_id:'id',operation:'update',update:{health_status:'tired'}});
  assert.deepEqual(JSON.parse(calls[1][1].body),{request_id:'id',operation:'rest',rest:{date:'2026-10-20',is_rest_day:true}});
  assert.equal(calls[0][1].signal,undefined);assert.equal(notified,1);
});

test('review errors and missing/expired status fail explicitly without a new search',async()=>{
  for(const reply of [{job_id:'x',state:'failed',error:{status:409,detail:'Review required'}},
    {job_id:'x',state:'unknown'},{}]) {
    let calls=0,notified=0;
    const request=createPlannerRequest({uuid:()=> 'id',announce:()=>notified++,
      fetcher:async()=>{calls++;return reply;}});
    await assert.rejects(request('/api/weekly-schedule/latest'),/Review required|unavailable|job ID/);
    assert.equal(calls,1);assert.equal(notified,0);
  }
});
