import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPlanEvents,onPlanLoaded} from '../../src/run_analysis/web/static/app/plan-events.js';
import {api} from '../../src/run_analysis/web/static/app/api.js';

test('plan events refresh locally and across tabs without transmitting a plan',()=>{
  let received,updates=0,posted;
  const events=createPlanEvents({listen:fn=>{received=fn;},broadcast:(key,value)=>{posted={key,newValue:value};},token:()=> 'nonce'});
  const off=events.subscribe(()=>updates++);
  events.loaded();
  assert.equal(updates,1);
  assert.equal(posted.newValue,'nonce');
  received({key:'unrelated',newValue:'x'});
  assert.equal(updates,1);
  received(posted);
  assert.equal(updates,2);
  off();events.loaded();
  assert.equal(updates,2);
});

test('blocked storage and a failed UI listener do not fail publication',()=>{
  let calls=0;
  const events=createPlanEvents({broadcast:()=>{throw Error('Blocked');}});
  events.subscribe(()=>{throw Error('Detached UI');});
  events.subscribe(()=>calls++);
  assert.doesNotThrow(()=>events.loaded());
  assert.equal(calls,1);
});

test('successful planner load/update/rest requests notify even when the initiating page is gone',async()=>{
  let calls=0;
  const off=onPlanLoaded(()=>calls++);
  globalThis.fetch=async()=>({ok:true,json:async()=>({planner_status:'ready'})});
  try {
    for(const path of ['/api/weekly-schedule/latest','/api/weekly-schedule','/api/weekly-schedule/rest-day']) await api(path);
    assert.equal(calls,3);
    await api('/api/dashboard');await api('/api/weekly-schedule/saved');
    assert.equal(calls,3); // Saved reads cannot loop into a planner refresh.
    globalThis.fetch=async()=>({ok:true,json:async()=>null});
    await api('/api/weekly-schedule/latest');assert.equal(calls,3);
    globalThis.fetch=async()=>({ok:false,status:409,json:async()=>({detail:'Needs review'})});
    await assert.rejects(api('/api/weekly-schedule/latest'),/Needs review/);
    assert.equal(calls,3);
  } finally {off();}
});
