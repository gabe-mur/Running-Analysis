import {test} from 'node:test';
import assert from 'node:assert/strict';
import {appSettings,invalidateSettings,primeSettings} from '../../src/run_analysis/web/static/app/api.js';

test('settings reads coalesce and an old in-flight response cannot overwrite a saved value',async()=>{
  const original=globalThis.fetch;
  const pending=[];
  globalThis.fetch=()=>new Promise(resolve=>pending.push(value=>resolve({ok:true,json:async()=>value})));
  try {
    invalidateSettings();
    const a=appSettings(),b=appSettings();assert.equal(pending.length,1);
    pending.shift()({target_hr:148});assert.deepEqual(await a,await b);
    await appSettings();assert.equal(pending.length,0);
    invalidateSettings();const old=appSettings();
    primeSettings({target_hr:150});pending.shift()({target_hr:140});await old;
    assert.equal((await appSettings()).target_hr,150);
    invalidateSettings();const fresh=appSettings();pending.shift()({target_hr:151});
    assert.equal((await fresh).target_hr,151);
  } finally {globalThis.fetch=original;invalidateSettings();}
});
