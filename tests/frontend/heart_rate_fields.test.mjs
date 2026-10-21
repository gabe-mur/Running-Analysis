import { test } from 'node:test';
import assert from 'node:assert/strict';
import { heartRateFields, syncZoneEditStatus } from '../../src/run_analysis/web/static/app/setup-fields.js';

const settings = {max_hr:194, resting_hr:46, profile:{max_hr_source:'measured'},
  zones:Object.fromEntries(['z1','z2','z3','z4','z5'].map((name,i)=>[name,
    {minimum_bpm:128+i*13,maximum_bpm:140+i*13}]))};

test('shared HR panel puts watch/method first, then both HR inputs and zone rows',()=>{
  const html=heartRateFields(settings);
  assert.ok(html.indexOf('Match my watch') < html.indexOf('Heart-rate reserve (HRR)'));
  assert.ok(html.indexOf('name="method"') < html.indexOf('name="max_hr"'));
  assert.ok(html.indexOf('name="resting_hr"') < html.indexOf('name="z1_min"'));
  for (const name of ['max_hr','resting_hr','max_hr_source', 'z1_min','z1_max','z2_min','z2_max']) {
    assert.equal((html.match(new RegExp(`name="${name}"`,'g'))||[]).length,1);
  }
  assert.match(html, /value="measured" selected/);
  assert.equal((html.match(/class="zone-calibration-row/g)||[]).length,5);
  assert.doesNotMatch(html,/Comparison|target_hr/);
});

test('read-only zone note explains the source and how to enable range edits',()=>{
  for (const [method,source] of [['device','matched to your watch'],['heart_rate_reserve','maximum and resting HR'],['percent_max','maximum HR']]) {
    const html=heartRateFields(settings,{method});
    assert.ok(html.includes(source));
    assert.match(html,/Choose Custom zones to edit ranges|choose Custom zones to edit ranges/);
    assert.match(html,/aria-describedby="zone-edit-status"/);
    assert.doesNotMatch(html,/data-zone-edit-status role="status" hidden/);
  }
  assert.match(heartRateFields(settings,{method:'custom'}),/data-zone-edit-status role="status" hidden/);
});

test('zone edit note follows the selected method and disappears for custom editing',()=>{
  const note={textContent:'',hidden:false};
  const container={querySelector:()=>note};
  syncZoneEditStatus(container,'device');
  assert.match(note.textContent,/Read-only: matched to your watch/);
  assert.equal(note.hidden,false);
  syncZoneEditStatus(container,'heart_rate_reserve');
  assert.match(note.textContent,/maximum and resting HR/);
  syncZoneEditStatus(container,'percent_max');
  assert.match(note.textContent,/Calculated from maximum HR/);
  assert.doesNotMatch(note.textContent,/resting/);
  syncZoneEditStatus(container,'custom');
  assert.equal(note.hidden,true);
  assert.equal(note.textContent,'');
  syncZoneEditStatus(container,'device');
  assert.equal(note.hidden,false);
});

test('shared HR panel preserves active locks and view-specific control identifiers',()=>{
  const html=heartRateFields(settings,{locked:true, method:'custom',
    estimateId:'settings-estimate-max',estimateStatusId:'settings-max-estimate-status'});
  assert.match(html,/name="max_hr"[^>]*disabled/);
  assert.match(html,/name="resting_hr"[^>]*disabled/);
  assert.match(html,/id="settings-estimate-max" disabled/);
  assert.match(html,/id="settings-max-estimate-status"/);
  assert.match(html,/data-method="custom"/);
  assert.match(html,/name="method" value="custom" checked/);
  assert.match(html,/locked for the active plan/);
});
