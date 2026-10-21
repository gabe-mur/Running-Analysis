import { test } from 'node:test';
import assert from 'node:assert/strict';

let panel;
const form = { addEventListener() {} };
const page = { innerHTML: '', querySelector: selector => selector === '[data-saved-next-run]' ? panel : form };
globalThis.document = { querySelector: () => page };
const { renderRunDetail, renderSavedNextRun, refreshSavedNextRun } = await import('../../src/run_analysis/web/static/app/views/run-detail.js');
const turn = () => new Promise(resolve => setImmediate(resolve));

const feedback = {
  run: { activity_id: 42, workout_type: 'easy', data_quality: 'good', gps_quality: 'good',
    start_time: '2026-10-12T18:00:00-04:00', distance_miles: 4.4, moving_pace_min_mile: 10,
    average_hr_bpm: 148, session_difficulty: { zone_breakdown: { zone_fractions: { z2: 1 } } } },
  metadata: { workout_type: 'easy', health_tag: 'normal', include_in_model: null, notes: '' },
  splits: [], cautions: [], positives: [], assessment: 'Easy aerobic running.',
  cardiac_drift: { valid: false, reason: 'Not enough steady running.' },
};
const weekly = { run_count: 4, projected_distance_range_miles: [20, 20], planner_status: 'ready',
  days: [{ date: '2026-10-14', recommendation: { title: 'Intervals', distance_target_miles: 3.8 } }] };

test('shared comparison equation includes HR/time as well as environmental adjustments',async()=>{
  const current={...feedback,run:{...feedback.run,fitness_observation:{
    pace_basis:'shared-aerobic-comparison-v1',confidence:'low',contributions:[],
    raw_pace_at_target_hr:{minutes_per_mile:11,display:'11:00/mi'},
    standardized_pace_at_target_hr:{minutes_per_mile:10,display:'10:00/mi'},
    environmental_adjustment_min_mile:-.2}}};
  panel={isConnected:true,outerHTML:''};
  globalThis.fetch=async path=>({ok:true,json:async()=>path.includes('/saved')?{needs_refresh:false,schedule:weekly}:current});
  await renderRunDetail(42);
  assert.match(page.innerHTML,/recorded sections/);
  assert.match(page.innerHTML,/comparable pace/);
  assert.match(page.innerHTML,/− 60 sec\/mi/);
  assert.doesNotMatch(page.innerHTML,/− 12 sec\/mi|before conditions|after conditions/);
});

test('run analysis renders without waiting for even the saved-plan read', async () => {
  let resolveSaved;
  const calls = [];
  panel = { isConnected: true, outerHTML: '' };
  globalThis.fetch = async path => {
    calls.push(path);
    if (path === '/api/weekly-schedule/saved') return new Promise(resolve => { resolveSaved = resolve; });
    assert.equal(path, '/api/runs/42');
    return { ok: true, json: async () => feedback };
  };
  await renderRunDetail(42);
  assert.match(page.innerHTML, /4\.4 miles/);
  assert.match(page.innerHTML, /Easy aerobic running/);
  assert.equal(panel.outerHTML, '');
  assert.deepEqual(calls, ['/api/runs/42', '/api/weekly-schedule/saved']);
  resolveSaved({ ok: true, json: async () => ({ needs_refresh: true, schedule: weekly }) });
  await turn();
  assert.match(panel.outerHTML, /Plan needs updating/);
  assert.doesNotMatch(panel.outerHTML, /Intervals|3\.8/);
});

test('a late saved-plan response cannot overwrite another route', async () => {
  let resolveSaved;
  panel = { isConnected: true, outerHTML: '' };
  globalThis.fetch = async path => path.includes('/saved')
    ? new Promise(resolve => { resolveSaved = resolve; })
    : { ok: true, json: async () => feedback };
  await renderRunDetail(42);
  panel.isConnected = false;
  resolveSaved({ ok: true, json: async () => ({ needs_refresh: false, schedule: weekly }) });
  await turn();
  assert.equal(panel.outerHTML, '');
});

test('fresh saved plan shows the next workout; stale/review plans do not prescribe it', () => {
  assert.match(renderSavedNextRun({ schedule: weekly, needs_refresh: false }), /Intervals/);
  assert.match(renderSavedNextRun({ schedule: weekly, needs_refresh: false }), /3\.8 mi/);
  assert.doesNotMatch(renderSavedNextRun({ schedule: weekly, needs_refresh: true }), /Intervals|3\.8/);
  assert.match(renderSavedNextRun({ schedule: { ...weekly, planner_status: 'review_required' } }), /Review your plan/);
});

test('next-run weekly mileage matches the committed Dashboard total without a duplicate range',()=>{
  const state={schedule:{...weekly,visible_7d_scheduled_miles:20.2},needs_refresh:false};
  assert.match(renderSavedNextRun(state),/4 runs · 20\.2 mi/);
  delete state.schedule.visible_7d_scheduled_miles;
  assert.match(renderSavedNextRun(state),/4 runs · 20\.0 mi/);
  state.schedule.projected_distance_range_miles=[20,21];
  assert.match(renderSavedNextRun(state),/4 runs · 20\.0–21\.0 mi/);
});

test('new saved-plan read replaces a pending panel and ignores an older in-flight response',async()=>{
  panel={isConnected:true,outerHTML:renderSavedNextRun({needs_refresh:true,schedule:weekly})};
  let oldResolve;
  const old=refreshSavedNextRun(()=>new Promise(resolve=>{oldResolve=resolve;}));
  await refreshSavedNextRun(async path=>{
    assert.equal(path,'/api/weekly-schedule/saved');
    return {needs_refresh:false,schedule:weekly};
  });
  assert.match(panel.outerHTML,/Intervals|3\.8 mi/);
  assert.match(panel.outerHTML,/data-saved-next-run/);
  oldResolve({needs_refresh:true,schedule:weekly});
  await old;
  assert.doesNotMatch(panel.outerHTML,/Plan needs updating/);
});

test('panel refresh does not overwrite analysis or form values and follows server freshness',async()=>{
  panel={isConnected:true,outerHTML:''};
  page.innerHTML='Analysis and unsaved form edits';
  await refreshSavedNextRun(async()=>({needs_refresh:true,schedule:weekly}));
  assert.match(panel.outerHTML,/Plan needs updating/);
  assert.equal(page.innerHTML,'Analysis and unsaved form edits');
});
