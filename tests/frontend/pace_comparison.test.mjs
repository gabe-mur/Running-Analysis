import {test} from 'node:test';
import assert from 'node:assert/strict';
const page={innerHTML:'',querySelectorAll:()=>[]};
globalThis.document={querySelector:()=>page};
globalThis.location={hash:'#pace-comparison'};
const {comparisonMarkup,renderPaceComparison}=await import('../../src/run_analysis/web/static/app/views/pace-comparison.js');

const point={activity_id:1,start_time:'2026-10-10T10:00:00Z',pace_min_mile:10,
  observed_pace_min_mile:10.5,workout_type:'easy',terrain_coverage:.8,weather_coverage:1,
  estimated_range_95_min_mile:[9.5,10.5],contributions_seconds_per_mile:{terrain:-12,wind:0,heart_rate:-8,duration:0,weather:-10}};
const data={runs:[point],series:[{activity_id:1,start_time:point.start_time,standardized_pace_min_mile:10,
  raw_pace_min_mile:10.5,trend_weight:1,uncertainty_95_min_mile:.5,distance_miles:4}],
  trend:[{as_of:point.start_time,pace_min_mile:10}],recorded_trend:[{as_of:point.start_time,pace_min_mile:10.5}],
  as_of:'2026-10-13T12:00:00Z',timezone:'America/New_York',window_days:90,available_windows:[14,28,90,365],
  trend_runs_in_view:1,calibration:{target_hr_bpm:148}};

test('preview overlays matched trends and exposes every stage without rewriting Progress',()=>{
  const html=comparisonMarkup(data);
  assert.match(html,/Recorded versus comparable pace/);
  assert.match(html,/chart-line trend-28/);
  assert.match(html,/chart-line raw-trend/);
  for(const label of ['Terrain','Wind','Heart rate','Time into run','Heat + humidity'])assert.ok(html.includes(label));
  assert.match(html,/80% terrain coverage/);
  assert.match(html,/#run\/1/);
  assert.match(html,/does not replace Progress/);
  assert.doesNotMatch(html,/NaN|undefined/);
});

test('empty preview is readable and still names comparison HR',()=>{
  const html=comparisonMarkup({...data,runs:[],series:[],trend:[],recorded_trend:[],trend_runs_in_view:0});
  assert.match(html,/148 bpm/);
  assert.match(html,/No comparable sections/);
  assert.doesNotMatch(html,/NaN|undefined/);
});

test('tiny adjustments round to zero without a negative-zero label',()=>{
  const html=comparisonMarkup({...data,runs:[{...point,contributions_seconds_per_mile:{wind:-.2}}]});
  assert.doesNotMatch(html,/-0 sec\/mi/);
  assert.match(html,/>0 sec\/mi/);
});

test('withheld terrain is not presented as a measured zero correction',()=>{
  const html=comparisonMarkup({...data,runs:[{...point,terrain_coverage:0,
    terrain_quality:{usable:false,status:'rejected',reason:'Conflicting elevations <test>.'},
    contributions_seconds_per_mile:{terrain:0}}],
    series:data.series.map(item=>({...item,terrain_quality:{usable:false}}))});
  assert.match(html,/<dt>Terrain<\/dt><dd>Not applied<\/dd>/);
  assert.match(html,/Terrain adjustment withheld/);
  assert.match(html,/Conflicting elevations &lt;test&gt;/);
  assert.match(html,/Terrain adjustment not applied/);
  assert.doesNotMatch(html,/<test>/);
});

test('preview fetches its own optional endpoint only',async()=>{
  const calls=[];globalThis.fetch=async path=>{calls.push(path);return {ok:true,json:async()=>data};};
  await renderPaceComparison();
  assert.deepEqual(calls,['/api/pace-comparability?window_days=90']);
  assert.match(page.innerHTML,/Compare your runs/);
});

test('provisional recorded bridge grades are labelled without claiming measured height',()=>{
  const quality={usable:true,status:'provisional',surface_profile_uncertain:true,source:'mixed_recorded_surface'};
  const html=comparisonMarkup({...data,runs:[{...point,terrain_quality:quality}],
    series:data.series.map(item=>({...item,terrain_quality:quality}))});
  assert.match(html,/Recorded grades used where the ground map may describe a different surface/);
  assert.match(html,/Surface grades estimated from recorded elevation/);
});

test('navigation away prevents stale comparison from replacing another page',async()=>{
  globalThis.fetch=async()=>{location.hash='#settings';return {ok:true,json:async()=>data};};
  await renderPaceComparison();
  assert.doesNotMatch(page.innerHTML,/Compare your runs/);
  location.hash='#pace-comparison';
});
