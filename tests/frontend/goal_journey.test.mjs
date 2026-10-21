import {test} from 'node:test';
import assert from 'node:assert/strict';
import {goalJourneyPanel,goalJourneyChart,paceOutlookChart,wireGoalOutlook} from '../../src/run_analysis/web/static/app/goal-journey.js';

const data={available:true,as_of:'2026-10-21T01:00:00Z',today:'2026-10-20',status:'supporting_volume',
  goal_kind:'general_fitness',target_long_miles:17,target_volume_per_7:42.5,current_long_miles:8.9,
  recent_long_run:{date:'2026-10-05',miles:9.536,activity_id:386},volume_per_7:19.4,
  supporting_volume_for_long:22.25,volume_growth_per_7:.04,long_growth_allowed:false,
  recorded:[{activity_id:386,date:'2026-10-05',miles:9.536}],scheduled:[{date:'2026-10-26',miles:8.9}],
  projection:{target_date:'2027-05-19',points:[{date:'2026-10-20',long_miles:8.9},{date:'2027-05-19',long_miles:17}]}};
const date=s=>s.slice(0,10);

test('distinguishes recorded accomplishment, prescription and future target',()=>{
  const html=goalJourneyPanel(data,date);
  assert.match(html,/Recent long run[^]*?9.5/);
  assert.match(html,/Current long-run prescription[^]*?8.9 mi · held/);
  assert.match(html,/17-mile weekly long run/);
  assert.match(html,/Building supporting volume/);
  assert.match(html,/4% per week/);assert.match(html,/22.3 mi/);
  assert.match(html,/May 2027/);
  assert.match(html,/not a future schedule/);
  assert.match(html,/href="#run\/386"/);
  assert.doesNotMatch(html,/NaN|undefined|VO2|race pace/i);
});
test('completed, scheduled and conditional marks have distinct visuals and inspection',()=>{
  const html=goalJourneyChart(data,date);
  assert.match(html,/journey-recorded/);assert.match(html,/journey-scheduled/);assert.match(html,/journey-projection/);
  assert.match(html,/data-run-title="2026-10-05"/);assert.match(html,/scheduled long run/);
  assert.doesNotMatch(html,/href="#run\/planned/);
});
test('paused or stale progression does not render an old completion date',()=>{
  for(const status of ['holding','update_pending','review','taper']) {
    const html=goalJourneyPanel({...data,status,projection:{points:[],target_date:null}},date);
    assert.doesNotMatch(html,/May 2027|Target at this rate|journey-projection"/);
  }
});
test('race prep reports its taper scenario instead of inventing a finish pace',()=>{
  const html=goalJourneyPanel({...data,goal_kind:'marathon',race_date:'2026-12-20',taper_start:'2026-11-29',
    projection:{points:[{date:'2026-10-20',long_miles:8.9},{date:'2026-11-29',long_miles:10.4}],
      target_date:null,taper:{long_miles:10.4}},race_projection_meets_target:false},date);
  assert.match(html,/Marathon · 2026-12-20/);assert.match(html,/10.4 mi by taper/);
  assert.match(html,/Below the preparation target at taper/);
  assert.doesNotMatch(html,/finish time|min\/mi|race pace/i);
});
test('new users get useful absence rather than fictional zero progress',()=>{
  const html=goalJourneyPanel({available:false,reason:'Create your Weekly Plan first.'},date);
  assert.match(html,/Create your Weekly Plan first/);
  assert.doesNotMatch(html,/0%|journey-chart|Target at this rate/);
});

const outlook={available:true,date:'2027-05-19',date_basis:'target',reference_hr_bpm:148,
  current_pace_min_mile:10.15,expected_pace_min_mile:9.1377,low_pace_min_mile:8.8787,high_pace_min_mile:9.3389,
  history:[{date:'2026-08-01',pace_min_mile:10.6},{date:'2026-10-19',pace_min_mile:10.15}],
  points:[{date:'2026-10-20',pace_min_mile:10.15,low_pace_min_mile:null,high_pace_min_mile:null},
    {date:'2026-10-27',pace_min_mile:10.1,low_pace_min_mile:null,high_pace_min_mile:null},
    {date:'2026-11-03',pace_min_mile:10.05,low_pace_min_mile:9.6,high_pace_min_mile:10.5},
    {date:'2027-05-19',pace_min_mile:9.1377,low_pace_min_mile:8.8787,high_pace_min_mile:9.3389}]};
test('compact goal window shares distance/pace space and retains forecast data in shorter plots',()=>{
  const input={...data,pace_outlook:outlook},before=structuredClone(input);
  const html=goalJourneyPanel(input,date,{compact:true});
  assert.match(html,/data-goal-outlook="distance"[^>]*aria-pressed="true"/);
  assert.match(html,/data-goal-panel="distance">/);
  assert.match(html,/data-goal-panel="pace" hidden/);
  assert.match(html,/<details class="journey-details"><summary>Progression details<\/summary>[^]*?Current long-run prescription[^]*?8.9 mi · held[^]*?<\/details>/);
  assert.match(html,/Recent long run[^]*?9.5/);
  assert.ok(html.includes(goalJourneyChart(input,date,{compact:true})));
  assert.ok(html.includes(paceOutlookChart(input,date,{compact:true})));
  assert.match(html,/goal-journey-title[^]*?<a class="journey-edit primary-button"/);
  assert.match(html,/goal-journey-chart[^]*?<div class="journey-key">/);
  assert.match(html,/pace-outlook-chart[^]*?<div class="journey-key">/);
  const paceHtml=goalJourneyPanel(input,date,{compact:true,outlook:'pace'});
  assert.match(paceHtml,/data-goal-panel="distance" hidden/);
  assert.match(paceHtml,/data-goal-panel="pace">/);
  assert.deepEqual(input,before);
});
test('compact outlook switch changes only visibility and selection, with no fetching',()=>{
  const buttons=['distance','pace'].map(key=>({dataset:{goalOutlook:key},attributes:{},
    classList:{toggle(name,value){this[name]=value;}},
    setAttribute(name,value){this.attributes[name]=value;},addEventListener(name,fn){this[name]=fn;}}));
  const panels=['distance','pace'].map(key=>({dataset:{goalPanel:key},hidden:key!=='distance'}));
  const changed=[],root={querySelectorAll:selector=>selector==='[data-goal-outlook]'?buttons:panels};
  wireGoalOutlook(root,key=>changed.push(key));
  buttons[1].click();
  assert.deepEqual(panels.map(p=>p.hidden),[true,false]);
  assert.deepEqual(buttons.map(b=>b.attributes['aria-pressed']),['false','true']);
  buttons[0].click();
  assert.deepEqual(panels.map(p=>p.hidden),[false,true]);
  assert.deepEqual(changed,['pace','distance']);
});
test('compact unavailable outlook stays useful and pending-goal warning survives either view',()=>{
  const html=goalJourneyPanel({...data,goal_change_pending:true},date,{compact:true,outlook:'pace'});
  assert.match(html,/More comparable runs are needed/);
  assert.match(html,/Goal change pending\. This shows the issued plan/);
  assert.doesNotMatch(html,/NaN|undefined|outlook-value/);
});
test('goal card shows expected aerobic pace and rough range, without a race-pace promise',()=>{
  const html=goalJourneyPanel({...data,pace_outlook:outlook},date);
  assert.match(html,/Projected aerobic pace/);assert.match(html,/9:10\/mi/);
  assert.match(html,/Rough range/);assert.match(html,/8:55\/mi–9:20\/mi/);
  assert.match(html,/Now 10:10\/mi · at 148 bpm/);
  assert.match(html,/Updates with completed training/);
  assert.match(html,/not a confidence interval/);
  assert.match(html,/midpoint, not a separately fitted expected value/);
  assert.match(html,/42-day aerobic trend/);assert.match(html,/errors predicting actual future runs/);
  assert.match(html,/If training stays consistent/);
  assert.match(html,/not a prescribed pace/);
  assert.doesNotMatch(html,/NaN|undefined/);
});
test('forecast band uses supported bounds without connecting to a false zero-certainty origin',()=>{
  const html=paceOutlookChart({...data,pace_outlook:outlook},date);
  assert.match(html,/outlook-range-marker/);assert.match(html,/outlook-history/);assert.match(html,/outlook-line/);
  assert.match(html,/Faster ↑/);assert.match(html,/data-run-title="2027-05-19"/);
  assert.match(html,/Projected pace: 9:10\/mi/);
  assert.doesNotMatch(html,/NaN|undefined|>0:00/);
  assert.match(html,/forecast-error band where estimates are available/);
  assert.equal((html.match(/class="outlook-band"/g)||[]).length,1);
  assert.equal((html.match(/class="outlook-bound"/g)||[]).length,2);
  const bandStart=Number(html.match(/class="outlook-band" d="M([^,]+)/)[1]);
  const firstSupported=Number(html.match(/data-run-key="pace-outlook-2" data-run-x="([^"]+)/)[1]);
  assert.ok(Math.abs(bandStart-firstSupported)<.051);
  const earlyHover=html.match(/<g [^>]*data-run-key="pace-outlook-1"[^]*?<\/g>/)?.[0];
  assert.ok(earlyHover);
  assert.doesNotMatch(earlyHover,/Rough range/);
  assert.equal((html.match(/class="outlook-bound-label"/g)||[]).length,2);
  assert.doesNotMatch(html,/href="#run/);
});
test('unsupported gaps split bands and a lone range only shows the endpoint marker',()=>{
  const base=outlook.points[2],last=outlook.points.at(-1);
  const points=[...outlook.points.slice(0,3),{...base,date:'2026-11-10'},
    {date:'2026-11-17',pace_min_mile:10,low_pace_min_mile:null,high_pace_min_mile:null},
    {...base,date:'2026-11-24'},last];
  const html=paceOutlookChart({...data,pace_outlook:{...outlook,points}},date);
  assert.equal((html.match(/class="outlook-band"/g)||[]).length,2);
  const single=paceOutlookChart({...data,pace_outlook:{...outlook,points:[outlook.points[0],last]}},date);
  assert.doesNotMatch(single,/class="outlook-band"/);
  assert.match(single,/outlook-range-marker/);
});
test('no forecast data yields a simple useful absence, not zeros',()=>{
  const html=goalJourneyPanel({...data,pace_outlook:{available:false,reason:'More completed training is needed.'}},date);
  assert.match(html,/More completed training is needed/);
  assert.doesNotMatch(html,/pace-outlook-chart|outlook-value|0:00/);
});
