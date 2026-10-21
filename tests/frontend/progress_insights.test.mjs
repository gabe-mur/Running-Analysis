import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performanceSummary, benchmarkHeadline, loadComparison, qualitySessionLabel, vo2EstimateMarkup } from '../../src/run_analysis/web/static/app/progress-insights.js';
import { fitnessChart, paceChartScale, vo2Chart } from '../../src/run_analysis/web/static/app/charts.js';
import { qualityDose, qualityProgress, qualityWorkHistory, durabilityProgress, trainingVolumeChart, comparisonHistory } from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

const improving={direction:'improving',evidence:'clear',pace_change_seconds_per_mile:-34,probability_faster:.98};
test('new-model headlines describe broad progress while the pace number remains current',()=>{
  const state={available:true,target_hr_bpm:148,current:{pace_min_mile:10},days_since_observation:1,
    progress_trend:{direction:'improving',evidence:'likely',span_days:84,early_estimate:false,pace_change_seconds_per_mile:-31.4},
    unadjusted:{available:true,current:{pace_min_mile:11},progress_trend:{direction:'declining',evidence:'clear',span_days:84}}};
  assert.equal(benchmarkHeadline(state).headline,'Your aerobic pace is likely improving.');
  assert.equal(benchmarkHeadline(state,'both').headline,benchmarkHeadline(state).headline);
  assert.equal(benchmarkHeadline(state,'raw').headline,'Your recorded pace is slowing.');
  assert.equal(benchmarkHeadline(state,'raw').value,11);
  assert.equal(benchmarkHeadline(state).valueLabel,'At 148 bpm');
  assert.equal(benchmarkHeadline({...state,progress_trend:{direction:'maintaining'}}).headline,'Maintaining aerobic efficiency.');
  assert.equal(benchmarkHeadline({...state,progress_trend:{direction:'insufficient_data'}}).headline,'Establishing your aerobic trend.');
  assert.equal(benchmarkHeadline(state).description,'12-week aerobic trend · adjusted for conditions.');
  assert.equal(benchmarkHeadline(state).basis,'Current smoothed pace');
  assert.equal(benchmarkHeadline(state).changeLabel,'Likely 31 sec/mi faster');
  assert.equal(benchmarkHeadline(state).changeBasis,'over the 12-week trend');
  const early=benchmarkHeadline({...state,progress_trend:{direction:'improving',evidence:'clear',span_days:28,early_estimate:true}});
  assert.equal(early.headline,'Your aerobic pace is likely improving.');
  assert.equal(early.description,'Early estimate · 4-week aerobic trend · adjusted for conditions.');
  assert.equal(benchmarkHeadline({...state,days_since_observation:29}).headline,'Your last aerobic pace estimate.');
  assert.equal(benchmarkHeadline(null).headline,'More comparable runs are needed.');
  assert.equal(benchmarkHeadline(null).value,null);
});
test('plain headline preserves evidence strength and separates different timescales',()=>{
  assert.equal(performanceSummary({period_change:improving}).changeLabel,'34 sec/mi faster');
  assert.match(performanceSummary({period_change:{...improving,evidence:'likely'}}).headline,/Likely faster/);
  const conflict=performanceSummary({period_change:improving,within_window_trend:{...improving,direction:'declining',probability_faster:.05}});
  assert.equal(conflict.direction,'improving');
  assert.equal(conflict.chosen,improving);
  assert.equal(conflict.headline,'Faster than the previous period.');
  assert.equal(conflict.recentTrajectory,'Recent pace is slowing.');
  assert.equal(conflict.changeLabel,'34 sec/mi faster');
  assert.equal(performanceSummary({period_change:improving},'4 weeks').headline,'Faster than the previous 4 weeks.');
  assert.equal(performanceSummary({}).direction,'insufficient_data');
  assert.equal(performanceSummary({fitness_trend:'uncertain'}).direction,'uncertain');
  assert.equal(performanceSummary({fitness_trend:'uncertain'}).headline,'Maintaining current pace.');
  assert.equal(performanceSummary({period_change:{...improving,direction:'uncertain',evidence:'unclear'}}).changeLabel,'No clear change');
});

test('VO2 displays shared confidence, pooled value, method and unavailable reason',()=>{
  const v={confidence:'moderate',value_ml_kg_min:52.7,uncertainty_95_ml_kg_min:4.7,
    method:'Shared ACSM & George',interpretation:'The equations agree.',limitations:['Not independent evidence.'],
    demographic_baseline_ml_kg_min:49.1,demographic_uncertainty_95_ml_kg_min:11.2};
  const markup=vo2EstimateMarkup(v);
  assert.match(markup,/52.7/); assert.doesNotMatch(markup,/Estimated range:|class="vo2-range"/);
  assert.match(markup,/quality moderate/); assert.match(markup,/Shared ACSM &amp; George/);
  assert.match(markup,/Not independent evidence/);
  assert.match(markup,/49.1 ± 11.2/); assert.match(markup,/population figure, not a measurement of you/);
  assert.doesNotMatch(markup,/not averaged together|Rough estimate/);
  const missing=vo2EstimateMarkup({...v,value_ml_kg_min:null,uncertainty_95_ml_kg_min:null,confidence:'unavailable',interpretation:'Set your profile <first>.'});
  assert.match(missing,/Set your profile &lt;first&gt;/);
  assert.doesNotMatch(missing,/Estimated range|NaN/);
});

const total=(low,high,key='same')=>({complete:true,cardio:{low,high},accounting_key:key});
test('load comparisons do not call extra load good or turn missing coverage into zero',()=>{
  assert.equal(loadComparison({current:total(135,135),prior:total(100,100)}).label,'About 35% more');
  assert.equal(loadComparison({current:total(95,105),prior:total(100,100)}).direction,'flat');
  assert.equal(loadComparison({current:{...total(135,135),complete:false},prior:total(100,100)}).direction,'none');
  assert.equal(loadComparison({current:total(135,135,'different'),prior:total(100,100)}).direction,'none');
});

test('workout label follows actual detected structure, not prescription name',()=>{
  assert.equal(qualitySessionLabel({continuous:{structure:'progression_finish'}}),'Progression');
  assert.equal(qualitySessionLabel({interval:{available:true},continuous:null}),'Intervals');
  assert.equal(qualitySessionLabel({continuous:{structure:'tempo'}}),'Sustained effort');
  assert.equal(qualitySessionLabel({prescription:{name:'Intervals'}}),null);
});

const series=[{activity_id:1,start_time:'2026-09-10T12:00:00Z',distance_miles:4,standardized_pace_min_mile:10,raw_pace_min_mile:10.5,uncertainty_95_min_mile:2,trend_weight:1},
  {activity_id:2,start_time:'2026-09-15T12:00:00Z',distance_miles:4,standardized_pace_min_mile:9.5,raw_pace_min_mile:10,uncertainty_95_min_mile:.3,trend_weight:0}];
test('pace chart keeps all markers and exposes full uncertainty without stretching the axis',()=>{
  for(const metric of ['standardized','raw','both']) assert.equal(fitnessChart(series,metric),fitnessChart(series,metric,[],null,null,{}));
  const wip=fitnessChart(series,'standardized',[],null,null,{plainLanguage:true,interactive:true});
  assert.equal((wip.match(/<circle/g)||[]).length,2);
  assert.match(wip,/context-only/);
  assert.doesNotMatch(wip,/% influence/);
  assert.doesNotMatch(wip,/uncertainty-mark/);
  assert.match(wip,/Estimated adjusted range: 8:00\/mi–12:00\/mi/);
  const axis=html=>[...html.matchAll(/<text x="48"[^>]*>([^<]+)<\/text>/g)].map(match=>match[1]);
  assert.deepEqual(axis(wip),axis(fitnessChart(series.map(item=>({...item,uncertainty_95_min_mile:50})),'standardized')));
  assert.doesNotMatch(fitnessChart([{...series[0],uncertainty_95_min_mile:null}],'standardized'),/NaN|Estimated adjusted range/);
});

test('pace axes use readable round ticks and retain a nonzero single-run span',()=>{
  for(const values of [[9.6,11.1],[10],[5.1,20.8]]) {
    const {min,max,ticks}=paceChartScale(values);
    assert.ok(min<Math.min(...values)&&max>Math.max(...values));
    assert.ok(ticks.length>=2&&ticks.length<=7);
    assert.ok(ticks.every(value=>Number.isInteger(value*2)));
    assert.ok(ticks.every((value,index)=>index===0||value-ticks[index-1]===ticks[1]-ticks[0]));
  }
});

test('adjusted, raw and both share a scale that includes both curves but not error ranges',()=>{
  const adjusted=[{as_of:series[0].start_time,pace_min_mile:9}],raw=[{as_of:series[0].start_time,pace_min_mile:11.2}];
  const axis=html=>[...html.matchAll(/<text x="48"[^>]*>([^<]+)<\/text>/g)].map(match=>match[1]);
  const displays=['standardized','raw','both'].map(metric=>fitnessChart(series,metric,adjusted,null,null,{rawTrend:raw}));
  assert.deepEqual(axis(displays[0]),axis(displays[1]));
  assert.deepEqual(axis(displays[1]),axis(displays[2]));
  assert.deepEqual(axis(displays[0]),['8:30','9:00','9:30','10:00','10:30','11:00','11:30']);
  const changed=series.map(item=>({...item,uncertainty_95_min_mile:100}));
  assert.deepEqual(axis(displays[2]),axis(fitnessChart(changed,'both',adjusted,null,null,{rawTrend:raw})));
});

test('raw and both show the appropriate visibly distinct trends',()=>{
  const a=[{as_of:'2026-09-10T12:00:00Z',pace_min_mile:10}],b=[{as_of:'2026-09-10T12:00:00Z',pace_min_mile:10.5}];
  const raw=fitnessChart(series,'raw',a,null,null,{rawTrend:b,fitUncertainty:true});
  assert.match(raw,/Unadjusted trend/);
  assert.doesNotMatch(raw,/Adjusted trend/);
  assert.match(raw,/class="chart-line raw-trend"/);
  const both=fitnessChart(series,'both',a,null,null,{rawTrend:b,fitUncertainty:true});
  assert.match(both,/Adjusted trend/); assert.match(both,/Unadjusted trend/);
  assert.match(both,/class="chart-line trend-28"/);
  assert.match(both,/class="chart-line raw-trend"/);
});

test('current-level VO2 history does not draw a trend across missing evidence',()=>{
  const s=[{as_of:'2026-09-01',value_ml_kg_min:50,uncertainty_95_ml_kg_min:5},{as_of:'2026-10-01',value_ml_kg_min:51,uncertainty_95_ml_kg_min:5}];
  const markup=vo2Chart(s,{maxGapDays:2});
  assert.match(markup,/<path class="vo2-line" d="M[^L]+ M/);
  assert.match(markup,/<path class="vo2-band" d="[^\"]+Z M/);
});

test('quality work is classified by actual bout duration, excluding recovery',()=>{
  assert.deepEqual(qualityDose({interval:{available:true,repetitions:[{kind:'work',duration_seconds:60},{kind:'recovery',duration_seconds:90},{kind:'work',duration_seconds:300}]}}),{short:1,long_rep:5,continuous:0});
  assert.deepEqual(qualityDose({continuous:{duration_seconds:1200}}),{short:0,long_rep:0,continuous:20});
  assert.deepEqual(qualityDose({interval:{available:false}}),{short:0,long_rep:0,continuous:0});
});

test('broader history charts render empty data and partial work honestly',()=>{
  const d={insights:{days:180,start:'2026-04-07T00:00:00Z'},as_of:'2026-10-04T00:00:00Z',quality:[],durability:[],exposure:{daily:[{as_of:'2026-10-03',distance_7d:0,distance_by_role:{easy:0,long:0,quality:0},distance_28d_weekly:0},{as_of:'2026-10-04',distance_7d:0,distance_by_role:{easy:0,long:0,quality:0},distance_28d_weekly:0}]}};
  const date=v=>v.slice(0,10);
  assert.doesNotMatch(trainingVolumeChart(d,date),/NaN/);
  assert.match(durabilityProgress(d,date),/No comparable/);
  d.quality=[{start_time:'2026-10-01T00:00:00Z',interval:{available:false}}];
  assert.match(qualityWorkHistory(d,date),/At least 0/);
  assert.match(qualityWorkHistory(d,date),/Unresolved sessions leave gaps/);
});

test('quality summaries include work before the display boundary',()=>{
  const d={insights:{days:180,start:'2026-04-07T00:00:00Z',quality_history:[{start_time:'2026-04-01T00:00:00Z',continuous:{duration_seconds:1200}}]},as_of:'2026-10-04T00:00:00Z',quality:[]};
  assert.match(qualityWorkHistory(d,v=>v.slice(0,10)),/2026-04-07: short reps 0.0 min; longer reps 0.0 min; continuous 20.0 min/);
});

const period=(count,median)=>({count,median,low:median,high:median,first:'2026-09-01',last:'2026-09-01'});
const comparison=(status='different_conditions_or_sparse')=>({recent:period(1,7),earlier:period(2,7.5),delta:-.5,status});
test('quality shows measured performance with sparse and condition caveats, not work minutes',()=>{
  const d={insights:{quality_comparisons:{repeats:[{label:'About 400 m repeats',total_sessions:3,comparison:comparison()}],sustained:{total_sessions:2,comparison:{status:'insufficient_history'}}}}};
  const html=qualityProgress(d,v=>v);
  assert.match(html,/Repeat speed/); assert.match(html,/7:00/);
  assert.match(html,/1 session/); assert.match(html,/No fitness-change claim/);
  assert.match(html,/No comparable trend yet/);
  assert.doesNotMatch(html,/recorded work minutes|NaN|Infinity/);
});
test('endurance comparison labels ranges as observed, not statistical confidence',()=>{
  const d={insights:{endurance_comparisons:[{label:'Around 100 minutes',reference_hr:148,comparison:{...comparison('matched_observations'),recent:period(3,2),earlier:period(3,4),delta:-2}}]}};
  const html=durabilityProgress(d,v=>v);
  assert.match(html,/2.0%/); assert.match(html,/less drift in similar conditions/);
  assert.match(html,/not confidence intervals/); assert.match(html,/Less drift ←/);
  assert.doesNotMatch(html,/NaN|Infinity/);
});

test('quality comparisons label the scale, periods and observed difference without hiding sparse evidence',()=>{
  const d={insights:{quality_comparisons:{repeats:[{label:'About 400 m repeats',total_sessions:3,comparison:comparison()}],sustained:{total_sessions:2,comparison:{status:'insufficient_history'}}}}};
  const html=qualityProgress(d,v=>v);
  assert.match(html,/Last 6 weeks/);
  assert.match(html,/Earlier in the 6-month history/);
  assert.match(html,/30 sec\/mi faster/);
  assert.match(html,/Faster ← · min\/mi/);
  assert.match(html,/Small sample or different conditions/);
  assert.match(html,/not statistical confidence/);
  assert.doesNotMatch(html,/proof of improvement|NaN|Infinity/);
});

test('drift changes use percentage points and do not turn observations into a fitness verdict',()=>{
  const d={insights:{endurance_comparisons:[{label:'Around 100 minutes',reference_hr:148,comparison:{...comparison(),recent:period(3,-1),earlier:period(3,1),delta:-2}}]}};
  const html=durabilityProgress(d,v=>v);
  assert.match(html,/2.0 percentage points less drift/);
  assert.match(html,/-1.0%/);
  assert.match(html,/No fitness-change claim/);
  assert.match(html,/not a recovery score/);
  assert.doesNotMatch(html,/NaN|Infinity/);
});

test('volume chart gracefully handles an empty display window',()=>{
  assert.match(trainingVolumeChart({exposure:{daily:[]}},v=>v),/No running history/);
});

test('comparison history plots only selected observations and never treats one session as a trend',()=>{
  const c={...comparison(),recent_days:42,earlier:{...period(2,7.5),activity_ids:[1,2]},recent:{...period(1,7),activity_ids:[3]}};
  const rows=[{activity_id:1,start_time:'2026-06-01',value:7.4},{activity_id:2,start_time:'2026-07-01',value:7.6},
    {activity_id:3,start_time:'2026-09-20',value:7},{activity_id:4,start_time:'2026-10-01',value:5}];
  const before=JSON.stringify({c,rows});
  const html=comparisonHistory(c,rows,'2026-04-07','2026-10-04',v=>v.slice(0,10));
  assert.equal((html.match(/<circle/g)??[]).length,3);
  assert.match(html,/href="#run\/3"/);
  assert.doesNotMatch(html,/href="#run\/4"|period-median recent|NaN|Infinity/);
  assert.match(html,/period-median earlier/);
  assert.equal(JSON.stringify({c,rows}),before);
  assert.equal(comparisonHistory(c,[],'2026-04-07','2026-10-04',v=>v),'');
});

test('quality history fallback matches the backend time divided by work distance',()=>{
  const c={...comparison(),earlier:{...period(2,8),activity_ids:[1,2]},recent:{...period(1,7),activity_ids:[3]}};
  const d={as_of:'2026-10-04',quality:[{activity_id:3,start_time:'2026-09-20',interval:{repetitions:[
    {kind:'work',duration_seconds:105,distance_miles:.25},{kind:'work',duration_seconds:105,distance_miles:.25},
    {kind:'recovery',duration_seconds:180,distance_miles:.1}]}}],insights:{start:'2026-04-07',quality_comparisons:{repeats:[{label:'About 400 m repeats',total_sessions:3,comparison:c}],sustained:{total_sessions:0,comparison:{status:'insufficient_history'}}}}};
  const html=qualityProgress(d,v=>v.slice(0,10));
  assert.match(html,/aria-label="2026-09-20: 7:00"/);
  assert.doesNotMatch(html,/period-median recent|typical session/);
});

test('daily run strip uses local recorded dates, preserves type splits and shows no-run gaps',()=>{
  const daily=['01','02','03'].map(day=>({as_of:`2026-10-${day}T16:00:00Z`,distance_by_role:{easy:3,long:8,quality:4},distance_7d:15,distance_28d_weekly:14}));
  const d={timezone:'America/New_York',exposure:{daily},activities:[
    {start_time:'2026-10-02T01:00:00Z',role:'easy',distance_miles:3},
    {start_time:'2026-10-01T23:00:00Z',role:'tempo_threshold',distance_miles:4},
    {start_time:'2026-10-03T16:00:00Z',role:'long',distance_miles:8}]};
  const html=trainingVolumeChart(d,v=>v.slice(0,10));
  assert.match(html,/2026-10-01: 7.0 miles · easy 3.0, long 0.0, quality 4.0/);
  assert.match(html,/2026-10-02: no recorded run/);
  assert.match(html,/2026-10-03: 8.0 miles/);
  assert.doesNotMatch(html,/NaN|Infinity/);
});
