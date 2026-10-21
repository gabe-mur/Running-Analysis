import {test} from 'node:test';
import assert from 'node:assert/strict';
import {windowedChartSeries} from '../../src/run_analysis/web/static/app/chart-window.js';
import {fitnessChart} from '../../src/run_analysis/web/static/app/charts.js';

const day=86400000,origin=Date.parse('2026-10-01T00:00:00Z');
const row=(d,value)=>({as_of:new Date(origin+d*day).toISOString(),value});
const clip=(rows,start,end)=>windowedChartSeries(rows,origin+start*day,origin+end*day,r=>[r.value],14);

test('clips the existing slope using preceding history, without fabricating run marks',()=>{
  const history=[row(0,10),row(4,8),row(8,12)];
  const short=clip(history,2,7),long=clip(history,-2,8);
  assert.deepEqual(short.groups[0].map(p=>[(p.at-origin)/day,p.values[0]]),[[2,9],[4,8],[7,11]]);
  assert.deepEqual(short.points.map(p=>p.row),[history[1]]);
  assert.equal(short.groups[0][0].row,undefined);
  assert.equal((short.groups[0][1].values[0]-short.groups[0][0].values[0])/(2*day),
    (long.groups[0][1].values[0]-long.groups[0][0].values[0])/(4*day));
});

test('does not extend a curve beyond its evidence or bridge unsupported periods',()=>{
  assert.deepEqual(clip([row(0,10)],1,4),{groups:[],points:[]});
  assert.deepEqual(clip([row(0,10),row(20,8)],18,24).groups[0].map(p=>p.row),[row(20,8)]);
  assert.deepEqual(clip([row(0,10),row(2,null),row(4,8)],3,6).groups[0].map(p=>p.row),[row(4,8)]);
});

test('a segment spanning the window is visible but never becomes an observation',()=>{
  const result=clip([row(0,10),row(10,20)],3,7);
  assert.deepEqual(result.groups[0].map(p=>p.values[0]),[13,17]);
  assert.deepEqual(result.points,[]);
  assert.deepEqual(clip([row(0,10),row(4,8)],0,4).groups[0].length,2);
});

test('pace curves enter at the boundary; hidden history cannot stretch axes or leak into the plot',()=>{
  const series=[{activity_id:1,start_time:row(4,0).as_of,standardized_pace_min_mile:9,raw_pace_min_mile:10,distance_miles:3}];
  const trend=[row(-100,500),row(0,10),row(4,8)].map(r=>({...r,pace_min_mile:r.value}));
  const end=row(8,0).as_of;
  const render=history=>fitnessChart(series,'both',history,end,6,{rawTrend:history,interactive:true});
  const html=render(trend);
  assert.match(html,/<path d="M55\.0,[\d.]+ L325\.0,[\d.]+" class="chart-line trend-28"/);
  assert.match(html,/<path d="M55\.0,[\d.]+ L325\.0,[\d.]+" class="chart-line raw-trend"/);
  assert.equal(html,render(trend.slice(1)));
  assert.equal((html.match(/data-run-key=/g)??[]).length,2); // One real run, two pace marks.
  assert.doesNotMatch(html,/NaN|Infinity/);
});
