import {test} from 'node:test';
import assert from 'node:assert/strict';
import {nearestRun,runChartAttributes,wireRunCharts,runTooltipPosition} from '../../src/run_analysis/web/static/app/chart-interaction.js';
import {fitnessChart} from '../../src/run_analysis/web/static/app/charts.js';
import {trainingVolumeChart} from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

test('nearest run follows time and breaks same-date ties by vertical proximity',()=>{
  const points=[{key:1,x:10,y:20},{key:2,x:90,y:30},{key:3,x:90,y:80}];
  assert.equal(nearestRun(points,12,200).key,1);
  assert.equal(nearestRun(points,88,70).key,3);
  assert.equal(nearestRun(points,88,25).key,2);
  assert.equal(nearestRun([],0),null);
  assert.equal(nearestRun([{x:NaN,y:1}],0),null);
});

test('overlays flip left near the right edge and stay vertically beside the run',()=>{
  const sizes={width:720,height:300,tooltipWidth:240,tooltipHeight:100};
  assert.deepEqual(runTooltipPosition({...sizes,x:400,y:160}),{left:412,top:110,side:'right'});
  const nearEdge=runTooltipPosition({...sizes,x:650,y:160});
  assert.deepEqual(nearEdge,{left:398,top:110,side:'left'});
  assert.ok(nearEdge.left+sizes.tooltipWidth<650);
  // Top/bottom boundaries and a wider multi-line tooltip still stay in bounds.
  assert.equal(runTooltipPosition({...sizes,x:650,y:5}).top,6);
  assert.equal(runTooltipPosition({...sizes,x:650,y:295}).top,194);
  assert.equal(runTooltipPosition({...sizes,x:650,y:160,tooltipWidth:400}).left,238);
});

test('narrow charts fall back above only when neither side can fit',()=>{
  const position=runTooltipPosition({width:220,height:240,tooltipWidth:208,tooltipHeight:100,x:110,y:180});
  assert.deepEqual(position,{left:6,top:68,side:'above'});
});

test('run metadata escapes text and only opt-in pace charts expose interaction',()=>{
  const attrs=runChartAttributes({id:1,x:10,y:20,title:'<img onerror="bad">',details:['<script>bad</script>']});
  assert.doesNotMatch(attrs,/<img|<script/);assert.match(attrs,/&lt;img/);
  const points=[{activity_id:1,start_time:'2026-10-01',distance_miles:4,standardized_pace_min_mile:10,raw_pace_min_mile:11,trend_weight:0}];
  assert.doesNotMatch(fitnessChart(points,'both'),/data-run-chart|data-run-key/);
  const html=fitnessChart(points,'both',[],null,null,{interactive:true});
  assert.equal((html.match(/data-run-key="1"/g)??[]).length,2);
  assert.match(html,/Adjusted pace: 10:00\/mi/);assert.match(html,/Unadjusted pace: 11:00\/mi/);
  assert.match(html,/Workout context only/);
});

test('volume overlay preserves separate runs on one date and uses one mileage axis',()=>{
  const d={timezone:'UTC',exposure:{daily:[{as_of:'2026-10-01',distance_7d:15,distance_28d_weekly:14,distance_by_role:{}},{as_of:'2026-10-02',distance_7d:18,distance_28d_weekly:14,distance_by_role:{}}]},activities:[
    {activity_id:1,start_time:'2026-10-01T10:00:00Z',role:'easy',distance_miles:3},
    {activity_id:2,start_time:'2026-10-01T18:00:00Z',role:'long',distance_miles:7}]};
  const before=JSON.stringify(d),html=trainingVolumeChart(d,v=>v.slice(0,10));
  assert.equal((html.match(/<svg/g)??[]).length,1);
  assert.match(html,/data-run-key="1"/);assert.match(html,/data-run-key="2"/);
  assert.match(html,/2026-10-01: 10.0 miles/);
  assert.match(html,/7-day total: 15.0 mi/);assert.match(html,/28-day average: 14.0 mi/);
  assert.match(html,/volume-total-line/);assert.match(html,/long-average/);
  assert.match(html,/class="dotted"><\/i>7-day total/);
  assert.match(html,/class=""><\/i>28-day average/);
  assert.ok(html.indexOf('class="volume-total-line"')<html.indexOf('class="volume-run"'));
  const tall=trainingVolumeChart(d,v=>v.slice(0,10),{height:600});
  assert.match(html,/viewBox="0 0 720 290"/);
  assert.match(tall,/viewBox="0 0 720 600"/);
  assert.deepEqual(tall.match(/data-run-details="[^"]*"/g),html.match(/data-run-details="[^"]*"/g));
  assert.deepEqual(tall.match(/fill="#[^"]*"/g),html.match(/fill="#[^"]*"/g));
  assert.match(tall,/class="dotted"><\/i>7-day total/);
  assert.match(tall,/class=""><\/i>28-day average/);
  assert.equal(JSON.stringify(d),before);
});

class Element {
  constructor(){this.children=[];this.attributes={};this.style={};this.dataset={};this.handlers={};this.classes=new Set();this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),toggle:(c,on)=>on?this.classes.add(c):this.classes.delete(c)};this.clientWidth=320;this.clientHeight=240;this.offsetWidth=170;this.offsetHeight=70;}
  setAttribute(k,v){this.attributes[k]=String(v);}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  getBoundingClientRect(){return {left:10,top:20};}
  addEventListener(k,fn){this.handlers[k]=fn;}
  contains(node){return node===this;}
  closest(){return null;}
}
function fixture(){
  const owner=new Element(),svg=new Element(),doc={activeElement:null,createElement:()=>new Element(),createElementNS:()=>new Element()};
  svg.ownerDocument=doc;svg.parentElement=owner;svg.dataset={runTop:'25',runBottom:'200'};
  const make=(key,x,y)=>{const mark=new Element();mark.dataset={runKey:key,runX:String(x),runY:String(y),runTitle:`Run ${key}`,runDetails:JSON.stringify(['4 miles','<script>not HTML</script>'])};mark.clicked=false;mark.closest=()=>({click:()=>{mark.clicked=true;}});return mark;};
  const marks=[make('1',20,60),make('2',180,100),make('2',180,140)];
  svg.querySelectorAll=()=>marks;
  svg.querySelector=()=>marks[0];
  // Rendered SVG has scaling AND letterboxing offsets, not just width ratio.
  const matrix={a:.5,d:.5,e:10,f:20,inverse:()=>({a:2,d:2,e:-20,f:-40})};
  svg.getScreenCTM=()=>matrix;
  svg.createSVGPoint=()=>({x:0,y:0,matrixTransform(m){return {x:this.x*m.a+m.e,y:this.y*m.d+m.f};}});
  wireRunCharts({querySelectorAll:()=>[svg]});
  return {svg,marks,owner,doc,tip:owner.children[0],guide:svg.children[0]};
}

test('pointer highlighting handles scaled SVGs, highlights both pace marks, and clears on leave',()=>{
  const f=fixture();
  f.svg.handlers.pointermove({clientX:99,clientY:70});
  assert.equal(f.guide.attributes.x1,'180');assert.equal(f.tip.hidden,false);
  assert.equal(f.marks[0].classes.has('is-nearest-run'),false);
  assert.ok(f.marks.slice(1).every(m=>m.classes.has('is-nearest-run')));
  assert.equal(f.tip.children[0].textContent,'Run 2');
  assert.equal(f.tip.children[2].textContent,'<script>not HTML</script>');
  assert.ok(parseFloat(f.tip.style.left)>=6&&parseFloat(f.tip.style.left)<=144);
  f.svg.handlers.pointerleave({pointerType:'mouse'});
  assert.equal(f.tip.hidden,true);assert.equal(f.guide.style.display,'none');
});

test('touch inspection persists; keyboard selects distinct runs, opens details and clears',()=>{
  const f=fixture();let prevented=0;
  f.svg.handlers.pointerdown({clientX:20,clientY:50});
  f.svg.handlers.focusin({target:f.svg});assert.equal(f.tip.children[0].textContent,'Run 1');
  f.svg.handlers.pointerdown({clientX:99,clientY:70});
  f.svg.handlers.focusin({target:f.svg});assert.equal(f.tip.children[0].textContent,'Run 2');
  f.svg.handlers.pointerleave({pointerType:'touch'});assert.equal(f.tip.hidden,false);
  let tapPrevented=false;f.svg.handlers.click({pointerType:'touch',preventDefault(){tapPrevented=true;}});assert.ok(tapPrevented);
  const key=k=>f.svg.handlers.keydown({key:k,target:f.svg,preventDefault(){prevented++;}});
  key('End');assert.equal(f.tip.children[0].textContent,'Run 2');
  key('ArrowLeft');assert.equal(f.tip.children[0].textContent,'Run 1');
  key('Enter');assert.equal(f.marks[0].clicked,true);
  key('Escape');assert.equal(f.tip.hidden,true);assert.equal(prevented,3);
});

test('scaled pointer and keyboard inspection use the same edge-aware placement',()=>{
  const f=fixture();
  // Screen scaling puts the second run near the right edge of its owner.
  f.svg.getScreenCTM=()=>({a:1.5,d:1,e:10,f:20,inverse:()=>({a:1/1.5,d:1,e:-10/1.5,f:-20})});
  f.svg.handlers.pointermove({clientX:280,clientY:120});
  assert.equal(f.tip.dataset.side,'left');
  assert.equal(f.tip.style.left,'88px');assert.equal(f.tip.style.top,'65px');
  f.svg.handlers.keydown({key:'End',target:f.svg,preventDefault(){}});
  assert.equal(f.tip.dataset.side,'left');assert.equal(f.tip.style.left,'88px');
});
