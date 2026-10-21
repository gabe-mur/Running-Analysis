import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadGuidanceDisplay} from '../../src/run_analysis/web/static/app/load-guidance-display.js';
import {fatiguePanel} from '../../src/run_analysis/web/static/app/progress-wip-charts.js';

function data(status='within') {
  const at='2026-10-10T16:00:00Z';
  const labels={within:'Within recommended load',below:'Below recommended load',above:'Above recommended load'};
  const row={as_of:at,status,label:labels[status],action:'Continue your planned training.',detail:'Recorded recommendation.',recommended_range:{low:24,high:39}};
  const sample=value=>({value,low:value,high:value,complete:true});
  return {as_of:at,timezone:'UTC',load_display:{current:{recent:sample(35)},daily:[{as_of:at,recent:sample(35),baseline:sample(30)}]},
    load_guidance:{current:row,daily:[row],ratio_reference:{low:.8,high:1.3}}};
}

test('range and actionable advice replace the unrelated legacy strain estimate',()=>{
  const html=loadGuidanceDisplay(data());
  assert.match(html,/Within recommended load/);
  assert.match(html,/Continue your planned training/);
  assert.match(html,/24–39/);
  assert.match(html,/adjusts gradually as your training changes/);
  assert.doesNotMatch(html,/https?:|Runalyze|Garmin|TRIMP|EPOC|80–130%|coaching boundaries/);
  assert.doesNotMatch(html,/No elevated strain flag|Building|fully recovered|NaN|undefined/);
  assert.match(loadGuidanceDisplay(null),/Load guidance unavailable/);
});

test('health and tired reports affect current advice, never historical load classification',()=>{
  const d=data();
  for(const status of ['pain_or_injury_concern','sick_or_recovering']) {
    d.current_check_in={status};
    assert.match(loadGuidanceDisplay(d),/health report takes priority/);
    assert.doesNotMatch(loadGuidanceDisplay(d),/Continue your planned training/);
    assert.equal(d.load_guidance.current.status,'within');
  }
  d.current_check_in={status:'little_tired'};
  assert.match(loadGuidanceDisplay(d),/Keep easy work familiar/);
  d.load_guidance.current.boundary_overlap=true;
  assert.match(loadGuidanceDisplay(d),/overlaps a range boundary/);
});

test('graph paints the recommended band and uses guidance colors on the line, not a strip',()=>{
  const d=data('above');
  d.status_history=[{as_of:d.as_of,status:'unknown',label:'Unclear comparison'}];
  const html=fatiguePanel(d,v=>new Date(v).toISOString().slice(0,10),{embedded:true});
  assert.match(html,/load-recommended-zone/);
  assert.match(html,/Recommended: 24.0–39.0 HR load\/day/);
  assert.match(html,/Above recommended load/);
  assert.match(html,/data-load-status="above" style="stroke:#e7bc68"/);
  assert.doesNotMatch(html,/training-status-cell|load-status-label/);
  assert.match(html,/moving recommended range/);
  assert.doesNotMatch(html,/Unclear comparison|load-comparison-zone|fatigue-level|not a target|Similar load/);
  assert.doesNotMatch(html,/NaN|undefined|fully recovered/);
  assert.doesNotMatch(html,/About training load|exponential|time constants|Runalyze|Garmin|EPOC/);
});
