import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bindQualityFocus, qualitySessionsCard } from '../../src/run_analysis/web/static/app/quality-session-settings.js';

const choices = { fartlek: true, short_intervals: true, long_intervals: true,
  threshold: true, progression: true, hill_repeats: false };

test('restores all six familiar checkboxes and their saved selections', () => {
  const html = qualitySessionsCard(choices);
  assert.equal((html.match(/type="checkbox"/g) ?? []).length, 6);
  assert.equal((html.match(/ checked/g) ?? []).length, 5);
  for (const name of Object.keys(choices)) assert.ok(html.includes(`name="quality_${name}"`));
  assert.match(html, /Hill Repeats/);
  assert.doesNotMatch(html, /disabled/);
});

test('adaptive settings now expose editable choices and send them rather than the old stored values', () => {
  const html = qualitySessionsCard(choices);
  assert.doesNotMatch(html, / disabled|not yet connected/);
  assert.match(html, /Disabled types will not be prescribed/);
  const view = readFileSync(new URL('../../src/run_analysis/web/static/app/views/settings.js',import.meta.url),'utf8');
  assert.ok(view.includes('qualitySessionsCard(settings.coaching.quality_sessions, settings.coaching.quality_focus)'));
  assert.ok(view.includes('quality_sessions: Object.fromEntries'));
  assert.ok(!view.includes('quality_sessions: adaptive ? settings.coaching.quality_sessions'));
});

test('quality focus is editable, defaults to variety and restores VO2max', () => {
  assert.match(qualitySessionsCard(choices), /value="variety" selected/);
  assert.match(qualitySessionsCard(choices, 'vo2max'), /value="vo2max" selected/);
  assert.match(qualitySessionsCard(choices), /name="quality_focus"/);
  assert.match(qualitySessionsCard(choices, 'vo2max'), /VO₂max/);
});

test('focus help explains enabled-workout limitations without changing choices', () => {
  const focus={value:'vo2max',addEventListener(_,fn){this.change=fn;}};
  const intervals={checked:true,addEventListener(_,fn){this.change=fn;}};
  const help={textContent:''};
  const form={querySelector(s){return s.includes('quality_focus') ? focus : s.includes('quality_long_intervals') ? intervals : help;}};
  bindQualityFocus(form);
  assert.match(help.textContent,/hard aerobic intervals/);
  intervals.checked=false;intervals.change();
  assert.match(help.textContent,/Enable Long Intervals/);
  assert.equal(intervals.checked,false);
  focus.value='variety';focus.change();
  assert.match(help.textContent,/Mixes/);
});
