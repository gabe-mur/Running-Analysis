import { test } from 'node:test';
import assert from 'node:assert/strict';
import { plannerSettingsCard, plannerSetupSection, plannerPreferenceFields,plannerLongGoalControl,readPlannerPreferences } from '../../src/run_analysis/web/static/app/planner-settings.js';
import {racePreparationCard} from '../../src/run_analysis/web/static/app/race-settings.js';

const planner = { status: 'adopted', long_target_miles: 8.9, long_goal_miles: 17,
  run_frequency: 'balanced', preferences_pending: false,
  detail: 'Technical diagnostic', issued_at: '2026-10-06T11:28:19Z' };

test('running preferences keep frequency and optional weekday; target moves to the training goal', () => {
  const card = plannerSettingsCard(planner);
  assert.match(card, /Running preferences/);
  assert.doesNotMatch(card,/planner_long_goal/);
  assert.match(card,/Long-run day/);
  assert.match(card,/<option value="" selected>Automatic/);
  assert.match(card,/Saturday/);
  const goal=racePreparationCard({distance:'general_fitness'},planner);
  assert.match(goal,/Training goal/);
  assert.doesNotMatch(goal,/Save target|save-planner-long-goal/);
  assert.match(goal,/Weekly Long-Run Target \(miles\)/);
  assert.ok(goal.includes('The weekly long-run distance you want to build toward. Not a race goal.'));
  assert.match(goal,/step="0.1" min="8.9" max="100" required  value="17"/);
  assert.match(card, /value="balanced" selected/);
  assert.match(card, /Fewer days/);
  assert.match(card, /More days/);
  assert.match(card, /id="save-planner-preferences">Save preferences/);
  assert.match(card, /type="button"/);
  assert.doesNotMatch(card, /<dl>|Technical diagnostic|2026-10-06|e-folding|quarter|cadence|volume-linked|No unapplied/);
});

test('setup shares both settings controls and explains progression without another save button', () => {
  const section = plannerSetupSection(planner);
  assert.ok(section.includes(plannerPreferenceFields(planner)));
  assert.equal((section.match(/name="planner_long_day"/g)??[]).length,1);
  assert.match(section,/Long-run day/);
  assert.match(section,/<option value="" selected>Automatic/);
  assert.match(section,/value="5" >Saturday/);
  assert.match(section, /Running progression/);
  assert.match(section, /Your long-run goal guides progression/);
  assert.match(section, /Run frequency guides how that training is spread across days/);
  assert.doesNotMatch(section, /<button/);
  assert.equal(plannerSetupSection(null), '');
  const unavailable = plannerSetupSection({ status: 'review_required' });
  assert.match(unavailable, /Add enough run history/);
  assert.doesNotMatch(unavailable, /<input|<select/);
});

test('pending saved values are displayed instead of stale issued values', () => {
  const pending={ ...planner, requested_long_goal_miles: 20,
    requested_run_frequency: 'fewer', requested_long_run_weekday:5,preferences_pending: true };
  const card = plannerSettingsCard(pending);
  assert.match(plannerLongGoalControl(pending),/value="20"/);
  assert.match(card, /value="fewer" selected/);
  assert.match(card,/value="5" selected>Saturday/);
  assert.match(card, /Saved for the next plan update/);
});

test('Automatic can explicitly clear a saved weekday and goal visibility follows general fitness',()=>{
  const card=plannerSettingsCard({...planner,long_run_weekday:5,requested_long_run_weekday:null});
  assert.match(card,/<option value="" selected>Automatic/);
  const race=racePreparationCard({distance:'marathon'},planner);
  assert.match(race,/data-general-long-goal hidden/);
  assert.match(race,/name="planner_long_goal"[^>]*disabled/);
});

test('each save changes only its own section; hidden race target preserves general destination',()=>{
  const fields={planner_long_goal:{value:'20',disabled:false},planner_frequency:{value:'more'},planner_long_day:{value:'5'}};
  const form={querySelector:s=>fields[s.match(/name="([^"]+)/)[1]]};
  assert.deepEqual(readPlannerPreferences(form,planner),{long_run_goal_miles:17,run_frequency:'more',long_run_weekday:5});
  assert.deepEqual(readPlannerPreferences(form,planner,{goalOnly:true}),{long_run_goal_miles:20,run_frequency:'balanced',long_run_weekday:null});
  assert.equal(readPlannerPreferences(form,planner,{setup:true}).long_run_goal_miles,20);
  assert.equal(readPlannerPreferences(form,planner,{setup:true}).long_run_weekday,5);
  fields.planner_long_goal.disabled=true;
  fields.planner_long_day.value='';
  assert.deepEqual(readPlannerPreferences(form,planner,{setup:true}),{long_run_goal_miles:17,run_frequency:'more',long_run_weekday:null});
});

test('missing or unverified planner does not expose editable invented values', () => {
  assert.equal(plannerSettingsCard(null), '');
  const card = plannerSettingsCard({ status: 'review_required', detail: 'Needs <review>' });
  assert.doesNotMatch(card, /<input|<select|<button/);
  assert.match(card, /Needs &lt;review&gt;/);
});
