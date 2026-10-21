import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const loader = {};
const page = { innerHTML: '', querySelector: selector => selector==='[data-planner-loader]'?loader:({ addEventListener() {} }) };
globalThis.document = { querySelector: () => page };
const { renderNextRun, weeklyScheduleMarkup } = await import('../../src/run_analysis/web/static/app/views/next-run.js');
const { strengthCompletionMarkup, updateStrengthCompletion } = await import('../../src/run_analysis/web/static/app/views/next-run.js');

const upper = {
  title: 'Upper body', muscle_groups: ['upper pecs', 'lats', 'core/trunk'],
  rationale: 'An isolated no-run day can carry your usual upper-body muscle groups together.',
};

test('lifting checkboxes support today, history, undo state and disabled future days', () => {
  const session = {...upper, session_type:'pull'};
  const day = {date:'2026-10-17', strength_suggestion:session};
  assert.match(strengthCompletionMarkup(day,'2026-10-17'),/data-strength-type="pull"/);
  assert.doesNotMatch(strengthCompletionMarkup(day,'2026-10-17'),/ checked| disabled/);
  assert.match(strengthCompletionMarkup({...day,date:'2026-10-18'},'2026-10-17'),/ disabled/);
  const receipt = {session, recorded_at:'2026-10-17T12:00:00Z'};
  const original = {days:[day], trailing_days:[day]};
  const updated = updateStrengthCompletion(original,day.date,receipt);
  assert.equal(original.days[0].strength_completion,undefined);
  assert.equal(updated.days[0].strength_completion,receipt);
  assert.equal(updated.trailing_days[0].strength_completion,receipt);
  assert.match(strengthCompletionMarkup(updated.days[0],'2026-10-17'),/ checked[^]*>Completed/);
  assert.equal(updateStrengthCompletion(updated,day.date,null).days[0].strength_completion,null);
});

test('past lifting appears beside run links without nesting a checkbox inside a link', () => {
  const session = {...upper,session_type:'pull'};
  const html = weeklyScheduleMarkup({...schedule,trailing_days:[{
    date:'2026-10-07',day_role:'run_day',strength_suggestion:session,
    activities:[{activity_id:42,distance_miles:3,workout_type:'easy',health_tag:'normal'}],
  }]},null,{},null,new Date('2026-10-17T12:00:00Z'));
  assert.match(html,/href="#run\/42"/);
  assert.match(html,/data-strength-date="2026-10-07"/);
  for (const anchor of html.matchAll(/<a\b[^>]*>[^]*?<\/a>/g)) {
    assert.doesNotMatch(anchor[0], /<input/);
  }
});
const schedule = {
  start_date: '2026-10-08', end_date: '2026-10-14', planner_status: 'ready',
  run_count: 1, target_run_count: 1, projected_distance_range_miles: [3.5, 4],
  target_distance_range_miles: [18, 19], target_evidence: { planning_mode: 'established', capacity_reference_miles: 18, rationale: 'Measured history.' },
  summary: 'Your plan.', trailing_days: [], days: [
    { date: '2026-10-08', day_role: 'rest_day', rationale: 'No run selected by the full projected training program.', strength_suggestion: upper },
    { date: '2026-10-09', day_role: 'run_day', planned_at: '2026-10-09T07:00:00-04:00', recommendation: {
      title: 'Easy aerobic run', workout_type: 'easy', readiness: 'caution', readiness_label: 'Weather caution',
      readiness_reason: 'Hot conditions.', planned_for: '2026-10-09T07:00:00-04:00', time_recommendation_basis: 'weather',
      distance_target_miles: 3.8, distance_range_miles: [3.5, 4], target_zones: ['Z1', 'Z2'],
      rule_trace: [], structure: [], reasons: ['Supporting aerobic volume.'],
    } },
    { date: '2026-10-10', day_role: 'rest_day', forced_rest: true, rationale: 'Your protected day.' },
    { date: '2026-10-11', day_role: 'race_event_day', rationale: 'Your scheduled race.' },
  ],
};
globalThis.fetch = async (path) => ({ ok: true, json: async () => ({
  '/api/current-status': { health_status: 'normal', health_report_current: true },
  '/api/weekly-schedule/latest': schedule,
  '/api/weekly-schedule/jobs': {job_id:'view-test',state:'complete',result:schedule},
  '/api/fitness-state': { recent_load: { trailing_7d: { distance_miles: 18 } }, running_days_28d: 16 },
  '/api/goal-progress': null,
})[path] });

test('previous-seven-day runs are full-card analysis links; rest days are not', () => {
  const html = weeklyScheduleMarkup({ ...schedule, trailing_days: [
    { date: '2026-10-07', day_role: 'run_day', activities: [{ activity_id: 42, distance_miles: 4.4, workout_type: 'tempo_threshold', health_tag: 'normal' }] },
    { date: '2026-10-06', day_role: 'rest_day', activities: [] },
    { date: '2026-10-05', day_role: 'run_day', activities: [
      { activity_id: 43, distance_miles: 3, workout_type: 'easy', health_tag: 'normal' },
      { activity_id: 44, distance_miles: 2, workout_type: 'easy', health_tag: 'normal' },
    ] },
  ] }, null);
  const trailing = html.split('class="week-grid trailing-week"')[1];
  assert.equal((trailing.match(/class="trailing-day logged-run-card"/g) ?? []).length, 3);
  for (const id of [42, 43, 44]) assert.match(trailing, new RegExp(`href="#run/${id}" aria-label="Run analysis:`));
  assert.match(trailing, /<a class="trailing-day logged-run-card"[^>]+><p class="eyebrow">[^<]+<\/p><h2>Run Day<\/h2>/);
  assert.match(trailing, /<article class="trailing-day rest-day">[^]*?No recorded activity<\/small><\/article>/);
  assert.match(trailing, /<div class="trailing-day-group"><a/);
});

test('weekly cards omit redundant labels and rest/upper-body boilerplate', async () => {
  await renderNextRun();
  assert.equal((page.innerHTML.match(/<h2>Rest Day<\/h2>/g) ?? []).length, 1);
  assert.equal((page.innerHTML.match(/<h2>Strength<\/h2>/g) ?? []).length, 1);
  assert.doesNotMatch(page.innerHTML, /suggested|· Run Day|No run planned|<h2>Run rest day|No run selected|An isolated no-run day|No weight, set, or rep target/i);
  assert.match(page.innerHTML, /Easy aerobic run/);
  assert.match(page.innerHTML, /How do you feel now\?/);
  assert.match(page.innerHTML, />Update plan<\/button>/);
  assert.doesNotMatch(page.innerHTML,/Save check-in|No check-in recorded|Last check-in/);
  assert.match(page.innerHTML, /3\.8 miles/);
  assert.match(page.innerHTML, /upper pecs · lats · core\/trunk/);
  assert.match(page.innerHTML, /standalone-strength/);
  assert.match(page.innerHTML, /<strong class="strength-label prescription-extent">Upper Body Day<\/strong>/);
  assert.match(page.innerHTML, /<p class="strength-groups">upper pecs · lats · core\/trunk<\/p>/);
  assert.doesNotMatch(page.innerHTML, /<p class="eyebrow">Strength<\/p>/);
  assert.doesNotMatch(page.innerHTML, /Optional strength/);
  assert.match(page.innerHTML, /Weather caution/);
  assert.match(page.innerHTML, /Why this warning/);
  assert.match(page.innerHTML, /Your protected day/);
  assert.match(page.innerHTML, /<h2>Race day<\/h2>/);
});

test('leg-day cards omit cadence boilerplate from cached plans while retaining the routine', () => {
  const saved = schedule.days[0].strength_suggestion;
  try {
    schedule.days[0].strength_suggestion = {
      session_type: 'legs', title: 'Leg strength', muscle_groups: ['quads', 'hamstrings', 'glutes', 'calves'],
      rationale: 'This is the best available no-run day for the 8-day leg cadence while keeping lower-body work away from long and quality runs when the calendar allows.',
    };
    const html = weeklyScheduleMarkup(schedule, null);
    assert.match(html, /Leg Day/);
    assert.match(html, /quads · hamstrings · glutes · calves/);
    assert.doesNotMatch(html, /best available no-run day|leg cadence|calendar allows/);
  } finally { schedule.days[0].strength_suggestion = saved; }
});

test('strength-disabled off days remain Rest Day without routine markup', async () => {
  const saved = schedule.days[0].strength_suggestion;
  schedule.days[0].strength_suggestion = null;
  try {
    await renderNextRun();
    assert.equal((page.innerHTML.match(/<h2>Rest Day<\/h2>/g) ?? []).length, 2);
    assert.doesNotMatch(page.innerHTML, /<h2>Strength<\/h2>|strength-label|strength-groups/);
  } finally {
    schedule.days[0].strength_suggestion = saved;
  }
});

test('weekly headers reserve a separate time line on strength and rest days', async () => {
  await renderNextRun();
  const headers = [...page.innerHTML.matchAll(/<p class="eyebrow schedule-date">(.*?)<\/p>/g)].map(match => match[1]);
  assert.equal(headers.length, schedule.days.length);
  assert.match(headers[0], /<\/span><span class="schedule-daypart" aria-hidden="true">&#160;<\/span>$/);
  assert.match(headers[1], /<\/span><span class="schedule-daypart">[^<]+<\/span>$/);
  assert.match(headers[2], /<span class="schedule-daypart">Your constraint<\/span>/);
  assert.match(headers[3], /<span class="schedule-daypart">Race<\/span>/);
});

test('routine styling explicitly uses regular-weight white text', () => {
  const css = readFileSync(new URL('../../src/run_analysis/web/static/styles.css', import.meta.url), 'utf8');
  const rule = css.match(/\.strength-suggestion \.strength-groups\s*\{([^}]+)\}/)?.[1];
  assert.ok(rule);
  assert.match(rule, /color:\s*var\(--flat\)/);
  assert.match(rule, /font-weight:\s*400/);
});

test('quality headings are concise without changing the prescription or effort guidance', async () => {
  const result = schedule.days[1].recommendation;
  const savedTitle = result.title;
  const savedStructure = result.structure;
  const labels = [
    ['Controlled intervals', 'Intervals'], ['Fartlek by feel', 'Fartlek'],
    ['Short controlled pickups', 'Pickups'], ['Long controlled efforts', 'Intervals'],
    ['Continuous threshold run', 'Threshold run'], ['Controlled progression run', 'Progression run'],
    ['Hills by terrain', 'Hill repeats'], ['Easy run with controlled pickups', 'Easy run with pickups'],
    ['Extended fartlek aerobic session', 'Fartlek'], ['Extended aerobic session with short pickups', 'Pickups'],
    ['Extended aerobic session with long efforts', 'Intervals'],
    ['Extended aerobic session with threshold blocks', 'Threshold intervals'],
    ['Extended aerobic progression session', 'Progression run'], ['Extended aerobic session with hills', 'Hill repeats'],
  ];
  try {
    result.structure = [{ instruction: 'Run at controlled effort; do not sprint.', target_zones: ['Z4 effort'] }];
    for (const [original, display] of labels) {
      result.title = original;
      await renderNextRun();
      assert.ok(page.innerHTML.includes(`<h2>${display}</h2>`));
      assert.match(page.innerHTML, /3\.8 miles/);
      assert.match(page.innerHTML, /Run at controlled effort; do not sprint\./);
      assert.equal(result.title, original);
      assert.equal(result.distance_target_miles, 3.8);
    }
  } finally {
    result.title = savedTitle;
    result.structure = savedStructure;
  }
});

test('strength paired with easy running retains the run heading and a separate section', async () => {
  schedule.days[1].strength_suggestion = { title: 'Upper-body push', muscle_groups: ['upper pecs', 'triceps'], rationale: '' };
  try {
    await renderNextRun();
    assert.match(page.innerHTML, /<h2>Easy aerobic run<\/h2>/);
    assert.match(page.innerHTML, /<h3>Strength<\/h3><strong class="strength-label prescription-extent">Push Day<\/strong>/);
    assert.match(page.innerHTML, /<p class="strength-groups">upper pecs · triceps<\/p>/);
  } finally {
    delete schedule.days[1].strength_suggestion;
  }
});

test('standalone strength uses concise day labels in the shared prescription row', async () => {
  const saved = schedule.days[0].strength_suggestion;
  try {
    for (const [session_type, label] of [['push', 'Push Day'], ['pull', 'Pull Day'], ['legs', 'Leg Day']]) {
      schedule.days[0].strength_suggestion = { ...saved, session_type };
      await renderNextRun();
      assert.ok(page.innerHTML.includes(`<h2>Strength</h2><strong class="strength-label prescription-extent">${label}</strong><div class="schedule-body">`));
    }
    assert.match(page.innerHTML, /class="week-grid planned-week"/);
    assert.match(page.innerHTML, /<h2>Easy aerobic run<\/h2><strong class="prescription-extent">3\.8 miles<\/strong><div class="schedule-body">/);
  } finally {
    schedule.days[0].strength_suggestion = saved;
  }
});

test('paused planning remains explicit rather than masquerading as a rest day', async () => {
  const savedDays = schedule.days;
  schedule.planner_status = 'review_required';
  schedule.days = [{ date: '2026-10-08', day_role: 'rest_day', rationale: 'Resolve the activity evidence before planning.' }];
  try {
    await renderNextRun();
    assert.match(page.innerHTML, /<h2>Planner paused<\/h2>/);
    assert.match(page.innerHTML, /Resolve the activity evidence before planning/);
    assert.doesNotMatch(page.innerHTML, /<h2>Rest Day<\/h2>/);
  } finally {
    schedule.planner_status = 'ready';
    schedule.days = savedDays;
  }
});

test('summary puts today beneath the count without range or scoring boilerplate',()=>{
  const before=JSON.stringify(schedule);
  for(const [day,title] of [[8,'Upper Body Day'],[9,'Easy aerobic run'],[10,'Rest Day'],[11,'Race day']]) {
    const html=weeklyScheduleMarkup(schedule,null,{longest_completed_run_30d_miles:9.1},
      {long_goal_miles:17},new Date(2026,9,day,12));
    const summary=html.split('schedule-summary')[1].split('</article>')[0];
    assert.match(summary,/<h2>1 planned run · 3\.8 miles this week<\/h2>/);
    assert.ok(summary.includes(`<h3>${title}</h3>`));
    assert.doesNotMatch(summary,/Your plan\.|How was this range|Measured history|3\.5–4\.0|Projected rolling load/);
    assert.match(html,/9\.1 <span>\/ 17\.0 miles/);
    assert.match(html,/54% of goal/);
    assert.match(html,/<p class="eyebrow">Your goal<\/p>/);
    assert.match(html,/<h2>Training for general fitness<\/h2>/);
    assert.match(html,/<p class="eyebrow">Weekly Long-Run Target<\/p>/);
  }
  assert.equal(JSON.stringify(schedule),before);
});

test('summary uses the committed visible mileage, not the old range midpoint',()=>{
  const html=weeklyScheduleMarkup({...schedule,run_count:4,visible_7d_scheduled_miles:20.2},
    {status:'no_goal',headline:'Training for general fitness',detail:'Old explanatory copy.'},
    {longest_completed_run_30d_miles:9.1},{long_goal_miles:17},new Date(2026,9,8,12));
  assert.match(html,/<h2>4 planned runs · 20\.2 miles this week<\/h2>/);
  assert.doesNotMatch(html,/20\.2–20\.2|Old explanatory copy/);
  assert.match(html,/<h2>Training for general fitness<\/h2>/);
});

test('goal progress stays bounded and missing evidence is not invented',()=>{
  const at=new Date(2026,9,8,12);
  const complete=weeklyScheduleMarkup(schedule,null,{longest_completed_run_30d_miles:20},{long_goal_miles:17},at);
  assert.match(complete,/aria-valuenow="100"/);
  const unknown=weeklyScheduleMarkup(schedule,null,{},null,at);
  assert.match(unknown,/Set a long-run goal/);
  assert.doesNotMatch(unknown,/aria-valuenow|NaN|undefined/);
  const saved=weeklyScheduleMarkup(schedule,null,{longest_completed_run_30d_miles:9.1},
    {long_goal_miles:17,requested_long_goal_miles:18},at);
  assert.match(saved,/\/ 18\.0 miles/);
});

test('long-run progress uses recorded accomplishment, not capped planner capacity',()=>{
  const html=weeklyScheduleMarkup(schedule,null,
    {longest_run_30d_miles:8.9,longest_completed_run_30d_miles:9.536},
    {long_goal_miles:17,requested_long_goal_miles:18},new Date(2026,9,20,12));
  assert.match(html,/9\.5 <span>\/ 18\.0 miles/);
  assert.doesNotMatch(html,/8\.9 <span>|Why this workout\?/);
  const old=weeklyScheduleMarkup(schedule,null,{longest_run_30d_miles:8.9},{long_goal_miles:17});
  assert.match(old,/Recorded progress not available yet/);
});
