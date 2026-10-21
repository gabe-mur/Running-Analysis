import { escapeHtml } from './format.js';

export function plannerLongGoalControl(planner,{setup=false,general=true}={}) {
  if (!planner || planner.status !== 'adopted') return '';
  return `<div data-general-long-goal ${general?'':'hidden'}>
    <label>Weekly Long-Run Target (miles)<input type="number" name="planner_long_goal" step="0.1" min="${planner.long_target_miles}" max="100" required ${general?'':'disabled'} value="${planner.requested_long_goal_miles ?? planner.long_goal_miles}">
      <small class="field-help">The weekly long-run distance you want to build toward. Not a race goal.</small>
    </label></div>`;
}

export function plannerPreferenceFields(planner) {
  if (!planner || planner.status !== 'adopted') return '';
  const frequency = planner.requested_run_frequency ?? planner.run_frequency;
  const weekday = Object.hasOwn(planner,'requested_long_run_weekday') ? planner.requested_long_run_weekday : planner.long_run_weekday;
  const low = planner.frequency_lower_per_7, high = planner.frequency_upper_per_7;
  const frequencyLabel = (label, position) => Number.isFinite(low) && Number.isFinite(high)
    ? `${label} · around ${Math.round(low + position * (high - low))} runs per 7 days` : label;
  return `<label>Run frequency<select name="planner_frequency">${[
      ['fewer', frequencyLabel('Fewer days', 0)], ['balanced', frequencyLabel('Balanced', .5)], ['more', frequencyLabel('More days', 1)],
    ].map(([value, label]) => `<option value="${value}" ${value === frequency ? 'selected' : ''}>${label}</option>`).join('')}</select>
      <small class="field-help">Your training volume and recovery still guide the schedule.</small>
    </label>
    <label>Long-run day<select name="planner_long_day"><option value="" ${weekday==null?'selected':''}>Automatic</option>${[
      'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday',
    ].map((label,value)=>`<option value="${value}" ${value===weekday?'selected':''}>${label}</option>`).join('')}</select>
      <small class="field-help">Long runs stay on the selected day. Automatic lets the planner choose.</small>
    </label>`;
}

export function readPlannerPreferences(form,planner,{goalOnly=false,setup=false}={}) {
  const goal=form.querySelector('[name="planner_long_goal"]');
  const day=form.querySelector('[name="planner_long_day"]');
  const savedGoal=planner.requested_long_goal_miles??planner.long_goal_miles;
  const savedDay=Object.hasOwn(planner,'requested_long_run_weekday')?planner.requested_long_run_weekday:planner.long_run_weekday;
  return {long_run_goal_miles:goalOnly||setup&&goal&&!goal.disabled?Number(goal.value):savedGoal,
    run_frequency:goalOnly?(planner.requested_run_frequency??planner.run_frequency):form.querySelector('[name="planner_frequency"]').value,
    long_run_weekday:goalOnly?(savedDay??null):day.value===''?null:Number(day.value)};
}

export function plannerSetupSection(planner, {step = 5} = {}) {
  if (!planner) return '';
  return `<section class="wide-card setup-block">
    <p class="eyebrow">Step ${step}</p><h2>Running progression</h2>
    <p class="chart-note">Your long-run goal guides progression. Run frequency guides how that training is spread across days. Volume and recovery determine the pace of progress.</p>
    ${planner.status === 'adopted'
      ? `<div class="setup-fields">${plannerPreferenceFields(planner)}</div>`
      : '<p class="chart-note">Add enough run history to establish your starting plan, then choose these preferences here.</p>'}
  </section>`;
}

export function plannerSettingsCard(planner) {
  if (!planner) return '';
  if (planner.status !== 'adopted') {
    return `<fieldset><legend>Running preferences</legend><p>${escapeHtml(planner.detail)}</p></fieldset>`;
  }
  return `<fieldset class="planner-preferences"><legend>Running preferences</legend>
    ${plannerPreferenceFields(planner)}
    <button type="button" id="save-planner-preferences">Save preferences</button>
    <span id="planner-preferences-status" class="form-status" role="status">${planner.preferences_pending ? 'Saved for the next plan update.' : ''}</span>
  </fieldset>`;
}
