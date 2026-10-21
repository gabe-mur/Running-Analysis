// Preview is read-only. Applying is a separate dated request, not race clearance.
import { api, invalidateSettings } from './api.js';
import { escapeHtml } from './format.js';
import { plannerLongGoalControl,readPlannerPreferences } from './planner-settings.js';

const DISTANCES = [['general_fitness', 'General fitness'], ['5k', '5K'], ['10k', '10K'],
  ['half_marathon', 'Half marathon'], ['marathon', 'Marathon']];

export function racePreparationFields(draft = {},planner=null,{setup=false}={}) {
  return `<label>Goal<select name="race_distance">${DISTANCES.map(([key, label]) =>
    `<option value="${key}" ${key === (draft.distance ?? 'general_fitness') ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    ${plannerLongGoalControl(planner,{setup,general:!draft.distance||draft.distance==='general_fitness'})}
    <label data-race-date>Race date<input type="date" name="race_date" value="${escapeHtml(draft.race_date ?? '')}"></label>
    <label data-race-mode>Preparation<select name="race_mode">
      <option value="shorter" ${draft.preparation_mode === 'shorter' ? 'selected' : ''}>Shorter preparation · less buffer</option>
      <option value="balanced" ${!draft.preparation_mode || draft.preparation_mode === 'balanced' ? 'selected' : ''}>Balanced preparation</option>
      <option value="more" ${draft.preparation_mode === 'more' ? 'selected' : ''}>More preparation · gentler build</option>
    </select></label>
    <div class="settings-card-actions" ${setup ? 'hidden' : ''}>
      <button type="button" data-race-preview>Check preparation</button>
      <button type="button" data-race-apply disabled>Apply training goal</button>
    </div>
    <p data-race-guidance role="status" aria-live="polite"></p>
    <div data-race-outcome hidden>
      <label>Race outcome<select data-race-outcome-choice><option value="">Select what happened</option></select></label>
      <button type="button" data-race-outcome-save>Save race outcome</button>
      <p data-race-outcome-message role="status"></p>
    </div>`;
}

export function racePreparationCard(draft,planner=null) {
  return `<fieldset><legend>Training goal</legend>${racePreparationFields(draft,planner)}</fieldset>`;
}

export function readRacePreparation(form) {
  const distance = form.querySelector('[name="race_distance"]').value;
  return { distance, race_date: distance === 'general_fitness' ? null : form.querySelector('[name="race_date"]').value || null,
    preparation_mode: distance === 'marathon' ? form.querySelector('[name="race_mode"]').value : 'balanced' };
}

export function preparationSummary(result, mode = 'balanced') {
  const option = result.options?.find((o) => o.mode === mode);
  if (!option) return result.detail || 'Preparation guidance needs review. No plan changed.';
  if (!option.earliest_conditional_date) return 'No preparation window found in this scenario. No plan changed.';
  const date = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' })
    .format(new Date(`${option.earliest_conditional_date}T12:00:00`));
  const dose = `around ${option.target.volume_per_7.toFixed(1)} miles per 7 days and a ${option.target.long_miles.toFixed(1)}-mile long run`;
  return `${option.requested_date_fits ? 'This date fits the conditional build estimate' : 'This date is earlier than the conditional build estimate'}: ${date}, ${dose}. Actual delivery, recovery and the full calendar still need checking.${option.interruption_allowance_fits === false ? ' There is little room for interruptions.' : ''}`;
}

export function bindRacePreparation(form, request = api,{planner=null}={}) {
  const goal = form.querySelector('[name="race_distance"]');
  const date = form.querySelector('[name="race_date"]');
  const mode = form.querySelector('[name="race_mode"]');
  const button = form.querySelector('[data-race-preview]');
  const output = form.querySelector('[data-race-guidance]');
  const apply = form.querySelector('[data-race-apply]');
  let goalStatus = null;
  function showOutcome(value) {
    const panel = form.querySelector('[data-race-outcome]');
    if (!panel) return;
    const outcome = value.outcome;
    panel.hidden = !outcome?.report_available;
    if (panel.hidden) return;
    const choice = panel.querySelector('[data-race-outcome-choice]');
    choice.innerHTML = `<option value="">Select what happened</option><option value="did_not_start">Did not start</option>${outcome.activities.map((r) =>
      `<option value="${escapeHtml(r.key)}">Uploaded run · ${escapeHtml(r.date)} · ${r.miles.toFixed(1)} miles</option>`).join('')}`;
    choice.value = outcome.status === 'did_not_start' ? 'did_not_start' : outcome.physical_key || '';
    panel.querySelector('[data-race-outcome-message]').textContent = outcome.status === 'unknown'
      ? 'Confirm the race before the plan moves on. If you ran it, upload or sync it first.' : 'Race outcome recorded.';
  }
  let revision = 0;
  function sync() {
    revision += 1;
    const general = goal.value === 'general_fitness';
    const longGoal=form.querySelector('[data-general-long-goal]');
    if(longGoal) {
      longGoal.hidden=!general;
      const input=form.querySelector('[name="planner_long_goal"]');
      if(input)input.disabled=!general;
    }
    date.disabled = general;
    date.required = !general;
    date.closest('label').hidden = general;
    mode.disabled = goal.value !== 'marathon';
    mode.closest('label').hidden = mode.disabled;
    button.hidden = general;
    apply.disabled = goalStatus?.status !== 'available' || (!general && !goalStatus.race_activation_available);
    if (mode.disabled) mode.value = 'balanced';
    output.textContent = general ? '' : goalStatus?.activation_note || '';
  }
  [goal, date, mode].forEach((field) => field.addEventListener('change', sync));
  date.addEventListener('input', sync);
  sync();
  const initialRevision = revision;
  request('/api/settings/race-preparation/status').then((value) => {
    goalStatus = value;
    showOutcome(value);
    apply.disabled = value.status !== 'available' || (goal.value !== 'general_fitness' && !value.race_activation_available);
    if (initialRevision !== revision) return;
    const saved = value.requested || value.applied;
    if (saved) {
      goal.value = saved.distance;
      date.value = saved.race_date || '';
      mode.value = saved.preparation_mode;
      sync();
    }
    if (value.pending) output.textContent = 'Goal saved for the next plan update.';
    else if (value.applied?.distance && value.applied.distance !== 'general_fitness')
      output.textContent = `Active goal: ${DISTANCES.find(([key]) => key === value.applied.distance)?.[1]} · ${value.applied.race_date}.`;
    else if (value.status !== 'available') output.textContent = value.detail || 'Finish training setup before applying a goal.';
  }).catch((error) => { output.textContent = error.message; });
  form.querySelector('[data-race-outcome-save]')?.addEventListener('click', async (event) => {
    const panel = form.querySelector('[data-race-outcome]');
    const choice = panel.querySelector('[data-race-outcome-choice]').value;
    const message = panel.querySelector('[data-race-outcome-message]');
    if (!choice || !goalStatus?.outcome?.expected_revision) return;
    event.currentTarget.disabled = true;
    try {
      goalStatus = await request('/api/settings/race-preparation/outcome', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({status: choice === 'did_not_start' ? 'did_not_start' : 'recorded',
          physical_key: choice === 'did_not_start' ? null : choice, expected_revision: goalStatus.outcome.expected_revision}),
      });
      showOutcome(goalStatus);
      message.textContent = goalStatus.detail;
    } catch (error) { message.textContent = error.message; }
    finally { panel.querySelector('[data-race-outcome-save]').disabled = false; }
  });
  apply.addEventListener('click', async () => {
    if (!date.disabled && !date.reportValidity()) return;
    if (!goalStatus?.expected_revision) return;
    const draft = readRacePreparation(form);
    const target=form.querySelector('[name="planner_long_goal"]');
    const includeTarget=draft.distance==='general_fitness'&&planner?.status==='adopted'&&target&&!target.disabled;
    if(includeTarget&&!target.reportValidity())return;
    apply.disabled = true;
    output.textContent = 'Applying training goal…';
    try {
      goalStatus = await request('/api/settings/race-preparation/apply', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: draft, expected_revision: goalStatus.expected_revision,
          ...(includeTarget?{planner_preferences:{...readPlannerPreferences(form,planner,{goalOnly:true}),
            expected_revision:planner.preferences_revision}}:{}) }),
      });
      if(goalStatus.planner)Object.assign(planner,goalStatus.planner);
      invalidateSettings();
      output.textContent = goalStatus.detail;
    } catch (error) { output.textContent = error.message; }
    finally { apply.disabled = goalStatus?.status !== 'available' ||
      (goal.value !== 'general_fitness' && !goalStatus.race_activation_available); }
  });
  button.addEventListener('click', async () => {
    if (!date.reportValidity()) return;
    const token = revision;
    const draft = readRacePreparation(form);
    button.disabled = true;
    output.textContent = 'Checking preparation…';
    try {
      const result = await request('/api/settings/race-preparation/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft),
      });
      if (token === revision) output.textContent = preparationSummary(result, draft.preparation_mode);
    } catch (error) { if (token === revision) output.textContent = error.message; }
    finally { button.disabled = false; }
  });
}
