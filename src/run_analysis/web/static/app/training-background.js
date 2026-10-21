import { escapeHtml } from './format.js';

export function trainingBackgroundFields(settings) {
  const value = settings.training_background;
  return `<section class="wide-card setup-block training-background" id="training-background">
    <h2>Your current running</h2>
    <label>Do you already run regularly?<select name="background_choice">
      <option value="new" ${!value ? 'selected' : ''}>I’m new or rebuilding from scratch</option>
      <option value="experienced" ${value ? 'selected' : ''}>I already run, but don’t have recorded history</option>
    </select></label>
    ${!value ? '<p>Start with a 10–30-minute easy recording. Continuous running is fine; take walking breaks if needed. Your starter schedule builds from what you demonstrate, then hands over to the full planner.</p>' : ''}
    <div data-background-fields ${value ? '' : 'hidden'}>
      <p>Enter your usual current training—not your best week or your future goal.</p>
      <div class="setup-fields">
        <label>Weekly volume (miles)<input name="background_volume" type="number" min="0.1" max="200" step="0.1" value="${value?.weekly_miles ?? ''}"></label>
        <label>Runs per week<input name="background_frequency" type="number" min="2" max="7" step="1" value="${value?.runs_per_week ?? ''}"></label>
        <label>Regular long run (miles)<input name="background_long" type="number" min="0.1" max="100" step="0.1" value="${value?.regular_long_miles ?? ''}"></label>
        <label>Are you maintaining this training?<select name="background_maintaining">
          <option value="yes" ${value?.maintaining_training !== false ? 'selected' : ''}>Yes</option>
          <option value="no" ${value?.maintaining_training === false ? 'selected' : ''}>Returning after a break</option>
        </select></label>
        <label>Last run (optional)<input name="background_last_run" type="date" value="${escapeHtml(value?.last_run_date ?? '')}"></label>
      </div>
      <p>Record one normal easy run to calibrate pace and effort. Your reported volume, frequency and long run remain the starting level.</p>
      <p data-background-break ${value?.maintaining_training === false ? '' : 'hidden'}>If you’re returning after a break, enter a comfortable current level or choose rebuilding from scratch. We won’t prescribe your former volume automatically.</p>
    </div>
  </section>`;
}

export function bindTrainingBackground(form) {
  const choice=form.querySelector('[name="background_choice"]');
  if (!choice) return;
  const update=()=>{
    const enabled=choice.value==='experienced';
    form.querySelector('[data-background-fields]').hidden=!enabled;
    for (const input of form.querySelectorAll('[data-background-fields] input, [data-background-fields] select')) {
      input.disabled=!enabled;
      input.required=enabled && ['background_volume','background_frequency','background_long'].includes(input.name);
    }
    form.querySelector('[data-background-break]').hidden=form.querySelector('[name="background_maintaining"]').value!=='no';
  };
  choice.addEventListener('change',update);
  form.querySelector('[name="background_maintaining"]').addEventListener('change',update);
  update();
}

export function readTrainingBackground(form) {
  const el=name=>form.querySelector(`[name="${name}"]`);
  if (!el('background_choice')) return undefined;
  if (el('background_choice').value!=='experienced') return null;
  return {weekly_miles:Number(el('background_volume').value),runs_per_week:Number(el('background_frequency').value),
    regular_long_miles:Number(el('background_long').value),maintaining_training:el('background_maintaining').value==='yes',
    last_run_date:el('background_last_run').value || null};
}
