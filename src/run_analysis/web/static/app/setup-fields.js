import { escapeHtml } from './format.js';

export const ZONE_METHODS = [
  ['device', 'Match my watch', 'Requires a FIT file containing your watch’s maximum HR, resting HR and zones.'],
  ['heart_rate_reserve', 'Heart-rate reserve (HRR)', 'Uses maximum and resting HR. A useful starting estimate if you don’t have watch zones.'],
  ['percent_max', 'Percent of maximum HR', 'Uses maximum HR only. Does not account for your resting HR.'],
  ['custom', 'Custom zones', 'Use ranges from a fitness test or a coach.'],
];
export const ZONE_NAMES = ['z1', 'z2', 'z3', 'z4', 'z5'];
export const MANUAL_METHODS = ['custom'];

const ZONE_EDIT_MESSAGES = {
  device: 'Read-only: matched to your watch. Choose Custom zones to edit ranges.',
  heart_rate_reserve: 'Calculated from maximum and resting HR. Adjust those above, or choose Custom zones to edit ranges.',
  percent_max: 'Calculated from maximum HR. Adjust it above, or choose Custom zones to edit ranges.',
  custom: '',
};

export function syncZoneEditStatus(container, method) {
  const note = container.querySelector('[data-zone-edit-status]');
  if (!note) return;
  note.textContent = ZONE_EDIT_MESSAGES[method] ?? 'Choose Custom zones to edit ranges.';
  note.hidden = !note.textContent;
}

function zoneNumberField(name, edge, value) {
  const label = `${name.toUpperCase()} ${edge === 'min' ? 'minimum' : 'maximum'}`;
  return `<span class="zone-number-field">
    <input type="number" name="${name}_${edge}" aria-label="${label} bpm" aria-describedby="zone-edit-status" min="40" max="250" step="1" value="${value}" required>
    <span class="zone-steppers">
      <button type="button" data-zone-step="1" aria-label="Increase ${label}"><svg viewBox="0 0 12 8" aria-hidden="true"><path d="M2 6 6 2 10 6"/></svg></button>
      <button type="button" data-zone-step="-1" aria-label="Decrease ${label}"><svg viewBox="0 0 12 8" aria-hidden="true"><path d="M2 2 6 6 10 2"/></svg></button>
    </span>
  </span>`;
}

export function bindZoneSteppers(container) {
  const buttons = [...container.querySelectorAll('[data-zone-step]')];
  for (const button of buttons) button.addEventListener('click', () => {
    const input = button.closest('.zone-number-field').querySelector('input');
    if (input.disabled || input.readOnly) return;
    input.stepUp(Number(button.dataset.zoneStep));
    input.dispatchEvent(new Event('input', {bubbles: true}));
  });
  return () => buttons.forEach(button => {
    const input = button.closest('.zone-number-field').querySelector('input');
    button.disabled = input.disabled || input.readOnly;
  });
}

export function profileFields(profile) {
  return `<label>Birth date<input type="date" name="profile_birth_date" value="${escapeHtml(profile?.birth_date ?? '')}" required></label>
    <label>Sex used by the equation<select name="profile_sex" required>
      <option value="" ${profile?.sex ? '' : 'selected'}>Choose</option>
      <option value="male" ${profile?.sex === 'male' ? 'selected' : ''}>Male</option>
      <option value="female" ${profile?.sex === 'female' ? 'selected' : ''}>Female</option>
    </select></label>
    <label>Weight (lb)<input type="number" name="profile_weight_lb" min="50.1" max="1000" step="0.1" value="${profile?.weight_lb ?? ''}" required></label>
    <label>Height (in)<input type="number" name="profile_height_in" min="36.1" max="100" step="0.1" value="${profile?.height_in ?? ''}" required></label>`;
}

export function zoneMethodFields(method) {
  return `<div class="method-options">${ZONE_METHODS.map(([value, label, detail]) => `
    <label class="method-option ${method === value ? 'chosen' : ''}" data-method="${value}">
      <input type="radio" name="method" value="${value}" ${method === value ? 'checked' : ''}>
      <b>${escapeHtml(label)}</b><small>${escapeHtml(detail)}</small>
    </label>`).join('')}</div>`;
}

// One calibration layout for the first-run journey and returning-user settings.
// Source ownership and any unsupported accounting-range migration are checked
// on the backend; watch values are previewed read-only in both views.
export function heartRateFields(settings, {locked = false,
  method = settings.setup?.zone_method ?? 'device',
  estimateId = 'estimate-max-hr', estimateStatusId = 'max-hr-estimate-status'} = {}) {
  const source = settings.profile?.max_hr_source ?? 'estimated';
  return `${zoneMethodFields(method)}
    <div class="setup-fields heart-rate-inputs">
      <label>Maximum HR (bpm)<input type="number" name="max_hr" min="100" max="250" value="${settings.max_hr}" ${locked ? 'disabled' : ''} required></label>
      <label>Resting HR (bpm)<input type="number" name="resting_hr" min="25" max="120" value="${settings.resting_hr}" ${locked ? 'disabled' : ''} required></label>
      <label data-manual-hr>Maximum HR source<select name="max_hr_source">
        <option value="measured" ${source === 'measured' ? 'selected' : ''}>Measured in a hard effort</option>
        <option value="estimated" ${source === 'estimated' ? 'selected' : ''}>Estimated</option>
        <option value="watch" ${source === 'watch' ? 'selected' : ''} hidden>From watch</option>
      </select></label>
    </div>
    <div class="heart-rate-estimate" data-manual-hr>
      <button type="button" id="${estimateId}" ${locked ? 'disabled' : ''}>Estimate max HR from age</button>
      <span id="${estimateStatusId}" class="form-status" role="status"></span>
    </div>
    ${locked ? '<small class="field-help">Maximum and resting HR are locked for the active plan.</small>' : ''}
    <p class="field-help zone-edit-status" id="zone-edit-status" data-zone-edit-status role="status" ${method === 'custom' ? 'hidden' : ''}>${escapeHtml(ZONE_EDIT_MESSAGES[method] ?? 'Choose Custom zones to edit ranges.')}</p>
    <div class="zone-calibration-list" id="zone-inputs">${zoneCalibrationRows(settings.zones)}</div>`;
}

// Keep manual drafts separate while browsing watch settings; a preview is not
// a save and must not silently overwrite someone's unsaved custom reference.
export function bindHeartRateSource(form, initialMethod) {
  const field = name => form.querySelector(`[name="${name}"]`);
  let method = initialMethod;
  let manual = null;
  if (method !== 'device' && field('max_hr_source').value === 'watch') {
    field('max_hr_source').value = 'estimated';
  }
  const read = () => Object.fromEntries(['max_hr','resting_hr','max_hr_source',
    ...ZONE_NAMES.flatMap(n => [`${n}_min`,`${n}_max`])].map(n => [n,field(n).value]));
  if (method !== 'device') manual = read();
  function sync(next = method) {
    if (next !== method) {
      if (method !== 'device') manual = read();
      if (next !== 'device' && manual) {
        for (const [name,value] of Object.entries(manual)) field(name).value = value;
      } else if (next !== 'device' && field('max_hr_source').value === 'watch') {
        field('max_hr_source').value = 'estimated';
      }
    }
    method = next;
    for (const name of ['max_hr','resting_hr']) field(name).readOnly = method === 'device';
    form.querySelectorAll('[data-manual-hr]').forEach(node => {node.hidden = method === 'device';});
  }
  sync();
  return {sync, imported(result) {
    if (method !== 'device' || !result.watch_recorded_at) return;
    field('max_hr').value = result.max_hr;
    field('resting_hr').value = result.resting_hr;
    field('max_hr_source').value = 'watch';
  }};
}

export function zoneFields(zones) {
  return ZONE_NAMES.map(name => `<label>${name.toUpperCase()}
    <span class="zone-pair">
      ${zoneNumberField(name,'min',zones[name].minimum_bpm)}
      <span>to</span>
      ${zoneNumberField(name,'max',zones[name].maximum_bpm)}
      <span>bpm</span>
    </span></label>`).join('');
}

export function zoneCalibrationRows(zones) {
  return ZONE_NAMES.map(name => `<div class="zone-calibration-row ${name === 'z2' ? 'zone-highlight' : ''}" data-zone="${name}">
    <span class="zone-label">${name.toUpperCase()}</span>
    <span class="zone-current" data-zone-current>${zones[name].minimum_bpm}–${zones[name].maximum_bpm} bpm</span>
    <span class="zone-pair">
      ${zoneNumberField(name,'min',zones[name].minimum_bpm)}
      <span>to</span>
      ${zoneNumberField(name,'max',zones[name].maximum_bpm)}
      <span>bpm</span>
    </span>
  </div>`).join('');
}
