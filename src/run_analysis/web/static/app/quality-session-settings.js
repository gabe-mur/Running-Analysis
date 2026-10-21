import { escapeHtml, titleCase } from './format.js';

export function qualitySessionsCard(sessions, focus = 'variety') {
  const fields = Object.entries(sessions).map(([name, enabled]) =>
    `<label class="settings-check"><input type="checkbox" name="quality_${escapeHtml(name)}" ${enabled ? 'checked' : ''}>${escapeHtml(titleCase(name))}</label>`
  ).join('');
  return `<fieldset><legend>Quality sessions</legend>
    <label>Quality focus<select name="quality_focus">
      <option value="variety" ${focus === 'variety' ? 'selected' : ''}>Balanced variety</option>
      <option value="vo2max" ${focus === 'vo2max' ? 'selected' : ''}>VO₂max</option>
    </select></label>
    <p id="quality-focus-help">${focus === 'vo2max' ? 'Favors hard aerobic intervals, with some sustained work.' : 'Mixes sustained work and faster efforts.'}</p>
    <div class="settings-quality-options">${fields}</div><p>Disabled types will not be prescribed.</p></fieldset>`;
}

export function bindQualityFocus(form) {
  const focus = form.querySelector('[name="quality_focus"]');
  const intervals = form.querySelector('[name="quality_long_intervals"]');
  const help = form.querySelector('#quality-focus-help');
  const update = () => {
    help.textContent = focus.value !== 'vo2max' ? 'Mixes sustained work and faster efforts.'
      : intervals.checked ? 'Favors hard aerobic intervals, with some sustained work.'
      : 'Enable Long Intervals for sustained hard aerobic repeats. Otherwise, only your enabled types are used.';
  };
  focus.addEventListener('change', update);
  intervals.addEventListener('change', update);
  update();
}
