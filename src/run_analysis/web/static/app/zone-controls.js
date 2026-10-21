import { MANUAL_METHODS, ZONE_NAMES, bindZoneSteppers, bindHeartRateSource, syncZoneEditStatus } from './setup-fields.js';

// Settings edits are drafts. Preview only reads data; Save waits for the exact
// current inputs, so a late response cannot submit a stale set of zones.
export function bindSettingsZones(form, settings, request) {
  const field = name => form.querySelector(`[name="${name}"]`);
  const status = form.querySelector('#settings-zone-status');
  let method = settings.setup?.zone_method ?? 'device';
  let dirty = false, generation = 0, cached = null, watchRecorded = false;
  const inflight = new Map();
  const syncSteppers = bindZoneSteppers(form);
  const hrSource = bindHeartRateSource(form, method);
  function values() {
    if (method === 'device') return {method};
    return {method, max_hr: Number(field('max_hr').value),
      resting_hr: Number(field('resting_hr').value),
      ...(MANUAL_METHODS.includes(method) ? {boundaries: Object.fromEntries(ZONE_NAMES.map(name => [name,
        {minimum_bpm: Number(field(`${name}_min`).value), maximum_bpm: Number(field(`${name}_max`).value)}]))} : {})};
  }
  function sync() {
    form.querySelectorAll('[data-method]').forEach(card => card.classList.toggle('chosen', card.dataset.method === method));
    for (const name of ZONE_NAMES) {
      field(`${name}_min`).readOnly = field(`${name}_max`).readOnly = !MANUAL_METHODS.includes(method)
        || (method === 'device' && watchRecorded);
    }
    syncSteppers();
    syncZoneEditStatus(form, method);
  }
  async function preview() {
    for (;;) {
      const input = values(), key = JSON.stringify(input);
      if (cached?.key === key) return cached.result;
      if (!inflight.has(key)) {
        const promise = request('/api/setup/zone-preview', {method: 'POST',
          headers: {'Content-Type': 'application/json'}, body: key});
        inflight.set(key, promise);
        promise.finally(() => inflight.delete(key)).catch(() => {});
      }
      const result = await inflight.get(key);
      if (JSON.stringify(values()) !== key) continue;
      if (!result.zones?.z2) throw new Error('Could not calculate heart-rate zones. No settings were saved.');
      cached = {key, result};
      hrSource.imported(result);
      cached.key = JSON.stringify(values());
      watchRecorded = Boolean(result.watch_recorded_at); sync();
      for (const name of ZONE_NAMES) {
        field(`${name}_min`).value = result.zones[name].minimum_bpm;
        field(`${name}_max`).value = result.zones[name].maximum_bpm;
        const current = form.querySelector(`[data-zone="${name}"] [data-zone-current]`);
        if (current) current.textContent = `${result.zones[name].minimum_bpm}–${result.zones[name].maximum_bpm} bpm`;
      }
      status.textContent = '';
      return result;
    }
  }
  const changed = () => {
    dirty = true; generation += 1;
    status.textContent = 'Calculating…';
    preview().catch(error => {status.textContent = error.message;});
  };
  form.querySelectorAll('input[name="method"]').forEach(input => input.addEventListener('change', () => {
    method = input.value; hrSource.sync(method); sync(); changed();
  }));
  for (const name of ['max_hr', 'resting_hr', ...ZONE_NAMES.flatMap(name => [`${name}_min`, `${name}_max`])]) {
    field(name).addEventListener('input', changed);
  }
  sync();
  // Reads never recalibrate or save the athlete; adoption is an explicit save.
  const initialGeneration = generation;
  if (method === 'device') preview().catch(error => {
    if (!dirty && generation === initialGeneration) status.textContent = error.message;
  });
  return {
    async calibration() {
      const result = dirty || method === 'device' ? await preview() : null;
      return {zones: result?.zones ?? settings.zones,
        max_hr: Number(field('max_hr').value), resting_hr: Number(field('resting_hr').value),
        setup: {...settings.setup, zone_method: method, comparison_hr_mode: 'automatic'}};
    },
    saved(updated) {
      Object.assign(settings, updated); generation += 1; cached = null; dirty = false;
      for (const name of ['max_hr','resting_hr']) field(name).value = updated[name];
      for (const name of ZONE_NAMES) {
        field(`${name}_min`).value = updated.zones[name].minimum_bpm;
        field(`${name}_max`).value = updated.zones[name].maximum_bpm;
        const current = form.querySelector(`[data-zone="${name}"] [data-zone-current]`);
        if (current) current.textContent = `${updated.zones[name].minimum_bpm}–${updated.zones[name].maximum_bpm} bpm`;
      }
    },
  };
}
