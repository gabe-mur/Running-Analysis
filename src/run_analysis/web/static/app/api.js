// The single fetch wrapper plus the cached settings every view reads the
// comparison heart rate from.

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `HTTP ${response.status}`);
  return payload;
}

// The comparison heart rate is configurable, so no copy may hard-code it.
// One cached fetch backs every "at N bpm" label in the app; invalidated when
// settings are saved.
let cachedSettings = null;
async function appSettings() {
  if (!cachedSettings) cachedSettings = await api("/api/settings");
  return cachedSettings;
}
function invalidateSettings() { cachedSettings = null; }
let targetHr = null;
const targetHrLabel = () => (Number.isFinite(targetHr) ? `${targetHr} bpm` : "your comparison heart rate");

// Views render synchronously, so the comparison heart rate is cached as a
// plain value and refreshed once per navigation rather than awaited inline.
export function setTargetHr(value) { targetHr = Number.isFinite(value) ? value : null; }
// Settings already holds the authoritative response after a save, so it seeds
// the cache directly instead of forcing another round trip.
export function primeSettings(settings) { cachedSettings = settings; setTargetHr(settings?.target_hr); }
export async function refreshTargetHr() {
  try { setTargetHr((await appSettings()).target_hr); } catch { setTargetHr(null); }
}

export { api, appSettings, invalidateSettings, targetHrLabel };
