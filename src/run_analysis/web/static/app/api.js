// The single fetch wrapper plus the cached settings every view reads the
// comparison heart rate from.
import { announcePlanLoaded } from './plan-events.js';

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `HTTP ${response.status}`);
  if(['/api/weekly-schedule/latest','/api/weekly-schedule','/api/weekly-schedule/rest-day'].includes(path)
      && ['ready','review_required'].includes(payload?.planner_status)) announcePlanLoaded();
  return payload;
}

// The comparison heart rate is configurable, so no copy may hard-code it.
// One cached fetch backs every "at N bpm" label in the app; invalidated when
// settings are saved.
let cachedSettings = null;
let settingsRequest = null, settingsGeneration = 0;
async function appSettings() {
  if(cachedSettings)return cachedSettings;
  if(settingsRequest)return settingsRequest;
  const generation=settingsGeneration;
  const request=api('/api/settings').then(value=>{
    if(generation===settingsGeneration)cachedSettings=value;
    return value;
  }).finally(()=>{if(settingsRequest===request)settingsRequest=null;});
  settingsRequest=request;
  return request;
}
function invalidateSettings() { cachedSettings = null; settingsRequest=null; settingsGeneration+=1; }
let targetHr = null;
const targetHrLabel = () => (Number.isFinite(targetHr) ? `${targetHr} bpm` : "your comparison heart rate");

// Views render synchronously, so the comparison heart rate is cached as a
// plain value and refreshed once per navigation rather than awaited inline.
export function setTargetHr(value) { targetHr = Number.isFinite(value) ? value : null; }
// Settings already holds the authoritative response after a save, so it seeds
// the cache directly instead of forcing another round trip.
export function primeSettings(settings) { invalidateSettings(); cachedSettings = settings; setTargetHr(settings?.target_hr); }
export async function refreshTargetHr() {
  try { setTargetHr((await appSettings()).target_hr); } catch { setTargetHr(null); }
}

export { api, appSettings, invalidateSettings, targetHrLabel };
