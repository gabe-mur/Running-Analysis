import { escapeHtml } from './format.js';

// Use saved calibration, never a preview or the shipped defaults. A first run
// supplies training history; it is not a prerequisite for entering a profile.
export function baselineInstructions(state, settings) {
  const background=settings.training_background;
  if (settings.planner?.status === 'adopted') return '';
  if (!background && state.steps.some(step => step.step === 'runs' && step.complete)) return settings.beginner_start
    ? '<article class="wide-card setup-blocked"><h2>Your starter schedule</h2><p>Open <a href="#next-run">Weekly Plan</a> to check in on your recording and build comfortable running time. You’ll move to the full planner automatically after establishing regular continuous easy runs.</p></article>' : '';
  if (background && settings.planner?.status === 'adopted') return '';
  const ready = ['profile', 'heart_rate', 'zones'].every(name =>
    state.steps.some(step => step.step === name && step.complete));
  const zone = settings.zones?.z2;
  if (!ready || !zone) return `<article class="wide-card setup-blocked">
    <h2>Next: your first baseline run</h2>
    <p>Save your personal details and heart-rate zones above first. Then we’ll show your first-run instructions.</p>
  </article>`;
  const estimated = settings.profile?.max_hr_source === 'estimated'
    && ['heart_rate_reserve', 'percent_max'].includes(settings.setup?.zone_method);
  if (background) {
    if (!background.maintaining_training) return `<article class="wide-card setup-blocked"><h2>Your starting training level</h2><p>Enter a comfortable current level after your break, or choose rebuilding from scratch. Your former volume won’t be prescribed automatically.</p></article>`;
    const miles=Math.round((background.weekly_miles-background.regular_long_miles)/(background.runs_per_week-1)*10)/10;
    return `<article class="wide-card setup-blocked"><h2>Your first recorded easy run</h2>
      <p>Run <b>${miles.toFixed(1)} miles</b> at conversational Z1/Z2 effort. Record GPS and heart rate.</p>
      <p>Your starting level remains ${escapeHtml(background.weekly_miles)} miles per week, ${escapeHtml(background.runs_per_week)} runs and a ${escapeHtml(background.regular_long_miles)}-mile regular long run. This recording calibrates pace and effort; it doesn’t replace that history.</p>
      <p>Stop for pain or feeling unwell. After uploading, open Weekly Plan to build your full schedule.</p></article>`;
  }
  return `<article class="wide-card setup-blocked">
    <p class="eyebrow">Next</p><h2>Your first baseline run</h2>
    <p>Already have recent running history? Upload FIT, TCX or GPX activity files, or a Garmin ZIP, or use Sync instead.</p>
    <p>Record your baseline as a running workout with GPS and heart rate enabled.</p>
    <p>Otherwise, run or run/walk for 10–30 minutes at conversational effort in <b>Z2: ${escapeHtml(zone.minimum_bpm)}–${escapeHtml(zone.maximum_bpm)} bpm</b>. Use your watch to follow the range; don’t chase a distance.</p>
    ${estimated ? '<p class="chart-note">These zones use an age-estimated maximum HR. Keep the effort conversational rather than forcing yourself to reach a number.</p>' : ''}
    <p>Stop if you feel unwell, develop pain, or the effort stops feeling easy. Finish within 30 minutes, then upload or sync the run and open Weekly Plan.</p>
  </article>`;
}

export function ageFromBirthDate(value, today = new Date()) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  const date = new Date(year, month - 1, day);
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) return null;
  const beforeBirthday = today.getMonth() + 1 < month
    || (today.getMonth() + 1 === month && today.getDate() < day);
  const age = today.getFullYear() - year - Number(beforeBirthday);
  return age > 5 && age <= 110 ? age : null;
}
