import {escapeHtml} from './format.js';

export function beginnerTrainingMarkup(starter) {
  if (!starter) return '';
  const next=starter.next_feedback;
  const timeFields=next ? `<div class="setup-fields"><label>Total running (minutes)<input name="running_minutes" type="number" min="0.1" max="180" step="0.1" ${next.needs_times?'required':''}></label>
    <label>Longest running stretch (minutes)<input name="longest_bout_minutes" type="number" min="0.1" max="180" step="0.1" ${next.needs_times?'required':''}></label></div>` : '';
  return `<article class="wide-card beginner-training"><p class="eyebrow">Starter training</p>
    <h2>${starter.running_minutes != null ? `${escapeHtml(starter.running_minutes)} minutes of easy running` : 'Find your comfortable starting session'}</h2>
    <p>${escapeHtml(starter.summary)}</p>
    ${next ? `<form data-beginner-feedback="${Number(next.activity_id)}">
      <p><a href="#run/${Number(next.activity_id)}">Your last outing</a> · Was the running conversational and comfortable?</p>
      ${next.needs_times ? `<p>We couldn’t identify clear running stretches. If you ran, enter your running time, excluding walks.</p>${timeFields}`
        : `<details><summary>Correct running/walking times</summary>${timeFields}</details>`}
      <button type="submit" name="comfortable" value="yes">Yes, comfortable</button>
      <button type="submit" name="comfortable" value="no" formnovalidate>No, too hard</button>
      <p role="status" data-feedback-status></p>
    </form>` : ''}
    <p class="chart-note">Continuous runs or run/walk both count. The full planner takes over after repeated comfortable 30-minute continuous runs across at least a week, at your recorded training level.</p>
  </article>`;
}

export function beginnerFeedbackPayload(form, submitter) {
  const value={comfortable:submitter?.value==='yes'};
  if(!value.comfortable) return value;
  for (const name of ['running_minutes','longest_bout_minutes']) {
    const input=form.querySelector(`[name="${name}"]`);
    if(input && input.value!=='') value[name]=Number(input.value);
  }
  return value;
}
