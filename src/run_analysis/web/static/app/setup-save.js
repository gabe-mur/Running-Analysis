import { api, primeSettings } from './api.js';

// The adaptive ledger owns preferences; calibration owns the settings overlay.
// Keep their API transactions separate, while offering one Save setup action.
export async function saveSetupSettings(settings, payload, preferences, request = api, {applyTrainingGoal = false} = {}) {
  let preferencesSaved = false;
  if (preferences && settings.planner?.status === 'adopted') {
    const saved = await request('/api/settings', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ planner_preferences: {
        ...preferences, expected_revision: settings.planner.preferences_revision,
      } }),
    });
    // Retain the new revision even if the second save fails, so retry is safe.
    Object.assign(settings.planner, saved.planner);
    primeSettings(saved);
    preferencesSaved = true;
  }
  const confirmed = new Set([
    ...(settings.setup?.confirmed_steps ?? []), ...payload.setup.confirmed_steps,
  ]);
  if (preferencesSaved) confirmed.add('progression');
  // An active planner's goal is confirmed only after activation succeeds.
  if (applyTrainingGoal && settings.planner?.status === 'adopted') confirmed.delete('goal');
  try {
    const updated = await request('/api/settings', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, setup: {
        ...payload.setup, confirmed_steps: [...confirmed],
      } }),
    });
    Object.assign(settings, updated);
    primeSettings(updated);
    if (applyTrainingGoal) {
      if (updated.planner?.status !== 'adopted') {
        updated.recalculation = [...(updated.recalculation ?? []), {name: 'goal_activation', status: 'deferred',
          detail: 'Training goal saved. It will need applying once your starting plan is available.'}];
      } else {
        try {
          const status = await request('/api/settings/race-preparation/status');
          if (status.status !== 'available' || !status.expected_revision)
            throw new Error(status.detail || 'Your starting plan is not available.');
          const goal = await request('/api/settings/race-preparation/apply', {method: 'POST',
            headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
              goal: payload.race_preparation, expected_revision: status.expected_revision,
            })});
          if (goal.planner) Object.assign(updated.planner, goal.planner);
          confirmed.add('goal');
          const confirmation = await request('/api/settings', {method: 'PATCH',
            headers: {'Content-Type': 'application/json'}, body: JSON.stringify({setup: {
              ...updated.setup, confirmed_steps: [...confirmed],
            }})});
          updated.setup = confirmation.setup;
          updated.recalculation = [...(updated.recalculation ?? []), {name: 'goal_activation', status: 'complete',
            detail: goal.detail || 'Training goal applied.'}];
        } catch (error) {
          throw new Error(`Personal details and HR settings are saved, but the training goal could not be applied: ${error.message}`);
        }
      }
    }
    Object.assign(settings, updated);
    primeSettings(updated);
    return updated;
  } catch (error) {
    if (preferencesSaved) throw new Error(`Running preferences are saved, but setup is not complete: ${error.message}`);
    throw error;
  }
}
