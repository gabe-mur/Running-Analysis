import { appSettings } from './api.js';

const windows = [14, 28, 42, 56, 90, 180, 365];
let selectedWindow = null;

// One browsing-session selection for Dashboard and Progress. No settings write.
export function selectFitnessWindow(days) {
  if (!windows.includes(days)) throw new Error('Unsupported fitness window');
  selectedWindow = days;
}

export async function fitnessWindow() {
  if (selectedWindow !== null) return selectedWindow;
  const settings = await appSettings();
  return selectedWindow ?? (windows.includes(settings.default_fitness_window) ? settings.default_fitness_window : 28);
}
