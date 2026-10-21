// Guided setup. Not a settings page: a settings page lists fields, this asks
// the few questions whose answers every other number depends on, shows what
// each answer does before it is saved, and says plainly when it is guessing.
//
// One form, one save action. Calibration is written together; planner
// preferences use their separate ledger transaction with explicit partial-
// failure reporting. Nothing is confirmed before its save succeeds.

import { api, primeSettings } from "../api.js";
import { view, loading } from "../dom.js";
import { escapeHtml } from "../format.js";
import { plannerSetupSection, readPlannerPreferences } from "../planner-settings.js";
import { saveSetupSettings } from "../setup-save.js";
import { ageFromBirthDate, baselineInstructions } from '../setup-baseline.js';
import { setupDeviceCard } from '../setup-device.js';
import { trainingBackgroundFields, bindTrainingBackground, readTrainingBackground } from '../training-background.js';
import { ZONE_NAMES, MANUAL_METHODS, profileFields, heartRateFields, bindZoneSteppers, bindHeartRateSource, syncZoneEditStatus } from '../setup-fields.js';
import { bindRacePreparation, racePreparationFields, readRacePreparation } from '../race-settings.js';

// Calibration steps are confirmed together. Progression is confirmed only
// after the adaptive preferences transaction has succeeded.
const COVERED_STEPS = ["profile", "heart_rate", "zones", "goal", "weather"];

function stepRow(step) {
  const state = step.complete ? "done" : step.blocking ? "blocked" : "todo";
  return `<li class="setup-step ${state}">
    <i aria-hidden="true">${step.complete ? "✓" : step.blocking ? "!" : ""}</i>
    <div><b>${escapeHtml(step.title)}</b><small>${escapeHtml(step.detail)}</small></div>
  </li>`;
}

function checklistMarkup(state) {
  const done = state.steps.filter((step) => step.complete).length;
  return `<div class="card-heading"><div><p class="eyebrow">Progress</p><h2>${done} of ${state.steps.length} confirmed</h2></div></div>
    <ol class="setup-steps">${state.steps.map(stepRow).join("")}</ol>`;
}

export async function renderSetup() {
  loading("Setup");
  const [state, settings] = await Promise.all([api("/api/setup"), api("/api/settings")]);

  // Local mirror of what the form currently proposes. Nothing is written until
  // Save, so this is the only place a half-made decision lives.
  const draft = {
    max_hr: settings.max_hr,
    resting_hr: settings.resting_hr,
    max_hr_source: settings.profile?.max_hr_source ?? "estimated",
    method: settings.setup?.zone_method ?? "device",
    zones: Object.fromEntries(ZONE_NAMES.map((name) => [name, { ...settings.zones[name] }])),
  };

  const blocked = state.steps.find((step) => step.step === 'runs' && !step.complete);
  const chooseHistory = blocked && !settings.training_background && !['profile', 'heart_rate', 'zones'].every(name =>
    state.steps.some(step => step.step === name && step.complete));
  view.innerHTML = `
    <section class="page narrow setup-page">
      <div class="page-heading"><div>
        <p class="eyebrow">Setup</p>
        <h1>${state.complete ? "You are set up" : "Set up your training"}</h1>
        <p>${blocked
          ? "Have recent runs? Import them first. If not, we’ll help you set up your heart-rate zones before your first baseline run."
          : state.complete
          ? "Every setting below has been confirmed rather than defaulted. Change any of them any time."
          : "Confirm your personal details, heart-rate zones and training preferences."}</p>
      </div></div>

      ${setupDeviceCard()}

      ${blocked ? `<article class="wide-card setup-block" id="setup-import">
        <h2>Add your runs first</h2>
        <p>Drop FIT, TCX or GPX activity files, or a Garmin ZIP, anywhere on this page, or choose <b>Upload Runs</b> at the top right.</p>
        <p>Already connected Intervals.icu? Use the <b>Sync</b> button next to Upload Runs.</p>
        ${chooseHistory ? '<button type="button" id="setup-no-history">I don’t have recorded runs</button>' : ''}
      </article>` : ''}

      <article class="wide-card" id="setup-checklist" ${chooseHistory ? 'hidden' : ''}>${checklistMarkup(state)}</article>

      <form id="setup-form" ${chooseHistory ? 'hidden' : ''}>
        ${settings.planner?.status !== 'adopted' && (blocked || settings.training_background || settings.beginner_start) ? trainingBackgroundFields(settings) : ''}
        <section class="wide-card setup-block">
          <p class="eyebrow">Step 1</p>
          <h2>About you</h2>
          <p class="chart-note">Used for your fitness estimate and an age-based maximum HR estimate if needed.</p>
          <div class="setup-fields">${profileFields(settings.profile)}</div>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 2</p>
          <h2>Heart-rate zones</h2>
          ${heartRateFields(settings, {method: draft.method})}
          <div id="zone-preview" class="zone-preview-slot"></div>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 3</p>
          <h2>What you are training for</h2>
          <div class="setup-fields">
            ${racePreparationFields(settings.race_preparation,settings.planner,{setup:true})}
          </div>
        </section>

        ${plannerSetupSection(settings.planner, {step: 4})}

        <section class="wide-card setup-block">
          <p class="eyebrow">Step ${settings.planner ? 5 : 4}</p>
          <h2>Weather and privacy</h2>
          <p class="chart-note">Heat and humidity change your pace at a given heart rate, so weather makes the comparison fairer. Only a date and a rounded, randomly offset location are ever sent — never your route.</p>
          <div class="setup-toggles">
            <label class="toggle"><input type="checkbox" name="historical_weather_enabled" ${settings.historical_weather_enabled ? "checked" : ""}><span>Look up weather for past runs</span></label>
            <label class="toggle"><input type="checkbox" name="forecast_weather_enabled" ${settings.forecast_weather_enabled ? "checked" : ""}><span>Look up forecasts when planning</span></label>
            <label>Location blur radius<input type="number" step="0.5" min="0" max="25" name="weather_privacy_radius_km" value="${settings.weather_privacy_radius_km}"><small>Kilometres. Larger is more private and slightly less accurate.</small></label>
          </div>
        </section>

        <div class="setup-save">
          <button class="primary-button" type="submit">Save setup</button>
          <span class="form-status" id="setup-status"></span>
        </div>
      </form>
      <div id="setup-baseline" ${chooseHistory ? 'hidden' : ''}>${baselineInstructions(state, settings)}</div>
    </section>`;

  bindRacePreparation(view.querySelector('#setup-form'),api,{planner:settings.planner});
  bindTrainingBackground(view.querySelector('#setup-form'));
  view.querySelector('#setup-no-history')?.addEventListener('click', () => {
    for (const id of ['setup-form', 'setup-checklist', 'setup-baseline']) {
      view.querySelector(`#${id}`).hidden = false;
    }
    view.querySelector('#setup-import').hidden = true;
    const birthDate = view.querySelector('[name="profile_birth_date"]');
    birthDate.scrollIntoView({block: 'center'});
    birthDate.focus({preventScroll: true});
  });
  wire(draft, settings);
}

function wire(draft, settings) {
  const form = view.querySelector("#setup-form");
  const el = (selector) => view.querySelector(selector);
  const field = (name) => form.elements[name];
  const syncSteppers = bindZoneSteppers(form);
  const hrSource = bindHeartRateSource(form, draft.method);

  function readDraft() {
    draft.max_hr = Number(field("max_hr").value);
    draft.resting_hr = Number(field("resting_hr").value);
    draft.max_hr_source = field("max_hr_source").value;
    if (MANUAL_METHODS.includes(draft.method)) {
      for (const name of ZONE_NAMES) {
        draft.zones[name] = {
          minimum_bpm: Number(field(`${name}_min`).value),
          maximum_bpm: Number(field(`${name}_max`).value),
        };
      }
    }
  }

  // The preview is the feedback loop, so it runs on its own rather than
  // behind a button: a section whose only control is disabled reads as broken.
  let pending = null;
  let sequence = 0, watchRecorded = false;
  function schedulePreview() {
    clearTimeout(pending);
    pending = setTimeout(runPreview, 300);
  }

  async function runPreview() {
    readDraft();
    const slot = el("#zone-preview");
    const ticket = ++sequence;
    slot.innerHTML = `<p class="chart-note">Calculating…</p>`;
    try {
      const preview = await api("/api/setup/zone-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method: draft.method,
          ...(draft.method !== 'device' ? {max_hr: draft.max_hr, resting_hr: draft.resting_hr} : {}),
          ...(MANUAL_METHODS.includes(draft.method) ? { boundaries: draft.zones } : {}),
        }),
      });
      // A slower earlier request must not overwrite a newer answer.
      if (ticket !== sequence) return false;
      draft.zones = preview.zones;
      renderPreview(preview);
      return true;
    } catch (error) {
      if (ticket !== sequence) return;
      slot.innerHTML = `<p class="chart-note error-text">${escapeHtml(error.message)}</p>`;
      return false;
    }
  }

  function renderPreview(preview) {
    hrSource.imported(preview);
    readDraft();
    watchRecorded = Boolean(preview.watch_recorded_at);
    for (const name of ZONE_NAMES) {
      const zone = preview.zones[name];
      el(`[data-zone="${name}"] [data-zone-current]`).textContent = `${zone.minimum_bpm}–${zone.maximum_bpm} bpm`;
      if (!MANUAL_METHODS.includes(draft.method) || watchRecorded) {
        field(`${name}_min`).value = zone.minimum_bpm;
        field(`${name}_max`).value = zone.maximum_bpm;
      }
    }
    syncZoneInputs();
    el("#zone-preview").innerHTML = '';
  }

  function syncZoneInputs() {
    for (const name of ZONE_NAMES) {
      field(`${name}_min`).readOnly = field(`${name}_max`).readOnly = !MANUAL_METHODS.includes(draft.method)
        || (draft.method === 'device' && watchRecorded);
    }
    syncSteppers();
    syncZoneEditStatus(form, draft.method);
  }

  view.querySelectorAll('input[name="method"]').forEach((input) =>
    input.addEventListener("change", () => {
      draft.method = input.value;
      hrSource.sync(draft.method);
      view.querySelectorAll(".method-option").forEach((option) =>
        option.classList.toggle("chosen", option.dataset.method === draft.method));
      syncZoneInputs();
      runPreview();
    }),
  );

  ["max_hr", "resting_hr"].forEach((name) =>
    field(name).addEventListener("input", schedulePreview));
  el("#zone-inputs").addEventListener("input", schedulePreview);

  el('#estimate-max-hr').addEventListener('click', async () => {
    const status = el('#max-hr-estimate-status');
    const age = ageFromBirthDate(field('profile_birth_date').value);
    if (age === null) {
      status.textContent = 'Enter a valid birth date above first.';
      return;
    }
    const button = el('#estimate-max-hr');
    button.disabled = true;
    const birthDate = field('profile_birth_date').value;
    const method = draft.method;
    try {
      const estimate = await api(`/api/setup/max-hr?age_years=${age}`);
      if (field('profile_birth_date').value !== birthDate || draft.method !== method) return;
      field('max_hr').value = estimate.estimated_max_hr;
      field('max_hr_source').value = 'estimated';
      status.textContent = 'Age estimate—not a measured maximum.';
      await runPreview();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    readDraft();
    const status = el("#setup-status");
    const button = form.querySelector('button[type="submit"]');
    status.textContent = "Saving and recalculating…";
    status.className = "form-status";
    button.disabled = true;
    try {
      // Resolve the current inputs before saving; a pending older preview must
      // not let new HR values be saved with stale calculated zones.
      clearTimeout(pending);
      if (!await runPreview()) throw new Error('Check your heart-rate zones before saving.');
      if (ageFromBirthDate(field('profile_birth_date').value) === null) throw new Error('Enter a valid birth date.');
      // Calibration still saves together; adaptive preferences use their own
      // ledger transaction. A failed second save is explicitly reported.
      const preferences = settings.planner?.status === 'adopted' ? readPlannerPreferences(form,settings.planner,{setup:true}) : null;
      const background = readTrainingBackground(form);
      const updated = await saveSetupSettings(settings, {
          ...(background !== undefined ? {training_background:background, beginner_start:background===null} : {}),
          max_hr: draft.max_hr,
          resting_hr: draft.resting_hr,
          zones: draft.zones,
          profile: {
            birth_date: field('profile_birth_date').value,
            sex: field('profile_sex').value,
            weight_lb: Number(field('profile_weight_lb').value),
            height_in: Number(field('profile_height_in').value),
            max_hr_source: draft.max_hr_source,
          },
          race_preparation: readRacePreparation(form),
          historical_weather_enabled: field("historical_weather_enabled").checked,
          forecast_weather_enabled: field("forecast_weather_enabled").checked,
          weather_privacy_radius_km: Number(field("weather_privacy_radius_km").value),
          setup: { confirmed_steps: COVERED_STEPS, zone_method: draft.method, comparison_hr_mode: 'automatic' },
      }, preferences, api, {applyTrainingGoal: true});
      Object.assign(settings, updated);
      primeSettings(updated);
      status.textContent = updated.recalculation.some(step => step.status === 'failed')
        ? 'Settings saved; some analysis needs attention.'
        : updated.recalculation.some(step => step.name === 'goal_activation' && step.status === 'deferred')
        ? 'Saved. Add your runs next; apply your training goal once a starting plan is available.'
        : 'Setup saved.';
      if (updated.planner?.preferences_pending) status.textContent += " Running preferences apply on the next plan update.";
      status.className = "form-status ok";
      // Refresh only the checklist. Re-rendering the whole page under someone
      // who just pressed Save throws away their scroll position for nothing.
      const refreshed = await api("/api/setup");
      el("#setup-checklist").innerHTML = checklistMarkup(refreshed);
      el('#setup-baseline').innerHTML = baselineInstructions(refreshed, settings);
    } catch (error) {
      status.textContent = error.message;
      status.className = "form-status error-text";
    } finally {
      button.disabled = false;
    }
  });

  syncZoneInputs();
  runPreview();
}
