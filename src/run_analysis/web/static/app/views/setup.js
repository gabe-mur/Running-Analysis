// Guided setup. Not a settings page: a settings page lists fields, this asks
// the few questions whose answers every other number depends on, shows what
// each answer does before it is saved, and says plainly when it is guessing.
//
// One form, one save. Saving step by step meant a half-configured state was
// reachable and the on-screen values could fall behind what was stored, so
// everything is reviewed together and written in a single request.

import { api, primeSettings } from "../api.js";
import { view, loading } from "../dom.js";
import { escapeHtml } from "../format.js";

const ZONE_METHODS = [
  ["device", "Copy from my watch", "What your watch already shows mid-run. Best option if you have it — two sets of zones means two answers to the same question."],
  ["heart_rate_reserve", "Calculate from max and resting", "Karvonen. Uses both numbers, so a low resting pulse moves your easy zone where it belongs."],
  ["percent_max", "Calculate from max only", "Percent of maximum. Cruder, but it needs one number instead of two."],
  ["custom", "Enter my own", "From a lab test, a field test, or a coach."],
];

const GOALS = [
  ["general_fitness", "General fitness"],
  ["5k", "5K"],
  ["10k", "10K"],
  ["half_marathon", "Half marathon"],
  ["marathon", "Marathon"],
];

const ZONE_NAMES = ["z1", "z2", "z3", "z4", "z5"];
const MANUAL_METHODS = ["device", "custom"];

//: Every step this page covers. Saving reviews all of them at once, so they
//: are confirmed together rather than one at a time.
const COVERED_STEPS = ["heart_rate", "zones", "comparison_hr", "goal", "weather"];

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
    comparison_hr: settings.target_hr,
  };

  view.innerHTML = `
    <section class="page narrow setup-page">
      <div class="page-heading"><div>
        <p class="eyebrow">Setup</p>
        <h1>${state.complete ? "You are set up" : "Two numbers decide everything else"}</h1>
        <p>${state.complete
          ? "Every setting below has been confirmed rather than defaulted. Change any of them any time."
          : "Your heart-rate zones decide what counts as easy, and your comparison heart rate is where every pace figure is measured. Until you confirm them, the app is using defaults that may not describe you."}</p>
      </div></div>

      <article class="wide-card" id="setup-checklist">${checklistMarkup(state)}</article>

      <form id="setup-form">
        <section class="wide-card setup-block">
          <p class="eyebrow">Step 1</p>
          <h2>Maximum and resting heart rate</h2>
          <p class="chart-note">Everything below is derived from these, so a wrong max quietly shifts every zone and the VO₂ estimate with it.</p>
          <div class="setup-fields">
            <label>Maximum heart rate<input type="number" name="max_hr" min="100" max="250" value="${draft.max_hr}" required></label>
            <label>Resting heart rate<input type="number" name="resting_hr" min="25" max="120" value="${draft.resting_hr}" required></label>
            <label>Where the max came from<select name="max_hr_source">
              <option value="measured" ${draft.max_hr_source === "measured" ? "selected" : ""}>Measured in a hard effort</option>
              <option value="estimated" ${draft.max_hr_source === "estimated" ? "selected" : ""}>Estimated from my age</option>
            </select><small>Measured narrows the VO₂ range considerably. An age formula is right on average and wrong for individuals by about 7 bpm.</small></label>
          </div>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 2</p>
          <h2>Heart-rate zones</h2>
          <div class="method-options">${ZONE_METHODS.map(([value, label, detail]) => `
            <label class="method-option ${draft.method === value ? "chosen" : ""}" data-method="${value}">
              <input type="radio" name="method" value="${value}" ${draft.method === value ? "checked" : ""}>
              <b>${escapeHtml(label)}</b><small>${escapeHtml(detail)}</small>
            </label>`).join("")}</div>
          <div class="zone-inputs" id="zone-inputs" ${MANUAL_METHODS.includes(draft.method) ? "" : "hidden"}>
            ${ZONE_NAMES.map((name) => `<label>${name.toUpperCase()}
              <span class="zone-pair">
                <input type="number" name="${name}_min" min="40" max="250" value="${draft.zones[name].minimum_bpm}">
                <input type="number" name="${name}_max" min="40" max="250" value="${draft.zones[name].maximum_bpm}">
              </span></label>`).join("")}
          </div>
          <div id="zone-preview" class="zone-preview-slot"></div>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 3</p>
          <h2>Comparison heart rate</h2>
          <p class="chart-note">Every "pace at X bpm" figure is measured here. Runs far from it have to be extrapolated, so the best choice is wherever your own history has the most evidence inside Z2.</p>
          <div id="comparison-recommendation" class="chart-note">Reading your history…</div>
          <div class="setup-fields">
            <label>Comparison heart rate<input type="number" name="target_hr" min="80" max="200" value="${draft.comparison_hr}" required></label>
          </div>
          <button type="button" id="use-recommended" disabled>Use recommended</button>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 4</p>
          <h2>What you are training for</h2>
          <p class="chart-note">Changes what the plan prioritises. It never overrides a health or load guardrail.</p>
          <div class="setup-fields">
            <label>Goal<select name="training_goal">${GOALS.map(([value, label]) => `<option value="${value}" ${settings.coaching.training_goal === value ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select>
            <small>A race goal needs a date and a target pace, which are set on the Settings page.</small></label>
          </div>
        </section>

        <section class="wide-card setup-block">
          <p class="eyebrow">Step 5</p>
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
    </section>`;

  wire(draft, settings);
}

function wire(draft, settings) {
  const form = view.querySelector("#setup-form");
  const el = (selector) => view.querySelector(selector);
  const field = (name) => form.elements[name];

  function readDraft() {
    draft.max_hr = Number(field("max_hr").value);
    draft.resting_hr = Number(field("resting_hr").value);
    draft.max_hr_source = field("max_hr_source").value;
    draft.comparison_hr = Number(field("target_hr").value);
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
  let sequence = 0;
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
          max_hr: draft.max_hr,
          resting_hr: draft.resting_hr,
          ...(MANUAL_METHODS.includes(draft.method) ? { boundaries: draft.zones } : {}),
        }),
      });
      // A slower earlier request must not overwrite a newer answer.
      if (ticket !== sequence) return;
      draft.zones = preview.zones;
      renderPreview(preview);
    } catch (error) {
      if (ticket !== sequence) return;
      slot.innerHTML = `<p class="chart-note error-text">${escapeHtml(error.message)}</p>`;
    }
  }

  function renderPreview(preview) {
    const rows = ZONE_NAMES.map((name) => {
      const zone = preview.zones[name];
      return `<tr class="${name === "z2" ? "zone-highlight" : ""}"><td>${name.toUpperCase()}</td><td>${zone.minimum_bpm}–${zone.maximum_bpm} bpm</td></tr>`;
    }).join("");
    const recommendation = preview.comparison_hr;
    el("#zone-preview").innerHTML = `<div class="zone-preview">
      <table class="zone-table"><tbody>${rows}</tbody></table>
      <div class="zone-effect">
        <p class="eyebrow">What these zones do to your comparison heart rate</p>
        <p><strong>${recommendation.recommended_bpm ?? "—"}${recommendation.recommended_bpm ? " bpm" : ""}</strong></p>
        <p class="chart-note">${escapeHtml(recommendation.rationale)}</p>
      </div>
    </div>`;
    updateRecommendation(recommendation);
  }

  let recommended = null;
  function updateRecommendation(recommendation) {
    recommended = recommendation.recommended_bpm ?? null;
    const target = el("#comparison-recommendation");
    if (target) target.textContent = recommendation.rationale;
    const button = el("#use-recommended");
    if (button) button.disabled = !recommended;
  }

  view.querySelectorAll('input[name="method"]').forEach((input) =>
    input.addEventListener("change", () => {
      draft.method = input.value;
      view.querySelectorAll(".method-option").forEach((option) =>
        option.classList.toggle("chosen", option.dataset.method === draft.method));
      el("#zone-inputs").hidden = !MANUAL_METHODS.includes(draft.method);
      runPreview();
    }),
  );

  ["max_hr", "resting_hr"].forEach((name) =>
    field(name).addEventListener("input", schedulePreview));
  el("#zone-inputs").addEventListener("input", schedulePreview);

  el("#use-recommended").addEventListener("click", () => {
    if (recommended) field("target_hr").value = recommended;
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
      // One request. Partial saves were how the page could end up showing a
      // comparison heart rate that no longer matched the stored zones.
      const updated = await api("/api/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          max_hr: draft.max_hr,
          resting_hr: draft.resting_hr,
          target_hr: draft.comparison_hr,
          zones: draft.zones,
          profile: settings.profile
            ? { ...settings.profile, max_hr_source: draft.max_hr_source }
            : undefined,
          coaching: { ...settings.coaching, training_goal: field("training_goal").value },
          historical_weather_enabled: field("historical_weather_enabled").checked,
          forecast_weather_enabled: field("forecast_weather_enabled").checked,
          weather_privacy_radius_km: Number(field("weather_privacy_radius_km").value),
          setup: { confirmed_steps: COVERED_STEPS, zone_method: draft.method },
        }),
      });
      primeSettings(updated);
      status.textContent = updated.recalculation.length ? "Saved, and your runs were re-analysed." : "Saved.";
      status.className = "form-status ok";
      // Refresh only the checklist. Re-rendering the whole page under someone
      // who just pressed Save throws away their scroll position for nothing.
      const refreshed = await api("/api/setup");
      el("#setup-checklist").innerHTML = checklistMarkup(refreshed);
    } catch (error) {
      status.textContent = error.message;
      status.className = "form-status error-text";
    } finally {
      button.disabled = false;
    }
  });

  runPreview();
}
