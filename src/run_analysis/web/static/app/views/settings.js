import { api, appSettings, invalidateSettings, primeSettings } from "../api.js";
import { view, loading } from "../dom.js";
import { escapeHtml, fitnessWindowLabel, titleCase } from "../format.js";
import { bindIntervalsSettings, intervalsSettingsCard } from "../intervals-settings.js";
import { plannerSettingsCard, readPlannerPreferences } from '../planner-settings.js';
import { bindRacePreparation, racePreparationCard, readRacePreparation } from '../race-settings.js';
import { bindQualityFocus, qualitySessionsCard } from '../quality-session-settings.js';
import { profileFields, heartRateFields } from '../setup-fields.js';
import { bindSettingsZones } from '../zone-controls.js';
import { ageFromBirthDate } from '../setup-baseline.js';

export async function renderSettings() {
  loading("Settings");
  invalidateSettings();
  const [settings, setup] = await Promise.all([
    appSettings(),
    // Setup needs a database; Settings must still work without one.
    api("/api/setup")
      .then((state) => ({
        complete: state.complete,
        total: state.steps.length,
        remaining: state.steps.filter((step) => !step.complete).length,
      }))
      .catch(() => ({ complete: true, remaining: 0, total: null })),
  ]);
  const adaptive = settings.planner !== null && settings.planner !== undefined;
  const planner = settings.planner;
  const strength = settings.coaching.strength_training;
  const strengthInputs = Object.entries(strength.muscle_groups).map(([name, groups]) => `<label>${titleCase(name)} groups<input type="text" name="strength_${name}" value="${escapeHtml(groups.join(", "))}" placeholder="Leave blank for default groups"></label>`).join("");
  const numberLabel = (value, decimals = 1) => value == null ? "Not available" : Number(value).toFixed(decimals);
  const plannerCard = plannerSettingsCard(planner);
  const movingLabels = {
    minimum_running_speed_mps: "Slowest speed counted as moving (m/s)",
    stopped_speed_mps: "Stopped-speed threshold (m/s)",
    gps_stopped_speed_mps: "GPS stopped-speed threshold (m/s)",
    stopped_distance_meters: "Maximum movement while stopped (m)",
    maximum_interval_seconds: "Largest allowed sensor gap (seconds)",
    minimum_stop_seconds: "Minimum stop length (seconds)",
    maximum_plausible_speed_mps: "Maximum believable speed (m/s)",
  };
  const coachingLabels = {
    general_fitness_progression_fraction: "General-fitness productive build",
    long_run_progression_factor: "Long-run single-session growth factor",
    long_run_target_progression_fraction: "Ordinary long-run target increase",
    high_load_ratio: "Training-status high-load warning",
    moderate_intensity_leakage_fraction: "Moderate-intensity warning",
    quality_recency_reference_days: "Quality-workout target spacing (days)",
    typical_rest_days_between_runs: "Legacy spacing reference (rest days)",
    continuous_fatigue_half_life_days: "Modeled accumulated-load half-life (days)",
    capacity_retention_half_life_days: "How gradually old capacity fades (days)",
    long_run_retention_half_life_days: "How gradually long-run durability fades (days)",
    long_run_retention_grace_days: "Days before long-run durability starts fading",
    capacity_retention_grace_days: "Short-break grace period (days)",
    minimum_running_days_28d_for_quality: "Run days needed before hard workouts",
    long_run_recency_reference_days: "Long-run target spacing (days)",
    reduced_volume_factor: "Reduced-run distance factor",
  };
  view.innerHTML = `
    <section class="page settings-page">
      <form id="settings-form">
      <div class="page-heading"><div><p class="eyebrow">Stored locally</p><h1>Settings</h1></div>
        <div class="settings-actions heading-actions"><button class="primary-button" type="submit">Save settings</button><span id="settings-status-top"></span></div>
      </div>
      <article class="wide-card setup-entry ${setup.complete ? "" : "pending"}">
        <div>
          <p class="eyebrow">Setup</p>
          <h2>${setup.complete ? setup.total ? `You’re set up · ${setup.total} of ${setup.total} essential settings confirmed` : "Review your setup" : `${setup.remaining} setup ${setup.remaining === 1 ? "step" : "steps"} remaining`}</h2>
          ${setup.complete ? "<p>Change them at any time.</p>" : ""}
        </div>
        <a class="primary-button" href="#setup">${setup.complete ? "Review setup" : "Finish setup"}</a>
      </article>

      <section class="settings-section" aria-labelledby="settings-training">
        <h2 id="settings-training">Training</h2>
        <div class="settings-grid">
          ${racePreparationCard(settings.race_preparation,planner)}
          ${plannerCard}
          ${qualitySessionsCard(settings.coaching.quality_sessions, settings.coaching.quality_focus)}
          <fieldset><legend>Strength</legend>
            <label class="settings-check"><input type="checkbox" name="strength_enabled" ${strength.enabled ? "checked" : ""}>Include strength suggestions</label>
            <label>Leg day every (days)<input type="number" name="strength_leg_days" min="1" max="60" value="${strength.leg_frequency_days}"></label>
            <label>Push/pull every (days)<input type="number" name="strength_upper_days" min="1" max="60" value="${strength.push_pull_frequency_days}"></label>
            <details class="settings-inline-details"><summary>Muscle groups</summary>
              <div class="settings-detail-fields">${strengthInputs}</div>
              <p>Separate groups with commas. Leave blank to use defaults.</p>
            </details>
            <p>Follows your run schedule. Lifting load is not included in running fatigue.</p>
          </fieldset>
        </div>
      </section>

      <section class="settings-section" aria-labelledby="settings-heart-rate">
        <h2 id="settings-heart-rate">Heart rate &amp; profile</h2>
        <div class="settings-grid">
          <fieldset class="settings-span"><legend>About you</legend>
            <div class="settings-profile-fields">${profileFields(settings.profile)}</div>
            <p>Used for your fitness estimate and an age-based maximum HR estimate if needed.</p>
          </fieldset>
          <fieldset class="settings-span"><legend>Heart-rate zones</legend>
            ${heartRateFields(settings, {
              estimateId: 'settings-estimate-max', estimateStatusId: 'settings-max-estimate-status'})}
            <small id="settings-zone-status" role="status"></small>
          </fieldset>
        </div>
      </section>

      <section class="settings-section" aria-labelledby="settings-connections">
        <h2 id="settings-connections">Sync &amp; weather</h2>
        <div class="settings-grid">
          ${intervalsSettingsCard()}
          <fieldset><legend>Weather</legend>
            <label class="settings-check"><input type="checkbox" name="historical_weather_enabled" ${settings.historical_weather_enabled ? "checked" : ""}>Add weather to past runs</label>
            <label class="settings-check"><input type="checkbox" name="forecast_weather_enabled" ${settings.forecast_weather_enabled ? "checked" : ""}>Show forecasts in the plan</label>
            <label>Location randomization (km)<input type="number" step="0.1" name="weather_privacy_radius_km" value="${settings.weather_privacy_radius_km}"></label>
            <p>Open-Meteo receives the run date or planned time and a rounded, randomized location—not your route.</p>
          </fieldset>
        </div>
      </section>

      <section class="settings-section" aria-labelledby="settings-display">
        <h2 id="settings-display">Display</h2>
        <div class="settings-grid">
          <fieldset class="settings-span"><legend>Progress</legend><label>Default timeframe<select name="default_fitness_window">${settings.available_fitness_windows.map((value) => `<option value="${value}" ${value === settings.default_fitness_window ? "selected" : ""}>${fitnessWindowLabel(value)}</option>`).join("")}</select></label></fieldset>
        </div>
      </section>

      <details class="wide-card advanced-settings">
        <summary><b>Advanced</b><span>Analysis &amp; model details</span></summary>
        <p class="chart-note">Technical settings that affect analysis and planning.</p>
        <div class="settings-grid">
          <fieldset><legend>Pace reference conditions</legend><label>Temperature (°F)<input type="number" step="0.1" name="reference_temperature_f" value="${settings.reference_temperature_f}"></label><label>Dew point (°F)<input type="number" step="0.1" name="reference_dewpoint_f" value="${settings.reference_dewpoint_f}"></label><label>Wind (mph)<input type="number" step="0.1" name="reference_wind_mph" value="${settings.reference_wind_mph}"></label><label>Grade (%)<input type="number" step="0.1" name="reference_grade_percent" value="${settings.reference_grade_percent}"></label><label>Time into run (minutes)<input type="number" step="0.5" name="reference_within_run_minutes" value="${settings.reference_within_run_minutes}"></label><p>Reference conditions for adjusted pace.</p></fieldset>
          <fieldset><legend>Movement detection</legend>${Object.entries(settings.moving_time).map(([name, value]) => `<label>${movingLabels[name] ?? titleCase(name)}<input type="number" step="0.01" name="moving_${name}" value="${value}" ${adaptive ? "disabled" : ""}></label>`).join("")}<p>${adaptive ? "Locked for the active plan." : "Changing these recalculates moving time."}</p></fieldset>
          <fieldset class="settings-span"><legend>${adaptive ? "Planner model" : "Planner parameters"}</legend>${adaptive ? `
            <dl class="settings-model-values">
              <div><dt>Load decay (e-folding days)</dt><dd>${numberLabel(planner.demand_memory_days)}</dd></div>
              <div><dt>Minimum run (minutes)</dt><dd>${numberLabel(planner.minimum_running_minutes)}</dd></div>
              <div><dt>Short-run penalty ends at (minutes)</dt><dd>${numberLabel(planner.penalty_free_running_minutes)}</dd></div>
              <div><dt>Load / delivery tradeoff</dt><dd>${numberLabel(planner.demand_delivery_exchange, 3)}</dd></div>
              <div><dt>Frequency preference weight</dt><dd>${numberLabel(planner.frequency_preference_fraction * 100)}%</dd></div>
            </dl>
            <p>Model assumptions—not measured recovery or proven training limits.</p>
            <details class="settings-inline-details"><summary>How these values are used</summary>
              <p>Load decay is not a recovery countdown. The frequency weight applies at the range edges, relative to the fixed reference fatigue. Older planner settings are retained but inactive.</p>
            </details>` : `${Object.entries(settings.coaching).filter(([, value]) => typeof value === "number").map(([name, value]) => `<label>${coachingLabels[name] ?? titleCase(name)}<input type="number" step="0.01" name="coaching_${name}" value="${value}"></label>`).join("")}<p>Spacing is advisory. Progression allows growth; it does not require an increase after every run.</p>`}</fieldset>
        </div>
        <div class="settings-actions reset-row">
          <button type="button" id="reset-advanced" ${adaptive ? "disabled" : ""}>Reset to shipped defaults</button>
          <small>Resets this section only—not your personal settings.</small>
          <span id="reset-status" class="form-status"></span>
        </div>
      </details>

      <div class="settings-actions"><button class="primary-button" type="submit">Save settings</button><span id="settings-status"></span></div></form>
    </section>`;
  const form = view.querySelector("#settings-form");
  const zoneControls = bindSettingsZones(form, settings, api);
  let maxHrSource = settings.profile?.max_hr_source ?? 'estimated';
  form.querySelector('#settings-estimate-max').addEventListener('click', async event => {
    const status = form.querySelector('#settings-max-estimate-status');
    const birthDate = form.querySelector('[name="profile_birth_date"]').value;
    const age = ageFromBirthDate(birthDate);
    if (age === null) {status.textContent = 'Enter a valid birth date first.'; return;}
    event.currentTarget.disabled = true;
    try {
      const estimate = await api(`/api/setup/max-hr?age_years=${age}`);
      if (form.querySelector('[name="profile_birth_date"]').value !== birthDate
          || form.querySelector('[name="method"]:checked')?.value === 'device') return;
      const input = form.querySelector('[name="max_hr"]');
      input.value = estimate.estimated_max_hr;
      maxHrSource = 'estimated';
      form.querySelector('[name="max_hr_source"]').value = maxHrSource;
      input.dispatchEvent(new Event('input', {bubbles: true}));
      status.textContent = 'Age estimate—not a measured maximum.';
    } catch (error) {status.textContent = error.message;}
    finally {form.querySelector('#settings-estimate-max').disabled = false;}
  });
  bindRacePreparation(form,api,{planner});
  bindQualityFocus(form);
  // Separate connection controls never run the settings recalculation pipeline.
  void bindIntervalsSettings(form);
  const savePreferences = async (event) => {
    const status = form.querySelector("#planner-preferences-status");
    const button = event.currentTarget;
    button.disabled = true;
    status.textContent = "Saving…";
    try {
      const updated = await api("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ planner_preferences: {
        ...readPlannerPreferences(form,planner),
        expected_revision: planner.preferences_revision,
      } }) });
      Object.assign(planner, updated.planner);
      primeSettings(updated);
      status.textContent = updated.planner.preferences_pending ? "Saved for the next plan update." : "Saved.";
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  };
  form.querySelector("#save-planner-preferences")?.addEventListener("click",event=>savePreferences(event));
  if (adaptive) {
    form.querySelector(".reset-row small").textContent = "Reset is unavailable for the active planner.";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form); const numeric = (name) => Number(data.get(name));
    const payload = {
      max_hr: numeric("max_hr"), resting_hr: numeric("resting_hr"),
      zones: Object.fromEntries(["z1","z2","z3","z4","z5"].map((name) => [name, { minimum_bpm: numeric(`${name}_min`), maximum_bpm: numeric(`${name}_max`) }])),
      reference_temperature_f: numeric("reference_temperature_f"), reference_dewpoint_f: numeric("reference_dewpoint_f"), reference_wind_mph: numeric("reference_wind_mph"), reference_grade_percent: numeric("reference_grade_percent"), reference_within_run_minutes: numeric("reference_within_run_minutes"), weather_privacy_radius_km: numeric("weather_privacy_radius_km"), historical_weather_enabled: data.get("historical_weather_enabled") === "on", forecast_weather_enabled: data.get("forecast_weather_enabled") === "on", default_fitness_window: numeric("default_fitness_window"),
      moving_time: adaptive ? settings.moving_time : Object.fromEntries(Object.keys(settings.moving_time).map((name) => [name, numeric(`moving_${name}`)])),
      race_preparation: readRacePreparation(form),
      coaching: {
        ...settings.coaching,
        quality_focus: data.get('quality_focus') ?? 'variety',
        ...Object.fromEntries(Object.entries(settings.coaching).filter(([, value]) => typeof value === "number").map(([name, value]) => [name, adaptive ? value : numeric(`coaching_${name}`)])),
        quality_sessions: Object.fromEntries(Object.keys(settings.coaching.quality_sessions).map((name) => [name, data.get(`quality_${name}`) === "on"])),
        strength_training: {
          enabled: data.get("strength_enabled") === "on",
          leg_frequency_days: numeric("strength_leg_days"),
          push_pull_frequency_days: numeric("strength_upper_days"),
          muscle_groups: Object.fromEntries(Object.keys(strength.muscle_groups).map((name) => [name, String(data.get(`strength_${name}`) ?? "").split(",").map((s) => s.trim()).filter(Boolean)])),
        },
      },
    };
    const statusNodes = [...form.querySelectorAll("#settings-status, #settings-status-top")];
    const status = {
      set textContent(value) { statusNodes.forEach((node) => { node.textContent = value; }); },
    };
    const buttons = [...form.querySelectorAll("button[type=submit]")];
    const button = { set disabled(value) { buttons.forEach((node) => { node.disabled = value; }); } };
    const profileBirthDate = data.get("profile_birth_date");
    const profileWeight = data.get("profile_weight_lb");
    const profileHeight = data.get("profile_height_in");
    if (!(profileBirthDate && profileWeight && profileHeight && data.get('profile_sex'))
        || ageFromBirthDate(profileBirthDate) === null) {
      status.textContent = "Complete your birth date, sex, weight and height in About you.";
      return;
    }
    {
      // Preserve profile facts; only the selected source owns HR provenance.
      payload.profile = {
        ...(settings.profile ?? {}),
        birth_date: profileBirthDate,
        sex: data.get("profile_sex"),
        weight_lb: Number(profileWeight),
        height_in: Number(profileHeight),
        max_hr_source: data.get('max_hr_source') ?? maxHrSource,
      };
    }
    status.textContent = "Validating and recalculating…"; button.disabled = true;
    try {
      Object.assign(payload, await zoneControls.calibration());
      payload.setup.confirmed_steps = [...new Set([...(settings.setup?.confirmed_steps ?? []),
        'profile', 'heart_rate', 'zones'])];
      const updated = await api("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      // A saved comparison HR must not leave stale "at N bpm" labels sitting
      // over numbers that were just recomputed at the new one.
      primeSettings(updated);
      zoneControls.saved(updated);
      const failed = updated.recalculation.filter((step) => step.status === "failed");
      status.textContent = failed.length ? `Saved; ${failed.map((step) => `${step.name}: ${step.detail}`).join("; ")}` : updated.recalculation.some((step) => step.name === 'race_preparation') ? 'Draft saved. Use Apply training goal to update the plan.' : "Saved.";
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  });

  view.querySelector("#reset-advanced")?.addEventListener("click", async (event) => {
    const status = view.querySelector("#reset-status");
    // Destructive-ish and one click away, so it asks. It cannot touch personal
    // settings, and the confirmation says so rather than being vague.
    if (!window.confirm("Restore the model and planner parameters to their shipped values?\n\nYour heart rates, zones, profile, goal, and weather choices are not affected.")) return;
    event.currentTarget.disabled = true;
    status.textContent = "Restoring and recalculating…";
    status.className = "form-status";
    try {
      const updated = await api("/api/settings/reset-advanced", { method: "POST" });
      primeSettings(updated);
      status.textContent = "Restored.";
      status.className = "form-status ok";
      await renderSettings();
    } catch (error) {
      status.textContent = error.message;
      status.className = "form-status error-text";
    } finally {
      const button = view.querySelector("#reset-advanced");
      if (button) button.disabled = false;
    }
  });
}
