import { api, appSettings, invalidateSettings, primeSettings } from "../api.js";
import { view, loading } from "../dom.js";
import { durationSeconds, fitnessWindowLabel, pace, titleCase } from "../format.js";

export async function renderSettings() {
  loading("Settings");
  invalidateSettings();
  const [settings, setup] = await Promise.all([
    appSettings(),
    // Setup needs a database; Settings must still work without one.
    api("/api/setup")
      .then((state) => ({
        complete: state.complete,
        remaining: state.steps.filter((step) => !step.complete).length,
      }))
      .catch(() => ({ complete: true, remaining: 0 })),
  ]);
  const zoneInputs = Object.entries(settings.zones).map(([name, zone]) => `<div class="zone-input"><strong>${name.toUpperCase()}</strong><input type="number" name="${name}_min" value="${zone.minimum_bpm}"><span>to</span><input type="number" name="${name}_max" value="${zone.maximum_bpm}"><span>bpm</span></div>`).join("");
  const qualitySessionInputs = Object.entries(settings.coaching.quality_sessions).map(([name, enabled]) => `<label><input type="checkbox" name="quality_${name}" ${enabled ? "checked" : ""}>${titleCase(name)}</label>`).join("");
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
    long_run_progression_factor: "Long-run growth warning",
    high_load_ratio: "High-load warning",
    moderate_intensity_leakage_fraction: "Moderate-intensity warning",
    minimum_days_between_quality_sessions: "Minimum days between hard workouts",
    quality_recency_reference_days: "Quality-workout target spacing (days)",
    typical_rest_days_between_runs: "Typical rest days between runs",
    capacity_retention_half_life_days: "How gradually old capacity fades (days)",
    capacity_retention_grace_days: "Short-break grace period (days)",
    minimum_running_days_28d_for_quality: "Run days needed before hard workouts",
    long_run_recency_reference_days: "Long-run target spacing (days)",
    reduced_volume_factor: "Reduced-run distance factor",
  };
  view.innerHTML = `
    <section class="page settings-page">
      <form id="settings-form">
      <div class="page-heading"><div><p class="eyebrow">Stored locally</p><h1>Settings</h1><p>Everything here stays on this computer. Only weather lookups you turn on contact Open-Meteo.</p></div>
        <div class="settings-actions heading-actions"><button class="primary-button" type="submit">Save settings</button><span id="settings-status-top"></span></div>
      </div>
      <article class="wide-card setup-entry ${setup.complete ? "" : "pending"}">
        <div>
          <p class="eyebrow">Setup</p>
          <h2>${setup.complete ? "Everything is confirmed" : `${setup.remaining} settings still on defaults`}</h2>
          <p>${setup.complete
            ? "The settings every other number depends on have all been confirmed. Walk through them again any time."
            : "A guided walk through the handful of settings everything else is built on, with each answer's effect shown before you save."}</p>
        </div>
        <a class="primary-button" href="#setup">${setup.complete ? "Review setup" : "Finish setup"}</a>
      </article>
      <div class="settings-grid">
        <fieldset><legend>Heart rate</legend><label>Maximum HR<input type="number" name="max_hr" value="${settings.max_hr}"></label><label>Resting HR<input type="number" name="resting_hr" value="${settings.resting_hr}"></label><label>Comparison HR<input type="number" name="target_hr" value="${settings.target_hr}"></label><p>The app compares runs at this same heart rate.</p></fieldset>
        <fieldset><legend>Optional VO₂ estimate</legend><label>Birth date<input type="date" name="profile_birth_date" value="${settings.profile?.birth_date ?? ""}"></label><label>Sex used by the equation<select name="profile_sex"><option value="male" ${settings.profile?.sex === "male" ? "selected" : ""}>Male</option><option value="female" ${settings.profile?.sex === "female" ? "selected" : ""}>Female</option></select></label><label>Weight (lb)<input type="number" step="0.1" name="profile_weight_lb" value="${settings.profile?.weight_lb ?? ""}"></label><label>Height (in)<input type="number" step="0.1" name="profile_height_in" value="${settings.profile?.height_in ?? ""}"></label><p>Used only for the local formula check. It is not Garmin's estimate.</p></fieldset>
        <fieldset><legend>Running zones</legend>${zoneInputs}</fieldset>
        <fieldset><legend>Weather and privacy</legend><label>Location randomization radius (km)<input type="number" step="0.1" name="weather_privacy_radius_km" value="${settings.weather_privacy_radius_km}"></label><label><input type="checkbox" name="historical_weather_enabled" ${settings.historical_weather_enabled ? "checked" : ""}>Add weather to past runs</label><label><input type="checkbox" name="forecast_weather_enabled" ${settings.forecast_weather_enabled ? "checked" : ""}>Add forecasts to weekly plans</label><p>Weather requests send only the run date or planned time and a rounded, randomized approximate location. Raw routes never leave this computer.</p></fieldset>
        <fieldset><legend>Training goal</legend><label>Goal<select name="training_goal"><option value="general_fitness" ${settings.coaching.training_goal === "general_fitness" ? "selected" : ""}>General fitness</option><option value="5k" ${settings.coaching.training_goal === "5k" ? "selected" : ""}>5K</option><option value="10k" ${settings.coaching.training_goal === "10k" ? "selected" : ""}>10K</option><option value="half_marathon" ${settings.coaching.training_goal === "half_marathon" ? "selected" : ""}>Half marathon</option><option value="marathon" ${settings.coaching.training_goal === "marathon" ? "selected" : ""}>Marathon</option></select></label><label>Race date<input type="date" name="goal_date" value="${settings.coaching.goal_date ?? ""}"></label><label>Goal pace per mile<input type="text" name="goal_pace" inputmode="numeric" placeholder="9:00" value="${settings.coaching.goal_pace_min_mile ? pace(settings.coaching.goal_pace_min_mile).replace('/mi', '') : ""}"></label><p>Race goals use your latest 10 comparable runs to reject unrealistic dates or paces.</p></fieldset>
        <fieldset><legend>Quality sessions</legend>${qualitySessionInputs}<p>Disabled types will not be prescribed. At least one type must remain enabled.</p></fieldset>
        <fieldset><legend>Display</legend><label>Default progress timeframe<select name="default_fitness_window">${settings.available_fitness_windows.map((value) => `<option value="${value}" ${value === settings.default_fitness_window ? "selected" : ""}>${fitnessWindowLabel(value)}</option>`).join("")}</select></label></fieldset>
      </div>

      <details class="wide-card advanced-settings">
        <summary><span class="eyebrow">Advanced</span><b>Model and planner parameters</b></summary>
        <p class="chart-note">These are the tuning constants behind the analysis, not facts about you. The shipped values are the ones every published figure in this app was validated against — changing them changes your results, and nothing will warn you that they no longer match the documentation.</p>
        <div class="settings-grid">
          <fieldset><legend>Comparison conditions</legend><label>Temperature (°F)<input type="number" step="0.1" name="reference_temperature_f" value="${settings.reference_temperature_f}"></label><label>Dew point (°F)<input type="number" step="0.1" name="reference_dewpoint_f" value="${settings.reference_dewpoint_f}"></label><label>Wind (mph)<input type="number" step="0.1" name="reference_wind_mph" value="${settings.reference_wind_mph}"></label><label>Grade (%)<input type="number" step="0.1" name="reference_grade_percent" value="${settings.reference_grade_percent}"></label><label>Point in each run (minute)<input type="number" step="0.5" name="reference_within_run_minutes" value="${settings.reference_within_run_minutes}"></label><p>Every run is adjusted to these conditions so runs on different days can be compared.</p></fieldset>
          <fieldset><legend>Movement detection</legend>${Object.entries(settings.moving_time).map(([name, value]) => `<label>${movingLabels[name] ?? titleCase(name)}<input type="number" step="0.01" name="moving_${name}" value="${value}"></label>`).join("")}<p>Decides what counts as stopped. Changing these re-derives every run.</p></fieldset>
          <fieldset><legend>Planning rules</legend>${Object.entries(settings.coaching).filter(([name]) => !["training_goal", "goal_date", "goal_pace_min_mile", "quality_sessions"].includes(name)).map(([name, value]) => `<label>${coachingLabels[name] ?? titleCase(name)}<input type="number" step="0.01" name="coaching_${name}" value="${value}"></label>`).join("")}<p>Thresholds the weekly plan is built from.</p></fieldset>
        </div>
        <div class="settings-actions reset-row">
          <button type="button" id="reset-advanced">Reset to recommended defaults</button>
          <small>Restores only the values in this section. Your heart rates, zones, profile, goal, and weather choices are left alone.</small>
          <span id="reset-status" class="form-status"></span>
        </div>
      </details>

      <div class="settings-actions"><button class="primary-button" type="submit">Save settings</button><span id="settings-status"></span></div></form>
    </section>`;
  const form = view.querySelector("#settings-form");

  // Race fields are meaningless for general fitness, so they empty and grey out
  // as soon as the goal changes instead of sitting there looking authoritative.
  const goalSelect = form.querySelector('select[name="training_goal"]');
  const raceFields = [form.querySelector('input[name="goal_date"]'), form.querySelector('input[name="goal_pace"]')].filter(Boolean);
  function syncRaceFields({ clear } = { clear: false }) {
    if (!goalSelect) return;
    const general = goalSelect.value === "general_fitness";
    raceFields.forEach((input) => {
      if (!input) return;
      if (general && clear) input.value = "";
      input.disabled = general;
      input.closest("label")?.classList.toggle("disabled-field", general);
    });
  }
  goalSelect?.addEventListener("change", () => syncRaceFields({ clear: true }));
  syncRaceFields();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form); const numeric = (name) => Number(data.get(name));
    const payload = {
      max_hr: numeric("max_hr"), resting_hr: numeric("resting_hr"), target_hr: numeric("target_hr"),
      zones: Object.fromEntries(["z1","z2","z3","z4","z5"].map((name) => [name, { minimum_bpm: numeric(`${name}_min`), maximum_bpm: numeric(`${name}_max`) }])),
      reference_temperature_f: numeric("reference_temperature_f"), reference_dewpoint_f: numeric("reference_dewpoint_f"), reference_wind_mph: numeric("reference_wind_mph"), reference_grade_percent: numeric("reference_grade_percent"), reference_within_run_minutes: numeric("reference_within_run_minutes"), weather_privacy_radius_km: numeric("weather_privacy_radius_km"), historical_weather_enabled: data.get("historical_weather_enabled") === "on", forecast_weather_enabled: data.get("forecast_weather_enabled") === "on", default_fitness_window: numeric("default_fitness_window"),
      moving_time: Object.fromEntries(Object.keys(settings.moving_time).map((name) => [name, numeric(`moving_${name}`)])),
      coaching: {
        ...Object.fromEntries(Object.keys(settings.coaching).filter((name) => !["training_goal", "goal_date", "goal_pace_min_mile", "quality_sessions"].includes(name)).map((name) => [name, numeric(`coaching_${name}`)])),
        training_goal: data.get("training_goal"),
        // A race date and pace only mean something for a race. Keeping them
        // when the goal is general fitness leaves a goal the app half-believes.
        goal_date: data.get("training_goal") === "general_fitness" ? null : data.get("goal_date") || null,
        goal_pace_min_mile:
          data.get("training_goal") === "general_fitness" || !durationSeconds(data.get("goal_pace"))
            ? null
            : durationSeconds(data.get("goal_pace")) / 60,
        quality_sessions: Object.fromEntries(Object.keys(settings.coaching.quality_sessions).map((name) => [name, data.get(`quality_${name}`) === "on"])),
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
    const profileHasAnyValue = Boolean(profileBirthDate || profileWeight || profileHeight);
    if (payload.coaching.training_goal !== "general_fitness" && (!payload.coaching.goal_date || !payload.coaching.goal_pace_min_mile)) {
      status.textContent = "Choose both a race date and a goal pace in mm:ss per mile.";
      return;
    }
    if (profileHasAnyValue && !(profileBirthDate && profileWeight && profileHeight)) {
      status.textContent = "Complete birth date, weight, and height together, or leave the optional VO₂ profile blank.";
      return;
    }
    if (profileHasAnyValue) {
      // Spread the stored profile first: this form does not show
      // max_hr_source, and rebuilding the object without it silently reset a
      // measured maximum to "estimated" and widened the VO2 interval.
      payload.profile = {
        ...(settings.profile ?? {}),
        birth_date: profileBirthDate,
        sex: data.get("profile_sex"),
        weight_lb: Number(profileWeight),
        height_in: Number(profileHeight),
      };
    }
    status.textContent = "Validating and recalculating…"; button.disabled = true;
    try {
      const updated = await api("/api/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      // A saved comparison HR must not leave stale "at N bpm" labels sitting
      // over numbers that were just recomputed at the new one.
      primeSettings(updated);
      status.textContent = updated.recalculation.length ? "Saved and updated the analysis." : "Saved.";
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
