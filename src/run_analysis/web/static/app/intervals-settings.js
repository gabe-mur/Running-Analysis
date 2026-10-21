import { api } from "./api.js";

export function intervalsSettingsCard() {
  return `<fieldset><legend>Intervals.icu connection</legend>
    <p>Import runs with downloadable original FIT files from Intervals.icu.</p>
    <p><a href="#setup">Device connection guide →</a></p>
    <label>Personal API key<input type="password" id="intervals-api-key" autocomplete="new-password" spellcheck="false" autocapitalize="none" maxlength="512" placeholder="Loading connection status…" disabled></label>
    <div class="settings-card-actions">
      <button type="button" id="save-intervals-key" disabled>Save API key</button>
      <button type="button" id="check-intervals-connection" disabled>Check connection</button>
      <button type="button" id="remove-intervals-key" disabled>Remove key</button>
    </div>
    <p id="intervals-status" class="form-status" role="status" aria-live="polite">Loading connection status…</p>
    <p>Use the sync button beside Upload Runs. Automatic imports are not enabled.</p>
    <details class="settings-inline-details"><summary>Key storage &amp; sync details</summary>
      <p>The key is stored in <code>data/credentials/intervals.json</code>, outside Git and browser storage. Only your Mac user can read it; it is not encrypted.</p>
      <p>Connection checks do not import runs. Sync downloads original FITs from the last 30 days.</p>
    </details>
  </fieldset>`;
}

export async function bindIntervalsSettings(form) {
  const input = form.querySelector("#intervals-api-key");
  const status = form.querySelector("#intervals-status");
  const save = form.querySelector("#save-intervals-key");
  const check = form.querySelector("#check-intervals-connection");
  const remove = form.querySelector("#remove-intervals-key");
  let configured = false, available = false, busy = true;
  const controls = () => {
    input.disabled = busy || !available;
    save.disabled = busy || !available || !input.value.trim();
    check.disabled = remove.disabled = busy || !available || !configured;
  };
  const update = (data) => {
    configured = data.configured;
    available = data.available;
    input.placeholder = configured ? "Key saved — enter only to replace" : "Paste your Intervals.icu API key";
    status.textContent = !available ? data.detail : configured
      ? "API key saved."
      : "No API key saved.";
  };
  const headers = { "X-Run-Analysis-Local": "1", "Content-Type": "application/json" };
  const act = async (operation) => {
    busy = true; controls();
    try { await operation(); }
    catch (error) { status.textContent = error.message; }
    finally { busy = false; controls(); }
  };
  input.addEventListener("input", controls);
  // Enter inside the secret field must not submit/recalculate general settings.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); if (!save.disabled) save.click(); }
  });
  save.addEventListener("click", () => act(async () => {
    status.textContent = "Saving the local API key…";
    const body = JSON.stringify({ api_key: input.value.trim() });
    input.value = ""; // Never refill/retrieve the secret through the UI.
    update(await api("/api/connections/intervals", { method: "PUT", headers, body }));
  }));
  check.addEventListener("click", () => act(async () => {
    status.textContent = "Checking Intervals.icu…";
    const result = await api("/api/connections/intervals/check", { method: "POST", headers });
    status.textContent = result.recent_activity_count
      ? `Connected. ${result.recent_activity_count} recent activities available; none imported.`
      : "Connected. No recent activities available yet—history sync may still be pending. Nothing imported.";
  }));
  remove.addEventListener("click", () => {
    if (!window.confirm("Delete the local Intervals.icu API key file? Your imported runs will stay untouched. You can save the key again later.")) return;
    act(async () => {
      input.value = "";
      update(await api("/api/connections/intervals", { method: "DELETE", headers }));
    });
  });
  try { update(await api("/api/connections/intervals")); }
  catch (error) { status.textContent = `Connection settings unavailable: ${error.message}`; }
  finally { busy = false; controls(); }
}
