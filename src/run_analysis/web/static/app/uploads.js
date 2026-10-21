// Activity-file drop target and the upload progress panel.

import { api } from "./api.js";
import { escapeHtml, titleCase } from "./format.js";
import { route } from "./router.js";
import { bindSync } from './sync.js';
import { view } from './dom.js';
import { refreshAfterTransfer } from './transfer-refresh.js';

const uploadButton = document.querySelector("#upload-button");
const uploadInput = document.querySelector("#upload-input");
const uploadPanel = document.querySelector("#upload-panel");
const uploadClose = document.querySelector("#upload-close");
const uploadSummary = document.querySelector("#upload-summary");
const uploadStages = document.querySelector("#upload-stages");
const syncButton = document.querySelector('#sync-button');
let busy = false;

function begin(label, title) {
  if (busy) return false;
  busy = true;
  uploadButton.disabled = syncButton.disabled = uploadInput.disabled = true;
  document.querySelector('#transfer-label').textContent = label;
  document.querySelector('#transfer-title').textContent = title;
  uploadPanel.hidden = false;
  uploadStages.replaceChildren();
  return true;
}

function end() {
  busy = false;
  uploadButton.disabled = syncButton.disabled = uploadInput.disabled = false;
}

function showStages(stages) {
  stages.forEach((stage) => {
    const item = document.createElement('li');
    item.dataset.state = stage.status;
    const names = { save: 'Save files', import: 'Read activities', process: 'Analyze runs', weather: 'Add weather', model: 'Update fitness', plan: 'Update plan', schedule: 'Update plan', download: 'Download run' };
    const statuses = { complete: 'Done', deferred: 'Skipped', skipped: 'Skipped', failed: 'Needs attention' };
    if (stage.name === 'schedule' && stage.status === 'deferred') statuses.deferred = 'On opening Weekly Plan';
    item.innerHTML = `<strong>${escapeHtml(names[stage.name] ?? titleCase(stage.name))}</strong><span>${escapeHtml(statuses[stage.status] ?? titleCase(stage.status))}</span>`;
    item.title = stage.detail;
    if (stage.status === 'failed') {
      const detail = document.createElement('span');
      detail.textContent = stage.detail;
      item.append(detail);
    }
    uploadStages.append(item);
  });
}

export function registerUploads() {
  bindSync({ button: syncButton, begin, end, summary: uploadSummary, showStages,
    refresh: () => refreshAfterTransfer(route, view) });
  uploadButton.addEventListener("click", () => uploadInput.click());
  uploadClose.addEventListener("click", () => { uploadPanel.hidden = true; });
  async function processUploads(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    if (!begin('Upload', 'Adding your runs')) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    uploadSummary.textContent = `Adding ${files.length} file${files.length === 1 ? "" : "s"}…`;
    try {
      const result = await api("/api/uploads", { method: "POST", body: form });
      showStages(result.stages);
      const accepted = result.files.filter((file) => file.status !== "failed").length;
      uploadSummary.textContent = accepted === result.files.length ? `${accepted} file${accepted === 1 ? "" : "s"} added.` : `${accepted} of ${result.files.length} files added.`;
      const warnings = [...new Set(result.files.flatMap(file => [
        ...(file.warnings ?? []),
        ...(file.status === 'failed' && file.error ? [`${file.filename}: ${file.error}`] : []),
      ])
        .filter(note => note !== 'Parser reported warnings'))];
      if (warnings.length) uploadSummary.textContent += ` ${warnings.join(' ')}`;
      const changed = result.stages.some(stage => stage.name === 'process' && stage.status !== 'skipped');
      if (result.primary_activity_id) {
        const target = `#run/${result.primary_activity_id}`;
        if (location.hash !== target) location.hash = target;
        else if (changed) await route();
      } else await refreshAfterTransfer(route, view, location.hash, changed);
    } catch (error) {
      uploadSummary.textContent = `Upload failed: ${error.message}`;
    } finally {
      end();
      uploadInput.value = "";
    }
  }
  uploadInput.addEventListener("change", () => processUploads(uploadInput.files));
  document.addEventListener("dragover", (event) => {
    if ([...event.dataTransfer.types].includes("Files")) {
      event.preventDefault(); document.body.classList.add("dragging");
    }
  });
  document.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) document.body.classList.remove("dragging");
  });
  document.addEventListener("drop", (event) => {
    event.preventDefault(); document.body.classList.remove("dragging"); processUploads(event.dataTransfer.files);
  });
}
