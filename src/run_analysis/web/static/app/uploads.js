// TCX drop target and the upload progress panel.

import { api } from "./api.js";
import { escapeHtml, titleCase } from "./format.js";
import { route } from "./router.js";

const uploadButton = document.querySelector("#upload-button");
const uploadInput = document.querySelector("#upload-input");
const uploadPanel = document.querySelector("#upload-panel");
const uploadClose = document.querySelector("#upload-close");
const uploadSummary = document.querySelector("#upload-summary");
const uploadStages = document.querySelector("#upload-stages");

export function registerUploads() {
  uploadButton.addEventListener("click", () => uploadInput.click());
  uploadClose.addEventListener("click", () => { uploadPanel.hidden = true; });
  async function processUploads(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    uploadPanel.hidden = false;
    uploadStages.replaceChildren();
    uploadSummary.textContent = `Adding ${files.length} file${files.length === 1 ? "" : "s"}…`;
    uploadButton.disabled = true;
    try {
      const result = await api("/api/uploads", { method: "POST", body: form });
      result.stages.forEach((stage) => {
        const item = document.createElement("li");
        item.dataset.state = stage.status;
        const stageNames = { save: "Save files", import: "Read activities", process: "Analyze runs", weather: "Add weather", model: "Update fitness", plan: "Update plan" };
        const stageStatus = { complete: "Done", deferred: "Skipped", failed: "Needs attention" };
        item.innerHTML = `<strong>${escapeHtml(stageNames[stage.name] ?? titleCase(stage.name))}</strong><span>${escapeHtml(stageStatus[stage.status] ?? titleCase(stage.status))}</span>`;
        item.title = stage.detail;
        uploadStages.append(item);
      });
      const accepted = result.files.filter((file) => file.status !== "failed").length;
      uploadSummary.textContent = accepted === result.files.length ? `${accepted} file${accepted === 1 ? "" : "s"} added.` : `${accepted} of ${result.files.length} files added.`;
      if (result.primary_activity_id) location.hash = `run/${result.primary_activity_id}`;
      else await route();
    } catch (error) {
      uploadSummary.textContent = `Upload failed: ${error.message}`;
    } finally {
      uploadButton.disabled = false;
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
