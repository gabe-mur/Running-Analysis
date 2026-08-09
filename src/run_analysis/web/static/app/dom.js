// The one element every view renders into, and the shared loading state.

import { escapeHtml } from "./format.js";

export const view = document.querySelector("#app-view");

export function loading(title = "Loading analysis") {
  view.innerHTML = `<section class="page narrow"><p class="eyebrow">Local analysis</p><h1>${escapeHtml(title)}</h1><div class="loading-line"></div></section>`;
}
