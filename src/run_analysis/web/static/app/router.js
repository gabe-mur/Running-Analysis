// Hash routing. Each route owns one view module; adding a screen means adding
// a module and one line here.

import { refreshTargetHr } from "./api.js";
import { view } from "./dom.js";
import { escapeHtml, titleCase } from "./format.js";
import { renderDashboard } from "./views/dashboard.js";
import { renderNextRun } from "./views/next-run.js";
import { renderProgress } from "./views/progress.js";
import { renderRunDetail } from "./views/run-detail.js";
import { renderRuns } from "./views/runs.js";
import { renderSettings } from "./views/settings.js";
import { renderSetup } from "./views/setup.js";

function renderPlaceholder(route) {
  view.innerHTML = `<section class="page narrow"><p class="eyebrow">In progress</p><h1>${titleCase(route)}</h1><p class="lede">The backend contract is ready. This page is being connected in the next implementation phase.</p></section>`;
}

export async function route() {
  const hash = location.hash.slice(1) || "dashboard";
  // Every view labels numbers with the comparison heart rate, so it is
  // refreshed once per navigation rather than by each view separately.
  await refreshTargetHr();
  document.querySelectorAll("nav a").forEach((link) => link.classList.toggle("active", hash === link.dataset.route || hash.startsWith(`${link.dataset.route}/`)));
  try {
    if (hash === "dashboard") await renderDashboard();
    else if (hash === "progress") await renderProgress();
    else if (hash === "runs") await renderRuns();
    else if (hash.startsWith("run/")) await renderRunDetail(Number(hash.split("/")[1]));
    else if (hash === "next-run") await renderNextRun();
    else if (hash === "settings") await renderSettings();
    else if (hash === "setup") await renderSetup();
    else renderPlaceholder(hash);
  } catch (error) {
    view.innerHTML = `<section class="page narrow"><p class="eyebrow error-text">Could not load</p><h1>Something needs attention.</h1><p>${escapeHtml(error.message)}</p></section>`;
  }
  view.focus({ preventScroll: true });
}
