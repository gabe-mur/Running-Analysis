// Render every view against a running instance so a missing import, a typo in
// a template literal, or a field the backend no longer returns fails here
// rather than silently in the browser.
//
// The ES-module split means an identifier can now be in scope in one file and
// undefined in another, which `node --check` cannot see. This loads the real
// module graph against a stub DOM and calls each view.
//
//   .venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
//     from run_analysis.web.app import create_app; import uvicorn; \
//     uvicorn.run(create_app('.'), port=8123, log_level='error')" &
//   node scripts/render_smoke.mjs src/run_analysis/web/static/app
const captured = [];
const stubEl = () => new Proxy(function(){}, {
  apply: () => stubEl(),
  get: (_t, k) => {
    if (k === "querySelector" || k === "closest" || k === "createElement") return stubEl;
    if (k === "querySelectorAll") return () => [];
    if (["addEventListener","focus","append","replaceChildren","click","preventDefault","reset"].includes(k)) return () => {};
    if (k === "classList") return { add(){}, remove(){}, toggle(){} };
    if (k === "dataset") return {};
    if (k === "files") return [];
    if (k === "value") return "";
    return stubEl();
  },
  set: (_t, k, v) => { if (k === "innerHTML") captured.push(String(v)); return true; },
});
globalThis.document = {
  querySelector: stubEl, querySelectorAll: () => [], addEventListener: () => {},
  createElement: stubEl, body: { classList: { add(){}, remove(){} } },
};
globalThis.window = { addEventListener: () => {}, location: { hash: "" } };
globalThis.location = { hash: "" };
globalThis.FormData = class { append(){} };
const BASE = "http://127.0.0.1:8123";
const realFetch = globalThis.fetch;
globalThis.fetch = (path, opts) => realFetch(BASE + path, opts);

const dir = process.argv[2];
const dashboard = await import(`${dir}/views/dashboard.js`);
const progress  = await import(`${dir}/views/progress.js`);
const runs      = await import(`${dir}/views/runs.js`);
const detail    = await import(`${dir}/views/run-detail.js`);
const next      = await import(`${dir}/views/next-run.js`);
const settings  = await import(`${dir}/views/settings.js`);
const setup     = await import(`${dir}/views/setup.js`);
const { refreshTargetHr } = await import(`${dir}/api.js`);
await refreshTargetHr();

const runList = await (await realFetch(`${BASE}/api/runs?limit=1`)).json();
const cases = [
  ["dashboard", () => dashboard.renderDashboard()],
  ["progress",  () => progress.renderProgress()],
  ["runs",      () => runs.renderRuns()],
  ["run-detail",() => detail.renderRunDetail(runList[0].activity_id)],
  ["next-run",  () => next.renderNextRun()],
  ["settings",  () => settings.renderSettings()],
  ["setup",     () => setup.renderSetup()],
];
// The colour vocabulary is split across two files: the views emit the class,
// the stylesheet gives it meaning. A rename that lands in one and not the
// other produces no error at all -- just uncoloured, un-laid-out markup. So
// every emitted state class is checked against a real rule in styles.css.
const css = await (await import("node:fs/promises")).readFile(
  new URL("../src/run_analysis/web/static/styles.css", import.meta.url), "utf8");
const STATEFUL = ["trend", "trend-arrow", "trend-status", "fitness-signal", "training-status"];
function unstyledStateClasses(html) {
  const missing = new Set();
  for (const base of STATEFUL) {
    const pattern = new RegExp(`class="${base} ([a-z-]+)"`, "g");
    for (const [, state] of html.matchAll(pattern)) {
      if (!css.includes(`.${base}.${state}`)) missing.add(`.${base}.${state}`);
    }
  }
  return [...missing];
}

let failed = 0;
for (const [name, run] of cases) {
  captured.length = 0;
  try {
    await run();
    const html = captured.join("");
    const bad = /undefined|\[object Object\]|NaN(?![a-zA-Z])/.exec(html);
    const unstyled = unstyledStateClasses(html);
    if (bad) { failed++; console.log(`WARN ${name}: rendered "${bad[0]}" near: ...${html.slice(Math.max(0,bad.index-70), bad.index+70)}...`); }
    else if (unstyled.length) { failed++; console.log(`WARN ${name}: emitted state classes with no rule in styles.css: ${unstyled.join(", ")}`); }
    else console.log(`OK   ${name} (${html.length} chars)`);
  } catch (e) { failed++; console.log(`FAIL ${name}: ${e.message}`); }
}
process.exit(failed ? 1 : 0);
