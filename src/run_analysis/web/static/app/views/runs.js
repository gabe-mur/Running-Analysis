import { api, targetHrLabel } from "../api.js";
import { view, loading } from "../dom.js";
import {
  dataQualityLabel, dateLabel, escapeHtml, gpsQualityLabel, number, pace, titleCase,
} from "../format.js";

let runFilters = { flag: "", sort_by: "date", sort_order: "desc", date_from: "", date_to: "" };
//: How many runs a page adds. Kept as a page size rather than a growing
//: limit, because "load more" now fetches only the next page and appends it.
const RUN_PAGE_SIZE = 100;

function runQuery(offset) {
  const query = new URLSearchParams({
    limit: String(RUN_PAGE_SIZE),
    offset: String(offset),
    sort_by: runFilters.sort_by,
    sort_order: runFilters.sort_order,
  });
  Object.entries(runFilters).forEach(([key, value]) => {
    if (value && !["sort_by", "sort_order"].includes(key)) query.set(key, value);
  });
  return query;
}


function runRow(run) {
  return `
    <a class="run-row" href="#run/${run.activity_id}">
      <span><strong>${dateLabel(run.start_time)}</strong><small>${titleCase(run.workout_type)}</small></span>
      <span><strong>${number(run.distance_miles)} mi</strong><small>${number(run.moving_minutes, 0)} min</small></span>
      <span><strong>${pace(run.moving_pace_min_mile)}</strong><small>${number(run.average_hr_bpm, 0)} avg · ${number(run.maximum_hr_bpm, 0)} max</small></span>
      <span><strong>${run.fitness_observation ? run.fitness_observation.standardized_pace_at_target_hr.display : "—"}</strong><small>adjusted at ${targetHrLabel()}</small></span>
      <span><strong>${Number.isFinite(run.temperature_f) ? `${number(run.temperature_f, 0)}°F` : "—"}</strong><small>${gpsQualityLabel(run.gps_quality)}</small></span>
      <span><strong>${escapeHtml(run.assessment_label)}</strong><small>${titleCase(run.health_tag)}</small></span>
      <span class="quality ${run.data_quality}">${dataQualityLabel(run.data_quality)}</span>
    </a>`;
}


export async function renderRuns() {
  loading("Run analysis");
  const query = runQuery(0);
  const runs = await api(`/api/runs?${query}`);
  const rows = runs.map(runRow).join("");

  view.innerHTML = `
    <section class="page">
      <div class="page-heading"><div><p class="eyebrow">History</p><h1>Run analysis</h1><p><span id="run-count">${runs.length}</span> activities shown. Open any run for the full analysis.</p></div></div>
      <form id="run-filters" class="filter-bar"><label>Show<select name="flag"><option value="">All activities</option><option value="easy">Easy</option><option value="hard">Hard</option><option value="long">Long</option><option value="illness">Illness / abnormal</option><option value="no_gps">No GPS</option><option value="excluded">Excluded</option></select></label><label>From<input type="date" name="date_from" value="${runFilters.date_from}"></label><label>To<input type="date" name="date_to" value="${runFilters.date_to}"></label><label>Sort<select name="sort_by"><option value="date">Date</option><option value="distance">Distance</option><option value="pace">Moving pace</option><option value="heart_rate">Average HR</option><option value="standardized">Adjusted pace at ${targetHrLabel()}</option></select></label><label>Order<select name="sort_order"><option value="desc">Newest / highest first</option><option value="asc">Oldest / lowest first</option></select></label><button type="submit">Apply</button></form>
      <div class="run-table-heading"><span>Run</span><span>Distance</span><span>Pace and HR</span><span>Adjusted pace</span><span>Conditions</span><span>Assessment</span><span>Data</span></div>
      <div class="run-table">${rows || '<div class="empty-state">No runs match these filters.</div>'}</div>
      ${runs.length === RUN_PAGE_SIZE ? `<button id="load-more-runs" class="load-more" type="button">Load ${RUN_PAGE_SIZE} more</button>` : ""}
    </section>`;
  const form = view.querySelector("#run-filters");
  form.flag.value = runFilters.flag; form.sort_by.value = runFilters.sort_by; form.sort_order.value = runFilters.sort_order;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runFilters = Object.fromEntries(new FormData(form).entries());
    renderRuns();
  });
  // Append rather than re-render. Rebuilding the page threw away the scroll
  // position, so "load more" put the rows you asked for above the fold you
  // were already looking at.
  const loadMore = view.querySelector("#load-more-runs");
  loadMore?.addEventListener("click", async () => {
    const table = view.querySelector(".run-table");
    const shown = table.querySelectorAll(".run-row").length;
    loadMore.disabled = true;
    loadMore.textContent = "Loading…";
    try {
      const next = await api(`/api/runs?${runQuery(shown)}`);
      table.insertAdjacentHTML("beforeend", next.map(runRow).join(""));
      const total = shown + next.length;
      const count = view.querySelector("#run-count");
      if (count) count.textContent = String(total);
      if (next.length < RUN_PAGE_SIZE) loadMore.remove();
      else { loadMore.disabled = false; loadMore.textContent = `Load ${RUN_PAGE_SIZE} more`; }
    } catch (error) {
      loadMore.disabled = false;
      loadMore.textContent = error.message;
    }
  });
}
