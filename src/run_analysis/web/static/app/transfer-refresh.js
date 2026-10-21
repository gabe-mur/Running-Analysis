// Finishing a transfer is not permission to start a new planner search.
export async function refreshAfterTransfer(refresh, view, hash = location.hash, changed = true) {
  if (!changed) return;
  if (hash !== '#next-run') return refresh();
  view.innerHTML = `<section class="page narrow"><p class="eyebrow">Weekly Plan</p><h1>Run data updated</h1><p>Your plan needs updating.</p><button type="button" data-update-weekly-plan>Update Weekly Plan</button></section>`;
  view.querySelector('[data-update-weekly-plan]').addEventListener('click', () => refresh());
}
