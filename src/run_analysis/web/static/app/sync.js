import { api } from './api.js';

// Shared transfer controls keep uploads/drop and sync mutually exclusive.
export function bindSync({ button, begin, end, summary, showStages, refresh }) {
  button.addEventListener('click', async () => {
    if (!begin('Sync', 'Syncing your runs')) return;
    button.setAttribute('aria-busy', 'true');
    summary.textContent = 'Checking Intervals.icu and analyzing new runs…';
    try {
      const result = await api('/api/connections/intervals/sync', {
        method: 'POST', headers: { 'X-Run-Analysis-Local': '1' },
      });
      const stages = [...(result.upload?.stages ?? [])];
      const errors = [...(result.errors ?? [])];
      for (const file of result.upload?.files ?? []) {
        if (file.status === 'failed') errors.push(file.error ?? 'An activity could not be read.');
      }
      errors.forEach(detail => stages.push({ name: 'download', status: 'failed', detail }));
      showStages(stages);
      const changed = stages.some(stage => stage.name === 'process' && stage.status === 'complete');
      const needsAttention = stages.some(stage => ['failed', 'partial'].includes(stage.status));
      summary.textContent = result.new_runs
        ? `Synced ${result.new_runs} new run${result.new_runs === 1 ? '' : 's'}.`
        : changed ? 'Run data updated.' : 'Up to date. No new runs.';
      if (needsAttention) {
        summary.textContent = result.new_runs || changed
          ? `${summary.textContent} Some steps need attention.`
          : 'Sync incomplete. Some runs could not be synced.';
      }
      if (result.remaining_runs) summary.textContent += ` ${result.remaining_runs} more to check—click sync again.`;
      if (changed) await refresh();
    } catch (error) {
      summary.textContent = `Sync failed: ${error.message}`;
    } finally {
      button.setAttribute('aria-busy', 'false');
      end();
    }
  });
}
