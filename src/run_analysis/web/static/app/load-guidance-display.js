import {escapeHtml, number} from './format.js';

// Guidance comes from the exact HR-load series plotted beside this panel.
// Never substitute the older distance-based strain flag or Building status.
export function loadGuidanceDisplay(overview) {
  const guidance=overview?.load_guidance, current=guidance?.current;
  if(!current) return '<h3 class="training-status none">Load guidance unavailable</h3><p>The personal load range could not be loaded. Your issued plan is unchanged.</p>';
  const report=overview.current_check_in;
  const adverse=['pain_or_injury_concern','sick_or_recovering'].includes(report?.status);
  const tired=report?.status==='little_tired';
  const tone=current.status==='within'?'good':current.status==='unavailable'?'none':'neutral';
  const band=current.recommended_range, recent=overview.load_display?.current?.recent?.value;
  const action=adverse?'Review your current symptoms before training.':tired?'Keep easy work familiar; reassess demanding sessions.':current.action;
  const detail=adverse?'Your recent health report takes priority over the load range. Being in range does not clear you to run.':tired?'You recently reported feeling a little tired. This load estimate cannot tell whether you are recovered.':current.detail;
  return `<h3 class="training-status ${tone}">${escapeHtml(current.label)}</h3>
    <p class="load-action">${escapeHtml(action)}</p><p class="status-detail">${escapeHtml(detail)}</p>
    ${band&&Number.isFinite(recent)?`<div class="load-range-numbers"><div><span>Recent load</span><strong>${number(recent,0)}</strong></div><div><span>Recommended range</span><strong>${number(band.low,0)}–${number(band.high,0)}</strong></div></div><small class="muted">HR load per day · same units as the graph</small>`:''}
    ${current.boundary_overlap?'<p class="load-boundary-note">The estimate overlaps a range boundary; avoid making a major adjustment from load alone.</p>':''}
    <details><summary>How this recommendation works</summary><p>Your recommended range adjusts gradually as your training changes. A brief dip can be normal recovery; sustained low load suggests rebuilding consistency.</p></details>`;
}
