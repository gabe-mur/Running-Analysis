import { escapeHtml, number } from './format.js';

export function longRunGoalMarkup(planner,state,trainingGoal=null) {
  const goal=planner?.requested_long_goal_miles??planner?.long_goal_miles;
  // Recorded exposure, not a future scheduled run or a lifetime personal best.
  // Capacity can be capped to a prescription; it is not the accomplishment.
  const current=state?.longest_completed_run_30d_miles;
  const available=Number.isFinite(goal)&&goal>0&&Number.isFinite(current);
  const percent=available?Math.min(100,Math.max(0,current/goal*100)):null;
  return `<a class="goal-panel panel long-run-goal" href="#progress"><p class="eyebrow">Your goal</p>
    <h2>${escapeHtml(trainingGoal?.headline??'Training for general fitness')}</h2>
    <div class="long-goal-progress"><p class="eyebrow">Weekly Long-Run Target</p>
    ${available?`<h3 class="long-goal-distance">${number(current)} <span>/ ${number(goal)} miles</span></h3><p class="goal-detail">Recent longest run</p>
      <div class="long-goal-track" role="progressbar" aria-label="Long-run distance toward your goal" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${number(percent,0)}" aria-valuetext="${escapeHtml(`${number(current)} of ${number(goal)} miles`)}"><span style="width:${percent}%"></span></div>
      <p class="long-goal-percent">${number(percent,0)}% of goal</p>`:
      `<h3 class="long-goal-distance">${Number.isFinite(goal)?`${number(goal)} miles`:'Set a long-run goal'}</h3><p class="goal-detail">${Number.isFinite(goal)?'Recorded progress not available yet.':'Choose your target in Settings.'}</p>`}
    </div><span class="panel-action">Progress →</span></a>`;
}
