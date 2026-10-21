// Evaluate the shipped wording on the Python audit's exact evidence payloads.
// No UI, settings, statistical thresholds or model parameters are changed.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {performanceSummary} from '../src/run_analysis/web/static/app/progress-insights.js';

export function auditHeadlines(report) {
  return {version:'progress-headline-responsiveness-v1',source_audit:report.version,
    production_changed:false,cases:report.pipeline_replays.map(p=>({
      case_kind:p.case_kind,onset:p.onset,gain_percent:p.gain_percent,ramp_days:p.ramp_days,
      recovered_speed_gain_percent:p.refitted_calibration_response_percent,
      monthly_change:p.refitted.monthly_change,
      baseline:performanceSummary(p.pace_headline_baseline),
      injected:performanceSummary(p.pace_headline_refitted),
      period:p.pace_headline_refitted.period_change,
      trajectory:p.pace_headline_refitted.within_window_trend}))};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) {
  const [input,output]=process.argv.slice(2);
  if(!input||!output) throw new Error('Usage: node scripts/audit_progress_headlines.mjs INPUT.json OUTPUT.json');
  const result=auditHeadlines(JSON.parse(fs.readFileSync(input,'utf8')));
  fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  for(const c of result.cases.filter(c=>c.case_kind==='recent')) console.log(JSON.stringify({
    ramp_days:c.ramp_days,gain_percent:c.gain_percent,recovered_speed_gain_percent:c.recovered_speed_gain_percent,
    headline:c.injected.headline,conflict:c.injected.conflict,
    period_direction:c.period?.direction,period_evidence:c.period?.evidence,
    trajectory_direction:c.trajectory?.direction,trajectory_evidence:c.trajectory?.evidence}));
}
