// The browser submits a short job request, then reads its result. Leaving a
// page never cancels accepted planner work; reopening joins the server's job.
import { api } from './api.js';
import { announcePlanLoaded } from './plan-events.js';

export function createPlannerRequest({fetcher=api,pause=ms=>new Promise(resolve=>setTimeout(resolve,ms)),
  uuid=()=>globalThis.crypto.randomUUID(),announce=announcePlanLoaded}={}) {
  const announced=new Set();
  return async function request(path,options={}) {
    const operation={'/api/weekly-schedule/latest':'refresh','/api/weekly-schedule':'update',
      '/api/weekly-schedule/rest-day':'rest'}[path];
    if(!operation)throw Error('Unknown planner operation');
    const requestId=new Headers(options.headers).get('X-Planner-Request')??uuid();
    const body={request_id:requestId,operation};
    if(operation!=='refresh')body[operation]=JSON.parse(options.body);
    let job=await fetcher('/api/weekly-schedule/jobs',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!job?.job_id)throw Error('The planner did not return a job ID. Please try again.');
    while(['queued','running'].includes(job.state)) {
      job=await fetcher(`/api/weekly-schedule/jobs/${job.job_id}`);
      if(['queued','running'].includes(job.state))await pause(750);
    }
    if(job.state==='failed')throw Error(job.error?.detail??'Could not update your plan. Try again.');
    if(job.state!=='complete')throw Error('Planner status unavailable. Please try again.');
    if(['ready','review_required'].includes(job.result?.planner_status)&&!announced.has(job.job_id)) {
      announced.add(job.job_id);
      if(announced.size>128)announced.delete(announced.values().next().value);
      announce();
    }
    return job.result;
  };
}

export const plannerRequest=createPlannerRequest();
