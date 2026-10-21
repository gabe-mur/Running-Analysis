import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { racePreparationFields, preparationSummary, readRacePreparation, bindRacePreparation } from '../../src/run_analysis/web/static/app/race-settings.js';

test('shared simple fields have no pace or false calendar activation', () => {
  const html = racePreparationFields({ distance: 'marathon', race_date: '2027-10-01', preparation_mode: 'more' });
  assert.match(html, /value="marathon" selected/);
  assert.match(html, /value="2027-10-01"/);
  assert.match(html, /value="more" selected/);
  assert.match(html, /Apply training goal/);
  assert.doesNotMatch(html, /Use Apply training goal to save this goal for your next plan update/);
  assert.doesNotMatch(html, /Save settings keeps a draft|successful plan refresh/);
  assert.doesNotMatch(html, /name="goal_pace|Riegel|latest 10|VO₂|safe|injury risk/);
});

test('only marathon can submit different preparation options; general clears date', () => {
  const values = { race_distance: '5k', race_date: '2027-10-01', race_mode: 'more' };
  const form = { querySelector: (s) => ({ value: values[s.match(/name="([^"]+)/)[1]] }) };
  assert.equal(readRacePreparation(form).preparation_mode, 'balanced');
  values.race_distance = 'general_fitness';
  assert.equal(readRacePreparation(form).race_date, null);
});

test('guidance describes conditional dates/doses, not race readiness', () => {
  const result = { options: [{ mode: 'balanced', target: { long_miles: 20, volume_per_7: 40 }, earliest_conditional_date: '2027-02-27', requested_date_fits: false }] };
  const text = preparationSummary(result);
  assert.match(text, /earlier than/); assert.match(text, /40.0 miles per 7 days/);
  assert.match(text, /Actual delivery, recovery and the full calendar still need checking/);
  assert.doesNotMatch(text, /ready|guaranteed|NaN|undefined/);
  assert.equal(preparationSummary({detail: 'Needs evidence', options: []}), 'Needs evidence');
});

test('Settings and Setup share fields and send the same draft, not legacy pace inputs', () => {
  for (const file of ['settings', 'setup']) {
    const source = readFileSync(new URL(`../../src/run_analysis/web/static/app/views/${file}.js`, import.meta.url), 'utf8');
    assert.match(source, /bindRacePreparation/);
    assert.match(source, /race_preparation: readRacePreparation\(form\)/);
    assert.doesNotMatch(source, /name="goal_pace|name="training_goal|both a race date and a goal pace|target pace, which are set/);
  }
});

function fakeForm() {
  const names=['race_distance','race_date','race_mode','preview','apply','output','outcome','outcomeChoice','outcomeSave','outcomeMessage'];
  const nodes=Object.fromEntries(names.map(name=>[name,{value:name==='race_distance'?'general_fitness':name==='race_mode'?'balanced':'',
    disabled:false,hidden:false,textContent:'',listeners:{},label:{hidden:false},
    addEventListener(type,fn){this.listeners[type]=fn;},closest(){return this.label;},reportValidity(){return true;}}]));
  const select = selector => {
    if(selector==='[data-general-long-goal]')return null;
    const name=selector.match(/name="([^"]+)"/)?.[1];
    const aliases={'[data-race-outcome]':'outcome','[data-race-outcome-choice]':'outcomeChoice',
      '[data-race-outcome-save]':'outcomeSave','[data-race-outcome-message]':'outcomeMessage'};
    return nodes[name || aliases[selector] || (selector.includes('preview')?'preview':selector.includes('apply')?'apply':'output')];
  };
  nodes.outcome.querySelector=select;
  return {nodes,querySelector:select};
}

test('general fitness has no redundant guidance; pending updates stay explicit', async () => {
  const form=fakeForm();
  bindRacePreparation(form,async()=>({status:'available',expected_revision:'fixture',pending:false}));
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(form.nodes.output.textContent,'');
  const pending=fakeForm();
  bindRacePreparation(pending,async()=>({status:'available',expected_revision:'fixture',pending:true}));
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(pending.nodes.output.textContent,'Goal saved for the next plan update.');
});

test('race outcome submits original activity identity, not a guessed completion or goal change',async()=>{
  const form=fakeForm();const calls=[];
  const status={status:'available',expected_revision:'goal',race_activation_available:false,
    outcome:{report_available:true,status:'unknown',expected_revision:'event-parent',
      activities:[{key:'original-fit',date:'2026-10-07',miles:6.2}]}};
  bindRacePreparation(form,async(path,options)=>{
    calls.push([path,options]);
    return options ? {...status,outcome:{...status.outcome,status:'recorded',physical_key:'original-fit'},
      detail:'Race outcome saved. Refresh Weekly Plan.'}:status;
  });
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(form.nodes.outcome.hidden,false);
  assert.match(form.nodes.outcomeChoice.innerHTML,/6.2 miles/);
  form.nodes.outcomeChoice.value='original-fit';
  await form.nodes.outcomeSave.listeners.click({currentTarget:form.nodes.outcomeSave});
  assert.deepEqual(JSON.parse(calls[1][1].body),{status:'recorded',physical_key:'original-fit',expected_revision:'event-parent'});
  assert.equal(calls[1][0],'/api/settings/race-preparation/outcome');
  assert.equal(calls.length,2);
  assert.match(form.nodes.outcomeMessage.textContent,/Refresh Weekly Plan/);
  assert.equal(form.nodes.outcomeSave.disabled,false);
});

test('nonparticipation carries no invented activity and a stale-report error remains visible',async()=>{
  const form=fakeForm();let body;
  bindRacePreparation(form,async(path,options)=>{
    if(options){body=JSON.parse(options.body);throw new Error('Race outcome changed; reload the form');}
    return {status:'available',expected_revision:'goal',outcome:{report_available:true,status:'unknown',
      expected_revision:'parent',activities:[]}};
  });
  await new Promise(resolve=>setImmediate(resolve));
  form.nodes.outcomeChoice.value='did_not_start';
  await form.nodes.outcomeSave.listeners.click({currentTarget:form.nodes.outcomeSave});
  assert.equal(body.physical_key,null);
  assert.equal(body.status,'did_not_start');
  assert.match(form.nodes.outcomeMessage.textContent,/reload/);
  assert.equal(form.nodes.outcomeSave.disabled,false);
});

test('unvalidated race cannot be applied while guidance and general fitness remain usable',async()=>{
  const form=fakeForm();
  bindRacePreparation(form,async()=>({status:'available',expected_revision:'fixture',race_activation_available:false,
    activation_note:'Race validation pending'}));
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(form.nodes.apply.disabled,false);
  form.nodes.race_distance.value='10k';form.nodes.race_distance.listeners.change();
  assert.equal(form.nodes.apply.disabled,true);
  assert.equal(form.nodes.preview.hidden,false);
  assert.equal(form.nodes.output.textContent,'Race validation pending');
});

test('released race apply queues an explicit pace-free goal without triggering a plan refresh',async()=>{
  const form=fakeForm();const calls=[];
  const status={status:'available',expected_revision:'parent',race_activation_available:true};
  bindRacePreparation(form,async(path,options)=>{
    calls.push([path,options]);
    return options ? {...status,pending:true,calendar_changed:false,detail:'Goal saved. Refresh Weekly Plan.'}:status;
  });
  await new Promise(resolve=>setImmediate(resolve));
  form.nodes.race_distance.value='10k';form.nodes.race_date.value='2027-05-01';
  form.nodes.race_distance.listeners.change();
  assert.equal(form.nodes.apply.disabled,false);
  await form.nodes.apply.listeners.click();
  assert.deepEqual(JSON.parse(calls[1][1].body),{goal:{distance:'10k',race_date:'2027-05-01',
    preparation_mode:'balanced'},expected_revision:'parent'});
  assert.equal(calls[1][0],'/api/settings/race-preparation/apply');
  assert.equal(calls.length,2);
  assert.match(form.nodes.output.textContent,/Refresh Weekly Plan/);
  assert.equal(form.nodes.apply.disabled,false);
});

test('reload uses durable saved goal, not a stale draft, without overwriting active edits',async()=>{
  const saved={distance:'marathon',race_date:'2027-05-01',preparation_mode:'more'};
  const form=fakeForm();
  bindRacePreparation(form,async()=>({status:'available',expected_revision:'fixture',requested:saved,pending:true,
    race_activation_available:false}));
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(form.nodes.race_distance.value,'marathon');
  assert.equal(form.nodes.race_date.value,'2027-05-01');
  assert.equal(form.nodes.race_mode.value,'more');
  const edited=fakeForm();let finish;
  bindRacePreparation(edited,()=>new Promise(resolve=>{finish=resolve;}));
  edited.nodes.race_distance.value='5k';edited.nodes.race_distance.listeners.change();
  finish({status:'available',expected_revision:'fixture',requested:saved,pending:true,race_activation_available:false});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(edited.nodes.race_distance.value,'5k');
});

test('Apply general fitness saves target atomically without unsaved running preferences',async()=>{
  const form=fakeForm(),calls=[];
  const target={value:'20',disabled:false,reportValidity:()=>true};
  const select=form.querySelector;
  form.querySelector=s=>s==='[name="planner_long_goal"]'?target:select(s);
  const planner={status:'adopted',long_goal_miles:17,run_frequency:'balanced',long_run_weekday:5,preferences_revision:'prefs'};
  const status={status:'available',expected_revision:'goal'};
  bindRacePreparation(form,async(path,options)=>{
    calls.push([path,options]);return options?{...status,detail:'Goal saved.',planner:{...planner,preferences_revision:'new',requested_long_goal_miles:20}}:status;
  },{planner});
  await new Promise(resolve=>setImmediate(resolve));
  await form.nodes.apply.listeners.click();
  assert.equal(calls.length,2);
  assert.deepEqual(JSON.parse(calls[1][1].body),{goal:{distance:'general_fitness',race_date:null,preparation_mode:'balanced'},
    expected_revision:'goal',planner_preferences:{long_run_goal_miles:20,run_frequency:'balanced',long_run_weekday:5,expected_revision:'prefs'}});
  assert.equal(planner.preferences_revision,'new');
  assert.equal(form.nodes.apply.disabled,false);
  target.reportValidity=()=>false;
  await form.nodes.apply.listeners.click();
  assert.equal(calls.length,2);
});
