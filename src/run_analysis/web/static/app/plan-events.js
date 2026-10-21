// Notifications carry no athlete data. Saved-plan reads remain authoritative,
// including a newer upload that still needs a refresh after this job finishes.
const KEY='running-coach:plan-loaded';

export function createPlanEvents({broadcast=()=>{},listen=()=>{},token=()=>`${Date.now()}:${Math.random()}`}={}) {
  const listeners=new Set();
  const notify=()=>{for(const listener of listeners) {try {listener();} catch {/* Display failures cannot fail a successful planner request. */}}};
  listen(event=>{if(event.key===KEY&&event.newValue) notify();});
  return {
    subscribe(listener) {listeners.add(listener);return ()=>listeners.delete(listener);},
    loaded() {notify();try {broadcast(KEY,token());} catch {/* Restricted storage cannot prevent local refresh. */}},
  };
}

const events=createPlanEvents({
  broadcast:(key,value)=>globalThis.window?.localStorage?.setItem(key,value),
  listen:handler=>globalThis.window?.addEventListener('storage',handler),
});
export const onPlanLoaded=listener=>events.subscribe(listener);
export const announcePlanLoaded=()=>events.loaded();
