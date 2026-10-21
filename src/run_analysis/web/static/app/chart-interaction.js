import { escapeHtml } from './format.js';
import { wireChartTypography } from './chart-typography.js';

// Optional metadata: static charts and their measurements are unchanged.
export function runChartAttributes({id,x,y,title,details=[]}) {
  return `aria-label="${escapeHtml(`${title} · ${details.join(' · ')}`)}" data-run-key="${escapeHtml(id)}" data-run-x="${x}" data-run-y="${y}" data-run-title="${escapeHtml(title)}" data-run-details="${escapeHtml(JSON.stringify(details))}"`;
}

export function nearestRun(points,x,y=0) {
  return points.reduce((best,p)=>{
    if(!Number.isFinite(p.x)||!Number.isFinite(p.y)) return best;
    if(!best) return p;
    const dx=Math.abs(p.x-x),bx=Math.abs(best.x-x);
    return dx<bx || (Math.abs(dx-bx)<.001&&Math.abs(p.y-y)<Math.abs(best.y-y))?p:best;
  },null);
}

export function runTooltipPosition({x,y,width,height,tooltipWidth,tooltipHeight}) {
  const inset=6,gap=12,maxLeft=Math.max(inset,width-tooltipWidth-inset);
  const right=x+gap,left=x-tooltipWidth-gap;
  // Flip beside the point instead of clamping an above-point overlay across it.
  const side=right<=maxLeft?'right':left>=inset?'left':'above';
  return {
    left:Math.max(inset,Math.min(side==='right'?right:side==='left'?left:x-tooltipWidth/2,maxLeft)),
    top:Math.max(inset,Math.min(side==='above'?y-tooltipHeight-gap:y-tooltipHeight/2,
      Math.max(inset,height-tooltipHeight-inset))),
    side,
  };
}

export function wireRunCharts(root) {
  wireChartTypography(root);
  root.querySelectorAll('svg[data-run-chart]').forEach((svg,index)=>{
    const marks=[...svg.querySelectorAll('[data-run-key]')];
    const points=marks.map(mark=>({mark,x:Number(mark.dataset.runX),y:Number(mark.dataset.runY),key:mark.dataset.runKey})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
    if(!points.length) return;
    // The custom inspector replaces native hover bubbles; labels and links stay.
    marks.forEach(mark=>mark.querySelector?.('title')?.remove());
    const owner=svg.parentElement,doc=svg.ownerDocument;
    owner.classList.add('run-chart');
    const tooltip=doc.createElement('div');
    tooltip.className='run-chart-tooltip';tooltip.hidden=true;tooltip.id=`run-chart-tip-${index}`;
    tooltip.setAttribute('role','tooltip');owner.append(tooltip);
    const guide=doc.createElementNS('http://www.w3.org/2000/svg','line');
    guide.setAttribute('class','run-hover-guide');guide.setAttribute('y1',svg.dataset.runTop??'25');guide.setAttribute('y2',svg.dataset.runBottom??'275');guide.hidden=true;
    guide.style.display='none';svg.append(guide);
    svg.setAttribute('tabindex','0');
    svg.setAttribute('aria-describedby',tooltip.id);
    svg.setAttribute('aria-description',svg.dataset.chartDescription??'Hover or tap to inspect a run. Left and right arrows select runs; Enter opens run details.');
    const ordered=[...new Map([...points].sort((a,b)=>a.x-b.x).map(p=>[p.key,p])).values()];
    let selected=null;
    function show(p) {
      if(!p) return;
      if(selected!==p) {
        selected=p;
        marks.forEach(mark=>mark.classList.toggle('is-nearest-run',mark.dataset.runKey===p.key));
        const heading=doc.createElement('strong');heading.textContent=p.mark.dataset.runTitle;
        let details=[];try {details=JSON.parse(p.mark.dataset.runDetails??'[]');} catch { /* Keep the date if optional metadata is unavailable. */ }
        tooltip.replaceChildren(heading,...details.map(text=>{const line=doc.createElement('span');line.textContent=text;return line;}));
      }
      guide.style.display='';guide.setAttribute('x1',p.x);guide.setAttribute('x2',p.x);tooltip.hidden=false;
      // Screen CTM handles SVG letterboxing, scaling and scrolling correctly.
      const matrix=svg.getScreenCTM();if(!matrix) return;
      const point=svg.createSVGPoint();point.x=p.x;point.y=p.y;
      const screen=point.matrixTransform(matrix),box=owner.getBoundingClientRect();
      const position=runTooltipPosition({x:screen.x-box.left,y:screen.y-box.top,
        width:owner.clientWidth,height:owner.clientHeight,
        tooltipWidth:tooltip.offsetWidth,tooltipHeight:tooltip.offsetHeight});
      tooltip.style.left=`${position.left}px`;tooltip.style.top=`${position.top}px`;
      tooltip.dataset.side=position.side;
    }
    function clear() {selected=null;marks.forEach(mark=>mark.classList.remove('is-nearest-run'));tooltip.hidden=true;guide.style.display='none';}
    function inspect(event) {
      const matrix=svg.getScreenCTM();if(!matrix) return;
      const pointer=svg.createSVGPoint();pointer.x=event.clientX;pointer.y=event.clientY;
      const at=pointer.matrixTransform(matrix.inverse());
      show(nearestRun(points,at.x,at.y));
    }
    svg.addEventListener('pointermove',inspect);
    svg.addEventListener('pointerdown',inspect);
    svg.addEventListener('pointerleave',event=>{if(event.pointerType!=='touch') clear();});
    svg.addEventListener('click',event=>{if(event.pointerType==='touch') event.preventDefault();});
    svg.addEventListener('focusin',event=>{
      const mark=event.target===svg?null:event.target.closest?.('[data-run-key]')??event.target.querySelector?.('[data-run-key]');
      show(mark?points.find(p=>p.mark===mark):selected??ordered.at(-1));
    });
    svg.addEventListener('focusout',event=>{if(!svg.contains(event.relatedTarget)) clear();});
    svg.addEventListener('keydown',event=>{
      if(event.key==='Escape') {clear();return;}
      if(event.target!==svg) return; // Keep native link keyboard behaviour.
      const at=ordered.findIndex(p=>p.key===selected?.key);
      const next={ArrowRight:Math.min(ordered.length-1,at+1),ArrowLeft:Math.max(0,at<0?ordered.length-1:at-1),Home:0,End:ordered.length-1}[event.key];
      if(next!==undefined) {event.preventDefault();show(ordered[next]);}
      else if(event.key==='Enter'&&selected) {event.preventDefault();(selected.mark.closest('a')??selected.mark.querySelector('a'))?.click();}
    });
  });
}
