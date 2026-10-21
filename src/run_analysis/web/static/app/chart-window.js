// Crop an existing curve, not its evidence. Boundary points only interpolate
// an already drawn segment; they are never observations or hover targets.
export function windowedChartSeries(history,start,end,valuesFor,maxGapDays) {
  const samples=history.map(row=>({at:+new Date(row.as_of),values:valuesFor(row),row}))
    .sort((a,b)=>a.at-b.at);
  const groups=[];
  let group=null;
  for(const sample of samples) {
    if(!Number.isFinite(sample.at)||!sample.values?.length||!sample.values.every(Number.isFinite)) {
      group=null;
      continue;
    }
    if(!group||sample.at-group.at(-1).at>maxGapDays*86400000) {
      group=[];
      groups.push(group);
    }
    group.push(sample);
  }
  const interpolate=(a,b,at)=>({at,values:a.values.map((v,i)=>v+(b.values[i]-v)*(at-a.at)/(b.at-a.at))});
  const clipped=groups.map(group=>{
    const visible=[];
    group.forEach((sample,i)=>{
      const previous=group[i-1];
      if(previous&&previous.at<start&&sample.at>start) visible.push(interpolate(previous,sample,start));
      if(sample.at>=start&&sample.at<=end) visible.push(sample);
      if(previous&&previous.at<end&&sample.at>end) visible.push(interpolate(previous,sample,end));
    });
    return visible;
  }).filter(group=>group.length);
  return {groups:clipped,points:samples.filter(s=>s.values?.length&&s.values.every(Number.isFinite)&&s.at>=start&&s.at<=end)};
}
