"""Bounded calendar-landscape map: exhaustive local neighborhoods + global controls.

No cheap-score pruning. Dates are rematerialized by the current allocator; this
does not enumerate all possible roles/doses/times on each calendar. Sampled
components are not proof of disconnected basins in the unobserved full space.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from math import comb, sqrt
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter
import json

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from run_analysis.prescription_distance import UnrepresentableDistanceTarget
from scripts.audit_continuation_upload import reconstruct, hard_score, evaluate_physical, unpack_soft
from scripts.audit_conditional_upload import components, score
from scripts.audit_calendar_materialization import prescriptions
from scripts.audit_regrowth_gates import check_candidate
from scripts.calendar_neighbor_prototype import calendar_neighbors
from scripts.continuation_replan import fingerprint


def edit_distance(a,b):
    """Unit add/remove or one-day move distance, on ordered distinct dates."""
    a,b=tuple(sorted(a)),tuple(sorted(b))
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+abs(x-y)))
        previous=current
    return previous[-1]


def panel(full_seed,cheap_seed,allowed,*,per_count=6,random_seed=20260923):
    """Complete local neighborhoods, independent stratified distant controls."""
    if per_count<0:
        raise ValueError('Nonnegative control count required')
    allowed=sorted(allowed)
    if not allowed or not set(full_seed)<=set(allowed) or not set(cheap_seed)<=set(allowed):
        raise ValueError('Seeds must use available dates')
    sources={}
    def add(offsets,label):
        sources.setdefault(tuple(offsets),set()).add(label)
    for label,seed in (('full_winner',full_seed),('cheap_winner',cheap_seed)):
        add(tuple(seed),label)
        for candidate in calendar_neighbors(seed,allowed):
            add(candidate,label+'_neighbor')
    rng=Random(random_seed)
    counts=range(max(1,min(len(full_seed),len(cheap_seed))-1),
                 min(len(allowed),max(len(full_seed),len(cheap_seed))+1)+1)
    populations={}
    for count in counts:
        populations[count]=comb(len(allowed),count)
        sampled=set()
        while len(sampled)<min(per_count,populations[count]):
            sampled.add(tuple(sorted(rng.sample(allowed,count))))
        for candidate in sorted(sampled):
            add(candidate,'stratified_control')
    return [(key,sorted(value)) for key,value in sorted(sources.items())],populations


def rank(values):
    ordered=sorted(range(len(values)),key=lambda i:values[i])
    ranks=[0.]*len(values)
    i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and values[ordered[j]]==values[ordered[i]]:
            j+=1
        for position in ordered[i:j]:
            ranks[position]=(i+j-1)/2
        i=j
    return ranks


def extend_panel(previous,allowed):
    """Reuse prior rows without mutation; complete one neighborhood of its best."""
    cache={tuple(r['offsets']):deepcopy(r) for r in previous['rows']}
    refined=tuple(previous['summary']['best']['offsets'])
    if refined not in cache or not set(refined)<=set(allowed):
        raise ValueError('Refinement seed must be an evaluated available calendar')
    sources={key:set(row['sources']) for key,row in cache.items()}
    sources[refined].add('refined_winner')
    for candidate in calendar_neighbors(refined,allowed):
        sources.setdefault(candidate,set()).add('refined_winner_neighbor')
    candidates=[(key,sorted(labels)) for key,labels in sorted(sources.items())]
    return candidates,cache,refined


def correlation(xs,ys):
    if len(xs)<2:
        return None
    xmean,ymean=sum(xs)/len(xs),sum(ys)/len(ys)
    denominator=sqrt(sum((x-xmean)**2 for x in xs)*sum((y-ymean)**2 for y in ys))
    return sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys))/denominator if denominator else None


def summarize(rows,*,seed=20260923):
    eligible=[r for r in rows if r.get('eligible')]
    result=dict(total=len(rows),full_scored=sum('full_cost' in r for r in rows),
                eligible=len(eligible),reasons=dict(Counter(r.get('status') for r in rows)))
    if not eligible:
        return result
    ordered=sorted(eligible,key=lambda r:(r['full_cost'],r['offsets']))
    result['best']={k:ordered[0][k] for k in ('offsets','full_cost','hard_cost','sources','prescriptions')}
    result['spearman_hard_full']=correlation(rank([r['hard_cost'] for r in eligible]),
                                             rank([r['full_cost'] for r in eligible]))
    result['strata']={}
    for label in ('full_winner_neighbor','cheap_winner_neighbor','refined_winner_neighbor','stratified_control'):
        subset=[r for r in rows if label in r['sources']]
        valid=[r for r in subset if r.get('eligible')]
        result['strata'][label]=dict(sampled=len(subset),eligible=len(valid),
            best=min((r['full_cost'] for r in valid),default=None),
            median=median(r['full_cost'] for r in valid) if valid else None)
    edges=[]
    far=[]
    for i,a in enumerate(eligible):
        for j in range(i):
            b=eligible[j]
            d=edit_distance(a['offsets'],b['offsets'])
            delta=abs(a['full_cost']-b['full_cost'])
            if d==1:
                edges.append((i,j,delta,(a['hard_cost']-b['hard_cost'])*(a['full_cost']-b['full_cost'])))
            elif d>=4:
                far.append(delta)
    result['local_smoothness']=dict(neighbor_pairs=len(edges),
        median_neighbor_delta=median(x[2] for x in edges) if edges else None,
        distant_pairs=len(far),median_distant_delta=median(far) if far else None,
        strict_order_reversals=sum(x[3]<-1e-9 for x in edges),
        concordant_pairs=sum(x[3]>1e-9 for x in edges))
    for label in ('full_winner','cheap_winner','refined_winner'):
        anchors=[r for r in eligible if label in r['sources']]
        if anchors:
            anchor=anchors[0]
            neighborhood=[r for r in eligible if label+'_neighbor' in r['sources']]
            result[label]=dict(cost=anchor['full_cost'],eligible_neighbors=len(neighborhood),
                improving_neighbors=sum(r['full_cost']<anchor['full_cost']-1e-9 for r in neighborhood),
                best_neighbor=min((r['full_cost'] for r in neighborhood),default=None))
    top_count=min(10,len(eligible))
    top={tuple(r['offsets']) for r in ordered[:top_count]}
    cheap=sorted(eligible,key=lambda r:(r['hard_cost'],r['offsets']))
    result['top10_recall_at_cheap10']=sum(tuple(r['offsets']) in top for r in cheap[:top_count])/top_count
    result['best_hard_rank']=1+sum(r['hard_cost']<ordered[0]['hard_cost'] for r in eligible)
    result['top_full_score_band_correlations']={}
    for count in (10,20):
        subset=ordered[:count]
        result['top_full_score_band_correlations'][str(count)]=dict(count=len(subset),
            spearman=correlation(rank([r['hard_cost'] for r in subset]),rank([r['full_cost'] for r in subset])),
            caveat='Conditioned on full-score rank; describes elite reordering, not predictive validation.')
    result['low_score_clusters']=[]
    rng=Random(seed)
    for margin in (1.,5.):
        low={i for i,r in enumerate(eligible) if r['full_cost']<=ordered[0]['full_cost']+margin}
        adjacency={i:set() for i in low}
        for i,j,_,_ in edges:
            if i in low and j in low:
                adjacency[i].add(j);adjacency[j].add(i)
        unseen=set(low);groups=[]
        while unseen:
            stack=[unseen.pop()];group=[]
            while stack:
                node=stack.pop();group.append(node)
                new=adjacency[node]&unseen
                unseen-=new;stack.extend(sorted(new))
            groups.append(group)
        observed=sum(i in low and j in low for i,j,_,_ in edges)
        random_edges=[]
        for _ in range(200):
            permuted=set(rng.sample(range(len(eligible)),len(low)))
            random_edges.append(sum(i in permuted and j in permuted for i,j,_,_ in edges))
        result['low_score_clusters'].append(dict(margin=margin,count=len(low),
            component_sizes=sorted((len(g) for g in groups),reverse=True),
            low_low_edges=observed,shuffled_label_mean_edges=sum(random_edges)/len(random_edges),
            caveat='Conditional on this deliberately sampled graph; missing bridges are not proven basins.'))
    result['best_calendars']=[{k:r[k] for k in ('offsets','full_cost','hard_cost','sources','prescriptions')}
                              for r in ordered[:10]]
    return result


def investigate(report,*,decision,per_count,emit,extend=None):
    if not 0<=decision<len(report['replans']):
        raise ValueError('Decision outside saved report')
    truncated=dict(report,replans=report['replans'][:decision+1])
    result={}
    def observe(index,inputs,days,evidence):
        if index!=decision:
            return
        from datetime import timedelta
        snapshot=report['replans'][index]
        saved=snapshot['planner_diagnostics']['continuation_selection']
        full=days+unpack_soft(saved['soft_sessions'])
        end=days[0].date+timedelta(days=21)
        native=score(inputs,evidence,full,end,end+timedelta(days=28),historical_pace=False)
        assert abs(native['components']['total']-saved['search_cost'])<1e-8
        result['native_control']=native['components']
        full_seed=tuple(i for i,d in enumerate(days) if d.recommendation)
        record=report['synthetic_sandbox']['calendar_neighbor_prototype'][index]
        cheap_seed=tuple(record['unconstrained']['offsets'])
        allowed=set(range(len(days)))-set(inputs['forced_rest_offsets'])-set(inputs['completed_run_offsets'])
        candidates,populations=panel(full_seed,cheap_seed,allowed,per_count=per_count)
        cache={}
        if extend is not None:
            if (extend['decision_at']!=snapshot['generated_at']
                    or tuple(extend['full_seed'])!=full_seed
                    or abs(extend['native_control']['total']-native['components']['total'])>1e-8):
                raise ValueError('Cached map does not match this fixed decision/control')
            candidates,cache,refined=extend_panel(extend,allowed)
            result['extension']=dict(reused=len(cache),new_candidates=len(candidates)-len(cache),
                refined_seed=refined,cache_scope='Same saved decision and native score; offline within-version reuse')
        result.update(decision_at=snapshot['generated_at'],full_seed=full_seed,cheap_seed=cheap_seed,
            calendar_space=2**len(allowed)-1,stratum_populations=populations,
            scope='One fixed decision, full one-edit neighborhoods and seeded count-stratified controls; no global optimality claim',
            candidate_count=len(candidates),rows=[])
        emit(dict(event='start',**{k:v for k,v in result.items() if k!='rows'}))
        for number,(offsets,sources) in enumerate(candidates,1):
            if offsets in cache:
                row=cache[offsets]
                row['sources']=sources
                result['rows'].append(row)
                emit(dict(event='cached_candidate',number=number,**row))
                continue
            started=perf_counter()
            row=dict(offsets=offsets,sources=sources,eligible=False)
            try:
                program,hard,details=hard_score(inputs,offsets)
                row['hard_cost']=hard.total_cost
                if program is None or details.get('unrepresentable_distance_target'):
                    row['status']='unrepresentable'
                else:
                    row['fingerprint']=fingerprint(program)
                    row['prescriptions']=prescriptions(program)
                    row['hard_components']=hard.model_dump(mode='json')
                    acceptance=assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
                        planning_mode=inputs['target_evidence'].planning_mode.value)
                    row['acceptance']=asdict(acceptance)
                    if acceptance.rejected or acceptance.reason=='missing_context':
                        row['status']='hard_acceptance_rejection'
                    else:
                        scored=evaluate_physical(inputs,program,evidence)
                        assert abs(scored['hard']['total_cost']-hard.total_cost)<1e-8
                        row['full_cost']=scored['continuation']['combined_cost']
                        row['full_components']=components(scored)
                        checked=scored['continuation']['opportunity_repair']
                        row['sequence_status']=checked['status']
                        row['gate_audit']=check_candidate(snapshot,inputs,program)
                        row['eligible']=checked['status']=='checked' and not row['gate_audit']['failures']
                        row['status']='eligible' if row['eligible'] else 'sequence_or_gate_rejection'
            except UnrepresentableDistanceTarget as error:
                row.update(status='unrepresentable',error=str(error))
            row['seconds']=perf_counter()-started
            result['rows'].append(row)
            emit(dict(event='candidate',number=number,**row))
            if number%5==0 or number==len(candidates):
                print(f"Mapped {number}/{len(candidates)}; eligible {sum(r['eligible'] for r in result['rows'])}; "
                      f"best {min((r['full_cost'] for r in result['rows'] if r['eligible']),default=None)}",flush=True)
        result['summary']=summarize(result['rows'])
    reconstruct(truncated,observe)
    return result


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--decision',type=int,required=True)
    parser.add_argument('--per-count',type=int,default=6)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--extend-report',type=Path,help='Reuse map and fully expand one neighborhood of its best plan')
    args=parser.parse_args()
    journal=args.output_json.with_suffix('.jsonl')
    if args.output_json.exists() or journal.exists():
        parser.error('Choose new report and journal paths')
    started=perf_counter()
    with journal.open('x') as stream:
        def emit(row):
            stream.write(json.dumps(row,default=str)+'\n');stream.flush()
        result=investigate(json.loads(args.report.read_text()),decision=args.decision,
                           per_count=args.per_count,emit=emit,
                           extend=json.loads(args.extend_report.read_text()) if args.extend_report else None)
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2,default=str);stream.write('\n')
    print(json.dumps({k:v for k,v in result['summary'].items() if k not in ('best','best_calendars')},indent=2),flush=True)
    print('Seconds',result['seconds'])


if __name__=='__main__':
    main()
