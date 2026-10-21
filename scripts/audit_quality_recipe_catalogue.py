"""Frozen structural catalogue audit, not personalized prescriptions or search."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import timedelta
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from time import perf_counter

from run_analysis.adaptive.quality_block_contract import adapt_v2, read_quality_contract
from run_analysis.adaptive.quality_prescription_contract import PaceReference, compile_quality, read_prescription
from run_analysis.adaptive.quality_recipe_catalogue import (
    DoseWindow, compile_catalogue, progression_options, reviewed_recipes)
from run_analysis.staged_progression import MILE
from run_analysis.workload_accounting import Quantity
from scripts.research_manifest import freeze_sources


def audit(before, output):
    input_hash = sha256(before.read_bytes()).hexdigest()
    saved = json.loads(before.read_text())
    # Current frozen handoff contains the authored AS179 hill clarification.
    # Earlier Q5 commands are a separate archive-read control, not a silently
    # regenerated comparator with today's hill recipe substituted into it.
    previous = [read_prescription(r['prescription']) for r in saved['menu']['prescriptions']]
    archive_path = Path('output/quality-q5-block-contract-20261008/before.json')
    archive = json.loads(archive_path.read_text())
    archive_hash = sha256(archive_path.read_bytes()).hexdigest()
    at = previous[0].request.issued_at
    def pace(seconds, purpose):
        return PaceReference(Quantity(seconds*.9, seconds*1.1, seconds,
            'explicit engineering sensitivity, not athlete estimate or confidence'),
            at-timedelta(seconds=1), ('engineering_'+purpose,), 'declared structural '+purpose+' scenario')
    paces = dict(easy_pace=pace(.4, 'easy'), threshold_pace=pace(.32, 'threshold'),
        aerobic_pace=pace(.28, 'hard_aerobic'), fast_pace=pace(.27, 'short_fast'),
        hill_pace=pace(.5, 'uphill'))
    def window(low, high, name):
        return DoseWindow(low, high, 10., 7200., 20, 'structural '+name+' envelope; not athlete stage admission')
    windows = {'sustained':window(360,900,'sustained'), 'faster':window(240,720,'faster')}
    broader = {'sustained':window(360,1440,'sustained'), 'faster':window(240,1080,'faster')}
    protocol = dict(scope='Catalogue1/2 structural candidate generation; no fitness prediction or accepted athlete progression',
        all_paces={k:asdict(v) for k,v in paces.items()},
        windows={k:asdict(v) for k,v in windows.items()}, broader_windows={k:asdict(v) for k,v in broader.items()},
        source='AS182 explicit coaching grids; do not copy elite dose totals',
        checks=['All declared valid target/count alternatives retained; full session extents/display/restart agree',
            'Broader nominal envelopes add doses without altering commands already generated',
            'All16 current frozen handoff v2 commands recompile exactly; no implicit migration',
            'All16 earlier Q5 archived contracts remain readable with original keys and numeric commands',
            'Explicit progression ramp preserved when varying its committed tenth-mile total'],
        limits=['No athlete-derived threshold/hard-aerobic pace calibration or purpose-credit allocation',
            'No new workload accuracy result, search, calendar test, live write or stride implementation',
            'Nominal generation bounds are not complete uncertainty-aware stage/health/recovery admission'])
    output.mkdir(parents=True,exist_ok=False)
    with (output/'protocol.json').open('x') as stream:
        json.dump(protocol,stream,indent=2,default=lambda v:v.isoformat())
    manifest = freeze_sources(output/'sources.zip',config={'structural_only':True},entrypoint=__name__,
        inputs=(before,archive_path,output/'protocol.json'))
    started=perf_counter()
    recipes=reviewed_recipes(issued_at=at,**paces,open_recovery_seconds=90)
    small=compile_catalogue(recipes,windows); large=compile_catalogue(recipes,broader)
    keys={c.prescription.key for c in small['candidates']}
    if not keys <= {c.prescription.key for c in large['candidates']}:
        raise AssertionError('Broader window changed existing commands')
    rows=[]
    for r in recipes:
        chosen=[c for c in small['candidates'] if c.recipe_key==r.key]
        larger=[c for c in large['candidates'] if c.recipe_key==r.key]
        rows.append(dict(recipe=r.key,family=r.base.family,purpose=r.purpose,count=len(chosen),
            broader_count=len(larger),primary_unit=r.base.blocks[1].primary.unit,
            work_seconds=(min(c.work_seconds.estimate for c in chosen),max(c.work_seconds.estimate for c in chosen)),
            rep_counts=sorted({c.repetitions for c in chosen}),
            primary_targets=sorted({c.prescription.request.blocks[1].primary.amount.estimate for c in chosen})))
    candidates=[]
    for c in large['candidates']:
        p=c.prescription
        if read_quality_contract(p.definition()['numeric_contract'])!=p:
            raise AssertionError('Restart changed a command')
        if abs(fsum(s.duration_minutes for s in p.steps())*60-fsum(q.seconds.estimate for q in p.phases))>1e-8:
            raise AssertionError('Display and scored durations disagree')
        if abs(fsum(s.distance_miles for s in p.steps())-fsum(q.meters.estimate for q in p.phases)/MILE)>1e-9:
            raise AssertionError('Display and scored distance disagree')
        candidates.append(dict(recipe=c.recipe_key,purpose=c.purpose,work_seconds=asdict(c.work_seconds),
            repetitions=c.repetitions,prescription=p.definition(),key=p.key))
    parity=[]
    for old in previous:
        new=compile_quality(old.request,easy_pace=old.easy_pace,work_pace=old.work_pace)
        exact=new==old and new.key==old.key and new.definition()==old.definition()
        parity.append(dict(key=old.key,exact=exact))
        if not exact:
            raise AssertionError('Current frozen v2 command changed')
    archived_keys=[]
    for row in archive:
        restored=read_prescription(row['definition']['numeric_contract'])
        if restored.key!=row['key']:
            raise AssertionError('Earlier Q5 archive identity changed')
        archived_keys.append(dict(key=restored.key,exact=True))
    progression=adapt_v2(next(p for p in previous if p.request.family=='progression'))
    gradual=progression_options(progression,(3.2,3.5,3.8,4.2,4.4),window=window(1,3600,'gradual'))
    if any(c.prescription.request.blocks!=progression.request.blocks for c in gradual):
        raise AssertionError('Progression ramp changed')
    examples=[]
    for key,n,target in (('threshold_repeats',3,300),('aerobic_timed',4,180),
                         ('pickups_distance',6,400),('threshold_continuous',1,900)):
        c=next(c for c in large['candidates'] if c.recipe_key==key and c.repetitions==n
            and c.prescription.request.blocks[1].primary.amount.estimate==target)
        examples.append(dict(recipe=key,miles=c.prescription.miles,work_seconds=c.work_seconds.estimate,
            text=[s.instruction for s in c.prescription.steps()],engineering_not_prescription=True))
    result=dict(protocol=protocol,manifest=manifest,recipes=rows,candidates=candidates,
        small_count=len(keys),broader_count=len(candidates),archived_parity=parity,examples=examples,
        earlier_archive_readable=archived_keys,
        progression_targets=[c.prescription.miles for c in gradual],gate=True,
        planner_influence=False,elapsed_seconds=perf_counter()-started)
    if (sha256(before.read_bytes()).hexdigest()!=input_hash
            or sha256(archive_path.read_bytes()).hexdigest()!=archive_hash):
        raise AssertionError('Frozen input changed')
    with (output/'report.json').open('x') as stream:
        json.dump(result,stream,indent=2,default=lambda v:v.isoformat(),allow_nan=False)
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.before,args.output)
    print(json.dumps({k:r[k] for k in ('recipes','small_count','broader_count','examples','progression_targets','gate','elapsed_seconds')},indent=2))


if __name__=='__main__':
    main()
