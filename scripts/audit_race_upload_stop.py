"""Reprice the two frozen race decisions; no search, upload or live mutation."""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision


def audit(root):
    before,after=load_decision(root,1),load_decision(root,2)
    original=tuple(map(tuple,before['event']['program']))
    remaining=tuple(p for p in original if p[0] in after['dates'])
    winner=tuple(map(tuple,after['event']['program']))
    # Preserve the original opening prescription and original date range. Do
    # not substitute measured post-upload work into the pre-upload counterfactual.
    back=tuple(sorted((*[p for p in winner if p[0] in before['dates']],
        *[p for p in original if p[0]<after['state'].context.decision.isoformat()])))
    def price(case,program):
        result=case['evaluate'](program)
        return dict(cost=float(result['cost']),eligible=bool(result['eligible']))
    old=price(before,original);back_price=price(before,back)
    comparison=json.loads((root/'execution-0001.json').read_text())['comparison']
    return dict(before_snapshot=before['state'].fingerprint,after_snapshot=after['state'].fingerprint,
        scope='Exact fixed-candidate repricing; not another optimizer or daily replay',
        original=old,original_remaining_after_upload=price(after,remaining),
        selected_after_upload=price(after,winner),later_calendar_before_upload=back_price,
        cheaper_before_upload_percent=100*(old['cost']-back_price['cost'])/old['cost'],
        same_stage=before['state'].stage_key==after['state'].stage_key,
        same_scales=before['state'].context.scales==after['state'].context.scales,
        same_goal=before['state'].race_goal==after['state'].race_goal,
        budget_exhausted=after['event']['budget_exhausted'],
        local_complete=after['event']['local_neighborhood_complete'],
        comparison=comparison,
        conclusion='An admitted better calendar existed before upload; search coverage contributes. Horizon/forecast effects are not fully separated.',
        live_changed=False)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=audit(args.root)
    (args.root/'backward-stop-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('original','later_calendar_before_upload',
        'cheaper_before_upload_percent','same_stage','same_scales','same_goal','conclusion')}))
