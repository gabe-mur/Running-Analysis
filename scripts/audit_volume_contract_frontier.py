"""Bounded same-calendar volume contrasts at saved lifecycle decision phases."""
from argparse import ArgumentParser
import json
from pathlib import Path
from scripts.audit_volume_contract_conflict import frozen_contrasts
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    a.output_dir.mkdir(exist_ok=False)
    indices=(1,10,17,23,26)
    paths=tuple(a.root/f'{kind}-{i:04d}.json' for i in indices for kind in ('decision','snapshot'))
    manifest=freeze_sources(a.output_dir/'sources.zip',config=load_config(),
        entrypoint='scripts.audit_volume_contract_frontier',inputs=paths)
    results=[frozen_contrasts(a.root,i) for i in indices]
    with (a.output_dir/'frontier.json').open('x') as f:
        json.dump(dict(manifest=manifest,results=results),f,indent=2)
    for r in results:
        print(json.dumps(dict(index=r['index'],as_of=r['as_of'],
            miles14=r['original']['first14_miles'],target14=r['target14'],
            count=r['alternative_count'],funded_miles=r['best_full_14_volume']['first14_miles']
            if r['best_full_14_volume'] else None,delta=r['delta'])),flush=True)


if __name__=='__main__':
    main()
