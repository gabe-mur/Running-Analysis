"""Content-addressed local experiment provenance, including dirty source bytes."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import yaml


def freeze_sources(destination, *, config, entrypoint, inputs=()):
    """Archive source/config, not private activity files; hash explicit inputs.

    Exclusive output prevents overwriting an earlier comparator. This is not a
    release manifest or a claim that historical artifacts used today's sources.
    """
    root=Path(__file__).resolve().parents[1]
    paths=sorted(set(p for folder in ('src','scripts','tests') for p in (root/folder).rglob('*.py')) |
                 {root/'pyproject.toml'})
    sources={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest=dict(version=1,entrypoint=entrypoint,python=platform.python_version(),
        source_sha256=sources,config=config,input_sha256={str(p):sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        reproducibility='archived dirty source and configuration; activity originals hashed separately')
    encoded=json.dumps(manifest,sort_keys=True,allow_nan=False,
                       default=lambda v:v.isoformat()).encode()
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'x',compression=ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,str(p.relative_to(root)))
        archive.writestr('manifest.json',encoded)
        archive.writestr('config.yaml',yaml.safe_dump(config))
    return dict(path=str(destination),sha256=sha256(destination.read_bytes()).hexdigest(),
                manifest_sha256=sha256(encoded).hexdigest(),source_count=len(sources))
