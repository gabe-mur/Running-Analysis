from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root=Path(SPECPATH).parent
staging=Path(os.environ['RUNNING_COACH_BUILD'])
datas=collect_data_files('run_analysis',include_py_files=True)
datas += collect_data_files('certifi')
datas += [(str(root/'config.example.yaml'),'.'),(str(root/'run_overrides.example.csv'),'.'),
          (str(staging/'licenses'),'licenses'),(str(staging/'build-info.json'),'.')]
hidden=collect_submodules('run_analysis')+collect_submodules('uvicorn')
for name in ('fitdecode','fastapi','starlette','pydantic','uvicorn','certifi'):
    datas+=copy_metadata(name)
a=Analysis([str(root/'desktop/server_entry.py')],pathex=[str(root/'src')],
    binaries=[],datas=datas,hiddenimports=hidden,hookspath=[],runtime_hooks=[],
    excludes=['pytest','httpx','keyring','IPython','matplotlib','pandas','scipy','tkinter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='RunningCoachServer',console=True,
    strip=False,upx=False,target_arch='arm64',codesign_identity=None)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='RunningCoachServer')
