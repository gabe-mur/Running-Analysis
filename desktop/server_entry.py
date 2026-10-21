"""Frozen loopback backend. Native launcher owns lifetime; user data is external.

No developer configuration, databases, activity files, reference corpus or live
runtime enters the bundle. --data-dir is explicit to support isolated testing.
"""
from argparse import ArgumentParser
from contextlib import closing
import json
import hashlib
import logging
import logging.handlers
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import threading
import time


def resource_root():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))


def prepare_workspace(root, resources):
    from run_analysis.privacy import private_file, private_tree
    from run_analysis.project_setup import initialize_project
    from run_analysis.config import load_config, resolve_project_path
    from run_analysis.db import connect, initialize
    private_tree(root)
    # Templates can be replaced with app updates; actual settings never are.
    for name in ('config.example.yaml','run_overrides.example.csv'):
        target=root/name
        if not target.exists():
            shutil.copyfile(resources/name,target)
            private_file(target)
    initialize_project(root)
    config=load_config(root/'config.yaml')
    database=resolve_project_path(root,config['paths']['database'])
    with connect(database) as connection:
        initialize(connection)
    return config,database


def main():
    parser=ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--ready-file',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    root=args.data_dir.expanduser().resolve()
    resources=resource_root()
    # Homebrew's OpenSSL trust paths do not exist on somebody else's Mac.
    # Ship an unmodified CA bundle/source/license, never disable verification.
    import certifi
    os.environ['SSL_CERT_FILE']=certifi.where()
    os.environ['REQUESTS_CA_BUNDLE']=certifi.where()
    from run_analysis.privacy import private_file, private_tree
    private_tree(root)
    private_tree(root/'logs')
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    handler=logging.handlers.RotatingFileHandler(root/'logs'/'app.log',maxBytes=2_000_000,backupCount=2)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(handler)
    private_file(root/'logs'/'app.log')
    from run_analysis.cli import _server_instance_lock
    from run_analysis.config import load_config, resolve_project_path
    # Own the workspace before any schema migration/setup. Never open the
    # development server's database automatically or start a reloader.
    lock_database=root/'data'/'run_analysis.sqlite'
    if (root/'config.yaml').exists():
        lock_database=resolve_project_path(root,load_config(root/'config.yaml')['paths']['database'])
    try:
        with _server_instance_lock(lock_database):
            config,database=prepare_workspace(root,resources)
            from run_analysis.web.app import create_app
            import uvicorn
            app=create_app(root)
            @app.get('/api/desktop-health',include_in_schema=False)
            def desktop_health():
                return {'application':'running-coach','pid':os.getpid(),'version':'0.2.0'}
            with closing(socket.socket(socket.AF_INET,socket.SOCK_STREAM)) as listener:
                listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                # Stable per workspace preserves browser-local view preferences;
                # bind atomically and fall back if another service owns it.
                preferred=19000+int(hashlib.sha256(str(root).encode()).hexdigest()[:8],16)%30000
                try:
                    listener.bind(('127.0.0.1',preferred))
                except OSError:
                    listener.bind(('127.0.0.1',0))
                listener.listen(128)
                port=listener.getsockname()[1]
                server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,
                    loop='asyncio',http='h11',ws='none',log_config=None,access_log=False))
                def announce():
                    while not server.started and not server.should_exit:
                        time.sleep(.05)
                    if server.started:
                        with sqlite3.connect(database) as initial:
                            empty=initial.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==0
                        payload={'url':f'http://127.0.0.1:{port}/'+('#setup' if empty else ''),'pid':os.getpid()}
                        temporary=args.ready_file.with_suffix('.tmp')
                        temporary.write_text(json.dumps(payload))
                        private_file(temporary)
                        temporary.replace(args.ready_file)
                def watch_parent():
                    while not server.should_exit:
                        try:
                            os.kill(args.parent_pid,0)
                        except ProcessLookupError:
                            server.should_exit=True
                            return
                        except PermissionError:
                            pass
                        time.sleep(1)
                threading.Thread(target=announce,daemon=True).start()
                if args.parent_pid:
                    threading.Thread(target=watch_parent,daemon=True).start()
                server.run(sockets=[listener])
    except Exception:
        logging.exception('Running Coach could not start')
        raise
    finally:
        args.ready_file.unlink(missing_ok=True)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
