"""Read-only monitor and artifact download for the bounded Docker test."""
import json
import os
from pathlib import Path
import time

import paramiko

REMOTE = os.environ.get('BACKFILL_MONITOR_REMOTE','/data/scientific-fulltext-source-id-docker-optimize-20261113-r3')
LOCAL = Path(os.environ.get('BACKFILL_MONITOR_LOCAL',str(Path(__file__).resolve().parents[1]/'outputs/fulltext_source_id_docker_optimization_20261113')))
LOCAL.mkdir(parents=True, exist_ok=True)


def read(ftp, name):
    try:
        with ftp.open(REMOTE+'/'+name) as stream:
            return json.loads(stream.read().decode())
    except FileNotFoundError:
        return None


while True:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('192.0.2.15',username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=15)
        with client.open_sftp() as ftp:
            status = read(ftp,'run/status.json') or {}
            host = read(ftp,'run/host_monitor.json') or {}
            summary = read(ftp,'host_summary.json')
            data = {k:status[k] for k in ['at','status','elapsed_seconds','phase','phases','handled',
                'newly_filled','no_source','duplicate_keys_skipped','pending_batches',
                'new_rows_per_second','error'] if k in status}
            data['host'] = {k:host.get(k) for k in ['safe','cpu_busy_percent','iowait_percent','services_unchanged']}
            if summary:
                data['host_summary'] = summary
            (LOCAL/'status_latest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            with (LOCAL/'observations.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps(data,ensure_ascii=False)+'\n')
            # Compact progress; full phase profiles stay in the local observation file.
            printable = dict(data)
            printable['phases'] = [{k:p[k] for k in ['name','rows_per_second','newly_filled']} for p in data.get('phases',[])]
            print(json.dumps(printable,ensure_ascii=False),flush=True)
            if summary:
                for name in ['host_summary.json','container.log','supervisor.log']:
                    ftp.get(REMOTE+'/'+name,str(LOCAL/name))
                for name in ['report.json','baseline.json','journal.sqlite','patch_manifest.json',
                    'patches_for_later_sync.jsonl.gz','host_monitor_history.jsonl','services_before.json','services_after.json']:
                    try:
                        ftp.get(REMOTE+'/run/'+name,str(LOCAL/name))
                    except FileNotFoundError:
                        pass
                if summary['container_exit_code'] != 0:
                    raise SystemExit(2)
                break
    except (OSError,paramiko.SSHException) as exc:
        print('monitor reconnect: '+repr(exc),flush=True)
    finally:
        client.close()
    time.sleep(45)
