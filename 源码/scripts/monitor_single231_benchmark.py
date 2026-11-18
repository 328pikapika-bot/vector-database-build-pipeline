"""Read-only monitor of the bounded single-231 benchmark."""
import json
from pathlib import Path
import time
import os
import paramiko

root='/data/scientific-fulltext-source-id-single231-20261113'
local=Path(__file__).resolve().parents[1]/'outputs/fulltext_source_id_single231_20261113'
local.mkdir(parents=True,exist_ok=True)
client=None
try:
    while True:
        if client is None:
            client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect('192.0.2.15',username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=15)
            client.get_transport().set_keepalive(30)
        try:
            with client.open_sftp() as s:
                with s.open(root+'/status.json') as f:status=json.loads(f.read().decode())
                try:
                    with s.open('/proc/'+str(status['pid'])+'/cmdline') as f:
                        alive=b'benchmark_fulltext_source_id_single231.py' in f.read()
                except FileNotFoundError:alive=False
                status['process_alive']=alive
                if status['status']=='completed_and_stopped':
                    for name in ['report.json','baseline.json','runner.log']:
                        s.get(root+'/'+name,str(local/name))
            (local/'status_latest.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
            with (local/'observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(status)+'\n')
            keys=['at','status','elapsed_seconds','phase','handled','newly_filled','no_source','duplicate_keys_skipped',
                  'completed_batches','pending_batches','new_rows_per_second','error','process_alive']
            print(json.dumps({k:status[k] for k in keys if k in status}),flush=True)
            if status['status']=='completed_and_stopped':break
            if status['status']=='stopped_error' or not alive:raise SystemExit(2)
        except (OSError,paramiko.SSHException) as exc:
            print('MONITOR_RECONNECT',repr(exc),flush=True)
            client.close();client=None
        time.sleep(60)
finally:
    if client:client.close()
