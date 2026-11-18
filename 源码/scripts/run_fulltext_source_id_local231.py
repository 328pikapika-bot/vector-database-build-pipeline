"""Continuous Windows-local, 231-only field patches with durable recovery.

Only id + hasSourcesysSourceId partial upserts are allowed. Resource problems
pause dispatch and recover automatically; integrity problems hold for review.
SSH is used for a shared exclusive lock and read-only resource observation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import threading
import time
import traceback

import paramiko
import requests

from benchmark_fulltext_source_id_single231 import Benchmark, TARGET
from benchmark_fulltext_source_id_100 import COLLECTION, FIELD, HOSTS, connect, snapshot
from fulltext_source_id_safe_common import IntegrityError, decode, encode, now, original_hash

PROJECT = Path(__file__).resolve().parents[1]
PREVIOUS = PROJECT/'outputs/fulltext_source_id_docker_adaptive_20261113_r2'
EXPECTED_COUNT = 195893017
DEFAULT_WORKERS = 2
DEFAULT_BATCH_SIZE = 2000
REMOTE_LOCK = '/data/scientific-fulltext-source-id-backfill-20261113/runner.lock'

# Read only: no files, configuration or existing service state are modified.
HOST_SAMPLE = r'''
import json,os,sys,urllib.request,subprocess
from pathlib import Path
a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
result={'cpu':[sum(a),a[3],a[4]],'available_ram_bytes':int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024,
 'disk_free_bytes':{p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/','/data']}}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
m=opener.open('http://127.0.0.1:9094/metrics',timeout=3).read().decode()
result.update(index_tasks_pending=0,index_tasks_running=0)
for l in m.splitlines():
 if l.startswith('milvus_datacoord_task_num_in_scheduler{') and 'task_type="Index"' in l:
  if 'task_state="JobStateInProgress"' in l:result['index_tasks_running']+=int(float(l.rsplit(' ',1)[1]))
  elif 'task_state="JobStateInit"' in l:result['index_tasks_pending']+=int(float(l.rsplit(' ',1)[1]))
result['health']={}
for port in [9093,9094,9095]:
 import time
 t=time.monotonic()
 with opener.open('http://127.0.0.1:%d/healthz'%port,timeout=2) as r:
  result['health'][str(port)]={'status':r.status,'seconds':time.monotonic()-t}
if len(sys.argv)>1:
 names=json.loads(sys.argv[1])
 if not names:names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
 raw=json.loads(subprocess.check_output(['docker','inspect',*names],text=True,timeout=6))
 result['services']={c['Name'].lstrip('/'):{'started':c['State']['StartedAt'],'status':c['State']['Status'],'health':c['State'].get('Health',{}).get('Status'),'restarts':c['RestartCount']} for c in raw}
print(json.dumps(result))
'''


class RemoteObserver:
    def __init__(self, folder):
        self.root = Path(folder)
        self.stop = threading.Event()
        self.mutex = threading.Lock()
        self.ssh = None
        self.lock_ssh = None
        self.lock_channel = None
        self.latest = {}
        saved_services = self.root/'services_before.json'
        self.services = json.loads(saved_services.read_text()) if saved_services.exists() else None
        self.previous_cpu = None
        self.thread = None

    def connect_ssh(self):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(TARGET, username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=12)
        c.get_transport().set_keepalive(15)
        return c

    def acquire(self):
        if self.lock_ssh:
            self.lock_ssh.close()
        self.lock_ssh = self.connect_ssh()
        # Open an existing task lock, without creating/truncating any file.
        code = "import fcntl,sys; f=open("+repr(REMOTE_LOCK)+",'r+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('LOCKED',flush=True); sys.stdin.buffer.read()"
        channel = self.lock_ssh.get_transport().open_session()
        channel.settimeout(15)
        channel.exec_command('python3 -u -c '+shlex.quote(code))
        if channel.recv(1024).strip() != b'LOCKED':
            raise RuntimeError('Could not acquire the shared runner lock')
        self.lock_channel = channel

    def lock_alive(self):
        return bool(self.lock_ssh and self.lock_ssh.get_transport() and
            self.lock_ssh.get_transport().is_active() and self.lock_channel and
            not self.lock_channel.closed and not self.lock_channel.exit_status_ready())

    def sample(self, services=False):
        if not self.ssh or not self.ssh.get_transport() or not self.ssh.get_transport().is_active():
            self.ssh = self.connect_ssh()
        cmd = 'python3 -c '+shlex.quote(HOST_SAMPLE)
        if services:
            cmd += ' '+shlex.quote(json.dumps(list(self.services) if self.services else []))
        _,out,err = self.ssh.exec_command(cmd,timeout=15)
        raw = out.read().decode()
        error = err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError('Read-only observer failed: '+error[-500:])
        result = json.loads(raw)
        cpu = result.pop('cpu')
        if self.previous_cpu:
            delta = max(cpu[0]-self.previous_cpu[0],1)
            result['cpu_busy_percent'] = 100*(1-(cpu[1]-self.previous_cpu[1])/delta)
            result['iowait_percent'] = 100*(cpu[2]-self.previous_cpu[2])/delta
        else:
            result['cpu_busy_percent'] = 0
            result['iowait_percent'] = 0
        self.previous_cpu = cpu
        if 'services' in result:
            if self.services is None:
                self.services = result['services']
                (self.root/'services_before.json').write_text(json.dumps(self.services,indent=2))
            result['services_unchanged'] = result.pop('services') == self.services
        else:
            with self.mutex:
                result['services_unchanged'] = self.latest.get('services_unchanged',False)
        result.update(timestamp=time.time(),lock_alive=self.lock_alive())
        with self.mutex:
            self.latest = result
        with (self.root/'host_observations.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(result)+'\n')

    def loop(self):
        i = 0
        while not self.stop.wait(3):
            try:
                if not self.lock_alive():
                    self.acquire()
                self.sample(services=i%5==0)
                i += 1
            except Exception as exc:
                with self.mutex:
                    self.latest = {'timestamp':time.time(),'error':repr(exc),'lock_alive':self.lock_alive()}
                if self.ssh:
                    self.ssh.close()
                    self.ssh = None

    def start(self):
        self.acquire()
        self.sample(services=True)
        time.sleep(2)
        self.sample()
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def read(self):
        with self.mutex:
            return dict(self.latest)


class PressureController:
    """Bounded aggregate pacing with host-pressure protection and auto recovery.

    Index jobs are expected Milvus maintenance. Their count never hard-gates
    writes; an existing backlog caps the next write rate so it can drain.
    """
    def __init__(self, initial_rate=480, max_rate=480):
        self.rate = initial_rate
        self.max_rate = max_rate
        self.quiet_rate = min(initial_rate,max_rate)
        self.next_write = 0
        self.resume_at = 0
        self.cpu_paused = False
        self.mutex = threading.Lock()

    def decision(self, host, rows, monotonic_now, wall_now):
        with self.mutex:
            reason = None
            if not host or wall_now-host.get('timestamp',0)>15 or host.get('error'):
                reason = 'monitor_unavailable'
            elif not host.get('lock_alive'):
                reason = 'shared_lock_unavailable'
            elif not host.get('services_unchanged',False):
                reason = 'service_state_changed'
            elif host.get('available_ram_bytes',0)<250*1024**3:
                reason = 'low_available_memory'
            elif host['disk_free_bytes']['/']<40*1024**3 or host['disk_free_bytes']['/data']<1024**4:
                reason = 'low_disk_space'
            elif any(h['status']!=200 or h['seconds']>0.25 for h in host['health'].values()):
                reason = 'service_response_slow'
            else:
                cpu = host['cpu_busy_percent']
                if cpu>80 or host['iowait_percent']>5:
                    self.cpu_paused = True
                elif cpu<=55 and host['iowait_percent']<=3:
                    self.cpu_paused = False
                if self.cpu_paused:
                    reason = 'host_pressure'
                # Avoid bursts that outrun the index builder and lead to long
                # CPU-only pauses. Index work limits pacing, never availability.
                cpu_budget = 480 if cpu<55 else (400 if cpu<70 else 240)
                pending = host.get('index_tasks_pending',0)
                index_budget = (40 if pending>=1000 else 80 if pending>=500
                    else 160 if pending>=200 else 240 if pending>=50 else 480)
                budget = min(cpu_budget,index_budget)
                if reason is None:
                    self.rate = min(self.max_rate,self.quiet_rate*budget/480)
            if reason:
                self.resume_at = max(self.resume_at,monotonic_now+20)
                return False,reason,2
            if monotonic_now<self.resume_at:
                return False,'cooldown',min(2,self.resume_at-monotonic_now)
            if rows==0:
                # Dispatch uses the same pressure rules without reserving writes.
                return True,'ready',0
            if monotonic_now<self.next_write:
                return False,'rate_pacing',min(1,self.next_write-monotonic_now)
            self.next_write = monotonic_now+rows/self.rate
            return True,'ready',0


class GuardedClient:
    def __init__(self, runner, client):
        self.runner = runner
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client,name)

    def upsert(self, collection, data, partial_update=False, **kwargs):
        if collection != COLLECTION or partial_update is not True:
            raise IntegrityError('Only the authorized partial update is allowed')
        if any(set(row)!={'id',FIELD} for row in data):
            raise IntegrityError('Unexpected mutation fields')
        started = time.monotonic()
        self.runner.wait_for_capacity(len(data))
        capacity_at = time.monotonic()
        # A pressure wait can be long. Recheck uniqueness after that wait, as
        # close as possible to mutation, so duplicate versions are preserved.
        ids = [row['id'] for row in data]
        if len(set(ids))!=len(ids) or self.runner.key_count(self.client,ids)!=len(ids):
            raise IntegrityError('Refusing to merge duplicate versions after capacity wait')
        guarded_at = time.monotonic()
        result = self.client.upsert(collection,data=data,partial_update=True,**kwargs)
        self.last_write_timing = {'resource_and_rate_wait':capacity_at-started,
            'prewrite_duplicate_guard':guarded_at-capacity_at,
            'actual_upsert_rpc':time.monotonic()-guarded_at}
        return result


def verify_protected_inventory(current, baseline):
    """Protect the original collections while allowing unrelated new ones."""
    for host, collections in baseline.items():
        for name, expected in collections.items():
            if current.get(host,{}).get(name)!=expected:
                raise IntegrityError('Protected collection inventory/schema/count changed: '+host+'/'+name)


class LocalRunner(Benchmark):
    def __init__(self, folder, batch_size=DEFAULT_BATCH_SIZE,
                 workers=DEFAULT_WORKERS, duration_seconds=None):
        if not 1 <= batch_size <= 4000:
            raise ValueError('batch_size must be between 1 and 4000')
        if not 1 <= workers <= 4:
            raise ValueError('workers must be between 1 and 4')
        if duration_seconds is not None and not 1 <= duration_seconds <= 3600:
            raise ValueError('duration_seconds must be between 1 and 3600')
        self.batch_size = batch_size
        self.workers = workers
        self.duration_seconds = duration_seconds
        self.dispatch_deadline = None
        self.dispatch_deadline_at = None
        self.dispatch_stop_reason = None
        self.draining_inflight = False
        self.trial_baseline_counts = None
        self.trial_final_counts = None
        self.trial_started_monotonic = None
        self.trial_started_at = None
        self.trial_finished_at = None
        self.trial_elapsed_seconds = None
        self.root = Path(folder).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.process_lock = open(self.root/'local_runner.lock','a+b')
        if os.name == 'nt':
            import msvcrt
            self.process_lock.seek(0)
            if self.process_lock.read(1)==b'':
                self.process_lock.write(b'0')
                self.process_lock.flush()
            self.process_lock.seek(0)
            msvcrt.locking(self.process_lock.fileno(),msvcrt.LK_NBLCK,1)
        self.db = sqlite3.connect(self.root/'journal.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS batches(seq INTEGER PRIMARY KEY,phase TEXT,status TEXT,payload BLOB,result TEXT);
            CREATE TABLE IF NOT EXISTS source_cache(doc_id TEXT PRIMARY KEY,value TEXT,reason TEXT);
            CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);
            CREATE INDEX IF NOT EXISTS batch_finished_at ON batches(json_extract(result,'$.finished_at')) WHERE status='completed';''')
        self.db.commit()
        self.db_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.gate_lock = threading.Lock()
        self.observer = RemoteObserver(self.root)
        self.controller = PressureController()
        self.previous = json.loads((PREVIOUS/'report.json').read_text())
        self.baseline = {}
        self.before_filled = {}
        self.fields = []
        self.original_fields = []
        self.main = None
        self.local = threading.local()
        self.all_worker_clients = []
        self.cursor = self.previous['end_cursor']
        self.bucket = int(self.cursor[:3],16)
        self.iterator = None
        self.buffer = []
        self.session = requests.Session()
        self.session.trust_env = False
        self.counts = dict.fromkeys(['handled','newly_filled','already_filled','no_source',
            'duplicate_keys_skipped','duplicate_extra_versions_preserved','completed_batches','retry_events'],0)
        self.start = time.monotonic()
        self.state = 'starting'
        self.reason = None
        self.started_at = None
        self.initial_elapsed = 0
        self.integrity_hold = None
        self.manual_stop = threading.Event()
        self.last_checkpoint = self.cursor
        self.last_checkpoint_seq = 0
        self.last_guard = 0
        self.initialized = False

    def save_state(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO state VALUES(?,?)',(key,json.dumps(value)))
        self.db.commit()

    def rebuild_counts(self):
        totals = dict.fromkeys(self.counts,0)
        for payload,result in self.db.execute("SELECT payload,result FROM batches WHERE status='completed' ORDER BY seq"):
            r = json.loads(result)
            for key,source in [('handled','rows'),('newly_filled','newly_filled'),('already_filled','already_filled'),
                ('no_source','no_source'),('duplicate_keys_skipped','duplicates'),('duplicate_extra_versions_preserved','extra_duplicate_versions')]:
                totals[key] += r[source]
            totals['completed_batches'] += 1
        totals['retry_events'] = self.counts.get('retry_events',0)
        self.counts = totals

    def status(self, state=None, **kwargs):
        if state:
            self.state = state
        with self.status_lock,self.db_lock:
            n = dict(self.counts)
            pending = self.db.execute("SELECT COUNT(*) FROM batches WHERE status!='completed'").fetchone()[0]
            elapsed = self.initial_elapsed+time.monotonic()-self.start
            # Use timestamps, so automatic waits are included in hourly rates.
            from datetime import datetime,timedelta,timezone
            cutoff = (datetime.now(timezone(timedelta(hours=8)))-timedelta(hours=1)).isoformat(timespec='seconds')
            last_hour = self.db.execute("SELECT COALESCE(SUM(json_extract(result,'$.newly_filled')),0) FROM batches WHERE status='completed' AND json_extract(result,'$.finished_at')>=?",(cutoff,)).fetchone()[0]
            avg = n['newly_filled']/max(elapsed,1)
            wall_elapsed = time.time()-datetime.fromisoformat(self.started_at).timestamp() if self.started_at else elapsed
            recent_rate = last_hour/min(3600,max(wall_elapsed,1))
            filled = self.before_filled.get(TARGET,self.previous['filled_after'][TARGET])+n['newly_filled']
            remaining = max(0,EXPECTED_COUNT-filled)
            data = {'at':now(),'status':self.state,'reason':self.reason,'pid':os.getpid(),
                'execution_host':'local_windows','write_host':TARGET,'read_only_hosts':list(HOSTS[1:]),
                'collection':COLLECTION,'field':FIELD,'started_at':self.started_at,
                'elapsed_seconds':round(elapsed,2),**n,'pending_batches':pending,
                'filled_records_estimated':filled,'remaining_records_estimated':remaining,
                'average_rows_per_second':round(avg,2),'last_hour_newly_filled':last_hour,
                'last_hour_rows_per_second':round(recent_rate,2),
                'estimated_days_remaining':remaining/recent_rate/86400 if recent_rate>0 else None,
                'rate_limit_rows_per_second':round(self.controller.rate,2),
                'batch_size':self.batch_size,'workers':self.workers,
                'trial_duration_seconds':self.duration_seconds,
                'dispatch_deadline_at':self.dispatch_deadline_at,
                'dispatch_stop_reason':self.dispatch_stop_reason,
                'trial':self.trial_metrics(n),
                'last_completed_cursor':self.last_checkpoint,'host':self.observer.read(),
                'integrity_hold':self.integrity_hold,**kwargs}
            tmp = self.root/'status.tmp'
            tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,self.root/'status.json')
            if self.initialized:
                self.save_state('elapsed',elapsed)
            return data

    def preflight(self):
        if not self.observer.thread or not self.observer.thread.is_alive():
            self.observer.start()
        self.main = connect(TARGET)
        stored = {k:json.loads(v) for k,v in self.db.execute('SELECT key,value FROM state')}
        self.started_at = stored.get('started_at',now())
        self.initial_elapsed = stored.get('elapsed',0)
        previous_baseline = json.loads((PREVIOUS/'baseline.json').read_text())['snapshots']
        for host in HOSTS:
            client = connect(host)
            try:
                self.baseline[host] = snapshot(client)
                self.before_filled[host] = int(client.query(COLLECTION,filter=FIELD+' is not null',
                    output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            finally:
                client.close()
        verify_protected_inventory(self.baseline,previous_baseline)
        if any(v[COLLECTION]['count']!=EXPECTED_COUNT for v in self.baseline.values()):
            raise IntegrityError('Target count changed since verified baseline')
        if 'initial_filled' not in stored:
            if self.before_filled != self.previous['filled_after']:
                raise IntegrityError('Unexpected external field updates')
            self.save_state('initial_filled',self.before_filled)
            self.save_state('started_at',self.started_at)
            self.save_state('checkpoint',self.cursor)
            self.save_state('checkpoint_seq',0)
            old = sqlite3.connect('file:'+str((PREVIOUS/'journal.sqlite').resolve()).replace('\\','/')+'?mode=ro',uri=True)
            for row in old.execute('SELECT doc_id,value,reason FROM source_cache'):
                self.db.execute('INSERT OR IGNORE INTO source_cache VALUES(?,?,?)',row)
            old.close()
            self.db.commit()
        else:
            live_filled = dict(self.before_filled)
            self.before_filled = stored['initial_filled']
            if any(int(self.main.query(COLLECTION,filter='',output_fields=['count(*)'],
                consistency_level='Strong',timeout=90)[0]['count(*)'])!=EXPECTED_COUNT for _ in range(1)):
                raise IntegrityError('Collection count changed')
            self.cursor = stored['checkpoint']
            self.last_checkpoint_seq = stored.get('checkpoint_seq',0)
            if self.db.execute("SELECT COUNT(*) FROM batches WHERE status!='completed'").fetchone()[0]==0:
                completed_new = self.db.execute("SELECT COALESCE(SUM(json_extract(result,'$.newly_filled')),0) FROM batches WHERE status='completed'").fetchone()[0]
                expected_filled = dict(self.before_filled)
                expected_filled[TARGET] += completed_new
                if live_filled != expected_filled:
                    raise IntegrityError('Restart filled counts differ from verified journal')
        self.bucket = int(self.cursor[:3],16)
        self.last_checkpoint = self.cursor
        self.fields = [f['name'] for f in self.baseline[TARGET][COLLECTION]['fields']]
        self.original_fields = [f for f in self.fields if f!=FIELD]
        self.rebuild_counts()
        self.initialized = True
        (self.root/'baseline.json').write_text(json.dumps({'snapshots':self.baseline,
            'initial_filled':self.before_filled,'start_cursor':self.previous['end_cursor']},indent=2))
        self.status('running')

    def worker_client(self):
        if not hasattr(self.local,'client'):
            raw = connect(TARGET)
            self.local.client = GuardedClient(self,raw)
            with self.db_lock:
                self.all_worker_clients.append(raw)
        return self.local.client

    def task(self, seq, selected, expected, reasons):
        client = self.worker_client()
        client.last_write_timing = {}
        result = super().task(seq,selected,expected,reasons)
        result['timing'].update(client.last_write_timing)
        with self.db_lock:
            self.db.execute('UPDATE batches SET result=? WHERE seq=? AND status=\'completed\'',
                (json.dumps(result),seq))
            self.db.commit()
        return result

    def wait_for_capacity(self, rows):
        # One rate gate controls all workers, including retries and recovery.
        with self.gate_lock:
            while True:
                if self.stop_reason():
                    # A submitted batch must finish safely. The deadline only
                    # closes dispatch; it never bypasses resource protection.
                    self.draining_inflight = True
                if self.integrity_hold:
                    raise IntegrityError(self.integrity_hold)
                if shutil.disk_usage(self.root).free<10*1024**3:
                    self.reason = 'local_audit_disk_low'
                    self.status('draining_inflight' if self.draining_inflight else 'waiting_for_resources')
                    time.sleep(15)
                    continue
                allowed,reason,delay = self.controller.decision(self.observer.read(),rows,time.monotonic(),time.time())
                self.reason = reason
                if allowed:
                    return
                self.status('draining_inflight' if self.draining_inflight else
                    ('waiting_for_resources' if reason!='rate_pacing' else 'running'))
                time.sleep(delay)

    def stop_reason(self):
        if (self.root/'STOP_REQUESTED').exists():
            return 'stop_requested'
        if self.dispatch_deadline is not None and time.monotonic()>=self.dispatch_deadline:
            return 'duration_elapsed'
        return None

    def stop_dispatch(self, reason):
        self.dispatch_stop_reason = reason
        self.draining_inflight = True
        self.status('draining_inflight')

    def return_unqueued_selection(self, selected, previous_cursor):
        # select() advances the in-memory cursor before Solr lookup. No journal
        # row exists yet, so put these IDs back and keep the durable checkpoint.
        self.buffer = selected+self.buffer
        self.cursor = previous_cursor

    def start_trial(self):
        if self.duration_seconds is None:
            return
        # Historical counters and recovery writes remain in the journal, but
        # do not count towards this trial. Its clock starts after recovery.
        self.trial_baseline_counts = dict(self.counts)
        self.trial_started_monotonic = time.monotonic()
        self.trial_started_at = now()
        self.dispatch_deadline = self.trial_started_monotonic+self.duration_seconds
        self.dispatch_deadline_at = datetime.fromtimestamp(
            time.time()+self.duration_seconds,timezone.utc).isoformat()

    def finish_trial(self):
        if self.trial_baseline_counts is None:
            return
        self.trial_final_counts = dict(self.counts)
        self.trial_elapsed_seconds = time.monotonic()-self.trial_started_monotonic
        self.trial_finished_at = now()

    def trial_metrics(self, counts=None):
        if self.trial_baseline_counts is None:
            return None
        current = self.trial_final_counts if self.trial_final_counts is not None else (
            self.counts if counts is None else counts)
        elapsed = self.trial_elapsed_seconds if self.trial_elapsed_seconds is not None else (
            time.monotonic()-self.trial_started_monotonic)
        added = current['newly_filled']-self.trial_baseline_counts['newly_filled']
        deltas = {key:current[key]-value for key,value in self.trial_baseline_counts.items()}
        return {'started_at':self.trial_started_at,'finished_at':self.trial_finished_at,
            'dispatch_window_seconds':self.duration_seconds,
            'elapsed_seconds_including_inflight_waits':round(elapsed,3),
            'baseline_counts':self.trial_baseline_counts,'new_counts':deltas,
            'verified_rows_per_second_including_waits':round(added/max(elapsed,0.001),3),
            'batch_size':self.batch_size,'workers':self.workers,
            'dispatch_stop_reason':self.dispatch_stop_reason}

    def recover_batch(self, seq, payload):
        if 'original_hashes' not in payload:
            selected = payload['selected']
            expected,reasons = self.fetch_source(selected)
            return self.task(seq,selected,expected,reasons)
        client = self.worker_client()
        rows = self.get_rows(client,payload['ids'])
        patches = {p['id']:p[FIELD] for p in payload['patches']}
        todo = []
        for key,row in rows.items():
            if original_hash(row,self.original_fields)!=payload['original_hashes'][key]:
                raise IntegrityError('Recovery original fields/vector mismatch')
            wanted = patches.get(key,payload['original_values'][key])
            if row.get(FIELD)==wanted:
                continue
            if key not in patches or row.get(FIELD)!=payload['original_values'][key]:
                raise IntegrityError('Recovery found a conflicting field value')
            todo.append({'id':key,FIELD:wanted})
        if todo:
            if self.key_count(client,[p['id'] for p in todo])!=len(todo):
                raise IntegrityError('Recovery duplicate primary keys')
            update = client.upsert(COLLECTION,data=todo,partial_update=True,timeout=120)
            if int(update.get('upsert_count',-1))!=len(todo):
                raise RuntimeError('Unexpected recovery update count')
        after = self.get_rows(client,payload['ids'])
        for key,row in after.items():
            if original_hash(row,self.original_fields)!=payload['original_hashes'][key] or row.get(FIELD)!=patches.get(key,payload['original_values'][key]):
                raise IntegrityError('Recovery readback validation failed')
        dup = payload['duplicate_keys']
        result = {'rows':len(payload['ids']),'newly_filled':len(patches),
            'already_filled':len(payload['ids'])-len(patches)-len(dup)-len(payload['no_source']),
            'no_source':len(payload['no_source']),'duplicates':len(dup),
            'extra_duplicate_versions':sum(v-1 for v in dup.values()),'timing':{},'finished_at':now(),
            'original_fields_and_vectors_unchanged':True,'source_values_match':True,'recovered':True}
        with self.db_lock:
            self.db.execute("UPDATE batches SET status='completed',result=? WHERE seq=?",(json.dumps(result),seq))
            self.db.commit()
        return result

    def advance_checkpoint(self):
        with self.db_lock:
            cursor = self.last_checkpoint
            seq = self.last_checkpoint_seq
            for batch_seq,status,blob in self.db.execute('SELECT seq,status,payload FROM batches WHERE seq>? ORDER BY seq',(self.last_checkpoint_seq,)):
                if status!='completed':
                    break
                payload = decode(blob)
                cursor = payload['selected'][-1]['id']
                seq = batch_seq
            self.last_checkpoint = cursor
            self.last_checkpoint_seq = seq
            self.db.execute('INSERT OR REPLACE INTO state VALUES(?,?)',('checkpoint',json.dumps(cursor)))
            self.db.execute('INSERT OR REPLACE INTO state VALUES(?,?)',('checkpoint_seq',json.dumps(seq)))
            self.db.commit()

    def recover_all(self):
        while True:
            with self.db_lock:
                pending = self.db.execute("SELECT seq,payload FROM batches WHERE status!='completed' ORDER BY seq").fetchall()
            if not pending:
                break
            for seq,blob in pending:
                try:
                    self.recover_batch(seq,decode(blob))
                    self.advance_checkpoint()
                    with self.db_lock:
                        self.rebuild_counts()
                    self.status('running')
                except IntegrityError:
                    raise
                except Exception as exc:
                    self.counts['retry_events'] += 1
                    self.reason = 'retrying_pending_batch: '+repr(exc)
                    self.status('retry_wait')
                    time.sleep(15)
                    break
        self.cursor = self.last_checkpoint
        self.bucket = int(self.cursor[:3],16)
        if self.iterator:
            self.iterator.close()
        self.iterator = None
        self.buffer = []

    def heartbeat(self):
        while not self.manual_stop.wait(10):
            try:
                self.status()
            except Exception:
                pass

    def periodic_count_guard(self):
        if time.monotonic()-self.last_guard<900:
            return
        n = int(self.main.query(COLLECTION,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
        if n!=EXPECTED_COUNT:
            raise IntegrityError('Target collection count changed')
        self.last_guard = time.monotonic()

    def export_patches(self):
        h = hashlib.sha256()
        n = 0
        with gzip.open(self.root/'patches_for_later_sync.jsonl.gz','wb') as out:
            for blob, in self.db.execute("SELECT payload FROM batches WHERE status='completed' ORDER BY seq"):
                for p in decode(blob)['patches']:
                    line = (json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                    out.write(line)
                    h.update(line)
                    n += 1
        (self.root/'patch_manifest.json').write_text(json.dumps({'rows':n,'jsonl_sha256':h.hexdigest()},indent=2))

    def run(self):
        threading.Thread(target=self.heartbeat,daemon=True).start()
        while True:
            try:
                self.preflight()
                break
            except IntegrityError as exc:
                self.integrity_hold = str(exc)
                self.reason = str(exc)
                self.status('integrity_hold')
                while True:
                    time.sleep(30)
            except Exception as exc:
                self.reason = repr(exc)
                self.status('preflight_retry_wait')
                time.sleep(15)
        self.recover_all()
        self.start_trial()
        while not self.stop_reason():
            pending = []
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    try:
                        for _ in range(100):
                            reason = self.stop_reason()
                            if reason:
                                self.stop_dispatch(reason)
                                break
                            if len(pending)>=self.workers:
                                pending.pop(0).result()
                                self.advance_checkpoint()
                            reason = self.stop_reason()
                            if reason:
                                self.stop_dispatch(reason)
                                break
                            # Capacity is checked again immediately before every write.
                            allowed,reason,_ = self.controller.decision(self.observer.read(),0,time.monotonic(),time.time())
                            if not allowed:
                                self.reason = reason
                                self.status('waiting_for_resources')
                                time.sleep(3)
                                continue
                            self.periodic_count_guard()
                            reason = self.stop_reason()
                            if reason:
                                self.stop_dispatch(reason)
                                break
                            previous_cursor = self.cursor
                            selected = self.select(self.batch_size)
                            if not selected:
                                self.reason = 'enumeration_finished'
                                self.manual_stop.set()
                                break
                            expected,reasons = self.fetch_source(selected)
                            reason = self.stop_reason()
                            if reason:
                                self.return_unqueued_selection(selected,previous_cursor)
                                self.stop_dispatch(reason)
                                break
                            with self.db_lock:
                                seq = self.db.execute("INSERT INTO batches(phase,status,payload) VALUES('continuous_local','queued',?)",
                                    (encode({'selected':selected}),)).lastrowid
                                self.db.commit()
                            pending.append(pool.submit(self.task,seq,selected,expected,reasons))
                    finally:
                        reason = self.stop_reason()
                        if reason and not self.dispatch_stop_reason:
                            self.stop_dispatch(reason)
                        errors = []
                        for future in pending:
                            try:
                                future.result()
                            except BaseException as exc:
                                errors.append(exc)
                        if errors:
                            raise errors[0]
                self.advance_checkpoint()
                self.status('draining_inflight' if self.dispatch_stop_reason else 'running')
                if self.manual_stop.is_set():
                    break
            except IntegrityError as exc:
                self.integrity_hold = str(exc)
                self.reason = str(exc)
                self.status('integrity_hold')
                while True:
                    time.sleep(30)
            except Exception as exc:
                self.reason = repr(exc)
                self.counts['retry_events'] += 1
                self.status('retry_wait')
                time.sleep(15)
                try:
                    self.recover_all()
                except IntegrityError as hold:
                    self.integrity_hold = str(hold)
                    self.reason = str(hold)
                    self.status('integrity_hold')
                    while True:
                        time.sleep(30)
        reason = self.stop_reason()
        if reason and not self.dispatch_stop_reason:
            self.stop_dispatch(reason)
        self.manual_stop.set()
        self.advance_checkpoint()
        # No submitted worker remains here. Include all dispatch, protection,
        # retries and drain time; exporting old patch logs is outside the trial.
        self.finish_trial()
        self.export_patches()
        final_state = ('completed' if self.reason=='enumeration_finished' else
            'trial_window_complete' if self.dispatch_stop_reason=='duration_elapsed' else
            'stopped_by_user')
        self.status(final_state)
        if self.trial_baseline_counts is not None:
            trial_path = self.root/('trial_report_'+str(time.time_ns())+'.json')
            trial_path.write_text(json.dumps(self.trial_metrics(),ensure_ascii=False,indent=2),encoding='utf-8')
        self.observer.stop.set()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--batch-size',type=int,default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    parser.add_argument('--duration-seconds',type=int)
    args = parser.parse_args()
    if not args.apply:
        parser.error('--apply is required')
    LocalRunner(args.workdir,batch_size=args.batch_size,workers=args.workers,
        duration_seconds=args.duration_seconds).run()
