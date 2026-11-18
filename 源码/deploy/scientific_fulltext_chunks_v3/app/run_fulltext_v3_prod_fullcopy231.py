"""Resume 231 production-instance V3 copy without indexes or old-collection writes.

The ten-minute sample is kept as the first 335825 confirmed rows. This runner
scans all hexadecimal doc_id prefixes, skips only those already confirmed in
the original sample, and journals each later insert. An ambiguous result stops
the run; there is no blind insert retry or automatic cutover.
"""
from __future__ import annotations
import hashlib, inspect, json, os, re, shlex, shutil, sqlite3, threading, time, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
try:
    import paramiko
except ImportError:
    paramiko=None
import requests
from pymilvus import MilvusClient
from pymilvus.client.grpc_handler import GrpcHandler
from benchmark_fulltext_source_id_isolated231_continuous import (
    SOURCE_COLLECTION, FIELD, SOLR, SOLR_FIELD, EXPECTED_SOURCE_COUNT,
    canonical, duplicate_counts, patch_rows, plain)
from fulltext_source_id_safe_common import IntegrityError, now, original_hash, valid_source

PROJECT=Path(__file__).resolve().parents[1]
OLD=Path(os.environ.get('V3_OLD_ROOT',str(PROJECT/'outputs/fulltext_v3_prod_10min_231_20261116_140301')))
ROOT=Path(os.environ.get('V3_WORK_ROOT',str(PROJECT/'outputs/fulltext_v3_prod_fullcopy_231_20261116')))
URI='http://192.0.2.15:19533'
TARGET='scientific_fulltext_chunks_v3'
BASE_COUNT=335825
MAX_WORKERS=max(1,min(4,int(os.environ.get('V3_MAX_WORKERS','4'))))
BATCH_SIZE=max(100,min(1000,int(os.environ.get('V3_BATCH_SIZE','200'))))
# Prefix shard: V3_PREFIXES selects which hex doc_id prefixes this process
# owns (default: all sixteen, single-process mode). Each shard gets its own
# lock/status/resources files; every shard shares the same journal.sqlite
# (WAL mode, single-writer locking) and the same per-prefix cursors, so no
# prefix can ever receive concurrent writers.
PREFIXES=tuple(os.environ.get('V3_PREFIXES','0123456789abcdef'))
SHARD_TAG=''.join(PREFIXES)
FULL_TAG='0123456789abcdef'
STATUS_NAME='status.json' if SHARD_TAG==FULL_TAG else f'status_{SHARD_TAG}.json'
LOCK_NAME='scientific_fulltext_chunks_v3.lock' if SHARD_TAG==FULL_TAG else f'scientific_fulltext_chunks_v3_{SHARD_TAG}.lock'
RESOURCES_LATEST_NAME='resources_latest.json' if SHARD_TAG==FULL_TAG else f'resources_latest_{SHARD_TAG}.json'
RESOURCES_LOG_NAME='resources.jsonl' if SHARD_TAG==FULL_TAG else f'resources_{SHARD_TAG}.jsonl'

REMOTE=r'''
import json,os,time,urllib.request
from pathlib import Path
def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
a=cpu();time.sleep(.5);b=cpu();d=[y-x for x,y in zip(a,b)]
m={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));h={}
for p in (9093,9094,9095):
 try:h[str(p)]=op.open('http://127.0.0.1:%d/healthz'%p,timeout=3).status
 except Exception as e:h[str(p)]=str(e)
print(json.dumps({'cpu_percent':100*(sum(d)-d[3]-d[4])/max(sum(d),1),
 'available_memory':m.get('MemAvailable',0),'health':h,
 'data_free':os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize}))
'''

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

class StageTimings:
    """Small in-memory batch counters, published only with the status heartbeat."""
    def __init__(self):
        self.lock=threading.Lock();self.prefixes={};self.previous={}

    def _prefix(self,prefix):
        if prefix not in self.prefixes:
            self.prefixes[prefix]={'started':time.perf_counter(),
                'stages':{},'confirmed_rows':0,'confirmed_batches':0}
        return self.prefixes[prefix]

    def record(self,prefix,stage,started,failed=False):
        elapsed=time.perf_counter()-started
        with self.lock:
            profile=self._prefix(prefix);profile['started']=min(profile['started'],started)
            item=profile['stages'].setdefault(stage,
                {'seconds':0.,'calls':0,'failed_calls':0})
            item['seconds']+=elapsed;item['calls']+=1;item['failed_calls']+=int(failed)

    def confirmed(self,prefix,rows):
        with self.lock:
            item=self._prefix(prefix);item['confirmed_rows']+=rows;item['confirmed_batches']+=1

    def snapshot(self):
        at=time.perf_counter();out={}
        with self.lock:
            for prefix,item in self.prefixes.items():
                elapsed=max(0.,at-item['started']);stages={k:dict(v) for k,v in item['stages'].items()}
                old=self.previous.get(prefix)
                interval=max(0.,at-old['at']) if old else elapsed
                window_rows=item['confirmed_rows']-old['rows'] if old else item['confirmed_rows']
                window={k:{'seconds':v['seconds']-(old['stages'].get(k,{}).get('seconds',0.) if old else 0.),
                           'calls':v['calls']-(old['stages'].get(k,{}).get('calls',0) if old else 0),
                           'failed_calls':v['failed_calls']-(old['stages'].get(k,{}).get('failed_calls',0) if old else 0)}
                        for k,v in stages.items()}
                out[prefix]={'elapsed_seconds':elapsed,'confirmed_rows':item['confirmed_rows'],
                    'confirmed_batches':item['confirmed_batches'],'stages':stages,
                    'window_seconds':interval,'window_confirmed_rows':window_rows,
                    'window_rows_per_second':window_rows/interval if interval>0 else 0.,
                    'window_stages':window}
                self.previous[prefix]={'at':at,'rows':item['confirmed_rows'],'stages':stages}
        return out

class Guard:
    def __init__(self,stop_event=None):
        self.lock=threading.Lock();self.last=0;self.value=None
        self.stop_event=stop_event or threading.Event()
        self.on_state=None
        self.on_231=os.environ.get('V3_MONITOR_LOCAL')=='1'
        self.milvus_pid=int(os.environ.get('V3_MILVUS_PID','0'))
        self.milvus_start=None
        self.ssh=None
        if not self.on_231:
            if paramiko is None:raise RuntimeError('Local SSH monitor requires paramiko')
            self.ssh=paramiko.SSHClient();self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect('192.0.2.15',username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=12)

    def native_resources(self):
        if not self.milvus_pid:return {}
        proc=Path('/proc')/str(self.milvus_pid)
        try:
            command=(proc/'cmdline').read_bytes()
            if b'milvus' not in command or b'standalone' not in command:
                raise RuntimeError('Configured Milvus PID does not identify the expected standalone process')
            start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            if self.milvus_start is not None and start!=self.milvus_start:
                raise RuntimeError('Milvus process changed; manual recovery is required')
            self.milvus_start=start
            with (proc/'maps').open() as f:maps=sum(1 for _ in f)
            status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (proc/'status').read_text().splitlines() if ':' in line}
            return {'milvus_pid':self.milvus_pid,'milvus_map_count':maps,
                    'milvus_map_limit':int(Path('/proc/sys/vm/max_map_count').read_text()),
                    'milvus_threads':int(status['Threads']),
                    'milvus_rss_bytes':int(status['VmRSS'].split()[0])*1024,
                    'milvus_virtual_bytes':int(status['VmSize'].split()[0])*1024}
        except (FileNotFoundError,ProcessLookupError) as exc:
            raise RuntimeError('Milvus process disappeared; copy paused without restarting the database') from exc

    def sample(self):
        if self.on_231:
            def cpu():return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            a=cpu();time.sleep(.5);b=cpu();d=[y-x for x,y in zip(a,b)]
            mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
            op=urllib.request.build_opener(urllib.request.ProxyHandler({}));health={}
            for port in (9093,9094,9095):
                try:health[str(port)]=op.open('http://127.0.0.1:%d/healthz'%port,timeout=3).status
                except Exception as exc:health[str(port)]=str(exc)
            value={'cpu_percent':100*(sum(d)-d[3]-d[4])/max(sum(d),1),
                   'available_memory':mem.get('MemAvailable',0),'health':health,
                   'data_free':shutil.disk_usage(ROOT).free}
            value.update(self.native_resources())
        else:
            _,out,err=self.ssh.exec_command(shlex.join(['python3','-c',REMOTE]),timeout=20)
            body=out.read().decode();error=err.read().decode()
            if out.channel.recv_exit_status():raise RuntimeError(error[-800:])
            value=json.loads(body)
        value['at']=now()
        atomic_json(ROOT/RESOURCES_LATEST_NAME,value)
        with (ROOT/RESOURCES_LOG_NAME).open('a',encoding='utf-8') as f:f.write(canonical(value)+'\n')
        return value

    @staticmethod
    def pressure_reasons(value):
        reasons=[]
        # An old deployment may still supply 90; recovery never exceeds 65%.
        cpu_limit=min(65.,float(os.environ.get('V3_GUARD_CPU_MAX','65')))
        if value['cpu_percent']>=cpu_limit:reasons.append('host CPU above %.1f%%'%cpu_limit)
        if value['available_memory']<200*1024**3:reasons.append('host available memory below 200 GiB')
        if value['data_free']<1024**4:reasons.append('data disk reserve below 1 TiB')
        if value.get('milvus_map_count',0)>=.70*value.get('milvus_map_limit',float('inf')):
            reasons.append('Milvus memory mappings above 70% of the host limit')
        if value.get('milvus_threads',0)>=1024:reasons.append('Milvus native threads above 1024')
        if any(value['health'].get(str(p))!=200 for p in (9093,9094,9095)):
            reasons.append('Milvus health endpoint temporarily unavailable')
        return reasons

    def check(self):
        while True:
            if self.stop_event.is_set() or (ROOT/'STOP_REQUESTED').exists():
                raise RuntimeError('Copy stop requested; existing data preserved')
            if shutil.disk_usage(ROOT).free<10*1024**3:
                raise RuntimeError('Local journal disk reserve below 10 GiB')
            with self.lock:
                if time.monotonic()-self.last>15:
                    self.value=self.sample();self.last=time.monotonic()
                value=self.value
            reasons=self.pressure_reasons(value)
            # Call outside the guard lock: Runner.status takes its own DB lock.
            if self.on_state:self.on_state(reasons,value)
            if not reasons:return
            # Wait without changing a service, and recheck stop requests often.
            self.stop_event.wait(1)

    def close(self):
        if self.ssh:self.ssh.close()

class Runner:
    def __init__(self):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.stop=threading.Event();self.guard=Guard(self.stop)
        self.timings=StageTimings()
        self.heartbeat_stop=threading.Event();self.heartbeat_thread=None
        self.db=sqlite3.connect(ROOT/'journal.sqlite',check_same_thread=False,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS batches(seq INTEGER PRIMARY KEY AUTOINCREMENT,prefix TEXT,
            first_id TEXT,last_id TEXT,status TEXT,ids TEXT,rows INTEGER,added INTEGER,
            missing INTEGER,original_hash_aggregate TEXT,submitted_at TEXT,confirmed_at TEXT);
          CREATE TABLE IF NOT EXISTS cursors(prefix TEXT PRIMARY KEY,last_id TEXT,scanned INTEGER DEFAULT 0,finished INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS duplicate_keys(id TEXT PRIMARY KEY,physical_count INTEGER);
        ''');self.db.commit()
        self.old_db=sqlite3.connect('file:'+OLD.joinpath('audit.sqlite').as_posix()+'?mode=ro',uri=True,check_same_thread=False)
        self.state={'started_at':now(),'pid':os.getpid(),'status':'preflight',
            'source_uri':URI,'target_uri':URI,'source_collection':SOURCE_COLLECTION,
            'target_collection':TARGET,'base_confirmed_rows':BASE_COUNT,
            'indexes_created':False,'production_cutover':False,'source_writes':0,
            'shard':SHARD_TAG,'shard_prefixes':PREFIXES,'max_workers':MAX_WORKERS,'batch_size':BATCH_SIZE,
            'all_source_prefixes':PREFIXES,
            'limitation':'Historical duplicate physical versions are recorded and skipped in V3; full lossless completion is not yet proven.'}
        self.guard.on_state=self.resource_state
        self.status()

    def resource_state(self,reasons,value):
        with self.lock:
            self.state['resources']=value
            if reasons:
                if self.state.get('status')!='waiting_resource':
                    self.state['status_before_resource_wait']=self.state.get('status','preflight')
                self.state.update(status='waiting_resource',wait_reason='; '.join(reasons))
            elif self.state.get('status')=='waiting_resource':
                self.state['status']=self.state.pop('status_before_resource_wait','running_no_index_copy')
                self.state.pop('wait_reason',None)

    def heartbeat(self):
        while not self.heartbeat_stop.wait(30):
            try:self.status()
            except Exception as exc:
                with self.lock:self.state['error']='Status heartbeat failed: '+str(exc)
                self.stop.set();return

    def status(self):
        with self.lock:
            confirmed=self.db.execute("SELECT coalesce(sum(rows),0),coalesce(sum(added),0),coalesce(sum(missing),0) FROM batches WHERE status='confirmed'").fetchone()
            self.state.update(updated_at=now(),confirmed_new_rows=confirmed[0],
                confirmed_new_source_ids=confirmed[1],confirmed_missing_source_ids=confirmed[2],
                target_expected_count=BASE_COUNT+confirmed[0],
                duplicate_keys_recorded=self.db.execute('SELECT count(*) FROM duplicate_keys').fetchone()[0],
                prefixes_finished=self.db.execute('SELECT count(*) FROM cursors WHERE finished=1').fetchone()[0],
                uncertain_batches=self.db.execute("SELECT count(*) FROM batches WHERE status='submitted'").fetchone()[0],
                stage_timings=self.timings.snapshot())
            atomic_json(ROOT/STATUS_NAME,self.state)

    def preflight(self):
        self.guard.check()
        with self.lock:
            marks=','.join('?' for _ in PREFIXES)
            pending=self.db.execute(f"SELECT count(*) FROM batches WHERE status='submitted' AND prefix IN ({marks})",PREFIXES).fetchone()[0]
        if pending:raise IntegrityError('Ambiguous submitted insert must be manually reconciled before any resume')
        source=MilvusClient(uri=URI,db_name='default',timeout=20)
        try:
            if TARGET not in source.list_collections():raise IntegrityError('Existing V3 target not found')
            if source.list_indexes(TARGET):raise IntegrityError('V3 has indexes; no-index runner refuses writes')
            n=int(source.query(SOURCE_COLLECTION,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            if n!=EXPECTED_SOURCE_COUNT:raise IntegrityError('Source count changed')
            actual=int(source.get_collection_stats(TARGET)['row_count'])
            expected=self.state['target_expected_count']
            if SHARD_TAG==FULL_TAG:
                if actual!=expected:
                    raise IntegrityError(f'Target count {actual} differs from confirmed journal {expected}')
            else:
                # Sharded mode: row_count does not include unflushed growing
                # segments, and sibling shards may hold just-inserted rows not
                # yet confirmed in the journal. Flush settles segment stats;
                # afterwards every journaled row must be physically present
                # while in-flight sibling inserts may only add rows.
                source.flush(TARGET,timeout=600)
                actual=int(source.get_collection_stats(TARGET)['row_count'])
                if actual<expected:
                    raise IntegrityError(f'Target count {actual} below confirmed journal {expected}')
            self.fields=[x['name'] for x in source.describe_collection(SOURCE_COLLECTION)['fields']]
            self.original_fields=[x for x in self.fields if x!=FIELD]
            target_fields=[x['name'] for x in source.describe_collection(TARGET)['fields']]
            if target_fields!=self.fields:raise IntegrityError('V3 schema differs from source')
            self.state.update(source_count=n,target_count_at_preflight=actual,status='running_no_index_copy')
            self.status()
        finally:source.close()

    def already_inserted(self,db,ids):
        if not ids:return set()
        result=set()
        for offset in range(0,len(ids),100):
            part=ids[offset:offset+100]
            result.update(r[0] for r in db.execute('SELECT id FROM expected_rows WHERE id IN ('+
                ','.join('?' for _ in part)+')',part))
        return result

    def source_ids(self,session,doc_ids):
        out={}
        ids=sorted(set(doc_ids))
        for offset in range(0,len(ids),200):
            part=ids[offset:offset+200]
            response=session.post(SOLR,data={'q':'{!terms f=id}'+','.join(part),
                'fl':'id,'+SOLR_FIELD,'rows':len(part),'wt':'json','timeAllowed':20000},timeout=(10,40))
            response.raise_for_status();body=response.json();docs=body['response']['docs']
            if body.get('responseHeader',{}).get('partialResults') or body['response']['numFound']!=len(docs) or len({d['id'] for d in docs})!=len(docs):
                raise IntegrityError('Ambiguous or partial Solr response')
            if not {d['id'] for d in docs}.issubset(part):raise IntegrityError('Solr returned unexpected document')
            found={d['id']:d.get(SOLR_FIELD) for d in docs}
            for key in part:
                value=found.get(key)
                out[key]=value if valid_source(value) else None
        return out

    def commit_cursor(self,prefix,last_id,scanned,finished=False):
        self.db.execute('INSERT INTO cursors(prefix,last_id,scanned,finished) VALUES(?,?,?,?) ON CONFLICT(prefix) DO UPDATE SET last_id=excluded.last_id,scanned=cursors.scanned+excluded.scanned,finished=excluded.finished',
                        (prefix,last_id,scanned,int(finished)))

    def process_batch(self,prefix,rows,source,target,session,old_db):
        if not rows:return
        started=time.perf_counter();self.guard.check();self.timings.record(prefix,'guard',started)
        all_ids=[r['id'] for r in rows]
        if all_ids!=sorted(all_ids) or len(all_ids)!=len(set(all_ids)):
            raise IntegrityError('Source iterator returned out-of-order or repeated IDs')
        started=time.perf_counter();existing=self.already_inserted(old_db,all_ids)
        rows=[r for r in rows if r['id'] not in existing]
        self.timings.record(prefix,'original_audit_lookup',started)
        started=time.perf_counter();duplicates=duplicate_counts(source,[r['id'] for r in rows])
        if duplicates:rows=[r for r in rows if r['id'] not in duplicates]
        self.timings.record(prefix,'duplicate_check',started)
        started=time.perf_counter();values=self.source_ids(session,[r['doc_id'] for r in rows])
        self.timings.record(prefix,'solr',started)
        started=time.perf_counter()
        original_digest=hashlib.sha256()
        for r in rows:original_digest.update(bytes.fromhex(original_hash(r,self.original_fields)))
        audit,added,missing=patch_rows(rows,values,self.original_fields)
        self.timings.record(prefix,'hash_patch',started)
        started=time.perf_counter()
        with self.lock:
            for key,n in duplicates.items():
                self.db.execute('INSERT OR IGNORE INTO duplicate_keys(id,physical_count) VALUES(?,?)',(key,n))
            if not rows:
                self.commit_cursor(prefix,all_ids[-1],len(all_ids));self.db.commit()
                self.timings.record(prefix,'sqlite_prepare',started);return
            self.db.execute('INSERT INTO batches(prefix,first_id,last_id,status,ids,rows,added,missing,original_hash_aggregate) VALUES(?,?,?,?,?,?,?,?,?)',
                (prefix,all_ids[0],all_ids[-1],'prepared',canonical([r['id'] for r in rows]),len(rows),added,missing,original_digest.hexdigest()))
            seq=self.db.execute('SELECT last_insert_rowid()').fetchone()[0];self.db.commit()
            self.db.execute('UPDATE batches SET status=?,submitted_at=? WHERE seq=?',('submitted',now(),seq));self.db.commit()
        self.timings.record(prefix,'sqlite_prepare',started)
        # The installed SDK retries ambiguous Insert by default; explicitly use
        # the raw single RPC once. Any exception leaves a submitted journal row
        # and stops the runner. No blind retry is permitted for duplicate PKs.
        raw=inspect.unwrap(GrpcHandler.insert_rows)
        started=time.perf_counter();failed=True
        try:
            result=raw(target._get_connection(),TARGET,rows,partition_name='',timeout=120,
                       context=target._generate_call_context())
            failed=False
        finally:self.timings.record(prefix,'insert_rpc',started,failed=failed)
        if result.insert_count!=len(rows):raise IntegrityError('Confirmed insert count mismatch')
        started=time.perf_counter()
        with self.lock:
            self.db.execute('UPDATE batches SET status=?,confirmed_at=? WHERE seq=?',('confirmed',now(),seq))
            self.commit_cursor(prefix,all_ids[-1],len(all_ids));self.db.commit()
            # The heartbeat publishes aggregates every 30s, not every 30 batches.
        self.timings.record(prefix,'sqlite_confirm',started)
        self.timings.confirmed(prefix,len(rows))

    def requested_batch_size(self,current=None):
        try:
            control=json.loads((ROOT/'tuning.json').read_text(encoding='utf-8'))
            size=max(100,min(1000,int(control.get('batch_size',BATCH_SIZE))))
        except FileNotFoundError:size=BATCH_SIZE
        except (OSError,ValueError,TypeError,AttributeError) as exc:
            with self.lock:self.state['tuning_error']=str(exc)
            return current if current is not None else BATCH_SIZE
        with self.lock:self.state.pop('tuning_error',None)
        return size

    def open_prefix_iterator(self,source,prefix,batch_size):
        # Always read the committed cursor, including on a live batch-size change.
        with self.lock:
            current=self.db.execute('SELECT last_id FROM cursors WHERE prefix=?',(prefix,)).fetchone()
        high=hex(int(prefix,16)+1)[2:] if prefix!='f' else 'g'
        expr=f'doc_id >= "{prefix}" and doc_id < "{high}"'
        if current and current[0]:expr+=' and id > '+canonical(current[0])
        return source.query_iterator(SOURCE_COLLECTION,filter=expr,batch_size=batch_size,
            output_fields=self.fields,consistency_level='Strong',reduce_stop_for_best=False,timeout=90)

    def run_prefix(self,prefix):
        if self.stop.is_set():return
        source=MilvusClient(uri=URI,db_name='default',timeout=20)
        target=MilvusClient(uri=URI,db_name='default',timeout=20)
        old_db=sqlite3.connect('file:'+OLD.joinpath('audit.sqlite').as_posix()+'?mode=ro',uri=True)
        session=requests.Session();session.trust_env=False
        try:
            with self.lock:
                cursor=self.db.execute('SELECT last_id,finished FROM cursors WHERE prefix=?',(prefix,)).fetchone()
            if cursor and cursor[1]:return
            batch_size=self.requested_batch_size()
            iterator=self.open_prefix_iterator(source,prefix,batch_size)
            with self.lock:self.state.setdefault('batch_size_by_prefix',{})[prefix]=batch_size
            try:
                while not self.stop.is_set():
                    started=time.perf_counter();self.guard.check();self.timings.record(prefix,'guard',started)
                    started=time.perf_counter();failed=True
                    try:
                        rows=[plain(dict(x)) for x in iterator.next()]
                        failed=False
                    finally:self.timings.record(prefix,'source_next',started,failed=failed)
                    if not rows:break
                    self.process_batch(prefix,rows,source,target,session,old_db)
                    # Only a fully committed batch may change the read iterator.
                    wanted=self.requested_batch_size(batch_size)
                    if wanted!=batch_size and not self.stop.is_set():
                        self.guard.check()
                        iterator.close();iterator=None
                        iterator=self.open_prefix_iterator(source,prefix,wanted)
                        batch_size=wanted
                        with self.lock:self.state['batch_size_by_prefix'][prefix]=batch_size
                if not self.stop.is_set():
                    with self.lock:
                        current=self.db.execute('SELECT last_id FROM cursors WHERE prefix=?',(prefix,)).fetchone()
                        last=current[0] if current else (cursor[0] if cursor else '')
                        self.commit_cursor(prefix,last,0,True);self.db.commit();self.status()
            finally:
                if iterator is not None:iterator.close()
        finally:session.close();old_db.close();target.close();source.close()

    def run(self):
        self.heartbeat_thread=threading.Thread(target=self.heartbeat,name='status-heartbeat',daemon=True)
        self.heartbeat_thread.start()
        try:
            self.preflight()
            # Four independent prefix streams. Each prefix has a durable cursor;
            # no prefix receives concurrent writers.
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                future={pool.submit(self.run_prefix,p):p for p in PREFIXES}
                for task in as_completed(future):
                    try:task.result()
                    except BaseException:
                        self.stop.set();raise
            self.state['status']='copy_scan_finished_pending_index_and_full_verification'
            self.status()
        except BaseException as e:
            self.state.update(status='paused_preserving_all_existing_data',error=type(e).__name__+': '+str(e),paused_at=now())
            self.status();raise
        finally:
            self.heartbeat_stop.set()
            if self.heartbeat_thread:self.heartbeat_thread.join()
            self.guard.close();self.old_db.close();self.db.close()

def main():
    if os.name=='nt':
        import msvcrt
    else:
        import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/LOCK_NAME).open('a+b') as lease:
        lease.seek(0)
        if not lease.read(1):lease.write(b'0');lease.flush()
        lease.seek(0)
        if os.name=='nt':msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
        else:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:Runner().run()
        finally:
            lease.seek(0)
            if os.name=='nt':msvcrt.locking(lease.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(lease.fileno(),fcntl.LOCK_UN)

if __name__=='__main__':main()
