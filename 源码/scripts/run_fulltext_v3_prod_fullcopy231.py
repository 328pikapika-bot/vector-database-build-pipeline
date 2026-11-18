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
PREFIXES=tuple('0123456789abcdef')

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

class Guard:
    def __init__(self):
        self.lock=threading.Lock();self.last=0;self.value=None
        self.on_231=os.environ.get('V3_MONITOR_LOCAL')=='1'
        self.ssh=None
        if not self.on_231:
            if paramiko is None:raise RuntimeError('Local SSH monitor requires paramiko')
            self.ssh=paramiko.SSHClient();self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect('192.0.2.15',username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=12)
    def check(self):
        if (ROOT/'STOP_REQUESTED').exists():raise RuntimeError('Local stop requested')
        if shutil.disk_usage(ROOT).free<10*1024**3:
            raise RuntimeError('Local journal disk reserve below10GiB')
        with self.lock:
            if time.monotonic()-self.last>15:
                if self.on_231:
                    def cpu():return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
                    a=cpu();time.sleep(.5);b=cpu();d=[y-x for x,y in zip(a,b)]
                    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
                    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));health={}
                    for port in (9093,9094,9095):
                        try:health[str(port)]=op.open('http://127.0.0.1:%d/healthz'%port,timeout=3).status
                        except Exception as exc:health[str(port)]=str(exc)
                    self.value={'cpu_percent':100*(sum(d)-d[3]-d[4])/max(sum(d),1),
                        'available_memory':mem.get('MemAvailable',0),'health':health,
                        'data_free':shutil.disk_usage(ROOT).free}
                else:
                    _,out,err=self.ssh.exec_command(shlex.join(['python3','-c',REMOTE]),timeout=20)
                    body=out.read().decode();error=err.read().decode()
                    if out.channel.recv_exit_status():raise RuntimeError(error[-800:])
                    self.value=json.loads(body)
                self.value['at']=now();self.last=time.monotonic()
                atomic_json(ROOT/'resources_latest.json',self.value)
                with (ROOT/'resources.jsonl').open('a',encoding='utf-8') as f:f.write(canonical(self.value)+'\n')
            v=self.value
        if v['cpu_percent']>=65 or v['available_memory']<200*1024**3 or v['data_free']<1024**4:
            raise RuntimeError('Host resource safety threshold reached; stopped without touching services')
        if any(v['health'].get(str(p))!=200 for p in (9093,9094,9095)):
            raise RuntimeError('Milvus health check failed; stopped without touching services')
    def close(self):
        if self.ssh:self.ssh.close()

class Runner:
    def __init__(self):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.guard=Guard();self.stop=threading.Event()
        self.db=sqlite3.connect(ROOT/'journal.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
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
            'all_source_prefixes':PREFIXES,
            'limitation':'Historical duplicate physical versions are recorded and skipped in V3; full lossless completion is not yet proven.'}
        self.status()

    def status(self):
        with self.lock:
            confirmed=self.db.execute("SELECT coalesce(sum(rows),0),coalesce(sum(added),0),coalesce(sum(missing),0) FROM batches WHERE status='confirmed'").fetchone()
            self.state.update(updated_at=now(),confirmed_new_rows=confirmed[0],
                confirmed_new_source_ids=confirmed[1],confirmed_missing_source_ids=confirmed[2],
                target_expected_count=BASE_COUNT+confirmed[0],
                duplicate_keys_recorded=self.db.execute('SELECT count(*) FROM duplicate_keys').fetchone()[0],
                prefixes_finished=self.db.execute('SELECT count(*) FROM cursors WHERE finished=1').fetchone()[0],
                uncertain_batches=self.db.execute("SELECT count(*) FROM batches WHERE status='submitted'").fetchone()[0])
            atomic_json(ROOT/'status.json',self.state)

    def preflight(self):
        self.guard.check()
        source=MilvusClient(uri=URI,db_name='default',timeout=20)
        try:
            if TARGET not in source.list_collections():raise IntegrityError('Existing V3 target not found')
            if source.list_indexes(TARGET):raise IntegrityError('V3 has indexes; no-index runner refuses writes')
            n=int(source.query(SOURCE_COLLECTION,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            if n!=EXPECTED_SOURCE_COUNT:raise IntegrityError('Source count changed')
            actual=int(source.get_collection_stats(TARGET)['row_count'])
            if actual!=self.state['target_expected_count']:
                raise IntegrityError(f'Target count {actual} differs from confirmed journal {self.state["target_expected_count"]}')
            if self.state['uncertain_batches']:
                raise IntegrityError('Ambiguous submitted insert must be manually reconciled before any resume')
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
        self.guard.check();all_ids=[r['id'] for r in rows]
        if all_ids!=sorted(all_ids) or len(all_ids)!=len(set(all_ids)):
            raise IntegrityError('Source iterator returned out-of-order or repeated IDs')
        existing=self.already_inserted(old_db,all_ids)
        rows=[r for r in rows if r['id'] not in existing]
        duplicates=duplicate_counts(source,[r['id'] for r in rows])
        if duplicates:rows=[r for r in rows if r['id'] not in duplicates]
        values=self.source_ids(session,[r['doc_id'] for r in rows])
        original_digest=hashlib.sha256()
        for r in rows:original_digest.update(bytes.fromhex(original_hash(r,self.original_fields)))
        audit,added,missing=patch_rows(rows,values,self.original_fields)
        with self.lock:
            for key,n in duplicates.items():
                self.db.execute('INSERT OR IGNORE INTO duplicate_keys(id,physical_count) VALUES(?,?)',(key,n))
            if not rows:
                self.commit_cursor(prefix,all_ids[-1],len(all_ids));self.db.commit();return
            self.db.execute('INSERT INTO batches(prefix,first_id,last_id,status,ids,rows,added,missing,original_hash_aggregate) VALUES(?,?,?,?,?,?,?,?,?)',
                (prefix,all_ids[0],all_ids[-1],'prepared',canonical([r['id'] for r in rows]),len(rows),added,missing,original_digest.hexdigest()))
            seq=self.db.execute('SELECT last_insert_rowid()').fetchone()[0];self.db.commit()
            self.db.execute('UPDATE batches SET status=?,submitted_at=? WHERE seq=?',('submitted',now(),seq));self.db.commit()
        # The installed SDK retries ambiguous Insert by default; explicitly use
        # the raw single RPC once. Any exception leaves a submitted journal row
        # and stops the runner. No blind retry is permitted for duplicate PKs.
        raw=inspect.unwrap(GrpcHandler.insert_rows)
        result=raw(target._get_connection(),TARGET,rows,partition_name='',timeout=120,
                   context=target._generate_call_context())
        if result.insert_count!=len(rows):raise IntegrityError('Confirmed insert count mismatch')
        with self.lock:
            self.db.execute('UPDATE batches SET status=?,confirmed_at=? WHERE seq=?',('confirmed',now(),seq))
            self.commit_cursor(prefix,all_ids[-1],len(all_ids));self.db.commit()
            if seq%30==0:self.status()

    def run_prefix(self,prefix):
        source=MilvusClient(uri=URI,db_name='default',timeout=20)
        target=MilvusClient(uri=URI,db_name='default',timeout=20)
        old_db=sqlite3.connect('file:'+OLD.joinpath('audit.sqlite').as_posix()+'?mode=ro',uri=True)
        session=requests.Session();session.trust_env=False
        try:
            with self.lock:
                cursor=self.db.execute('SELECT last_id,finished FROM cursors WHERE prefix=?',(prefix,)).fetchone()
            if cursor and cursor[1]:return
            high=hex(int(prefix,16)+1)[2:] if prefix!='f' else 'g'
            expr=f'doc_id >= "{prefix}" and doc_id < "{high}"'
            if cursor and cursor[0]:expr+=' and id > '+canonical(cursor[0])
            iterator=source.query_iterator(SOURCE_COLLECTION,filter=expr,batch_size=200,
                output_fields=self.fields,consistency_level='Strong',reduce_stop_for_best=False,timeout=90)
            try:
                while not self.stop.is_set():
                    self.guard.check()
                    rows=[plain(dict(x)) for x in iterator.next()]
                    if not rows:break
                    self.process_batch(prefix,rows,source,target,session,old_db)
                if not self.stop.is_set():
                    with self.lock:
                        last=rows[-1]['id'] if rows else (cursor[0] if cursor else '')
                        self.commit_cursor(prefix,last,0,True);self.db.commit();self.status()
            finally:iterator.close()
        finally:session.close();old_db.close();target.close();source.close()

    def run(self):
        try:
            self.preflight()
            # Four independent prefix streams. Each prefix has a durable cursor;
            # no prefix receives concurrent writers.
            with ThreadPoolExecutor(max_workers=4) as pool:
                future={pool.submit(self.run_prefix,p):p for p in PREFIXES}
                for task in as_completed(future):
                    try:task.result()
                    except BaseException:
                        self.stop.set();raise
            self.state['status']='copy_scan_finished_pending_index_and_full_verification'
            self.status()
        except BaseException as e:
            self.state.update(status='paused_preserving_all_existing_data',error=repr(e),paused_at=now())
            self.status();raise
        finally:self.guard.close();self.old_db.close();self.db.close()

def main():
    if os.name=='nt':
        import msvcrt
    else:
        import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'scientific_fulltext_chunks_v3.lock').open('a+b') as lease:
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
