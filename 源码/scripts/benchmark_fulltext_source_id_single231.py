"""Authorized 30-minute field-only run on 231, with measured bounded tuning.

230 and 232 are read-only baselines. All request bodies for mutations target the
fixed 231 endpoint, contain only id + hasSourcesysSourceId, and use partial upsert.
Original fields/vectors are hashed before/after every update; duplicate PKs skip.
"""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Windows continuation uses its own remote exclusive lock.
    fcntl = None
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

from benchmark_fulltext_source_id_100 import COLLECTION, FIELD, HOSTS, SOLR, SOLR_FIELD, connect, digest, plain, snapshot
from fulltext_source_id_safe_common import encode, decode, original_hash, valid_source, now, IntegrityError

OLD = Path('/data/scientific-fulltext-source-id-backfill-20261113')
TARGET = '192.0.2.15'


class Benchmark:
    def __init__(self, folder, seconds):
        if fcntl is None:
            raise RuntimeError('Use run_fulltext_source_id_local231.py on Windows')
        self.root = Path(folder)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(OLD/'runner.lock','a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (OLD/'STOP_REQUESTED').exists():
            raise RuntimeError('Three-node runner must remain stopped')
        self.db = sqlite3.connect(self.root/'journal.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS batches(seq INTEGER PRIMARY KEY,phase TEXT,status TEXT,payload BLOB,result TEXT);
        CREATE TABLE IF NOT EXISTS source_cache(doc_id TEXT PRIMARY KEY,value TEXT,reason TEXT);
        ''')
        self.db.commit()
        if self.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]:
            raise RuntimeError('Existing run journal: explicit recovery required; refusing fresh enumeration')
        self.db_lock = threading.Lock()
        old = sqlite3.connect('file:'+str(OLD/'progress.sqlite')+'?mode=ro',uri=True)
        self.old_state = {k:json.loads(v) for k,v in old.execute('SELECT key,value FROM state')}
        for row in old.execute('SELECT doc_id,result,reason FROM source_cache'):
            self.db.execute('INSERT OR IGNORE INTO source_cache VALUES(?,?,?)',row)
        self.db.commit()
        old.close()
        self.cursor = self.old_state['cursor']
        self.bucket = self.old_state['document_id_bucket']
        self.iterator = None
        self.buffer = []
        self.main = connect(TARGET)
        self.local = threading.local()
        self.all_worker_clients = []
        self.seconds = seconds
        self.start = None
        self.deadline = None
        self.baseline = {}
        self.before_filled = {}
        self.phases = []
        self.phase = None
        self.counts = {'handled':0,'newly_filled':0,'already_filled':0,'no_source':0,
                       'duplicate_keys_skipped':0,'duplicate_extra_versions_preserved':0,
                       'completed_batches':0,'retry_events':0}
        self.last_guard = 0.0
        self.guard_count=0
        self.status_lock = threading.Lock()
        self.last_progress = 0.0
        self.session = requests.Session()
        self.session.trust_env = False

    def log(self,event,**kwargs):
        print(json.dumps({'at':now(),'event':event,**kwargs},ensure_ascii=False),flush=True)

    def status(self,state='running',**kwargs):
        with self.status_lock:
            with self.db_lock:
                counts=dict(self.counts)
                pending=self.db.execute("SELECT COUNT(*) FROM batches WHERE status!='completed'").fetchone()[0]
            elapsed=time.monotonic()-self.start if self.start else 0
            data={'at':now(),'status':state,'pid':os.getpid(),'write_host':TARGET,
                  'read_only_hosts':list(HOSTS[1:]),'duration_seconds':self.seconds,
                  'elapsed_seconds':round(elapsed,2),'phase':self.phase,'phases':self.phases,
                  **counts,'pending_batches':pending,'cursor':self.cursor,'bucket':self.bucket,
                  'new_rows_per_second':round(counts['newly_filled']/max(elapsed,1),2),**kwargs}
            tmp=self.root/'status.tmp'
            tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,self.root/'status.json')
            return data

    def preflight(self):
        self.status('preflight')
        def read(h):
            c=connect(h)
            try:
                return snapshot(c),int(c.query(COLLECTION,filter=FIELD+' is not null',
                    output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            finally:c.close()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={h:pool.submit(read,h) for h in HOSTS}
            for h,f in futures.items():
                self.baseline[h],self.before_filled[h]=f.result()
        if self.baseline!=self.old_state['baseline'] or len(set(self.before_filled.values()))!=1:
            raise IntegrityError('Stopped replica baseline no longer matches')
        self.fields=[f['name'] for f in self.baseline[TARGET][COLLECTION]['fields']]
        self.original_fields=[f for f in self.fields if f!=FIELD]
        self.expected_count=self.baseline[TARGET][COLLECTION]['count']
        (self.root/'baseline.json').write_text(json.dumps({'snapshots':self.baseline,
            'filled_counts':self.before_filled,'start_cursor':self.cursor,'start_bucket':self.bucket},
            ensure_ascii=False,indent=2),encoding='utf-8')
        self.guard(force=True)

    def worker_client(self):
        if not hasattr(self.local,'client'):
            self.local.client=connect(TARGET)
            with self.db_lock:self.all_worker_clients.append(self.local.client)
        return self.local.client

    def get_rows(self,c,ids):
        result={r['id']:{k:(v if k=='text_embed' else plain(v)) for k,v in r.items()}
            for r in c.get(COLLECTION,ids=ids,output_fields=self.fields,consistency_level='Strong',timeout=90)}
        if len(result)!=len(ids) or set(result)!=set(ids):
            raise IntegrityError('Missing or duplicated returned primary key')
        return result

    def key_count(self,c,ids):
        return int(c.query(COLLECTION,filter='id in '+json.dumps(ids),output_fields=['count(*)'],
            consistency_level='Strong',timeout=90)[0]['count(*)'])

    def duplicates(self,c,ids):
        n=self.key_count(c,ids)
        if n<len(ids):raise IntegrityError('Primary key disappeared')
        if n==len(ids):return {}
        if len(ids)==1:return {ids[0]:n}
        mid=len(ids)//2
        return {**self.duplicates(c,ids[:mid]),**self.duplicates(c,ids[mid:])}

    def fetch_source(self,rows):
        values={}; missing=[]; reasons={}
        with self.db_lock:
            for i in sorted({r['doc_id'] for r in rows}):
                row=self.db.execute('SELECT value,reason FROM source_cache WHERE doc_id=?',(i,)).fetchone()
                if row:values[i],reasons[i]=json.loads(row[0]),row[1]
                else:missing.append(i)
        for offset in range(0,len(missing),300):
            ids=missing[offset:offset+300]
            if any(len(i)!=32 or any(ch not in '0123456789abcdef' for ch in i) for i in ids):
                raise IntegrityError('Unexpected document identifier')
            response=self.session.post(SOLR,data={'q':'{!terms f=id}'+','.join(ids),
                'fl':'id,'+SOLR_FIELD,'rows':len(ids),'wt':'json','timeAllowed':15000},timeout=(10,40))
            response.raise_for_status();body=response.json()
            if body.get('responseHeader',{}).get('partialResults'):
                raise RuntimeError('Incomplete Solr response')
            docs=body['response']['docs']
            if body['response']['numFound']!=len(docs) or len({r['id'] for r in docs})!=len(docs):
                raise IntegrityError('Ambiguous Solr response')
            found={r['id']:r for r in docs}
            with self.db_lock:
                for i in ids:
                    value=found.get(i,{}).get(SOLR_FIELD)
                    reason=None if valid_source(value) else ('document_missing' if i not in found else 'source_id_missing')
                    value=value if reason is None else None
                    values[i],reasons[i]=value,reason
                    self.db.execute('INSERT INTO source_cache VALUES(?,?,?)',(i,json.dumps(value),reason))
                self.db.commit()
        return {r['id']:values[r['doc_id']] for r in rows},reasons

    def select(self,n):
        while len(self.buffer)<n and self.bucket<4096:
            if self.iterator is None:
                lower=format(self.bucket,'03x');upper=format(self.bucket+1,'03x') if self.bucket<4095 else 'g'
                expr='doc_id >= '+json.dumps(lower)+' and doc_id < '+json.dumps(upper)
                if self.cursor:expr+=' and id > '+json.dumps(self.cursor)
                self.iterator=self.main.query_iterator(COLLECTION,filter=expr,batch_size=2000,
                    output_fields=['id','doc_id'],consistency_level='Strong',timeout=90)
            rows=self.iterator.next()
            if not rows:
                self.iterator.close();self.iterator=None;self.bucket+=1;continue
            self.buffer.extend(dict(r) for r in rows)
        rows=self.buffer[:n];self.buffer=self.buffer[n:]
        if rows:
            ids=[r['id'] for r in rows]
            if ids!=sorted(ids) or len(set(ids))!=len(ids) or (self.cursor and ids[0]<=self.cursor):
                raise IntegrityError('Enumeration ordering violated')
            if any(r['id'].rsplit('_',1)[0]!=r['doc_id'] for r in rows):
                raise IntegrityError('Unexpected chunk/document relation')
            self.cursor=ids[-1]
        return rows

    def task(self,seq,selected,expected,reasons):
        c=self.worker_client();ids=[r['id'] for r in selected];timing={};t0=time.monotonic()
        t=time.monotonic();duplicates=self.duplicates(c,ids);timing['duplicate_check']=time.monotonic()-t
        t=time.monotonic();before=self.get_rows(c,ids);timing['read_before']=time.monotonic()-t
        t=time.monotonic();hashes={i:original_hash(r,self.original_fields) for i,r in before.items()};timing['hash_before']=time.monotonic()-t
        patches=[];already=0;no_source=[]
        for row in selected:
            i=row['id'];old=before[i].get(FIELD);value=expected[i]
            if before[i]['doc_id']!=row['doc_id']:raise IntegrityError('Original document ID changed')
            if i in duplicates:continue
            if value is None:no_source.append({'id':i,'doc_id':row['doc_id'],'reason':reasons[row['doc_id']]});continue
            if old is not None:
                if old!=value:raise IntegrityError('Existing source value conflicts')
                already+=1;continue
            patches.append({'id':i,FIELD:value})
        payload={'ids':ids,'selected':selected,'patches':patches,'original_hashes':hashes,
                 'original_values':{i:r.get(FIELD) for i,r in before.items()},
                 'duplicate_keys':duplicates,'no_source':no_source}
        with self.db_lock:
            self.db.execute("UPDATE batches SET status='prepared',payload=? WHERE seq=?",(encode(payload),seq))
            self.db.commit()
        del before
        t=time.monotonic()
        if patches:
            if self.key_count(c,[r['id'] for r in patches])!=len(patches):
                raise IntegrityError('Refusing to merge duplicate versions')
            result=c.upsert(COLLECTION,data=patches,partial_update=True,timeout=120)
            if int(result.get('upsert_count',-1))!=len(patches):raise RuntimeError('Unexpected update count')
        timing['write_including_final_duplicate_guard']=time.monotonic()-t
        t=time.monotonic();after=self.get_rows(c,ids);timing['read_after']=time.monotonic()-t
        t=time.monotonic();patchmap={r['id']:r[FIELD] for r in patches}
        for i,row in after.items():
            if original_hash(row,self.original_fields)!=hashes[i]:raise IntegrityError('Original fields/vector changed')
            if row.get(FIELD)!=patchmap.get(i,payload['original_values'][i]):raise IntegrityError('Source readback mismatch')
        timing['hash_and_verify_after']=time.monotonic()-t
        timing['total']=time.monotonic()-t0
        result={'rows':len(ids),'newly_filled':len(patches),'already_filled':already,
                'no_source':len(no_source),'duplicates':len(duplicates),
                'extra_duplicate_versions':sum(v-1 for v in duplicates.values()),'timing':timing,
                'finished_at':now(),'original_fields_and_vectors_unchanged':True,'source_values_match':True}
        with self.db_lock:
            self.db.execute("UPDATE batches SET status='completed',result=? WHERE seq=?",(json.dumps(result),seq))
            self.db.commit()
            for key,incr in [('handled',len(ids)),('newly_filled',len(patches)),('already_filled',already),
                             ('no_source',len(no_source)),('duplicate_keys_skipped',len(duplicates)),
                             ('duplicate_extra_versions_preserved',sum(v-1 for v in duplicates.values())),
                             ('completed_batches',1)]:self.counts[key]+=incr
        return result

    def guard(self,force=False):
        if not force and time.monotonic()-self.last_guard<120:return
        import paramiko
        ssh=paramiko.SSHClient();ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(TARGET,username='root',password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=10)
        try:
            _,o,e=ssh.exec_command("python3 -c 'import os,json; print(json.dumps({p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in [\"/\",\"/data\"]}))'",timeout=20)
            disk=json.loads(o.read().decode())
            if disk['/']<40*1024**3 or disk['/data']<1024**4:raise RuntimeError('Free-space guard reached')
        finally:ssh.close()
        if self.guard_count%3==0:
            count=int(self.main.query(COLLECTION,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            if count!=self.expected_count:raise IntegrityError('Target collection count changed')
        self.guard_count+=1;self.last_guard=time.monotonic()
        self.disk=disk

    def phase_run(self,name,batch,workers,until):
        self.phase={'name':name,'batch_size':batch,'workers':workers}
        t0=time.monotonic();before=dict(self.counts);pending=[];timing_totals={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                while time.monotonic()<until and time.monotonic()<self.deadline:
                    if (self.root/'STOP_REQUESTED').exists():break
                    if len(pending)>=workers:
                        result=pending.pop(0).result()
                        for k,v in result['timing'].items():timing_totals[k]=timing_totals.get(k,0)+v
                    self.guard()
                    previous_cursor=self.cursor
                    selected=self.select(batch)
                    if not selected:break
                    expected,reasons=self.fetch_source(selected)
                    if time.monotonic()>=until or time.monotonic()>=self.deadline:
                        # No skipped keys: these selected records are not dispatched.
                        self.buffer=selected+self.buffer
                        self.cursor=previous_cursor
                        break
                    with self.db_lock:
                        seq=self.db.execute("INSERT INTO batches(phase,status,payload) VALUES(?,'queued',?)",
                            (name,encode({'selected':selected}))).lastrowid
                        self.db.commit()
                    pending.append(pool.submit(self.task,seq,selected,expected,reasons))
                    if time.monotonic()-self.last_progress>=15:
                        self.status();self.log('progress',phase=self.phase,**self.counts)
                        self.last_progress=time.monotonic()
            finally:
                # Drain every in-flight batch; no partial worker left behind.
                for future in pending:
                    result=future.result()
                    for k,v in result['timing'].items():timing_totals[k]=timing_totals.get(k,0)+v
        elapsed=time.monotonic()-t0
        result={'name':name,'batch_size':batch,'workers':workers,'seconds':elapsed,
                'handled':self.counts['handled']-before['handled'],
                'newly_filled':self.counts['newly_filled']-before['newly_filled'],
                'rows_per_second':(self.counts['newly_filled']-before['newly_filled'])/max(elapsed,1),
                'timing_totals':timing_totals}
        self.phases.append(result);self.status();self.log('phase_complete',**result)
        return result

    def finish(self):
        self.status('verifying_final_read_only')
        def read(h):
            c=connect(h)
            try:return snapshot(c),int(c.query(COLLECTION,filter=FIELD+' is not null',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            finally:c.close()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={h:pool.submit(read,h) for h in HOSTS}
            final={h:f.result() for h,f in futures.items()}
        for h,(snap,filled) in final.items():
            if snap!=self.baseline[h]:raise IntegrityError('Collection inventory/count/schema changed on '+h)
            want=self.before_filled[h]+(self.counts['newly_filled'] if h==TARGET else 0)
            if filled!=want:raise IntegrityError('Unexpected final filled count on '+h)
        with self.db_lock:
            pending=self.db.execute("SELECT COUNT(*) FROM batches WHERE status!='completed'").fetchone()[0]
            if pending:raise IntegrityError('Incomplete batches remain')
            last=self.db.execute("SELECT payload FROM batches ORDER BY seq DESC LIMIT 1").fetchone()
        last_id=decode(last[0])['ids'][-1] if last else self.old_state['cursor']
        report={'status':'completed_and_stopped','finished_at':now(),'write_host':TARGET,
                'read_only_hosts':list(HOSTS[1:]),'duration_seconds':self.seconds,
                'write_window_seconds':self.write_window_seconds,'counts':self.counts,'phases':self.phases,
                'chosen_configuration':self.chosen,'pending_batches':pending,'end_cursor':last_id,
                'filled_before':self.before_filled,'filled_after':{h:v[1] for h,v in final.items()},
                'collection_counts':{h:v[0][COLLECTION]['count'] for h,v in final.items()},
                'all_original_fields_vectors_verified_each_batch':True,
                'other_two_nodes_unchanged':True,'last_disk_bytes':self.disk}
        (self.root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.status('completed_and_stopped',end_cursor=last_id)
        self.log('completed_and_stopped',**report)

    def run(self):
        try:
            self.preflight()
            self.start=time.monotonic();self.deadline=self.start+self.seconds
            self.log('single231_started',seconds=self.seconds,start_cursor=self.cursor)
            # Comparable first pilot to the old 500-row flow; then a small ladder
            # measures batching and concurrency independently on real records.
            pilots=[]
            for name,batch,workers in [('baseline500_w1',500,1),('batch2000_w1',2000,1),
                                       ('batch2000_w2',2000,2),('batch2000_w4',2000,4)]:
                pilots.append(self.phase_run(name,batch,workers,min(time.monotonic()+45,self.deadline)))
            best=max(pilots,key=lambda x:x['rows_per_second'])
            # Prefer lower pressure if performance is within 5% of measured best.
            chosen=min([p for p in pilots if p['rows_per_second']>=best['rows_per_second']*0.95],
                       key=lambda x:(x['workers'],x['batch_size']))
            self.chosen={'batch_size':chosen['batch_size'],'workers':chosen['workers'],
                         'measured_pilot_rows_per_second':chosen['rows_per_second']}
            self.phase_run('sustained_selected',chosen['batch_size'],chosen['workers'],self.deadline)
            self.write_window_seconds=time.monotonic()-self.start
            self.finish()
        except BaseException as exc:
            self.status('stopped_error',error=repr(exc),traceback=traceback.format_exc())
            self.log('stopped_error',error=repr(exc));raise
        finally:
            if self.iterator:self.iterator.close()
            for c in self.all_worker_clients:c.close()
            self.main.close();self.db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',required=True);p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--apply',action='store_true');args=p.parse_args()
    if not args.apply or args.seconds>1800 or args.seconds<=0:
        p.error('--apply and a bounded duration 1..1800 seconds are required')
    Benchmark(args.workdir,args.seconds).run()
