"""Ten-minute bounded V3 copy probe on 231:19533.

Only the new scientific_fulltext_chunks_v3 collection is writable. The existing
collections are read-only. No vector or scalar index is built in this probe.
"""
from __future__ import annotations
import json, os, shlex, time
from pathlib import Path
import paramiko
from pymilvus import MilvusClient
import benchmark_fulltext_source_id_isolated231_continuous as b
from fulltext_source_id_safe_common import IntegrityError, original_hash, now

PROD = 'http://192.0.2.15:19533'
COLLECTION = 'scientific_fulltext_chunks_v3'
PROJECT = Path(__file__).resolve().parents[1]


def only_new_prod_collection(source, target, collection):
    if source != PROD or target != PROD or collection != COLLECTION:
        raise ValueError('writes are limited to 231:19533/scientific_fulltext_chunks_v3')


b.SOURCE_URI = PROD
b.TARGET_URI = PROD
b.COLLECTION_PREFIX = COLLECTION
b.validate_destination = only_new_prod_collection

REMOTE = r'''
import json,os,time,urllib.request
from pathlib import Path
def cpu(): return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
a=cpu();time.sleep(.4);z=cpu();d=[y-x for x,y in zip(a,z)]
mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));health={}
for p in (9093,9094,9095):
 try: health[str(p)]=op.open('http://127.0.0.1:%d/healthz'%p,timeout=2).status
 except Exception as e: health[str(p)]=str(e)
print(json.dumps({'cpu_percent':100*(sum(d)-d[3]-d[4])/max(sum(d),1),'available_memory':mem.get('MemAvailable',0),'health':health}))
'''


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


class ProductionGuard:
    def __init__(self, root):
        self.root, self.latest, self.last_at = root, None, 0
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect('192.0.2.15', username='root',
                         password=os.environ['BACKFILL_SSH_PASSWORD'], timeout=12)

    def sample(self, force=False):
        if not force and self.latest and time.monotonic()-self.last_at < 10:
            return self.latest
        _, out, err = self.ssh.exec_command(shlex.join(['python3','-c',REMOTE]), timeout=15)
        body, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(error[-800:])
        value = json.loads(body); value['at'] = now(); self.latest = value
        self.last_at = time.monotonic(); write_json(self.root/'resources_latest.json',value)
        with (self.root/'resources.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(value,ensure_ascii=False)+'\n')
        return value

    def check(self, operation):
        value = self.sample()
        if value['cpu_percent'] >= 65: raise RuntimeError('CPU safety limit before '+operation)
        if value['available_memory'] < 200*1024**3: raise RuntimeError('memory safety limit before '+operation)
        if any(value['health'].get(str(p)) != 200 for p in (9093,9094,9095)):
            raise RuntimeError('Milvus health failure before '+operation)


class ProductionV3Probe(b.Experiment):
    def __init__(self, root, guard):
        # Upper bound only: dispatch is stopped at ten minutes, normally far below 800k.
        super().__init__(root, COLLECTION, rows=800000, seconds=600, skip_duplicate_keys=True)
        self.guard, self.workers = guard, 4
        self.report.update({'target_uri':PROD,'test_collection':COLLECTION,
            'writes_allowed_only_to_new_collection':True,'index_created':False,
            'index_plan':'none during this ten-minute probe; build once after full copy',
            'production_cutover':False,'full_copy':False,'copy_window_seconds':600})

    def before_operation(self, operation): self.guard.check(operation)

    def preflight(self):
        """Accept only the empty collection created by this probe's failed start."""
        self.source = MilvusClient(uri=PROD, db_name='default', timeout=20)
        self.target = MilvusClient(uri=PROD, db_name='default', timeout=20)
        self.target_collections_before = set(self.target.list_collections())
        self.source_schema = self.source.describe_collection(b.SOURCE_COLLECTION)
        self.source_fields = [f['name'] for f in self.source_schema['fields']]
        self.original_fields = [f for f in self.source_fields if f != b.FIELD]
        source_count = self.count(self.source, b.SOURCE_COLLECTION)
        if source_count != b.EXPECTED_SOURCE_COUNT:
            raise IntegrityError('Protected source count changed before V3 probe')
        if COLLECTION in self.target_collections_before:
            existing = self.target.describe_collection(COLLECTION)
            if [f['name'] for f in existing['fields']] != self.source_fields:
                raise IntegrityError('Existing V3 schema differs; refusing to write')
            if self.target.list_indexes(COLLECTION):
                raise IntegrityError('Existing V3 has an index; refusing to write')
            existing_count = int(self.target.get_collection_stats(COLLECTION).get('row_count', -1))
            if existing_count != 0:
                raise IntegrityError('Existing V3 is not empty; refusing to append or overwrite')
            self.report['resumed_existing_empty_collection'] = True
        else:
            schema = self.clone_schema(self.source_schema)
            b.RestrictedTarget(self.target, COLLECTION).create(schema)
            if self.target.list_indexes(COLLECTION):
                raise IntegrityError('New V3 unexpectedly has an index')
        self.report['source_count_before'] = source_count
        self.report['source_field_count_before'] = self.count(self.source, b.SOURCE_COLLECTION, b.FIELD+' is not null')
        self.report['target_collections_before'] = sorted(self.target_collections_before)
        (self.root/'source_schema.json').write_text(json.dumps(self.source_schema,ensure_ascii=False,indent=2),encoding='utf-8')
        self.save()

    def verify_without_index(self):
        self.report['status']='flushing_v3_without_index'; self.progress()
        started=time.monotonic(); self.target.flush(COLLECTION,timeout=300)
        self.report['seconds']['flush_without_index']=time.monotonic()-started
        count=int(self.target.get_collection_stats(COLLECTION).get('row_count',-1))
        if count != self.report['inserted_rows_pending_verification']:
            raise IntegrityError('V3 count differs from confirmed inserts')
        self.report['v3_count_after_flush']=count
        ids=[r[0] for r in self.db.execute('SELECT id FROM expected_rows ORDER BY seq,row_no LIMIT 200')]
        rows=self.target.get(COLLECTION,ids=ids,output_fields=self.original_fields+[b.FIELD],
                             consistency_level='Strong',timeout=90)
        expected={k:(digest,after) for k,digest,before,after in self.db.execute(
            'SELECT id,original_hash,source_field_before,source_field_after FROM expected_rows LIMIT 200')}
        for raw in rows:
            row=b.plain(dict(raw)); wanted=expected.get(row['id'])
            if wanted is None or (original_hash(row,self.original_fields),b.canonical(row.get(b.FIELD))) != wanted:
                raise IntegrityError('V3 sample readback mismatch for '+row['id'])
        if len(rows)!=len(ids): raise IntegrityError('V3 sample readback incomplete')
        self.report['v3_sample_rows_verified_without_index']=len(rows)
        self.report['index_states_after_test']=self.target.list_indexes(COLLECTION)
        if self.report['index_states_after_test']: raise IntegrityError('Unexpected V3 index')

    def run(self):
        started=time.monotonic()
        try:
            self.preflight(); self.report['status']='copying_v3_without_indexes'
            self.last_ids = {}
            self.copy_phase('batch200_workers4_no_index',200,self.row_limit,self.seconds)
            self.report['seconds']['copy_without_indexes']=sum(
                p['elapsed_seconds_including_inflight_waits'] for p in self.report['phases'])
            self.report['copy_rows_per_second_without_indexes']=self.report['inserted_rows_pending_verification']/max(self.report['seconds']['copy_without_indexes'],.001)
            self.verify_without_index(); self.report['status']='completed_10min_partial_v3_no_index'
            self.report['full_copy']=False; self.report['production_writes']=self.report['inserted_rows_pending_verification']
        except BaseException as exc:
            self.report['status']='stopped_v3_partial_collection_preserved_no_index'; self.report['error']=repr(exc); raise
        finally:
            self.report['seconds']['wall_clock_until_final_flush']=time.monotonic()-started; self.report['finished_at']=now(); self.progress()
            for iterator in self.iterators.values(): iterator.close()
            for source,target,session in self.worker_resources: session.close(); target.close(); source.close()
            if self.target: self.target.close()
            if self.source: self.source.close()
            self.db.close()


def main():
    root=PROJECT/'outputs'/('fulltext_v3_prod_10min_231_'+time.strftime('%Y%m%d_%H%M%S'))
    import msvcrt
    with (PROJECT/'outputs/scientific_fulltext_chunks_v3.lock').open('a+b') as lease:
        lease.seek(0)
        if not lease.read(1): lease.write(b'0'); lease.flush()
        lease.seek(0); msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
        guard=ProductionGuard(root)
        try: ProductionV3Probe(root,guard).run()
        finally: guard.sample(force=True); guard.ssh.close(); lease.seek(0); msvcrt.locking(lease.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
