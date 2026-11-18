"""Bounded 231 copy-throughput experiment; never a production migration tool.

Production 19533 is read only. Only a fresh, specially named collection in the
existing isolated 19539 instance may be written. Copying has no indexes; a FLAT
index is built afterwards solely to permit full readback verification. Timings
do not claim readiness with the production collection's indexes.

QueryIterator may suppress historical duplicate primary keys. By default any
identified duplicate aborts the experiment. The explicit benchmark-only skip
option records and excludes those keys, never deduplicating the production
data, and makes the sample unsuitable as proof of a lossless full migration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from urllib.parse import urlsplit

from pymilvus import DataType, MilvusClient
import requests

from benchmark_fulltext_source_id_100 import plain
from fulltext_source_id_safe_common import IntegrityError, now, original_hash, valid_source


SOURCE_URI = 'http://192.0.2.15:19533'
TARGET_URI = 'http://192.0.2.15:19539'
SOURCE_COLLECTION = 'scientific_fulltext_chunks_v2'
EXPECTED_SOURCE_COUNT = 195893017
COLLECTION_PREFIX = 'fulltext_source_id_throughput_231_'
FIELD = 'hasSourcesysSourceId'
SOLR_FIELD = 'stringIMS_hasSourcesysSourceId'
SOLR = 'http://service.example.invalid:8984/solr/paper6.6/select'
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKETS = tuple(format(i*4096//30, '03x') for i in range(30))
MAX_VECTOR_ROWS = 200
PREFIX_TURN_ROWS = 2000


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def validate_destination(source_uri, target_uri, collection):
    def endpoint(uri):
        u = urlsplit(uri)
        if u.scheme != 'http' or u.username or u.password or u.path or u.query or u.fragment:
            raise ValueError('Only the exact approved HTTP Milvus endpoints are allowed')
        return u.hostname, u.port
    if endpoint(source_uri) != ('192.0.2.15', 19533):
        raise ValueError('Production source must be 231:19533')
    if endpoint(target_uri) != ('192.0.2.15', 19539):
        raise ValueError('Writes must be limited to isolated 231:19539')
    if not re.fullmatch(re.escape(COLLECTION_PREFIX)+r'[A-Za-z0-9_]{1,100}', collection):
        raise ValueError('A fresh benchmark collection with the required prefix is required')


def duplicate_counts(client, ids):
    """Find all physical duplicates without reading their vectors or changing them."""
    if not ids:
        return {}
    if len(set(ids)) != len(ids):
        raise IntegrityError('Source iterator returned a repeated primary key in one batch')
    n = int(client.query(SOURCE_COLLECTION, filter='id in '+canonical(ids),
        output_fields=['count(*)'], consistency_level='Strong', timeout=90)[0]['count(*)'])
    if n < len(ids):
        raise IntegrityError('Source primary key disappeared during the read-only sample')
    if n == len(ids):
        return {}
    if len(ids) == 1:
        return {ids[0]:n}
    middle = len(ids)//2
    return {**duplicate_counts(client, ids[:middle]), **duplicate_counts(client, ids[middle:])}


def patch_rows(rows, source_ids, original_fields):
    """Keep all returned rows and every original value; change only the new field."""
    audit = []
    added = missing = 0
    for row in rows:
        doc_id = row['doc_id']
        if row['id'].rsplit('_', 1)[0] != doc_id:
            raise IntegrityError('Unexpected chunk/document identifier relationship')
        before_hash = original_hash(row, original_fields)
        old = row.get(FIELD)
        new = source_ids.get(doc_id)
        if valid_source(new):
            if old is not None and old != new:
                raise IntegrityError('Existing source ID conflicts with Solr for '+row['id'])
            if old is None:
                row[FIELD] = new
                added += 1
        elif old is None:
            row[FIELD] = None
            missing += 1
        if original_hash(row, original_fields) != before_hash:
            raise IntegrityError('Copy preparation changed an original scalar or vector')
        audit.append((row['id'], before_hash, canonical(old), canonical(row.get(FIELD))))
    return audit, added, missing


class RestrictedTarget:
    """No deletion or update API exists on this writer; only one fresh collection."""
    def __init__(self, raw, collection):
        validate_destination(SOURCE_URI, TARGET_URI, collection)
        self.raw = raw
        self.collection = collection

    def _name(self, name):
        if name != self.collection:
            raise IntegrityError('Attempt to address an existing or production collection')

    def create(self, schema):
        if self.collection in self.raw.list_collections():
            raise IntegrityError('Benchmark collection already exists; no overwrite allowed')
        return self.raw.create_collection(self.collection, schema=schema, timeout=90)

    def insert(self, name, rows):
        self._name(name)
        if not 1 <= len(rows) <= MAX_VECTOR_ROWS:
            raise IntegrityError('Vector batches must contain 1 through 200 rows')
        return self.raw.insert(name, data=rows, timeout=120)


class Experiment:
    def __init__(self, root, collection, rows=600000, seconds=900,
                 skip_duplicate_keys=False, buckets=DEFAULT_BUCKETS):
        validate_destination(SOURCE_URI, TARGET_URI, collection)
        if not 200000 <= rows <= 800000 or not 600 <= seconds <= 900:
            raise ValueError('Bounded experiment requires 200000..800000 rows and 600..900 seconds')
        if len(buckets) != 30 or len(set(buckets)) != 30 or any(
                not re.fullmatch(r'[0-9a-f]{3}', p) for p in buckets):
            raise ValueError('Exactly 30 distinct hexadecimal doc_id prefixes are required')
        self.root = Path(root).resolve()
        if self.root.exists():
            raise RuntimeError('Use a new output directory; previous evidence is preserved')
        self.root.mkdir(parents=True)
        self.collection = collection
        self.row_limit = rows
        self.seconds = seconds
        self.skip_duplicates = skip_duplicate_keys
        self.buckets = tuple(buckets)
        self.lock = threading.RLock()
        self.local = threading.local()
        self.worker_resources = []
        self.source = None
        self.target = None
        self.iterators = {}
        self.active_prefix = None
        self.active_batch_size = None
        self.active_prefix_rows = 0
        self.bucket_rows = {prefix:0 for prefix in self.buckets}
        self.exhausted = set()
        self.next_bucket = 0
        self.batch_sequence = 0
        self.db = sqlite3.connect(self.root/'audit.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE expected_rows(seq INTEGER,row_no INTEGER,id TEXT,original_hash TEXT,
                source_field_before TEXT,source_field_after TEXT,verified INTEGER DEFAULT 0,
                PRIMARY KEY(seq,row_no));
            CREATE INDEX expected_id ON expected_rows(id);
            CREATE TABLE excluded_duplicate_keys(id TEXT PRIMARY KEY,physical_count INTEGER);
            CREATE TABLE source_cache(doc_id TEXT PRIMARY KEY,value TEXT,reason TEXT);
            CREATE TABLE batches(seq INTEGER PRIMARY KEY,phase TEXT,status TEXT,rows INTEGER,
                inserted INTEGER DEFAULT 0,result TEXT);
        ''')
        self.db.commit()
        self.report = {'started_at':now(),'status':'preflight','source_uri':SOURCE_URI,
            'target_uri':TARGET_URI,'source_collection':SOURCE_COLLECTION,
            'test_collection':collection,'planned_max_scanned_rows':rows,
            'copy_dispatch_window_seconds':seconds,'workers':2,
            'max_vector_rows_per_response':MAX_VECTOR_ROWS,'sample_prefixes':self.buckets,
            'maximum_simultaneously_open_source_iterators':1,
            'prefix_rows_per_round':PREFIX_TURN_ROWS,
            'skip_duplicate_keys_for_benchmark':skip_duplicate_keys,
            'integrity_boundary':(
                'BENCHMARK ONLY: identified duplicate primary keys are recorded and excluded; '
                'this sample cannot prove a lossless full migration; automatic production switching is forbidden'
                if skip_duplicate_keys else
                'Any identified historical duplicate aborts the benchmark; API iterator samples cannot prove full-version preservation'),
            'read_rows':0,'inserted_rows_pending_verification':0,'source_ids_added':0,
            'source_ids_missing':0,'excluded_duplicate_keys':0,
            'excluded_physical_versions':0,'excluded_duplicate_details_file':'excluded_duplicate_keys.json',
            'phases':[],'seconds':{},
            'production_writes':0,'production_ready':False,
            'timing_scope':'Copy phases exclude indexes. FLAT index after copying is solely for full readback; production index readiness is not measured.'}
        self.save()

    def save(self):
        with self.lock:
            self.report['updated_at'] = now()
            self.report['prefix_rows_read'] = dict(self.bucket_rows)
            self.report['rows_pending_readback'] = max(0,
                self.report['inserted_rows_pending_verification']-self.report.get('verified_rows',0))
            tmp = self.root/'report.tmp'
            tmp.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(self.root/'report.json')

    def progress(self):
        self.save()
        print(canonical({key:self.report.get(key) for key in ('updated_at','status',
            'read_rows','inserted_rows_pending_verification','excluded_duplicate_keys')}), flush=True)

    def worker(self):
        if not hasattr(self.local, 'resources'):
            source = MilvusClient(uri=SOURCE_URI, db_name='default', timeout=20)
            target = MilvusClient(uri=TARGET_URI, db_name='default', timeout=20)
            session = requests.Session()
            session.trust_env = False
            self.local.resources = source, RestrictedTarget(target, self.collection), session
            with self.lock:
                self.worker_resources.append((source,target,session))
        return self.local.resources

    def sources(self, doc_ids, session):
        values = {}
        missing = []
        with self.lock:
            for doc_id in sorted(set(doc_ids)):
                if not re.fullmatch('[0-9a-f]{32}', doc_id):
                    raise IntegrityError('Unexpected Solr document ID')
                cached = self.db.execute('SELECT value FROM source_cache WHERE doc_id=?',(doc_id,)).fetchone()
                if cached:
                    values[doc_id] = json.loads(cached[0])
                else:
                    missing.append(doc_id)
        for offset in range(0,len(missing),200):
            chunk = missing[offset:offset+200]
            response = session.post(SOLR, data={'q':'{!terms f=id}'+','.join(chunk),
                'fl':'id,'+SOLR_FIELD,'rows':len(chunk),'wt':'json','timeAllowed':20000}, timeout=(10,40))
            response.raise_for_status()
            body = response.json()
            docs = body['response']['docs']
            if body.get('responseHeader',{}).get('partialResults') or (
                    body['response']['numFound'] != len(docs)) or (
                    len({row['id'] for row in docs}) != len(docs)) or (
                    not {row['id'] for row in docs}.issubset(chunk)):
                raise IntegrityError('Incomplete or ambiguous Solr response')
            found = {row['id']:row.get(SOLR_FIELD) for row in docs}
            with self.lock:
                for doc_id in chunk:
                    value = found.get(doc_id)
                    reason = None if valid_source(value) else 'missing_document_or_source_id'
                    value = value if reason is None else None
                    encoded = canonical(value)
                    old = self.db.execute('SELECT value FROM source_cache WHERE doc_id=?',(doc_id,)).fetchone()
                    if old and old[0] != encoded:
                        raise IntegrityError('Solr source ID changed during the sample')
                    self.db.execute('INSERT OR IGNORE INTO source_cache VALUES(?,?,?)',(doc_id,encoded,reason))
                    values[doc_id] = value
                self.db.commit()
        return values

    def count(self, client, collection, expression=''):
        return int(client.query(collection, filter=expression, output_fields=['count(*)'],
            consistency_level='Strong', timeout=90)[0]['count(*)'])

    def clone_schema(self, source_schema):
        if source_schema.get('auto_id'):
            raise IntegrityError('Auto-ID sources are outside this benchmark')
        schema = self.target.create_schema(auto_id=False,
            enable_dynamic_field=source_schema.get('enable_dynamic_field',False),
            description='Bounded isolated 231 source-ID copy benchmark; not for production switching')
        for field in source_schema['fields']:
            args = {'field_name':field['name'],'datatype':field['type'],
                'description':field.get('description',''), **field.get('params',{})}
            for name in ('is_primary','nullable','element_type','default_value','is_partition_key','is_clustering_key'):
                if name in field:
                    args[name] = field[name]
            schema.add_field(**args)
        if FIELD not in {field['name'] for field in source_schema['fields']}:
            schema.add_field(field_name=FIELD, datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR, max_capacity=4096, max_length=65535, nullable=True)
        return schema

    def preflight(self):
        self.source = MilvusClient(uri=SOURCE_URI, db_name='default', timeout=20)
        self.target = MilvusClient(uri=TARGET_URI, db_name='default', timeout=20)
        self.target_collections_before = set(self.target.list_collections())
        if self.collection in self.target_collections_before:
            raise IntegrityError('Refusing to touch a pre-existing test collection')
        self.source_schema = self.source.describe_collection(SOURCE_COLLECTION)
        fields = {field['name']:field for field in self.source_schema['fields']}
        if 'text_embed' not in fields or fields['text_embed']['type'] != DataType.FLOAT_VECTOR:
            raise IntegrityError('Expected original float32 text_embed vector is absent')
        if FIELD in fields and (fields[FIELD]['type'] != DataType.ARRAY or
                fields[FIELD].get('element_type') != DataType.VARCHAR):
            raise IntegrityError('Source-ID field must remain an array of strings')
        self.source_fields = list(fields)
        self.original_fields = [name for name in fields if name != FIELD]
        self.report['source_count_before'] = self.count(self.source,SOURCE_COLLECTION)
        if self.report['source_count_before'] != EXPECTED_SOURCE_COUNT:
            raise IntegrityError('Production count differs from the protected original count')
        self.report['source_field_count_before'] = (self.count(self.source,SOURCE_COLLECTION,FIELD+' is not null')
            if FIELD in fields else None)
        (self.root/'source_schema.json').write_text(json.dumps(self.source_schema,ensure_ascii=False,indent=2),encoding='utf-8')
        schema = self.clone_schema(self.source_schema)
        started = time.monotonic()
        RestrictedTarget(self.target,self.collection).create(schema)
        if self.target.list_indexes(self.collection):
            raise IntegrityError('Copy benchmark requires an initially unindexed new collection')
        self.report['seconds']['create_fresh_unindexed_collection'] = time.monotonic()-started
        self.report['target_collections_before'] = sorted(self.target_collections_before)
        self.save()

    def next_rows(self, batch_size):
        self.before_operation('source_read')
        if batch_size not in (100,200):
            raise IntegrityError('Only the approved 100/200 vector read batches are allowed')
        remaining = self.row_limit-self.report['read_rows']
        if remaining <= 0:
            return None, []
        while True:
            if self.active_prefix is not None:
                quota = self.row_limit//len(self.buckets)+(self.buckets.index(self.active_prefix)<self.row_limit % len(self.buckets))
                if (self.active_batch_size != batch_size or
                        self.active_prefix_rows >= PREFIX_TURN_ROWS or
                        self.bucket_rows[self.active_prefix] >= quota):
                    self.iterators.pop(self.active_prefix).close()
                    self.active_prefix = None
            if self.active_prefix is None:
                prefix = None
                for _ in range(len(self.buckets)):
                    candidate = self.buckets[self.next_bucket % len(self.buckets)]
                    self.next_bucket += 1
                    quota = self.row_limit//len(self.buckets)+(self.buckets.index(candidate)<self.row_limit % len(self.buckets))
                    if candidate not in self.exhausted and self.bucket_rows[candidate] < quota:
                        prefix = candidate
                        break
                if prefix is None:
                    return None,[]
                self.active_prefix = prefix
                self.active_batch_size = batch_size
                self.active_prefix_rows = 0
                high = format(int(prefix,16)+1,'03x') if prefix!='fff' else 'g'
                expression = f'doc_id >= "{prefix}" and doc_id < "{high}"'
                cursor = self.last_ids.get(prefix)
                if cursor:
                    expression += ' and id > '+canonical(cursor)
                limit = min(PREFIX_TURN_ROWS,quota-self.bucket_rows[prefix],remaining)
                self.iterators[prefix] = self.source.query_iterator(SOURCE_COLLECTION, filter=expression,
                    batch_size=min(batch_size,limit), limit=limit,
                    output_fields=self.source_fields, consistency_level='Strong', timeout=90,
                    reduce_stop_for_best=False)
            prefix = self.active_prefix
            iterator = self.iterators[prefix]
            started = time.monotonic()
            rows = [plain(dict(row)) for row in iterator.next()]
            elapsed = time.monotonic()-started
            if len(rows)>MAX_VECTOR_ROWS:
                raise IntegrityError('Source returned an oversized vector batch')
            if not rows:
                iterator.close()
                del self.iterators[prefix]
                self.active_prefix = None
                self.exhausted.add(prefix)
                continue
            ids = [row['id'] for row in rows]
            if ids != sorted(ids) or len(ids) != len(set(ids)) or (
                    prefix in self.last_ids and ids[0]<=self.last_ids[prefix]):
                raise IntegrityError('Source iterator ordering would repeat or skip a primary key')
            if any(not row['doc_id'].startswith(prefix) for row in rows):
                raise IntegrityError('Source prefix filter was not respected')
            self.last_ids[prefix] = ids[-1]
            self.active_prefix_rows += len(rows)
            self.bucket_rows[prefix] += len(rows)
            self.report['read_rows'] += len(rows)
            return elapsed,rows

    def copy_batch(self, seq, phase, rows, read_seconds):
        source,target,session = self.worker()
        timing = {'source_iterator_read':read_seconds}
        started = time.monotonic()
        duplicates = duplicate_counts(source,[row['id'] for row in rows])
        timing['physical_duplicate_check'] = time.monotonic()-started
        if duplicates:
            with self.lock:
                for key,n in duplicates.items():
                    self.db.execute('INSERT OR IGNORE INTO excluded_duplicate_keys VALUES(?,?)',(key,n))
                self.db.commit()
                self.report['excluded_duplicate_keys'] = self.db.execute('SELECT COUNT(*) FROM excluded_duplicate_keys').fetchone()[0]
                self.report['excluded_physical_versions'] = self.db.execute('SELECT COALESCE(SUM(physical_count),0) FROM excluded_duplicate_keys').fetchone()[0]
                (self.root/'excluded_duplicate_keys.json').write_text(json.dumps([
                    {'id':key,'physical_count':count} for key,count in self.db.execute(
                        'SELECT id,physical_count FROM excluded_duplicate_keys ORDER BY id')],
                    ensure_ascii=False,indent=2),encoding='utf-8')
            if not self.skip_duplicates:
                raise IntegrityError('Historical duplicate primary keys identified; details in audit.sqlite; no silent deduplication allowed')
            # The explicit switch excludes entire duplicated keys solely from
            # this benchmark. All their original versions stay untouched.
            rows = [row for row in rows if row['id'] not in duplicates]
        started = time.monotonic()
        values = self.sources([row['doc_id'] for row in rows],session)
        timing['solr_and_cache'] = time.monotonic()-started
        started = time.monotonic()
        audit,added,missing = patch_rows(rows,values,self.original_fields)
        timing['original_hash_and_prepare'] = time.monotonic()-started
        with self.lock:
            for row_no,(key,digest,before,after) in enumerate(audit):
                if self.db.execute('SELECT 1 FROM expected_rows WHERE id=?',(key,)).fetchone():
                    raise IntegrityError('Iterator repeated a previously copied key; benchmark cannot prove sample completeness')
                self.db.execute('INSERT INTO expected_rows(seq,row_no,id,original_hash,source_field_before,source_field_after) VALUES(?,?,?,?,?,?)',
                    (seq,row_no,key,digest,before,after))
            self.db.execute('UPDATE batches SET status=? WHERE seq=?',('prepared',seq))
            self.db.commit()
        started = time.monotonic()
        if rows:
            self.before_operation('target_insert')
            with self.lock:
                self.db.execute('UPDATE batches SET status=? WHERE seq=?',('submitted_outcome_not_yet_confirmed',seq))
                self.db.commit()
            result = target.insert(self.collection,rows)
            if int(result.get('insert_count',-1)) != len(rows):
                raise IntegrityError('Isolated target insertion count mismatch')
        timing['isolated_insert_rpc'] = time.monotonic()-started
        result = {'rows_scanned':len(audit)+len(duplicates),'rows_inserted':len(rows),
            'source_ids_added':added,'source_ids_missing':missing,'timing_seconds':timing,
            'original_hashes_persisted':True,'readback_verified':False,'finished_at':now()}
        with self.lock:
            self.db.execute('UPDATE batches SET status=?,inserted=?,result=? WHERE seq=?',
                ('inserted_pending_verification',len(rows),canonical(result),seq))
            self.db.commit()
            self.report['inserted_rows_pending_verification'] += len(rows)
            self.report['source_ids_added'] += added
            self.report['source_ids_missing'] += missing
        return result

    def copy_phase(self, name, batch_size, row_budget, seconds):
        workers = getattr(self,'workers',2)
        if workers not in (1,2,4):
            raise IntegrityError('Only 1, 2 or 4 bounded workers are allowed')
        self.report['status'] = 'copying_unindexed_'+name
        phase = {'name':name,'batch_size':batch_size,'workers':workers,
            'dispatch_window_seconds':seconds,'maximum_scanned_rows':row_budget}
        started = time.monotonic()
        before_read = self.report['read_rows']
        before_insert = self.report['inserted_rows_pending_verification']
        deadline = started+seconds
        pending = []
        results = []
        last_progress = 0
        error = None
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    while self.report['read_rows']-before_read<row_budget and time.monotonic()<deadline:
                        if (self.root/'STOP_REQUESTED').exists():
                            phase['dispatch_stop_reason'] = 'local_stop_requested'
                            break
                        if shutil.disk_usage(self.root).free<10*1024**3:
                            raise RuntimeError('Local audit disk has less than 10 GiB free')
                        if len(pending)>=workers:
                            results.append(pending.pop(0).result())
                        if time.monotonic()>=deadline:
                            break
                        read_seconds,rows = self.next_rows(batch_size)
                        if not rows:
                            phase['dispatch_stop_reason'] = 'sample_row_limit_or_prefixes_exhausted'
                            break
                        if time.monotonic()>=deadline or (self.root/'STOP_REQUESTED').exists():
                            # Read-only discarded selections are listed explicitly;
                            # this benchmark has no resumption or migration claim.
                            phase['read_not_dispatched_rows'] = len(rows)
                            break
                        self.batch_sequence += 1
                        with self.lock:
                            self.db.execute('INSERT INTO batches(seq,phase,status,rows) VALUES(?,?,?,?)',
                                (self.batch_sequence,name,'queued',len(rows)))
                            self.db.commit()
                        pending.append(pool.submit(self.copy_batch,self.batch_sequence,name,rows,read_seconds))
                        if time.monotonic()-last_progress>=15:
                            self.progress()
                            last_progress = time.monotonic()
                finally:
                    # Network or integrity failures never kill an in-flight
                    # insertion; collect outcomes and keep the test collection.
                    errors = []
                    for future in pending:
                        try:
                            results.append(future.result())
                        except BaseException as exc:
                            errors.append(exc)
                    if errors:
                        raise errors[0]
        except BaseException as exc:
            error = exc
        finally:
            elapsed = time.monotonic()-started
            inserted = self.report['inserted_rows_pending_verification']-before_insert
            phase.update(elapsed_seconds_including_inflight_waits=elapsed,
                rows_scanned=self.report['read_rows']-before_read,rows_inserted=inserted,
                copy_rows_per_second_including_waits=inserted/max(elapsed,0.001),
                readback_verified=False,timing_totals_seconds={})
            for result in results:
                for key,value in result['timing_seconds'].items():
                    phase['timing_totals_seconds'][key] = phase['timing_totals_seconds'].get(key,0)+value
            if error:
                phase['error'] = repr(error)
            self.report['phases'].append(phase)
            # Close old phase iterators before changing the read batch size.
            for iterator in self.iterators.values():
                iterator.close()
            self.iterators.clear()
            self.active_prefix = None
            self.active_batch_size = None
            self.active_prefix_rows = 0
            self.save()
        if error:
            raise error

    def before_operation(self, operation):
        """Override for the resource guard; no retries of ambiguous inserts."""

    def build_readback_index(self):
        self.report['status'] = 'building_flat_index_for_verification_only'
        self.progress()
        indexes = self.target.prepare_index_params()
        indexes.add_index(field_name='text_embed',index_name='verification_only_text_embed',
            index_type='FLAT',metric_type='COSINE')
        started = time.monotonic()
        self.target.create_index(self.collection,index_params=indexes,sync=True,timeout=14400)
        self.report['seconds']['flat_index_for_verification_only'] = time.monotonic()-started

    def verify(self):
        if (self.root/'STOP_REQUESTED').exists():
            raise RuntimeError('Stopped before target flush and verification index; test collection retained')
        inserted = self.report['inserted_rows_pending_verification']
        if not inserted:
            raise IntegrityError('No inserted rows are available to verify')
        self.report['status'] = 'flushing_new_test_collection'
        self.progress()
        started = time.monotonic()
        self.target.flush(self.collection,timeout=300)
        self.report['seconds']['target_flush'] = time.monotonic()-started
        # Milvus needs a vector index to load and read all vector fields. FLAT
        # avoids production IVF_PQ training and is created only after copy timing.
        if (self.root/'STOP_REQUESTED').exists():
            raise RuntimeError('Stopped before verification index; test collection retained')
        self.before_operation('index_build')
        self.build_readback_index()
        if (self.root/'STOP_REQUESTED').exists():
            raise RuntimeError('Stopped after verification index; no load or readback started')
        started = time.monotonic()
        self.target.load_collection(self.collection,timeout=600)
        self.report['seconds']['load_test_collection'] = time.monotonic()-started
        self.report['status'] = 'verifying_all_copied_rows_and_vectors'
        self.progress()
        started = time.monotonic()
        count = self.count(self.target,self.collection)
        if count != inserted:
            raise IntegrityError('New test collection count differs from insert confirmations')
        verified = 0
        iterator = self.target.query_iterator(self.collection,batch_size=MAX_VECTOR_ROWS,
            output_fields=self.original_fields+[FIELD],consistency_level='Strong',timeout=90,
            reduce_stop_for_best=False)
        try:
            while True:
                self.before_operation('target_verify_read')
                rows = iterator.next()
                if not rows:
                    break
                if len(rows)>MAX_VECTOR_ROWS:
                    raise IntegrityError('Verification returned an oversized vector batch')
                with self.lock:
                    for raw in rows:
                        row = plain(dict(raw))
                        entry = self.db.execute('SELECT rowid,original_hash,source_field_after FROM expected_rows WHERE id=? AND verified=0',
                            (row['id'],)).fetchall()
                        wanted = (original_hash(row,self.original_fields),canonical(row.get(FIELD)))
                        match = next((item for item in entry if (item[1],item[2])==wanted),None)
                        if match is None:
                            raise IntegrityError('Target original text/vector/source multiset mismatch for '+row['id'])
                        self.db.execute('UPDATE expected_rows SET verified=1 WHERE rowid=?',(match[0],))
                        verified += 1
                    self.db.commit()
                if verified % 10000 == 0:
                    self.report['verified_rows'] = verified
                    self.progress()
        finally:
            iterator.close()
        unverified = self.db.execute("SELECT COUNT(*) FROM expected_rows WHERE verified=0 AND seq IN (SELECT seq FROM batches WHERE status IN ('inserted_pending_verification','completed_and_verified'))").fetchone()[0]
        if verified != inserted or unverified:
            raise IntegrityError('Full target stream omitted an original row/version')
        self.report['seconds']['all_target_rows_text_vector_source_verification'] = time.monotonic()-started
        self.report['verified_rows'] = verified
        self.report['target_count_after'] = count
        # Read the original source again in <=200-row groups to substantiate the
        # sampled original fields/vector values remaining unchanged.
        self.report['status'] = 'verifying_production_sample_remains_unchanged'
        self.progress()
        started = time.monotonic()
        cursor = self.db.execute("SELECT id,original_hash,source_field_before FROM expected_rows WHERE seq IN (SELECT seq FROM batches WHERE status IN ('inserted_pending_verification','completed_and_verified')) ORDER BY seq,row_no")
        checked = 0
        while True:
            self.before_operation('source_verify_read')
            expected = cursor.fetchmany(MAX_VECTOR_ROWS)
            if not expected:
                break
            rows = self.source.get(SOURCE_COLLECTION,ids=[item[0] for item in expected],
                output_fields=self.source_fields,consistency_level='Strong',timeout=90)
            if len(rows) != len(expected):
                raise IntegrityError('Production readback omitted sampled primary keys')
            actual = {row['id']:plain(dict(row)) for row in rows}
            for key,digest,field in expected:
                row = actual.get(key)
                if row is None or original_hash(row,self.original_fields)!=digest or canonical(row.get(FIELD))!=field:
                    raise IntegrityError('Original source sample changed for '+key)
                checked += 1
            if checked % 10000 == 0:
                self.report['source_sample_rows_unchanged'] = checked
                self.progress()
        self.report['seconds']['all_source_sample_rows_unchanged_verification'] = time.monotonic()-started
        self.report['source_sample_rows_unchanged'] = checked
        self.report['source_count_after'] = self.count(self.source,SOURCE_COLLECTION)
        self.report['source_field_count_after'] = (self.count(self.source,SOURCE_COLLECTION,FIELD+' is not null')
            if FIELD in self.source_fields else None)
        if (self.report['source_count_after'],self.report['source_field_count_after']) != (
                self.report['source_count_before'],self.report['source_field_count_before']):
            raise IntegrityError('Production collection counts changed during the experiment')
        if self.source.describe_collection(SOURCE_COLLECTION) != self.source_schema:
            raise IntegrityError('Production collection schema changed during the experiment')
        if set(self.target.list_collections()) != self.target_collections_before|{self.collection}:
            raise IntegrityError('Isolated instance collection inventory changed unexpectedly')
        with self.lock:
            self.db.execute("UPDATE batches SET status='completed_and_verified',result=json_set(result,'$.readback_verified',json('true')) WHERE status='inserted_pending_verification'")
            self.db.commit()
            for phase in self.report['phases']:
                phase['readback_verified'] = True
        self.report['all_copied_original_fields_vectors_and_source_ids_verified'] = True
        self.report['source_global_counts_and_schema_unchanged'] = True

    def run(self):
        self.last_ids = {}
        overall = time.monotonic()
        try:
            self.preflight()
            self.copy_phase('batch100_workers2',100,self.row_limit//2,self.seconds/2)
            if not (self.root/'STOP_REQUESTED').exists():
                self.copy_phase('batch200_workers2',200,self.row_limit-self.report['read_rows'],self.seconds/2)
            self.report['seconds']['copy_phases_including_waits'] = sum(
                phase['elapsed_seconds_including_inflight_waits'] for phase in self.report['phases'])
            self.report['copy_rows_per_second_excluding_index_and_verification'] = (
                self.report['inserted_rows_pending_verification']/max(self.report['seconds']['copy_phases_including_waits'],0.001))
            if (self.root/'STOP_REQUESTED').exists():
                self.report['status'] = 'stopped_by_user_test_collection_preserved_unverified'
                self.report['finished_at'] = now()
                return
            self.verify()
            self.report['status'] = 'completed_sample_and_verified_not_production_ready'
            self.report['finished_at'] = now()
        except BaseException as exc:
            self.report['status'] = 'failed_new_test_collection_preserved'
            self.report['error'] = repr(exc)
            self.report['finished_at'] = now()
            raise
        finally:
            self.report['seconds']['total_wall_including_preflight_and_readback'] = time.monotonic()-overall
            self.progress()
            for iterator in self.iterators.values():
                iterator.close()
            for source,target,session in self.worker_resources:
                session.close()
                target.close()
                source.close()
            if self.target:
                self.target.close()
            if self.source:
                self.source.close()
            self.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--rows',type=int,default=600000)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--collection')
    parser.add_argument('--workdir')
    parser.add_argument('--skip-duplicate-keys-for-benchmark',action='store_true')
    args = parser.parse_args()
    if not args.apply:
        parser.error('--apply is required; no source or destination is contacted otherwise')
    suffix = time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)
    collection = args.collection or COLLECTION_PREFIX+suffix
    root = args.workdir or PROJECT/'outputs'/('fulltext_source_id_isolated231_throughput_'+suffix)
    Experiment(root,collection,args.rows,args.seconds,args.skip_duplicate_keys_for_benchmark).run()


if __name__=='__main__':
    main()
