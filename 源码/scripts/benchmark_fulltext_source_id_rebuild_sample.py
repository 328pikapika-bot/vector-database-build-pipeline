"""Bounded real-row rebuild benchmark in a NEW isolated 230 Milvus instance.

Source collection is read only. The test collection is created only in the
separate instance at port 19539. No existing collection is updated or deleted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from pymilvus import MilvusClient
import requests

from fulltext_source_id_safe_common import original_hash, valid_source, now
from benchmark_fulltext_source_id_100 import plain


SOURCE = os.environ.get('BENCH_SOURCE_URI', 'http://192.0.2.14:19533')
BENCH = os.environ.get('BENCH_TARGET_URI', 'http://192.0.2.14:19539')
COLLECTION = 'scientific_fulltext_chunks_v2'
TEST_COLLECTION = os.environ.get('BENCH_COLLECTION', 'fulltext_source_id_rebuild_sample_20261114_r4')
FIELD = 'hasSourcesysSourceId'
SOLR_FIELD = 'stringIMS_hasSourcesysSourceId'
SOLR = 'http://service.example.invalid:8984/solr/paper6.6/select'
BUCKETS = tuple(json.loads(os.environ['BENCH_BUCKETS'])) if 'BENCH_BUCKETS' in os.environ else tuple(sorted({*(format(i*100, '03x') for i in range(40)), '020'}))
ROOT = Path(os.environ.get('BENCH_REPORT_ROOT',
    str(Path(__file__).resolve().parents[1] / 'outputs/fulltext_source_id_rebuild_sample_20261114')))


def main() -> None:
    p = argparse.ArgumentParser()
    authorized_rows = int(os.environ.get('BENCH_AUTHORIZED_ROWS', '1435000'))
    p.add_argument('--rows', type=int, default=authorized_rows)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if not a.apply or a.rows != authorized_rows or not BENCH.endswith(':19539') or not SOURCE.endswith(':19533') or not TEST_COLLECTION.startswith('fulltext_source_id_'):
        p.error('Explicit apply, exact authorized row count and isolated test endpoint required')
    ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / os.environ.get('BENCH_REPORT_NAME', 'report_r4.json')
    if report_path.exists():
        raise RuntimeError('Report already exists; refusing to repeat or overwrite test')
    report = {'started_at': now(), 'source': SOURCE, 'benchmark': BENCH,
              'source_collection': COLLECTION, 'test_collection': TEST_COLLECTION,
              'planned_rows': a.rows, 'status': 'preflight', 'seconds': {},
              'source_rows': 0, 'inserted_rows': 0, 'source_ids_added': 0,
              'source_ids_missing': 0, 'duplicate_primary_keys': 0}
    def save(verbose=True):
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if verbose:
            print(json.dumps({k: report[k] for k in ('status','source_rows','inserted_rows','source_ids_added','source_ids_missing')}, ensure_ascii=False), flush=True)
    save()
    src = MilvusClient(uri=SOURCE, db_name='default', timeout=20)
    dst = MilvusClient(uri=BENCH, db_name='default', timeout=20)
    session = requests.Session()
    session.trust_env = False
    expected = Counter()
    seen_ids = set()
    try:
        if TEST_COLLECTION in dst.list_collections():
            raise RuntimeError('Test collection already exists; refusing to alter it')
        expected_collections = set(json.loads(os.environ.get('BENCH_EXISTING_COLLECTIONS',
            '["fulltext_source_id_rebuild_sample_20261114", "fulltext_source_id_rebuild_sample_20261114_r2", "fulltext_source_id_rebuild_sample_20261114_r3"]')))
        if set(dst.list_collections()) != expected_collections:
            raise RuntimeError('Independent test instance is not empty')
        src_schema = src.describe_collection(COLLECTION)
        original_fields = [f['name'] for f in src_schema['fields'] if f['name'] != FIELD]
        report['source_count_before'] = int(src.query(COLLECTION, filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)'])
        report['source_field_count_before'] = int(src.query(COLLECTION, filter=FIELD+' is not null', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)'])
        if report['source_count_before'] != 195893017:
            raise RuntimeError('Source count unexpected')
        schema = dst.create_schema(auto_id=False, enable_dynamic_field=False,
                                   description=f'Isolated {a.rows} real-row source-ID rebuild benchmark')
        for f in src_schema['fields']:
            args = {'field_name': f['name'], 'datatype': f['type'],
                    'description': f.get('description', ''), **f.get('params', {})}
            if f.get('is_primary'):
                args['is_primary'] = True
            if f.get('nullable'):
                args['nullable'] = True
            if f.get('element_type') is not None:
                args['element_type'] = f['element_type']
            schema.add_field(**args)
        t = time.monotonic()
        dst.create_collection(TEST_COLLECTION, schema=schema, timeout=60)
        report['seconds']['create_no_index_collection'] = time.monotonic()-t
        report['status'] = 'copying_real_rows'
        save()
        per_bucket = a.rows // len(BUCKETS)
        for prefix in BUCKETS:
            high = format(int(prefix, 16)+1, '03x')
            iterator = src.query_iterator(COLLECTION,
                filter=f'doc_id >= "{prefix}" and doc_id < "{high}"',
                batch_size=100, limit=per_bucket, output_fields=['*'],
                consistency_level='Strong', timeout=90)
            bucket_rows = 0
            try:
                while bucket_rows < per_bucket:
                    t = time.monotonic()
                    rows = [plain(dict(r)) for r in iterator.next()]
                    report['seconds']['read'] = report['seconds'].get('read', 0) + time.monotonic()-t
                    if not rows:
                        break
                    bucket_rows += len(rows)
                    report['source_rows'] += len(rows)
                    doc_ids = sorted({r['doc_id'] for r in rows})
                    t = time.monotonic()
                    response = session.get(SOLR, params={'q': '{!terms f=id}' + ','.join(doc_ids),
                        'fl': 'id,'+SOLR_FIELD, 'rows': len(doc_ids), 'wt': 'json',
                        'timeAllowed': 20000}, timeout=(10, 40))
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get('responseHeader', {}).get('partialResults'):
                        raise RuntimeError('Solr returned partial results')
                    docs = payload['response']['docs']
                    if payload['response']['numFound'] != len(docs) or len({d['id'] for d in docs}) != len(docs):
                        raise RuntimeError('Ambiguous Solr results')
                    source_ids = {d['id']: d.get(SOLR_FIELD) for d in docs}
                    report['seconds']['solr'] = report['seconds'].get('solr', 0) + time.monotonic()-t
                    t = time.monotonic()
                    for r in rows:
                        old = r.get(FIELD)
                        new = source_ids.get(r['doc_id'])
                        if valid_source(new):
                            if old is not None and old != new:
                                raise RuntimeError('Existing source ID conflicts with Solr for '+r['id'])
                            if old is None:
                                r[FIELD] = new
                                report['source_ids_added'] += 1
                        elif old is None:
                            report['source_ids_missing'] += 1
                        if r['id'] in seen_ids:
                            report['duplicate_primary_keys'] += 1
                        seen_ids.add(r['id'])
                        expected[(original_hash(r, original_fields), json.dumps(r.get(FIELD), ensure_ascii=False))] += 1
                    report['seconds']['hash'] = report['seconds'].get('hash', 0) + time.monotonic()-t
                    t = time.monotonic()
                    result = dst.insert(TEST_COLLECTION, data=rows, timeout=120)
                    report['seconds']['insert'] = report['seconds'].get('insert', 0) + time.monotonic()-t
                    if result.get('insert_count') != len(rows):
                        raise RuntimeError('Insert count mismatch')
                    report['inserted_rows'] += len(rows)
                    save(verbose=(report['inserted_rows'] % 5000 == 0))
            finally:
                iterator.close()
            if bucket_rows != per_bucket:
                raise RuntimeError(f'Bucket {prefix} returned only {bucket_rows}/{per_bucket} rows')
        t = time.monotonic()
        dst.flush(TEST_COLLECTION, timeout=120)
        report['seconds']['flush'] = time.monotonic()-t
        report['status'] = 'building_indexes'
        save()
        indexes = dst.prepare_index_params()
        indexes.add_index(field_name='text_embed', index_type='IVF_PQ', index_name='text_embed',
                          metric_type='COSINE', params={'nlist': int(os.environ.get('BENCH_NLIST', '32768')), 'm': 64, 'nbits': 8})
        for name in ('doc_id','year','accessType','classificationCode',
                     'hasAuthorname','hasCreateOrganizationname','language'):
            index_name = next(i for i in src.list_indexes(COLLECTION)
                if src.describe_index(COLLECTION,index_name=i)['field_name'] == name)
            indexes.add_index(field_name=name, index_type='INVERTED', index_name=index_name)
        t = time.monotonic()
        dst.create_index(TEST_COLLECTION, index_params=indexes, timeout=14400, sync=True)
        report['seconds']['all_indexes'] = time.monotonic()-t
        t = time.monotonic()
        dst.load_collection(TEST_COLLECTION, timeout=300)
        report['seconds']['load'] = time.monotonic()-t
        report['status'] = 'verifying'
        save()
        t = time.monotonic()
        actual_count = int(dst.query(TEST_COLLECTION, filter='', output_fields=['count(*)'],
            consistency_level='Strong', timeout=90)[0]['count(*)'])
        report['test_count_after'] = actual_count
        if actual_count != a.rows:
            raise RuntimeError('Test count mismatch')
        got = Counter()
        it = dst.query_iterator(TEST_COLLECTION, batch_size=100, limit=a.rows,
            output_fields=['*'], consistency_level='Strong', timeout=90)
        try:
            while True:
                rows = it.next()
                if not rows:
                    break
                for r in rows:
                    row = plain(dict(r))
                    got[(original_hash(row, original_fields), json.dumps(row.get(FIELD), ensure_ascii=False))] += 1
        finally:
            it.close()
        if got != expected:
            raise RuntimeError(f'Full-row/vector/source multiset mismatch: {sum(got.values())} vs {sum(expected.values())}')
        report['seconds']['verification'] = time.monotonic()-t
        report['source_count_after'] = int(src.query(COLLECTION, filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)'])
        report['source_field_count_after'] = int(src.query(COLLECTION, filter=FIELD+' is not null', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)'])
        if (report['source_count_after'], report['source_field_count_after']) != (report['source_count_before'], report['source_field_count_before']):
            raise RuntimeError('Original source collection counts changed')
        report['status'] = 'completed_and_verified'
        report['finished_at'] = now()
        save()
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        save()
        raise
    finally:
        session.close()
        dst.close()
        src.close()


if __name__ == '__main__':
    main()
