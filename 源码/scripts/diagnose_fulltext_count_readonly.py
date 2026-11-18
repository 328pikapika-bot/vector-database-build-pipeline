"""Read-only historical-MVCC audit of count changes; no Milvus mutations."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import time

from pymilvus.client.constants import ITERATOR_FIELD
from pymilvus.grpc_gen import milvus_pb2
from benchmark_fulltext_source_id_100 import COLLECTION, connect, plain
from run_fulltext_source_id_backfill import decode, original_hash

ROOT = Path('/data/scientific-fulltext-source-id-backfill-20261113')
HIST_TS = int(datetime(2026, 11, 13, 13, 54, tzinfo=timezone(timedelta(hours=8))).timestamp()*1000)<<18


def run():
    started = time.monotonic()
    db = sqlite3.connect('file:'+str(ROOT/'progress.sqlite')+'?mode=ro', uri=True)
    cursor = json.loads(db.execute("SELECT value FROM state WHERE key='cursor'").fetchone()[0])
    batches = [decode(r[0]) for r in db.execute("SELECT payload FROM batches WHERE status='completed' ORDER BY seq")]
    expected_ids = {i for b in batches for i in b['ids']}
    expected_hashes = {i: h for b in batches for i,h in b['original_hashes'].items()}
    c = connect('192.0.2.15')
    stub = c._get_connection()._stub

    def count(expr, historical=False):
        req = milvus_pb2.QueryRequest(db_name='default', collection_name=COLLECTION, expr=expr,
            output_fields=['count(*)'], guarantee_timestamp=HIST_TS if historical else 0,
            consistency_level=0, use_default_consistency=False)
        if historical:
            req.query_params.add(key=ITERATOR_FIELD, value='True')
            req.query_params.add(key='reduce_stop_for_best', value='False')
        result = stub.Query(req, timeout=60)
        if result.status.error_code or result.status.code:
            raise RuntimeError(str(result.status))
        return int(result.fields_data[0].scalars.long_data.data[0])

    expr = 'doc_id >= "000" and doc_id < "003" and id <= '+json.dumps(cursor)
    report = {'historical_ts': HIST_TS, 'historical_time': '2026-11-13T13:54:00+08:00',
              'scope_cursor': cursor, 'journal_unique_keys': len(expected_ids),
              'global_count_before': count('', True), 'global_count_now': count(''),
              'scope_count_before': count(expr, True), 'scope_count_now': count(expr),
              'original_baseline_count': json.loads(db.execute("SELECT value FROM state WHERE key='initial_count'").fetchone()[0]),
              'duplicate_keys': [], 'duplicate_extra_versions': 0,
              'historical_unavailable_keys': 0,
              'historical_snapshot_note': 'Old MVCC results can lose compacted versions; absence in this view is not evidence of current data loss.'}
    print(json.dumps(report), flush=True)
    historical_ids = set()
    for bucket in range(3):
        expr_bucket = 'doc_id >= '+json.dumps(format(bucket,'03x'))+' and doc_id < '+json.dumps(format(bucket+1,'03x'))+' and id <= '+json.dumps(cursor)
        it = c.query_iterator(COLLECTION, filter=expr_bucket, batch_size=1000,
            output_fields=['id'], consistency_level='Strong', timeout=60)
        it._session_ts = HIST_TS
        it._kwargs['guarantee_timestamp'] = HIST_TS
        try:
            while True:
                rows = it.next()
                if not rows:
                    break
                historical_ids.update(r['id'] for r in rows)
        finally:
            it.close()
        print('HISTORICAL_KEYS', bucket, len(historical_ids), flush=True)
    report['historical_unique_keys'] = len(historical_ids)
    report['historical_keyset_matches_journal'] = historical_ids == expected_ids
    report['missing_keys'] = sorted(historical_ids-expected_ids)
    report['extra_journal_keys'] = sorted(expected_ids-historical_ids)

    def find_duplicates(ids, total=None):
        n = count('id in '+json.dumps(ids), True) if total is None else total
        if n < len(ids):
            report['historical_unavailable_keys'] += len(ids)-n
            return
        if n == len(ids):
            return
        if len(ids) == 1:
            report['duplicate_keys'].append({'id':ids[0], 'historical_versions':n})
            report['duplicate_extra_versions'] += n-1
            (ROOT/'count_diagnosis_partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            return
        half = len(ids)//2
        find_duplicates(ids[:half])
        find_duplicates(ids[half:])

    for index, batch in enumerate(batches):
        find_duplicates(batch['ids'])
        if (index+1) % 25 == 0:
            print('BATCHES_AUDITED', index+1, 'EXTRA_VERSIONS', report['duplicate_extra_versions'], flush=True)

    fields = [f['name'] for f in c.describe_collection(COLLECTION)['fields'] if f['name']!='hasSourcesysSourceId']
    ids = [d['id'] for d in report['duplicate_keys']]
    if ids:
        rows = c._get_connection().query(collection_name=COLLECTION,
            expr='id in '+json.dumps(ids), output_fields=fields, partition_names=[], timeout=60,
            limit=16000, guarantee_timestamp=HIST_TS, consistency_level=0,
            **{ITERATOR_FIELD:'True'})
        hashes = {}
        for row in rows:
            row = {k:(v if k=='text_embed' else plain(v)) for k,v in row.items()}
            hashes[row['id']] = original_hash(row, fields)
        report['historical_duplicate_visible_originals_match_journal'] = (
            set(hashes)==set(ids) and all(hashes[i]==expected_hashes[i] for i in ids))
        # A local audit archive of historical visible originals, never inserted.
        import gzip
        with gzip.open(ROOT/'historical_duplicate_visible_originals.jsonl.gz','wt',encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(plain(dict(row)),ensure_ascii=False)+'\n')
    report['count_delta_exactly_duplicate_extras'] = (
        report['original_baseline_count']-report['global_count_now'] == report['duplicate_extra_versions'])
    report['elapsed_seconds'] = time.monotonic()-started
    (ROOT/'count_diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSIS_COMPLETE',json.dumps(report),flush=True)
    c.close()
    db.close()


if __name__=='__main__':
    run()
