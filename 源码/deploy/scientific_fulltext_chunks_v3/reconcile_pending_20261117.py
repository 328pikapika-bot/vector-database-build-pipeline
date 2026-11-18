"""One-time replay of two inserts proven absent after the 2026-11-17 crash.

Run only while all copy workers are stopped. Evidence and the original journal
are retained. Every replay still uses the runner's single-RPC insert journal.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, '/app')
import run_fulltext_v3_prod_fullcopy231 as copy


BACKUP = copy.ROOT / 'recovery_backups_20261117'
copy.STATUS_NAME = str(BACKUP / 'status_reconciliation.json')
EXPECTED_SEQUENCES = {486813, 486814}
EXPECTED_BASELINE = 97687028


def main():
    evidence_path = BACKUP / 'pending_reconciliation_evidence.json'
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    ids_evidence = json.loads((BACKUP / 'pending_ids_and_last_confirmed.json').read_text())
    direct = evidence['direct_pk_results']
    assert evidence['target_collection'] == copy.TARGET
    assert evidence['flushed_rows'] == EXPECTED_BASELINE
    assert evidence['journal_confirmed_plus_base'] == EXPECTED_BASELINE
    assert evidence['growing_segments'] == evidence['sealed_segments'] == evidence['flushing_segments'] == 0
    assert {r['seq'] for r in direct['pending']} == EXPECTED_SEQUENCES
    assert all(r['ids_checked'] == 200 and r['found'] == 0 for r in direct['pending'])
    assert len(direct['confirmed_controls']) == 3
    assert all(r['ids_checked'] == r['found'] == 200 for r in direct['confirmed_controls'])
    assert evidence['checkpoint_tso'] > direct['max_insert_tso']
    id_proofs = {r['seq']: r for r in ids_evidence['pending']}
    assert set(id_proofs) == EXPECTED_SEQUENCES

    runner = copy.Runner()
    source = copy.MilvusClient(uri=copy.URI, db_name='default', timeout=20)
    target = copy.MilvusClient(uri=copy.URI, db_name='default', timeout=20)
    session = copy.requests.Session()
    session.trust_env = False
    report = {'started_at': copy.now(), 'evidence_sha256': hashlib.sha256(evidence_bytes).hexdigest(), 'batches': []}
    try:
        runner.guard.check()
        assert not source.list_indexes(copy.TARGET)
        assert int(source.get_collection_stats(copy.TARGET)['row_count']) == EXPECTED_BASELINE
        fields = [f['name'] for f in source.describe_collection(copy.SOURCE_COLLECTION)['fields']]
        originals = [f for f in fields if f != copy.FIELD]
        runner.db.row_factory = sqlite3.Row
        pending = [dict(r) for r in runner.db.execute("SELECT * FROM batches WHERE status='submitted' ORDER BY seq")]
        runner.db.row_factory = None
        assert {r['seq'] for r in pending} == EXPECTED_SEQUENCES
        prepared = []
        for batch in pending:
            ids = json.loads(batch['ids'])
            assert ids == sorted(ids) and len(ids) == len(set(ids)) == batch['rows'] == 200
            assert ids == json.loads(id_proofs[batch['seq']]['ids'])
            rows = [copy.plain(dict(r)) for r in source.query(
                copy.SOURCE_COLLECTION, filter='id in ' + copy.canonical(ids),
                output_fields=fields, consistency_level='Strong', timeout=90)]
            rows.sort(key=lambda r: r['id'])
            assert [r['id'] for r in rows] == ids
            assert not runner.already_inserted(runner.old_db, ids)
            assert not copy.duplicate_counts(source, ids)
            digest = hashlib.sha256()
            for row in rows:
                digest.update(bytes.fromhex(copy.original_hash(row, originals)))
            assert digest.hexdigest() == batch['original_hash_aggregate']
            cursor = runner.db.execute('SELECT last_id FROM cursors WHERE prefix=?', (batch['prefix'],)).fetchone()
            assert cursor and cursor[0] < ids[0]
            assert ids[0] == batch['first_id'] and ids[-1] == batch['last_id']
            prepared.append((batch, rows))

        # Both complete source batches have passed integrity checks first.
        with runner.db:
            runner.db.execute('''CREATE TABLE IF NOT EXISTS batch_reconciliations(
                seq INTEGER PRIMARY KEY, previous_status TEXT, result TEXT,
                reconciled_at TEXT, evidence_sha256 TEXT, original_batch_json TEXT)''')
            for batch, _ in prepared:
                runner.db.execute('INSERT INTO batch_reconciliations VALUES(?,?,?,?,?,?)',
                    (batch['seq'], 'submitted', 'not_inserted_after_durable_pk_check',
                     copy.now(), report['evidence_sha256'], copy.canonical(batch)))
                changed = runner.db.execute("UPDATE batches SET status='not_inserted_reconciled' WHERE seq=? AND status='submitted'", (batch['seq'],))
                assert changed.rowcount == 1
        runner.preflight()
        for batch, rows in prepared:
            runner.process_batch(batch['prefix'], rows, source, target, session, runner.old_db)
            result = runner.db.execute('SELECT seq,status,rows,original_hash_aggregate FROM batches ORDER BY seq DESC LIMIT 1').fetchone()
            assert result[0] > max(EXPECTED_SEQUENCES)
            assert result[1] == 'confirmed' and result[2] == 200
            assert result[3] == batch['original_hash_aggregate']
            cursor = runner.db.execute('SELECT last_id FROM cursors WHERE prefix=?', (batch['prefix'],)).fetchone()
            assert cursor[0] == batch['last_id']
            report['batches'].append({'old_seq': batch['seq'], 'replay_seq': result[0], 'rows_confirmed': result[2], 'prefix': batch['prefix']})
            print(json.dumps(report['batches'][-1]), flush=True)
            runner.status()

        source.flush(copy.TARGET, timeout=180)
        count = int(source.get_collection_stats(copy.TARGET)['row_count'])
        assert count == EXPECTED_BASELINE + 400
        assert not source.list_indexes(copy.TARGET)
        assert runner.db.execute("SELECT count(*) FROM batches WHERE status='submitted'").fetchone()[0] == 0
        report.update(finished_at=copy.now(), status='reconciled_and_replayed', target_rows=count, indexes=[])
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(finished_at=copy.now(), status='stopped_for_review', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        (BACKUP / 'reconciliation_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        session.close()
        target.close()
        source.close()
        runner.guard.close()
        runner.old_db.close()
        runner.db.close()


if __name__ == '__main__':
    main()
