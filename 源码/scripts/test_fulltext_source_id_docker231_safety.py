"""Offline tests for field-only protection; no network or database connections."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from benchmark_fulltext_source_id_docker231 import DockerBenchmark
from fulltext_source_id_safe_common import IntegrityError, decode


class FakeClient:
    def __init__(self, duplicate=False, conflict=False, corrupt=False):
        self.row = {'id':'abc_0','doc_id':'abc','text_embed':[1.0,2.0],
            'text':'keep original','hasSourcesysSourceId':['conflict'] if conflict else None}
        self.duplicate = duplicate
        self.corrupt = corrupt
        self.writes = 0

    def query(self, *args, **kwargs):
        return [{'count(*)':2 if self.duplicate else 1}]

    def get(self, *args, **kwargs):
        return [copy.deepcopy(self.row)]

    def upsert(self, collection, data, partial_update, **kwargs):
        assert partial_update is True
        assert set(data[0]) == {'id','hasSourcesysSourceId'}
        self.writes += 1
        self.row['hasSourcesysSourceId'] = data[0]['hasSourcesysSourceId']
        if self.corrupt:
            self.row['text_embed'][1] = 9.0
        return {'upsert_count':len(data)}


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.b = DockerBenchmark.__new__(DockerBenchmark)
        self.b.fields = ['id','doc_id','text','text_embed','hasSourcesysSourceId']
        self.b.original_fields = self.b.fields[:-1]
        self.b.db = sqlite3.connect(':memory:')
        self.b.db.execute('CREATE TABLE batches(seq INTEGER, status TEXT,payload BLOB,result TEXT)')
        self.b.db.execute("INSERT INTO batches VALUES(1,'queued',NULL,NULL)")
        self.b.db_lock = threading.Lock()
        self.b.counts = dict.fromkeys(['handled','newly_filled','already_filled','no_source',
            'duplicate_keys_skipped','duplicate_extra_versions_preserved','completed_batches'],0)

    def tearDown(self):
        self.b.db.close()

    def task(self, fake):
        self.b.worker_client = lambda:fake
        return self.b.task(1,[{'id':'abc_0','doc_id':'abc'}],{'abc_0':['source']},{'abc':None})

    def test_only_target_field_written_and_originals_verified(self):
        c = FakeClient()
        self.assertEqual(self.task(c)['newly_filled'],1)
        self.assertEqual(c.row['text_embed'],[1.0,2.0])
        payload = decode(self.b.db.execute('SELECT payload FROM batches').fetchone()[0])
        self.assertEqual(payload['patches'],[{'id':'abc_0','hasSourcesysSourceId':['source']}])

    def test_duplicate_versions_never_upserted(self):
        c = FakeClient(duplicate=True)
        self.assertEqual(self.task(c)['duplicates'],1)
        self.assertEqual(c.writes,0)

    def test_conflicting_value_stops_before_write(self):
        c = FakeClient(conflict=True)
        with self.assertRaises(IntegrityError):
            self.task(c)
        self.assertEqual(c.writes,0)

    def test_vector_corruption_detected_and_batch_not_completed(self):
        c = FakeClient(corrupt=True)
        with self.assertRaises(IntegrityError):
            self.task(c)
        self.assertEqual(self.b.db.execute('SELECT status FROM batches').fetchone()[0],'prepared')

    def test_missing_and_stale_watchdog_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.b.root = Path(folder)
            self.b.stop_reason = None
            self.assertFalse(self.b.guard())
            (self.b.root/'host_monitor.json').write_text(json.dumps({'timestamp':time.time()-60,'safe':True}))
            self.b.stop_reason = None
            self.assertFalse(self.b.guard())
            self.assertIn('stale',self.b.stop_reason)

    def test_adaptive_pressure_waits_without_selecting_or_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.b.root = Path(folder)
            (self.b.root/'host_monitor.json').write_text(json.dumps({'cpu_busy_percent':5,
                'index_tasks_pending':2,'index_tasks_running':1}))
            self.b.adaptive = True
            self.b.pressure_wait_seconds = 0
            self.b.deadline = time.monotonic()+0.05
            self.b.guard = lambda:True
            self.b.stop_reason = None
            self.b.phases = []
            self.b.status = lambda *a,**k:None
            self.b.log = lambda *a,**k:None
            self.b.select = lambda n:self.fail('No enumeration during index pressure')
            result = self.b.phase_run('pressure_test',2000,2,self.b.deadline)
            self.assertEqual(result['newly_filled'],0)
            self.assertGreater(self.b.pressure_wait_seconds,0)

    def test_guard_refusal_stops_dispatch_without_enumeration(self):
        self.b.deadline = time.monotonic()+1
        self.b.guard = lambda:False
        self.b.stop_reason = 'protected'
        self.b.phases = []
        self.b.status = lambda *a,**k:None
        self.b.log = lambda *a,**k:None
        self.b.select = lambda n:self.fail('No enumeration after protection')
        result = self.b.phase_run('guard_test',2000,2,self.b.deadline)
        self.assertEqual(result['newly_filled'],0)
        self.assertEqual(result['stop_reason'],'protected')


if __name__ == '__main__':
    unittest.main()
