"""Offline pressure recovery and ambiguous-write recovery tests."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from run_fulltext_source_id_local231 import LocalRunner,PressureController,GuardedClient,verify_protected_inventory
from fulltext_source_id_safe_common import IntegrityError,decode,original_hash,encode


def host(wall=100):
    return {'timestamp':wall,'lock_alive':True,'services_unchanged':True,
        'available_ram_bytes':900*1024**3,'disk_free_bytes':{'/':100*1024**3,'/data':2*1024**4},
        'health':{'9094':{'status':200,'seconds':0.01}},'cpu_busy_percent':10,'iowait_percent':0,
        'index_tasks_pending':0,'index_tasks_running':0}


class PressureTests(unittest.TestCase):
    def test_high_cpu_waits_and_automatically_recovers(self):
        c=PressureController()
        h=host();h['cpu_busy_percent']=97
        self.assertFalse(c.decision(h,1000,10,100)[0])
        self.assertEqual(c.rate,480)
        self.assertFalse(c.decision(host(),1000,29,100)[0])
        self.assertTrue(c.decision(host(),1000,31,100)[0])

    def test_no_burst_after_idle_and_one_gate_for_all_workers(self):
        c=PressureController(initial_rate=100)
        self.assertTrue(c.decision(host(),1000,10,100)[0])
        self.assertFalse(c.decision(host(),1000,11,100)[0])
        self.assertTrue(c.decision(host(),1000,20,100)[0])

    def test_stale_monitor_and_lost_lock_wait(self):
        c=PressureController()
        self.assertEqual(c.decision(host(),1000,10,120)[1],'monitor_unavailable')
        h=host();h['lock_alive']=False
        self.assertEqual(c.decision(h,1000,10,100)[1],'shared_lock_unavailable')

    def test_index_jobs_never_gate_writes_when_cpu_is_safe(self):
        c=PressureController()
        h=host();h['index_tasks_pending']=3000;h['index_tasks_running']=12
        self.assertTrue(c.decision(h,2000,10,100)[0])
        self.assertEqual(c.rate,40)

    def test_backlog_caps_rate_but_does_not_stop_writes(self):
        for pending,expected in [(0,480),(49,480),(50,240),(199,240),
                                  (200,160),(499,160),(500,80),(999,80),(1000,40)]:
            with self.subTest(pending=pending):
                c=PressureController()
                h=host();h['index_tasks_pending']=pending
                self.assertTrue(c.decision(h,2000,10,100)[0])
                self.assertEqual(c.rate,expected)

    def test_moderate_cpu_reduces_budget_without_index_pause(self):
        c=PressureController()
        h=host();h['cpu_busy_percent']=61;h['index_tasks_running']=10
        self.assertTrue(c.decision(h,2000,10,100)[0])
        self.assertEqual(c.rate,400)
        self.assertTrue(c.decision(host(),2000,16,100)[0])
        self.assertEqual(c.rate,480)

    def test_cpu_hysteresis_avoids_repeated_resume_near_limit(self):
        c=PressureController()
        h=host();h['cpu_busy_percent']=90
        self.assertFalse(c.decision(h,2000,10,100)[0])
        h['cpu_busy_percent']=70
        self.assertFalse(c.decision(h,2000,30,100)[0])
        self.assertTrue(c.decision(host(),2000,51,100)[0])

    def test_dispatch_check_never_consumes_write_budget(self):
        c=PressureController()
        for _ in range(5):self.assertTrue(c.decision(host(),0,10,100)[0])
        self.assertTrue(c.decision(host(),2000,10,100)[0])
        self.assertFalse(c.decision(host(),2000,10,100)[0])

    def test_unrelated_new_collection_allowed_protected_changes_rejected(self):
        baseline={'231':{'fulltext':{'count':100,'schema':'original'}}}
        current=copy.deepcopy(baseline);current['231']['unrelated_new']={'count':10}
        verify_protected_inventory(current,baseline)
        current['231']['fulltext']['count']=99
        with self.assertRaises(IntegrityError):verify_protected_inventory(current,baseline)

    def test_duplicate_appearing_during_wait_is_rejected_before_write(self):
        c=FakeClient()
        r=LocalRunner.__new__(LocalRunner)
        r.wait_for_capacity=lambda rows:None
        r.key_count=lambda client,ids:2
        wrapped=GuardedClient(r,c)
        with self.assertRaises(IntegrityError):
            wrapped.upsert('scientific_fulltext_chunks_v2',data=[{'id':'abc_0','hasSourcesysSourceId':['source']}],partial_update=True)
        self.assertEqual(c.writes,0)

    def test_valid_patch_has_separate_wait_rpc_metrics_and_preserves_originals(self):
        c=FakeClient();original=copy.deepcopy(c.row)
        r=LocalRunner.__new__(LocalRunner)
        r.wait_for_capacity=lambda rows:None
        r.key_count=lambda client,ids:len(ids)
        wrapped=GuardedClient(r,c)
        result=wrapped.upsert('scientific_fulltext_chunks_v2',data=[{'id':'abc_0','hasSourcesysSourceId':['source']}],partial_update=True)
        self.assertEqual(result['upsert_count'],1)
        self.assertEqual(c.writes,1)
        self.assertEqual({k:v for k,v in c.row.items() if k!='hasSourcesysSourceId'},
            {k:v for k,v in original.items() if k!='hasSourcesysSourceId'})
        self.assertEqual(set(wrapped.last_write_timing),{'resource_and_rate_wait','prewrite_duplicate_guard','actual_upsert_rpc'})


class FakeClient:
    def __init__(self):
        self.row={'id':'abc_0','doc_id':'abc','text':'original','text_embed':[1.0,2.0],'hasSourcesysSourceId':None}
        self.writes=0
    def query(self,*a,**kw):return [{'count(*)':1}]
    def get(self,*a,**kw):return [copy.deepcopy(self.row)]
    def upsert(self,collection,data,partial_update,**kw):
        assert partial_update and set(data[0])=={'id','hasSourcesysSourceId'}
        self.writes+=1
        self.row['hasSourcesysSourceId']=data[0]['hasSourcesysSourceId']
        return {'upsert_count':len(data)}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.r=LocalRunner.__new__(LocalRunner)
        self.r.fields=['id','doc_id','text','text_embed','hasSourcesysSourceId']
        self.r.original_fields=self.r.fields[:-1]
        self.r.db=sqlite3.connect(':memory:')
        self.r.db.execute('CREATE TABLE batches(seq INTEGER,status TEXT,payload BLOB,result TEXT)')
        self.r.db.execute("INSERT INTO batches VALUES(1,'prepared',NULL,NULL)")
        self.r.db_lock=threading.Lock()
        self.c=FakeClient();self.r.worker_client=lambda:self.c
        self.payload={'ids':['abc_0'],'selected':[{'id':'abc_0','doc_id':'abc'}],
            'patches':[{'id':'abc_0','hasSourcesysSourceId':['source']}],
            'original_hashes':{'abc_0':original_hash(self.c.row,self.r.original_fields)},
            'original_values':{'abc_0':None},'duplicate_keys':{},'no_source':[]}
    def tearDown(self):self.r.db.close()
    def test_prepared_not_written_is_patched_and_verified(self):
        result=self.r.recover_batch(1,self.payload)
        self.assertEqual(result['newly_filled'],1)
        self.assertEqual(self.c.writes,1)
    def test_already_written_ambiguous_request_is_not_written_again(self):
        self.c.row['hasSourcesysSourceId']=['source']
        self.r.recover_batch(1,self.payload)
        self.assertEqual(self.c.writes,0)
    def test_recovery_conflict_or_original_corruption_never_writes(self):
        self.c.row['text']='unexpected'
        with self.assertRaises(IntegrityError):self.r.recover_batch(1,self.payload)
        self.assertEqual(self.c.writes,0)
    def test_wrong_mutation_fields_rejected(self):
        wrapped=GuardedClient(None,self.c)
        with self.assertRaises(IntegrityError):wrapped.upsert('scientific_fulltext_chunks_v2',data=[{'id':'abc_0','text_embed':[9]}],partial_update=True)
        self.assertEqual(self.c.writes,0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_never_skips_unfinished_batch_and_then_advances(self):
        r=LocalRunner.__new__(LocalRunner)
        r.db=sqlite3.connect(':memory:')
        r.db.executescript('CREATE TABLE batches(seq INTEGER,status TEXT,payload BLOB); CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT);')
        r.db_lock=threading.Lock()
        r.last_checkpoint='start'
        r.last_checkpoint_seq=0
        for seq,status in [(1,'completed'),(2,'prepared'),(3,'completed')]:
            r.db.execute('INSERT INTO batches VALUES(?,?,?)',(seq,status,encode({'selected':[{'id':'id'+str(seq)}]})))
        r.advance_checkpoint()
        self.assertEqual((r.last_checkpoint,r.last_checkpoint_seq),('id1',1))
        r.db.execute("UPDATE batches SET status='completed' WHERE seq=2")
        r.advance_checkpoint()
        self.assertEqual((r.last_checkpoint,r.last_checkpoint_seq),('id3',3))
        r.db.close()


class TrialControlTests(unittest.TestCase):
    def trial_runner(self):
        r=LocalRunner.__new__(LocalRunner)
        r.duration_seconds=20
        r.batch_size=2000
        r.workers=2
        r.counts={'newly_filled':1000000,'handled':1000010,'completed_batches':500}
        r.trial_baseline_counts=None
        r.trial_final_counts=None
        r.trial_started_monotonic=None
        r.trial_started_at=None
        r.trial_finished_at=None
        r.trial_elapsed_seconds=None
        r.dispatch_stop_reason=None
        return r

    def test_trial_rate_excludes_historical_counts_and_includes_drain_wait(self):
        r=self.trial_runner()
        with patch('run_fulltext_source_id_local231.time.monotonic',return_value=100):
            r.start_trial()
        self.assertEqual(r.dispatch_deadline,120)
        r.counts['newly_filled']+=2000
        r.counts['handled']+=2000
        r.counts['completed_batches']+=1
        r.dispatch_stop_reason='duration_elapsed'
        # The 20-second dispatch window plus 10 seconds draining a submitted
        # batch is a 30-second trial, rather than an artificial 20-second rate.
        with patch('run_fulltext_source_id_local231.time.monotonic',return_value=130):
            r.finish_trial()
        with patch('run_fulltext_source_id_local231.time.monotonic',return_value=900):
            result=r.trial_metrics()
        self.assertEqual(result['new_counts']['newly_filled'],2000)
        self.assertEqual(result['baseline_counts']['newly_filled'],1000000)
        self.assertEqual(result['elapsed_seconds_including_inflight_waits'],30)
        self.assertEqual(result['verified_rows_per_second_including_waits'],66.667)
        self.assertIsNotNone(result['finished_at'])
        self.assertEqual(r.counts['newly_filled'],1002000)

    def test_live_trial_rate_includes_resource_waits(self):
        r=self.trial_runner()
        with patch('run_fulltext_source_id_local231.time.monotonic',return_value=100):
            r.start_trial()
        with patch('run_fulltext_source_id_local231.time.monotonic',return_value=119):
            result=r.trial_metrics()
        self.assertEqual(result['verified_rows_per_second_including_waits'],0)
        self.assertEqual(result['elapsed_seconds_including_inflight_waits'],19)
        self.assertIsNone(result['finished_at'])

    def test_default_continuous_run_does_not_create_trial_or_deadline(self):
        r=self.trial_runner()
        r.duration_seconds=None
        r.dispatch_deadline=None
        r.start_trial()
        r.finish_trial()
        self.assertIsNone(r.trial_metrics())
        self.assertIsNone(r.dispatch_deadline)

    def test_stop_reason_checks_deadline_and_local_stop_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            r=LocalRunner.__new__(LocalRunner)
            r.root=Path(directory)
            r.dispatch_deadline=time.monotonic()+60
            self.assertIsNone(r.stop_reason())
            r.dispatch_deadline=time.monotonic()-1
            self.assertEqual(r.stop_reason(),'duration_elapsed')
            (r.root/'STOP_REQUESTED').touch()
            self.assertEqual(r.stop_reason(),'stop_requested')

    def test_unqueued_selection_returns_to_buffer_without_advancing_cursor(self):
        r=LocalRunner.__new__(LocalRunner)
        r.cursor='new_id'
        r.buffer=[{'id':'later_id'}]
        r.return_unqueued_selection([{'id':'first_id'}],'old_id')
        self.assertEqual(r.cursor,'old_id')
        self.assertEqual([row['id'] for row in r.buffer],['first_id','later_id'])

    def test_stop_does_not_bypass_capacity_gate_for_inflight_work(self):
        with tempfile.TemporaryDirectory() as directory:
            r=LocalRunner.__new__(LocalRunner)
            r.root=Path(directory)
            (r.root/'STOP_REQUESTED').touch()
            r.dispatch_deadline=None
            r.dispatch_stop_reason=None
            r.draining_inflight=False
            r.integrity_hold=None
            r.gate_lock=threading.Lock()
            r.observer=type('Observer',(),{'read':lambda self:{}})()
            calls=[]
            class Gate:
                def decision(self,*args):
                    calls.append(True)
                    return (len(calls)>1,'resource_wait',0)
            r.controller=Gate()
            states=[]
            r.status=lambda state:states.append(state)
            r.wait_for_capacity(10)
            self.assertEqual(len(calls),2)
            self.assertTrue(r.draining_inflight)
            self.assertEqual(states,['draining_inflight'])


if __name__=='__main__':unittest.main()
