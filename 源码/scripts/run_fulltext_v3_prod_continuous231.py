"""Continue the real 231:19533 V3 copy from the retained ten-minute checkpoint.

Only scientific_fulltext_chunks_v3 is writable. No index is created here.
"""
from __future__ import annotations
import json, os, shutil, sqlite3, time
from pathlib import Path
from pymilvus import MilvusClient

import run_fulltext_v3_prod_10min231 as one
import benchmark_fulltext_source_id_isolated231_continuous as b
from fulltext_source_id_safe_common import IntegrityError, now

PROJECT = Path(__file__).resolve().parents[1]
OLD_ROOT = PROJECT/'outputs/fulltext_v3_prod_10min_231_20261116_140301'
SOURCE = 'scientific_fulltext_chunks_v2'
TARGET = 'scientific_fulltext_chunks_v3'
PROD = one.PROD


class ResumeProbe(one.ProductionV3Probe):
    def __init__(self, root, guard, old_report):
        super().__init__(root, guard)
        self.db.close()
        shutil.copy2(OLD_ROOT/'audit.sqlite', root/'audit.sqlite')
        self.db = sqlite3.connect(root/'audit.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.report.update(old_report)
        self.report.update({'status':'resuming_v3_copy_without_indexes',
            'resume_started_at':now(),'continuation_output':str(root),
            'index_created':False,'full_copy':False,'production_cutover':False})
        self.report['resume_inserted_rows'] = int(old_report['inserted_rows_pending_verification'])
        self.root = root
        self.row_limit = one.b.EXPECTED_SOURCE_COUNT
        self.seconds = 600
        self.workers = 4
        self.bucket_rows = dict(old_report['prefix_rows_read'])
        self.exhausted = set()
        self.next_bucket = 0
        self.last_ids = {}
        for key, in self.db.execute('SELECT id FROM expected_rows ORDER BY id'):
            prefix = key[:3]
            if key > self.last_ids.get(prefix, ''): self.last_ids[prefix] = key

    def preflight(self):
        self.source = MilvusClient(uri=PROD, db_name='default', timeout=20)
        self.target = MilvusClient(uri=PROD, db_name='default', timeout=20)
        self.target_collections_before = set(self.target.list_collections())
        if TARGET not in self.target_collections_before:
            raise IntegrityError('V3 target collection is missing; refusing to create a second copy')
        self.source_schema = self.source.describe_collection(SOURCE)
        self.source_fields = [f['name'] for f in self.source_schema['fields']]
        self.original_fields = [f for f in self.source_fields if f != b.FIELD]
        if self.count(self.source,SOURCE) != b.EXPECTED_SOURCE_COUNT:
            raise IntegrityError('Protected source count changed before continuation')
        if self.target.list_indexes(TARGET):
            raise IntegrityError('V3 already has indexes; no append allowed')
        target_count = int(self.target.get_collection_stats(TARGET).get('row_count',-1))
        expected = int(self.report['inserted_rows_pending_verification'])
        if target_count != expected:
            raise IntegrityError(f'V3 row count {target_count} differs from audit {expected}')
        self.report['source_count_before'] = b.EXPECTED_SOURCE_COUNT
        self.report['target_count_at_resume'] = target_count
        self.save()

    def run(self):
        started=time.monotonic(); cycles=0
        try:
            self.preflight()
            while not self.exhausted and self.report['read_rows'] < self.row_limit:
                if (self.root/'STOP_REQUESTED').exists(): break
                cycles += 1
                before=self.report['inserted_rows_pending_verification']
                self.copy_phase('continuation_%03d_batch200_workers4_no_index'%cycles,200,
                                self.row_limit-self.report['read_rows'],600)
                self.target.flush(TARGET,timeout=300)
                target_count=int(self.target.get_collection_stats(TARGET).get('row_count',-1))
                if target_count != self.report['inserted_rows_pending_verification']:
                    raise IntegrityError('V3 count differs after continuation flush')
                self.report['continuation_cycles']=cycles
                self.report['last_cycle_inserted']=self.report['inserted_rows_pending_verification']-before
                self.report['copy_rows_per_second_without_indexes']=(
                    (self.report['inserted_rows_pending_verification']-int(self.report.get('resume_inserted_rows',before))) /
                    max(time.monotonic()-started,.001))
                self.save()
            self.report['status']='completed_v3_copy_without_indexes'
            self.report['full_copy'] = bool(self.exhausted or self.report['read_rows'] >= self.row_limit)
            self.report['finished_at']=now()
        except BaseException as exc:
            self.report['status']='paused_v3_copy_collection_preserved_no_index'
            self.report['error']=repr(exc); self.report['finished_at']=now(); raise
        finally:
            self.report['wall_clock_seconds']=time.monotonic()-started; self.save()
            for iterator in self.iterators.values(): iterator.close()
            for source,target,session in self.worker_resources:
                session.close();target.close();source.close()
            if self.target:self.target.close()
            if self.source:self.source.close()
            self.db.close()


def main():
    stamp=time.strftime('%Y%m%d_%H%M%S')
    root=PROJECT/'outputs'/('fulltext_v3_prod_continuous_231_'+stamp)
    old=json.loads((OLD_ROOT/'report.json').read_text(encoding='utf-8'))
    guard=one.ProductionGuard(root)
    try: ResumeProbe(root,guard,old).run()
    finally: guard.sample(force=True); guard.ssh.close()


if __name__=='__main__': main()
