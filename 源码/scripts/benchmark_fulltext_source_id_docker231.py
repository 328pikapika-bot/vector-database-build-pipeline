"""Bounded Docker tuning: only partial patches on 231; preserve every original field."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import benchmark_fulltext_source_id_single231 as core
from fulltext_source_id_safe_common import IntegrityError, decode, encode, now

PREVIOUS = Path('/previous')


class DockerBenchmark(core.Benchmark):
    def __init__(self, folder, seconds, adaptive=False):
        super().__init__(folder, seconds)
        self.adaptive = adaptive
        self.pressure_wait_seconds = 0.0
        self.previous = json.loads((PREVIOUS/'report.json').read_text())
        if self.previous['status'] != 'completed_and_stopped' or self.previous['pending_batches']:
            raise IntegrityError('Previous run is not fully completed')
        self.cursor = self.previous['end_cursor']
        self.bucket = int(self.cursor[:3], 16)
        self.stop_reason = None
        self.phase_measurements = []
        # Reuse validated source cache, without modifying the previous journal.
        # Previous runner is stopped/closed; immutable prevents WAL/SHM creation
        # through the strictly read-only mount of this stable audit snapshot.
        old = sqlite3.connect('file:'+str(PREVIOUS/'journal.sqlite')+'?mode=ro&immutable=1', uri=True)
        for row in old.execute('SELECT doc_id,value,reason FROM source_cache'):
            self.db.execute('INSERT OR REPLACE INTO source_cache VALUES(?,?,?)', row)
        self.db.commit()
        old.close()

    def preflight(self):
        self.status('preflight')
        def read(host):
            c = core.connect(host)
            try:
                return core.snapshot(c), int(c.query(core.COLLECTION, filter=core.FIELD+' is not null',
                    output_fields=['count(*)'], consistency_level='Strong', timeout=90)[0]['count(*)'])
            finally:
                c.close()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {h:pool.submit(read,h) for h in core.HOSTS}
            for h,f in futures.items():
                self.baseline[h],self.before_filled[h] = f.result()
        if self.baseline != self.old_state['baseline']:
            raise IntegrityError('Collection inventory/schema/count baseline changed')
        if self.before_filled != self.previous['filled_after']:
            raise IntegrityError('Unexpected external field writes since the previous run')
        if any(s[core.COLLECTION]['count'] != 195893017 for s in self.baseline.values()):
            raise IntegrityError('Strict collection count guard failed')
        self.fields = [f['name'] for f in self.baseline[core.TARGET][core.COLLECTION]['fields']]
        self.original_fields = [f for f in self.fields if f != core.FIELD]
        self.expected_count = 195893017
        (self.root/'baseline.json').write_text(json.dumps({'snapshots':self.baseline,
            'filled_counts':self.before_filled,'start_cursor':self.cursor,'start_bucket':self.bucket}, indent=2))
        if not self.guard(force=True):
            raise RuntimeError('Preflight host protection: '+str(self.stop_reason))

    def guard(self, force=False):
        if (self.root/'STOP_REQUESTED').exists():
            self.stop_reason = self.stop_reason or 'stop marker'
            return False
        # Host monitor runs outside the container; fail closed on stale/missing data.
        try:
            host = json.loads((self.root/'host_monitor.json').read_text())
            if time.time()-host['timestamp'] > 35:
                raise RuntimeError('Host monitor is stale')
            if not host['safe']:
                raise RuntimeError('; '.join(host['reasons']))
            self.disk = host['disk_free_bytes']
        except Exception as exc:
            self.stop_reason = 'host protection: '+str(exc)
            return False
        if force or time.monotonic()-self.last_guard >= 120:
            n = int(self.main.query(core.COLLECTION, filter='', output_fields=['count(*)'],
                consistency_level='Strong', timeout=90)[0]['count(*)'])
            if n != self.expected_count:
                raise IntegrityError('Target collection count changed')
            self.last_guard = time.monotonic()
        return True

    def phase_run(self, name, batch, workers, until):
        self.phase = {'name':name,'batch_size':batch,'workers':workers}
        t0 = time.monotonic()
        before = dict(self.counts)
        pending = []
        timing = {}
        def collect(future):
            result = future.result()
            for k,v in result['timing'].items():
                timing[k] = timing.get(k,0)+v
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                while time.monotonic() < min(until,self.deadline):
                    if len(pending) >= workers:
                        collect(pending.pop(0))
                    if not self.guard():
                        break
                    if self.adaptive:
                        host = json.loads((self.root/'host_monitor.json').read_text())
                        if host['cpu_busy_percent'] > 35 or host.get('index_tasks_pending',0) > 0 or host.get('index_tasks_running',0) > 0:
                            wait_start = time.monotonic()
                            self.status('waiting_for_host_capacity',host_cpu_percent=host['cpu_busy_percent'],
                                index_tasks_pending=host.get('index_tasks_pending',0),
                                index_tasks_running=host.get('index_tasks_running',0))
                            time.sleep(min(2,max(0,min(until,self.deadline)-time.monotonic())))
                            self.pressure_wait_seconds += time.monotonic()-wait_start
                            continue
                    previous_cursor = self.cursor
                    selected = self.select(batch)
                    if not selected:
                        break
                    expected,reasons = self.fetch_source(selected)
                    if time.monotonic() >= min(until,self.deadline) or not self.guard():
                        self.buffer = selected+self.buffer
                        self.cursor = previous_cursor
                        break
                    with self.db_lock:
                        seq = self.db.execute("INSERT INTO batches(phase,status,payload) VALUES(?,'queued',?)",
                            (name,encode({'selected':selected}))).lastrowid
                        self.db.commit()
                    pending.append(pool.submit(self.task,seq,selected,expected,reasons))
                    if time.monotonic()-self.last_progress >= 15:
                        self.status()
                        self.log('progress',phase=self.phase,**self.counts)
                        self.last_progress = time.monotonic()
            finally:
                # Even on guard failure, finish and verify every dispatched update.
                errors = []
                for f in pending:
                    try:
                        collect(f)
                    except BaseException as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
        elapsed = time.monotonic()-t0
        result = {'name':name,'batch_size':batch,'workers':workers,'seconds':elapsed,
            'handled':self.counts['handled']-before['handled'],
            'newly_filled':self.counts['newly_filled']-before['newly_filled'],
            'rows_per_second':(self.counts['newly_filled']-before['newly_filled'])/max(elapsed,1),
            'timing_totals':timing,'stop_reason':self.stop_reason}
        self.phases.append(result)
        self.status()
        self.log('phase_complete',**result)
        return result

    def export_patches(self):
        digest = hashlib.sha256()
        n = 0
        with gzip.open(self.root/'patches_for_later_sync.jsonl.gz','wb') as out:
            for payload, in self.db.execute("SELECT payload FROM batches WHERE status='completed' ORDER BY seq"):
                for patch in decode(payload)['patches']:
                    line = (json.dumps(patch,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                    out.write(line)
                    digest.update(line)
                    n += 1
        if n != self.counts['newly_filled']:
            raise IntegrityError('Patch export count mismatch')
        (self.root/'patch_manifest.json').write_text(json.dumps({'rows':n,'jsonl_sha256':digest.hexdigest(),
            'write_host':core.TARGET,'read_only_hosts':list(core.HOSTS[1:])},indent=2))

    def run(self):
        try:
            self.preflight()
            self.start = time.monotonic()
            self.deadline = self.start+self.seconds
            self.log('docker_tuning_started',seconds=self.seconds,start_cursor=self.cursor)
            pilots = []
            configs = ([('adaptive2000_w2',2000,2),('adaptive3000_w2',3000,2)] if self.adaptive else
                [('baseline2000_w4',2000,4),('batch4000_w4',4000,4),
                 ('batch2000_w6',2000,6),('batch4000_w6',4000,6),('batch4000_w8',4000,8)])
            for name,batch,workers in configs:
                if self.stop_reason or time.monotonic() >= self.deadline:
                    break
                pilots.append(self.phase_run(name,batch,workers,min(time.monotonic()+90,self.deadline)))
            if not pilots:
                raise RuntimeError('No completed benchmark phase')
            best = max(pilots,key=lambda p:p['rows_per_second'])
            chosen = min([p for p in pilots if p['rows_per_second'] >= best['rows_per_second']*0.95],
                key=lambda p:(p['workers'],p['batch_size']))
            self.chosen = {'batch_size':chosen['batch_size'],'workers':chosen['workers'],
                'measured_pilot_rows_per_second':chosen['rows_per_second']}
            if not self.stop_reason:
                self.phase_run('sustained_selected',chosen['batch_size'],chosen['workers'],
                    min(time.monotonic()+(180 if self.adaptive else 300),self.deadline))
            if not self.stop_reason and not self.adaptive:
                self.phase_run('baseline_repeat2000_w4',2000,4,self.deadline)
            self.write_window_seconds = time.monotonic()-self.start
            self.export_patches()
            self.finish()
            report = json.loads((self.root/'report.json').read_text())
            report.update(stop_reason=self.stop_reason,container_test=True,
                adaptive_pressure_control=self.adaptive,pressure_wait_seconds=self.pressure_wait_seconds,
                previous_sustained_rows_per_second=477.64038493596514)
            (self.root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        except BaseException as exc:
            self.status('stopped_error',error=repr(exc),traceback=traceback.format_exc())
            self.log('stopped_error',error=repr(exc))
            raise
        finally:
            if self.iterator:
                self.iterator.close()
            for c in self.all_worker_clients:
                c.close()
            self.main.close()
            self.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',required=True)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    args = parser.parse_args()
    if not args.apply or not 1 <= args.seconds <= 1200:
        parser.error('Explicit --apply and bounded duration <=1200s required')
    DockerBenchmark(args.workdir,args.seconds,args.adaptive).run()
