"""Recover only this copy job from retained source-vector mmap cache pressure.

Runs on 231 with stdlib + Docker CLI. Never deletes database records, builds an
index, changes the host, or restarts a database. An unexpected state fails closed
and preserves the stop request and evidence for manual review.
"""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import urllib.request

BASE = Path('/data/scientific_fulltext_chunks_v3')
ROOT = BASE / 'runtime'
SHARDS = ('01', '23', '45', '67', '89', 'ab', 'cd', 'ef')
WORKERS = tuple('scientific_fulltext_chunks_v3_worker_' + x for x in SHARDS)
IMAGE = 'scientific_fulltext_chunks_v3_worker:20261117-tuning'
PID = 870208
PROCESS_START = '1466939877'
SOURCE_TOTAL = 195893017
BASE_COUNT = 335825
MAP_REASON = 'Milvus memory mappings above 70% of the host limit'
POLL_SECONDS = 30
STATUS = ROOT / 'cache_recovery_status.json'


def now():
    return datetime.now().astimezone().isoformat()


def publish(state, **details):
    value = {'at': now(), 'pid': os.getpid(), 'state': state, **details}
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def command(args, timeout=45):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT,
                                   timeout=timeout)


def worker_states():
    states = json.loads(command(['docker', 'inspect', *WORKERS]))
    active_names = command(['docker', 'ps', '--filter', 'name=scientific_fulltext_chunks_v3_worker_',
                            '--format', '{{.Names}}']).splitlines()
    if set(active_names) - set(WORKERS):
        raise RuntimeError('Unexpected additional active copy worker')
    prefixes = []
    for state, shard in zip(states, SHARDS):
        if state['Name'] != '/' + 'scientific_fulltext_chunks_v3_worker_' + shard:
            raise RuntimeError('Worker identity changed')
        if state['Config']['Image'] != IMAGE:
            raise RuntimeError('Worker image changed')
        env = dict(x.split('=', 1) for x in state['Config']['Env'] if '=' in x)
        if (env.get('V3_PREFIXES') != shard or env.get('V3_MAX_WORKERS') != '1'
                or env.get('V3_MILVUS_PID') != str(PID)
                or env.get('V3_WORK_ROOT') != '/runtime'):
            raise RuntimeError('Worker ownership/configuration changed')
        prefixes.extend(shard)
    if sorted(prefixes) != list('0123456789abcdef'):
        raise RuntimeError('Worker prefix overlap or omission')
    return states


def native_resources():
    proc = Path('/proc') / str(PID)
    cmdline = (proc / 'cmdline').read_bytes()
    if b'milvus' not in cmdline or b'standalone' not in cmdline:
        raise RuntimeError('Native Milvus identity changed')
    # Tie the monitor to the original process, including PID reuse protection.
    start = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    maps = sum(1 for _ in (proc / 'maps').open())
    limit = int(Path('/proc/sys/vm/max_map_count').read_text())
    mem = {x.split(':')[0]: int(x.split()[1]) * 1024
           for x in Path('/proc/meminfo').read_text().splitlines()}
    def cpu():
        return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    a = cpu()
    time.sleep(.5)
    b = cpu()
    delta = [y - x for x, y in zip(a, b)]
    health = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in (9093, 9094, 9095):
        with opener.open('http://127.0.0.1:%d/healthz' % port, timeout=5) as response:
            health[str(port)] = response.status
    return {'pid': PID, 'start': start, 'maps': maps, 'map_limit': limit,
            'available_memory': mem['MemAvailable'], 'data_free': shutil.disk_usage(ROOT).free,
            'cpu': 100 * (sum(delta) - delta[3] - delta[4]) / max(sum(delta), 1),
            'health': health}


def statuses():
    return {shard: json.loads((ROOT / ('status_' + shard + '.json')).read_text())
            for shard in SHARDS}


def waiting_for_maps(values, states, resources, process_start, at=None):
    if resources['start'] != process_start:
        raise RuntimeError('Milvus process changed; manual review required')
    timestamp = at or datetime.now().astimezone()
    waiting = 0
    by_name = {x['Name'].rsplit('_', 1)[1]: x for x in states}
    for shard, value in values.items():
        state = by_name[shard]['State']
        if value.get('status') == 'copy_scan_finished_pending_index_and_full_verification':
            if state['Running'] or state.get('ExitCode') != 0:
                return False
            continue
        if not state['Running']:
            return False
        age = (timestamp - datetime.fromisoformat(value['updated_at'])).total_seconds()
        if age < -5 or age > 120:
            return False
        if (value.get('status') != 'waiting_resource'
                or value.get('wait_reason') != MAP_REASON
                or value.get('uncertain_batches') != 0):
            return False
        waiting += 1
    return (waiting > 0 and resources['maps'] >= .70 * resources['map_limit']
            and resources['cpu'] < 65
            and resources['available_memory'] >= 200 * 1024**3
            and resources['data_free'] >= 1024**4
            and all(code == 200 for code in resources['health'].values()))


def journal_snapshot():
    db = sqlite3.connect('file:' + str(ROOT / 'journal.sqlite') + '?mode=ro', uri=True)
    try:
        pending = db.execute("SELECT count(*),coalesce(sum(rows),0) FROM batches WHERE status='submitted'").fetchone()
        confirmed = db.execute("SELECT coalesce(sum(rows),0) FROM batches WHERE status='confirmed'").fetchone()[0]
        return {'pending': list(pending), 'confirmed': confirmed,
                'expected_total': BASE_COUNT + confirmed,
                'cursors': [list(x) for x in db.execute('SELECT * FROM cursors ORDER BY prefix')]}
    finally:
        db.close()


SDK_CODE = r'''
import json,os
from datetime import datetime
from pymilvus import MilvusClient
c=MilvusClient(uri='http://127.0.0.1:19533',db_name='default',timeout=30)
source='scientific_fulltext_chunks_v2';target='scientific_fulltext_chunks_v3'
def report(stage,**kwargs):print(json.dumps({'at':datetime.now().astimezone().isoformat(),'stage':stage,**kwargs}),flush=True)
try:
 if c.describe_collection(source)['collection_id']!=467035880531671253:raise RuntimeError('Source identity changed')
 if c.describe_collection(target)['collection_id']!=470177903194309891:raise RuntimeError('Target identity changed')
 if c.list_indexes(target):raise RuntimeError('V3 index appeared; no-index copy must be reviewed')
 original_indexes=sorted(c.list_indexes(source))
 n=int(c.query(source,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=120)[0]['count(*)'])
 if n!=195893017:raise RuntimeError('Source total changed')
 c.flush(target,timeout=600)
 actual=int(c.get_collection_stats(target)['row_count'])
 if actual!=int(os.environ['V3_EXPECTED_TOTAL']):raise RuntimeError('Target persisted count differs from quiescent journal')
 report('counts_verified',source_count=n,target_count=actual)
 c.release_collection(source,timeout=180)
 report('source_query_cache_released',load_state=str(c.get_load_state(source)))
 c.load_collection(source,replica_number=1,timeout=1800)
 n=int(c.query(source,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=120)[0]['count(*)'])
 if n!=195893017 or sorted(c.list_indexes(source))!=original_indexes:raise RuntimeError('Source changed after reload')
 if c.list_indexes(target):raise RuntimeError('Target indexes changed')
 report('source_reloaded_verified',source_count=n,target_indexes=[])
finally:c.close()
'''


def resume_workers(marker, directory, expected_marker):
    if marker.read_text() != expected_marker:
        raise RuntimeError('Stop request changed; refusing resume')
    os.rename(marker, directory / 'STOP_REQUESTED_handled')
    try:
        command(['docker', 'update', '--restart=on-failure:5', *WORKERS])
        command(['docker', 'start', *WORKERS])
    except BaseException:
        # Docker may have started a subset before returning an error. Request a
        # safe boundary stop again, preserving any stop request from an operator.
        try:
            with marker.open('x') as file:
                file.write('V3 cache recovery resume failed; manual review required\n')
        except FileExistsError:
            pass
        command(['docker', 'update', '--restart=no', *WORKERS])
        deadline = time.monotonic() + 240
        while any(x['State']['Running'] for x in worker_states()):
            if time.monotonic() > deadline:
                raise RuntimeError('Partial resume safe stop timed out; stop request preserved')
            time.sleep(2)
        raise


def recover(process_start):
    if (ROOT / 'STOP_REQUESTED').exists():
        raise RuntimeError('An existing stop request belongs to a different action')
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    # Re-read the gate immediately before taking ownership of the stop request.
    if not waiting_for_maps(statuses(), worker_states(), native_resources(), process_start):
        return
    directory = ROOT / ('cache_recovery_' + stamp)
    directory.mkdir()
    command(['docker', 'update', '--restart=no', *WORKERS])
    marker = ROOT / 'STOP_REQUESTED'
    with marker.open('x') as file:
        file.write('Owned by V3 cache recovery ' + stamp + '\n')
    publish('stopping_copy_at_checkpoint',evidence=str(directory))
    deadline = time.monotonic() + 240
    while any(x['State']['Running'] for x in worker_states()):
        if time.monotonic() > deadline:
            raise RuntimeError('Safe stop timed out; no force stop permitted')
        time.sleep(2)
    checkpoint = journal_snapshot()
    if checkpoint['pending'][0]:
        raise RuntimeError('Ambiguous submitted inserts require manual reconciliation')
    (directory / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2))
    source = sqlite3.connect('file:' + str(ROOT / 'journal.sqlite') + '?mode=ro', uri=True)
    dest = sqlite3.connect(str(directory / 'journal_before_reload.sqlite'))
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()
    for path in ROOT.glob('status_*.json'):
        shutil.copy2(path, directory / path.name)
    current = native_resources()
    if current['start'] != process_start:
        raise RuntimeError('Milvus changed before cache release')
    (directory / 'resources_before.json').write_text(json.dumps(current, indent=2))
    name = 'scientific_fulltext_chunks_v3_cache_reload_' + stamp
    publish('reloading_source_query_cache',evidence=str(directory),checkpoint=checkpoint)
    args = ['docker', 'run', '--name', name, '--network', 'host', '--cpus', '1',
            '--memory', '512m', '--pids-limit', '128',
            '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
            '-e', 'V3_EXPECTED_TOTAL=' + str(checkpoint['expected_total']),
            IMAGE, 'python', '-u', '-c', SDK_CODE]
    with (directory / 'collection_cache_reload.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3000)
    current = native_resources()
    if (current['start'] != process_start or current['maps'] >= .70 * current['map_limit']
            or current['cpu'] >= 65 or current['available_memory'] < 200 * 1024**3
            or current['data_free'] < 1024**4 or any(x != 200 for x in current['health'].values())):
        raise RuntimeError('Native mapping reserve not recovered')
    if any(x['State']['Running'] for x in worker_states()):
        raise RuntimeError('Another action started workers during maintenance')
    if journal_snapshot() != checkpoint:
        raise RuntimeError('Journal changed during maintenance')
    expected_marker = 'Owned by V3 cache recovery ' + stamp + '\n'
    resume_workers(marker, directory, expected_marker)
    (directory / 'resources_after.json').write_text(json.dumps(current, indent=2))
    publish('resumed_copy',evidence=str(directory),resources=current)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'cache_recovery_supervisor.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        streak = 0
        try:
            process_start = native_resources()['start']
            if process_start != PROCESS_START:
                raise RuntimeError('Expected Milvus process start changed')
            while True:
                if (ROOT / 'STOP_REQUESTED').exists():
                    publish('idle_existing_stop_request')
                    return
                values = statuses()
                if values and all(x.get('status') == 'copy_scan_finished_pending_index_and_full_verification'
                                  for x in values.values()):
                    publish('copy_scan_finished_no_more_cache_recovery')
                    return
                resources = native_resources()
                eligible = waiting_for_maps(values, worker_states(), resources, process_start)
                streak = streak + 1 if eligible else 0
                publish('watching_existing_copy',resources=resources,eligible_samples=streak)
                # Three consecutive fresh observations before any maintenance.
                if streak >= 3:
                    recover(process_start)
                    streak = 0
                time.sleep(POLL_SECONDS)
        except Exception as exc:
            publish('manual_review_required',error=type(exc).__name__ + ': ' + str(exc))
            raise


if __name__ == '__main__':
    main()
