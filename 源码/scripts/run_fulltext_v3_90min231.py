"""90-minute isolated, real-data source-ID evaluation. Never a migration/cutover.

Only fresh collections on 231:19539 are writable. Production 19533 is read only.
All collections/data are retained. Each bounded sample finishes production-index
build and complete original-field/vector readback before being counted complete.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import threading
import time
import traceback

import paramiko
from pymilvus import MilvusClient

from benchmark_fulltext_source_id_isolated231_continuous import (
    COLLECTION_PREFIX, Experiment, RestrictedTarget, SOURCE_COLLECTION,
    SOURCE_URI, TARGET_URI, canonical,
)
from fulltext_source_id_safe_common import IntegrityError, now

PROJECT = Path(__file__).resolve().parents[1]
TARGET_CONTAINER = 'fulltext-source-id-phase-231-20261115-standalone-1'
REMOTE = r'''
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
def cpu(): return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
a=cpu();time.sleep(0.5);b=cpu();d=[y-x for x,y in zip(a,b)]
r={'cpu_percent':100*(sum(d)-d[3]-d[4])/max(sum(d),1),'iowait_percent':100*d[4]/max(sum(d),1),'cpus':os.cpu_count()}
mem=dict((s.split(':')[0],int(s.split()[1])*1024) for s in Path('/proc/meminfo').read_text().splitlines())
r['available_memory']=mem['MemAvailable'];r['data_free']=os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
r['services']={}
for x in json.loads(subprocess.check_output(['docker','inspect',*names],text=True)):
 n=x['Name'].lstrip('/');h=x['HostConfig']
 r['services'][n]={'id':x['Id'],'started':x['State']['StartedAt'],'restarts':x['RestartCount'],'status':x['State']['Status'],'health':x['State'].get('Health',{}).get('Status'),'config_hash':hashlib.sha256(json.dumps({'config':x['Config'],'host':h,'mounts':sorted(x['Mounts'],key=lambda m:m['Destination'])},sort_keys=True).encode()).hexdigest()}
 if n=='fulltext-source-id-phase-231-20261115-standalone-1':
  pid=x['State']['Pid'];r['target_limits']={k:h[k] for k in ['NanoCpus','CpuQuota','CpuPeriod','Memory']}
  cg=Path('/proc/%d/cgroup'%pid).read_text().splitlines();r['target_memory']=0
  for line in cg:
   parts=line.split(':',2)
   if parts[0]=='0':
    p=Path('/sys/fs/cgroup'+parts[2]+'/memory.current')
    if p.exists():r['target_memory']=int(p.read_text())
   elif 'memory' in parts[1].split(','):
    p=Path('/sys/fs/cgroup/memory'+parts[2]+'/memory.usage_in_bytes')
    if p.exists():r['target_memory']=int(p.read_text())
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));r['health']={};r['target_index_pending']=0;r['target_index_running']=0
for p in [9093,9094,9095,9099]:
 try:
  t=time.monotonic();resp=op.open('http://127.0.0.1:%d/healthz'%p,timeout=3);r['health'][str(p)]={'status':resp.status,'seconds':time.monotonic()-t}
 except Exception as e:r['health'][str(p)]={'error':str(e)}
metrics=op.open('http://127.0.0.1:9099/metrics',timeout=3).read().decode()
for line in metrics.splitlines():
 if line.startswith('milvus_datacoord_task_num_in_scheduler{') and 'task_type="Index"' in line:
  if 'task_state="JobStateInit"' in line:r['target_index_pending']+=int(float(line.rsplit(' ',1)[1]))
  if 'task_state="JobStateInProgress"' in line:r['target_index_running']+=int(float(line.rsplit(' ',1)[1]))
print(json.dumps(r))
'''


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def pressure_reason(sample):
    if sample.get('cpu_percent',100) >= 65:
        return 'host_cpu_at_or_above_65_percent'
    if sample.get('available_memory',0) < 200*1024**3:
        return 'host_memory_reserve_below_200GiB'
    if sample.get('data_free',0) < 1024**4:
        return 'server_data_free_below_1TiB'
    if sample.get('target_memory',0) > 105*1024**3:
        return 'isolated_target_memory_above_105GiB'
    if any(sample.get('health',{}).get(str(p),{}).get('status') != 200
           for p in (9093,9094,9095,9099)):
        return 'database_health_check_failed'
    return None


class Guard:
    def __init__(self, root):
        self.root=root
        self.lock=threading.RLock()
        self.sampled=0.0
        self.latest=None
        self.baseline=None
        self.wait_seconds=0.0
        self.ssh=paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect('192.0.2.15',username='root',
            password=os.environ['BACKFILL_SSH_PASSWORD'],timeout=12)

    def sample(self, force=False):
        with self.lock:
            if not force and self.latest and time.monotonic()-self.sampled<15:
                return self.latest
            _,out,err=self.ssh.exec_command(shlex.join(['python3','-c',REMOTE]),timeout=25)
            body=out.read().decode();error=err.read().decode()
            if out.channel.recv_exit_status():raise RuntimeError('Read-only monitor failed: '+error[-1000:])
            sample=json.loads(body);sample['at']=now()
            limits=sample.get('target_limits',{})
            if limits.get('NanoCpus')!=24000000000 or limits.get('Memory')!=128*1024**3:
                raise IntegrityError('Independent target must retain its 24-core/128GiB limits')
            if self.baseline is None:
                self.baseline=sample
                write_json(self.root/'server_baseline.json',sample)
            elif sample['services']!=self.baseline['services']:
                raise IntegrityError('Existing service identity/configuration/start/restart/health changed; stop this test')
            self.latest=sample;self.sampled=time.monotonic()
            write_json(self.root/'resources_latest.json',sample)
            with (self.root/'resources.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(canonical(sample)+'\n')
            return sample

    def check(self, operation):
        started=time.monotonic()
        while True:
            if (self.root/'STOP_REQUESTED').exists():raise RuntimeError('User requested bounded test stop; existing data retained')
            if shutil.disk_usage(self.root).free<10*1024**3:raise RuntimeError('Local audit disk reserve below10GiB')
            reason=pressure_reason(self.sample())
            if reason is None:
                self.wait_seconds+=time.monotonic()-started
                return
            write_json(self.root/'guard_wait.json',{'at':now(),'operation':operation,'reason':reason})
            if time.monotonic()-started>300:
                raise RuntimeError('Resource pressure remained for5minutes; test dispatch stopped: '+reason)
            time.sleep(5)


class OneShotTarget(RestrictedTarget):
    def insert(self, name, rows):
        self._name(name)
        if not 1<=len(rows)<=200:raise IntegrityError('Unsafe vector batch size')
        # The installed SDK's retry decorator prioritizes timeout over retry_times.
        # Invoke its inspected raw operation once, rather than blindly retrying a
        # potentially committed Insert after an ambiguous network response.
        connection=self.raw._get_connection()
        from pymilvus.client.grpc_handler import GrpcHandler
        raw=inspect.unwrap(GrpcHandler.insert_rows)
        result=raw(connection,name,rows,partition_name='',timeout=120,
            context=self.raw._generate_call_context())
        return {'insert_count':result.insert_count,'ids':result.primary_keys}


class ProductionIndexSample(Experiment):
    def __init__(self, root, collection, guard, prior_ids, batch_size, rows, seconds, workers=2):
        super().__init__(root,collection,rows=rows,seconds=seconds,skip_duplicate_keys=True)
        self.guard=guard;self.last_ids=dict(prior_ids);self.batch_size=batch_size
        self.workers=workers
        self.report['workers']=workers
        self.report['timing_scope']='Complete sample: read + Solr/cache + hash/audit + one-shot insert + production IVF_PQ and7scalar indexes + full target/source vector readback. No production cutover.'

    def before_operation(self, operation):self.guard.check(operation)

    def worker(self):
        source,target,session=super().worker()
        return source,OneShotTarget(target.raw,self.collection),session

    def preflight(self):
        self.guard.check('preflight')
        super().preflight()
        self.source_indexes=sorted([self.source.describe_index(SOURCE_COLLECTION,n)
                             for n in self.source.list_indexes(SOURCE_COLLECTION)],key=lambda i:i['index_name'])
        vector=[i for i in self.source_indexes if i['field_name']=='text_embed']
        if len(vector)!=1 or vector[0]['index_type']!='IVF_PQ' or len(self.source_indexes)!=8:
            raise IntegrityError('Production index definitions changed; inspect before testing')
        self.report['production_index_definitions']=self.source_indexes
        self.save()

    def build_readback_index(self):
        self.report['status']='building_exact_production_indexes_after_copy'
        self.progress()
        indexes=self.target.prepare_index_params()
        for index in self.source_indexes:
            args=dict(field_name=index['field_name'],index_name=index['index_name'],index_type=index['index_type'])
            if index['field_name']=='text_embed':
                args.update(metric_type=index['metric_type'],params={k:int(index[k]) for k in ('nlist','m','nbits')})
            indexes.add_index(**args)
        started=time.monotonic()
        self.target.create_index(self.collection,index_params=indexes,sync=False,timeout=60)
        while True:
            self.guard.sample()
            states=[self.target.describe_index(self.collection,i['index_name']) for i in self.source_indexes]
            self.report['target_index_states']=states
            self.progress()
            if any(i.get('state')=='Failed' for i in states):raise IntegrityError('Sample index creation failed')
            if all(i.get('state')=='Finished' and i.get('pending_index_rows',0)==0 for i in states):break
            if time.monotonic()-started>1800:raise RuntimeError('Index build exceeded30minute bound; collection retained')
            time.sleep(10)
        self.report['seconds']['exact_production_indexes']=time.monotonic()-started

    def run(self):
        started=time.monotonic()
        try:
            self.preflight()
            self.copy_phase('batch%d_workers%d'%(self.batch_size,self.workers),self.batch_size,self.row_limit,self.seconds)
            self.report['seconds']['copy_phases_including_waits']=sum(p['elapsed_seconds_including_inflight_waits'] for p in self.report['phases'])
            self.verify()
            self.report['status']='completed_sample_and_verified_not_production_ready'
            self.report['verified_new_source_ids']=self.db.execute("SELECT count(*) FROM expected_rows WHERE verified=1 AND source_field_before='null' AND source_field_after!='null'").fetchone()[0]
            self.report['source_documents_looked_up']=self.db.execute('SELECT count(*) FROM source_cache').fetchone()[0]
            self.report['source_documents_missing']=self.db.execute("SELECT count(*) FROM source_cache WHERE reason IS NOT NULL").fetchone()[0]
            self.report['index_definitions_match_production']=True
            # Only this newly created test collection is released from RAM.
            # Its rows/indexes remain on disk, and no existing collection changes.
            self.target.release_collection(self.collection,timeout=60)
            self.report['new_test_collection_released_from_memory_data_preserved']=True
        except BaseException as exc:
            self.report['status']='failed_new_test_collection_preserved'
            self.report['error']=repr(exc)
            self.report['traceback']=traceback.format_exc()
            raise
        finally:
            self.report['finished_at']=now()
            self.report['seconds']['complete_cycle']=time.monotonic()-started
            self.progress()
            for iterator in self.iterators.values():iterator.close()
            for source,target,session in self.worker_resources:
                session.close();target.close();source.close()
            if self.target:self.target.close()
            if self.source:self.source.close()
            self.db.close()


def execute(root, seconds=5400, resume=False):
    from datetime import datetime
    import sqlite3
    root.mkdir(parents=True,exist_ok=resume)
    start=time.monotonic();deadline=start+seconds
    state={'started_at':now(),'pid':os.getpid(),'status':'preflight','planned_seconds':seconds,
           'purpose':'Highest efficient source-ID completion; rebuilding itself is not the objective',
           'source_read_only':True,'production_writes':0,'cycles':[],
           'verified_rows':0,'verified_new_source_ids':0,
           'limitation':'Unique-key distributed samples only; source duplicate versions are retained and excluded from the benchmark, not solved for full migration.'}
    if resume:
        state=json.loads((root/'status.json').read_text())
        write_json(root/('state_before_resume_'+str(time.time_ns())+'.json'),state)
        elapsed=(datetime.fromisoformat(now())-datetime.fromisoformat(state['started_at'])).total_seconds()
        start=time.monotonic()-elapsed;deadline=start+seconds
        state.update(pid=os.getpid(),resumed_at=now(),status='resuming_after_read_response_limit')
        state.pop('error',None);state.pop('traceback',None);state.pop('finished_at',None)
    write_json(root/'status.json',state)
    guard=None
    try:
        guard=Guard(root);guard.check('initial_preflight')
        prior_ids={};durations=[c['complete_seconds'] for c in state['cycles']];cycle=0
        for directory in sorted(root.glob('cycle_*')):
            cycle=max(cycle,int(directory.name.split('_')[-1]))
            report=json.loads((directory/'report.json').read_text())
            audit=sqlite3.connect(directory/'audit.sqlite')
            for key, in audit.execute('SELECT id FROM expected_rows'):
                prefix=key[:3];prior_ids[prefix]=max(prior_ids.get(prefix,''),key)
            if report['status']=='failed_new_test_collection_preserved':
                if audit.execute("SELECT count(*) FROM batches WHERE status='submitted_outcome_not_yet_confirmed'").fetchone()[0]:
                    raise IntegrityError('Ambiguous insertion requires reconciliation; never auto-resume')
                recovery=ProductionIndexSample.__new__(ProductionIndexSample)
                recovery.root=directory;recovery.report=report;recovery.collection=report['test_collection'];recovery.lock=threading.RLock();recovery.bucket_rows=report['prefix_rows_read'];recovery.guard=guard
                recovery.source=MilvusClient(uri=SOURCE_URI,timeout=20);recovery.target=MilvusClient(uri=TARGET_URI,timeout=20)
                recovery.source_schema=json.loads((directory/'source_schema.json').read_text());recovery.source_fields=[f['name'] for f in recovery.source_schema['fields']];recovery.original_fields=[f for f in recovery.source_fields if f!='hasSourcesysSourceId']
                recovery.target_collections_before=set(report['target_collections_before']);recovery.source_indexes=report['production_index_definitions'];recovery.db=audit
                try:
                    recovery.verify();recovery.target.release_collection(recovery.collection,timeout=60)
                    recovery.report.update(status='interrupted_read_sample_recovered_and_verified',recovery_finished_at=now(),recovery_no_inserts_performed=True);recovery.save()
                    state.setdefault('diagnostic_verified_rows',0);state['diagnostic_verified_rows']+=report['verified_rows']
                finally:recovery.target.close();recovery.source.close()
            audit.close()
        while True:
            remaining=deadline-time.monotonic()
            # Reserve the previous observed complete cycle plus20%, so the test
            # has a real90minute budget instead of90minutes of inserts plus hours
            # of uncounted index work. Finish in-flight verification if it overruns.
            if cycle and remaining<max(600,max(durations[-2:])*1.2):
                state['dispatch_stop_reason']='remaining_budget_reserved_for_complete_cycle';break
            if (root/'STOP_REQUESTED').exists():
                state['dispatch_stop_reason']='user_stop';break
            cycle+=1
            batch=100 if len(state['cycles'])==0 else 200
            workers=4 if len(state['cycles'])==2 else 2
            if len(state['cycles'])>2:
                best=max(state['cycles'],key=lambda c:c['rows']/c['copy_seconds'])
                batch=best['batch_size'];workers=best['workers']
            name=COLLECTION_PREFIX+root.name.split('90min_')[-1]+'_c'+str(cycle)
            subroot=root/('cycle_%02d'%cycle)
            state.update(status='running_cycle',current_cycle=cycle,current_report=str(subroot/'report.json'))
            write_json(root/'status.json',state)
            sample=ProductionIndexSample(subroot,name,guard,prior_ids,batch,200000,600,workers)
            sample.run()
            prior_ids=dict(sample.last_ids)
            report=sample.report
            durations.append(report['seconds']['complete_cycle'])
            state['cycles'].append({'cycle':cycle,'collection':name,'report':str(subroot/'report.json'),
                'batch_size':batch,'workers':workers,'rows':report['verified_rows'],'new_source_ids':report['verified_new_source_ids'],
                'copy_seconds':report['seconds']['copy_phases_including_waits'],
                'complete_seconds':report['seconds']['complete_cycle'],
                'index_seconds':report['seconds']['exact_production_indexes'],
                'excluded_duplicate_keys':report['excluded_duplicate_keys']})
            state['verified_rows']+=report['verified_rows']
            state['verified_new_source_ids']+=report['verified_new_source_ids']
            state['elapsed_seconds']=time.monotonic()-start
            state['verified_copy_rows_per_second']=state['verified_rows']/sum(x['copy_seconds'] for x in state['cycles'])
            state['verified_end_to_end_rows_per_second']=state['verified_rows']/state['elapsed_seconds']
            state['verified_new_source_ids_per_second']=state['verified_new_source_ids']/state['elapsed_seconds']
            write_json(root/'status.json',state)
        guard.sample(force=True)
        state['services_unchanged']=True
        state['status']='completed_real_samples_and_verified_no_production_cutover'
        state['finished_at']=now()
    except BaseException as exc:
        state.update(status='test_stopped_existing_data_retained',error=repr(exc),traceback=traceback.format_exc(),finished_at=now())
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-start
        if guard:
            state['guard_check_and_wait_seconds']=guard.wait_seconds
            guard.ssh.close()
        write_json(root/'status.json',state)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.apply:parser.error('--apply required')
    # A local exclusive lock prevents a second benchmark writer after a restart.
    import msvcrt
    with (PROJECT/'outputs/fulltext_v3_benchmark_231.lock').open('a+b') as lease:
        lease.seek(0)
        if lease.read(1)==b'':lease.write(b'0');lease.flush()
        lease.seek(0)
        msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
        try:execute(args.workdir.resolve(),resume=args.resume)
        finally:
            lease.seek(0);msvcrt.locking(lease.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
