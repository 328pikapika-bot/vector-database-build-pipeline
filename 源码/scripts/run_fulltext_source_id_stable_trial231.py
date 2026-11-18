"""Bounded real backfill trials with a reversible Milvus CPU quota.

Only the user's 231 source-ID field is written, through the existing audited
runner. This wrapper never deletes rows, collections, indexes, or containers.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

import paramiko

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / 'outputs/fulltext_source_id_local231_continuous_20261113'
OUT = PROJECT / 'outputs/fulltext_source_id_stable_trial_231_20261115'
TARGET = 'scientific-milvus-standalone'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


REMOTE_SAMPLE = r'''
import subprocess,json,hashlib,urllib.request,os
from pathlib import Path
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
containers=json.loads(subprocess.check_output(['docker','inspect',*names],text=True))
r={'cpu':list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9])), 'services':{},'host_cpu_count':os.cpu_count()}
for x in containers:
 n=x['Name'].lstrip('/'); hc=dict(x['HostConfig'])
 if n=='scientific-milvus-standalone':
  hc.pop('CpuQuota',None);hc.pop('CpuPeriod',None)
 mounts=sorted(x['Mounts'],key=lambda m:m['Destination'])
 r['services'][n]={'id':x['Id'],'started':x['State']['StartedAt'],'restarts':x['RestartCount'],'status':x['State']['Status'],'health':x['State'].get('Health',{}).get('Status'),'config_hash':hashlib.sha256(json.dumps({'config':x['Config'],'host_config':hc,'mounts':mounts},sort_keys=True).encode()).hexdigest()}
 if n=='scientific-milvus-standalone':r['milvus_cpu_config']={k:x['HostConfig'][k] for k in ['NanoCpus','CpuQuota','CpuPeriod','CpusetCpus']}
r['index_pending']=0;r['index_running']=0;r['compaction_pending']=0;r['compaction_running']=0;r['compaction_metrics']=[]
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
metrics=op.open('http://127.0.0.1:9094/metrics',timeout=3).read().decode()
for line in metrics.splitlines():
 if line.startswith('milvus_datacoord_task_num_in_scheduler{') and 'task_type="Index"' in line:
  if 'task_state="JobStateInit"' in line:r['index_pending']+=int(float(line.rsplit(' ',1)[1]))
  if 'task_state="JobStateInProgress"' in line:r['index_running']+=int(float(line.rsplit(' ',1)[1]))
 if line.startswith('milvus_datacoord_compaction_task_num{'):
  r['compaction_metrics'].append(line)
  if 'status="pending"' in line:r['compaction_pending']+=int(float(line.rsplit(' ',1)[1]))
  if 'status="executing"' in line:r['compaction_running']+=int(float(line.rsplit(' ',1)[1]))
r['compaction_pending']=max(0,r['compaction_pending']);r['compaction_running']=max(0,r['compaction_running'])
r['health']={}
import time
for port in [9093,9094,9095]:
 t=time.monotonic()
 with op.open('http://127.0.0.1:%d/healthz'%port,timeout=2) as h:r['health'][str(port)]={'status':h.status,'seconds':time.monotonic()-t}
print(json.dumps(r))
'''


class Remote:
    def __init__(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect('192.0.2.15', username='root',
                            password=os.environ['BACKFILL_SSH_PASSWORD'], timeout=12)
        self.previous_cpu = None

    def command(self, args):
        _, out, err = self.client.exec_command(shlex.join(args), timeout=20)
        value, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-1000:])
        return value

    def sample(self):
        r = json.loads(self.command(['python3', '-c', REMOTE_SAMPLE]))
        cpu = r.pop('cpu')
        if self.previous_cpu is not None:
            delta = [b-a for a,b in zip(self.previous_cpu,cpu)]
            r['host_cpu_percent'] = 100*(sum(delta)-delta[3])/max(sum(delta),1)
            r['host_iowait_percent'] = 100*delta[4]/max(sum(delta),1)
        self.previous_cpu = cpu
        r['at'] = now()
        return r


def journal():
    db = sqlite3.connect('file:'+WORK.joinpath('journal.sqlite').as_posix()+'?mode=ro', uri=True)
    try:
        return {'pending':db.execute("SELECT count(*) FROM batches WHERE status!='completed'").fetchone()[0],
                'max_seq':db.execute('SELECT max(seq) FROM batches').fetchone()[0],
                'verified_newly_filled':db.execute("SELECT coalesce(sum(json_extract(result,'$.newly_filled')),0) FROM batches WHERE status='completed'").fetchone()[0]}
    finally:
        db.close()


def field_counts():
    from benchmark_fulltext_source_id_100 import COLLECTION, FIELD, connect
    c = connect('192.0.2.15')
    try:
        return {'total':int(c.query(COLLECTION,filter='',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)']),
                'filled':int(c.query(COLLECTION,filter=FIELD+' is not null',output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])}
    finally:
        c.close()


def child_main(rate, seconds, batch_size):
    import run_fulltext_source_id_local231 as module

    class TrialController(module.PressureController):
        def __init__(self):
            super().__init__(initial_rate=rate, max_rate=rate)

        def decision(self, host, rows, monotonic_now, wall_now):
            # Keep every original guard. The trial tests actual index capacity
            # instead of using an arbitrary pending-count rate budget.
            trial_host = dict(host) if host else host
            if trial_host:
                if trial_host.get('index_tasks_pending',0) >= 500:
                    stop = WORK/'STOP_REQUESTED'
                    if not stop.exists():
                        stop.write_text('Bounded trial: index queue reached 500; stop dispatch and drain.\n')
                trial_host['index_tasks_pending'] = 0
            return super().decision(trial_host, rows, monotonic_now, wall_now)

    runner = module.LocalRunner(WORK, batch_size=batch_size, workers=2,
                                duration_seconds=seconds)
    runner.controller = TrialController()
    runner.run()


class AdoptedProcess:
    """Observe an existing bounded writer without interrupting or restarting it."""
    def __init__(self, pid):
        self.pid=pid
        self.returncode=None
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        self.handle=kernel.OpenProcess(0x100000|0x1000,False,pid)
        if not self.handle:raise RuntimeError('Could not open bounded writer for observation')
    def poll(self):
        if ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(self.handle),0)==0x102:return None
        result=ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(ctypes.c_void_p(self.handle),ctypes.byref(result)):
            raise RuntimeError('Could not read bounded writer exit code')
        self.returncode=result.value
        return self.returncode


def orchestrate(adopt_pid=None, resume_drain=None, extra_trial=False):
    resuming=adopt_pid is not None or resume_drain is not None
    if not resuming:
        OUT.mkdir(parents=True, exist_ok=False)
        report = {'at':now(),'pid':os.getpid(),'status':'preflight','stages':[],
                  'source_workdir':str(WORK),'cpu_quota_cores':56}
    else:
        report=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
        write_json(OUT/('coordinator_handoff_before_'+str(time.time_ns())+'.json'),report)
        report['previous_coordinator_pid']=report['pid']
        report['pid']=os.getpid()
        report['coordinator_handoff_at']=now()
        if adopt_pid is not None and report['stages'][0]['child_pid']!=adopt_pid:raise RuntimeError('Unexpected writer PID for handoff')
        if resume_drain is not None:
            if resume_drain not in (1,2,3) or len(report['stages'])!=resume_drain or 'runner_final' not in report['stages'][resume_drain-1]:
                raise RuntimeError('Drain resume requires the corresponding completed bounded writer')
            report['resumed_drain_at']=now()
            completed=report['stages'][resume_drain-1]['runner_final']
            if completed.get('integrity_hold') or completed.get('pending_batches'):
                raise RuntimeError('Cannot resume drain with unresolved data verification')
            if report.get('error'):
                report['previous_observation_error']=report.pop('error')
                report['previous_observation_stop_reason']=report.pop('stop_reason',None)
    child = None
    quota_applied = False
    remote = None
    mix_baseline=None
    mix_may_be_changed=False
    property_key='collection.autocompaction.enabled'
    owner_path=OUT/'mix_property_owner.json'
    owner={'pid':os.getpid(),'status':'baseline_unchanged','at':now()}
    def save():
        report['at'] = now()
        write_json(OUT/'status.json', report)
    def stop_dispatch(reason):
        report['stop_reason'] = reason
        if not (WORK/'STOP_REQUESTED').exists():
            (WORK/'STOP_REQUESTED').write_text('Bounded trial: '+reason+'\n', encoding='utf-8')
        save()
    def sample():
        observation = remote.sample()
        with (OUT/'host_observations.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(observation)+'\n')
        report['host'] = observation
        save()
        return observation
    def property_client():
        from benchmark_fulltext_source_id_100 import connect
        return connect('192.0.2.15')
    def owner_save():
        owner['at']=now();write_json(owner_path,owner)
    def disable_mix():
        nonlocal mix_may_be_changed
        from benchmark_fulltext_source_id_100 import COLLECTION
        c=property_client()
        try:
            current=dict(c.describe_collection(COLLECTION).get('properties') or {})
            if current!=mix_baseline:raise RuntimeError('Properties changed before Mix-off trial')
            mix_may_be_changed=True
            owner['status']='applying_mix_off';owner_save()
            c.alter_collection_properties(collection_name=COLLECTION,properties={property_key:'false'},timeout=30)
            current=dict(c.describe_collection(COLLECTION).get('properties') or {})
            expected=dict(mix_baseline);expected[property_key]='false'
            if current!=expected:raise RuntimeError('Mix-off property verification failed')
            owner['properties_during_trial']=current;owner['mix_off_applied_at']=now();owner['status']='mix_off_verified';owner_save()
            report['mix_off_applied_at']=owner['mix_off_applied_at'];save()
        finally:c.close()
    def restore_mix():
        nonlocal mix_may_be_changed
        from benchmark_fulltext_source_id_100 import COLLECTION
        while mix_may_be_changed:
            c=None
            try:
                owner['status']='restoring_original_properties';owner_save()
                c=property_client()
                current=dict(c.describe_collection(COLLECTION).get('properties') or {})
                if current!=mix_baseline:
                    if property_key in mix_baseline:
                        c.alter_collection_properties(collection_name=COLLECTION,properties={property_key:mix_baseline[property_key]},timeout=30)
                    else:
                        c.drop_collection_properties(collection_name=COLLECTION,property_keys=[property_key],timeout=30)
                current=dict(c.describe_collection(COLLECTION).get('properties') or {})
                if current!=mix_baseline:raise RuntimeError('Original collection properties not fully restored')
                mix_may_be_changed=False
                owner['properties_after_restore']=current;owner['property_restored_at']=now();owner['status']='original_properties_restored';owner_save()
                report['mix_property_restored_at']=owner['property_restored_at'];save()
            except Exception as exc:
                owner['restore_error']=repr(exc);owner['status']='restore_retry_cpu_cap_retained';owner_save()
                report['status']='cpu_cap_retained_until_property_restore';save();time.sleep(15)
            finally:
                if c is not None:c.close()
    def drain(label, flush=True):
        started = time.monotonic()
        quiet_since = None
        report['status'] = label
        save()
        from benchmark_fulltext_source_id_100 import COLLECTION, connect
        if flush:
            c=connect('192.0.2.15')
            try:c.flush(COLLECTION,timeout=120)
            finally:c.close()
            report['last_flush_at']=now()
        while True:
            try:s = sample()
            except Exception as exc:
                report['observation_retry_error']=repr(exc);report['status']=label+'_observation_retry_cap_retained';quiet_since=None;save()
                time.sleep(5)
                try:remote.client.close()
                except Exception:pass
                try:
                    replacement=Remote();remote.client=replacement.client;remote.previous_cpu=None
                except Exception:pass
                continue
            report['status']=label
            if s['services'] != report['services_before']:
                report['service_state_warning'] = 'Container state/configuration changed during index drain'
            drain_cpu_max=20
            control_path=OUT/'control.json'
            if control_path.exists():drain_cpu_max=float(json.loads(control_path.read_text()).get('drain_cpu_max',20))
            report['drain_cpu_max']=drain_cpu_max
            idle=(s['services']==report['services_before'] and all(h['status']==200 and h['seconds']<=0.25 for h in s['health'].values()) and not s['index_pending'] and not s['index_running'] and not s['compaction_pending'] and not s['compaction_running'] and s.get('host_cpu_percent',100)<drain_cpu_max)
            if idle:
                if quiet_since is None:quiet_since=time.monotonic()
                report['background_quiet_seconds']=time.monotonic()-quiet_since
                if report['background_quiet_seconds']>=90:break
            else:
                quiet_since=None
                report['background_quiet_seconds']=0
            save();time.sleep(5)
        return time.monotonic()-started
    save()
    try:
        if resuming:
            remote=Remote();quota_applied=True
            if adopt_pid is not None:child=AdoptedProcess(adopt_pid)
            s=sample()
            if s['services']!=report['services_before'] or s['milvus_cpu_config']['CpuQuota']!=5600000 or s['milvus_cpu_config']['NanoCpus']!=0:
                raise RuntimeError('Handoff protected service state/CPU quota verification failed')
        if not resuming:
            raw = subprocess.check_output(['powershell.exe','-NoProfile','-Command',
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],text=True)
            processes = json.loads(raw) if raw.strip() else []
            if isinstance(processes,dict): processes=[processes]
            conflicts = [p for p in processes if p['ProcessId']!=os.getpid() and
                         any(s in (p.get('CommandLine') or '') for s in
                             ['run_fulltext_source_id_local231.py','benchmark_fulltext_source_id','run_fulltext_source_id_backfill.py'])]
            if conflicts: raise RuntimeError('Another local source-ID process exists')
            report['journal_before'] = journal()
            if report['journal_before']['pending']:
                raise RuntimeError('Pending journal batches need explicit review before trial')
            shutil.copy2(WORK/'launcher.json',OUT/'launcher_before.json')
            shutil.copy2(WORK/'status.json',OUT/'source_status_before.json')
            if (WORK/'STOP_REQUESTED').exists():
                shutil.copy2(WORK/'STOP_REQUESTED',OUT/'STOP_REQUESTED_before')
            remote = Remote()
            lock_code="import fcntl; f=open('/data/scientific-fulltext-source-id-backfill-20261113/runner.lock','r+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('FREE')"
            if remote.command(['python3','-c',lock_code]).strip()!='FREE':
                raise RuntimeError('Shared backfill lock is not free')
            s = sample();time.sleep(2);s = sample()
            report['services_before'] = s['services']
            report['cpu_config_before'] = s['milvus_cpu_config']
            if s['milvus_cpu_config']['NanoCpus']!=0 or s['milvus_cpu_config']['CpuQuota'] not in (0,-1) or s['milvus_cpu_config']['CpuPeriod'] not in (0,100000):
                raise RuntimeError('Unexpected production CPU configuration')
            if s.get('host_cpu_percent',100)>20 or s['index_pending'] or s['index_running']:
                raise RuntimeError('Host/index baseline is not idle enough for bounded test')
            report['counts_before'] = field_counts()
            if report['counts_before']['total']!=195893017:
                raise RuntimeError('Source count mismatch before trial')
            quota_applied = True
            remote.command(['docker','update','--cpu-period','100000','--cpu-quota','5600000',TARGET])
            report['quota_applied_at'] = now()
            s = sample()
            if s['services']!=report['services_before'] or s['milvus_cpu_config']!={'NanoCpus':0,'CpuQuota':5600000,'CpuPeriod':100000,'CpusetCpus':''}:
                raise RuntimeError('CPU quota verification or protected service state check failed')
        c=property_client()
        try:mix_baseline=dict(c.describe_collection('scientific_fulltext_chunks_v2').get('properties') or {})
        finally:c.close()
        if owner_path.exists() and resume_drain is not None and resume_drain>=2:
            previous_owner=json.loads(owner_path.read_text(encoding='utf-8'))
            if previous_owner.get('status')!='mix_off_verified':raise RuntimeError('Cannot adopt an uncertain Mix property state')
            intended=dict(previous_owner['original_properties']);intended[property_key]='false'
            if mix_baseline!=intended:raise RuntimeError('Current Mix-off properties differ from owned state')
            mix_baseline=previous_owner['original_properties'];mix_may_be_changed=True
            owner.update(previous_owner);owner['previous_pid']=owner['pid'];owner['pid']=os.getpid();owner['adopted_at']=now()
        elif (OUT/'mix_ab_state.json').exists():
            old_ab=json.loads((OUT/'mix_ab_state.json').read_text(encoding='utf-8'))
            if mix_baseline!=old_ab['original_properties']:raise RuntimeError('Property baseline differs during coordinator takeover')
        owner['original_properties']=mix_baseline;owner_save()
        configurations=[(240,2000),(400,2000)]+([(600,3000)] if extra_trial else [])
        for ordinal,(rate,batch_size) in enumerate(configurations,1):
            if resume_drain is not None and ordinal<resume_drain:continue
            if (adopt_pid is not None and ordinal==1) or resume_drain==ordinal:
                stage=report['stages'][ordinal-1]
            else:
                if ordinal==2:disable_mix()
                if (WORK/'STOP_REQUESTED').exists():
                    (WORK/'STOP_REQUESTED').rename(OUT/f'STOP_REQUESTED_preserved_before_stage{ordinal}')
                stage={'ordinal':ordinal,'rate_ceiling':rate,'batch_size':batch_size,'workers':2,
                       'dispatch_seconds':600,'started_at':now(),'journal_before':journal()}
                report['stages'].append(stage)
                report['status']='starting_stage';save()
                with (OUT/f'stage{ordinal}.stdout.log').open('w',encoding='utf-8') as out, (OUT/f'stage{ordinal}.stderr.log').open('w',encoding='utf-8') as err:
                    child=subprocess.Popen([sys.executable,__file__,'--child','--rate',str(rate),'--seconds','600','--batch-size',str(batch_size)],cwd=PROJECT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                stage['child_pid']=child.pid
                write_json(WORK/'launcher.json',{'pid':child.pid,'script':str(Path(__file__).resolve()),'workdir':str(WORK),'execution_host':'local_windows','write_host':'192.0.2.15','launched_at':now(),'batch_size':batch_size,'workers':2,'rate_ceiling':rate,'bounded_seconds':600,'trial_wrapper_pid':os.getpid(),'trial_output':str(OUT)})
            report['status']='running_stage' if child is not None else 'resuming_completed_stage_drain';save()
            while child is not None and child.poll() is None:
                try:s=sample()
                except Exception as exc:
                    report['observation_retry_error']=repr(exc);report['status']='writer_observation_retry_cpu_cap_retained';save()
                    time.sleep(5)
                    try:remote.client.close()
                    except Exception:pass
                    try:
                        replacement=Remote();remote.client=replacement.client;remote.previous_cpu=None
                    except Exception:pass
                    continue
                report['status']='running_stage'
                if s['services']!=report['services_before']:
                    stop_dispatch('protected service state/configuration changed')
                elif any(h['status']!=200 or h['seconds']>0.25 for h in s['health'].values()):
                    stop_dispatch('protected service health became slow/unhealthy')
                elif s.get('host_cpu_percent',0)>75 or s.get('host_iowait_percent',0)>5:
                    stop_dispatch('host pressure threshold reached')
                elif s['index_pending']>=500:
                    stop_dispatch('index queue reached 500')
                if (WORK/'status.json').exists():
                    current=json.loads((WORK/'status.json').read_text(encoding='utf-8'))
                    if current.get('pid')==child.pid:
                        stage['progress']=current.get('trial')
                        stage['runner_state']=current.get('status')
                        stage['runner_reason']=current.get('reason')
                        if current.get('integrity_hold'):
                            stop_dispatch('integrity hold: '+str(current['integrity_hold']))
                save();time.sleep(5)
            if child is not None:stage['returncode']=child.returncode
            child=None
            final=json.loads((WORK/'status.json').read_text(encoding='utf-8'))
            stage['runner_final']=final
            stage['journal_after']=journal()
            stage.setdefault('finished_writing_at',now())
            stage['counts_after']=field_counts()
            if stage['returncode']!=0 or stage['journal_after']['pending'] or final.get('integrity_hold') or stage['counts_after']['total']!=195893017:
                raise RuntimeError('Bounded trial did not end with a clean verified journal')
            if resume_drain==ordinal:
                stage.setdefault('drain_started_at',report.get('last_flush_at',now()))
            else:stage['drain_started_at']=now()
            stage['drain_seconds_this_coordinator']=drain('draining_index_after_stage'+str(ordinal),flush=resume_drain!=ordinal)
            stage['index_drained_at']=now()
            stage['index_drain_seconds']=datetime.fromisoformat(stage['index_drained_at']).timestamp()-datetime.fromisoformat(stage['drain_started_at']).timestamp()
            trial=final.get('trial') or {}
            stage['verified_trial_newly_filled']=(trial.get('new_counts') or {}).get('newly_filled',0)
            elapsed=trial.get('elapsed_seconds_including_inflight_waits',0)
            stage['rows_per_second_including_index_drain']=stage['verified_trial_newly_filled']/max(elapsed+stage['index_drain_seconds'],1)
            if trial.get('started_at'):
                stage['full_wall_seconds_until_background_ready']=datetime.fromisoformat(stage['index_drained_at']).timestamp()-datetime.fromisoformat(trial['started_at']).timestamp()
                stage['verified_rows_per_second_until_background_ready']=stage['verified_trial_newly_filled']/max(stage['full_wall_seconds_until_background_ready'],1)
            save()
            if report.get('stop_reason') or (trial and trial.get('dispatch_stop_reason')!='duration_elapsed'):
                break
        report['counts_after']=field_counts()
        report['journal_after']=journal()
        if report['counts_after']['filled']-report['counts_before']['filled']!=report['journal_after']['verified_newly_filled']-report['journal_before']['verified_newly_filled']:
            raise RuntimeError('Final source-field count differs from the verified journal delta')
        report['status']='completed_and_verified'
    except BaseException as exc:
        report['error']=repr(exc)
        report['status']='error_draining'
        stop_dispatch('wrapper error: '+type(exc).__name__)
        if child is not None:
            while child.poll() is None:
                if remote is not None:
                    try:sample()
                    except Exception:pass
                time.sleep(5)
        report['status']='failed'
    finally:
        if not (WORK/'STOP_REQUESTED').exists():
            (WORK/'STOP_REQUESTED').write_text('Bounded real trial finished; preserve stop until user-directed next action.\n',encoding='utf-8')
        if remote is not None and quota_applied:
            if mix_may_be_changed:restore_mix()
            while True:
                try:
                    report['final_index_drain_seconds']=drain('final_index_drain_before_quota_restore')
                    break
                except Exception as exc:
                    report['status']='cpu_cap_retained_until_background_verified'
                    report['drain_observation_error']=repr(exc);save()
                    time.sleep(15)
                    try:remote.client.close()
                    except Exception:pass
                    try:remote=Remote()
                    except Exception:pass
            while True:
                try:
                    report['status']='restoring_cpu_quota';save()
                    remote.command(['docker','update','--cpu-quota','-1',TARGET])
                    after=remote.sample()
                    if after['milvus_cpu_config']['NanoCpus']!=0 or after['milvus_cpu_config']['CpuQuota']!=-1 or after['services']!=report['services_before']:
                        raise RuntimeError('CPU restoration could not be fully verified')
                    report['cpu_quota_restored_at']=now();report['after_restore']=after
                    report['unlimited_semantics_restored']=True;report['other_services_unchanged']=True
                    report['status']='completed_and_verified' if 'error' not in report else 'failed'
                    break
                except Exception as exc:
                    report['status']='cpu_restore_uncertain_reapplying_safe_cap';report['cpu_restore_error']=repr(exc);save()
                    while True:
                        try:
                            try:remote.client.close()
                            except Exception:pass
                            remote=Remote()
                            remote.command(['docker','update','--cpu-period','100000','--cpu-quota','5600000',TARGET])
                            confirmed=remote.sample()
                            if confirmed['milvus_cpu_config']['CpuQuota']!=5600000:raise RuntimeError('Safe cap verification failed')
                            report['host']=confirmed;report['status']='safe_cap_reapplied_waiting_for_background';save()
                            break
                        except Exception as retry:
                            report['safe_cap_retry_error']=repr(retry);save();time.sleep(15)
                    while True:
                        try:drain('final_index_drain_after_cpu_restore_retry');break
                        except Exception as retry:
                            report['drain_observation_error']=repr(retry);save();time.sleep(15)
                            try:remote.client.close()
                            except Exception:pass
                            try:remote=Remote()
                            except Exception:pass
        report['finished_at']=now()
        save()
        if remote is not None:remote.client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',action='store_true')
    p.add_argument('--adopt-pid',type=int)
    p.add_argument('--resume-drain',type=int)
    p.add_argument('--extra-trial',action='store_true')
    p.add_argument('--rate',type=int,default=240)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=2000)
    a=p.parse_args()
    if a.child:child_main(a.rate,a.seconds,a.batch_size)
    else:orchestrate(a.adopt_pid,a.resume_drain,a.extra_trial)
