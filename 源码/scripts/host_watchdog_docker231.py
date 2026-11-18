"""Dedicated test supervisor; observes existing services, never stops or edits them."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.request


def cpu():
    values = list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(values),values[3],values[4]


def services(names=None):
    if names is None:
        names = subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
    raw = json.loads(subprocess.check_output(['docker','inspect',*names],text=True,timeout=8))
    return {c['Name'].lstrip('/'):{'started':c['State']['StartedAt'],'status':c['State']['Status'],
        'health':c['State'].get('Health',{}).get('Status'),'restarts':c['RestartCount']} for c in raw}


def disks():
    return {r.split()[2]:int(r.split()[12]) for r in Path('/proc/diskstats').read_text().splitlines()
        if r.split()[2] in ('sda','sdb')}


class Monitor:
    def __init__(self, folder):
        self.root = Path(folder)
        self.baseline = services()
        self.stop = threading.Event()
        self.streak = {}
        self.previous_cpu = cpu()
        self.previous_disks = disks()
        self.previous_time = time.monotonic()
        self.samples = []
        (self.root/'services_before.json').write_text(json.dumps(self.baseline,indent=2))

    def sample(self):
        t = time.monotonic()
        current = cpu()
        delta = max(current[0]-self.previous_cpu[0],1)
        busy = 100*(1-(current[1]-self.previous_cpu[1])/delta)
        iowait = 100*(current[2]-self.previous_cpu[2])/delta
        disk = disks()
        util = {d:100*(n-self.previous_disks.get(d,n))/max((t-self.previous_time)*1000,1) for d,n in disk.items()}
        self.previous_cpu,self.previous_disks,self.previous_time = current,disk,t
        available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines()
            if r.startswith('MemAvailable:')))*1024
        free = {p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/','/data']}
        reasons = []
        now_services = services(list(self.baseline))
        if now_services != self.baseline:
            reasons.append('Existing service state/restart/health changed')
        checks = {'cpu_busy':busy>60,'iowait':iowait>10,
                  'disk_util':max(util.values(),default=0)>75}
        latency = {}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:9094/metrics',timeout=5) as r:
            metrics = r.read().decode()
        index_tasks = {'pending':0,'running':0}
        for line in metrics.splitlines():
            if line.startswith('milvus_datacoord_task_num_in_scheduler{') and 'task_type="Index"' in line:
                if 'task_state="JobStateInProgress"' in line:
                    index_tasks['running'] += int(float(line.rsplit(' ',1)[1]))
                elif 'task_state="JobStateInit"' in line:
                    index_tasks['pending'] += int(float(line.rsplit(' ',1)[1]))
        for port in (9093,9094,9095):
            start = time.monotonic()
            try:
                with opener.open('http://127.0.0.1:%d/healthz'%port,timeout=2) as r:
                    if r.status != 200:
                        raise RuntimeError('health endpoint returned non-200')
                    r.read()
                latency[str(port)] = time.monotonic()-start
                checks['latency_'+str(port)] = latency[str(port)]>0.5
            except Exception as exc:
                latency[str(port)] = str(exc)
                checks['latency_'+str(port)] = True
        for k,triggered in checks.items():
            self.streak[k] = self.streak.get(k,0)+1 if triggered else 0
            if self.streak[k] >= 3:
                reasons.append(k+' exceeded protection threshold for 3 samples')
        if available < 250*1024**3:
            reasons.append('Available RAM below 250GiB')
        if free['/'] < 40*1024**3 or free['/data'] < 1024**4:
            reasons.append('Disk free-space guard')
        data = {'timestamp':time.time(),'safe':not reasons,'reasons':reasons,'cpu_busy_percent':busy,
            'iowait_percent':iowait,'disk_util_percent':util,'available_ram_bytes':available,
            'disk_free_bytes':free,'health_latency_seconds':latency,'services_unchanged':now_services==self.baseline}
        data.update(index_tasks_pending=index_tasks['pending'],index_tasks_running=index_tasks['running'])
        tmp = self.root/'host_monitor.tmp'
        tmp.write_text(json.dumps(data,indent=2))
        os.replace(tmp,self.root/'host_monitor.json')
        with (self.root/'host_monitor_history.jsonl').open('a') as out:
            out.write(json.dumps(data)+'\n')
        self.samples.append(data)
        if reasons:
            (self.root/'STOP_REQUESTED').write_text('; '.join(reasons))
        return data

    def loop(self):
        while not self.stop.wait(10):
            try:
                self.sample()
            except Exception as exc:
                (self.root/'STOP_REQUESTED').write_text('Monitor error: '+repr(exc))
                (self.root/'monitor_error.txt').write_text(repr(exc))
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--previous',default='/data/scientific-fulltext-source-id-single231-20261113')
    parser.add_argument('--adaptive',action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    if (root/'run'/'host_monitor_history.jsonl').exists():
        raise RuntimeError('Refusing to overwrite an existing test')
    monitor = Monitor(root/'run')
    time.sleep(1)
    if not monitor.sample()['safe']:
        raise RuntimeError('Unsafe host preflight')
    thread = threading.Thread(target=monitor.loop)
    thread.start()
    name = root.name
    old = '/data/scientific-fulltext-source-id-backfill-20261113'
    previous = args.previous
    adaptive = args.adaptive
    args = ['docker','run','--name',name,'--user','0:0','--network','host','--cpus','8','--cpu-shares','128',
        '--memory','16g','--memory-swap','16g','--pids-limit','512','--read-only',
        '--tmpfs','/tmp:rw,noexec,nosuid,size=256m','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--log-opt','max-size=20m','--log-opt','max-file=2',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
        '-v',str(root/'scripts')+':/test:ro','-v',str(root/'run')+':/work:rw']
    for filename,mode in [('progress.sqlite','ro'),('progress.sqlite-wal','ro'),
                          ('progress.sqlite-shm','ro'),('STOP_REQUESTED','ro'),('runner.lock','rw')]:
        args += ['-v',old+'/'+filename+':'+old+'/'+filename+':'+mode]
    for filename in ['report.json','journal.sqlite']:
        args += ['-v',previous+'/'+filename+':/previous/'+filename+':ro']
    args += ['--entrypoint','python',
        'registry.example.invalid/nstl/vector-sync-service:v3-metadata-prefilter-20261016',
        '/test/benchmark_fulltext_source_id_docker231.py','--workdir','/work','--seconds',
        '420' if adaptive else '900','--apply']
    if adaptive:
        args += ['--adaptive']
    # The monitor writes into run/, the only writable mount visible to the test.
    # root is the run directory's parent; monitor data is linked by directory choice below.
    try:
        with (root/'container.log').open('w') as log:
            code = subprocess.call(args,stdout=log,stderr=subprocess.STDOUT)
        monitor.stop.set()
        thread.join(timeout=20)
        monitor.sample()
        after = services(list(monitor.baseline))
        (root/'services_after.json').write_text(json.dumps(after,indent=2))
        summary = {'container_exit_code':code,'services_unchanged':after==monitor.baseline,
            'samples':len(monitor.samples),
            'max_cpu_busy_percent':max(s['cpu_busy_percent'] for s in monitor.samples),
            'max_iowait_percent':max(s['iowait_percent'] for s in monitor.samples),
            'min_available_ram_bytes':min(s['available_ram_bytes'] for s in monitor.samples),
            'protection_triggered':any(not s['safe'] for s in monitor.samples)}
        (root/'host_summary.json').write_text(json.dumps(summary,indent=2))
    finally:
        monitor.stop.set()
        thread.join(timeout=20)


if __name__ == '__main__':
    main()
