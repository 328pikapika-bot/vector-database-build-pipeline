"""Read-only live dashboard for the scientific_fulltext_chunks_v3 copy."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3

NAME='scientific_fulltext_chunks_v3'
ROOT=Path(os.environ.get('V3_WORK_ROOT','/data/scientific_fulltext_chunks_v3/run'))
SOURCE_TOTAL=195893017
BASE=335825
JOB_START_FALLBACK='2026-11-16T14:03:01+08:00'
TZ=timezone(timedelta(hours=8))

def metrics():
    now=datetime.now(TZ)
    statuses=[]
    for p in sorted(ROOT.glob('status*.json')):
        try:
            value=json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(value,dict):raise ValueError('status must be an object')
            statuses.append(value)
        except Exception as exc:
            statuses.append({'shard':p.stem.removeprefix('status_'),'status':'unreadable',
                'error':'状态文件读取失败：'+str(exc)})
    def age_of(s):
        try:return (now-datetime.fromisoformat(s.get('updated_at'))).total_seconds()
        except Exception:return None
    shard_statuses=[]
    for s in statuses:
        age=age_of(s)
        state=str(s.get('status') or 'unknown')
        fresh=age is not None and -5<=age<120
        if state=='running_no_index_copy':
            kind='running' if fresh else 'stale'
        elif state=='waiting_resource':
            kind='waiting' if fresh else 'stale'
        elif state.startswith('copy_scan_finished'):
            kind='finished'
        elif state.startswith('paused') or s.get('error'):
            kind='paused'
        else:
            kind='checking' if fresh and state=='preflight' else 'unknown'
        shard_statuses.append({'shard':s.get('shard','all'),'status':state,
            'kind':kind,'updated_at':s.get('updated_at'),'age_seconds':age,
            'error':s.get('error') or s.get('wait_reason')})
    live_shards=sum(s['kind']=='running' for s in shard_statuses)
    paused_shards=sum(s['kind']=='paused' for s in shard_statuses)
    waiting_shards=sum(s['kind']=='waiting' for s in shard_statuses)
    stale_shards=sum(s['kind'] in ('stale','unknown') for s in shard_statuses)
    blocked_shards=sum(s['kind'] not in ('running','finished') for s in shard_statuses)
    best=max(statuses,key=lambda s:s.get('updated_at','')) if statuses else {}
    age=age_of(best)
    live=live_shards>0
    status_text=';'.join(sorted({str(s.get('status')) for s in statuses if s.get('status')})) or None
    db=sqlite3.connect('file:'+ROOT.joinpath('journal.sqlite').as_posix()+'?mode=ro',uri=True,timeout=5)
    try:
        confirmed=db.execute("SELECT coalesce(sum(rows),0),coalesce(sum(added),0),max(confirmed_at),min(confirmed_at) FROM batches WHERE status='confirmed'").fetchone()
        start=now-timedelta(seconds=60)
        minute=db.execute("SELECT coalesce(sum(rows),0) FROM batches WHERE status='confirmed' AND confirmed_at>=?",(start.isoformat(timespec='seconds'),)).fetchone()[0]
        submitted=db.execute("SELECT count(*) FROM batches WHERE status='submitted'").fetchone()[0]
        unresolved=db.execute("SELECT count(*),coalesce(sum(rows),0) FROM batches WHERE status='submitted' AND (submitted_at IS NULL OR submitted_at<=?)",((now-timedelta(seconds=120)).isoformat(timespec='seconds'),)).fetchone()
        duplicate=db.execute('SELECT count(*) FROM duplicate_keys').fetchone()[0]
        finished=db.execute('SELECT count(*) FROM cursors WHERE finished=1').fetchone()[0]
    finally:db.close()
    total=BASE+confirmed[0]
    remaining=max(0,SOURCE_TOTAL-total)
    rate=minute/60
    starts=[t for t in (confirmed[3],JOB_START_FALLBACK) if t]
    job_start=min(datetime.fromisoformat(t) for t in starts)
    if job_start.tzinfo is None:job_start=job_start.replace(tzinfo=TZ)
    elapsed=max(1.0,(now-job_start).total_seconds())
    overall_rate=total/elapsed
    scan_finished=bool(shard_statuses) and all(s['kind']=='finished' for s in shard_statuses)
    resource_wait_only=waiting_shards>0 and all(s['kind'] in ('waiting','finished') for s in shard_statuses)
    if scan_finished:
        eta_reason='复制扫描已结束，统一建索引和最终验收尚未完成。'
    elif resource_wait_only:
        eta_reason='正在等待资源恢复，恢复持续写入后再估算完成时间。'
    elif not live:
        eta_reason='程序已暂停或状态未更新，恢复后再估算完成时间。'
    elif blocked_shards:
        eta_reason='部分分片暂停、检查中或状态过期，恢复后再估算完成时间。'
    elif unresolved[0]:
        eta_reason='存在结果待核对的写入批次，暂不估算全部复制完成时间。'
    elif not minute:
        eta_reason='最近一分钟没有确认写入，暂无法估算完成时间。'
    else:
        eta_reason=None
    eta=(now+timedelta(seconds=remaining/overall_rate)).isoformat() if eta_reason is None and overall_rate>0 else None
    if live:
        state_label='部分运行，存在待处理问题' if blocked_shards or unresolved[0] else '正在复制'
    elif resource_wait_only:
        state_label='等待资源恢复'
    elif scan_finished:
        state_label='复制扫描结束，等待验收'
    elif paused_shards:
        state_label='已暂停，等待处理'
    else:
        state_label='状态未更新，需检查程序'
    return {'collection':NAME,'at':now.isoformat(),'source_collection':'scientific_fulltext_chunks_v2',
        'source_total':SOURCE_TOTAL,'target_rows_confirmed':total,'copied_after_transfer':confirmed[0],
        'source_ids_added_after_transfer':confirmed[1],
        'source_ids_added_in_first_10_minutes':311825,
        'source_ids_added_total':311825+confirmed[1],
        'remaining_estimate':remaining,'progress_percent':100*total/SOURCE_TOTAL,
        'last_minute_rows':minute,'last_minute_rows_per_second':rate,
        'job_start_at':job_start.isoformat(),'elapsed_since_start_seconds':elapsed,
        'overall_rows_per_second':overall_rate,
        'estimated_copy_finish_at':eta,'estimated_days_remaining':remaining/overall_rate/86400 if eta is not None else None,
        'eta_unavailable_reason':eta_reason,
        'status':status_text,'state_label':state_label,'live':live,'status_age_seconds':age,
        'shards_live':live_shards,'shards_total':len(statuses),
        'shards_paused':paused_shards,'shards_waiting':waiting_shards,'shards_stale':stale_shards,'shard_statuses':shard_statuses,
        'last_confirmed_at':confirmed[2],'submitted_batches_total':submitted,
        'submitted_inflight_batches':submitted-unresolved[0],
        'unresolved_submitted_batches':unresolved[0],'unresolved_submitted_rows':unresolved[1],
        'duplicate_keys_quarantined_after_transfer':duplicate,
        'prefixes_finished':finished,'prefixes_total':16,
        'indexes_built':False,'note':'仅复制阶段预估；完成时间按从开始复制到现在的整体平均速度估算（含中途停顿）。最后建索引和验收另计。历史重复主键隔离记录，尚待无损方案处理。'}

class Handler(BaseHTTPRequestHandler):
    def send(self,body,content_type,status=200):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store, max-age=0')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/','/index.html'):
            self.send(Path('/app/scientific_fulltext_chunks_v3_dashboard.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/api/status':
            try:body=json.dumps(metrics(),ensure_ascii=False).encode('utf-8');self.send(body,'application/json; charset=utf-8')
            except Exception as exc:
                self.send(json.dumps({'error':str(exc)},ensure_ascii=False).encode(),'application/json; charset=utf-8',503)
        elif self.path=='/healthz':
            self.send(b'ok','text/plain; charset=utf-8')
        else:self.send(b'not found','text/plain; charset=utf-8',404)

if __name__=='__main__':
    ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
