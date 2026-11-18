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
TZ=timezone(timedelta(hours=8))

def metrics():
    now=datetime.now(TZ)
    status=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
    db=sqlite3.connect('file:'+ROOT.joinpath('journal.sqlite').as_posix()+'?mode=ro',uri=True,timeout=5)
    try:
        confirmed=db.execute("SELECT coalesce(sum(rows),0),coalesce(sum(added),0),max(confirmed_at) FROM batches WHERE status='confirmed'").fetchone()
        start=now-timedelta(seconds=60)
        minute=db.execute("SELECT coalesce(sum(rows),0) FROM batches WHERE status='confirmed' AND confirmed_at>=?",(start.isoformat(timespec='seconds'),)).fetchone()[0]
        submitted=db.execute("SELECT count(*) FROM batches WHERE status='submitted'").fetchone()[0]
        duplicate=db.execute('SELECT count(*) FROM duplicate_keys').fetchone()[0]
        finished=db.execute('SELECT count(*) FROM cursors WHERE finished=1').fetchone()[0]
    finally:db.close()
    total=BASE+confirmed[0]
    remaining=max(0,SOURCE_TOTAL-total)
    rate=minute/60
    eta=(now+timedelta(seconds=remaining/rate)).isoformat() if rate>0 else None
    updated=status.get('updated_at')
    try:age=(now-datetime.fromisoformat(updated)).total_seconds()
    except Exception:age=None
    live=status.get('status')=='running_no_index_copy' and age is not None and age<120
    return {'collection':NAME,'at':now.isoformat(),'source_collection':'scientific_fulltext_chunks_v2',
        'source_total':SOURCE_TOTAL,'target_rows_confirmed':total,'copied_after_transfer':confirmed[0],
        'source_ids_added_after_transfer':confirmed[1],
        'source_ids_added_in_first_10_minutes':311825,
        'source_ids_added_total':311825+confirmed[1],
        'remaining_estimate':remaining,'progress_percent':100*total/SOURCE_TOTAL,
        'last_minute_rows':minute,'last_minute_rows_per_second':rate,
        'estimated_copy_finish_at':eta,'estimated_days_remaining':remaining/rate/86400 if rate>0 else None,
        'status':status.get('status'),'live':live,'status_age_seconds':age,
        'last_confirmed_at':confirmed[2],'submitted_inflight_batches':submitted,
        'duplicate_keys_quarantined_after_transfer':duplicate,
        'prefixes_finished':finished,'prefixes_total':16,
        'indexes_built':False,'note':'仅复制阶段预估；最后建索引和验收另计。历史重复主键隔离记录，尚待无损方案处理。'}

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
