"""Read-only local status endpoint for container health and operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def start_status_server(state_dir: Path, port: int = 8099) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/health", "/status"):
                self.send_error(404)
                return
            try:
                report = json.loads((state_dir / "progress.json").read_text(encoding="utf-8"))
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(report["at"])).total_seconds()
                healthy = age <= 45 and report.get("status") not in {"safety_pause", "storage_safety_pause"}
            except (OSError, KeyError, ValueError):
                report, healthy = {"status": "starting"}, False
            body = json.dumps(report if self.path == "/status" else {"healthy": healthy,
                              "status": report.get("status")}, ensure_ascii=False).encode()
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    return server
