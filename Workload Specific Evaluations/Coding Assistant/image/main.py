#!/usr/bin/env python3
"""AgentCore shim: runs the firewall as root, then serves /ping and /invocations on :8080."""
import json, os, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

t0 = time.time()
try:
    r = subprocess.run(["/opt/firewall/start.sh"], capture_output=True, text=True, timeout=60)
    FW = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-500:]}
except Exception as e:  # start.sh decides pass/fail; here we only report
    FW = {"rc": -1, "out": repr(e)}
FW["secs"] = round(time.time() - t0, 2)


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/ping":
            self._send({"status": "Healthy" if FW["rc"] == 0 else "Unhealthy", "firewall": FW["rc"]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self._send({"firewall": FW, "uid": os.getuid()})

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
