"""
SIH26174 - Monitoring dashboard.

Serves a mission-control style web UI from inside the running process:
  /            ui/index.html
  /style.css   ui/style.css
  /app.js      ui/app.js
  /stream      MJPEG video (degraded, as a bandwidth-constrained downlink)
  /state       JSON snapshot, polled by the page

The front end lives in ui/ as ordinary files so it can be edited without
touching Python. Nothing is fetched from a CDN and no webfonts are loaded:
the system has to keep working with the link to ground pulled out, so a
dashboard that phones home for a stylesheet would fail the very requirement
it exists to demonstrate.
"""

import json
import mimetypes
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class Dashboard:
    """Serves the UI, the MJPEG stream and a JSON state snapshot."""

    def __init__(self, port=8080, width=480, every=3, ui_dir=None):
        self.jpeg = None
        self.lock = threading.Lock()
        self.state = {
            "protocol": "", "current": "", "confidence": 0.0, "fps": 0.0,
            "steps": [], "alerts": [], "log": [], "next_prompt": "",
            "recording": False, "session": "", "n_events": 0,
            "alerting": False, "probs": [], "report_note": "",
        }
        self.report_source = None    # callable -> html, set by run_live
        self.voice_api = None        # object with voice_list/voice_current/set_voice
        self.width = width
        self.every = every
        self.count = 0
        self.port = port
        self.url = f"http://{local_ip()}:{port}/"
        self.ui_dir = os.path.abspath(ui_dir or self._find_ui())
        if not os.path.isdir(self.ui_dir):
            raise SystemExit(
                f"UI folder not found: {self.ui_dir}\n"
                "Expected ui/index.html, ui/style.css, ui/app.js. "
                "Run from the repo root or pass ui_dir=.")
        srv = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                path = self.path.split("?")[0]
                if path != "/voice":
                    self.send_error(404)
                    return
                try:
                    # Cap the body: this binds 0.0.0.0, so never trust its size.
                    n = int(self.headers.get("Content-Length", "0"))
                    if not (0 < n <= 512):
                        raise ValueError("bad length")
                    idx = json.loads(self.rfile.read(n).decode("utf-8")).get("index")
                    api = srv.voice_api
                    ok = bool(api is not None and isinstance(idx, int)
                              and not isinstance(idx, bool) and api.set_voice(idx))
                    body = json.dumps({"ok": ok, "current": getattr(
                        api, "voice_current", 0)}).encode("utf-8")
                    self._send(body, "application/json")
                except Exception:
                    self.send_error(400)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body, ctype = srv.read_ui("index.html")
                    if body is None:
                        self.send_error(500, "ui/index.html missing")
                        return
                    self._send(body, ctype)
                elif path in ("/style.css", "/app.js"):
                    body, ctype = srv.read_ui(path.lstrip("/"))
                    if body is None:
                        self.send_error(404)
                        return
                    self._send(body, ctype)
                elif path == "/state":
                    with srv.lock:
                        body = json.dumps(srv.state).encode("utf-8")
                    self._send(body, "application/json")
                elif path == "/voices":
                    api = srv.voice_api
                    payload = {"voices": list(getattr(api, "voice_list", []) or []),
                               "current": getattr(api, "voice_current", 0)} \
                        if api is not None else {"voices": [], "current": 0}
                    self._send(json.dumps(payload).encode("utf-8"),
                               "application/json")
                elif path == "/report":
                    try:
                        # Preferred: the full-fidelity report from the live
                        # validator, not the truncated state snapshot.
                        if srv.report_source is not None:
                            page = srv.report_source()
                        else:
                            with srv.lock:
                                page = srv.build_report_html()
                    except Exception as exc:
                        page = f"<pre>report failed: {exc}</pre>"
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with srv.lock:
                                f = srv.jpeg
                            if f is not None:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    b"Content-Length: " +
                                    str(len(f)).encode() + b"\r\n\r\n" +
                                    f + b"\r\n")
                            time.sleep(0.07)
                    except Exception:
                        pass
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def _find_ui():
        """Look for ui/ next to the repo root, then in the working directory."""
        here = os.path.dirname(os.path.abspath(__file__))       # src/har
        root = os.path.abspath(os.path.join(here, "..", ".."))  # repo root
        for cand in (os.path.join(root, "ui"), os.path.join(os.getcwd(), "ui")):
            if os.path.isdir(cand):
                return cand
        return os.path.join(root, "ui")

    def read_ui(self, name):
        """Read a UI file fresh each request so edits show on refresh."""
        safe = os.path.basename(name)
        path = os.path.join(self.ui_dir, safe)
        if not os.path.isfile(path):
            return None, None
        with open(path, "rb") as f:
            body = f.read()
        ctype = mimetypes.guess_type(safe)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return body, ctype

    def push_frame(self, frame):
        self.count += 1
        if self.count % self.every:
            return
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(h * self.width / w)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()

    def build_report_html(self):
        """
        A one-page mission report, printable straight to PDF from the browser.

        No PDF library is used on purpose: reportlab/weasyprint are extra
        dependencies that can fail to install on a machine you have not
        tested them on, and this is being built the day before a submission.
        The browser's own Print dialog (Ctrl+P, Save as PDF) does the actual
        PDF conversion, so this only has to be a clean HTML document.
        """
        s = self.state
        rows = "".join(
            f"<tr><td>{e.get('t', 0):.1f}s</td><td>{e.get('event','')}</td>"
            f"<td>{e.get('step','') or ''}</td></tr>"
            for e in s.get("log", []))
        alerts_html = "".join(
            f"<li><b>{a.get('t',0):.1f}s</b> - {a.get('message','')}"
            + (f" <span class='exp'>(expected: {a.get('expected')}, "
               f"detected: {a.get('detected')})</span>"
               if a.get("expected") else "")
            + "</li>"
            for a in s.get("alerts", [])) or "<li>No alerts this session.</li>"
        steps_html = "".join(
            f"<tr><td>{st['id']}</td><td>{st['name'].replace('_',' ')}</td>"
            f"<td class='st-{st['state']}'>{st['state'].upper()}</td></tr>"
            for st in s.get("steps", []))
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Mission Report - {s.get('session','')}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;color:#111;max-width:780px;margin:32px auto;padding:0 20px}}
h1{{font-size:20px;margin-bottom:2px}}
.sub{{color:#666;font-size:12px;margin-bottom:24px}}
h2{{font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:#333;
    border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:28px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px}}
td,th{{padding:5px 8px;border-bottom:1px solid #e5e5e5;text-align:left}}
.st-done{{color:#1a7a4c}} .st-current{{color:#0a6}} .st-skipped{{color:#c0392b}}
.st-next{{color:#a06a00}} .st-pending{{color:#999}}
ul{{font-size:12.5px;padding-left:18px}}
.exp{{color:#666}}
.meta td{{border:0;padding:2px 8px}}
@media print {{ body{{margin:0}} }}
.noprint{{margin-bottom:20px}}
button{{padding:8px 14px;font-size:13px;cursor:pointer}}
</style></head><body>
<div class="noprint"><button onclick="window.print()">Print / Save as PDF</button></div>
<h1>On-board HAR Assistant &mdash; Mission Report</h1>
<div class="sub">{s.get('protocol','')}</div>
<table class="meta">
<tr><td><b>Session</b></td><td>{s.get('session','')}</td></tr>
<tr><td><b>Total events</b></td><td>{s.get('n_events',0)}</td></tr>
<tr><td><b>Recording</b></td><td>{'Yes' if s.get('recording') else 'No'}</td></tr>
</table>
<h2>Protocol status</h2>
<table><tr><th>#</th><th>Step</th><th>State</th></tr>{steps_html}</table>
<h2>Alerts</h2>
<ul>{alerts_html}</ul>
<h2>Event log</h2>
<table><tr><th>Time</th><th>Event</th><th>Step</th></tr>{rows}</table>
</body></html>"""

    def push_state(self, **kw):
        with self.lock:
            self.state.update(kw)

    def stop(self):
        try:
            self.httpd.shutdown()
        except Exception:
            pass