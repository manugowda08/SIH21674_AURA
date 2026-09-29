"""
SIH26174 - Mission report.

One builder, two callers: the dashboard's /report page (live, on demand) and
the automatic save that fires when the protocol completes.

PDF conversion uses the browser already on the machine. Edge ships with
Windows and Chrome is near-universal, and both can print a page to PDF
headlessly. That keeps the system dependency-free and offline: no reportlab,
no wkhtmltopdf, nothing to install the day before a demo. If neither browser
is found the HTML report is still saved and can be printed by hand, so a
missing browser degrades the feature rather than breaking it.
"""

import html as _html
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _e(x):
    return _html.escape(str(x if x is not None else ""))


def _pretty(name):
    return _e(str(name or "").replace("_", " "))


def build_report_html(protocol_name, session, steps, events, alerts,
                      recording=False, complete=False):
    """
    steps   - [{"id","name","state"}]  as returned by Validator.progress()
    events  - the full event list from Validator.events (not a truncated tail)
    alerts  - [{"message","expected","detected","t"}]
    """
    done_at = {}
    for ev in events:
        if ev.get("event") == "STEP_COMPLETED":
            done_at[ev["step"]] = ev["t"]

    n_done = sum(1 for s in steps if s["state"] == "done")
    duration = max((ev["t"] for ev in events), default=0.0)

    if complete and not alerts:
        outcome, tone = "COMPLETE - all steps verified, no violations", "ok"
    elif complete:
        outcome, tone = f"COMPLETE - {len(alerts)} violation(s) flagged", "warn"
    else:
        outcome, tone = (f"INCOMPLETE - {n_done} of {len(steps)} steps verified",
                         "bad")

    step_rows = "".join(
        f"<tr><td>{_e(s['id'])}</td><td>{_pretty(s['name'])}</td>"
        f"<td class='st-{_e(s['state'])}'>{_e(s['state']).upper()}</td>"
        f"<td>{('%.1fs' % done_at[s['id']]) if s['id'] in done_at else '-'}</td></tr>"
        for s in steps)

    alert_items = "".join(
        f"<li><b>{a.get('t', 0):.1f}s</b> &mdash; {_e(a.get('message', ''))}"
        + (f"<br><span class='exp'>expected: {_pretty(a.get('expected'))} "
           f"&rarr; detected: {_pretty(a.get('detected') or 'idle')}</span>"
           if a.get("expected") else "")
        + "</li>"
        for a in alerts) or "<li>No violations this session.</li>"

    log_rows = "".join(
        f"<tr><td>{ev.get('t', 0):.1f}s</td><td>{_e(ev.get('event'))}</td>"
        f"<td>{_pretty(ev.get('step_name') or '')}</td>"
        f"<td>{_e(ev.get('message', ''))}</td></tr>"
        for ev in events)

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Mission Report - {_e(session)}</title>
<style>
@page {{ size: A4; margin: 16mm; }}
body{{font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#111;
     max-width:780px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px;margin:0 0 2px}}
.sub{{color:#666;font-size:12px;margin-bottom:18px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#333;
   border-bottom:1px solid #ccc;padding-bottom:4px;margin:24px 0 8px}}
.outcome{{padding:10px 14px;border-radius:6px;font-weight:600;margin:14px 0}}
.outcome.ok{{background:#e6f4ec;color:#14683f}}
.outcome.warn{{background:#fdf1dc;color:#8a5a00}}
.outcome.bad{{background:#fbe6e4;color:#a02a20}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
td,th{{padding:4px 8px;border-bottom:1px solid #e6e6e6;text-align:left;vertical-align:top}}
th{{color:#555;font-weight:600}}
.st-done{{color:#1a7a4c}} .st-current{{color:#0a6}} .st-skipped{{color:#c0392b}}
.st-next{{color:#a06a00}} .st-pending{{color:#999}}
ul{{padding-left:18px;font-size:12.5px}} li{{margin-bottom:6px}}
.exp{{color:#666;font-size:11.5px}}
.meta td{{border:0;padding:1px 8px 1px 0}}
.foot{{margin-top:28px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:8px}}
.noprint button{{padding:7px 13px;font-size:13px;cursor:pointer}}
@media print {{ .noprint{{display:none}} body{{margin:0;max-width:none}} }}
</style></head><body>
<div class="noprint"><button onclick="window.print()">Print / Save as PDF</button></div>
<h1>On-board HAR Assistant &mdash; Mission Report</h1>
<div class="sub">{_e(protocol_name)}</div>
<table class="meta">
<tr><td><b>Session</b></td><td>{_e(session)}</td></tr>
<tr><td><b>Duration</b></td><td>{duration:.1f} s</td></tr>
<tr><td><b>Events logged</b></td><td>{len(events)}</td></tr>
<tr><td><b>Video recorded locally</b></td><td>{'Yes' if recording else 'No'}</td></tr>
<tr><td><b>Generated</b></td><td>{generated} (offline, on-board)</td></tr>
</table>
<div class="outcome {tone}">{_e(outcome)}</div>
<h2>Step outcomes</h2>
<table><tr><th>#</th><th>Step</th><th>Status</th><th>Verified at</th></tr>{step_rows}</table>
<h2>Violations</h2><ul>{alert_items}</ul>
<h2>Event log</h2>
<table><tr><th>Time</th><th>Event</th><th>Step</th><th>Detail</th></tr>{log_rows}</table>
<div class="foot">Generated locally with no network connection. This structured
summary and the JSONL log are the downlink artifacts; the raw video stays on board.</div>
</body></html>"""


# --------------------------------------------------------------- PDF export

def find_browser():
    """Locate Edge or Chrome. Returns a path, or None."""
    env = os.environ
    bases = [env.get("ProgramFiles"), env.get("ProgramFiles(x86)"),
             env.get("LOCALAPPDATA")]
    rel = [
        ("Google", "Chrome", "Application", "chrome.exe"),
        ("Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for base in bases:
        if not base:
            continue
        for parts in rel:
            p = os.path.join(base, *parts)
            if os.path.isfile(p):
                return p
    for name in ("chrome", "msedge", "google-chrome", "chromium",
                 "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    return None


def pdf_command(browser, html_path, pdf_path, profile_dir):
    """
    Build the headless print command.

    --user-data-dir matters: if Chrome or Edge is already open (it will be,
    the dashboard is in it) a headless launch on the default profile hands the
    request to the running instance and exits without writing anything. A
    throwaway profile forces a separate process.
    """
    return [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf_path}",
        Path(html_path).resolve().as_uri(),
    ]


def html_to_pdf(html_path, pdf_path, timeout=45):
    browser = find_browser()
    if not browser:
        return False, "no Edge or Chrome found"
    profile = tempfile.mkdtemp(prefix="har_pdf_")
    try:
        cmd = pdf_command(browser, html_path, pdf_path, profile)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
        ok = os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 500
        return ok, (os.path.basename(browser) if ok else "browser ran but wrote no PDF")
    except Exception as exc:
        return False, str(exc)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def save_report(html, outdir, session, tag=""):
    """
    Write the HTML, then try for a PDF beside it.
    Returns {"html": path, "pdf": path-or-None, "note": str}.
    """
    os.makedirs(outdir, exist_ok=True)
    stem = f"report_{session}{('_' + tag) if tag else ''}"
    html_path = os.path.join(outdir, stem + ".html")
    pdf_path = os.path.join(outdir, stem + ".pdf")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    ok, note = html_to_pdf(html_path, pdf_path)
    return {"html": html_path, "pdf": pdf_path if ok else None, "note": note}