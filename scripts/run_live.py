#!/usr/bin/env python3
"""
SIH26174 - Live on-board assistant.

Everything the brief asks for, in one offline process:
  continuous local video processing      - camera -> pose -> features -> model
  next-step suggestion                   - on screen and spoken
  voice alerts on skip / out-of-order    - offline TTS, no network
  timestamped structured log             - JSONL, the downlink artifact
  local recording AND stream to an IP    - MP4 on disk, MJPEG over HTTP
  monitoring GUI                         - the window you are looking at

Keys
  r   reset the protocol run
  m   mute / unmute voice
  q   quit

Usage
  python scripts/run_live.py --camera 0
  python scripts/run_live.py --replay data/raw/take_001__A__s01__nominal.mp4
"""

import argparse
import os
import queue
import sys
import threading
import time

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from har.pipeline import (  # noqa: E402
    DeskCalibration, PoseDetector, STEP_NAMES, WINDOW,
    featurise, frame_features, model_path,
)
from har.protocol import EventLog, Protocol, Smoother, Validator  # noqa: E402
from har.webui import Dashboard  # noqa: E402
from har.report import build_report_html, save_report  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (110, 210, 120)
AMBER = (60, 170, 235)
RED = (70, 70, 235)
GREY = (150, 150, 150)
WHITE = (235, 235, 235)
CYAN = (200, 200, 90)


# --------------------------------------------------------------- voice

class Voice:
    """
    Offline TTS on a worker thread so speech never stalls the video loop.

    Alerts are not just another message. They jump ahead of routine prompts,
    are preceded by an attention tone, are spoken at full volume and slightly
    slower, and are repeated once. A warning that queues behind "next, step
    four" arrives after the operator has already made the mistake.
    """

    ALERT, ROUTINE = 0, 1

    def __init__(self, enabled=True, volume=1.0, rate=170,
                 alert_rate=150, repeat=2, tone=True):
        self.q = queue.PriorityQueue()
        self.enabled = enabled
        self.muted = False
        self.ok = False
        self.volume = volume
        self.rate = rate
        self.alert_rate = alert_rate
        self.repeat = max(1, repeat)
        self.tone = tone
        self._seq = 0
        self._lock = threading.Lock()
        # Installed voices, discovered by the worker thread once it has a
        # speaker. Selection is a request that the worker applies itself,
        # because a COM speech object may only be touched from the thread that
        # created it. Changing it from the web server's thread would fail
        # intermittently, which is the worst kind of bug to debug live.
        self.voice_list = []
        self.voice_current = 0
        self._pending_voice = None
        self.ok = True   # provisional; the worker confirms or clears it
        if enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _beep():
        """Two short rising tones. Offline, no audio file, no dependency."""
        try:
            import winsound
            winsound.Beep(784, 140)
            winsound.Beep(1047, 200)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    def _make_speaker(self):
        """
        Prefer Windows SAPI directly over pyttsx3.

        pyttsx3 stops responding after runAndWait() is called repeatedly from a
        worker thread - the first prompt speaks and everything after it is
        silent. Talking to SAPI through COM avoids that entirely. COM must be
        initialised on the thread that will use the object, so this is called
        from inside the worker, never from __init__.
        """
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            sapi.Volume = int(max(0.0, min(1.0, self.volume)) * 100)
            print("  voice: Windows SAPI")
            return ("sapi", sapi)
        except Exception:
            pass
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("volume", self.volume)
            print("  voice: pyttsx3 fallback")
            return ("pyttsx3", eng)
        except Exception as exc:
            print(f"  voice unavailable ({exc}); alerts on-screen only")
            return (None, None)

    _FEMALE = {"zira", "hazel", "susan", "heera", "eva", "linda", "catherine",
               "aria", "jenny", "sonia", "natasha", "clara", "libby", "samantha"}
    _MALE = {"david", "mark", "ravi", "george", "james", "richard", "guy",
             "ryan", "sean", "liam", "prabhat", "hemant"}

    @classmethod
    def _guess_gender(cls, label, attr=""):
        a = (attr or "").strip().lower()
        if a.startswith("f"):
            return "Female"
        if a.startswith("m"):
            return "Male"
        first = label.lower().split()[0] if label else ""
        if first in cls._FEMALE:
            return "Female"
        if first in cls._MALE:
            return "Male"
        return ""

    @staticmethod
    def _clean_label(desc):
        """"Microsoft Zira Desktop - English (United States)" -> "Zira - English (United States)"."""
        name, _, lang = desc.partition(" - ")
        name = name.replace("Microsoft ", "").replace(" Desktop", "").strip()
        return f"{name} - {lang}" if lang else name

    def _enumerate(self, kind, eng):
        found = []
        try:
            if kind == "sapi":
                vs = eng.GetVoices()
                for i in range(vs.Count):
                    tok = vs.Item(i)
                    label = self._clean_label(tok.GetDescription())
                    try:
                        attr = tok.GetAttribute("Gender")
                    except Exception:
                        attr = ""
                    found.append({"i": i, "label": label,
                                  "gender": self._guess_gender(label, attr)})
            elif kind == "pyttsx3":
                for i, v in enumerate(eng.getProperty("voices")):
                    label = self._clean_label(v.name or f"voice {i}")
                    found.append({"i": i, "label": label,
                                  "gender": self._guess_gender(label, str(v.gender or ""))})
        except Exception as exc:
            print(f"  could not list voices: {exc}")
        self.voice_list = found
        if found:
            print("  voices: " + ", ".join(v["label"].split(" - ")[0] for v in found))

    def _apply_pending(self, kind, eng):
        idx, self._pending_voice = self._pending_voice, None
        if idx is None:
            return
        try:
            if kind == "sapi":
                eng.Voice = eng.GetVoices().Item(idx)
            elif kind == "pyttsx3":
                eng.setProperty("voice", eng.getProperty("voices")[idx].id)
            self.voice_current = idx
        except Exception as exc:
            print(f"  could not switch voice: {exc}")

    def set_voice(self, idx):
        """Request a voice by index. Safe to call from any thread."""
        # bool is a subclass of int, so True would otherwise pass as index 1.
        if type(idx) is not int or not (0 <= idx < len(self.voice_list)):
            return False
        self._pending_voice = idx
        # Speak a short sample so the change is audible immediately and the
        # blocked worker wakes up to apply it.
        with self._lock:
            self._seq += 1
            self.q.put((self.ROUTINE, self._seq, "This is the selected voice.", False))
        return True

    def _speak(self, kind, eng, text, urgent):
        if kind == "sapi":
            # Rate is -10..10 on SAPI, not words per minute.
            eng.Rate = -2 if urgent else 0
            eng.Volume = int(max(0.0, min(1.0, self.volume)) * 100)
            eng.Speak(text)
        elif kind == "pyttsx3":
            eng.setProperty("rate", self.alert_rate if urgent else self.rate)
            eng.setProperty("volume", self.volume)
            eng.say(text)
            eng.runAndWait()

    def _worker(self):
        kind, eng = self._make_speaker()
        self.ok = kind is not None
        if not self.ok:
            return
        self._enumerate(kind, eng)
        while True:
            prio, _, text, urgent = self.q.get()
            if text is None:
                break
            try:
                self._apply_pending(kind, eng)
                if urgent and self.tone:
                    self._beep()
                for k in range(self.repeat if urgent else 1):
                    self._speak(kind, eng, text, urgent)
                    if k + 1 < (self.repeat if urgent else 1):
                        time.sleep(0.2)
            except Exception as exc:
                print(f"  speech failed: {exc}")

    def _drain_routine(self):
        """Throw away queued routine prompts so an alert is heard immediately."""
        kept = []
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == self.ALERT:
                    kept.append(item)
        except queue.Empty:
            pass
        for item in kept:
            self.q.put(item)

    def say(self, text, urgent=False):
        if not (self.ok and self.enabled) or self.muted:
            return
        with self._lock:
            if urgent:
                self._drain_routine()
            elif self.q.qsize() > 2:
                return
            self._seq += 1
            self.q.put((self.ALERT if urgent else self.ROUTINE,
                        self._seq, text, urgent))


# --------------------------------------------------------------- GUI

def draw_gui(frame, proto, validator, step_id, conf, alerts, fps,
             rec_on, stream_url, muted, names=None):
    names = names or STEP_NAMES
    h, w = frame.shape[:2]
    panel_w = 330
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (26, 26, 28)
    x0 = w + 16

    cv2.putText(canvas, "ON-BOARD HAR ASSISTANT", (x0, 34),
                FONT, 0.55, WHITE, 1)
    cv2.putText(canvas, proto.name[:34], (x0, 56), FONT, 0.4, GREY, 1)
    cv2.line(canvas, (x0, 70), (w + panel_w - 16, 70), (60, 60, 64), 1)

    y = 100
    cv2.putText(canvas, "CURRENT", (x0, y), FONT, 0.42, GREY, 1)
    label = (names[step_id] if step_id < len(names) else str(step_id)) \
        if step_id else "idle"
    col = GREEN if step_id else GREY
    cv2.putText(canvas, label[:22], (x0, y + 26), FONT, 0.62, col, 2)
    cv2.putText(canvas, f"confidence {conf * 100:.0f}%", (x0, y + 48),
                FONT, 0.4, GREY, 1)

    y += 78
    cv2.putText(canvas, "PROTOCOL", (x0, y), FONT, 0.42, GREY, 1)
    y += 20
    for s in validator.progress():
        state = s["state"]
        if state == "done":
            mark, c = "[x]", GREEN
        elif state == "current":
            mark, c = "[>]", GREEN
        elif state == "next":
            mark, c = "[ ]", AMBER
        else:
            mark, c = "[ ]", (105, 105, 110)
        cv2.putText(canvas, f"{mark} {s['id']}. {s['name'][:19]}",
                    (x0, y), FONT, 0.42, c, 1)
        y += 21

    y += 8
    cv2.putText(canvas, "ALERTS", (x0, y), FONT, 0.42, GREY, 1)
    y += 20
    if not alerts:
        cv2.putText(canvas, "none", (x0, y), FONT, 0.4, (105, 105, 110), 1)
        y += 18
    for a in alerts[-4:]:
        cv2.putText(canvas, a[:34], (x0, y), FONT, 0.4, RED, 1)
        y += 18

    base = h - 76
    cv2.line(canvas, (x0, base - 12), (w + panel_w - 16, base - 12),
             (60, 60, 64), 1)
    cv2.putText(canvas, f"{fps:.1f} fps   OFFLINE", (x0, base + 6),
                FONT, 0.4, GREY, 1)
    cv2.putText(canvas, f"REC {'on' if rec_on else 'off'}   "
                        f"voice {'muted' if muted else 'on'}",
                (x0, base + 24), FONT, 0.4, GREY, 1)
    cv2.putText(canvas, stream_url[:34], (x0, base + 42), FONT, 0.36, CYAN, 1)
    cv2.putText(canvas, "r reset  m mute  [ ] dwell  q quit", (x0, base + 62),
                FONT, 0.36, (105, 105, 110), 1)
    return canvas


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0")
    ap.add_argument("--replay", default=None, help="Play a recorded take instead")
    ap.add_argument("--realtime", action="store_true",
                    help="Pace replay at the source frame rate. Without this a "
                         "replay runs at whatever speed inference allows, which "
                         "looks wrong on a projector.")
    ap.add_argument("--loop", action="store_true",
                    help="Restart the replay when it ends, resetting the protocol")
    ap.add_argument("--model", default="data/runs/model.joblib")
    ap.add_argument("--calib", default="data/desk_calibration.json")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--logdir", default="data/logs")
    ap.add_argument("--recdir", default="data/sessions")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--volume", type=float, default=1.0,
                    help="TTS volume 0..1")
    ap.add_argument("--repeat", type=int, default=2,
                    help="How many times an alert is spoken")
    ap.add_argument("--no-tone", action="store_true",
                    help="Skip the attention beep before alerts")
    ap.add_argument("--reportdir", default="data/reports")
    ap.add_argument("--no-report", action="store_true",
                    help="Do not auto-generate a report on completion or exit")
    ap.add_argument("--headless", action="store_true",
                    help="No OpenCV window; use the web dashboard only")
    # Runtime overrides. The tuned values are a conservative estimate from an
    # unseen actor; the shipped model has seen everyone, so the demo can often
    # run looser. Adjustable here so rehearsal does not need a retrain.
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--dwell", type=float, default=None)
    ap.add_argument("--grace", type=float, default=None)
    args = ap.parse_args()

    for path, what in ((args.model, "model"), (args.calib, "calibration"),
                       (args.protocol, "protocol")):
        if not os.path.exists(path):
            sys.exit(f"Missing {what}: {path}")

    bundle = joblib.load(args.model)
    scaler, clf = bundle["scaler"], bundle["clf"]
    tuned_conf = bundle.get("min_conf", 0.70)
    tuned_dwell = bundle.get("dwell_seconds", 1.8)
    tuned_grace = bundle.get("skip_grace_s", 6.0)
    # A subset or merged model has its own class list. Using the pipeline's
    # 9-step default would mislabel every step on screen and in the log.
    names = bundle.get("step_names") or STEP_NAMES
    print(f"classes: {names}")
    if args.conf is not None:
        tuned_conf = args.conf
    if args.dwell is not None:
        tuned_dwell = args.dwell
    if args.grace is not None:
        tuned_grace = args.grace
    if bundle.get("merged") and args.protocol.endswith("protocol.yaml"):
        alt = args.protocol.replace("protocol.yaml", "protocol7.yaml")
        if os.path.exists(alt):
            print(f"model is merged 7-step; using {alt}")
            args.protocol = alt
    print(f"thresholds: conf={tuned_conf}, dwell={tuned_dwell}s, "
          f"skip grace={tuned_grace}s")
    calib = DeskCalibration.load(args.calib)
    proto = Protocol.load(args.protocol)
    print(f"Protocol: {proto.name}")

    if args.replay:
        cap = cv2.VideoCapture(args.replay)
        src = os.path.basename(args.replay)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        if not (1.0 < src_fps < 120.0):
            src_fps = 20.0
    else:
        cap = cv2.VideoCapture(int(args.camera))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 20)
        src = f"camera {args.camera}"
        src_fps = 20.0
    if not cap.isOpened():
        sys.exit(f"Could not open {src}")
    ok, frame = cap.read()
    if not ok:
        sys.exit("No frame from source")
    h, w = frame.shape[:2]
    print(f"Source: {src}  {w}x{h}")

    detector = PoseDetector(model_path(SCRIPT_DIR))
    voice = Voice(enabled=not args.no_voice, volume=args.volume,
                  repeat=args.repeat, tone=not args.no_tone)
    dash = Dashboard(port=args.port)
    print(f"\n  DASHBOARD:  {dash.url}")
    print(f"  (open on a phone on the same wifi too)\n")

    session = time.strftime("%Y%m%d_%H%M%S")
    log = EventLog(os.path.join(args.logdir, f"session_{session}.jsonl"), session)

    writer = None
    if args.replay and not args.no_record:
        print("  (replay mode: disabling session recording)")
        args.no_record = True
    if not args.no_record:
        os.makedirs(args.recdir, exist_ok=True)
        rec_path = os.path.join(args.recdir, f"session_{session}.mp4")
        writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, (w, h))
        print(f"Recording: {rec_path}")

    smoother = Smoother(min_conf=tuned_conf, dwell_seconds=tuned_dwell,
                        update_hz=20.0)
    validator = Validator(proto, skip_grace_s=tuned_grace)

    report_state = {"runs": 0, "saved_this_run": False, "note": ""}

    def report_html(complete=None):
        if complete is None:
            complete = all(s["state"] == "done" for s in validator.progress())
        return build_report_html(
            proto.name, session, validator.progress(), validator.events,
            alerts, recording=writer is not None, complete=complete)

    def auto_report(reason, blocking=False):
        """Save the report and try for a PDF, off the video thread."""
        if args.no_report or report_state["saved_this_run"]:
            return
        report_state["saved_this_run"] = True
        report_state["runs"] += 1
        tag = f"run{report_state['runs']}"
        html = report_html(complete=(reason == "complete") or all(
            s["state"] == "done" for s in validator.progress()))

        def work():
            res = save_report(html, args.reportdir, session, tag)
            if res["pdf"]:
                report_state["note"] = f"Report saved: {os.path.basename(res['pdf'])}"
                print(f"\n  REPORT ({reason}): {res['pdf']}")
            else:
                report_state["note"] = f"Report saved: {os.path.basename(res['html'])}"
                print(f"\n  REPORT ({reason}): {res['html']}")
                print(f"  (no PDF: {res['note']}. Open the HTML and press Ctrl+P.)")

        if blocking:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    dash.report_source = report_html
    dash.voice_api = voice
    for ev in validator.events:
        log.add(ev)
    voice.say(proto.prompt_of(1))

    history = []
    alerts = []
    fps_win = []
    last = time.perf_counter()
    step_id, conf = 0, 0.0
    alert_flash = 0.0
    last_probs = None
    n = 0
    n_pred = 0
    raw_hist = {}
    commits = 0

    print("\nRunning. r reset, m mute, q quit.\n")

    frame_period = 1.0 / src_fps
    next_due = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.replay and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                validator.reset()
                smoother = Smoother(min_conf=tuned_conf, dwell_seconds=tuned_dwell,
                                    update_hz=20.0)
                alerts.clear()
                history.clear()
                print("  replay looped, protocol reset")
                continue
            if args.replay:
                print("Replay finished.")
                break
            continue

        if args.replay and args.realtime:
            # Hold each frame to the source rate. Inference is usually faster
            # than 20 fps, so an unpaced replay sprints and the voice prompts
            # pile up out of sync with what is on screen.
            next_due += frame_period
            slack = next_due - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_due = time.perf_counter()

        now = time.perf_counter()
        fps_win.append(now - last)
        last = now
        if len(fps_win) > 20:
            fps_win.pop(0)
        fps = 1.0 / (sum(fps_win) / len(fps_win)) if fps_win else 0.0

        clean = frame.copy()
        pts, vis = detector.process(frame, timestamp_ms=int(n * 50))
        history.append(frame_features(pts, vis, calib))
        if len(history) > WINDOW * 2:
            history.pop(0)
        n += 1

        if len(history) >= WINDOW:
            x = featurise(history, len(history) - 1).reshape(1, -1)
            probs = clf.predict_proba(scaler.transform(x))[0]
            raw = int(np.argmax(probs))
            conf = float(probs[raw])
            last_probs = probs
            n_pred += 1
            raw_hist[raw] = raw_hist.get(raw, 0) + 1
            committed = smoother.update(raw, conf)
            if committed != step_id:
                commits += 1
            step_id = committed
            for ev in validator.update(committed, now=time.time() - validator.t0):
                log.add(ev)
                if ev["severity"] == "alert":
                    # Carry expected/detected through to the UI, not just the
                    # spoken sentence, so the alert panel can show WHY it
                    # fired rather than only that it did.
                    alerts.append({
                        "message": ev["message"],
                        "expected": ev.get("expected", ""),
                        "detected": ev.get("detected", ""),
                        "t": ev["t"],
                    })
                    voice.say(ev["message"], urgent=True)
                    alert_flash = time.time()
                elif ev["event"] in ("NEXT_STEP", "PROTOCOL_COMPLETE"):
                    voice.say(ev["message"])
                if ev["event"] == "PROTOCOL_COMPLETE":
                    auto_report("complete")

        if pts is not None:
            for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
                if vis[a] > 0.5 and vis[b] > 0.5:
                    cv2.line(frame, tuple(map(int, pts[a])),
                             tuple(map(int, pts[b])), GREEN, 2)

        if time.time() - alert_flash < 2.5:
            h_, w_ = frame.shape[:2]
            cv2.rectangle(frame, (2, 2), (w_ - 3, h_ - 3), (60, 60, 235), 6)

        if writer is not None:
            writer.write(clean)
        dash.push_frame(clean)

        nxt = [s for s in validator.progress() if s["state"] == "next"]
        prob_list = ([{"name": names[i] if i < len(names) else str(i),
                       "p": round(float(p), 3)}
                      for i, p in enumerate(last_probs)]
                     if last_probs is not None else [])
        dash.push_state(
            protocol=proto.name,
            current=(names[step_id] if step_id < len(names) else "") if step_id else "",
            confidence=round(conf, 3), fps=round(fps, 1),
            steps=validator.progress(),
            alerts=alerts[-6:],
            probs=prob_list,
            log=[{"t": e["t"], "event": e["event"], "step": e["step"]}
                 for e in validator.events[-40:]],
            next_prompt=(proto.prompt_of(nxt[0]["id"]) if nxt else ""),
            alerting=(time.time() - alert_flash) < 4.0,
            report_note=report_state["note"],
            recording=writer is not None, session=session,
            n_events=len(validator.events))

        canvas = draw_gui(frame, proto, validator, step_id, conf,
                          [a["message"] for a in alerts],
                          fps, writer is not None, dash.url, voice.muted,
                          names)
        if args.headless:
            key = 255
            time.sleep(0.001)
        else:
            cv2.imshow("On-board HAR Assistant", canvas)
            key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            if len(validator.events) > 1:      # something actually happened
                auto_report("reset")            # keep the aborted run's record
            report_state["saved_this_run"] = False
            validator.reset()
            smoother = Smoother(min_conf=tuned_conf, dwell_seconds=tuned_dwell,
                                update_hz=20.0)
            alerts.clear()
            for ev in validator.events:
                log.add(ev)
            voice.say("Protocol reset. " + proto.prompt_of(1))
            print("  protocol reset")
        if key == ord("m"):
            voice.muted = not voice.muted
        if key in (ord("["), ord("]")):
            tuned_dwell = max(0.4, tuned_dwell + (0.3 if key == ord("]") else -0.3))
            smoother = Smoother(min_conf=tuned_conf, dwell_seconds=tuned_dwell,
                                update_hz=20.0)
            print(f"  dwell -> {tuned_dwell:.1f}s")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if len(validator.events) > 1 and not report_state["saved_this_run"] \
            and not args.no_report:
        print("\n  Generating report (a few seconds)...")
        auto_report("exit", blocking=True)
    dash.stop()
    log.close()
    print(f"\nLog: {log.path}")
    print(f"Events: {len(validator.events)}")
    print(f"Frames read: {n}   windows scored: {n_pred}   step commits: {commits}")
    if raw_hist:
        total = sum(raw_hist.values())
        print("Raw per-window predictions (before smoothing):")
        for k in sorted(raw_hist, key=lambda z: -raw_hist[z]):
            nm = names[k] if k < len(names) else str(k)
            print(f"   {nm:<24} {raw_hist[k]:6d}  ({100.0*raw_hist[k]/total:.0f}%)")
    if n < 100:
        print("\n  Very few frames decoded. Try --no-record, and check the file plays.")
    elif commits == 0:
        print("\n  Model never committed a step. Lower the dwell: --dwell 1.0")


if __name__ == "__main__":
    main()