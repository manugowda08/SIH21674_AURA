"""
SIH26174 - Protocol definition and sequence validator.

The classifier says WHAT is happening. This says whether it was allowed to
happen, which is the part the problem statement actually asks for.

Deliberately a deterministic state machine rather than something learned:
the procedure is known and written down, so hard-coding it makes the system
auditable. A judge can read this file and verify the logic.
"""

import json
import os
import time
from collections import deque


def speakable(text):
    """
    "retrieve_cartridge" -> "retrieve cartridge".

    Step names double as identifiers elsewhere (feature arrays, the GUI's
    machine-readable state), so they stay snake_case at the source. This is
    the one place that matters: TTS should never read a variable name aloud.
    """
    return text.replace("_", " ").strip()


class Protocol:
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.steps = cfg["steps"]                  # ordered list of dicts
        self.n = len(self.steps)
        self.by_id = {s["id"]: s for s in self.steps}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        return cls(cfg)

    def name_of(self, step_id):
        """Raw name for logs, features and the GUI's machine-readable state."""
        s = self.by_id.get(step_id)
        return s["name"] if s else "idle"

    def spoken_name_of(self, step_id):
        """Human-readable form, for anything read aloud or shown as a label."""
        return speakable(self.name_of(step_id))

    def prompt_of(self, step_id):
        s = self.by_id.get(step_id)
        return s["prompt"] if s else ""


class Smoother:
    """
    Turns noisy per-frame predictions into committed steps.

    Guards, each catching a different failure:
      confidence gate - a weak prediction is treated as idle, not acted on
      majority vote   - kills single-frame flicker
      dwell time      - a step must persist before it counts, so a momentary
                        misread never reaches the state machine

    dwell is expressed in SECONDS and converted using the rate it is actually
    being fed at. Counting raw updates silently means different things offline
    (strided) and live (every frame), which makes tuned thresholds meaningless.
    """

    def __init__(self, vote_seconds=0.9, min_conf=0.70, dwell_seconds=1.8,
                 update_hz=20.0):
        self.update_hz = float(update_hz)
        vote_n = max(3, int(round(vote_seconds * self.update_hz)))
        self.dwell_n = max(2, int(round(dwell_seconds * self.update_hz)))
        self.buf = deque(maxlen=vote_n)
        self.min_conf = min_conf
        self.candidate = 0
        self.candidate_count = 0
        self.committed = 0

    def update(self, pred, conf):
        if conf < self.min_conf:
            pred = 0
        self.buf.append(pred)
        counts = {}
        for p in self.buf:
            counts[p] = counts.get(p, 0) + 1
        winner = max(counts, key=counts.get)
        # Require the winner to actually dominate, not merely lead a split vote.
        if counts[winner] < 0.6 * len(self.buf):
            winner = self.committed

        if winner == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = winner
            self.candidate_count = 1

        if self.candidate_count >= self.dwell_n:
            self.committed = self.candidate
        return self.committed


class Validator:
    """
    Tracks progress and emits events.

    Events: STEP_STARTED, STEP_COMPLETED, STEP_SKIPPED, OUT_OF_ORDER,
            NEXT_STEP, PROTOCOL_COMPLETE
    """

    def __init__(self, protocol, skip_grace_s=6.0):
        self.p = protocol
        # A suspected skip is held this long before it is announced. Detection
        # of a step can lag by a second or two, and firing instantly turns
        # every late detection into a false alarm on a perfectly correct run.
        # If the step shows up inside the window the concern is dropped in
        # silence; if it does not, the alert fires as normal.
        self.skip_grace_s = skip_grace_s
        self.reset()

    def reset(self):
        self.done = set()
        self.skipped = set()
        self.current = 0
        self.expected = 1
        self.completed_announced = False
        self.pending = {}          # step id -> time it becomes an alert
        self.events = []
        self.t0 = time.time()
        self._now = None
        self._emit("NEXT_STEP", self.expected,
                   f"Begin with step 1, {self.p.spoken_name_of(1)}")

    def _clock(self):
        return self._now if self._now is not None else (time.time() - self.t0)

    def _emit(self, kind, step_id, message, severity="info"):
        ev = {
            "t": round(self._clock(), 2),
            "event": kind,
            "step": step_id,
            "step_name": self.p.name_of(step_id),
            "message": message,
            "severity": severity,
        }
        self.events.append(ev)
        return ev

    def tick(self, now=None):
        """Advance the clock and release any skip concerns that timed out."""
        if now is not None:
            self._now = now
        new = []
        for step in sorted([s for s, due in self.pending.items()
                            if self._clock() >= due]):
            self.pending.pop(step, None)
            self.skipped.add(step)
            ev = self._emit(
                "STEP_SKIPPED", step,
                f"Warning. Step {step}, {self.p.spoken_name_of(step)}, was skipped.",
                "alert")
            # Explains WHY this fired, not just that it did. A bare beep tells
            # an operator something is wrong; this tells them what to check.
            ev["expected"] = self.p.name_of(step)
            ev["detected"] = self.p.name_of(self.current)
            new.append(ev)
        return new

    def update(self, committed_step, now=None):
        """Feed the smoothed step id. Returns any new events."""
        if now is not None:
            self._now = now
        new = self.tick()
        # A step that turns up while its concern is still pending was simply
        # detected late, not omitted. Drop the concern, and remember it was
        # late so the ordering check below does not re-flag it as out of
        # sequence - it is the same detection lag, not a second fault.
        was_late = self.pending.pop(committed_step, None) is not None
        if committed_step == self.current:
            return new
        prev, self.current = self.current, committed_step

        if prev != 0 and prev not in self.done:
            self.done.add(prev)
            new.append(self._emit("STEP_COMPLETED", prev,
                                  f"Step {prev} complete, {self.p.spoken_name_of(prev)}"))
            if prev == self.expected:
                self.expected = prev + 1

        if committed_step == 0:
            outstanding = [s["id"] for s in self.p.steps
                           if s["id"] not in self.done]
            if outstanding:
                nxt = outstanding[0]
                self.expected = nxt
                new.append(self._emit(
                    "NEXT_STEP", nxt,
                    f"Next, step {nxt}, {self.p.spoken_name_of(nxt)}"))
            elif not self.completed_announced:
                # Announce completion exactly once. Without this guard every
                # return to idle re-announces it and the voice loops forever.
                self.completed_announced = True
                new.append(self._emit("PROTOCOL_COMPLETE", 0,
                                      "Protocol complete. All steps verified."))
            return new

        new.append(self._emit("STEP_STARTED", committed_step,
                              f"Step {committed_step} started"))

        if committed_step in self.skipped:
            # Checked first: the idle branch rewinds `expected` to the next
            # outstanding step, so a recovered step can compare equal to
            # expected and would otherwise fall through unnoticed.
            self.skipped.discard(committed_step)
            new.append(self._emit(
                "STEP_RECOVERED", committed_step,
                f"Step {committed_step} now being completed.", "info"))
            self.expected = max(self.expected, committed_step + 1)

        elif committed_step > self.expected:
            missing = [s for s in range(self.expected, committed_step)
                       if s not in self.done and s not in self.skipped]
            if missing:
                due = self._clock() + self.skip_grace_s
                for mstep in missing:
                    self.pending.setdefault(mstep, due)
            self.expected = committed_step + 1

        elif committed_step < self.expected and committed_step not in self.done:
            if was_late:
                self.expected = max(self.expected, committed_step + 1)
            else:
                ev = self._emit(
                    "OUT_OF_ORDER", committed_step,
                    f"Warning. Step {committed_step} performed out of sequence.",
                    "alert")
                ev["expected"] = self.p.name_of(self.expected)
                ev["detected"] = self.p.name_of(committed_step)
                new.append(ev)
        else:
            self.expected = max(self.expected, committed_step + 1)
        return new

    def progress(self):
        return [{"id": s["id"], "name": s["name"],
                 "state": ("done" if s["id"] in self.done else
                           "current" if s["id"] == self.current else
                           "next" if s["id"] == self.expected else "pending")}
                for s in self.p.steps]


class EventLog:
    """
    Append-only JSONL. This is the downlink artifact.

    Video cannot be sent to ground on a constrained link. This file can - a
    full session is a few kilobytes, so it is what actually reaches mission
    control. Written line by line so a crash never costs more than one event.
    """

    def __init__(self, path, session):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "a", buffering=1)
        self._write({"event": "SESSION_START", "session": session,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def _write(self, obj):
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def add(self, ev):
        self._write(ev)

    def close(self):
        self._write({"event": "SESSION_END",
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.f.close()