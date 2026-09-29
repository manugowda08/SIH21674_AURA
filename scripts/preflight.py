#!/usr/bin/env python3
"""
SIH26174 - Preflight camera check (pose + desk calibration).

Run this BEFORE recording any takes.

WHY POSE AND NOT HAND LANDMARKS
  The deployment condition is a gloved operator. Consumer hand landmark models
  are trained on bare skin and fail completely on nitrile gloves - measured 0%
  on our own bench. Body pose tracks the whole arm silhouette, so a covered
  hand is a small perturbation rather than total loss of signal.

WHY DESK CALIBRATION
  Zone boundaries used to be fixed fractions of the image, which never line up
  with a desk seen in perspective. Instead you click the four corners of the
  working area once. That gives a homography mapping image pixels to real desk
  centimetres, so:
    - drawn zone lines follow the desk's perspective and match your tape
    - wrist positions become physical coordinates, not pixel coordinates
    - the same calibration is reused by feature extraction later
  Calibration is saved and reloaded, so you only do it when the camera moves.

Keys
  c     (re)calibrate the desk corners
  s     save a reference frame
  g     cycle guides (zones -> thirds -> off)
  q     quit

Usage
  python preflight.py --camera 0 --desk-width 90 --desk-depth 60
"""

import argparse
import json
from collections import deque
import math
import os
import time
import urllib.request

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_NAMES = ["LEFT STOWAGE", "CENTRE BENCH", "RIGHT STOWAGE"]
ZONE_SHORT = ["LEFT", "CENTRE", "RIGHT"]
CORNER_PROMPTS = [
    "FRONT-LEFT corner of the working area",
    "FRONT-RIGHT corner",
    "BACK-RIGHT corner",
    "BACK-LEFT corner",
]

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task"
)

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
# BlazePose also reports coarse hand points. These are what let us estimate a
# grip point rather than relying on the wrist joint, which sits 10-18 cm behind
# wherever the hand is actually touching something.
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

# (wrist, index_knuckle, pinky_knuckle, thumb) per side
HAND_CHAIN = {
    "L": (L_WRIST, L_INDEX, L_PINKY, L_THUMB),
    "R": (R_WRIST, R_INDEX, R_PINKY, R_THUMB),
}

UPPER_BODY_LINKS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
]
VIS_THRESHOLD = 0.5
GRIP_EXTEND = 0.35


def grip_point(pts, vis, side, extend=GRIP_EXTEND):
    """
    Estimate where the hand is actually holding something.

    The wrist joint is a poor proxy for contact: it trails the palm by most of
    a hand length, so a wrist 15 cm short of a vial may still be gripping it.
    Instead take the knuckle line (index and pinky MCPs) and push a little
    further along the wrist-to-knuckle direction, which lands near the fingers
    where a grasped object sits.

    Returns (x_px, y_px) or None if the hand points are not visible.
    """
    w_i, idx_i, pky_i, _ = HAND_CHAIN[side]
    if vis[idx_i] <= VIS_THRESHOLD or vis[pky_i] <= VIS_THRESHOLD:
        return None
    kx = (pts[idx_i][0] + pts[pky_i][0]) / 2.0
    ky = (pts[idx_i][1] + pts[pky_i][1]) / 2.0
    if vis[w_i] > VIS_THRESHOLD:
        wx, wy = pts[w_i]
        kx += (kx - wx) * extend
        ky += (ky - wy) * extend
    return int(round(kx)), int(round(ky))

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# ----------------------------------------------------------------------------
# Desk calibration
# ----------------------------------------------------------------------------

class DeskCalibration:
    """Maps between image pixels and desk centimetres via a homography."""

    def __init__(self, img_corners, width_cm, depth_cm, tolerance_cm=12.0):
        corners, self.reordered = self._canonical_order(
            [tuple(map(float, p)) for p in img_corners])

        self.img_corners = corners
        self.width_cm = float(width_cm)
        self.depth_cm = float(depth_cm)
        self.tolerance_cm = float(tolerance_cm)

        src = np.float32(self.img_corners)
        # Desk frame: x rightwards 0..width, y away from camera 0..depth.
        dst = np.float32([
            [0.0, 0.0],
            [self.width_cm, 0.0],
            [self.width_cm, self.depth_cm],
            [0.0, self.depth_cm],
        ])
        self.img2desk = cv2.getPerspectiveTransform(src, dst)
        self.desk2img = cv2.getPerspectiveTransform(dst, src)

    @staticmethod
    def _canonical_order(pts):
        """
        Sort four clicked corners into front-left, front-right, back-right,
        back-left regardless of the order they were clicked in.

        Click order is a needless source of error: clicking from the performer's
        side mirrors the desk and swaps LEFT with RIGHT, while starting on a
        side edge rotates it 90 degrees and makes the zone dividers run the
        wrong way. Both are silent failures. Deriving the order from geometry
        removes the whole class of mistake.

        For a camera looking down at a bench, image y grows toward the viewer,
        so the two corners below the centroid are the front pair.
        """
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        front = sorted([p for p in pts if p[1] >= cy], key=lambda p: p[0])
        back = sorted([p for p in pts if p[1] < cy], key=lambda p: p[0])

        # Degenerate split (nearly edge-on view): fall back to angular sort.
        if len(front) != 2 or len(back) != 2:
            ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            return ordered, True

        ordered = [front[0], front[1], back[1], back[0]]
        return ordered, ordered != list(pts)

    def to_desk(self, x_px, y_px):
        pt = np.float32([[[x_px, y_px]]])
        out = cv2.perspectiveTransform(pt, self.img2desk)
        return float(out[0][0][0]), float(out[0][0][1])

    def to_image(self, x_cm, y_cm):
        pt = np.float32([[[x_cm, y_cm]]])
        out = cv2.perspectiveTransform(pt, self.desk2img)
        return int(round(out[0][0][0])), int(round(out[0][0][1]))

    def zone_of(self, x_px, y_px):
        """
        Zone index 0/1/2, or None if well off the working area.

        A tolerance band is allowed because the pose model reports the WRIST
        JOINT, which sits a few centimetres behind the point where the hand
        actually contacts the bench. Without this, hands resting at the back
        edge are wrongly reported as off-desk.
        """
        x_cm, y_cm = self.to_desk(x_px, y_px)
        t = self.tolerance_cm
        if not (-t <= x_cm <= self.width_cm + t and -t <= y_cm <= self.depth_cm + t):
            return None
        clamped = min(max(x_cm, 0.0), self.width_cm - 1e-6)
        return min(int(clamped / (self.width_cm / 3.0)), 2)

    def draw(self, frame):
        """Outline the desk and draw zone dividers that follow its perspective."""
        outline = np.int32([self.to_image(x, y) for x, y in (
            (0, 0), (self.width_cm, 0), (self.width_cm, self.depth_cm), (0, self.depth_cm)
        )])
        cv2.polylines(frame, [outline], True, (120, 190, 120), 2)

        for i in (1, 2):
            x_cm = self.width_cm * i / 3.0
            pts = np.int32([self.to_image(x_cm, y) for y in
                            np.linspace(0, self.depth_cm, 12)])
            cv2.polylines(frame, [pts], False, (200, 200, 90), 2)

        for i, name in enumerate(ZONE_NAMES):
            cx = self.width_cm * (i + 0.5) / 3.0
            px, py = self.to_image(cx, self.depth_cm * 0.10)
            size = cv2.getTextSize(name, FONT, 0.45, 1)[0]
            cv2.putText(frame, name, (px - size[0] // 2, py),
                        FONT, 0.45, (200, 200, 90), 1)

    def to_dict(self):
        return {
            "img_corners": self.img_corners,
            "width_cm": self.width_cm,
            "depth_cm": self.depth_cm,
            "tolerance_cm": self.tolerance_cm,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["img_corners"], d["width_cm"], d["depth_cm"],
                   d.get("tolerance_cm", 12.0))


def calibrate(cap, width_cm, depth_cm, tolerance_cm=12.0):
    """Interactive four-click corner picker. Returns a DeskCalibration."""
    clicks = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    win = "calibrate desk"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("\n--- Desk calibration ---")
    print("Click the four corners of the WORKING AREA. Order does not matter -")
    print("they are sorted automatically from their positions in the frame.")
    print("u = undo last point,  Enter = accept,  Esc = cancel\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for i, p in enumerate(clicks):
            cv2.circle(frame, p, 7, (60, 200, 240), 2)
            cv2.putText(frame, str(i + 1), (p[0] + 10, p[1] - 8),
                        FONT, 0.6, (60, 200, 240), 2)
        if len(clicks) > 1:
            cv2.polylines(frame, [np.int32(clicks)], len(clicks) == 4,
                          (120, 190, 120), 2)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
        if len(clicks) < 4:
            msg = f"Click corner {len(clicks) + 1} of 4  (any order)"
            cv2.putText(frame, msg, (12, 24), FONT, 0.6, (60, 200, 240), 2)
            cv2.putText(frame, "Click all four corners of the working area."
                        "   u = undo   Esc = cancel", (12, 48),
                        FONT, 0.45, (180, 180, 180), 1)
        else:
            cv2.putText(frame, "All 4 set. Enter = accept, u = undo, Esc = cancel",
                        (12, 24), FONT, 0.6, (90, 210, 120), 2)
            cv2.putText(frame, f"Working area {width_cm:.0f} x {depth_cm:.0f} cm",
                        (12, 48), FONT, 0.45, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            return None
        if key == ord("u") and clicks:
            clicks.pop()
        if key in (13, 10) and len(clicks) == 4:
            cv2.destroyWindow(win)
            return DeskCalibration(clicks, width_cm, depth_cm, tolerance_cm)


# ----------------------------------------------------------------------------
# Pose
# ----------------------------------------------------------------------------

class PoseDetector:
    def __init__(self):
        self._ensure_model()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()

    @staticmethod
    def _ensure_model():
        if os.path.exists(POSE_MODEL_PATH):
            return
        print("Downloading pose model (about 3 MB, one-time only)...")
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
            print(f"Saved to {POSE_MODEL_PATH}")
        except Exception as exc:
            raise SystemExit(
                f"Could not download the pose model: {exc}\n"
                f"Download manually:\n  {POSE_MODEL_URL}\n"
                f"and save as:\n  {POSE_MODEL_PATH}"
            )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None, None
        lm = result.pose_landmarks[0]
        return ([(int(p.x * w), int(p.y * h)) for p in lm],
                [p.visibility for p in lm])

    @staticmethod
    def draw(frame, pts, vis):
        for a, b in UPPER_BODY_LINKS:
            if vis[a] > VIS_THRESHOLD and vis[b] > VIS_THRESHOLD:
                cv2.line(frame, pts[a], pts[b], (90, 210, 120), 2)
        for i in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW):
            if vis[i] > VIS_THRESHOLD:
                cv2.circle(frame, pts[i], 4, (200, 200, 90), -1)
        for i in (L_WRIST, R_WRIST):
            if vis[i] > VIS_THRESHOLD:
                cv2.circle(frame, pts[i], 7, (60, 200, 240), 2)
        # Grip point: magenta cross. This, not the wrist, is what contact
        # detection should use.
        for side in ("L", "R"):
            g = grip_point(pts, vis, side)
            if g is None:
                continue
            w_i = HAND_CHAIN[side][0]
            if vis[w_i] > VIS_THRESHOLD:
                cv2.line(frame, pts[w_i], g, (200, 80, 200), 2)
            cv2.drawMarker(frame, g, (220, 90, 220), cv2.MARKER_CROSS, 18, 2)
            cv2.circle(frame, g, 11, (220, 90, 220), 1)


def torso_axis(pts, vis):
    """Shoulder-line angle in degrees. The canonicalisation reference frame."""
    if vis[L_SHOULDER] <= VIS_THRESHOLD or vis[R_SHOULDER] <= VIS_THRESHOLD:
        return None
    (lx, ly), (rx, ry) = pts[L_SHOULDER], pts[R_SHOULDER]
    return math.degrees(math.atan2(ry - ly, rx - lx))


# ----------------------------------------------------------------------------

def exposure_report(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(grey.mean())
    clipped = float((grey > 250).sum()) / grey.size * 100.0
    if mean < 65:
        return mean, clipped, "TOO DARK - add light", (60, 60, 235)
    if mean > 195:
        return mean, clipped, "TOO BRIGHT - reduce light", (60, 60, 235)
    if clipped > 4.0:
        return mean, clipped, "BLOWN HIGHLIGHTS - move the lamp", (60, 170, 235)
    return mean, clipped, "EXPOSURE OK", (90, 210, 120)


def draw_thirds(frame):
    h, w = frame.shape[:2]
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h), (110, 110, 110), 1)
    for y in (h // 3, 2 * h // 3):
        cv2.line(frame, (0, y), (w, y), (110, 110, 110), 1)


def open_camera(args):
    if args.camera.isdigit():
        cap = cv2.VideoCapture(int(args.camera))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    else:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Camera preflight (pose + desk calibration).")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--desk-width", type=float, default=90.0,
                    help="Working area width in cm, left edge to right edge")
    ap.add_argument("--desk-depth", type=float, default=60.0,
                    help="Working area depth in cm, front edge to back edge")
    ap.add_argument("--tolerance", type=float, default=12.0,
                    help="cm of slack outside the desk still counted as on-bench. "
                         "The pose model reports the wrist joint, which sits behind "
                         "the hand's contact point.")
    ap.add_argument("--recalibrate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    calib_path = os.path.join(args.outdir, "desk_calibration.json")

    cap = open_camera(args)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {args.camera}\n"
                         "Close Zoom/Teams/Camera app, then try --camera 1 or 2.")
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Camera opened but returned no frame. Privacy shutter closed?")

    real_h, real_w = frame.shape[:2]
    print(f"\nRequested {args.width}x{args.height}, camera delivering {real_w}x{real_h}")

    calib = None
    if os.path.exists(calib_path) and not args.recalibrate:
        try:
            with open(calib_path) as f:
                calib = DeskCalibration.from_dict(json.load(f))
            print(f"Loaded desk calibration from {calib_path}")

            # The corner clicks stay valid if only the measured size changed,
            # so rescale in place rather than making the user re-click.
            if (abs(calib.width_cm - args.desk_width) > 0.5 or
                    abs(calib.depth_cm - args.desk_depth) > 0.5):
                print(f"  Stored size {calib.width_cm:.0f} x {calib.depth_cm:.0f} cm "
                      f"differs from --desk-width/--desk-depth "
                      f"({args.desk_width:.0f} x {args.desk_depth:.0f} cm).")
                print("  Keeping the corner clicks, updating the measurements.")
                calib = DeskCalibration(calib.img_corners, args.desk_width,
                                        args.desk_depth, args.tolerance)
                with open(calib_path, "w") as f:
                    json.dump(calib.to_dict(), f, indent=2)
            print(f"  Working area {calib.width_cm:.0f} x {calib.depth_cm:.0f} cm")
        except Exception as exc:
            print(f"Calibration file unreadable ({exc}), recalibrating.")
    if calib is None:
        calib = calibrate(cap, args.desk_width, args.desk_depth, args.tolerance)
        if calib is not None:
            with open(calib_path, "w") as f:
                json.dump(calib.to_dict(), f, indent=2)
            print(f"Saved calibration to {calib_path}")
            if calib.reordered:
                print("  NOTE: corners were reordered automatically so that LEFT is the")
                print("  camera's left and the zone dividers run front-to-back.")
        else:
            print("Calibration skipped. Zone overlay disabled until you press c.")

    detector = None
    if HAS_MEDIAPIPE:
        try:
            detector = PoseDetector()
            print("Pose model loaded. Gloves and sleeves are fine.")
        except SystemExit:
            raise
        except Exception as exc:
            print(f"Pose model failed to load ({exc}). No overlay.")
    else:
        print("MediaPipe not installed: pip install mediapipe")

    print("\nc = recalibrate   s = save frame   g = cycle guides   q = quit\n")

    guide_mode = 0
    frames = both_wrist_frames = torso_frames = person_frames = on_desk_frames = 0
    # Cumulative counters include frames from before the performer sat down,
    # which makes them useless for spotting WHICH step loses tracking. Keep a
    # short rolling window for the live readout; cumulative goes in the summary.
    ROLL = 120  # about 6 seconds at 20 fps
    roll_wrist = deque(maxlen=ROLL)
    roll_torso = deque(maxlen=ROLL)
    worst_wrist = worst_torso = 100.0
    fps_window = []
    last = time.perf_counter()
    mean = clipped = measured_fps = 0.0
    verdict, verdict_colour = "", (200, 200, 200)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        now = time.perf_counter()
        fps_window.append(now - last)
        last = now
        if len(fps_window) > 30:
            fps_window.pop(0)
        measured_fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0.0

        clean = frame.copy()
        mean, clipped, verdict, verdict_colour = exposure_report(frame)
        frames += 1

        wrist_note = "no person detected"
        n_wrists = 0
        angle = None
        saw_both_wrists = False
        saw_torso = False

        if detector is not None:
            pts, vis = detector.process(frame)
            if pts is not None:
                person_frames += 1
                detector.draw(frame, pts, vis)
                left_ok = vis[L_WRIST] > VIS_THRESHOLD
                right_ok = vis[R_WRIST] > VIS_THRESHOLD
                n_wrists = int(left_ok) + int(right_ok)
                if n_wrists == 2:
                    both_wrist_frames += 1
                    saw_both_wrists = True

                angle = torso_axis(pts, vis)
                if angle is not None:
                    torso_frames += 1
                    saw_torso = True
                    cv2.line(frame, pts[L_SHOULDER], pts[R_SHOULDER], (240, 160, 60), 3)

                parts = []
                any_on_desk = False
                for ok_flag, side in ((left_ok, "L"), (right_ok, "R")):
                    if not ok_flag:
                        continue
                    g = grip_point(pts, vis, side)
                    if g is None:
                        parts.append(f"{side}:no-grip-pt")
                        continue
                    if calib is None:
                        parts.append(f"{side}:uncalibrated")
                        continue
                    z = calib.zone_of(*g)
                    if z is None:
                        parts.append(f"{side}:off-desk")
                    else:
                        any_on_desk = True
                        xc, yc = calib.to_desk(*g)
                        parts.append(f"{side}:{ZONE_SHORT[z]} ({xc:.0f},{yc:.0f})cm")
                if any_on_desk:
                    on_desk_frames += 1
                wrist_note = "   ".join(parts) if parts else "hands not visible"

        roll_wrist.append(1 if saw_both_wrists else 0)
        roll_torso.append(1 if saw_torso else 0)
        roll_w_pct = 100.0 * sum(roll_wrist) / len(roll_wrist)
        roll_t_pct = 100.0 * sum(roll_torso) / len(roll_torso)
        if len(roll_wrist) == ROLL:
            worst_wrist = min(worst_wrist, roll_w_pct)
            worst_torso = min(worst_torso, roll_t_pct)

        if guide_mode == 0 and calib is not None:
            calib.draw(frame)
        elif guide_mode == 1:
            draw_thirds(frame)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 124), (0, 0, 0), -1)
        cv2.putText(frame, f"{real_w}x{real_h}  {measured_fps:4.1f} fps",
                    (12, 30), FONT, 0.7, (220, 220, 220), 2)
        cv2.putText(frame, f"brightness {mean:5.1f}   clipped {clipped:4.1f}%   {verdict}",
                    (12, 56), FONT, 0.5, verdict_colour, 1)
        if detector is not None:
            wc = (90, 210, 120) if roll_w_pct > 80 else (60, 170, 235)
            tc = (90, 210, 120) if roll_t_pct > 90 else (60, 170, 235)
            cv2.putText(frame, f"wrists {roll_w_pct:4.0f}% (6s)  worst {worst_wrist:3.0f}%",
                        (12, 78), FONT, 0.5, wc, 1)
            atxt = f"{angle:+.0f}deg" if angle is not None else "--"
            cv2.putText(frame, f"torso {roll_t_pct:4.0f}% (6s)  worst {worst_torso:3.0f}%  tilt {atxt}",
                        (330, 78), FONT, 0.5, tc, 1)
            cv2.putText(frame, wrist_note, (12, 104), FONT, 0.48, (200, 200, 90), 1)

        cv2.imshow("preflight", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("g"):
            guide_mode = (guide_mode + 1) % 3
        if key == ord("c"):
            new_calib = calibrate(cap, args.desk_width, args.desk_depth, args.tolerance)
            if new_calib is not None:
                calib = new_calib
                with open(calib_path, "w") as f:
                    json.dump(calib.to_dict(), f, indent=2)
                print(f"Saved calibration to {calib_path}")
        if key == ord("s"):
            path = os.path.join(args.outdir, "setup_reference.jpg")
            cv2.imwrite(path, clean)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Preflight summary ---")
    print(f"  Resolution        {real_w}x{real_h}")
    print(f"  Measured fps      {measured_fps:.1f}")
    print(f"  Brightness        {mean:.1f}  ({verdict})")
    print(f"  Desk calibrated   {'yes' if calib else 'NO'}")
    if detector is not None:
        wr = 100.0 * both_wrist_frames / max(frames, 1)
        tr = 100.0 * torso_frames / max(frames, 1)
        print(f"  Person detected   {100.0 * person_frames / max(frames,1):.0f}%")
        print(f"  Both wrists       {wr:.0f}% overall, worst 6s window {worst_wrist:.0f}%   (gate 80%)")
        print(f"  Torso axis        {tr:.0f}% overall, worst 6s window {worst_torso:.0f}%   (gate 90%)")
        print("  (The worst window is the number that matters. Overall includes")
        print("   frames before the performer was in position.)")

        problems = []
        if worst_wrist < 80:
            problems.append("wrists")
        if worst_torso < 90:
            problems.append("torso")
        if calib is None:
            problems.append("calibration")

        if not problems:
            print("\n  Framing is good. Start recording.")
        else:
            print(f"\n  FAILED on: {', '.join(problems)}. Do not start recording.")
            if "wrists" in problems:
                print("\n  Wrists below 80%:")
                print("    1. Performer sits back, hands away from the torso")
                print("    2. Raise the camera or slide it back")
                print("    3. More light on the bench")
            if "torso" in problems:
                print("\n  Torso below 90% - canonicalisation has no reference frame:")
                print("    1. Both shoulders must stay in frame through ALL eight steps")
                print("    2. Raise the camera 5 cm or slide it back 5 cm")
            if "calibration" in problems:
                print("\n  Run again and press c to click the four desk corners.")


if __name__ == "__main__":
    main()