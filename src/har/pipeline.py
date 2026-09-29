"""
SIH26174 - Shared perception and feature extraction.

CRITICAL: training and live inference both import featurise() from here.
If the two ever compute features differently the model silently breaks, and
it is the hardest class of bug to find. One function, two callers.
"""

import json
import math
import os
import time
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------- pose model

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
VIS = 0.5
GRIP_EXTEND = 0.35

STEP_NAMES = [
    "idle", "don_gloves", "sanitise_surface", "retrieve_cartridge",
    "draw_medium", "inoculate_vial", "seal_and_label",
    "stow_in_incubator", "confirm_on_interface",
]
N_CLASSES = len(STEP_NAMES)

# Window of frames fed to the classifier. 40 frames at 20 fps = 2 seconds.
# Subsampled so the feature vector stays small enough for a shallow model.
WINDOW = 60          # 3 seconds at 20 fps
SUBSAMPLE = [0, 6, 13, 20, 26, 33, 39, 46, 52, 59]
FEATS_PER_FRAME = 27
# Indices within a frame vector for the two grip points, used to derive motion.
GRIP_SLICES = [(0, 2), (7, 9)]
N_MOTION = 4 * (len(SUBSAMPLE) - 1) + 4 + 2 + 2
FEATURE_DIM = len(SUBSAMPLE) * FEATS_PER_FRAME + N_MOTION


def model_path(script_dir):
    return os.path.join(script_dir, "pose_landmarker_lite.task")


def ensure_pose_model(path):
    if os.path.exists(path):
        return
    print("Downloading pose model (about 3 MB, one-time)...")
    urllib.request.urlretrieve(POSE_MODEL_URL, path)
    print(f"  saved {path}")


class PoseDetector:
    """MediaPipe Pose, Tasks API. Robust to gloves, unlike the hand model."""

    def __init__(self, task_path, video_mode=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        ensure_pose_model(task_path)
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._t0 = time.perf_counter()
        self._frame_no = 0
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms=None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            timestamp_ms = int((time.perf_counter() - self._t0) * 1000)
        # MediaPipe VIDEO mode demands strictly increasing timestamps for the
        # whole life of the landmarker, not per clip. Batch extraction restarts
        # its counter at every take, so guard it here rather than making every
        # caller track a global clock.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        self._frame_no += 1
        res = self._landmarker.detect_for_video(image, timestamp_ms)
        if not res.pose_landmarks:
            return None, None
        lm = res.pose_landmarks[0]
        return ([(p.x * w, p.y * h) for p in lm], [p.visibility for p in lm])


# ------------------------------------------------------------- calibration

class DeskCalibration:
    def __init__(self, img_corners, width_cm, depth_cm, tolerance_cm=12.0):
        corners, self.reordered = self._canonical_order(
            [tuple(map(float, p)) for p in img_corners])
        self.img_corners = corners
        self.width_cm = float(width_cm)
        self.depth_cm = float(depth_cm)
        self.tolerance_cm = float(tolerance_cm)
        src = np.float32(corners)
        dst = np.float32([[0, 0], [self.width_cm, 0],
                          [self.width_cm, self.depth_cm], [0, self.depth_cm]])
        self.img2desk = cv2.getPerspectiveTransform(src, dst)
        self.desk2img = cv2.getPerspectiveTransform(dst, src)

    @staticmethod
    def _canonical_order(pts):
        cy = sum(p[1] for p in pts) / 4.0
        cx = sum(p[0] for p in pts) / 4.0
        front = sorted([p for p in pts if p[1] >= cy], key=lambda p: p[0])
        back = sorted([p for p in pts if p[1] < cy], key=lambda p: p[0])
        if len(front) != 2 or len(back) != 2:
            ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            return ordered, True
        ordered = [front[0], front[1], back[1], back[0]]
        return ordered, ordered != list(pts)

    def to_desk(self, x, y):
        out = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.img2desk)
        return float(out[0][0][0]), float(out[0][0][1])

    def to_image(self, x_cm, y_cm):
        out = cv2.perspectiveTransform(np.float32([[[x_cm, y_cm]]]), self.desk2img)
        return int(round(out[0][0][0])), int(round(out[0][0][1]))

    def zone_of(self, x_px, y_px):
        x, y = self.to_desk(x_px, y_px)
        t = self.tolerance_cm
        if not (-t <= x <= self.width_cm + t and -t <= y <= self.depth_cm + t):
            return None
        c = min(max(x, 0.0), self.width_cm - 1e-6)
        return min(int(c / (self.width_cm / 3.0)), 2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["img_corners"], d["width_cm"], d["depth_cm"],
                   d.get("tolerance_cm", 12.0))


# ---------------------------------------------------------------- features

def grip_point(pts, vis, side, extend=GRIP_EXTEND):
    """Where the hand is actually holding something, not the wrist joint."""
    w_i = L_WRIST if side == "L" else R_WRIST
    i_i = L_INDEX if side == "L" else R_INDEX
    p_i = L_PINKY if side == "L" else R_PINKY
    if vis[i_i] <= VIS or vis[p_i] <= VIS:
        return None
    kx = (pts[i_i][0] + pts[p_i][0]) / 2.0
    ky = (pts[i_i][1] + pts[p_i][1]) / 2.0
    if vis[w_i] > VIS:
        kx += (kx - pts[w_i][0]) * extend
        ky += (ky - pts[w_i][1]) * extend
    return kx, ky


def frame_features(pts, vis, calib):
    """
    27 numbers describing one frame.

    Everything positional is expressed in DESK units normalised to 0..1, not
    pixels, so the features survive a resolution change or a camera nudge.
    Returns zeros when no person is detected, which is itself informative.
    """
    f = np.zeros(FEATS_PER_FRAME, dtype=np.float32)
    if pts is None:
        return f

    W, D = calib.width_cm, calib.depth_cm

    def norm(px, py):
        x, y = calib.to_desk(px, py)
        return x / W, y / D

    i = 0
    # Grip points (the contact proxy) and their zone one-hots.
    for side in ("L", "R"):
        g = grip_point(pts, vis, side)
        if g is None:
            f[i:i + 2] = 0.0
            i += 2
            f[i] = 0.0
            i += 1
            f[i:i + 4] = [1, 0, 0, 0]   # off/unknown
            i += 4
        else:
            gx, gy = norm(*g)
            f[i:i + 2] = [gx, gy]
            i += 2
            f[i] = 1.0                   # valid flag
            i += 1
            z = calib.zone_of(*g)
            onehot = [0, 0, 0, 0]
            onehot[0 if z is None else z + 1] = 1
            f[i:i + 4] = onehot
            i += 4

    # Wrists and elbows give arm configuration.
    for idx in (L_WRIST, R_WRIST, L_ELBOW, R_ELBOW):
        if vis[idx] > VIS:
            f[i:i + 2] = norm(*pts[idx])
        i += 2

    # Torso axis: the canonicalisation reference frame.
    if vis[L_SHOULDER] > VIS and vis[R_SHOULDER] > VIS:
        (lx, ly), (rx, ry) = pts[L_SHOULDER], pts[R_SHOULDER]
        ang = math.atan2(ry - ly, rx - lx)
        f[i] = math.sin(ang)
        f[i + 1] = math.cos(ang)
        mx, my = norm((lx + rx) / 2.0, (ly + ry) / 2.0)
        f[i + 2] = mx
        f[i + 3] = my
    i += 4

    # Distance between the two grip points: open vs together hands.
    gl = grip_point(pts, vis, "L")
    gr = grip_point(pts, vis, "R")
    if gl and gr:
        ax, ay = norm(*gl)
        bx, by = norm(*gr)
        f[i] = math.hypot(ax - bx, ay - by)
    i += 1
    return f


def featurise(frame_feats, end_idx):
    """
    Build one model input from the WINDOW frames ending at end_idx.

    Causal by construction: it only ever looks backwards, so the exact same
    function works offline on a recorded take and live on a camera stream.

    Position alone cannot separate steps that share a zone - draw medium,
    inoculate and seal all happen with hands in the centre of the bench, and a
    positional model collapses them together. Motion does separate them:
    wiping is broad and repetitive, dispensing is a near-static hold, labelling
    is small and rhythmic. So the window also contributes velocity, spread and
    path length of both grip points.
    """
    start = end_idx - WINDOW + 1
    win = np.zeros((WINDOW, FEATS_PER_FRAME), dtype=np.float32)
    for k in range(WINDOW):
        j = start + k
        if 0 <= j < len(frame_feats):
            win[k] = frame_feats[j]

    sub = win[SUBSAMPLE]
    parts = [sub.reshape(-1)]

    grips = np.concatenate(
        [sub[:, a:b] for a, b in GRIP_SLICES], axis=1)      # (S, 4)
    parts.append(np.diff(grips, axis=0).reshape(-1))         # velocities

    full = np.concatenate(
        [win[:, a:b] for a, b in GRIP_SLICES], axis=1)       # (WINDOW, 4)
    parts.append(full.std(axis=0))                           # spread
    step = np.abs(np.diff(full, axis=0))
    parts.append(np.array([step[:, 0:2].sum(), step[:, 2:4].sum()],
                          dtype=np.float32))                 # path length
    parts.append(np.array([step[:, 0:2].mean(), step[:, 2:4].mean()],
                          dtype=np.float32))                 # mean speed

    return np.concatenate(parts).astype(np.float32)