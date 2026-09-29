#!/usr/bin/env python3
"""
SIH26174 - Feature extraction.

Runs pose over every take and writes per-frame feature arrays to disk, so
training can iterate in seconds instead of re-decoding video each time.

Resumable: already-processed takes are skipped, so you can stop and restart.

Usage
  python scripts/extract_features.py
  python scripts/extract_features.py --raw data/raw --out data/features
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from har.pipeline import (  # noqa: E402
    DeskCalibration, PoseDetector, STEP_NAMES, frame_features, model_path,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_take(stem):
    """take_001__A__s01__nominal -> (1, 'A', 's01', 'nominal')"""
    base = os.path.basename(stem)
    parts = base.split("__")
    if len(parts) != 4:
        return None
    try:
        num = int(parts[0].replace("take_", ""))
    except ValueError:
        return None
    return num, parts[1], parts[2], parts[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--calib", default="data/desk_calibration.json")
    ap.add_argument("--force", action="store_true", help="Redo completed takes")
    args = ap.parse_args()

    if not os.path.exists(args.calib):
        sys.exit(f"No calibration at {args.calib}. Run preflight first.")
    calib = DeskCalibration.load(args.calib)
    print(f"Calibration: {calib.width_cm:.0f} x {calib.depth_cm:.0f} cm")

    os.makedirs(args.out, exist_ok=True)
    mp4s = sorted(glob.glob(os.path.join(args.raw, "*.mp4")))
    if not mp4s:
        sys.exit(f"No takes in {args.raw}")

    todo = []
    for m in mp4s:
        stem = os.path.splitext(m)[0]
        meta = parse_take(stem)
        if meta is None:
            print(f"  skipping oddly named {os.path.basename(m)}")
            continue
        out_npz = os.path.join(args.out, os.path.basename(stem) + ".npz")
        if os.path.exists(out_npz) and not args.force:
            continue
        todo.append((m, stem, meta, out_npz))

    print(f"{len(mp4s)} takes found, {len(todo)} to process\n")
    if not todo:
        print("Nothing to do. Use --force to redo.")
        return

    t_start = time.time()

    for n, (mp4, stem, meta, out_npz) in enumerate(todo, 1):
        num, actor, session, seqtype = meta
        csv_path = stem + ".csv"
        if not os.path.exists(csv_path):
            print(f"  [{n}/{len(todo)}] {os.path.basename(stem)}  NO CSV, skipped")
            continue

        with open(csv_path) as f:
            labels = [int(r["label"]) for r in csv.DictReader(f)]

        # Fresh landmarker per take: detect_for_video requires timestamps
        # to be monotonically increasing for the life of the landmarker,
        # but each take's ts resets to 0, so one shared instance across
        # takes throws "Input timestamp must be monotonically increasing."
        detector = PoseDetector(model_path(SCRIPT_DIR))

        cap = cv2.VideoCapture(mp4)
        feats = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(len(feats) * 1000 / 20)
            pts, vis = detector.process(frame, timestamp_ms=ts)
            feats.append(frame_features(pts, vis, calib))
        cap.release()

        # Tail-flush means the encoder can drop the last fraction of a second.
        # Frame index alignment holds for everything that exists, so truncate.
        n_use = min(len(feats), len(labels))
        X = np.array(feats[:n_use], dtype=np.float32)
        y = np.array(labels[:n_use], dtype=np.int64)

        np.savez_compressed(
            out_npz, X=X, y=y,
            actor=actor, session=session, seqtype=seqtype, take=num,
        )
        rate = n / max(time.time() - t_start, 1e-9) * 60
        print(f"  [{n}/{len(todo)}] {os.path.basename(stem):<44} "
              f"{n_use:5d} frames  ({rate:.1f} takes/min)")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()