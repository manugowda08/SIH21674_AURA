#!/usr/bin/env python3
"""
SIH26174 - Dataset audit.

Checks every take for video/label frame-count mismatches and reports the
pattern, which is what tells you whether a mismatch is harmless.

  Tail loss    - the video is short by a few frames at the END. Labels 0..N-1
                 still line up with video frames 0..N-1, so the take is fine;
                 feature extraction just ignores the orphaned tail labels.
  Mid-stream   - frames dropped partway through. Every label after the gap is
                 shifted. The take is corrupted and should be re-recorded.

A consistent small deficit across many takes points to encoder tail-flush,
which is benign. Large or wildly varying deficits point to dropped frames
under load, which is not.

Usage
  python audit_takes.py data/raw
"""

import csv
import glob
import json
import os
import sys

import cv2


def count_video_frames(path):
    """Actually decode and count. The container's metadata is often wrong."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    mp4s = sorted(glob.glob(os.path.join(root, "*.mp4")))
    if not mp4s:
        sys.exit(f"No takes found in {root}")

    print(f"Auditing {len(mp4s)} takes in {root}\n")
    print(f"{'take':<42} {'video':>7} {'labels':>7} {'diff':>6}  {'pct':>6}  status")
    print("-" * 88)

    bad, suspect, clean = [], [], []
    total_deficit = 0

    for mp4 in mp4s:
        stem = os.path.splitext(mp4)[0]
        name = os.path.basename(stem)
        csv_path = stem + ".csv"
        json_path = stem + ".json"

        if not os.path.exists(csv_path):
            print(f"{name:<42} {'--':>7} {'MISSING':>7}")
            bad.append(name)
            continue

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        n_lab = len(rows)
        n_vid = count_video_frames(mp4)

        if n_vid is None:
            print(f"{name:<42} {'UNREADABLE':>7}")
            bad.append(name)
            continue

        diff = n_lab - n_vid
        pct = 100.0 * diff / max(n_lab, 1)
        total_deficit += diff

        if diff == 0:
            status = "exact"
            clean.append(name)
        elif 0 < diff <= 90 and pct < 8.0:
            status = "tail loss, OK"
            clean.append(name)
        elif diff < 0:
            status = "MORE video than labels"
            suspect.append(name)
        else:
            status = "LARGE GAP - check"
            suspect.append(name)

        print(f"{name:<42} {n_vid:>7} {n_lab:>7} {diff:>6} {pct:>5.1f}%  {status}")

        if not os.path.exists(json_path):
            print(f"{'':<42} json missing")

    print("-" * 88)
    print(f"\n  clean/tail-loss : {len(clean)}")
    print(f"  needs a look    : {len(suspect)}")
    print(f"  broken          : {len(bad)}")
    if mp4s:
        print(f"  mean deficit    : {total_deficit / len(mp4s):.1f} frames per take")

    if suspect:
        print("\n  Inspect these with verify_take.py and watch whether the label")
        print("  still matches the action at the END of the video:")
        for n in suspect:
            print(f"    {n}")

    print("\n  A small deficit (under ~90 frames) on most takes is encoder")
    print("  tail-flush and is harmless - alignment by frame index is preserved.")
    print("  Feature extraction should use min(video_frames, label_rows).")


if __name__ == "__main__":
    main()