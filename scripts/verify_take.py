#!/usr/bin/env python3
"""
SIH26174 - Take verification.

Replays a recorded take with its CSV labels burned onto the frame so you can
confirm the labels line up with what the actor was actually doing. Run this on
three random takes at the end of every session.

Usage
  python verify_take.py data/raw/take_001__A__s01__nominal.mp4
"""

import csv
import os
import sys

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python verify_take.py <take>.mp4")

    mp4_path = sys.argv[1]
    csv_path = os.path.splitext(mp4_path)[0] + ".csv"

    if not os.path.exists(csv_path):
        sys.exit(f"No label file found at {csv_path}")

    with open(csv_path) as handle:
        labels = [row["step_name"] for row in csv.DictReader(handle)]

    if not labels:
        sys.exit("Label file is empty. This take is unusable, re-record it.")

    cap = cv2.VideoCapture(mp4_path)
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        name = labels[idx] if idx < len(labels) else "NO LABEL"
        colour = (80, 80, 235) if name in ("idle", "NO LABEL") else (90, 210, 120)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(frame, f"{idx}  {name}", (12, 32), FONT, 0.8, colour, 2)

        cv2.imshow("verify_take", frame)
        if cv2.waitKey(40) & 0xFF == ord("q"):
            break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if abs(idx - len(labels)) > 3:
        print(f"WARNING: {idx} video frames but {len(labels)} label rows. Investigate.")
    else:
        print(f"OK: {idx} frames, {len(labels)} labels.")


if __name__ == "__main__":
    main()
