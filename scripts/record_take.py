#!/usr/bin/env python3
"""
SIH26174 - Dataset recording harness.

Records one take of the experiment protocol to an MP4 while an operator marks
step boundaries live with the number keys. Writes a per-frame label CSV so no
manual video scrubbing is ever needed.

Keys during recording
  1-8   mark the start of that protocol step
  0     mark idle / no step in progress
  u     undo: revert to the previous label
  q     end the take and save
  ESC   abort the take and delete the files

Usage
  python record_take.py --actor A --session s01 --seqtype nominal
  python record_take.py --actor B --session s02 --seqtype skip-03
  python record_take.py --actor A --session s01 --seqtype idle --camera 1
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import cv2

# --------------------------------------------------------------------------
# Protocol definition. Keep these ids frozen once recording begins.
# --------------------------------------------------------------------------
STEPS = {
    0: "idle",
    1: "don_gloves",
    2: "sanitise_surface",
    3: "retrieve_cartridge",
    4: "draw_medium",
    5: "inoculate_vial",
    6: "seal_and_label",
    7: "stow_in_incubator",
    8: "confirm_on_interface",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def next_take_id(outdir):
    """Scan the output directory and return the next free take number."""
    ids = [0]
    for path in glob.glob(os.path.join(outdir, "take_*.mp4")):
        match = re.search(r"take_(\d+)__", os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    return max(ids) + 1


def draw_overlay(frame, label, elapsed, frame_idx, seqtype):
    """Draw the operator HUD onto a copy of the frame. Never onto the saved frame."""
    hud = frame.copy()
    height, width = hud.shape[:2]

    cv2.rectangle(hud, (0, 0), (width, 78), (0, 0, 0), -1)
    cv2.rectangle(hud, (0, height - 34), (width, height), (0, 0, 0), -1)

    colour = (80, 80, 235) if label == 0 else (90, 210, 120)
    cv2.putText(hud, f"[{label}] {STEPS[label]}", (12, 34), FONT, 0.85, colour, 2)
    cv2.putText(
        hud,
        f"{elapsed:6.1f}s   frame {frame_idx}   {seqtype}",
        (12, 64),
        FONT,
        0.55,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        hud,
        "1-8 step   0 idle   u undo   q save   ESC abort",
        (12, height - 12),
        FONT,
        0.5,
        (170, 170, 170),
        1,
    )
    return hud


RESET_CHECKLIST = [
    "GLOVES off the performer's hands, folded back into the pouch",
    "Wipe back on top of the packet, centre",
    "Vial back in the rack, LEFT zone, cap on",
    "Label tape peeled off the vial and discarded",
    "Syringe plunger pushed in, laid centre-left, dry",
    "Amber bottle capped, upright, centre-left",
    "Marker capped, centre",
    "Incubator box empty, LID CLOSED, right zone",
    "Tablet flat, screen up, centre-right",
    "Whole layout matches the reference photo",
    "Performer seated, hands on the bench, still",
]


def run_checklist(seqtype):
    """Walk the solo operator through the reset. No floor manager to do it."""
    print("\n" + "=" * 58)
    print(f"  RESET  \u2014  next take is  {seqtype}")
    print("=" * 58)
    for item in RESET_CHECKLIST:
        print(f"   [ ] {item}")
    print("=" * 58)
    try:
        input("  Reset done? Press Enter to record, Ctrl+C to stop.  ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="Record one labelled take.")
    parser.add_argument("--actor", required=True, help="Actor id, e.g. A")
    parser.add_argument("--session", required=True, help="Session id, e.g. s01")
    parser.add_argument(
        "--seqtype",
        required=True,
        help="nominal | skip-NN | swap-NN-NN | idle",
    )
    parser.add_argument("--outdir", default="data/raw")
    parser.add_argument(
        "--camera",
        default="0",
        help="Webcam index (0, 1, 2) or a full RTSP URL for an IP camera",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--lighting", default="overhead")
    parser.add_argument("--camera-pos", default="pos1")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--no-checklist", action="store_true",
        help="Skip the reset checklist prompt (only if a floor manager is doing it)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not args.no_checklist:
        run_checklist(args.seqtype)

    take_id = next_take_id(args.outdir)
    stem = f"take_{take_id:03d}__{args.actor}__{args.session}__{args.seqtype}"
    mp4_path = os.path.join(args.outdir, stem + ".mp4")
    csv_path = os.path.join(args.outdir, stem + ".csv")
    json_path = os.path.join(args.outdir, stem + ".json")

    is_network = not args.camera.isdigit()
    source = args.camera if is_network else int(args.camera)

    if is_network:
        # Force TCP and disable buffering. Without this OpenCV hands back stale
        # frames and every keypress label lands late by the buffer depth.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0",
        )
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        sys.exit(
            f"Could not open camera source: {args.camera}\n"
            "Webcam: try --camera 1 or 2.\n"
            "IP camera: check the RTSP path, credentials, and that the host is reachable."
        )

    ok, frame = cap.read()
    if not ok:
        cap.release()
        sys.exit("Camera opened but returned no frame. Check it is not in use elsewhere.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, args.fps, (width, height))

    label = 0
    prev_label = 0
    rows = []
    frame_idx = 0
    aborted = False

    print(f"\nRecording {stem}")
    print("Press a number key the moment each step begins. q to save, ESC to abort.\n")

    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, ending take.")
            break

        elapsed = time.perf_counter() - start

        writer.write(frame)
        rows.append((frame_idx, round(elapsed, 4), label, STEPS[label]))

        cv2.imshow("record_take", draw_overlay(frame, label, elapsed, frame_idx, args.seqtype))
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            aborted = True
            break
        if key == ord("q"):
            break
        if key == ord("u"):
            label, prev_label = prev_label, label
        elif ord("0") <= key <= ord("8"):
            new_label = key - ord("0")
            if new_label != label:
                prev_label = label
                label = new_label
                print(f"  {elapsed:6.1f}s  ->  [{label}] {STEPS[label]}")

        frame_idx += 1

    duration = time.perf_counter() - start
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if aborted:
        for path in (mp4_path, csv_path, json_path):
            if os.path.exists(path):
                os.remove(path)
        print("\nTake aborted. Files deleted.")
        return

    with open(csv_path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame_idx", "t_seconds", "label", "step_name"])
        out.writerows(rows)

    counts = {}
    for row in rows:
        counts[row[3]] = counts.get(row[3], 0) + 1

    measured_fps = round(frame_idx / duration, 2) if duration > 0 else 0.0

    meta = {
        "take_id": take_id,
        "actor": args.actor,
        "session": args.session,
        "seqtype": args.seqtype,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "frames": frame_idx,
        "requested_fps": args.fps,
        "measured_fps": measured_fps,
        "resolution": [width, height],
        "camera_source": re.sub(r"//[^@]+@", "//***@", args.camera),
        "lighting": args.lighting,
        "camera_pos": args.camera_pos,
        "notes": args.notes,
        "label_counts": counts,
    }
    with open(json_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\nSaved {stem}")
    print(f"  {frame_idx} frames, {duration:.1f}s, measured {measured_fps} fps")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {name:24s} {count:5d} frames")

    distinct = len([k for k in counts if k != "idle"])
    if distinct < 2 and args.seqtype != "idle":
        print("\n  WARNING: fewer than 2 steps labelled. Did the operator miss keypresses?")


if __name__ == "__main__":
    main()