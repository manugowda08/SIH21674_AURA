#!/usr/bin/env python3
"""
SIH26174 - Train the step classifier.

Splits BY ACTOR, never randomly by frame. A random split puts adjacent frames
of the same take on both sides and reports ~99% accuracy that collapses the
moment a stranger stands in front of the camera. The number this prints is
the number you can quote.

Usage
  python scripts/train.py
  python scripts/train.py --holdout C
  python scripts/train.py --loso          # leave-one-subject-out, all actors
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from har.pipeline import STEP_NAMES, WINDOW, featurise  # noqa: E402
from har.protocol import Protocol, Smoother, Validator  # noqa: E402


NOMINAL = list(range(1, 9))

# Steps 4 (draw medium) and 5 (inoculate) are the same visual event: syringe in
# hand, centre zone. Without object detection there is nothing to separate them,
# and the pair drags each other's scores down. Merging gives a 7-step protocol
# that the model can actually resolve. Seven steps that work beat eight that do
# not - and the merge is reversible once a prop detector exists.
MERGE45 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 7}
MERGED_NAMES = ["idle", "don_gloves", "sanitise_surface", "retrieve_cartridge",
                "draw_and_inoculate", "seal_and_label", "stow_in_incubator",
                "confirm_on_interface"]
# Under the merge these take types no longer contain a detectable violation,
# because the omitted or reordered step folds into its neighbour.
MERGE_NEUTRALISED = {"skip-04", "skip-05", "swap-45"}


def apply_merge(takes):
    lut = np.array([MERGE45[i] for i in range(9)], dtype=np.int64)
    for t in takes:
        t["y"] = lut[t["y"]]
    return takes


def expected_order_merged(seqtype):
    base = list(range(1, 8))
    t = seqtype.lower()
    if t == "idle":
        return []
    if t in MERGE_NEUTRALISED or t == "nominal":
        return base
    if t.startswith("skip-"):
        k = MERGE45[int(t.split("-")[1])]
        return [s for s in base if s != k]
    if t.startswith("swap-"):
        d = t.split("-")[1]
        a, b = MERGE45[int(d[0])], MERGE45[int(d[1])]
        o = list(base)
        i, j = o.index(a), o.index(b)
        o[i], o[j] = o[j], o[i]
        return o
    if t.startswith("jump-"):
        d = t.split("-")[1]
        a, b = MERGE45[int(d[0])], MERGE45[int(d[1])]
        if int(d[0]) == 8:
            return [a] + [s for s in base if s != a]
        o = [s for s in base if s != b]
        return o[:o.index(a) + 1] + [b] + o[o.index(a) + 1:]
    return base


ACTIVE_PROTOCOL = "configs/protocol.yaml"
MERGED = False
KEEP = None          # set of original step ids to model; others become idle
KEEP_MAP = None      # original id -> compact id


def apply_keep(takes, keep_ids):
    """
    Model only the steps the sensing can actually resolve; everything else
    becomes idle.

    Reporting a step the system cannot detect guarantees a false alarm on every
    correct run, because the validator sees it as missing. A shorter protocol
    that is genuinely verified is worth more than a longer one that cries wolf.
    """
    global KEEP_MAP
    keep = sorted(keep_ids)
    KEEP_MAP = {orig: i + 1 for i, orig in enumerate(keep)}
    lut = np.zeros(9, dtype=np.int64)
    for orig, new in KEEP_MAP.items():
        lut[orig] = new
    for t in takes:
        t["y"] = lut[t["y"]]
    return takes


def expected_order_keep(seqtype):
    base = [KEEP_MAP[o] for o in sorted(KEEP_MAP)]
    t = seqtype.lower()
    if t == "idle":
        return []
    if t == "nominal":
        return base
    if t.startswith("skip-"):
        k = int(t.split("-")[1])
        return [v for o, v in sorted(KEEP_MAP.items()) if o != k]
    if t.startswith(("swap-", "jump-")):
        d = t.split("-")[1]
        a, b = int(d[0]), int(d[1])
        order = [o for o in sorted(KEEP_MAP)]
        if t.startswith("swap-"):
            if a in order and b in order:
                i, j = order.index(a), order.index(b)
                order[i], order[j] = order[j], order[i]
        else:
            if a == 8 and a in order:
                order = [a] + [o for o in order if o != a]
            elif a in order and b in order:
                order = [o for o in order if o != b]
                order = order[:order.index(a) + 1] + [b] + order[order.index(a) + 1:]
        return [KEEP_MAP[o] for o in order]
    return base


def build_protocol_cfg(keep_ids, path):
    import yaml
    full = yaml.safe_load(open("configs/protocol.yaml"))
    steps = [s for s in full["steps"] if s["id"] in keep_ids]
    for i, s in enumerate(steps, 1):
        s["id"] = i
    cfg = {"name": full["name"] + f" ({len(steps)}-step verified subset)",
           "description": "Auto-generated: only steps the sensing resolves.",
           "steps": steps}
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def expected_order(seqtype):
    """The step order a take of this type should contain."""
    if KEEP is not None:
        return expected_order_keep(seqtype)
    if MERGED:
        return expected_order_merged(seqtype)
    t = seqtype.lower()
    if t == "nominal":
        return list(NOMINAL)
    if t == "idle":
        return []
    if t.startswith("skip-"):
        k = int(t.split("-")[1])
        return [s for s in NOMINAL if s != k]
    if t.startswith("swap-"):
        digits = t.split("-")[1]
        a, b = int(digits[0]), int(digits[1])
        o = list(NOMINAL)
        i, j = o.index(a), o.index(b)
        o[i], o[j] = o[j], o[i]
        return o
    if t.startswith("jump-"):
        digits = t.split("-")[1]
        a, b = int(digits[0]), int(digits[1])
        if a == 8:
            return [8] + [s for s in NOMINAL if s != 8]
        o = [s for s in NOMINAL if s != b]
        return o[:o.index(a) + 1] + [b] + o[o.index(a) + 1:]
    return list(NOMINAL)


def take_probs(scaler, clf, take, stride=3):
    """Model posteriors for a take. Computed once, reused across threshold sweeps."""
    X = take["X"]
    ends = list(range(WINDOW - 1, len(X), stride))
    if not ends:
        return None
    feats = np.array([featurise(X, e) for e in ends], dtype=np.float32)
    return clf.predict_proba(scaler.transform(feats))


def decode_from_probs(probs, conf, dwell, stride=3, fps=20.0):
    """Returns (sequence, timed) where timed is [(t_seconds, step), ...]."""
    sm = Smoother(min_conf=conf, dwell_seconds=dwell, update_hz=fps / stride)
    seq, timed, last = [], [], None
    for i, p in enumerate(probs):
        t = (WINDOW - 1 + i * stride) / fps
        committed = sm.update(int(np.argmax(p)), float(p.max()))
        if committed != last:
            timed.append((t, committed))
            last = committed
        if committed != 0 and (not seq or seq[-1] != committed):
            seq.append(committed)
    return seq, timed


def alerts_for(timed, proto, grace):
    val = Validator(proto, skip_grace_s=grace)
    alerted = False
    for t, s in timed:
        for ev in val.update(s, now=t):
            if ev["severity"] == "alert":
                alerted = True
    end = (timed[-1][0] if timed else 0.0) + 30.0
    for ev in val.tick(end):
        if ev["severity"] == "alert":
            alerted = True
    return alerted


def oof_probs(takes, train_actors, stride=3):
    """
    Out-of-fold posteriors: for each training actor, predict them using a model
    trained on the OTHER training actors. Tuning on in-fold predictions is what
    produced a threshold that scored +0.90 in training and 0.00 on a new actor.
    """
    out = []
    acts = sorted(train_actors)
    if len(acts) < 2:
        return None
    for held in acts:
        rest = {a for a in acts if a != held}
        Xtr, ytr = build(takes, rest)
        if Xtr is None:
            continue
        sc, cl = train_one(Xtr, ytr)
        for t in takes:
            if t["actor"] != held or t["seqtype"] == "idle":
                continue
            pr = take_probs(sc, cl, t, stride)
            if pr is not None:
                out.append((t, pr))
    return out


def sweep_thresholds(scaler, clf, takes, stride=3, cache=None):
    """
    Pick confidence and dwell on the TRAINING actors, never on the held-out one.

    Scored by detection minus false alarms: a system that alerts on everything
    scores zero, which is the correct verdict for it.
    """
    if cache is None:
        cache = []
        for t in takes:
            if t["seqtype"] == "idle":
                continue
            pr = take_probs(scaler, clf, t, stride)
            if pr is not None:
                cache.append((t, pr))
    proto = Protocol.load(ACTIVE_PROTOCOL)
    best, best_score = (0.70, 1.8, 6.0), -9.9
    for conf in (0.45, 0.55, 0.65, 0.75):
        for dwell in (1.0, 1.8, 2.6):
            for grace in (0.0, 4.0, 8.0, 14.0):
                det = det_n = fa = fa_n = 0
                for t, pr in cache:
                    _, timed = decode_from_probs(pr, conf, dwell, stride)
                    alerted = alerts_for(timed, proto, grace)
                    clean = (t["seqtype"] == "nominal"
                             or (MERGED and t["seqtype"] in MERGE_NEUTRALISED))
                    if clean:
                        fa_n += 1
                        fa += int(alerted)
                    else:
                        det_n += 1
                        det += int(alerted)
                score = (det / max(det_n, 1)) - (fa / max(fa_n, 1))
                if score > best_score:
                    best_score, best = score, (conf, dwell, grace)
    print(f"  tuned out-of-fold: conf={best[0]}, dwell={best[1]}s, "
          f"grace={best[2]}s (score {best_score:+.2f})")
    return best


def sequence_report(scaler, clf, takes, label="", conf=0.70, dwell=1.8,
                    grace=6.0):
    """
    Operational metrics: does the system recover the right step order, does it
    catch real violations, and does it stay quiet on correct runs. These are
    what an operations person would ask for; frame accuracy is not.
    """
    proto = Protocol.load(ACTIVE_PROTOCOL)
    exact = total = 0
    _ = grace
    err_caught = err_total = 0
    false_alarm = nominal_total = 0

    for t in takes:
        if t["seqtype"] == "idle":
            continue
        want = expected_order(t["seqtype"])
        pr = take_probs(scaler, clf, t, stride=3)
        if pr is None:
            continue
        got, timed = decode_from_probs(pr, conf, dwell, stride=3)
        total += 1
        if got == want:
            exact += 1
        alerted = alerts_for(timed, proto, grace)
        is_clean = (t["seqtype"] == "nominal"
                    or (MERGED and t["seqtype"] in MERGE_NEUTRALISED))
        if is_clean:
            nominal_total += 1
            if alerted:
                false_alarm += 1
        else:
            err_total += 1
            if alerted:
                err_caught += 1

    print(f"\n--- {label} sequence-level ---")
    if total:
        print(f"exact step order recovered : {exact}/{total} "
              f"({100.0 * exact / total:.0f}%)")
    if err_total:
        print(f"violations detected        : {err_caught}/{err_total} "
              f"({100.0 * err_caught / err_total:.0f}%)")
    if nominal_total:
        print(f"false alarms on nominal    : {false_alarm}/{nominal_total} "
              f"({100.0 * false_alarm / nominal_total:.0f}%)")
    return {
        "exact_order": exact / total if total else 0.0,
        "violation_detection": err_caught / err_total if err_total else 0.0,
        "false_alarm": false_alarm / nominal_total if nominal_total else 0.0,
    }


def load_takes(feat_dir):
    takes = []
    for p in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        takes.append({
            "X": d["X"], "y": d["y"],
            "actor": str(d["actor"]), "session": str(d["session"]),
            "seqtype": str(d["seqtype"]), "name": os.path.basename(p),
        })
    return takes


def windows_from_take(take, stride=2):
    """Every window ends at a labelled frame; the label is that frame's."""
    X, y = take["X"], take["y"]
    out_X, out_y = [], []
    for end in range(WINDOW - 1, len(X), stride):
        out_X.append(featurise(X, end))
        out_y.append(y[end])
    if not out_X:
        return None, None
    return np.array(out_X, dtype=np.float32), np.array(out_y, dtype=np.int64)


def build(takes, actors, stride=2):
    Xs, ys = [], []
    for t in takes:
        if t["actor"] not in actors:
            continue
        wx, wy = windows_from_take(t, stride)
        if wx is None:
            continue
        Xs.append(wx)
        ys.append(wy)
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)


def train_one(Xtr, ytr, seed=0):
    scaler = StandardScaler().fit(Xtr)
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.12,
        random_state=seed,
        verbose=False,
    )
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def evaluate(scaler, clf, Xte, yte, label=""):
    pred = clf.predict(scaler.transform(Xte))
    acc = float((pred == yte).mean())
    present = sorted(set(yte.tolist()) | set(pred.tolist()))
    names = [STEP_NAMES[i] for i in present]
    print(f"\n--- {label} ---")
    print(f"frame accuracy: {acc * 100:.1f}%")
    print(classification_report(yte, pred, labels=present,
                                target_names=names, zero_division=0, digits=3))
    return acc, pred, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--holdout", default=None, help="Actor to hold out, eg C")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--merge45", action="store_true",
                    help="Merge draw_medium and inoculate_vial into one step")
    ap.add_argument("--keep", default=None,
                    help="Comma-separated original step ids to model, "
                         "eg --keep 1,3,7,8. Everything else becomes idle.")
    args = ap.parse_args()

    global MERGED, STEP_NAMES, KEEP, ACTIVE_PROTOCOL
    if args.merge45:
        MERGED = True
        STEP_NAMES = MERGED_NAMES
        ACTIVE_PROTOCOL = "configs/protocol7.yaml"
        print("MERGED MODE: steps 4 and 5 combined -> 7-step protocol")
    if args.keep:
        KEEP = {int(x) for x in args.keep.split(",")}
        base = ["idle"] + [n for i, n in enumerate(STEP_NAMES) if i in KEEP]
        STEP_NAMES = base
        ACTIVE_PROTOCOL = build_protocol_cfg(KEEP, "configs/protocol_subset.yaml")
        print(f"SUBSET MODE: modelling steps {sorted(KEEP)} "
              f"-> {len(KEEP)}-step protocol, wrote {ACTIVE_PROTOCOL}")

    takes = load_takes(args.features)
    if takes and args.merge45:
        takes = apply_merge(takes)
    if takes and args.keep:
        takes = apply_keep(takes, KEEP)
    if not takes:
        sys.exit(f"No features in {args.features}. Run extract_features.py first.")

    actors = sorted({t["actor"] for t in takes})
    print(f"{len(takes)} takes, actors: {', '.join(actors)}")
    for a in actors:
        n = sum(1 for t in takes if t["actor"] == a)
        print(f"   {a}: {n} takes")
    if len(actors) < 2:
        sys.exit("\nNeed at least 2 actors to split honestly.")

    os.makedirs(args.out, exist_ok=True)

    if args.loso:
        accs = {}
        for held in actors:
            tr = [a for a in actors if a != held]
            Xtr, ytr = build(takes, set(tr), args.stride)
            Xte, yte = build(takes, {held}, args.stride)
            if Xtr is None or Xte is None:
                continue
            sc, clf = train_one(Xtr, ytr)
            acc, _, _ = evaluate(sc, clf, Xte, yte, f"holdout actor {held}")
            tr_takes = [t for t in takes if t["actor"] != held]
            c, d, g = sweep_thresholds(sc, clf, None,
                                       cache=oof_probs(takes, set(tr)))
            sequence_report(sc, clf, [t for t in takes if t["actor"] == held],
                            f"holdout actor {held}", c, d, g)
            accs[held] = acc
        print("\n=== Leave-one-subject-out ===")
        for k, v in accs.items():
            print(f"  holdout {k}: {v * 100:.1f}%")
        vals = list(accs.values())
        print(f"  mean {np.mean(vals) * 100:.1f}%  std {np.std(vals) * 100:.1f}%")
        return

    held = args.holdout or actors[-1]
    if held not in actors:
        sys.exit(f"No actor '{held}' in the dataset. Available: {', '.join(actors)}")
    train_actors = {a for a in actors if a != held}
    print(f"\ntrain on {sorted(train_actors)}, hold out {held}")

    Xtr, ytr = build(takes, train_actors, args.stride)
    Xte, yte = build(takes, {held}, args.stride)
    print(f"train windows {Xtr.shape}, test windows {Xte.shape}")

    t0 = time.time()
    scaler, clf = train_one(Xtr, ytr)
    print(f"trained in {time.time() - t0:.1f}s")

    acc, pred, present = evaluate(scaler, clf, Xte, yte, f"held-out actor {held}")

    print("\n  computing out-of-fold predictions for threshold tuning...")
    cache = oof_probs(takes, train_actors)
    conf, dwell, grace = sweep_thresholds(scaler, clf, None, cache=cache)
    held_takes = [t for t in takes if t["actor"] == held]
    seq_metrics = sequence_report(scaler, clf, held_takes,
                                  f"held-out actor {held}", conf, dwell, grace)

    print("\nconfusion matrix (rows true, cols predicted)")
    cm = confusion_matrix(yte, pred, labels=present)
    hdr = "".join(f"{STEP_NAMES[i][:8]:>9}" for i in present)
    print(f"{'':>22}{hdr}")
    for r, i in enumerate(present):
        row = "".join(f"{v:>9}" for v in cm[r])
        print(f"{STEP_NAMES[i]:>22}{row}")

    # Retrain on everything for the model that actually ships.
    Xall, yall = build(takes, set(actors), args.stride)
    scaler_all, clf_all = train_one(Xall, yall)
    bundle = os.path.join(args.out, "model.joblib")
    joblib.dump({"scaler": scaler_all, "clf": clf_all,
                 "step_names": STEP_NAMES, "window": WINDOW,
                 "merged": args.merge45, "keep": sorted(KEEP) if KEEP else None,
                 "protocol": ACTIVE_PROTOCOL,
                 "min_conf": conf, "dwell_seconds": dwell,
                 "skip_grace_s": grace}, bundle)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"holdout_actor": held, "holdout_frame_accuracy": acc,
                   "sequence": seq_metrics,
                   "actors": actors, "n_takes": len(takes),
                   "train_windows": int(Xtr.shape[0])}, f, indent=2)

    print(f"\nsaved {bundle}")
    print("\n=== numbers to quote ===")
    print(f"  violation detection : {seq_metrics['violation_detection'] * 100:.0f}% "
          f"on unseen actor {held}")
    print(f"  false alarm rate    : {seq_metrics['false_alarm'] * 100:.0f}%")
    print(f"  exact step order    : {seq_metrics['exact_order'] * 100:.0f}%")
    print(f"  frame accuracy      : {acc * 100:.1f}%  (harsh metric, report for honesty)")


if __name__ == "__main__":
    main()