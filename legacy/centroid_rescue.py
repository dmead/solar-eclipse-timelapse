"""Recover a capture the circle fit cannot handle, using a calibrated centroid.

14_05_08 is good data - a clean, well exposed crescent - but the ray-march circle
fit scatters over 595 px across it while every other capture sits at 0.1-0.8 px.
Rather than lose 2.4 s of the partial phase, this places that capture from the
brightness centroid instead.

A centroid is NOT the disc centre for a crescent, which is exactly the bug that
made the first version of this timelapse bounce. But the offset between the two is
a smooth function of how thin the crescent is, so it can be measured where both
methods work and carried across where only one does: the neighbouring captures are
fitted by both, their centroid-minus-circle offset is computed, and that offset is
interpolated in time across the broken capture.

Run with the Umbra venv's Python, which has numpy:

    D:/projects/umbra/venv/Scripts/python.exe centroid_rescue.py --file 14_05_08.ser
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serlib import SerFile  # noqa: E402
from ecl import paths  # noqa: E402

# Fraction of the frame's peak used to select lit pixels. High enough to exclude
# the glow around the crescent, which is asymmetric and would drag the centroid.
LIT_FRACTION = 0.35


def centroid(ser, index):
    """Intensity-weighted centroid of the lit region, in half-resolution px."""
    raw = np.frombuffer(ser.read_frame(index), dtype=np.uint16)
    a = raw.reshape(ser.height, ser.width)
    # Green channel via one of the two G sites, at half resolution.
    g = a[1::2, 0::2].astype(np.float32)
    thr = g.max() * LIT_FRACTION
    m = g > thr
    if m.sum() < 200:
        return None
    ys, xs = np.nonzero(m)
    w = g[ys, xs] - thr
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=paths.out_dir())
    ap.add_argument("--data", default=paths.data_dir())
    ap.add_argument("--file", default="14_05_08.ser")
    ap.add_argument("--calib", default="14_03_31.ser,14_06_45.ser",
                    help="neighbouring captures where the circle fit works")
    args = ap.parse_args()

    det = json.load(open(os.path.join(args.out, "diag", "centres.json")))
    man = json.load(open(os.path.join(args.out, "segments.json")))
    t0 = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"]) for f in man["files"]}
    fps = {f["name"]: f["fps"] for f in man["files"]}
    epoch = min(t0.values())

    def utc(name, idx):
        return (t0[name] - epoch).total_seconds() + idx / fps[name]

    by = {}
    for c in det["centres"]:
        by.setdefault(c["file"], []).append(c)

    # Calibrate: centroid minus fitted circle centre, on captures where the
    # circle fit is trustworthy.
    calib = []
    for name in args.calib.split(","):
        cs = [c for c in by.get(name, []) if c["method"] == "kasa" and c["arc"] >= 100]
        if len(cs) < 5:
            print("  %s: no trustworthy fits to calibrate on" % name)
            continue
        with SerFile(os.path.join(args.data, name)) as ser:
            dx, dy = [], []
            for c in cs[::max(1, len(cs) // 8)]:
                p = centroid(ser, c["index"])
                if p is None:
                    continue
                dx.append(p[0] - c["cx"])
                dy.append(p[1] - c["cy"])
        if not dx:
            continue
        t = sum(utc(name, c["index"]) for c in cs) / len(cs)
        calib.append((t, float(np.median(dx)), float(np.median(dy))))
        print("  %-14s centroid offset (%+7.2f, %+7.2f) px from %d frames"
              % (name, calib[-1][1], calib[-1][2], len(dx)))

    if len(calib) < 2:
        raise SystemExit("need two calibration captures either side")
    calib.sort()

    # Place the broken capture: centroid, minus the offset interpolated to its own
    # time. Linear because the crescent thins steadily, so the offset does too.
    (ta, ax, ay), (tb, bx, by_) = calib[0], calib[-1]
    target = [c for c in by.get(args.file, [])]
    target.sort(key=lambda c: c["index"])
    print("  %s: %d frames to place" % (args.file, len(target)))

    fixed = 0
    with SerFile(os.path.join(args.data, args.file)) as ser:
        for c in target:
            p = centroid(ser, c["index"])
            if p is None:
                continue
            t = utc(args.file, c["index"])
            f = 0.0 if tb == ta else (t - ta) / (tb - ta)
            ox = ax + (bx - ax) * f
            oy = ay + (by_ - ay) * f
            c["cx"] = p[0] - ox
            c["cy"] = p[1] - oy
            c["method"] = "centroid"
            c["arc"] = 360
            c["n"] = 999
            c["rms"] = 1.0
            fixed += 1

    xs = [c["cx"] for c in target]
    ys = [c["cy"] for c in target]
    print("  placed %d frames; cx %.1f..%.1f  cy %.1f..%.1f"
          % (fixed, min(xs), max(xs), min(ys), max(ys)))

    out = os.path.join(args.out, "diag", "centres.json")
    json.dump(det, open(out, "w"))
    print("  updated %s" % out)


if __name__ == "__main__":
    main()
