"""Find the small-bead window, before anything else in Stage F has run.

The diamond ring and Baily's beads are both clipped photosphere, so the amount of
clipping cannot tell them apart - it falls smoothly through both. Measured on
14_13_00, the clipped area goes 56453 px at f1090 to 5 px at f1300 without a step
anywhere in it.

SHAPE separates them. The diamond ring is ONE connected blob; beads are several
smaller ones with lunar ridges between. So the discriminator is the fraction of
the clipped area sitting in its largest connected component:

    raw    area  blobs  largest  largest%
    1090  56453      1    56413      100     one solid blob - diamond ring
    1168    419      3      391       93
    1204    136      2       72       53     broken up - beads
    1240     69      2       18       26     small beads, ridges between
    1300      5      0        1       20     gone

Connected components need no disc centre, which is what lets this run BEFORE
`tl_centres` - and it has to, because `gen_timelapse` decides screen time and
runs first of all.

Only captures that contain a filter change are scanned. Beads exist for a couple
of seconds either side of a contact and nowhere else, and scanning all of
totality would read three thousand frames to find nothing in 90% of them.

    python -m ecl.beadwindow --out S:/solar-eclipse/out --data Z:/solar-eclipse/Sun

Writes `diag/beads.json`. `gen_timelapse` reads it if present and falls back to
its saturation heuristic if not.
"""

import argparse
import json
import os

import cv2
import numpy as np

from .source import open_source

__all__ = ["scan_capture", "find_window", "main"]

# Fraction of full scale counted as clipped.
SAT = 0.90

# A bead frame: enough clipped area to see, and not dominated by one blob.
#
# Both bounds are set from the table above. Below MIN_AREA the clipped region is
# a handful of specks - f1258 is 36 px spread over two components, which renders
# as nothing. Above MAX_BLOB_FRAC one component holds nearly all the light, which
# is the ring rather than beads: it sits at 100% for the whole approach, breaks
# through 83% at f1180 and is under 55% from f1204 on.
MIN_AREA = 40
MAX_BLOB_FRAC = 0.70

# Components smaller than this are read noise and hot pixels, not beads.
MIN_BLOB = 8

# Raw frames a window may jump to stay one window.
MAX_GAP = 24


def tune(out_dir, log=None):
    """Resolve the bead thresholds from the config against the surveyed radius.

    MIN_AREA and MIN_BLOB are areas and scale as r^2, so a sensor with twice the
    disc radius needs a four times larger bead before it means the same thing.
    """
    global MIN_AREA, MIN_BLOB, SAT, MAX_BLOB_FRAC
    from .params import load

    P = load(out_dir, create=False)
    MIN_AREA = max(1.0, P.area("beads.min_area_r2"))
    MIN_BLOB = max(1.0, P.area("beads.min_blob_r2"))
    SAT = P.get("beads.sat", SAT)
    MAX_BLOB_FRAC = P.get("beads.max_blob_frac", MAX_BLOB_FRAC)
    if log:
        log(f"  tuned to r={P.radius_px:.0f}px: min bead {MIN_AREA:.0f} px2, "
            f"min blob {MIN_BLOB:.0f} px2")
    return P


def scan_capture(path, lo, hi, step=2, log=print):
    """[(index, area, n_blobs, largest_frac)] over raw frames [lo, hi)."""
    rows = []
    with open_source(path) as ser:
        hi = min(hi, ser.frame_count)
        for i in range(lo, hi, step):
            g = ser.green(i) * 65535.0
            m = (g >= SAT * 65535.0).astype(np.uint8)
            if not m.any():
                rows.append((i, 0, 0, 0.0))
                continue
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]),
                           reverse=True)
            tot = sum(areas)
            big = areas[0] if areas else 0
            rows.append((i, tot, sum(1 for a in areas if a >= MIN_BLOB),
                         big / tot if tot else 0.0))
    return rows


def find_window(rows):
    """Longest run of bead-like frames: (lo, hi) inclusive, or None."""
    ok = [r for r in rows if r[1] >= MIN_AREA and r[3] <= MAX_BLOB_FRAC]
    if not ok:
        return None
    runs, cur = [], [ok[0]]
    for r in ok[1:]:
        if r[0] - cur[-1][0] <= MAX_GAP:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    best = max(runs, key=lambda g: g[-1][0] - g[0][0])
    return (best[0][0], best[-1][0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="S:/solar-eclipse/out")
    ap.add_argument("--data", default="S:/solar-eclipse/Sun")
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args(argv)
    tune(args.out, log=print)

    with open(os.path.join(args.out, "segments.json")) as fh:
        man = json.load(fh)

    out = {}
    for f in man["files"]:
        states = {s["state"] for s in f["segments"]}
        if len(states) < 2:
            continue                       # no filter change here
        unf = [s for s in f["segments"] if s["state"] == "unfiltered"]
        if not unf:
            continue
        lo = min(s["start"] for s in unf)
        hi = max(s["start"] + s["count"] for s in unf)
        print(f"{f['name']}: scanning unfiltered f{lo}-{hi}")
        rows = scan_capture(os.path.join(args.data, f["name"]), lo, hi, args.step)
        w = find_window(rows)
        if not w:
            print("  no bead window found")
            continue
        inside = [r for r in rows if w[0] <= r[0] <= w[1]]
        out[f["name"]] = {
            "lo": w[0], "hi": w[1],
            "frames": w[1] - w[0] + 1,
            "seconds": (w[1] - w[0] + 1) / f["fps"],
            "area": [min(r[1] for r in inside), max(r[1] for r in inside)],
            "blob_frac": [round(min(r[3] for r in inside), 2),
                          round(max(r[3] for r in inside), 2)],
        }
        d = out[f["name"]]
        print(f"  beads f{w[0]}-{w[1]}  {d['frames']} raw frames "
              f"({d['seconds']:.2f} s), area {d['area'][0]}-{d['area'][1]} px, "
              f"largest blob {d['blob_frac'][0]:.0%}-{d['blob_frac'][1]:.0%}")

    dest = os.path.join(args.out, "diag", "beads.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
