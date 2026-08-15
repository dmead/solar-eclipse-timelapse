"""Stage A — measure the capture and segment it by exposure state.

A single eclipse SER can contain several different exposure states: the white-light
filter comes off mid-file and the exposure is then ridden down by hand. Stacking a
whole file would blend all of that together, so this builds a sampled light curve,
finds the state changes, and writes a frame-range manifest the later stages consume.

Nothing here is tuned to a particular file. Change points are found by comparing
each step against the spread of the series' own steps, so a uniform capture yields
exactly one segment and a file with a filter change yields a boundary where the
brightness actually moves.

    python scan_ser.py --data S:/solar-eclipse/Sun --out S:/solar-eclipse/out

Outputs `lightcurve.json` (cached; delete to force a rescan), `segments.json` and a
human-readable `segments.txt`.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

from serlib import SerFile

# Sampling: every Nth frame, and this many evenly spaced rows within each frame.
DEFAULT_STEP = 20
DEFAULT_ROWS = 24

# A segment must last at least this long to count as a stable state; shorter runs
# are the exposure being ridden and get merged into a `transition` segment.
MIN_SEGMENT_S = 2.0

# Change-point sensitivity. A boundary needs a step in log2(median) larger than
# both this floor (a ~27% brightness change) and several times the typical step
# size of the series itself.
CP_FLOOR_LOG2 = 0.35
CP_MAD_MULT = 6.0

# Exposure clustering: stable segments whose log2(median) differ by more than this
# belong to different exposure levels.
LEVEL_GAP_LOG2 = 0.5


def log2s(v):
    return math.log(max(v, 1), 2)


def mad(xs):
    """Median absolute deviation — robust spread, no numpy."""
    if not xs:
        return 0.0
    s = sorted(xs)
    m = s[len(s) // 2]
    d = sorted(abs(x - m) for x in xs)
    return d[len(d) // 2]


# ---------------------------------------------------------------- light curve

def sample_file(path, step, nrows, log):
    """Sample one SER into a list of per-frame stat dicts."""
    with SerFile(path) as ser:
        rows = [r * ser.height // nrows for r in range(nrows)]
        ts = ser.timestamps()
        t0 = ts[0]
        idx = list(range(0, ser.frame_count, step))
        if idx[-1] != ser.frame_count - 1:
            idx.append(ser.frame_count - 1)

        pts = []
        for i in idx:
            st = ser.sample_stats(i, rows)
            st["i"] = i
            st["t"] = (ts[i] - t0).total_seconds()
            pts.append(st)

        log("  %s: %d frames, %d samples, %.1fs @ %.2ffps" % (
            os.path.basename(path), ser.frame_count, len(pts),
            ser.duration(), ser.fps()))
        return {
            "path": path.replace("\\", "/"),
            "name": os.path.basename(path),
            "width": ser.width, "height": ser.height, "depth": ser.depth,
            "color_id": ser.color_id, "frame_count": ser.frame_count,
            "max_value": ser.max_value,
            "t0_utc": ts[0].isoformat(), "t1_utc": ts[-1].isoformat(),
            "duration": ser.duration(), "fps": ser.fps(),
            "step": step, "rows": nrows,
            "points": pts,
        }


def build_lightcurve(data_dir, out_dir, step, nrows, log):
    cache = os.path.join(out_dir, "lightcurve.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            lc = json.load(fh)
        if lc.get("step") == step and lc.get("rows") == nrows:
            log("using cached light curve (%d files)" % len(lc["files"]))
            return lc
        log("cache parameters changed, rescanning")

    paths = sorted(glob.glob(os.path.join(data_dir, "*.ser")))
    if not paths:
        raise SystemExit("no .ser files under %s" % data_dir)
    log("scanning %d files, every %dth frame, %d rows/frame" % (len(paths), step, nrows))

    started = time.time()
    files = [sample_file(p, step, nrows, log) for p in paths]
    lc = {"step": step, "rows": nrows, "data_dir": data_dir.replace("\\", "/"),
          "files": files}
    with open(cache, "w") as fh:
        json.dump(lc, fh)
    log("light curve built in %.1f min -> %s" % ((time.time() - started) / 60, cache))
    return lc


# ------------------------------------------------------------- change points

def find_boundaries(pts):
    """Sample indices where the brightness state changes.

    Returns positions into `pts` at which a new segment begins.
    """
    if len(pts) < 3:
        return []
    series = [log2s(p["med"]) for p in pts]
    steps = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    thr = max(CP_FLOOR_LOG2, CP_MAD_MULT * mad(steps))
    return [i for i in range(1, len(series)) if steps[i - 1] > thr]


def refine_boundary(ser, rows, lo, hi, lo_med, hi_med):
    """Binary-search the exact frame where the brightness crosses the midpoint.

    Sampling every 20th frame only locates a transition to within 20 frames; this
    narrows it to the frame, for ~5 extra reads instead of 20.
    """
    target = (log2s(lo_med) + log2s(hi_med)) / 2.0
    rising = hi_med > lo_med
    while hi - lo > 1:
        mid = (lo + hi) // 2
        m = log2s(ser.sample_stats(mid, rows)["med"])
        if (m >= target) == rising:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------- segmentation

def segment_file(fdesc, log):
    """Split one file into frame ranges of constant exposure state."""
    pts = fdesc["points"]
    maxv = fdesc["max_value"]
    bounds = find_boundaries(pts)

    # Refine each sampled boundary down to an exact frame index.
    exact = []
    with SerFile(fdesc["path"]) as ser:
        rows = [r * ser.height // fdesc["rows"] for r in range(fdesc["rows"])]
        for b in bounds:
            lo, hi = pts[b - 1], pts[b]
            exact.append(refine_boundary(ser, rows, lo["i"], hi["i"],
                                         lo["med"], hi["med"]))

    edges = [0] + sorted(set(exact)) + [fdesc["frame_count"]]
    fps = fdesc["fps"] or 1.0
    blown_level = 0.5 * maxv

    segs = []
    for k in range(len(edges) - 1):
        start, end = edges[k], edges[k + 1]
        if end <= start:
            continue
        inside = [p for p in pts if start <= p["i"] < end]
        if not inside:
            inside = [min(pts, key=lambda p: abs(p["i"] - start))]
        med = sorted(p["med"] for p in inside)[len(inside) // 2]
        segs.append({
            "start": start, "count": end - start,
            "seconds": (end - start) / fps,
            "med": med,
            "p99": sorted(p["p99"] for p in inside)[len(inside) // 2],
            "lit": round(sorted(p["lit"] for p in inside)[len(inside) // 2], 5),
            "satfrac": round(sorted(p["satfrac"] for p in inside)[len(inside) // 2], 5),
            "blown": med >= blown_level,
        })

    return merge_short(segs, fdesc, fps)


def merge_short(segs, fdesc, fps):
    """Collapse sub-threshold runs into `transition` segments.

    While the operator rides the exposure down after pulling the filter, every
    sample differs from the last. Those frames belong to no stable state, so they
    are grouped rather than emitted as a dozen one-second segments.
    """
    out = []
    for s in segs:
        short = s["seconds"] < MIN_SEGMENT_S and not s["blown"]
        if short and out and out[-1].get("kind") == "transition":
            prev = out[-1]
            prev["count"] = s["start"] + s["count"] - prev["start"]
            prev["seconds"] = prev["count"] / fps
            prev["med"] = max(prev["med"], s["med"])
            continue
        s["kind"] = "transition" if short else ("blown" if s["blown"] else "stable")
        out.append(s)
    return out


def classify(files_segs, log):
    """Label segments filtered/unfiltered and assign exposure levels.

    Totality is bracketed physically rather than by a brightness threshold: pulling
    the filter while the exposure is still set for the photosphere blows the frame
    out, and the sun reappearing at third contact blows it out again. Everything
    between the first and last blowout is unfiltered.
    """
    flat = []
    for f in files_segs:
        for s in f["segments"]:
            flat.append((f, s))

    blown_at = [i for i, (_, s) in enumerate(flat) if s["kind"] == "blown"]
    if blown_at:
        first, last = blown_at[0], blown_at[-1]
        log("totality bracket: %s f%d -> %s f%d" % (
            flat[first][0]["name"], flat[first][1]["start"],
            flat[last][0]["name"], flat[last][1]["start"]))
    else:
        first = last = -1
        log("WARNING: no blown-out transition found; treating all as filtered")

    for i, (_, s) in enumerate(flat):
        s["state"] = "unfiltered" if first <= i <= last else "filtered"

    # Cluster the stable unfiltered segments into exposure levels by background
    # median, splitting wherever a gap in log space exceeds the threshold.
    corona = [s for _, s in flat if s["state"] == "unfiltered" and s["kind"] == "stable"]
    ordered = sorted(corona, key=lambda s: s["med"])
    level = 0
    for i, s in enumerate(ordered):
        if i and log2s(s["med"]) - log2s(ordered[i - 1]["med"]) > LEVEL_GAP_LOG2:
            level += 1
        s["level"] = level
    names = {0: "short", 1: "mid", 2: "long"} if level == 2 else {}
    for s in ordered:
        s["level_name"] = names.get(s["level"], "L%d" % s["level"])
    log("exposure levels among corona segments: %d" % (level + 1))
    return flat


# --------------------------------------------------------------------- report

def write_report(files_segs, out_dir, log):
    lines = ["%-14s %-6s %-7s %-6s %-11s %-6s %6s %8s %8s" % (
        "file", "start", "count", "sec", "state", "kind", "level", "med", "p99"), "-" * 88]
    for f in files_segs:
        for s in f["segments"]:
            lines.append("%-14s %-6d %-7d %-6.1f %-11s %-6s %6s %8d %8d" % (
                f["name"], s["start"], s["count"], s["seconds"], s["state"],
                s["kind"], s.get("level_name", "-"), s["med"], s["p99"]))
    txt = os.path.join(out_dir, "segments.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s (%d segments)" % (txt, sum(len(f["segments"]) for f in files_segs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="S:/solar-eclipse/Sun")
    ap.add_argument("--out", default="S:/solar-eclipse/out")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logpath = os.path.join(args.out, "scan.log")
    logfh = open(logpath, "w")

    def log(msg):
        logfh.write(msg + "\n")
        logfh.flush()

    lc = build_lightcurve(args.data, args.out, args.step, args.rows, log)

    files_segs = []
    for f in lc["files"]:
        files_segs.append({
            "name": f["name"], "path": f["path"],
            "frame_count": f["frame_count"], "fps": f["fps"],
            "t0_utc": f["t0_utc"], "t1_utc": f["t1_utc"],
            "color_id": f["color_id"], "width": f["width"], "height": f["height"],
            "depth": f["depth"],
            "segments": segment_file(f, log),
        })

    classify(files_segs, log)

    manifest = os.path.join(args.out, "segments.json")
    with open(manifest, "w") as fh:
        json.dump({"data_dir": lc["data_dir"], "files": files_segs}, fh, indent=1)
    write_report(files_segs, args.out, log)
    logfh.close()

    print("segments -> %s" % manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
