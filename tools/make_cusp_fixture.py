"""Regenerate the cusp-smoothing fixtures in tests/data from a finished run.

    python tools/make_cusp_fixture.py --out <run-dir> --data <captures> \
        --dest tests/data --name cusp_track_2024.json [--truth]

`tests/test_cusp_smoothing.py` guards `fit_cusp_track` against two opposite
failures, and it needs real measurements to do it: the smoother's whole job is
to leave accurate per-frame answers where they are, and no synthetic series
proves that on the data anyone actually shipped.

What is stored is ANGLES, one pair per frame, which is why a fixture measured
from hundreds of gigabytes of SER and XISF is a few tens of kilobytes. Nothing
here is a picture and nothing here is reversible into one.

Two things go in each file:

  rows       find_cusps' answer per filtered frame - the smoother's INPUT
  reference  fit_cusp_track's answer, recorded at the time the fixture was
             made, so a later change that moves it has to say why
  truth      (--truth) the same cusps read off the frames by a deliberately
             different method: one circle at 0.97 R, nearest-neighbour
             sampled, ends of the longest lit run. It shares no code with
             find_cusps' multi-radius band and sub-pixel edge walk, so a fault
             common to both would have to be a coincidence rather than a
             shared assumption. Only worth generating where the cusps sweep -
             the 2024 captures are 30-60 s each and barely move.

The 2024 fixture was made from a rebuild of the published run, verified against
its documented figures first: survey radius 291.8 px, rSun 279.0, rMoon 292,
2299 frames, a 1180x880 window.
"""

import argparse
import json
import math
import os

import numpy as np

from ecl import gen_insets as gi
from ecl.source import open_source

# Angles are rounded to this many places: 1e-6 rad is 0.0003 px at r=300, which
# is four orders below any tolerance the tests assert, and it roughly halves
# the file.
ROUND = 6


def measure_rows(out_dir, cfg_path, log=print):
    """find_cusps on every filtered frame of a run, grouped by capture."""
    gi.tune(out_dir, log=lambda *_a: None)
    with open(cfg_path, encoding="utf-8-sig") as fh:
        cfg = json.load(fh)
    with open(os.path.join(out_dir, "diag", "centres.json"),
              encoding="utf-8-sig") as fh:
        r_sun = float(json.load(fh)["rSun"])

    cache = [None, None]

    def plane(f):
        if cache[0] != f["src"]:
            if cache[1] is not None:
                cache[1].close()
            cache[1] = open_source(f["src"])
            cache[0] = f["src"]
        return gi.green_plane(cache[1], f["index"])

    per = {}
    try:
        for f in cfg["frames"]:
            if f["state"] != "filtered":
                continue
            cu = gi.find_cusps(plane(f), f["cx"], f["cy"], r_sun)
            if cu is None:
                continue
            a, b, _mid, _opens = cu
            if a[1] > b[1]:                       # upper first, by image position
                a, b = b, a
            per.setdefault(f["file"], []).append([
                f["index"],
                round(math.atan2(a[1] - f["cy"], a[0] - f["cx"]), ROUND),
                round(math.atan2(b[1] - f["cy"], b[0] - f["cx"]), ROUND)])
    finally:
        if cache[1] is not None:
            cache[1].close()
    log(f"  measured cusps on {sum(len(v) for v in per.values())} frames "
        f"of {len(per)} capture(s)")
    return r_sun, {k: sorted(v) for k, v in per.items()}, cfg


def pixel_truth(cfg, r_sun, every, log=print):
    """Cusp angles from a single-radius scan - a second opinion, not a copy."""
    N = 2880
    th = np.arange(N)*(2*math.pi/N)
    cache = [None, None]
    truth = {}
    try:
        for f in cfg["frames"][::every]:
            if f["state"] != "filtered":
                continue
            if cache[0] != f["src"]:
                if cache[1] is not None:
                    cache[1].close()
                cache[1] = open_source(f["src"])
                cache[0] = f["src"]
            g = gi.green_plane(cache[1], f["index"])
            h, w = g.shape
            xs = f["cx"] + 0.97*r_sun*np.cos(th)
            ys = f["cy"] + 0.97*r_sun*np.sin(th)
            if xs.min() < 0 or ys.min() < 0 or xs.max() > w-1 or ys.max() > h-1:
                continue
            v = g[np.round(ys).astype(int), np.round(xs).astype(int)]
            lit = v > 0.5*np.percentile(v, 95)
            if lit.all() or not lit.any():
                continue
            ii = np.nonzero(~lit)[0]
            gaps = np.diff(np.concatenate([ii, ii[:1] + N])) - 1
            k = int(np.argmax(gaps))
            if gaps[k] < 20:                     # too short an arc to locate
                continue
            truth[str(f["index"])] = [
                round(float(th[(ii[k] + 1) % N]), ROUND),
                round(float(th[(ii[k] + gaps[k]) % N]), ROUND)]
    finally:
        if cache[1] is not None:
            cache[1].close()
    log(f"  independent truth on {len(truth)} frames")
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="a finished run directory")
    ap.add_argument("--config", default=None,
                    help="config to read (default: the run's timelapse.json). "
                         "Point this at a pre-insets copy if the run has "
                         "already been through gen_insets.")
    ap.add_argument("--dest", default="tests/data")
    ap.add_argument("--name", required=True, help="output file name")
    ap.add_argument("--note", default="")
    ap.add_argument("--truth", action="store_true",
                    help="also record the independent single-radius reading")
    ap.add_argument("--truth-every", type=int, default=4,
                    help="sample every Nth frame for the truth set")
    args = ap.parse_args(argv)

    cfg_path = args.config or os.path.join(args.out, "configs", "timelapse.json")
    r_sun, per, cfg = measure_rows(args.out, cfg_path)

    caps = []
    for name in sorted(per):
        rows = per[name]
        if len(rows) < gi.CUSP_MIN_FRAMES:
            print(f"  {name}: only {len(rows)} measurements - skipped")
            continue
        deg = 2 if len(rows) >= gi.CUSP_QUAD_MIN else 1
        # Tolerate both arities on purpose, so this can be pointed at an older
        # checkout to capture what THAT produced. A reference generated by the
        # same code the test guards would only assert that the code reproduces
        # itself; the 2024 file was made against the whole-capture polynomial.
        got = gi.fit_cusp_track([tuple(r) for r in rows], deg)
        fitted = got[0]
        if fitted is None:
            continue
        caps.append({"name": name, "deg": deg, "rows": rows,
                     "reference": [[round(v, ROUND) for v in p] for p in fitted]})

    doc = {"note": args.note, "r_sun_plane_px": round(r_sun, 4)}
    if len(caps) == 1 and args.truth:
        doc.update({"capture": caps[0]["name"], "deg": caps[0]["deg"],
                    "rows": caps[0]["rows"], "reference": caps[0]["reference"],
                    "truth": pixel_truth(cfg, r_sun, args.truth_every)})
    else:
        doc["captures"] = caps
        if args.truth:
            doc["truth"] = pixel_truth(cfg, r_sun, args.truth_every)

    os.makedirs(args.dest, exist_ok=True)
    path = os.path.join(args.dest, args.name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB, "
          f"{len(caps)} capture(s))")


if __name__ == "__main__":
    main()
