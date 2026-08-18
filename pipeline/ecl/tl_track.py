"""Measure pointing during totality by correlating the corona against itself —
port of tl-corona-track.js.

The disc fit that stabilises the partial phases has nothing to fit during
totality: there is no photosphere. `tl_centres` falls back to scoring candidate
centres on the gradient around the Moon's limb, and that scatters about 7 px with
a lag-1 autocorrelation of only +0.2 to +0.5 — noise rather than motion — so
`smooth_track.py` refuses to follow it and places totality on a straight line.
The line removes the drift. It cannot remove the mount's actual wobble, and what
it leaves behind is visible in the video, magnified threefold inside the zoom
panels: measured against this pass, the line sits 0.74 px RMS from where the
picture says the frame is.

Phase correlation does not have that problem. The frame is not a smooth blob: it
carries the lunar limb, a hard edge right around the disc, and the prominences on
it. That is exactly what the drizzle already relies on to stack twenty raw frames
inside one group — this measures the same quantity BETWEEN groups.

Every frame of a capture is measured against ONE reference rather than against its
predecessor. A chain of frame-to-frame shifts accumulates its own errors, and over
seventy frames a small bias becomes a drift that looks exactly like the thing
being corrected. A common reference cannot accumulate. The mount was nudged
between captures, so no reference is carried across one.

The reference is the middle frame of the capture's LONGEST constant-exposure run,
not simply its first frame. Phase correlation is indifferent to a change of gain,
because it throws away magnitude and keeps only phase — but it is not indifferent
to CLIPPING, which is a change of structure rather than of scale, and the inner
corona clips at the long exposures.
"""

import argparse
import json
import os
import time
from collections import Counter, OrderedDict

from lunation.core.fftreg import PhaseCorrelator

from . import serio
from .source import open_source

# A shift larger than this is a failed correlation, not the mount. The mount
# drifts ~40 px over a 60 s capture, so this has to be generous.
MAX_SHIFT_PX = 80.0

# "skimage" is an upsampled-DFT estimator good to ~0.01 px; "ported" is the
# literal fftalign.jsh port, which lunation's own tests hold only to 0.75 px.
# The quantity this pass exists to measure is a 0.74 px RMS residual, so the
# exact port is not accurate enough to measure it — default to the better one.
DEFAULT_ENGINE = "skimage"

__all__ = ["track_corona", "DEFAULT_ENGINE", "MAX_SHIFT_PX"]


def _unpack(shift):
    """PhaseCorrelator.evaluate returns a dict or a pair depending on engine."""
    if isinstance(shift, dict):
        return float(shift["dx"]), float(shift["dy"])
    dx, dy = shift
    return float(dx), float(dy)


def _reference_frame(rows):
    """Middle frame of the capture's modal exposure level."""
    keys = [str(int(round(r.get("gain", 1.0) * 1e4))) for r in rows]
    counts = Counter(keys)
    # Strict > on first encounter, matching the original's tie-breaking.
    best_key, best_n = None, 0
    for k in keys:
        if counts[k] > best_n:
            best_key, best_n = k, counts[k]
    run = [r for r, k in zip(rows, keys) if k == best_key]
    return run[len(run) // 2], best_n


def track_corona(frames, engine=DEFAULT_ENGINE, log=print):
    """Correlate every distinct totality frame against its capture's reference."""
    t0 = time.time()

    # One entry per DISTINCT raw frame. The prominence level is resampled with
    # repeats to hold it on screen, and correlating the same frame twice is
    # wasted work — the answer cannot differ.
    # The second-contact resolve is excluded even though it is unfiltered.
    #
    # This pass measures the corona's pointing by correlating one totality frame
    # against another, which assumes the two carry the same structure. A resolve
    # frame does not: it is a saturated plateau covering up to a third of the
    # limb, and correlating a plateau against a corona returns the offset between
    # two blobs. Including them moved the whole totality run 160 px - every
    # measurement after them inherits the error, because the shifts are chained.
    #
    # They lose nothing by it. Their positions come from the modelled track in
    # smooth_track.py, interpolated across a span the corona frames either side
    # already pin down to a fraction of a pixel.
    seen = set()
    want = []
    for fr in frames:
        if fr["state"] != "unfiltered" or fr.get("resolve"):
            continue
        key = (fr["file"], fr["index"])
        if key in seen:
            continue
        seen.add(key)
        want.append(fr)
    log(f"corona track: {len(want)} distinct totality frames  [engine={engine}]")

    by_file = OrderedDict()
    for fr in want:
        by_file.setdefault(fr["file"], []).append(fr)

    results = []
    failed = done = 0

    for name, rows in by_file.items():
        rows.sort(key=lambda r: r["index"])
        ref, best_n = _reference_frame(rows)
        log(f"  {name}: {len(rows)} frames, reference f{ref['index']} "
            f"(level {ref.get('gain')}, {best_n} frames)")

        with open_source(rows[0]["src"]) as ser:
            aligner = PhaseCorrelator(use_gradient=False, engine=engine)
            aligner.initialize(ser.green(ref["index"]))

            for fr in rows:
                if fr["index"] == ref["index"]:
                    results.append({"file": fr["file"], "index": fr["index"],
                                    "dx": 0.0, "dy": 0.0, "ok": True})
                else:
                    dx, dy = _unpack(aligner.evaluate(ser.green(fr["index"])))
                    ok = abs(dx) <= MAX_SHIFT_PX and abs(dy) <= MAX_SHIFT_PX
                    if not ok:
                        failed += 1
                    results.append({"file": fr["file"], "index": fr["index"],
                                    "dx": dx if ok else 0.0,
                                    "dy": dy if ok else 0.0, "ok": ok})
                done += 1
                if done % 100 == 0:
                    log(f"  {done}/{len(want)}")

    log(f"  {len(results)} measured, {failed} rejected, "
        f"in {(time.time() - t0) / 60:.1f} min")
    return {"shifts": results, "maxShift": MAX_SHIFT_PX, "engine": engine}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="S:/solar-eclipse/out/configs/timelapse.json")
    ap.add_argument("--out", default="S:/solar-eclipse/out/diag/corona_track.json")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, choices=["skimage", "ported"])
    ap.add_argument("--data-dir", default=None,
                    help="read the captures from here instead of the paths "
                         "baked into the config")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    serio.restage(cfg, args.data_dir)

    out = track_corona(cfg["frames"], engine=args.engine)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {args.out}  {len(out['shifts'])} shifts")


if __name__ == "__main__":
    main()
