"""Slice and stack — replaces run-eclipse.mjs and the PJSR stacker.

Two phases, as before:

  B. slice   cut each segment out of its 22 GB capture      (`ecl.slicer`)
  C. stack   drizzle-stack each slice, one job per channel  (lunation)

The stacker itself needed no porting. `lunation.stack.stacker` is the validated
Python port of the very `pjsr/ser-stack.js` this pipeline was calling by absolute
path, and it deliberately froze that script's config schema, log sentinels and
report field set. The job dicts `run-eclipse.mjs` was already writing —
`{ser, channel, out, log, report, bestFraction, minFrames, maxFrames,
alignOnGradient, localAlign, drizzle, drizzleMargin}` — are accepted as-is.

What disappears with the PixInsight launch: `withPiLaunchLock` (two instances
starting in the same second raced the same instance slot and hung at 0 CPU), the
`-r="script,args"` marshalling, the swap-directory TMP/TEMP juggling, and reading
results back by grepping a log file for `=== STACK OK` because a missing PJSR API
kills a script with exit 0 and no output. A Python call either returns or raises.
"""

import argparse
import json
import os
import time

from .vendor.stack import stacker

from . import slicer

__all__ = ["run_stack_job", "run_dataset", "expected_slice_bytes", "FIELD_WARP"]

# Single-resample drizzle placement, on by default.
#
# The plain ported path interpolates twice per frame — upsample x2, then shift by
# the sub-pixel remainder — and each pass adds its own kernel ringing. The field
# engine folds the shift into the resample so the frame is interpolated once. It
# is gated behind localAlign in lunation, so both are set together.
#
# Measured on 14_14_36_f251 G, 35 frames, against the PJSR reference stack:
#
#   variant             10-90 limb   undershoot   sky noise   secs
#   PixInsight              2.600 px     -2.06%    0.001805     23
#   ported/bicubic          2.000 px     -1.84%    0.001809     37
#   ported/lanczos          2.000 px     -1.89%    0.001810     32
#   stsci/pixfrac 1.0       2.400 px     -1.97%    0.001805     37
#   stsci/pixfrac 0.7       2.300 px     -1.95%    0.001807     32
#   field/localAlign        2.000 px     -0.16%    0.001828     43
#
# Confirmed across the whole dataset (12 stacks, 4 segments x RGB): sharper in
# 12 of 12, median limb 10-90 2.50 px -> 1.95 px, median undershoot -1.42% ->
# -0.10%.
#
# THE NOISE COST IS SEGMENT-DEPENDENT, and it splits by segment length rather
# than at random. Sky noise relative to the PixInsight stack:
#
#   14_13_00_f1170   93 frames    0.75 - 0.92   (better)
#   14_14_36_f251    35 frames    1.01 - 1.01   (neutral)
#   14_14_36_f338   422 frames    1.12 - 1.33   (worse)
#   14_16_14_f0     556 frames    1.09 - 1.19   (worse)
#
# ISOLATED 2026-08-16 on 14_14_36_f338 G, re-stacking the same segment+channel
# with only the placement engine flipped:
#
#   variant        frames   sky noise   10-90 limb   undershoot
#   no field warp     408    0.001062        1.700       -1.37%
#   field warp        422    0.001197        1.800       +0.53%
#   PixInsight        360    0.001067            -            -
#
# Field warp is 12.7% noisier here, and that holds up: it used MORE frames than
# the plain path, which should have reduced noise, so the residual confound
# works against it. The plain path lands within 0.5% of PixInsight's noise.
# (The frame counts still differ, so the limb figures stay confounded and are
# not relied on — frame selection turned out NOT to be independent of alignment
# as assumed.)
#
# So the engine is chosen per segment. Short segments get the ringing win; long
# ones keep the noise. The split in the table above is clean at roughly 200
# stacked frames, with 35 and 93 on the good side and 422 and 556 on the bad.
#
# Same limb acuity as the best of the others, with 11x less ringing. Checked for
# geometric safety, which is
# the thing that would disqualify a local warp on a corona — `corona-drift`
# measures a 0.279 px/s differential rate off these stacks and the radial
# profile is the science: Moon centre moves 0.137 px, radial profile median
# 0.109%, local displacement around the limb maxes at 0.151 px with no spread,
# and there is no 128/256 px tile seam in the difference.
#
# lanczos buys nothing and rings slightly more; both stsci kernels are blunter
# here, independently confirming lunation's own 2026-07-24 default flip.
FIELD_WARP = True

# Stacked-frame count above which the field engine's noise penalty outweighs its
# ringing win. Pass field_warp=True/False to override the per-segment choice.
FIELD_WARP_MAX_FRAMES = 200


def wants_field_warp(seg, default=None):
    """Whether this segment should use single-resample placement.

    Estimates the stacked frame count the way the stacker will: `bestFraction`
    of the segment, capped by `maxFrames`.
    """
    if default is not None:
        return default
    if not FIELD_WARP:
        return False
    n = seg.get("count", 0) * (seg.get("bestFraction") or 0.10)
    n = min(n, seg.get("maxFrames") or 1_000_000)
    return n <= FIELD_WARP_MAX_FRAMES


def expected_slice_bytes(seg, header_bytes=178):
    """Size a finished slice should have, or None if geometry is unknown.

    `run-eclipse.mjs` skipped re-cutting a slice whose size already matched the
    requested frame range — worth keeping, since re-running to redo one channel
    should not re-cut 22 GB off a slow drive.
    """
    g = seg.get("geometry")
    if not g:
        return None
    planes = 3 if g["colorId"] >= 100 else 1
    bpp = 2 if g["depth"] > 8 else 1
    frame_bytes = g["width"] * g["height"] * planes * bpp
    trailer = seg["count"] * 8 if g.get("hasTrailer", True) else 0
    return header_bytes + seg["count"] * frame_bytes + trailer


def _slice_is_current(seg):
    if not os.path.exists(seg["slice"]):
        return False
    want = expected_slice_bytes(seg)
    # Without known geometry, presence is all that can be checked; with it, a
    # size mismatch means the slice was cut for a different frame range.
    return True if want is None else os.path.getsize(seg["slice"]) == want


def run_stack_job(job, log=print):
    """Run one segment x channel stack. Returns the report dict."""
    t0 = time.time()
    ok = stacker.run(dict(job))
    if not ok:
        raise RuntimeError(f"stack failed for {job['out']} — see {job.get('log')}")
    report = {}
    if job.get("report") and os.path.exists(job["report"]):
        with open(job["report"], encoding="utf-8-sig") as f:
            report = json.load(f)
    log(f"  [{os.path.basename(job['out'])}] "
        f"{report.get('stacked', '?')}/{report.get('frames', '?')} frames "
        f"in {time.time() - t0:.1f}s")
    return report


def run_dataset(cfg, channels=None, only=None, slice_only=False,
                stack_only=False, dry=False, field_warp=None, log=print):
    """Slice then stack every selected segment, for every selected channel.

    `field_warp` forces single-resample placement on or off for every segment;
    left as None, each segment is decided by `wants_field_warp`.
    """
    channels = channels or cfg.get("channels") or ["R", "G", "B"]
    segments = cfg["segments"]
    if only:
        segments = [s for s in segments if s["id"] in only]
    if not segments:
        raise ValueError("no segments selected")

    for d in (cfg["sliceDir"], cfg["stackDir"], f"{cfg['outDir']}/logs",
              f"{cfg['outDir']}/configs"):
        os.makedirs(d, exist_ok=True)

    log(f"{cfg.get('name', 'eclipse')}: {len(segments)} segment(s) "
        f"x {len(channels)} channel(s)")

    if not stack_only:
        todo = [s for s in segments if not _slice_is_current(s)]
        log(f"slice: {len(todo)} to cut, {len(segments) - len(todo)} already present")
        for seg in todo:
            if dry:
                log(f"  [dry] {seg['id']} f{seg['start']}+{seg['count']}")
                continue
            slicer.slice_ser(seg["src"], seg["slice"], seg["start"], seg["count"],
                             log=lambda m: log(f"  {m}"))
    if slice_only:
        log("slice-only: done")
        return []

    reports = []
    for seg in segments:
        for ch in channels:
            jid = f"{seg['id']}_{ch}"
            job = {
                "ser": seg["slice"],
                "channel": ch,
                "out": f"{cfg['stackDir']}/{jid}.xisf",
                "log": f"{cfg['outDir']}/logs/{jid}_stack.log",
                "report": f"{cfg['stackDir']}/{jid}.json",
                "bestFraction": seg.get("bestFraction"),
                "minFrames": seg.get("minFrames"),
                "maxFrames": seg.get("maxFrames"),
                "alignOnGradient": seg.get("alignOnGradient"),
                "drizzle": seg.get("drizzle"),
                "drizzleMargin": seg.get("drizzleMargin"),
            }
            # The segment configs carry `localAlign: false` from the PJSR era,
            # when local alignment meant tile warping and nothing else. It now
            # also gates single-resample placement, so it is decided here rather
            # than read from the segment — see FIELD_WARP for the measurements
            # and why the choice is per-segment.
            if wants_field_warp(seg, field_warp):
                job["localAlign"] = True
                job["drizzleFieldWarp"] = True
            else:
                job["localAlign"] = seg.get("localAlign")
            # The config file is still written: it is the record of what was run,
            # and lunation's CLI can replay it directly.
            cfg_path = f"{cfg['outDir']}/configs/{jid}.json"
            if not dry:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(job, f, indent=2)
            if dry:
                log(f"  [dry] stack {jid}")
                continue
            log(f"  [{jid}] stacking")
            reports.append(run_stack_job(job, log=log))
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="S:/solar-eclipse/out/configs/eclipse.json")
    ap.add_argument("--channels", default=None, help="comma-separated, e.g. R,G,B")
    ap.add_argument("--only", default=None, help="comma-separated segment ids")
    ap.add_argument("--slice-only", action="store_true")
    ap.add_argument("--stack-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-field-warp", action="store_true",
                    help="use the plain ported drizzle placement (two "
                         "interpolations per frame) instead of the "
                         "single-resample field engine")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)

    run_dataset(cfg,
                channels=args.channels.split(",") if args.channels else None,
                only=args.only.split(",") if args.only else None,
                slice_only=args.slice_only, stack_only=args.stack_only,
                dry=args.dry,
                field_warp=False if args.no_field_warp else None)


if __name__ == "__main__":
    main()
