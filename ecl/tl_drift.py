"""Where the Sun is during totality — measured from the PARTIAL phases.

    python -m ecl.tl_drift --out <out> --data <captures>

WHY THIS PASS EXISTS. The disc fit has nothing to fit during totality: there is
no photosphere, so `tl_centres` falls back to the Moon's limb and `tl_track`
correlates a frame whose dominant feature is that same limb. Both measure the
MOON. The partial phases either side measure the SUN. Those are different
reference frames — the Moon crosses the corona at its own synodic rate — so
placing totality on the Moon leaves the corona walking across the frame and the
video stepping sideways at each phase boundary.

`smooth_track` has always had the correction and has never been able to apply it.
It read the rate from `final/drift.json`, written by the CORONA pipeline, which
the timelapse path does not run; the read sat inside a bare `except`, so the term
silently evaluated to zero on every run that has ever been made.

WHY NOT MEASURE IT DURING TOTALITY, as the corona pipeline does. That was tried
first and it is the wrong place to stand. The differential over one 60 s capture
is about 8 px, and it has to be separated from 40-66 px of mount drift, so the
Moon must be registered to roughly a pixel. A full-frame phase correlation will
not do it: the brightest, most structured thing in a totality frame is the inner
corona, which is fixed to the SUN, so the correlation locks onto a mixture of
the two and returns a rate 75% above what the Moon can physically manage. The
corona pipeline avoids this only because it has a separate registration pass.

MEASURING IT FROM THE PARTIAL PHASES INVERTS EVERY ONE OF THOSE PROBLEMS. The
Sun is directly tracked there, so the answer arrives in exactly the frame
totality needs to be placed in. The Moon's centre comes from the terminator it
casts on the disc, which needs no ephemeris and which `gen_insets` already fits.
And the baseline is the whole eclipse rather than one capture: the same rate
that hides under 8 px of totality shows up as hundreds of pixels of accumulated
offset.

The fit never sees totality, so extrapolating into it is a real prediction, and
the geometry checks it: an eclipse is total only while the centres are closer
than r_moon - r_sun. On the 2024-04-08 data a track fitted to the partial phases
alone passes within 1.3 px of the Sun's centre and predicts 225.9 s of totality,
against the 213.2 s the light curve measured independently. A track that
predicts the eclipse it was not shown can be trusted to say where the Sun is
inside it.

The output is a POSITION, not just a rate. A rate alone fixes the drift across
totality and leaves the step at the boundary, because it says nothing about
where the line should sit. The fitted offset carries its intercept, so totality
lands continuous with the partial phases by construction.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

from . import paths, serio

__all__ = ["measure_track", "moon_offset", "main"]

# Frames sampled for the fit. Each one is a disc read and a terminator fit, and
# the relative motion is very nearly linear over 46 minutes, so more samples buy
# precision slowly. The residual reports whether linearity held.
N_SAMPLES = 40

# The fit is rejected unless extrapolating it produces the eclipse that actually
# happened: a closest approach inside r_moon - r_sun, and a predicted duration
# within this fraction of the one the light curve measured.
DURATION_TOLERANCE = 0.25


def moon_offset(track, t):
    """Moon centre minus Sun centre at time t, in plane px."""
    return (float(np.polyval(track["dx"], t)),
            float(np.polyval(track["dy"], t)))


def _utc_of(cfg, man):
    """Seconds since the first capture, per frame, keyed by (file, index)."""
    fps_of = {f["name"]: f["fps"] for f in man["files"]}
    t0 = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"])
          for f in man["files"]}
    epoch = min(t0.values())
    out = {}
    for f in cfg["frames"]:
        out[(f["file"], f["index"])] = (
            (t0[f["file"]] - epoch).total_seconds()
            + f["index"] / fps_of[f["file"]])
    return out


def measure_track(out_dir, cfg, centres, man, log=print):
    """Fit the Moon's offset from the Sun over the partial phases."""
    from . import gen_insets as gi

    gi.tune(out_dir, log=lambda *_a: None)
    r_sun = float(centres.get("rSun") or 279.0)
    r_moon = float(centres.get("rMoon") or 292.0)
    utc = _utc_of(cfg, man)

    # The SUN's centre comes from the detections, not from the smoothed model.
    # This pass runs BEFORE the smoother - the smoother is what needs its answer
    # - and during the partial phases the raw disc fits are the well conditioned
    # case anyway.
    det = {(c["file"], c["index"]): c for c in centres.get("centres", [])}
    frames = [f for f in cfg["frames"] if f["state"] == "filtered"
              and (f["file"], f["index"]) in det]
    if len(frames) < 3 * N_SAMPLES:
        log("only %d usable partial frames - too few to fit a track"
            % len(frames))
        return None

    step = max(1, len(frames) // N_SAMPLES)
    ts, ox, oy, bad = [], [], [], 0
    for f in frames[::step]:
        d = det[(f["file"], f["index"])]
        try:
            r = gi.fit_moon(f, d["cx"], d["cy"], r_sun, r_moon)
        except Exception as e:                              # noqa: BLE001
            log("  %s f%d: %s" % (f["file"], f["index"], e))
            bad += 1
            continue
        if r is None or r[2] > gi.MOON_FIT_MAX_RMS:
            bad += 1
            continue
        ts.append(utc[(f["file"], f["index"])])
        ox.append(r[0] - d["cx"])
        oy.append(r[1] - d["cy"])

    log("terminator fitted on %d of %d sampled frames" % (len(ts), len(ts) + bad))
    if len(ts) < 6:
        log("not enough terminator fits to trust a track")
        return None

    t = np.array(ts)
    dx, dy = np.polyfit(t, np.array(ox), 1), np.polyfit(t, np.array(oy), 1)
    resid = float(np.sqrt(np.mean(
        (np.array(ox) - np.polyval(dx, t)) ** 2
        + (np.array(oy) - np.polyval(dy, t)) ** 2)))
    rate = float(np.hypot(dx[0], dy[0]))
    log("moon vs sun %+.5f, %+.5f plane px/s (%.5f), residual %.2f px"
        % (dx[0], dy[0], rate, resid))

    track = {"dx": list(dx), "dy": list(dy)}

    # --- the check: does this track produce the eclipse that happened? -----
    tot = [f for f in cfg["frames"] if f["state"] == "unfiltered"]
    if not tot:
        log("no totality in this data; track written unchecked")
        return _payload(track, rate, resid, len(ts), r_sun, r_moon, None)

    t_lo = min(utc[(f["file"], f["index"])] for f in tot)
    t_hi = max(utc[(f["file"], f["index"])] for f in tot)
    grid = np.linspace(t_lo - 600, t_hi + 600, 40000)
    sep = np.hypot(np.polyval(dx, grid), np.polyval(dy, grid))
    limit = r_moon - r_sun
    inside = grid[sep < limit]
    observed = t_hi - t_lo
    predicted = float(inside[-1] - inside[0]) if len(inside) else 0.0

    log("closest approach %.1f px (totality needs < %.1f)" % (sep.min(), limit))
    log("predicts %.1f s of totality; the light curve measured %.1f s"
        % (predicted, observed))

    ok = len(inside) > 0 and observed > 0 and \
        abs(predicted - observed) <= DURATION_TOLERANCE * observed
    if not ok:
        log("*** the fitted track does not reproduce the observed totality; "
            "refusing to correct on it")
        return None

    check = {"closestApproachPx": float(sep.min()),
             "totalityLimitPx": limit,
             "predictedTotalitySeconds": predicted,
             "observedTotalitySeconds": observed}
    return _payload(track, rate, resid, len(ts), r_sun, r_moon, check)


def _payload(track, rate, resid, n, r_sun, r_moon, check):
    return {
        "units": "plane-px, seconds since the first capture",
        "track": track,
        "ratePlanePxPerSec": rate,
        "residualPx": resid,
        "samples": n,
        "rSun": r_sun,
        "rMoon": r_moon,
        "check": check,
        "source": "ecl.tl_drift, fitted on the partial phases",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=paths.out_dir())
    ap.add_argument("--data", default=paths.data_dir())
    args = ap.parse_args(argv)

    cfgp = os.path.join(args.out, "configs", "timelapse.json")
    with open(cfgp, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    serio.restage(cfg, args.data)
    with open(os.path.join(args.out, "diag", "centres.json"),
              encoding="utf-8-sig") as f:
        centres = json.load(f)
    with open(os.path.join(args.out, "segments.json"),
              encoding="utf-8-sig") as f:
        man = json.load(f)

    res = measure_track(args.out, cfg, centres, man)
    dst = os.path.join(args.out, "diag", "drift.json")
    if res is None:
        # Not fatal - the smoother falls back to placing totality on the Moon,
        # which is what it has always done. But say so plainly: a silently
        # absent correction is exactly what put the step in the video.
        print("\nNO SUN TRACK MEASURED. Totality will be placed in the MOON's\n"
              "frame, which leaves the corona drifting across it and a step at\n"
              "each phase boundary.")
        if os.path.exists(dst):
            os.remove(dst)
        return 1

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
