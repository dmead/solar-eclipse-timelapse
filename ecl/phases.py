"""Name the eclipse phase each frame belongs to.

    python -m ecl.phases --out <out-dir>

Writes `diag/phases.json` and stamps a `phase` on every frame in
`configs/timelapse.json`, so the renderer can caption a frame without knowing
anything about eclipses.

THE CLASSIFIER IS GEOMETRY, NOT PICTURE-READING. Every boundary here comes from
one number - the centre separation `d` between Sun and Moon - measured against
the two radii the survey already established:

    d >= r_sun + r_moon     the discs do not touch          no eclipse
    d  < r_sun + r_moon     the limbs cross                 partial
    d <= r_moon - r_sun     the Sun is wholly covered       totality

The separation comes from `diag/drift.json`, which `ecl.tl_drift` fits to the
PARTIAL phases where the geometry is well conditioned, and which self-checks by
predicting the length of totality against the observed one. Nothing in this
module looks at a pixel, so the phase of a frame does not depend on whether a
feature detector found anything, and re-running it on the same numbers gives the
same answer every time.

Contacts are the instants where those inequalities flip. They are found by
bisection on the fitted d(t) rather than by scanning frames, so a contact lands
at its true time even when no frame was captured near it - the operator's clip
boundaries have nothing to do with where the Moon actually was.

The bead phases are the exception, and deliberately so. Baily's beads are not a
geometric state - they are the last photosphere shining through lunar valleys,
which depends on the Moon's limb profile and not on d. Their extent is taken
from `diag/beads.json` where that pass found it, and otherwise from a symmetric
window around each contact, whose width is a config key rather than a constant
discovered here.
"""

import argparse
import datetime
import io
import json
import os

from . import paths
from .tl_drift import moon_offset

__all__ = ["PHASES", "classify", "main"]

# In the order they occur. The renderer captions with these strings verbatim.
PHASES = [
    "before first contact",
    "partial",
    "baily's beads",
    "totality",
    "partial",
    "after fourth contact",
]

# Seconds either side of second and third contact that read as beads when the
# beads pass found no window of its own. Overridable as `phases.bead_window_s`.
BEAD_WINDOW_S = 2.0

# Bisection tolerance for a contact time, in seconds. The frame interval is
# ~0.04 s, so a millisecond is far below anything that can matter.
CONTACT_TOL_S = 0.001


def tune(out_dir, log=None):
    """Resolve the one setting this pass has."""
    global BEAD_WINDOW_S
    from .params import load

    P = load(out_dir, create=False)
    BEAD_WINDOW_S = P.get("phases.bead_window_s", BEAD_WINDOW_S)
    if log:
        log("  bead window %.1f s either side of contact" % BEAD_WINDOW_S)
    return P


def separation(track, t):
    """Centre separation in plane px at time t, from the fitted Moon track."""
    ox, oy = moon_offset(track, t)
    return (ox * ox + oy * oy) ** 0.5


def _cross(track, target, lo, hi):
    """Time in [lo, hi] where separation crosses `target`, or None.

    Bisection, so the answer is the time the MODEL crosses rather than the time
    of the nearest captured frame. Returns None when the interval does not
    bracket a crossing, which is the honest answer for data that never reaches
    totality at all.
    """
    f_lo = separation(track, lo) - target
    f_hi = separation(track, hi) - target
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None
    while hi - lo > CONTACT_TOL_S:
        mid = 0.5 * (lo + hi)
        f_mid = separation(track, mid) - target
        if f_mid == 0.0:
            return mid
        if (f_lo > 0) == (f_mid > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def closest_approach(track, t0, t1, steps=20000):
    """Time of minimum separation in [t0, t1].

    Found by a coarse scan then a golden-section refine. The scan matters: the
    separation is a smooth U, and every contact is a crossing on one arm of it,
    so this time is the only correct place to split the interval for bisection.
    """
    lo, hi = t0, t1
    step = (hi - lo) / float(steps)
    best_t, best_d = lo, separation(track, lo)
    for i in range(1, steps + 1):
        t = lo + i * step
        d = separation(track, t)
        if d < best_d:
            best_t, best_d = t, d
    a, b = max(t0, best_t - step), min(t1, best_t + step)
    while b - a > CONTACT_TOL_S:
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if separation(track, m1) < separation(track, m2):
            b = m2
        else:
            a = m1
    return 0.5 * (a + b)


def contacts(track, r_sun, r_moon, t0, t1):
    """The four contact times, any of which may be None if never reached.

    C1/C4 are where the limbs first and last cross; C2/C3 bracket totality. A
    partial eclipse yields C1 and C4 with C2 and C3 both None, and the caller
    must cope with that rather than assuming totality happened.

    EVERY INTERVAL IS SPLIT AT CLOSEST APPROACH, never at the midpoint of the
    search window. The separation falls to a minimum and rises again, so each
    threshold is crossed twice, once on each arm - and a midpoint split only
    brackets them if the minimum happens to sit near it. It generally does not:
    on the 2024-04-08 data the window midpoint is t=1350 s where the separation
    is 62.8 px, while closest approach is at t=1882 s. Bisecting from the
    midpoint reported no totality at all on data that plainly contains it.
    """
    outer, inner = r_sun + r_moon, r_moon - r_sun
    mid = closest_approach(track, t0, t1)
    return {
        "c1": _cross(track, outer, t0, mid),
        "c2": _cross(track, inner, t0, mid) if inner > 0 else None,
        "c3": _cross(track, inner, mid, t1) if inner > 0 else None,
        "c4": _cross(track, outer, mid, t1),
    }


def classify(t, c, bead_spans, r_sun, r_moon, track):
    """The phase name for one frame time.

    Beads win over the geometric phase where they overlap, because a frame
    showing beads is a bead frame whatever the separation says.
    """
    for lo, hi in bead_spans:
        if lo <= t <= hi:
            return "baily's beads"

    d = separation(track, t)
    if c.get("c2") is not None and c.get("c3") is not None:
        if c["c2"] <= t <= c["c3"]:
            return "totality"
    elif r_moon > r_sun and d <= r_moon - r_sun:
        return "totality"

    if d < r_sun + r_moon:
        return "partial"
    if c.get("c1") is not None and t < c["c1"]:
        return "before first contact"
    return "after fourth contact"


def bead_windows(out_dir, c, t_of):
    """Bead spans in run-relative seconds.

    Prefers what `ecl.beadwindow` measured; falls back to a symmetric window on
    each contact so a run without that diagnostic still captions its beads.
    """
    spans, measured = [], 0
    path = os.path.join(out_dir, "diag", "beads.json")
    try:
        with io.open(path, encoding="utf-8-sig") as fh:
            found = json.load(fh)
    except (OSError, ValueError):
        found = {}

    for name, w in sorted(found.items()):
        lo, hi = t_of(name, w.get("lo")), t_of(name, w.get("hi"))
        if lo is not None and hi is not None:
            spans.append((min(lo, hi), max(lo, hi)))
            measured += 1

    if not spans:
        for k in ("c2", "c3"):
            if c.get(k) is not None:
                spans.append((c[k] - BEAD_WINDOW_S, c[k] + BEAD_WINDOW_S))
    return spans, measured


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out or paths.out_dir()
    tune(out_dir)

    def rd(*p):
        with io.open(os.path.join(out_dir, *p), encoding="utf-8-sig") as fh:
            return json.load(fh)

    cfg_path = os.path.join(out_dir, "configs", "timelapse.json")
    cfg = rd("configs", "timelapse.json")
    man = rd("segments.json")
    drift = rd("diag", "drift.json")
    track = drift["track"]

    # THE RADII MUST COME FROM THE DRIFT PASS, not from the survey.
    #
    # They are not interchangeable. The survey's radius is the photosphere as the
    # disc fit sees it through the filter; tl_drift's is measured off the
    # terminator on the same frames it fitted the track to. On the 2024-04-08
    # data they are 291.8 and 279.0 - and since totality is `d < r_moon - r_sun`,
    # the survey pair gives a limit of 0.2 px, which no real eclipse ever meets,
    # so every totality frame came back "partial".
    #
    # The drift pair is the self-consistent one: it is what that pass checked
    # itself against, reporting a closest approach of 1.6 px inside a 13.0 px
    # limit and predicting 219.1 s of totality against 213.2 s observed. Using
    # the track without the radii it was validated with is mixing two geometries.
    r_sun = float(drift.get("rSun") or rd("survey.json")["radius_plane_px"])
    r_moon = float(drift.get("rMoon")
                   or rd("diag", "centres.json")["rMoon"])

    fps = {f["name"]: f["fps"] for f in man["files"]}
    t0s = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"])
           for f in man["files"]}
    epoch = min(t0s.values())

    def t_of(name, index):
        if index is None or name not in fps:
            return None
        return ((t0s[name] - epoch).total_seconds() + index / fps[name])

    times = sorted(t_of(f["file"], f["index"]) for f in cfg["frames"])
    span_lo, span_hi = times[0], times[-1]

    # Search a window sized by the eclipse, not by a round number of seconds.
    # First and fourth contact sit `r_sun + r_moon` of separation away from
    # closest approach, so at the fitted approach rate that is how long the
    # search has to reach - with margin, and never narrower than the capture.
    # A fixed +/-3600 s missed C4 on the 2024-04-08 data, whose fourth contact
    # falls about 4800 s after closest approach.
    rate = max(abs(separation(track, span_hi) - separation(track, span_lo))
               / max(span_hi - span_lo, 1e-6), 1e-9)
    reach = 2.0 * (r_sun + r_moon) / rate
    c = contacts(track, r_sun, r_moon,
                 min(span_lo, span_lo - reach), max(span_hi, span_hi + reach))
    spans, measured = bead_windows(out_dir, c, t_of)

    print("r_sun %.2f, r_moon %.2f plane px" % (r_sun, r_moon))
    for k in ("c1", "c2", "c3", "c4"):
        v = c[k]
        print("  %s %s" % (k.upper(),
                           "not reached" if v is None else "t=%+.3f s" % v))
    if c["c2"] is not None and c["c3"] is not None:
        print("  totality lasts %.1f s" % (c["c3"] - c["c2"]))
    print("  bead spans: %d (%s)"
          % (len(spans), "measured" if measured else "from contacts"))

    counts = {}
    for f in cfg["frames"]:
        t = t_of(f["file"], f["index"])
        name = classify(t, c, spans, r_sun, r_moon, track)
        f["phase"] = name
        counts[name] = counts.get(name, 0) + 1

    for name in dict.fromkeys(PHASES):      # PHASES lists "partial" twice
        if name in counts:
            print("  %-22s %5d frames" % (name, counts[name]))

    with io.open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, separators=(",", ":"))

    diag = {"r_sun": r_sun, "r_moon": r_moon, "contacts": c,
            "bead_spans": spans, "bead_source": "measured" if measured
            else "contacts", "counts": counts}
    os.makedirs(os.path.join(out_dir, "diag"), exist_ok=True)
    with io.open(os.path.join(out_dir, "diag", "phases.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(diag, fh, indent=1)
    print("wrote diag/phases.json and stamped %d frames" % len(cfg["frames"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
