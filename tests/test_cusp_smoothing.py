"""The cusp angle smoother must follow the motion, not impose a shape on it.

`fit_cusp_track` takes the wobble out of per-frame cusp measurements without
moving them: raw scatter of a couple of pixels is magnified several times over
in a cusp panel, and the first version of that panel had the horn leaving its
own box. It used to smooth with ONE polynomial across a whole capture, which is
right only while a capture is a short clip.

Given a single capture spanning a whole partial eclipse it is not. The cusp
angle sweeps about a quarter turn and accelerates hard through maximum, and the
best parabola through that shape is tens of degrees out in the middle - far
enough that `keep_on_limb` discarded the panels as having wandered off the limb.

Two datasets are checked, and they are checked for opposite things:

  2024-04-08  the published totality run, 20 short captures. Nothing here may
              move: this is the no-regression half, and the reference values in
              the fixture are what the code produced before the change.
  2026-08-12  a UK partial recorded as ONE 78-minute capture, which is the case
              that broke. The fit has to stay as close to the limb as the raw
              measurements it was given.

Both fixtures hold ANGLES measured off the real frames, so they are tens of KB
rather than the hundreds of GB they were measured from. `tools/make_cusp_fixture.py`
regenerates them.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ecl.gen_insets import fit_cusp_track, _cusp_noise

DATA = Path(__file__).parent / "data"
SIGMA = 0.004          # per-frame scatter, radians; about 1.2 px at r=303


def _load(name):
    with open(DATA / name, encoding="utf-8") as fh:
        return json.load(fh)


def _sep(a, b):
    """Angular distance, the short way round."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _pair(got, want):
    """Error on both cusps, under whichever pairing of the two fits better.

    Which cusp is "first" is not meaningful - upper and lower are named by
    image position and the pair can swap as the geometry rotates - so a metric
    that assumes an order measures the labelling, not the accuracy. Sorting the
    angles numerically is NOT a fix: they wrap, and sorting pairs them across
    the wrap. Scored that way this comparison read 140 deg where the true answer
    was 2.
    """
    return min(((_sep(got[0], want[j]), _sep(got[1], want[1 - j]))
                for j in (0, 1)), key=sum)


def _score(fixture, get):
    """Both cusps of every frame that has an independent truth, in degrees."""
    idx = {r[0]: i for i, r in enumerate(fixture["rows"])}
    errs = []
    for frame, truth in fixture["truth"].items():
        i = idx.get(int(frame))
        if i is not None:
            errs += list(_pair(get(i), truth))
    return np.degrees(np.array(errs))


def _single_parabola(rows):
    """The whole-capture fit this replaced, for use as a control."""
    ts = [r[0] for r in rows]
    t0 = sum(ts)/len(ts)
    ts = [t - t0 for t in ts]
    out = []
    for which in (1, 2):
        vs = [r[which] for r in rows]
        for i in range(1, len(vs)):
            while vs[i] - vs[i-1] > math.pi:
                vs[i] -= 2*math.pi
            while vs[i] - vs[i-1] < -math.pi:
                vs[i] += 2*math.pi
        c = np.polyfit(ts, vs, 2)
        out.append([float(np.polyval(c, t)) for t in ts])
    return out


# --------------------------------------------------------------------------
# 2024-04-08: the published run. Nothing may move.
# --------------------------------------------------------------------------

def test_the_published_totality_run_is_unchanged():
    """Every capture of the 2024 run must fit as it did before the change.

    The reference values were produced by the previous whole-capture
    polynomial. A cusp panel is ~42 px half-box there, so the bound is a small
    fraction of one: this is asserting "did not move", not "moved acceptably".
    """
    fx = _load("cusp_track_2024.json")
    r_sun = fx["r_sun_plane_px"]
    worst, worst_name = 0.0, None
    for cap in fx["captures"]:
        rows = [tuple(r) for r in cap["rows"]]
        out, _dev, _win = fit_cusp_track(rows, cap["deg"])
        ref = cap["reference"]
        for i in range(len(rows)):
            d = max(abs(out[i][0] - ref[i][0]), abs(out[i][1] - ref[i][1]))
            if d > worst:
                worst, worst_name = d, cap["name"]
    assert worst*r_sun < 6.0, (
        f"cusp fit moved {worst*r_sun:.2f} px on {worst_name}")


def test_short_captures_keep_the_whole_capture_window():
    """The mechanism behind the test above, asserted directly.

    A 30-60 s capture's cusps really do move like a parabola, so the window
    search must come straight back with "all of it" and reproduce the old
    behaviour exactly. Two of the twenty narrow - both sit just after third
    contact, where the cusps reappear and sweep fast - and those two are the
    only captures in the run whose panels move at all.
    """
    fx = _load("cusp_track_2024.json")
    whole = 0
    for cap in fx["captures"]:
        rows = [tuple(r) for r in cap["rows"]]
        _out, _dev, win = fit_cusp_track(rows, cap["deg"])
        if win == (len(rows), len(rows)):
            whole += 1
    assert whole >= len(fx["captures"]) - 3, (
        f"only {whole} of {len(fx['captures'])} captures kept the whole-capture "
        "window; the no-regression guarantee rests on this")


# --------------------------------------------------------------------------
# 2026-08-12: one capture over a whole partial eclipse. The case that broke.
# --------------------------------------------------------------------------

def test_a_whole_eclipse_capture_stays_on_the_limb():
    """The fit must be as close to the limb as the measurements it was handed.

    Scored against cusps read off the frames by a plainly different method - a
    single-radius scan for the ends of the lit run, sharing no code with
    `find_cusps`' multi-radius band and sub-pixel edge walk.

    The bar is the RAW measurements' own accuracy rather than an absolute
    angle, because that is the only thing this function is responsible for: it
    is a smoother, and it cannot be asked to beat its input. Around maximum the
    two measurement methods genuinely differ by ~9 deg on a crescent a pixel
    thick, and a few frames in the cloudy sunset tail are worse still; both
    show up identically in the raw series, which is why the comparison is
    relative.
    """
    fx = _load("cusp_track_2026.json")
    rows = [tuple(r) for r in fx["rows"]]
    out, _dev, _win = fit_cusp_track(rows, fx["deg"])

    raw = _score(fx, lambda i: (rows[i][1], rows[i][2]))
    fit = _score(fx, lambda i: out[i])

    assert np.median(fit) < np.median(raw) + 0.5, (
        f"smoothing pulled the fit off the limb: median {np.median(fit):.2f} deg "
        f"against {np.median(raw):.2f} deg raw")
    assert np.percentile(fit, 90) < np.percentile(raw, 90) + 1.5, (
        f"p90 {np.percentile(fit, 90):.2f} deg against "
        f"{np.percentile(raw, 90):.2f} deg raw")


def test_one_parabola_across_the_whole_eclipse_would_fail_that():
    """The control that gives the test above its meaning.

    Without this, a smoother that quietly did nothing at all would pass. This
    is the previous implementation run on the same measurements, and it has to
    miss by a wide margin or the fixture has stopped exercising the fault.
    """
    fx = _load("cusp_track_2026.json")
    rows = [tuple(r) for r in fx["rows"]]
    par = _single_parabola(rows)
    old = _score(fx, lambda i: (par[0][i], par[1][i]))
    raw = _score(fx, lambda i: (rows[i][1], rows[i][2]))
    assert np.median(old) > 4*np.median(raw), (
        f"the whole-capture parabola is only {np.median(old):.2f} deg out; "
        "this fixture no longer reproduces the bug")


def test_the_window_narrows_where_the_sweep_is_steep():
    """One window for the capture is not enough, even a well chosen one.

    Picking a single window on the median residual settles where the quiet
    majority of frames wants it - 140 frames here - and still left 83 deg of
    error through maximum. The window is therefore chosen per point, and on
    this capture it has to span a wide range to prove it.
    """
    fx = _load("cusp_track_2026.json")
    rows = [tuple(r) for r in fx["rows"]]
    _out, _dev, win = fit_cusp_track(rows, fx["deg"])
    lo, hi = win
    assert lo < 32, f"window never tightened below {lo} frames"
    assert hi > 4*lo, (
        f"window barely varied ({lo}-{hi}); it should smooth hard over the slow "
        "approach and tighten through maximum")


# --------------------------------------------------------------------------
# The mechanism, on signals whose answer is known exactly.
# --------------------------------------------------------------------------

def _rows(f1, f2, n, seed):
    rng = np.random.default_rng(seed)
    return [(i, f1(i) + rng.normal(0, SIGMA), f2(i) + rng.normal(0, SIGMA))
            for i in range(n)]


def test_parabolic_motion_keeps_the_whole_capture_window():
    n = 400
    def f1(i):
        return 1e-6*(i - 200)**2 + 0.002*(i - 200) + 0.3
    def f2(i):
        return -8e-7*(i - 200)**2 + 0.001*(i - 200) - 1.1

    out, _dev, win = fit_cusp_track(_rows(f1, f2, n, 7), 2)
    assert win == (n, n), f"window shrank to {win} on genuinely parabolic motion"
    worst = max(max(abs(out[i][0] - f1(i)), abs(out[i][1] - f2(i)))
                for i in range(n))
    assert math.degrees(worst) < 0.5


def test_a_steep_sweep_is_followed_rather_than_averaged_away():
    """An arctangent is the shape a cusp angle really takes through a deep
    partial: slow, a fast swing through maximum, slow again."""
    n = 400
    def f1(i):
        return 2.6*math.atan((i - 200)/45.0)
    def f2(i):
        return 0.9*math.atan((i - 200)/70.0) - 0.8

    rows = _rows(f1, f2, n, 7)
    out, _dev, win = fit_cusp_track(rows, 2)
    assert win[0] < n
    worst = max(max(abs(out[i][0] - f1(i)), abs(out[i][1] - f2(i)))
                for i in range(n))
    assert math.degrees(worst) < 2.0

    par = _single_parabola(rows)
    old = max(abs(par[0][i] - f1(i)) for i in range(n))
    assert math.degrees(old) > 30, "control is no longer exercising the failure"


def test_noise_estimate_survives_a_steep_sweep():
    """The window search is judged against this number, so it has to measure
    the scatter and not how far the series travels."""
    n = 400
    rng = np.random.default_rng(3)
    vs = [2.6*math.atan((i - 200)/45.0) + rng.normal(0, 0.01) for i in range(n)]
    assert _cusp_noise(vs) == pytest.approx(0.01, rel=0.5)
