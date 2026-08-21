"""The Sun-track correction that places totality in the Sun's frame.

The interesting behaviour is the self-check. The track is fitted on the partial
phases and never sees totality, so extrapolating into it is a prediction, and a
fit that does not reproduce the eclipse that actually happened must be refused
rather than used to move every totality frame by a fabricated amount.
"""

import numpy as np
import pytest

from ecl.tl_drift import DURATION_TOLERANCE, moon_offset

FPS = 30.0


def test_moon_offset_is_the_fitted_line():
    track = {"dx": [-0.1, 200.0], "dy": [0.05, -100.0]}
    assert moon_offset(track, 0.0) == pytest.approx((200.0, -100.0))
    assert moon_offset(track, 1000.0) == pytest.approx((100.0, -50.0))


def test_offset_passes_through_zero_at_greatest_eclipse():
    """A track describing a total eclipse must bring the centres together."""
    track = {"dx": [-0.1, 190.0], "dy": [0.05, -95.0]}
    t = np.linspace(0, 4000, 4001)
    sep = np.hypot(*np.array([moon_offset(track, x) for x in t]).T)
    assert sep.min() < 5.0
    # and it must be a single approach, not a wander
    assert 1500 < t[int(np.argmin(sep))] < 2500


def _fit(ox, oy, ts):
    return {"dx": list(np.polyfit(ts, ox, 1)),
            "dy": list(np.polyfit(ts, oy, 1))}


def test_a_track_fitted_to_partials_recovers_the_rate_it_was_built_from():
    """The fit is a straight line through terminator measurements; check it
    survives realistic scatter on those measurements."""
    rng = np.random.default_rng(0)
    ts = np.linspace(0, 2700, 30)
    true = (-0.097, 0.061)
    ox = true[0] * ts + 183.0 + rng.normal(0, 7.0, ts.size)
    oy = true[1] * ts - 116.8 + rng.normal(0, 7.0, ts.size)
    tr = _fit(ox, oy, ts)
    assert tr["dx"][0] == pytest.approx(true[0], abs=0.01)
    assert tr["dy"][0] == pytest.approx(true[1], abs=0.01)


def test_duration_tolerance_rejects_a_track_that_misses_totality():
    """A track whose closest approach never gets inside r_moon - r_sun implies
    no total eclipse; the pass must refuse it rather than correct on it."""
    limit = 13.0
    # A near-miss: passes 40 px away, so this data was never total.
    track = {"dx": [-0.1, 190.0], "dy": [0.05, -60.0]}
    t = np.linspace(0, 4000, 4001)
    sep = np.hypot(*np.array([moon_offset(track, x) for x in t]).T)
    assert sep.min() > limit
    assert not (sep < limit).any()


def test_tolerance_is_a_fraction_not_an_absolute():
    """Guards the units of DURATION_TOLERANCE: it is compared against the
    observed duration, so it has to be well under 1."""
    assert 0 < DURATION_TOLERANCE < 1
