"""The crescent measurement, and the failure mode it was written to avoid.

The first version thresholded at half the frame's 99.5th percentile. That is the
photosphere only while the crescent is bigger than 0.5% of the frame; once it is
thinner the percentile lands on sky and the "photosphere level" silently becomes a
sky level. On real data it read 0.025 where the answer was 0.73 - not an error,
just a wrong number, which is worse.

So the property that matters is not accuracy on one frame. It is that the answer
does not depend on how much of the Sun is left.
"""

import numpy as np
import pytest

from gen_timelapse import crescent_level

H, W = 540, 960
SKY = 0.004
LEVEL = 0.62


def frame(area_px, level=LEVEL, sky=SKY, noise=0.0, seed=0):
    """A flat sky with a `area_px` patch of photosphere in it."""
    a = np.full((H, W), sky, np.float32)
    side = max(int(round(area_px ** 0.5)), 1)
    a[10:10 + side, 10:10 + side] = level
    if noise:
        a += np.random.default_rng(seed).normal(0, noise, a.shape).astype(np.float32)
    return a


@pytest.mark.parametrize("area", [200_000, 50_000, 10_000, 2_500, 400])
def test_level_is_independent_of_how_much_sun_is_left(area):
    got, n = crescent_level(frame(area))
    assert got == pytest.approx(LEVEL, abs=0.01), (
        f"crescent of {area} px measured as {got:.3f}, not {LEVEL}")
    assert n == pytest.approx(area, rel=0.15)


def test_survives_noise():
    got, n = crescent_level(frame(20_000, noise=0.002))
    assert got == pytest.approx(LEVEL, abs=0.02)
    assert n > 0


def test_no_crescent_reports_nothing():
    """Totality: sky only. Reporting a level here would normalize on noise."""
    a = np.full((H, W), SKY, np.float32)
    got, n = crescent_level(a)
    assert (got, n) == (0.0, 0)


def test_saturated_crescent_still_reads_as_saturated():
    """The caller needs to SEE the clipping to know the measurement is useless;
    three captures in this data set are entirely in that state."""
    got, _ = crescent_level(frame(20_000, level=0.999))
    assert got > 0.97


def test_a_percentile_threshold_would_have_failed_here():
    """Guards the reason for the design, not just the behaviour.

    0.5% of this frame is 2592 px, so a crescent below that defeats the old
    threshold while the peak-relative one still works.
    """
    a = frame(1_000)
    old = np.median(a[a > 0.5 * np.percentile(a, 99.5)])
    new, _ = crescent_level(a)
    assert old < 0.5 * LEVEL, "the old rule should be badly wrong here"
    assert new == pytest.approx(LEVEL, abs=0.01)
