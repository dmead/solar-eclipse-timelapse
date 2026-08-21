"""The geometric consistency check on the Moon fit.

How much of the Sun is showing fixes how far apart the centres are. That is what
lets a bad terminator fit be recognised: a bootstrap claiming the centres are
235 px apart on a frame where 6% of the disc is lit has locked onto something
that is not the terminator, and it does so confidently - the two frames that did
this passed every quality gate and took the fitted track's residual from 7 px
to 72.
"""

import math

import pytest

from ecl.gen_insets import lit_fraction_at, overlap_area

R_SUN, R_MOON = 279.0, 292.0


def test_no_overlap_when_far_apart():
    assert overlap_area(R_SUN, R_MOON, R_SUN + R_MOON + 1) == 0.0
    assert lit_fraction_at(R_SUN, R_MOON, 600.0) == pytest.approx(1.0)


def test_total_eclipse_hides_the_whole_disc():
    """Inside r_moon - r_sun the Sun is entirely covered."""
    for sep in (0.0, 5.0, 12.0):
        assert lit_fraction_at(R_SUN, R_MOON, sep) == pytest.approx(0.0,
                                                                    abs=1e-9)


def test_lit_fraction_is_monotonic_in_separation():
    seps = [0, 20, 60, 120, 200, 300, 400, 500, 571]
    got = [lit_fraction_at(R_SUN, R_MOON, s) for s in seps]
    assert got == sorted(got)
    assert got[-1] == pytest.approx(1.0, abs=0.01)


def test_overlap_matches_the_area_of_the_smaller_disc_when_contained():
    a = overlap_area(R_SUN, R_MOON, 0.0)
    assert a == pytest.approx(math.pi * R_SUN ** 2)


def test_centres_a_solar_radius_apart_leaves_over_half_showing():
    """A sanity anchor. With the Moon's centre sitting on the Sun's limb it
    covers a lens narrower than half the disc, so a little OVER half stays
    lit - 0.578 here. The first version of this test asserted "a bit under
    half" from intuition and was simply wrong about the geometry."""
    f = lit_fraction_at(R_SUN, R_MOON, R_SUN)
    assert 0.55 < f < 0.62


def test_the_case_that_caused_the_bug():
    """6% lit cannot mean a 235 px separation - that is the check."""
    measured = 0.0626
    bad = lit_fraction_at(R_SUN, R_MOON, 234.7)
    good = lit_fraction_at(R_SUN, R_MOON, 29.8)
    assert abs(measured - bad) > 0.35, "the bad bootstrap must be rejected"
    assert abs(0.0438 - good) < 0.35, "a good one must survive"


def test_symmetric_in_the_two_radii():
    for d in (0.0, 50.0, 250.0, 560.0):
        assert overlap_area(R_SUN, R_MOON, d) == pytest.approx(
            overlap_area(R_MOON, R_SUN, d))
