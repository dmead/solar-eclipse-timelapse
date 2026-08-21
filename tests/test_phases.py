"""The phase classifier is geometry, and must stay geometry.

These tests build a synthetic Moon track - a straight line passing the Sun at a
chosen miss distance - and check the classifier against it. Nothing here reads
an image, which is the point: the phase of a frame must not depend on whether
any feature detector found anything.
"""
import pytest

from ecl import phases


def track(rate=0.1, closest=0.0, t_close=1000.0):
    """A Moon crossing the Sun in +x, offset by `closest` px in y.

    Returned in the same shape `tl_drift` writes: polynomial coefficients in
    numpy's highest-power-first order, evaluated on run-relative seconds.
    """
    return {"dx": [rate, -rate * t_close], "dy": [0.0, closest]}


R_SUN, R_MOON = 279.0, 292.0            # the 2024-04-08 pair, plane px


def test_separation_is_minimum_at_closest_approach():
    tr = track(closest=5.0)
    t = phases.closest_approach(tr, 0.0, 2000.0)
    assert t == pytest.approx(1000.0, abs=0.5)
    assert phases.separation(tr, t) == pytest.approx(5.0, abs=0.01)


def test_all_four_contacts_on_a_total_eclipse():
    tr = track(closest=2.0)
    c = phases.contacts(tr, R_SUN, R_MOON, -20000.0, 20000.0)
    assert all(c[k] is not None for k in ("c1", "c2", "c3", "c4"))
    assert c["c1"] < c["c2"] < c["c3"] < c["c4"]
    # Totality is symmetric about closest approach for a straight-line pass.
    assert 0.5 * (c["c2"] + c["c3"]) == pytest.approx(1000.0, abs=1.0)


def test_a_partial_eclipse_has_no_second_or_third_contact():
    """Miss distance greater than r_moon - r_sun never covers the Sun."""
    tr = track(closest=100.0)          # far outside the 13 px totality limit
    c = phases.contacts(tr, R_SUN, R_MOON, -20000.0, 20000.0)
    assert c["c1"] is not None and c["c4"] is not None
    assert c["c2"] is None and c["c3"] is None
    assert phases.classify(1000.0, c, [], R_SUN, R_MOON, tr) == "partial"


def test_the_phases_run_in_order_across_the_eclipse():
    tr = track(closest=2.0)
    c = phases.contacts(tr, R_SUN, R_MOON, -20000.0, 20000.0)
    seen = [phases.classify(t, c, [], R_SUN, R_MOON, tr)
            for t in (c["c1"] - 100, 0.5 * (c["c1"] + c["c2"]), 1000.0,
                      0.5 * (c["c3"] + c["c4"]), c["c4"] + 100)]
    assert seen == ["before first contact", "partial", "totality",
                    "partial", "after fourth contact"]


def test_beads_win_over_the_geometric_phase():
    """A frame showing beads is a bead frame whatever the separation says."""
    tr = track(closest=2.0)
    c = phases.contacts(tr, R_SUN, R_MOON, -20000.0, 20000.0)
    assert phases.classify(1000.0, c, [], R_SUN, R_MOON, tr) == "totality"
    assert phases.classify(1000.0, c, [(995.0, 1005.0)], R_SUN, R_MOON,
                           tr) == "baily's beads"


def test_contacts_are_bracketed_at_closest_approach_not_the_midpoint():
    """The bug this pins: a window whose midpoint is far from the minimum.

    Splitting at the midpoint only brackets the crossings when the minimum
    happens to sit near it. Here it deliberately does not, and the real data
    behaved exactly this way - reporting no totality at all.
    """
    tr = track(closest=2.0, t_close=1882.0)
    c = phases.contacts(tr, R_SUN, R_MOON, -6000.0, 12000.0)   # midpoint 3000
    assert c["c2"] is not None and c["c3"] is not None
    assert c["c2"] < 1882.0 < c["c3"]


def test_totality_duration_matches_the_chord_it_should_be():
    """Geometry check, independent of the code path that finds the contacts."""
    import math

    closest, rate = 2.0, 0.1
    tr = track(rate=rate, closest=closest)
    c = phases.contacts(tr, R_SUN, R_MOON, -20000.0, 20000.0)
    limit = R_MOON - R_SUN
    half_chord = math.sqrt(limit * limit - closest * closest)
    assert c["c3"] - c["c2"] == pytest.approx(2 * half_chord / rate, rel=1e-3)


def test_an_annular_geometry_reports_no_totality():
    """r_moon < r_sun cannot cover the Sun at any separation."""
    tr = track(closest=0.0)
    c = phases.contacts(tr, 292.0, 279.0, -20000.0, 20000.0)
    assert c["c2"] is None and c["c3"] is None
    assert phases.classify(1000.0, c, [], 292.0, 279.0, tr) == "partial"
