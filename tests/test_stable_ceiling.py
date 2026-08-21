"""A stable frame is bounded on its own p99, not just on its segment's gain.

The bug this pins: one gain per stable segment is chosen for the scene the
segment settles into. For about two seconds after second contact the frame still
carries residual photosphere, and that gain drives it to white - the beads bloom
and the chromosphere under them is lost.

The cap has to bind on the bright frames and leave the settled ones exactly as
they were, so both halves are tested. A cap that simply darkened the segment
would pass the first test and fail the second.
"""
import math

from ecl import gen_timelapse as gt


def cap_for(p99, state="unfiltered"):
    """The bound the stable branch applies, as the module computes it."""
    return (gt.STABLE_CEILING * gt.TARGET[state] * gt.FULL_SCALE
            / max(p99, 1.0 / 65535.0))


def test_cap_binds_on_a_frame_still_holding_photosphere():
    # p99 measured on the 2024-04-08 run just after second contact.
    seg_gain = 27.31
    assert cap_for(0.11882) < seg_gain
    assert min(seg_gain, cap_for(0.11882)) / seg_gain < 0.35


def test_cap_relaxes_to_the_segment_gain_once_the_scene_settles():
    # p99 from the same segment, 126 frames later, deep in totality.
    seg_gain = 27.31
    assert cap_for(0.03204) >= seg_gain
    assert min(seg_gain, cap_for(0.03204)) == seg_gain


def test_cap_is_monotonic_in_p99():
    """Brighter frame, tighter bound. Nothing clever, but it is the whole idea."""
    p99s = [0.11882, 0.08678, 0.05640, 0.04028, 0.03204]
    caps = [cap_for(q) for q in p99s]
    assert caps == sorted(caps), "a dimmer frame must not be capped harder"


def test_a_filtered_frame_uses_its_own_target():
    """The partial phases are allowed to sit brighter than the corona."""
    assert gt.TARGET["filtered"] > gt.TARGET["unfiltered"]
    assert cap_for(0.5, "filtered") > cap_for(0.5, "unfiltered")


def test_ceiling_is_configurable_and_defaults_to_the_transition_value():
    """Both ceilings answer the same question, so they start from one number."""
    assert gt.STABLE_CEILING == gt.TRANSITION_CEILING

    from ecl.params import DEFAULTS
    assert "stable_ceiling" in DEFAULTS["select"]
    assert DEFAULTS["select"]["stable_ceiling"] == gt.STABLE_CEILING


def test_an_infinite_ceiling_restores_the_old_behaviour():
    """The escape hatch the config comment promises actually works."""
    old = gt.STABLE_CEILING
    try:
        gt.STABLE_CEILING = math.inf
        assert min(27.31, cap_for(0.11882)) == 27.31
    finally:
        gt.STABLE_CEILING = old
