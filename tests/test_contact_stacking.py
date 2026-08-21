"""Frames beside a contact are left unstacked; frames beside a GAP are not.

The distinction matters because only a contact has beads. A boundary in the
frame list can also be one capture ending and the next opening minutes later
with the eclipse well past third contact — unstacking there costs a twentyfold
noise penalty and preserves nothing.
"""

import pytest

from ecl.gen_timelapse import unstack_near_contacts

BEAD_FRAMES = 3


def seq(spec):
    """(file, state) pairs -> frames, all starting at full stack."""
    return [{"file": f, "state": st, "index": i, "stack": 20}
            for i, (f, st) in enumerate(spec)]


def test_unstacks_at_a_boundary_with_measured_beads():
    frames = seq([("a.ser", "filtered")] * 6 + [("a.ser", "unfiltered")] * 6)
    marked, spared = unstack_near_contacts(
        frames, {"a.ser": {"lo": 4, "hi": 8}}, BEAD_FRAMES, log=None)
    assert marked > 0 and spared == 0
    assert frames[5]["stack"] == 1 and frames[6]["stack"] == 1
    assert frames[0]["stack"] == 20 and frames[-1]["stack"] == 20


def test_leaves_a_boundary_alone_when_no_beads_were_found():
    """Third contact fell in a recording gap: the boundary is a capture change,
    and the bead pass found nothing on either side."""
    frames = seq([("a.ser", "unfiltered")] * 6 + [("b.ser", "filtered")] * 6)
    marked, spared = unstack_near_contacts(frames, {}, BEAD_FRAMES, log=None)
    # Empty beads means the pass was never run -> keep the old behaviour.
    assert marked > 0

    frames = seq([("a.ser", "unfiltered")] * 6 + [("b.ser", "filtered")] * 6)
    marked, spared = unstack_near_contacts(
        frames, {"c.ser": {"lo": 1, "hi": 2}}, BEAD_FRAMES, log=None)
    assert marked == 0 and spared > 0
    assert all(f["stack"] == 20 for f in frames)


def test_one_end_with_beads_and_one_without():
    """The real shape of this data: beads at second contact, a gap at third."""
    frames = seq([("a.ser", "filtered")] * 5
                 + [("a.ser", "unfiltered")] * 5
                 + [("b.ser", "unfiltered")] * 5
                 + [("c.ser", "filtered")] * 5)
    unstack_near_contacts(frames, {"a.ser": {"lo": 5, "hi": 9}},
                          BEAD_FRAMES, log=None)
    assert frames[4]["stack"] == 1, "second contact must be unstacked"
    assert frames[5]["stack"] == 1
    assert frames[-1]["stack"] == 20, "the gap must keep its stack"
    assert frames[-6]["stack"] == 20


def test_dense_frames_are_never_touched():
    """Their groups overlap by design and are set elsewhere."""
    frames = seq([("a.ser", "filtered")] * 4 + [("a.ser", "unfiltered")] * 4)
    for f in frames:
        f["dense"] = True
    marked, _ = unstack_near_contacts(frames, {"a.ser": {}}, BEAD_FRAMES,
                                      log=None)
    assert marked == 0
    assert all(f["stack"] == 20 for f in frames)


def test_no_boundary_means_nothing_changes():
    frames = seq([("a.ser", "unfiltered")] * 10)
    marked, spared = unstack_near_contacts(frames, {"a.ser": {}}, BEAD_FRAMES,
                                           log=None)
    assert (marked, spared) == (0, 0)
    assert all(f["stack"] == 20 for f in frames)


@pytest.mark.parametrize("bead_frames", [1, 5, 12])
def test_window_width_is_honoured(bead_frames):
    n = 40
    frames = seq([("a.ser", "filtered")] * n + [("a.ser", "unfiltered")] * n)
    unstack_near_contacts(frames, {"a.ser": {}}, bead_frames, log=None)
    unstacked = [i for i, f in enumerate(frames) if f["stack"] == 1]
    assert min(unstacked) == n - bead_frames
    assert max(unstacked) == n + bead_frames - 1
