"""The GIF preview's chapter picker.

The thing worth testing is that it degrades: the same rules have to give four
segments for a full shoot, fewer for data missing a phase, and something rather
than nothing for data that has no eclipse in it at all.
"""

import pytest

from ecl.encode import GIF, pick_segments, runs_where

FPS = 30


def frames(spec):
    """Build a frame list from (count, kwargs) pairs."""
    out = []
    for n, kw in spec:
        out += [dict(kw) for _ in range(n)]
    return {"frames": out}


FILTERED = {"state": "filtered"}
RESOLVE = {"state": "unfiltered", "resolve": True}
BEAD = {"state": "unfiltered", "bead": True}
CORONA = {"state": "unfiltered"}

FULL = frames([(600, FILTERED), (270, RESOLVE), (135, BEAD),
               (600, CORONA), (600, FILTERED)])


def names(segs):
    return [s[2] for s in segs]


def test_runs_where_finds_disjoint_runs():
    fs = FULL["frames"]
    assert runs_where(fs, lambda f: f.get("bead")) == [(870, 135)]
    # Partial phases appear twice: before the filter comes off and after.
    assert runs_where(fs, lambda f: f["state"] == "filtered") == \
        [(0, 600), (1605, 600)]


def test_full_shoot_gets_every_chapter_in_time_order():
    segs = pick_segments(FULL, FPS)
    assert names(segs) == ["partial phases", "the filter coming off",
                           "second contact", "totality"]
    starts = [s[0] for s in segs]
    assert starts == sorted(starts)


def test_partials_chosen_from_before_totality():
    """The post-totality run is the same length here and starts later; picking
    by length would open the preview on the wrong side of the eclipse."""
    seg = next(s for s in pick_segments(FULL, FPS) if s[2] == "partial phases")
    start, count, _ = seg
    assert start + count <= 600


def test_budget_is_split_not_multiplied():
    segs = pick_segments(FULL, FPS)
    total = sum(s[1] for s in segs)
    assert total == pytest.approx(GIF["seconds"] * FPS, abs=FPS)


def test_totality_only_redistributes_the_budget():
    segs = pick_segments(
        frames([(270, RESOLVE), (135, BEAD), (600, CORONA)]), FPS)
    assert names(segs) == ["the filter coming off", "second contact",
                           "totality"]
    assert sum(s[1] for s in segs) == pytest.approx(GIF["seconds"] * FPS,
                                                    abs=FPS)


def test_partials_only_still_produces_one_segment():
    segs = pick_segments(frames([(900, FILTERED)]), FPS)
    assert names(segs) == ["partial phases"]


def test_no_frames_falls_back_rather_than_failing():
    assert pick_segments({}, FPS)
    assert pick_segments({"frames": []}, FPS)


def test_short_sequence_is_not_over_read():
    short = frames([(40, FILTERED)])
    for start, count, _ in pick_segments(short, FPS):
        assert start + count <= 40


def test_chapter_below_the_floor_is_dropped_not_flashed():
    """A budget too small to give every chapter min_chapter_s must drop some
    rather than emit a handful of frames that reads as a glitch."""
    spec = dict(GIF, seconds=2.0, min_chapter_s=1.0)
    segs = pick_segments(FULL, FPS, spec)
    assert segs
    assert all(s[1] >= spec["min_chapter_s"] * FPS for s in segs)
