"""Shard boundaries, and the bug they exist to prevent.

A resume drops the frames already on disk, so what reaches the sharder is several
separate runs of video with gaps between them. Slicing that list into equal pieces
puts a join inside a shard, and the renderer cross-dissolves across it against a
frame from a different part of the video. That shipped once: a ghost decaying over
the dissolve length, which reads as the Moon's limb stepping backwards.

The invariant is simply stated, so it is worth stating: every span the sharder
returns is contiguous in `seq`.
"""

import math

import pytest

from ecl.tl_render import shard_spans


def frames(seqs):
    return [{"seq": s} for s in seqs]


def spans_are_contiguous(fs, spans):
    for a, b in spans:
        for i in range(a + 1, b):
            if fs[i]["seq"] != fs[i - 1]["seq"] + 1:
                return False
    return True


def test_spans_tile_the_list_exactly():
    fs = frames(range(100))
    spans = shard_spans(fs, 8)
    assert spans[0][0] == 0 and spans[-1][1] == len(fs)
    for (_, b), (c, _) in zip(spans, spans[1:]):
        assert b == c, "spans must not overlap or leave a hole"


def test_full_render_is_unaffected():
    """No discontinuities means the old equal-slice behaviour, exactly."""
    fs = frames(range(2393))
    spans = shard_spans(fs, 24)
    step = math.ceil(2393 / 24)
    assert spans == [(a, min(a + step, 2393)) for a in range(0, 2393, step)]


def test_a_shard_never_spans_a_gap():
    # 17 runs of 13, the shape a kill actually left behind.
    seqs = []
    for r in range(17):
        seqs += list(range(r * 100, r * 100 + 13))
    fs = frames(seqs)
    spans = shard_spans(fs, 24)
    assert spans_are_contiguous(fs, spans)


@pytest.mark.parametrize("workers", [1, 2, 3, 7, 24, 64])
def test_contiguous_for_any_worker_count(workers):
    seqs = list(range(0, 40)) + list(range(200, 213)) + [999]
    fs = frames(seqs)
    spans = shard_spans(fs, workers)
    assert spans_are_contiguous(fs, spans)
    assert sum(b - a for a, b in spans) == len(fs)


def test_the_shipped_case_would_have_been_caught():
    """seq 176 was the end of one run and 177 the start of the next.

    Under the old rule they landed in one shard and the dissolve fired between
    them. Whatever else changes, they must not share a span.
    """
    fs = frames(list(range(100, 177)) + list(range(277, 300)))
    spans = shard_spans(fs, 24)
    join = next(i for i in range(1, len(fs))
                if fs[i]["seq"] != fs[i - 1]["seq"] + 1)
    assert any(a == join for a, _ in spans), "the join must start a new shard"


def test_empty_list():
    assert shard_spans([], 24) == []
