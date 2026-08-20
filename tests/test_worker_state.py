"""The tuned render constants have to reach the worker processes.

`tune()` writes module globals. The render pool is SPAWNED, not forked, so on
Windows every worker re-imports the module and gets the built-in defaults unless
the parent hands its state over explicitly. That was silently true for a while:
the parent tuned itself, logged the tuned values, and then handed the rendering
to 24 processes that had never read the config.

The first test is the one that matters. It reads the `global` declarations out
of `tune` and fails if `TUNED` does not name exactly the same set — so adding a
setting to `tune()` and forgetting the workers breaks the suite rather than
quietly disabling that setting for everyone not running `--workers 1`.
"""

import ast
import inspect

from ecl import tl_render


def _globals_declared_in(func):
    """Names in every `global` statement in a function's source."""
    src = inspect.getsource(func)
    tree = ast.parse(src.lstrip())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


def test_TUNED_matches_what_tune_actually_writes():
    declared = _globals_declared_in(tl_render.tune)
    named = set(tl_render.TUNED)
    assert declared == named, (
        "tune() and TUNED disagree; workers would silently use defaults for: "
        f"{sorted(declared - named)}"
        + (f" / TUNED names nothing tune() writes: {sorted(named - declared)}"
           if named - declared else ""))


def test_tuned_state_round_trips():
    before = tl_render.tuned_state()
    assert set(before) == set(tl_render.TUNED)
    try:
        tl_render.apply_tuned(dict(before, DRIZZLE=99, GAMMA=0.5))
        assert tl_render.DRIZZLE == 99
        assert tl_render.GAMMA == 0.5
    finally:
        tl_render.apply_tuned(before)
    assert tl_render.DRIZZLE == before["DRIZZLE"]


def test_init_worker_adopts_the_parents_state():
    """The pool's initializer is the only chance a worker gets."""
    before = tl_render.tuned_state()
    try:
        tl_render._init_worker(state=dict(before, DRIZZLE=7))
        assert tl_render.DRIZZLE == 7
    finally:
        tl_render.apply_tuned(before)


def test_init_worker_without_state_changes_nothing():
    before = tl_render.tuned_state()
    tl_render._init_worker()
    assert tl_render.tuned_state() == before


def test_apply_tuned_clears_the_shoulder_cache():
    """The highlight curve is cached per parameter set; a worker adopting new
    shoulder values must not keep the shape solved for the old ones."""
    tl_render._shoulder_shape()
    assert tl_render._SHOULDER_CACHE
    tl_render.apply_tuned(tl_render.tuned_state())
    assert not tl_render._SHOULDER_CACHE
