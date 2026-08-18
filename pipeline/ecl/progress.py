"""Progress view for a running render.

`tl_render` logs once per hundred frames PER SHARD, which is close to useless
while it runs: with the shards interleaved the lines arrive out of order, a
background job buffers them anyway, and the one number worth having - when it
will finish - is not in them at all.

The frames on disk are the honest progress record, so this counts those instead
and needs nothing from the running job. It works on a job started by any means,
survives being started and stopped, and can be pointed at a finished render to
get its timings back.

RATE IS MEASURED FROM THE RECENT PAST, not from the start. The first frames of a
run are far slower than the rest - cold cache, and every worker opening its first
SER - so an average over the whole run understates the rate badly and the ETA it
gives is wrong in the direction that matters. The window below is in frames, so
it adapts to whatever the current rate happens to be.

    python -m ecl.progress                       # one look
    python -m ecl.progress --watch               # refresh until done
"""

import argparse
import json
import math
import os
import re
import time

__all__ = ["scan", "render_progress"]

_SEQ = re.compile(r"^seq_(\d+)\.png$")

# Frames whose timestamps set the reported rate.
RATE_WINDOW = 120


def scan(frame_dir):
    """(seq, mtime) for every rendered frame, sorted by seq."""
    if not os.path.isdir(frame_dir):
        return []
    out = []
    with os.scandir(frame_dir) as it:
        for e in it:
            m = _SEQ.match(e.name)
            if m:
                try:
                    out.append((int(m.group(1)), e.stat().st_mtime))
                except OSError:
                    pass
    out.sort()
    return out


def _bar(frac, width=28):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def _hms(sec):
    if sec is None or sec < 0 or not math.isfinite(sec):
        return "  --:--"
    sec = int(sec)
    return f"{sec // 3600:3d}:{sec // 60 % 60:02d}:{sec % 60:02d}".lstrip()


def render_progress(config, frame_dir, workers, log=print):
    """One progress report. Returns True once every frame is on disk."""
    with open(config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    total = len(cfg["frames"])
    frame_dir = frame_dir or cfg["outDir"]
    done = scan(frame_dir)
    n = len(done)

    log(f"{os.path.basename(frame_dir)}  {n}/{total} frames "
        f"({100.0 * n / max(total, 1):.1f}%)  [{_bar(n / max(total, 1))}]")
    if not n:
        log("  nothing rendered yet")
        return False

    # Shards are contiguous and equal, the same split tl_render makes.
    step = math.ceil(total / max(workers, 1))
    got = {s for s, _ in done}
    rows = []
    for k in range(workers):
        a, b = k * step, min(total, (k + 1) * step)
        if a >= b:
            continue
        c = sum(1 for i in range(a, b) if i in got)
        rows.append((k, a, b, c))

    # A shard that is behind decides the finish time, not the average, because
    # the run is over only when the LAST one is.
    if len(rows) > 1:
        for k, a, b, c in rows:
            log(f"  shard {k:2d}  {a:5d}-{b - 1:<5d} {c:5d}/{b - a:<5d} "
                f"[{_bar(c / max(b - a, 1), 18)}]")

    ts = [t for _, t in done]
    ts.sort()
    win = ts[-RATE_WINDOW:] if len(ts) > 1 else ts
    span = win[-1] - win[0]
    rate = (len(win) - 1) / span if span > 0 else 0.0
    idle = time.time() - ts[-1]

    left = total - n
    log(f"  rate {rate * 60:.1f} frames/min ({1 / rate:.2f} s/frame)" if rate
        else "  rate unknown")
    if left <= 0:
        log(f"  DONE — {_hms(ts[-1] - ts[0])} wall from first frame to last")
        return True
    if rate > 0:
        # Slowest shard, since the job ends with it - and the per-shard rate is
        # shared out over the shards STILL RUNNING, not over all of them. Shards
        # finish at very different times here (the bead dwell is all stack=1 and
        # lands early, the corona shards each read twenty raw frames per output
        # frame), and counting the finished ones as active understates the rate
        # of the survivors and pushes the ETA out by the same factor.
        active = [r for r in rows if r[3] < r[2] - r[1]] or rows
        worst = max((b - a - c) for _, a, b, c in active)
        per = rate / len(active)
        log(f"  {left} left — eta {_hms(worst / per if per else None)} "
            f"(finish {time.strftime('%H:%M:%S', time.localtime(time.time() + worst / per))})"
            if per else f"  {left} left")
    if idle > 120:
        log(f"  !! no new frame for {_hms(idle)} — the job may have stopped")
    return False


def main(argv=None):
    from .tl_render import default_workers

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="S:/solar-eclipse/out/configs/timelapse.json")
    ap.add_argument("--frames", default=None, help="frame dir (default: cfg outDir)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="shard count the render was started with")
    ap.add_argument("--watch", action="store_true", help="refresh until finished")
    ap.add_argument("--every", type=float, default=30.0, help="--watch period, s")
    args = ap.parse_args(argv)

    while True:
        print(time.strftime("[%H:%M:%S]"))
        finished = render_progress(args.config, args.frames, args.workers)
        if finished or not args.watch:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
