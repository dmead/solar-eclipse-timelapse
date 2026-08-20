"""Encode the rendered frames to video — port of run-timelapse.mjs's encode half
and encode-deliverables.mjs.

Nothing here ever touched PixInsight; it was Node only because the driver that
launched PixInsight happened to be Node. With the renderer in Python the last
reason to keep a Node dependency goes away.

Three video cuts and a GIF. The middle cut is the fussy one: Instagram
re-encodes anything it is not happy with, and a re-encode of a black frame with
a thin bright corona is exactly where its encoder produces blocking. A capped
bitrate with a closed 2 s GOP and no scene-cut detection keeps the file inside
what its pipeline passes through untouched.

The GIF exists because that is what a README can actually play. GitHub strips
`<video>` out of rendered markdown — its own /markdown API returns the
surrounding text with nothing between — while an animated GIF comes back tagged
`data-animated-image` and plays inline. So the one format that will not embed is
the one every encoder here produces, and a short GIF is the fix.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from . import paths

__all__ = ["encode", "encode_gif", "encode_deliverables", "pick_segments",
           "runs_where", "tune", "CUTS", "GIF", "CHAPTERS"]

CUTS = [
    {"name": "timelapse.mp4", "vf": None,
     "extra": ["-crf", "17", "-preset", "slow"]},
    {"name": "timelapse_instagram.mp4",
     "vf": "scale=1080:-2:flags=lanczos",
     "extra": ["-crf", "18", "-preset", "slow",
               "-profile:v", "high", "-level", "4.0",
               "-maxrate", "8M", "-bufsize", "16M",
               "-g", "60", "-keyint_min", "60",
               "-x264-params", "scenecut=0:open-gop=0"]},
    {"name": "timelapse_preview.mp4",
     "vf": "scale=960:-2:flags=lanczos",
     "extra": ["-crf", "26", "-preset", "slow"]},
]

# The GIF preview. Not a cut of the whole video: 74 seconds at a size worth
# looking at runs to hundreds of megabytes in a format with no interframe
# compression, and a six-second excerpt of one moment is not a preview of an
# eclipse either. It is a handful of segments, chosen from what the earlier
# passes already marked in the config — see `pick_segments`.
GIF = {
    "name": "timelapse.gif",
    "width": 480,
    "fps": 10,
    "seconds": 9.0,           # total screen time, split across the chapters
    "min_chapter_s": 1.0,     # below this a chapter reads as a glitch; drop it
    # The subject is a white corona on black sky; the palette is spent on the
    # tones between, and 64 entries covers them. `stats_mode=diff` weights it
    # toward what actually changes rather than the acres of unchanging sky.
    "colors": 64,
}


def _partials(f):
    return f.get("state") == "filtered" and not f.get("resolve")


def _resolve(f):
    return bool(f.get("resolve"))


def _beads(f):
    return bool(f.get("bead"))


def _corona(f):
    return (f.get("state") == "unfiltered" and not f.get("bead")
            and not f.get("resolve"))


# What a preview of a total eclipse has to contain, in order, with how the clip
# is taken from each. The predicates read the marks the earlier passes left in
# the config; nothing here knows a frame number or a timestamp.
#
# WEIGHT is share of the screen-time budget. WHERE says which part of the chosen
# run to take, and each has a reason: the partial phases are worth seeing at
# their thinnest, so take the END of the run that leads into totality; the
# filter coming off and second contact are events, so take the START of them;
# the corona just sits there, so take the MIDDLE, away from the exposure changes
# at either end.
#
# A chapter whose predicate matches nothing is skipped and its budget goes to
# the others. Data that is totality only gets two chapters, data with no
# totality at all gets one, and neither case is special-cased anywhere.
CHAPTERS = [
    ("partial phases",        _partials, 1.0, "end"),
    ("the filter coming off", _resolve,  1.0, "start"),
    ("second contact",        _beads,    1.5, "start"),
    ("totality",              _corona,   1.5, "middle"),
]


def tune(out_dir, log=None):
    """Resolve the GIF settings from the config, if there is one."""
    from .params import load

    P = load(out_dir, create=False)
    for k in ("width", "fps", "seconds", "min_chapter_s", "colors"):
        GIF[k] = P.get("gif." + k, GIF[k])
    if log:
        log(f"  gif {GIF['width']}px {GIF['fps']}fps, "
            f"{GIF['seconds']:.0f}s budget")
    return P


def runs_where(frames, pred):
    """Contiguous [start, count) runs of frames satisfying `pred`."""
    out, start = [], None
    for i, f in enumerate(frames):
        if pred(f):
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(frames) - start))
    return out


def _take(run, n, where):
    """`n` frames from a (start, count) run, at one end or the middle."""
    start, count = run
    n = min(n, count)
    if where == "start":
        return start, n
    if where == "end":
        return start + count - n, n
    return start + (count - n) // 2, n


def pick_segments(cfg, fps, spec=GIF, log=None):
    """[(start_seq, count, name)] for the preview, in time order.

    The budget is split by weight across whichever chapters this data actually
    has, so the same rules give four segments for a full shoot and one for a
    folder of partial phases. Frames carry no `seq` of their own: a frame's
    sequence number is its index in the config, which is what the renderer
    numbers by too.
    """
    frames = (cfg or {}).get("frames") or []
    budget = int(round(spec["seconds"] * fps))
    if not frames:
        return [(0, budget, "the start")]

    present = []
    for name, pred, weight, where in CHAPTERS:
        rs = runs_where(frames, pred)
        if rs:
            present.append((name, rs, weight, where))
    if not present:
        return [(0, min(budget, len(frames)), "the whole sequence")]

    total_w = sum(w for _n, _r, w, _p in present)
    floor = int(round(spec["min_chapter_s"] * fps))
    out = []
    for name, rs, weight, where in present:
        want = int(round(budget * weight / total_w))
        if want < floor:
            continue
        # Which run: the partial phases are the ones that LEAD INTO totality,
        # not whichever happens to be longest. A shoot that resumes after third
        # contact has a second run of them, longer in this data than the run
        # before totality - picking by length opened the preview on the wrong
        # side of the eclipse and told the story backwards.
        if name == "partial phases":
            t0 = _first_totality(frames)
            before = [r for r in rs if r[0] + r[1] <= t0]
            # The run that ends nearest totality, of those that end before it.
            run = (max(before, key=lambda r: r[0] + r[1]) if before
                   else max(rs, key=lambda r: r[1]))
        else:
            run = max(rs, key=lambda r: r[1])
        seg = _take(run, want, where)
        if seg[1] >= floor:
            out.append((seg[0], seg[1], name))

    out.sort(key=lambda s: s[0])
    if log:
        for st, n, name in out:
            log(f"    {name:<24s} seq {st}-{st + n - 1}  ({n / fps:.1f}s)")
    return out or [(0, min(budget, len(frames)), "the whole sequence")]


def _first_totality(frames):
    for i, f in enumerate(frames):
        if f.get("state") == "unfiltered":
            return i
    return len(frames)


_SEQ = re.compile(r"^seq_(\d+)\.png$")


def count_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(frame_dir)
    return sum(1 for f in os.listdir(frame_dir) if _SEQ.match(f))


def first_frame(frame_dir):
    """Lowest sequence number present, or 0.

    Frames keep their position in the WHOLE video even when only part of it was
    rendered, so a totality-only render starts at seq 638 and not at zero.
    ffmpeg's numbered-sequence reader begins at 0 and stops at the first gap, so
    without telling it where to start it finds nothing at all and writes an empty
    file rather than failing.
    """
    ns = [int(_SEQ.match(f).group(1)) for f in os.listdir(frame_dir)
          if _SEQ.match(f)]
    return min(ns) if ns else 0


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not on PATH")
    return exe


def encode(frame_dir, out_path, fps, vf=None, extra=None, log=print,
           start_number=None):
    """One cut. Frames are read as a numbered sequence, never a glob — this
    ffmpeg build has no glob support, which is why the renderer numbers frames
    by their position in the video rather than within their shard."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if start_number is None:
        start_number = first_frame(frame_dir)
    cmd = [_ffmpeg(), "-y", "-framerate", str(fps),
           "-start_number", str(start_number),
           "-i", f"{frame_dir}/seq_%05d.png"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-an", "-c:v", "libx264", *(extra or []),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]

    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed on {out_path} "
                           f"(exit {r.returncode})\n{tail}")
    mb = os.path.getsize(out_path) / 1e6
    log(f"  {os.path.basename(out_path):26s} {mb:8.1f} MB  "
        f"({time.time() - t0:.0f}s)")
    return out_path


def encode_gif(frame_dir, out_path, fps, segments, spec, log=print):
    """One GIF from a list of (start, count, name) segments, in two passes.

    THE SEGMENTS ARE STAGED BEFORE ENCODING. ffmpeg reads a numbered sequence
    and stops at the first gap, so disjoint spans cannot be handed to it as one
    input; the alternative, a `select` expression over the whole sequence,
    decodes all 2228 full-size frames to keep ninety of them. Hard-linking the
    chosen frames into a temp directory, renumbered contiguously, costs no bytes
    on the same volume and leaves ffmpeg reading exactly what it will use.

    Then a palette pass and an apply pass. A single pass would quantise to the
    standard 216-colour web palette, which on a corona is visible banding in the
    one place the picture has any gradient. The palette is built from the staged
    frames, so it is spent on the excerpt rather than on the whole video.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="eclgif")
    try:
        n_src = _stage(frame_dir, tmp, segments)
        if not n_src:
            raise RuntimeError(f"no frames on disk for the preview segments "
                               f"in {frame_dir}")
        n_out = max(1, int(round(n_src * spec["fps"] / float(fps))))
        chain = f"fps={spec['fps']},scale={spec['width']}:-1:flags=lanczos"
        src = ["-framerate", str(fps), "-start_number", "0",
               "-i", f"{tmp}/seq_%05d.png"]
        palette = os.path.join(tmp, "palette.png")
        _run([_ffmpeg(), "-y", *src, "-frames:v", str(n_out),
              "-vf", f"{chain},palettegen=max_colors={spec['colors']}"
                     ":stats_mode=diff", palette], palette)
        _run([_ffmpeg(), "-y", *src, "-i", palette, "-frames:v", str(n_out),
              "-lavfi", f"{chain}[x];[x][1:v]paletteuse=dither=bayer"
                        ":bayer_scale=5", "-loop", "0", out_path], out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    mb = os.path.getsize(out_path) / 1e6
    log(f"  {os.path.basename(out_path):26s} {mb:8.1f} MB  "
        f"({time.time() - t0:.0f}s, {n_out} frames, {len(segments)} segment(s))")
    return out_path


def _stage(frame_dir, tmp, segments):
    """Hard-link the segments' frames into `tmp`, renumbered from zero.

    Falls back to copying where a link cannot be made — a different volume, or a
    filesystem without them. Frames the renderer never wrote are skipped rather
    than raising: a run that dropped frames should still get a preview.
    """
    k = 0
    for start, count, _name in segments:
        for i in range(start, start + count):
            src = os.path.join(frame_dir, "seq_%05d.png" % i)
            if not os.path.exists(src):
                continue
            dst = os.path.join(tmp, "seq_%05d.png" % k)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
            k += 1
    return k


def _run(cmd, out_path):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed on {out_path} "
                           f"(exit {r.returncode})\n{tail}")


def encode_deliverables(frame_dir, out_dir, fps, cuts=None, log=print,
                        cfg=None, gif=GIF):
    n = count_frames(frame_dir)
    if not n:
        raise ValueError(f"no frames in {frame_dir}")
    cuts = CUTS if cuts is None else cuts
    start = first_frame(frame_dir)
    log(f"{n} frames at {fps} fps -> {len(cuts) + (1 if gif else 0)} outputs"
        + (f" (starting at seq {start})" if start else ""))
    out = [encode(frame_dir, f"{out_dir}/{c['name']}", fps,
                  c["vf"], c["extra"], log=log, start_number=start)
           for c in cuts]
    if gif:
        log("  gif preview segments:")
        segments = pick_segments(cfg, fps, gif, log=log)
        out.append(encode_gif(frame_dir, f"{out_dir}/{gif['name']}", fps,
                              segments, gif, log=log))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=paths.in_out("configs", "timelapse.json"))
    ap.add_argument("--frames", default=None,
                    help="frame directory (default: the config's outDir)")
    ap.add_argument("--out-dir", default=paths.in_out("final"))
    ap.add_argument("--only", default=None,
                    help="comma-separated output names, e.g. timelapse.mp4")
    ap.add_argument("--no-gif", action="store_true",
                    help="skip the GIF preview")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args(argv)

    # <out>/configs/timelapse.json -> <out>, where eclipse.toml lives.
    try:
        tune(os.path.dirname(os.path.dirname(os.path.abspath(args.config))),
             log=print)
    except SystemExit:
        print("  no survey/config found - using built-in GIF defaults")

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    frames = args.frames or cfg["outDir"]
    fps = args.fps or cfg.get("fps", 30)
    cuts, gif = CUTS, (None if args.no_gif else GIF)
    if args.only:
        want = set(args.only.split(","))
        cuts = [c for c in CUTS if c["name"] in want]
        if gif and GIF["name"] not in want:
            gif = None
        if not cuts and not gif:
            raise SystemExit(f"no output matches {args.only}")

    encode_deliverables(frames, args.out_dir, fps, cuts, cfg=cfg, gif=gif)


if __name__ == "__main__":
    main()
