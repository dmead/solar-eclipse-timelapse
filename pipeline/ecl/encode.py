"""Encode the rendered frames to video — port of run-timelapse.mjs's encode half
and encode-deliverables.mjs.

Nothing here ever touched PixInsight; it was Node only because the driver that
launched PixInsight happened to be Node. With the renderer in Python the last
reason to keep a Node dependency goes away.

Three cuts, and the middle one is the fussy one. Instagram re-encodes anything it
is not happy with, and a re-encode of a black frame with a thin bright corona is
exactly where its encoder produces blocking. A capped bitrate with a closed 2 s
GOP and no scene-cut detection keeps the file inside what its pipeline passes
through untouched.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time

__all__ = ["encode", "encode_deliverables", "CUTS"]

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

_SEQ = re.compile(r"^seq_\d+\.png$")


def count_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(frame_dir)
    return sum(1 for f in os.listdir(frame_dir) if _SEQ.match(f))


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not on PATH")
    return exe


def encode(frame_dir, out_path, fps, vf=None, extra=None, log=print):
    """One cut. Frames are read as a numbered sequence, never a glob — this
    ffmpeg build has no glob support, which is why the renderer numbers frames
    by their position in the video rather than within their shard."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-framerate", str(fps),
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


def encode_deliverables(frame_dir, out_dir, fps, cuts=None, log=print):
    n = count_frames(frame_dir)
    if not n:
        raise ValueError(f"no frames in {frame_dir}")
    cuts = cuts or CUTS
    log(f"{n} frames at {fps} fps -> {len(cuts)} cuts")
    return [encode(frame_dir, f"{out_dir}/{c['name']}", fps,
                   c["vf"], c["extra"], log=log) for c in cuts]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="S:/solar-eclipse/out/configs/timelapse.json")
    ap.add_argument("--frames", default=None,
                    help="frame directory (default: the config's outDir)")
    ap.add_argument("--out-dir", default="S:/solar-eclipse/out/final")
    ap.add_argument("--only", default=None,
                    help="comma-separated cut names, e.g. timelapse.mp4")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    frames = args.frames or cfg["outDir"]
    fps = args.fps or cfg.get("fps", 30)
    cuts = CUTS
    if args.only:
        want = set(args.only.split(","))
        cuts = [c for c in CUTS if c["name"] in want]
        if not cuts:
            raise SystemExit(f"no cut matches {args.only}")

    encode_deliverables(frames, args.out_dir, fps, cuts)


if __name__ == "__main__":
    main()
