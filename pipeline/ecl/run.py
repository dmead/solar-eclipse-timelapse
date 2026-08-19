"""One command, one required argument: the data directory.

    python -m ecl.run "D:\\eclipse\\data"

Surveys the data, writes a config if there is not one, then runs every pass in
order. Each pass is resumable and skippable, because they cost minutes to hours
and the reason to re-run one is almost never a reason to re-run all of them.

WHY THE ORDER IS FIXED AND WHY IT STARTS AT THE TOP. Every pass after the first
rewrites `configs/timelapse.json` in place, adding what it measured to what the
last one left. Running a later pass against a config an earlier one has already
rewritten is the single most common way to break this pipeline - it has happened
twice, once by re-running the track smoother over its own output and once by two
chains racing on the same file. `--from` exists, and it re-runs everything from
that pass onward rather than only that pass.

Outputs land beside the data by default, in `<data>/../out`, so a read-only or
network data directory still works.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from . import params, survey as survey_mod

__all__ = ["PASSES", "main"]

# name, module-or-script, what it produces
PASSES = [
    ("survey", None, "survey.json - sensor, radius, memory plan"),
    ("beads", "ecl.beadwindow", "diag/beads.json - the Baily's bead window"),
    ("select", "gen_timelapse.py", "configs/timelapse.json - frames, gains, dwells"),
    ("centres", "ecl.tl_centres", "diag/centres.json - a disc centre per frame"),
    ("track", "ecl.tl_track", "diag/corona_track.json - totality pointing"),
    ("smooth", "smooth_track.py", "cx/cy on every frame; drops untrustworthy ones"),
    ("insets", "gen_insets.py", "insets on every frame"),
    ("render", "ecl.tl_render", "frames/*.png"),
    ("encode", "ecl.encode", "final/*.mp4"),
]

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preflight(log=print):
    """Fail in seconds on a missing dependency, not an hour into a render.

    Everything here is a declared PyPI dependency, so a correct install passes
    this trivially - it exists for the half-installed case. ffmpeg is the one
    worth checking hard: it is not a Python package, it is only used by the last
    pass, and without this an absent binary surfaces after the entire render.
    """
    import shutil

    missing = []
    for mod, hint in (("cv2", "opencv-python-headless"), ("psutil", "psutil"),
                      ("numpy", "numpy"), ("scipy", "scipy"), ("PIL", "pillow"),
                      ("skimage", "scikit-image"), ("tifffile", "tifffile")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"{mod} - python -m pip install {hint}")
    if not shutil.which("ffmpeg"):
        missing.append("\n".join([
            "ffmpeg is not on PATH - winget install --id Gyan.FFmpeg -e",
            "      (needed only by the final encode pass)",
        ]))
    if missing:
        log("missing prerequisites:")
        for m in missing:
            log(f"  - {m}")
        raise SystemExit(1)


def _run(mod, args, log):
    cmd = [sys.executable]
    cmd += ["-m", mod] if not mod.endswith(".py") else [os.path.join(HERE, mod)]
    cmd += args
    log("    " + " ".join(str(c) for c in cmd[1:]))
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"pass failed: {mod}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="passes: " + ", ".join(p[0] for p in PASSES))
    ap.add_argument("data_dir", help="folder of .ser files, or of images, or of "
                                     "one folder per capture")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <data_dir>/../out)")
    ap.add_argument("--frames", default=None,
                    help="where rendered PNGs go (default: <out>/frames)")
    ap.add_argument("--from", dest="start", default=None,
                    help="begin at this pass and run the rest")
    ap.add_argument("--only", default=None, help="run just these, comma separated")
    ap.add_argument("--sample", action="store_true",
                    help="render a representative subset for review, not the "
                         "whole video (see tl_render --sample)")
    ap.add_argument("--dry-run", action="store_true",
                    help="survey, write the config, print the plan, run nothing")
    args = ap.parse_args(argv)

    data = os.path.abspath(args.data_dir)
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(data), "out"))
    frames = os.path.abspath(args.frames or os.path.join(out, "frames"))
    os.makedirs(out, exist_ok=True)

    names = [p[0] for p in PASSES]
    want = names
    if args.only:
        want = [n.strip() for n in args.only.split(",")]
        bad = [n for n in want if n not in names]
        if bad:
            raise SystemExit(f"unknown pass(es) {bad}; known: {names}")
    elif args.start:
        if args.start not in names:
            raise SystemExit(f"unknown pass {args.start!r}; known: {names}")
        want = names[names.index(args.start):]

    preflight()

    t0 = time.time()
    print(f"data   {data}")
    print(f"out    {out}")
    print(f"frames {frames}\n")

    # The survey always runs when it is in the plan, and its results are what
    # the config resolves against.
    sv = None
    if "survey" in want:
        sv = survey_mod.survey(data)
        with open(os.path.join(out, "survey.json"), "w", encoding="utf-8") as f:
            json.dump(sv, f, indent=1)

    P = params.load(out, sv, log=print)
    print(f"\nradius {P.radius_px:.1f} px, drizzle x{P['render']['drizzle']}, "
          f"{P['render']['workers']} workers\n")

    if args.dry_run:
        for n, mod, what in PASSES:
            mark = "*" if n in want else " "
            print(f" {mark} {n:8s} {what}")
        print("\n(dry run - nothing executed)")
        return

    common = ["--out", out]
    for name, mod, _what in PASSES:
        if name not in want or mod is None:
            continue
        print(f"[{name}] {time.strftime('%H:%M:%S')}")
        a = list(common)
        if name in ("beads", "insets"):
            a += ["--data", data]
        elif name in ("centres", "track"):
            a = ["--data-dir", data, "--out",
                 os.path.join(out, "diag",
                              "centres.json" if name == "centres"
                              else "corona_track.json"),
                 "--config", os.path.join(out, "configs", "timelapse.json")]
        elif name == "render":
            a = ["--data-dir", data, "--out-dir", frames,
                 "--config", os.path.join(out, "configs", "timelapse.json")]
            if args.sample:
                a.append("--sample")
        elif name == "encode":
            a = ["--frames", frames, "--out-dir", os.path.join(out, "final"),
                 "--config", os.path.join(out, "configs", "timelapse.json")]
        _run(mod, a, print)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
