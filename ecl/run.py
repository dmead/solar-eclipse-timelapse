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
    ("segment", "ecl.segment", "segments.json - exposure states, frame exact"),
    ("beads", "ecl.beadwindow", "diag/beads.json - the Baily's bead window"),
    ("select", "ecl.gen_timelapse", "configs/timelapse.json - frames, gains, dwells"),
    ("centres", "ecl.tl_centres", "diag/centres.json - a disc centre per frame"),
    ("track", "ecl.tl_track", "diag/corona_track.json - totality pointing"),
    ("drift", "ecl.tl_drift", "diag/drift.json - where the Sun is behind the Moon"),
    ("smooth", "ecl.smooth_track", "cx/cy on every frame; drops untrustworthy ones"),
    ("insets", "ecl.gen_insets", "insets on every frame"),
    ("phases", "ecl.phases", "diag/phases.json - the contacts, and a phase per frame"),
    ("render", "ecl.tl_render", "frames/*.png"),
    ("encode", "ecl.encode", "final/*.mp4"),
]


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


def unbuffer():
    """Make this run readable when it is piped to a file.

    Python block-buffers stdout when it is not a terminal, and a long render is
    always watched through a file or a pipe. The parent's progress lines then
    land thousands of characters after the child stderr they were meant to
    introduce: a failing pass shows its traceback ABOVE the header naming the
    pass, and above the `data/out/frames` banner printed before anything ran.
    The error was never lost, but it reads as though the run died without
    saying anything.

    Line-buffering this process and unbuffering the children puts the log back
    in the order the events happened.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    os.environ["PYTHONUNBUFFERED"] = "1"


def _run(mod, args, log):
    """Run one pass as `python -m <module>`, in its own process.

    Every pass is a module in this package rather than a loose script beside
    it. It used to be a mix: three passes were run by PATH, from the directory
    holding `ecl/`, which works in an editable checkout and nowhere else - an
    installed wheel puts `ecl` in site-packages and there is no script there to
    find. Nothing here depends on the working directory any more.

    They stay separate PROCESSES on purpose. Each pass holds hundreds of frames
    of numpy at once, and a process boundary is what actually returns that
    memory to the OS between passes.
    """
    cmd = [sys.executable, "-m", mod] + [str(a) for a in args]
    log("    " + " ".join(cmd[1:]))
    sys.stdout.flush()
    rc = subprocess.run(cmd).returncode
    sys.stdout.flush()
    if rc != 0:
        # Name the command as well as the pass. A pass fails inside a
        # subprocess, so the traceback above belongs to a different process and
        # nothing else in the log connects the two.
        raise SystemExit(
            f"\npass failed: {mod} (exit {rc})\n"
            f"  the traceback above is from this command:\n"
            f"    {' '.join(cmd[1:])}\n"
            f"  re-run it on its own to see it without the other passes "
            f"interleaved.")


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

    unbuffer()
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
        if name in ("segment", "beads", "drift", "insets"):
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

        if name == "select":
            """
            Record where the frames actually go.

            The planner writes `outDir` from its own argument, and every caller
            here passes --frames separately, so the two could disagree forever
            without anyone noticing: `ecl.progress` with no arguments believed
            the config and counted 1660 frames in a directory left over from an
            earlier run, when the answer was 2228 somewhere else.
            """
            cfgp = os.path.join(out, "configs", "timelapse.json")
            with open(cfgp, encoding="utf-8-sig") as fh:
                cfg = json.load(fh)
            if cfg.get("outDir") != frames:
                cfg["outDir"] = frames
                with open(cfgp, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh)
                print(f"  outDir -> {frames}")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
