"""Build the aligned/unaligned comparison in the README, end to end.

    python tools/make_alignment_demo.py --out <out-dir> --data <captures>

Renders the same frames twice from one finished `timelapse.json` — once through
the disc track, once through a window pinned to the median centre and never
moved — then composes them side by side and encodes a GIF. It is what produced
`docs/media/alignment.gif`, and exists so that asset can be regenerated rather
than being a picture nobody can reproduce.

Three details are load-bearing:

  - THE TWO CONFIGS DIFFER IN ONE FIELD. Same frames, same gains, same stack
    depths, same window; only `cx`/`cy`. Anything else and the comparison stops
    being about the crop.
  - FRAMES ARE PAIRED BY SEQUENCE NUMBER, never by position in a sorted
    listing. The renderer fills 24 shards spread across the video rather than
    working front to back, so two runs at different stages have completely
    different frames on disk and pairing by position puts a totality frame
    beside a partial one.
  - INSETS COME OFF BOTH SIDES. They are drawn content at hard edges; left in,
    they are the loudest thing in the frame and the eye follows them instead of
    the Sun.

Rendered at drizzle 1 by default. The GIF's panels are a few hundred px wide, so
a full-size render is thrown away in the downsample — this is four times cheaper
and indistinguishable.
"""

import argparse
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

SEQ = re.compile(r"^seq_(\d+)\.png$")


def build_configs(out_dir, work, frames_root, drizzle):
    """Two configs and an eclipse.toml, differing only in the crop centre."""
    os.makedirs(os.path.join(work, "configs"), exist_ok=True)
    shutil.copy(os.path.join(out_dir, "survey.json"),
                os.path.join(work, "survey.json"))

    toml = io.open(os.path.join(out_dir, "eclipse.toml"),
                   encoding="utf-8").read()
    lines, seen = [], False
    for ln in toml.splitlines():
        if ln.strip().startswith("drizzle") and not seen:
            ln, seen = "drizzle = %d" % drizzle, True
        lines.append(ln)
    io.open(os.path.join(work, "eclipse.toml"), "w",
            encoding="utf-8").write("\n".join(lines) + "\n")

    with io.open(os.path.join(out_dir, "configs", "timelapse.json"),
                 encoding="utf-8-sig") as f:
        base = json.load(f)

    xs = [f["cx"] for f in base["frames"]]
    ys = [f["cy"] for f in base["frames"]]
    cx0, cy0 = statistics.median(xs), statistics.median(ys)
    print("%d frames, window %dx%d" % (len(base["frames"]), base["outW"],
                                       base["outH"]))
    print("the Sun travels %.0f x %.0f plane px = %.0f%% x %.0f%% of the window"
          % (max(xs) - min(xs), max(ys) - min(ys),
             100 * (max(xs) - min(xs)) / base["outW"],
             100 * (max(ys) - min(ys)) / base["outH"]))

    for tag, fixed in (("aligned", False), ("unaligned", True)):
        cfg = json.loads(json.dumps(base))
        cfg["outDir"] = "%s/%s" % (frames_root, tag)
        cfg.pop("insetPanel", None)
        for fr in cfg["frames"]:
            fr.pop("insets", None)
            if fixed:
                fr["cx"], fr["cy"] = cx0, cy0
        cfg["sunFixed"] = not fixed
        with io.open(os.path.join(work, "configs", "%s.json" % tag), "w",
                     encoding="utf-8") as f:
            json.dump(cfg, f)


def render(work, frames_root, data, workers):
    for tag in ("unaligned", "aligned"):
        print("rendering %s" % tag)
        subprocess.run([sys.executable, "-m", "ecl.tl_render",
                        "--config", os.path.join(work, "configs",
                                                 "%s.json" % tag),
                        "--data-dir", data,
                        "--out-dir", "%s/%s" % (frames_root, tag),
                        "--workers", str(workers)], check=True)


def compose(frames_root, staged, panel_w, step, labels):
    from PIL import Image, ImageDraw, ImageFont

    def by_seq(d):
        return {int(SEQ.match(f).group(1)): os.path.join(d, f)
                for f in os.listdir(d) if SEQ.match(f)}

    L = by_seq(os.path.join(frames_root, "unaligned"))
    R = by_seq(os.path.join(frames_root, "aligned"))
    keys = sorted(set(L) & set(R))[::step]
    if not keys:
        raise SystemExit("no frames in common")

    w0, h0 = Image.open(L[keys[0]]).size
    ph = int(round(h0 * panel_w / w0))
    bar = max(18, panel_w // 16)
    gap = 2
    W, H = panel_w * 2 + gap, ph + bar
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf",
                                  int(bar * 0.62))
    except OSError:
        font = ImageFont.load_default()

    os.makedirs(staged, exist_ok=True)
    for i, k in enumerate(keys):
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        for x0, src in ((0, L[k]), (panel_w + gap, R[k])):
            im = Image.open(src).convert("RGB").resize((panel_w, ph),
                                                       Image.LANCZOS)
            sheet.paste(im, (x0, bar))
        d = ImageDraw.Draw(sheet)
        d.rectangle([0, 0, W, bar - 1], fill=(16, 16, 16))
        for x0, text in ((0, labels[0]), (panel_w + gap, labels[1])):
            tw = d.textlength(text, font=font)
            d.text((x0 + (panel_w - tw) / 2, bar * 0.18), text, font=font,
                   fill=(235, 235, 235))
        d.line([(panel_w, 0), (panel_w, H)], fill=(60, 60, 60), width=gap)
        sheet.save(os.path.join(staged, "seq_%05d.png" % i))
    print("composed %d pairs at %dx%d" % (len(keys), W, H))
    return len(keys)


def encode(staged, out_gif, fps, colors):
    """Palette pass then apply, as `ecl.encode` does and for the same reason:
    the single-pass web palette bands a corona visibly."""
    from ecl.encode import _ffmpeg, _run

    pal = os.path.join(staged, "palette.png")
    src = ["-framerate", str(fps), "-i", "%s/seq_%%05d.png" % staged]
    _run([_ffmpeg(), "-y", *src,
          "-vf", "palettegen=max_colors=%d:stats_mode=diff" % colors, pal], pal)
    _run([_ffmpeg(), "-y", *src, "-i", pal,
          "-lavfi", "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=4",
          "-loop", "0", out_gif], out_gif)
    print("wrote %s (%.1f MB)" % (out_gif, os.path.getsize(out_gif) / 1e6))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="a finished run's output dir (holds configs/, "
                         "eclipse.toml, survey.json)")
    ap.add_argument("--data", required=True, help="the captures")
    ap.add_argument("--work", default=None,
                    help="where the two configs go (default <out>/align-demo)")
    ap.add_argument("--frames", default=None,
                    help="where the two renders go (default <work>/frames)")
    ap.add_argument("--gif", default="alignment.gif")
    ap.add_argument("--drizzle", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--panel-w", type=int, default=320)
    ap.add_argument("--step", type=int, default=18,
                    help="use every Nth frame pair")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--skip-render", action="store_true",
                    help="frames are already there; just compose and encode")
    args = ap.parse_args(argv)

    work = args.work or os.path.join(args.out, "align-demo")
    frames_root = args.frames or os.path.join(work, "frames")
    workers = args.workers or (os.cpu_count() or 4)

    if not args.skip_render:
        build_configs(args.out, work, frames_root, args.drizzle)
        render(work, frames_root, args.data, workers)
    staged = os.path.join(work, "composed")
    compose(frames_root, staged, args.panel_w, args.step,
            ("NO TRACKING", "DISC TRACK"))
    encode(staged, args.gif, args.fps, args.colors)


if __name__ == "__main__":
    main()
