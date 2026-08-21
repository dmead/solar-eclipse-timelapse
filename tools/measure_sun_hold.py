"""How still does the rendered video actually hold the Sun?

    python tools/measure_sun_hold.py --out <out> --frames <dir>

Detects the limb in the RENDERED frames — not in the pipeline's own numbers,
which would only confirm that it agrees with itself — and reports how far the
Sun sits from frame centre through each phase. Zero means held still.

WHY IT DOES NOT SIMPLY MEASURE THE DISC. During the partial phases the visible
edge is the Sun's limb, and measuring it is the answer. During totality the
visible edge is the MOON, which is a different object sitting up to 13 px away,
and once the crop is correctly centred on the Sun the Moon is SUPPOSED to be
off-centre by exactly that. Measuring the Moon therefore makes a correct fix
look like a regression — which it duly did, for several rounds, before this
script existed. So the Moon's measured position is converted:

    sun_in_frame = moon_in_frame - (moon - sun) offset at that instant

using the track from `ecl.tl_drift`. The two phases then become directly
comparable, and the step between them is the thing to minimise.

The detector searches only for the CENTRE of a circle whose radius is known from
the survey, which is far better conditioned than fitting three parameters to
edge points: on a partial phase most rays cross empty sky, and a free fit to
that returns 12-100 px of residual and a radius that wanders between the Sun's
and the Moon's.
"""

import argparse
import datetime
import json
import os
import re

import numpy as np
from PIL import Image

SEQ = re.compile(r"^seq_(\d+)\.png$")
N_TH = 360
MIN_RESPONSE = 0.06        # below this the search found no edge at all


def load_plane(path, drizzle):
    im = Image.open(path).convert("L")
    w, h = im.size
    if drizzle > 1:
        im = im.resize((w // drizzle, h // drizzle), Image.BOX)
    return np.asarray(im, np.float64) / 255.0


def ring_response(img, cx, cy, R, dr=3.0, signed=0):
    """Mean radial step across a circle: outside minus inside.

    `signed` +1 keeps dark-inside/bright-outside (the Moon against the corona),
    -1 keeps bright-inside/dark-outside (the Sun against sky). Polarity is what
    separates the two limbs when both are in frame.
    """
    th = np.linspace(0, 2 * np.pi, N_TH, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    h, w = img.shape

    def at(rad):
        x = np.clip((cx + ct * rad).astype(int), 0, w - 1)
        y = np.clip((cy + st * rad).astype(int), 0, h - 1)
        return img[y, x]

    step = at(R + dr) - at(R - dr)
    if signed > 0:
        step = np.clip(step, 0, None)
    elif signed < 0:
        step = np.clip(-step, 0, None)
    else:
        step = np.abs(step)
    return step.mean()


def find_centre(img, R, cx0, cy0, signed=0, span=70):
    cx, cy, best = cx0, cy0, -1e9
    for step in (8, 3, 1):
        rng = np.arange(-span, span + 1, step)
        bx, by = cx, cy
        for dy in rng:
            for dx in rng:
                v = ring_response(img, cx + dx, cy + dy, R, signed=signed)
                if v > best:
                    best, bx, by = v, cx + dx, cy + dy
        cx, cy = bx, by
        span = max(step * 2, 3)
    return cx, cy, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the run's output directory")
    ap.add_argument("--frames", required=True, help="the rendered frames")
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--drizzle", type=int, default=2)
    args = ap.parse_args(argv)

    cfg = json.load(open(os.path.join(args.out, "configs", "timelapse.json"),
                         encoding="utf-8-sig"))
    det = json.load(open(os.path.join(args.out, "diag", "centres.json"),
                         encoding="utf-8-sig"))
    r_sun = float(det.get("rSun") or 279.0)
    r_moon = float(det.get("rMoon") or 292.0)
    cx0, cy0 = cfg["outW"] / 2.0, cfg["outH"] / 2.0

    track = None
    dpath = os.path.join(args.out, "diag", "drift.json")
    if os.path.exists(dpath):
        track = json.load(open(dpath, encoding="utf-8-sig"))["track"]
        man = json.load(open(os.path.join(args.out, "segments.json"),
                             encoding="utf-8-sig"))
        fps = {f["name"]: f["fps"] for f in man["files"]}
        t0 = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"])
              for f in man["files"]}
        ep = min(t0.values())
    else:
        print("no diag/drift.json - totality will be reported as the MOON's\n"
              "position, which is not the same question. Run ecl.tl_drift.")

    groups = {}
    for i in range(0, len(cfg["frames"]), args.step):
        f = cfg["frames"][i]
        if f.get("resolve"):
            continue
        p = os.path.join(args.frames, "seq_%05d.png" % i)
        if not os.path.exists(p):
            continue
        moon = f["state"] == "unfiltered"
        R, sign = (r_moon, +1) if moon else (r_sun, -1)
        img = load_plane(p, args.drizzle)
        cx, cy, v = find_centre(img, R, cx0, cy0, signed=sign)
        if v < MIN_RESPONSE:
            continue
        dx, dy = cx - cx0, cy - cy0
        if moon and track is not None:
            t = ((t0[f["file"]] - ep).total_seconds()
                 + f["index"] / fps[f["file"]])
            dx -= float(np.polyval(track["dx"], t))
            dy -= float(np.polyval(track["dy"], t))
        key = ("totality" if moon else
               ("partials before" if i < len(cfg["frames"]) // 2
                else "partials after"))
        groups.setdefault(key, []).append((dx, dy))

    print("\ndistance of the SUN from frame centre (plane px; 0 = held still)")
    print("%-18s %5s %9s %9s %9s" % ("", "n", "median", "mean", "sd"))
    med = {}
    for key in ("partials before", "totality", "partials after"):
        v = groups.get(key)
        if not v:
            continue
        a = np.array(v)
        d = np.hypot(a[:, 0], a[:, 1])
        med[key] = np.median(a, axis=0)
        print("%-18s %5d %9.1f %9.1f %9.1f"
              % (key, len(a), np.median(d), d.mean(), d.std()))

    if "totality" in med:
        for side in ("partials before", "partials after"):
            if side in med:
                s = np.hypot(*(med["totality"] - med[side]))
                print("step %s <-> totality: %.1f plane px (%.1f in the render)"
                      % (side, s, args.drizzle * s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
