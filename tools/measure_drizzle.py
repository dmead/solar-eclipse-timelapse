"""Measure what drizzle actually recovers, on the features rather than the sky.

    python tools/measure_drizzle.py --d1 <frames> --d2 <frames> --config <cfg>

Compares a drizzle-1 render against a drizzle-2 render of the same frames by the
share of power ABOVE the drizzle-1 Nyquist: drizzle 2 native against drizzle 1
upscaled with Lanczos. A resampler cannot invent signal above the source
Nyquist, so anything up there in the drizzled version is detail drizzle
recovered. The numbers in `docs/STATE.md` came from this.

WHERE YOU MEASURE DECIDES THE ANSWER, and getting it wrong is easy. Measured over
a large crop this reports no benefit — 99.999% of the power in a frame of corona
sits below a quarter of Nyquist and swamps the signal. Measured over small
patches on the sunspot and the prominences, the only genuinely fine structure in
the data, the supra-Nyquist power rises a hundredfold and the answer reverses.

The features do not need detecting. The panel planner already located them, per
frame, and left them in the config as `insets`. The render holds the Sun at the
centre of the frame, so a feature sits at

    output = window/2 + (feature - disc centre)

in plane px, times the drizzle factor. Both renders share a window and a track,
so one expression places it in either.

Read the ratio per band, not the total. Noise sits flat across bands and
interpolation ringing favours the upscaled side; an advantage that GROWS with
frequency is what recovered detail looks like.

One caveat on any result from this: the renders are 8-bit and stretched, so
quantisation and the highlight shoulder both work against fine low-contrast
detail. Whatever it reports is a lower bound.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image


def radial_power(a):
    """Radially averaged power spectrum, normalised, windowed."""
    n = a.shape[0]
    a = a - a.mean()
    w = np.hanning(n)
    a = a * w[:, None] * w[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2, yy - n / 2).astype(int)
    prof = (np.bincount(r.ravel(), P.ravel())[:n // 2]
            / np.maximum(np.bincount(r.ravel())[:n // 2], 1))
    tot = prof[1:].sum()
    return prof / tot if tot else prof


def patches(frames, seq, label, d1, d2, patch):
    fr = frames[seq]
    hit = [i for i in (fr.get("insets") or []) if i.get("label") == label]
    if not hit:
        return None
    f = hit[0]
    out = []
    for root, scale in ((d1, 1.0), (d2, 2.0)):
        p = os.path.join(root, "seq_%05d.png" % seq)
        if not os.path.exists(p):
            return None
        im = Image.open(p).convert("L")
        W, H = im.size
        px = (W / scale) / 2.0 + (f["cx"] - fr["cx"])
        py = (H / scale) / 2.0 + (f["cy"] - fr["cy"])
        side = int(patch * scale / 2.0)
        x, y = int(px * scale), int(py * scale)
        if not (side < x < W - side and side < y < H - side):
            return None
        c = im.crop((x - side, y - side, x + side, y + side))
        if scale == 1.0:
            c = c.resize((patch, patch), Image.LANCZOS)
        out.append(np.asarray(c, np.float64) / 255.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d1", required=True, help="drizzle 1 frames")
    ap.add_argument("--d2", required=True, help="drizzle 2 frames")
    ap.add_argument("--config", required=True, help="the timelapse.json WITH "
                                                    "insets, for the features")
    ap.add_argument("--labels", default="sunspot,prominence")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=24,
                    help="frames per label")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        frames = json.load(f)["frames"]

    have = sorted(int(n[4:9]) for n in os.listdir(args.d1)
                  if n.startswith("seq_") and n.endswith(".png"))

    for label in args.labels.split(","):
        ups, his = [], []
        for seq in have:
            if len(ups) >= args.limit:
                break
            got = patches(frames, seq, label, args.d1, args.d2, args.patch)
            if got:
                ups.append(radial_power(got[0]))
                his.append(radial_power(got[1]))
        if not ups:
            print("%-12s no frames carry this feature" % label)
            continue
        U, H = np.mean(ups, axis=0), np.mean(his, axis=0)
        k = len(U)
        half = k // 2
        print("%s  (%d frames, %dx%d patch)"
              % (label.upper(), len(ups), args.patch, args.patch))
        print("  above the drizzle-1 Nyquist: d1 %.5f  d2 %.5f  ratio %.2fx"
              % (U[half:].sum(), H[half:].sum(),
                 H[half:].sum() / U[half:].sum() if U[half:].sum() else 0))
        print("  %-12s %10s %10s %8s" % ("band", "drizzle1", "drizzle2",
                                         "ratio"))
        for lab, sl in (("0-25%", slice(1, k // 4)),
                        ("25-50%", slice(k // 4, half)),
                        ("50-75%", slice(half, 3 * k // 4)),
                        ("75-100%", slice(3 * k // 4, k))):
            a, b = U[sl].sum(), H[sl].sum()
            print("  %-12s %10.5f %10.5f %7.2fx"
                  % (lab, a, b, b / a if a else 0))
        print()


if __name__ == "__main__":
    main()
