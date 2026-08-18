"""Measurement and export helpers — ports of sharpness.js, radial-probe.js,
crop-export.js and xisf-export.js.

None of these are pipeline stages; they exist to answer questions about the
products. Kept together because each is a few dozen lines and they share the
same "read an XISF, measure or write a picture" shape.
"""

import argparse
import math
import os

import numpy as np
from lunation.core.warp import resample

from .imgio import read_xisf, write_png

__all__ = ["sharpness", "radial_probe", "crop_export", "xisf_export"]


def _js_exp(v, digits=6):
    """JavaScript toExponential formatting: no zero padding in the exponent."""
    s = f"{v:.{digits}e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


# ---------------------------------------------------------------- sharpness

def sharpness(paths, log=print):
    """Edge contrast in the brightest region against noise in blank sky.

    The ratio is what matters: a stack can be made to look sharper by amplifying
    noise, and comparing edge strength alone would reward that.
    """
    rows = []
    for p in paths:
        img = read_xisf(p)
        a = img[:, :, 0] if img.ndim == 3 else img
        h, w = a.shape
        srt = np.sort(a, axis=None)
        hi = srt[int(srt.size * 0.99)]
        sky = srt[int(srt.size * 0.20)]

        core = a[1:h - 1, 1:w - 1]
        m = core >= hi
        gx = a[1:h - 1, 2:] - a[1:h - 1, :w - 2]
        gy = a[2:, 1:w - 1] - a[:h - 2, 1:w - 1]
        cnt = int(m.sum())
        edge = float(np.sqrt(((gx[m] ** 2 + gy[m] ** 2).sum()) / cnt)) if cnt else 0.0

        # Sky noise from the median absolute first difference, scaled to sigma.
        sub = a[1:h - 1:3, 1:w - 1:3]
        nxt = a[1:h - 1:3, 2:w:3]
        nxt = nxt[:, :sub.shape[1]]
        keep = sub <= sky
        devs = np.abs(nxt[keep] - sub[keep])
        noise = (float(np.sort(devs)[devs.size // 2]) / 0.6745 / math.sqrt(2)
                 if devs.size else 0.0)

        ratio = f"{edge / noise:.2f}" if noise > 0 else "inf"
        log(f"{os.path.basename(p)}  {w}x{h}  edge={_js_exp(edge, 4)}  "
            f"noise={_js_exp(noise, 4)}  edge/noise={ratio}  maskPx={cnt}")
        rows.append({"path": p, "width": w, "height": h, "edge": edge,
                     "noise": noise, "ratio": edge / noise if noise > 0 else None,
                     "maskPx": cnt})
    return rows


# ------------------------------------------------------------- radial probe

def radial_probe(in_path, cx, cy, out_path, log=print):
    """Azimuthally averaged radial profile as CSV.

    A diagnostic for locating which stage produced a ring: run it on each
    product and the radius where the profiles diverge is the culprit.
    """
    img = read_xisf(in_path)
    g = img[:, :, 1] if img.ndim == 3 else img
    H, W = g.shape
    log(f"probe {in_path} {W}x{H} about ({cx:.1f}, {cy:.1f})")

    rmax = int(math.ceil(math.hypot(max(cx, W - cx), max(cy, H - cy))))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.floor(np.hypot(xx - cx, yy - cy) + 0.5).astype(np.int64)
    inb = r <= rmax
    cnt = np.bincount(r[inb], minlength=rmax + 1)
    tot = np.bincount(r[inb], weights=g[inb].astype(np.float64),
                      minlength=rmax + 1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("r,mean,count\n")
        for rr in range(rmax + 1):
            mean = _js_exp(tot[rr] / cnt[rr]) if cnt[rr] > 0 else "0"
            f.write(f"{rr},{mean},{cnt[rr]}\n")
    log(f"  wrote {out_path} ({rmax} radii)")
    return out_path


# -------------------------------------------------------------- crop export

def crop_export(paths, out_dir, size=512, gamma=0.5, log=print):
    """Matching crops around the brightest structure, on one shared scale.

    The centre and the normalisation are measured ONCE on the first file, so the
    crops are directly comparable — separately normalised crops would hide
    exactly the brightness differences they are meant to show.
    """
    os.makedirs(out_dir, exist_ok=True)
    cx = cy = -1
    norm = 0.0
    written = []
    for p in paths:
        img = read_xisf(p)
        a = img[:, :, 0] if img.ndim == 3 else img
        h, w = a.shape
        if cx < 0:
            srt = np.sort(a, axis=None)
            hi = srt[int(srt.size * 0.995)]
            norm = float(srt[-1])
            m = a >= hi
            yy, xx = np.mgrid[0:h, 0:w]
            sw = float(a[m].sum())
            cx = int(round(float((xx[m] * a[m]).sum()) / sw))
            cy = int(round(float((yy[m] * a[m]).sum()) / sw))
            log(f"crop centred on ({cx}, {cy}), {size}x{size}")

        x0 = max(0, min(w - size, cx - (size >> 1)))
        y0 = max(0, min(h - size, cy - (size >> 1)))
        v = np.clip(a[y0:y0 + size, x0:x0 + size] / norm, 0.0, 1.0) ** gamma
        name = os.path.splitext(os.path.basename(p))[0]
        out = f"{out_dir}/{name}_crop.png"
        write_png(out, v.astype(np.float32), bit_depth=8)
        written.append(out)
        log(f"  wrote {name}_crop.png")
    return written


# -------------------------------------------------------------- xisf export

def xisf_export(in_path, out_path, max_width=0, log=print):
    """XISF to 8-bit PNG with NO tone change.

    Deliberately unlike pix-planetary's xisf-preview, which rescales and applies
    a gamma: this is for looking at what a stage actually produced, so anything
    outside [0, 1] is clipped rather than remapped, and clipping is the
    information you want to see.
    """
    img = read_xisf(in_path)
    W = img.shape[1]
    if max_width > 0 and W > max_width:
        img = resample(img, max_width / W, "mitchell")
    write_png(out_path, np.clip(img, 0.0, 1.0).astype(np.float32), bit_depth=8)
    log(f"exported {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sharpness", help="edge contrast against sky noise")
    s.add_argument("paths", nargs="+")

    r = sub.add_parser("probe", help="radial profile to CSV")
    r.add_argument("in_path")
    r.add_argument("--cx", type=float, required=True)
    r.add_argument("--cy", type=float, required=True)
    r.add_argument("--out", required=True)

    c = sub.add_parser("crop", help="matching crops on one shared scale")
    c.add_argument("paths", nargs="+")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--size", type=int, default=512)
    c.add_argument("--gamma", type=float, default=0.5)

    e = sub.add_parser("export", help="XISF to PNG with no tone change")
    e.add_argument("in_path")
    e.add_argument("out_path")
    e.add_argument("--max-width", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "sharpness":
        sharpness(a.paths)
    elif a.cmd == "probe":
        radial_probe(a.in_path, a.cx, a.cy, a.out)
    elif a.cmd == "crop":
        crop_export(a.paths, a.out_dir, a.size, a.gamma)
    else:
        xisf_export(a.in_path, a.out_path, a.max_width)


if __name__ == "__main__":
    main()
