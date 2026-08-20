"""Measure the shift between exposure levels directly — port of corona-register.js.

Locating the Moon independently in each level does not work across a 60x exposure
range. The radius comes out fine, but the centre does not: at the shortest
exposure the limb is marked only by a chromospheric arc on one side, and at the
longest it is washed out by bleed from a saturated inner corona — its ring score
is ten times weaker than a well-behaved level's, and the search happily locks
onto the edge of a saturated blob instead. Fits that disagree by hundreds of
pixels put two Moons in the merged image.

Comparing levels against each other avoids the problem entirely. It never has to
decide where the Moon is, only how far one frame has moved relative to another,
and for that the whole edge structure contributes rather than one fitted circle.

Both frames are reduced to the gradient magnitude of LOG brightness first. That
is what makes a 60x exposure difference comparable: in log space a given edge has
roughly the same height at every exposure. Normalized cross-correlation then
removes any residual scale and offset.

The search is a coarse-to-fine pyramid because the Moon travels a few hundred
pixels across the ladder, far too wide a window to brute-force at full resolution.
"""

import argparse
import json
import os
import time

import numpy as np

from .corona import log_gradient
from .imgio import read_xisf
from . import paths

# Decimation factors, coarsest first, with the half-width of the search at each
# level expressed in that level's own pixels.
PYRAMID = [(16, 22), (4, 8), (1, 6)]

# A correlation this weak means the two frames share no usable structure.
MIN_NCC = 0.25

# Too little overlap to trust a correlation.
MIN_OVERLAP = 1000

__all__ = ["register_levels", "match_pyramid", "ncc", "MIN_NCC"]


def ncc(a, b, dx, dy):
    """Normalized cross-correlation of `b` shifted by (dx, dy) against `a`."""
    h, w = a.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 <= y0 or x1 <= x0:
        return -2.0
    A = a[y0:y1, x0:x1].astype(np.float64, copy=False)
    B = b[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float64, copy=False)
    n = A.size
    if n < MIN_OVERLAP:
        return -2.0
    sa, sb = A.sum(), B.sum()
    ma, mb = sa / n, sb / n
    ca = float((A * A).sum()) - n * ma * ma
    cb = float((B * B).sum()) - n * mb * mb
    if ca <= 0 or cb <= 0:
        return -2.0
    return float((float((A * B).sum()) - n * ma * mb) / np.sqrt(ca * cb))


def decimate(img, k):
    """Block mean by an integer factor, discarding any ragged remainder."""
    if k == 1:
        return img
    h, w = img.shape
    hh, ww = h // k, w // k
    return img[:hh * k, :ww * k].reshape(hh, k, ww, k).mean(axis=(1, 3))


def build_pyramid(full):
    return {ds: decimate(full, ds) for ds, _ in PYRAMID}


def match_pyramid(pa, pb):
    """Coarse-to-fine match of pyramid `pa` onto reference pyramid `pb`.

    Each step searches a small window around the previous step's answer, scaled
    into its own resolution, so the total window is hundreds of pixels while the
    work stays bounded.
    """
    bx = by = 0
    for i, (ds, win) in enumerate(PYRAMID):
        A, B = pa[ds], pb[ds]
        cx, cy = int(round(bx / ds)), int(round(by / ds))
        best = (cx, cy, -2.0)
        for dy in range(cy - win, cy + win + 1):
            for dx in range(cx - win, cx + win + 1):
                s = ncc(B, A, dx, dy)
                if s > best[2]:
                    best = (dx, dy, s)
        bx, by = best[0] * ds, best[1] * ds

    A, B = pa[1], pb[1]
    c = ncc(B, A, bx, by)
    xm, xp = ncc(B, A, bx - 1, by), ncc(B, A, bx + 1, by)
    ym, yp = ncc(B, A, bx, by - 1), ncc(B, A, bx, by + 1)
    ddx, ddy = xm + xp - 2 * c, ym + yp - 2 * c
    ox = 0.5 * (xm - xp) / ddx if ddx < 0 else 0.0
    oy = 0.5 * (ym - yp) / ddy if ddy < 0 else 0.0
    if not abs(ox) <= 1:
        ox = 0.0
    if not abs(oy) <= 1:
        oy = 0.0
    # Returned as the shift to APPLY to this image to bring it onto the reference.
    return {"dx": -(bx + ox), "dy": -(by + oy), "ncc": c}


def _luma(path):
    """Green plane of a level image — the channel every stage measures on."""
    img = read_xisf(path)
    return img[:, :, 1] if img.ndim == 3 else img


def register_levels(specs, ref_path, log=print):
    """`specs` is a list of {path, level, t}. Returns the registration payload."""
    t0 = time.time()
    ref_idx = next((i for i, s in enumerate(specs) if s["path"] == ref_path), 0)
    log(f"reference: {os.path.basename(specs[ref_idx]['path'])}")

    pyrs, shape = [], None
    for s in specs:
        lum = _luma(s["path"])
        if shape is None:
            shape = lum.shape
        elif lum.shape != shape:
            raise ValueError(f"geometry mismatch on {s['path']}")
        pyrs.append(build_pyramid(log_gradient(lum)))
        log(f"  prepared {os.path.basename(s['path'])}")

    out = []
    for i, s in enumerate(specs):
        if i == ref_idx:
            out.append({"path": s["path"], "level": s.get("level"),
                        "t": s.get("t"), "dx": 0.0, "dy": 0.0, "ncc": 1.0})
            continue
        m = match_pyramid(pyrs[i], pyrs[ref_idx])
        flag = "  *** WEAK" if m["ncc"] < MIN_NCC else ""
        log(f"  L{s.get('level')} {os.path.basename(s['path'])} -> shift "
            f"({m['dx']:.2f}, {m['dy']:.2f}) ncc={m['ncc']:.4f}{flag}")
        out.append({"path": s["path"], "level": s.get("level"), "t": s.get("t"),
                    **m})

    log(f"  registered {len(specs)} levels in {time.time() - t0:.0f} s")
    return {"refPath": ref_path, "shifts": out}


def register_dataset(cfg, levels_dir=None, out_path=None, log=print):
    """Build the level list from an eclipse config and register it."""
    levels_dir = levels_dir or f"{cfg['outDir']}/levels"
    out_path = out_path or f"{cfg['outDir']}/final/registration.json"

    images = []
    for seg in cfg["segments"]:
        img = f"{levels_dir}/{seg['id']}.xisf"
        side = f"{levels_dir}/{seg['id']}_moon.json"
        if not (os.path.exists(img) and os.path.exists(side)):
            continue
        images.append({"path": img, "level": seg.get("level"),
                       "t": seg.get("t_mid")})
    if len(images) < 2:
        raise ValueError(f"need at least 2 levels, have {len(images)}")

    # The shortest exposure is the reference: it is the level that carries the
    # limb, chromosphere and prominences, so it is the one whose Moon should end
    # up sharp.
    ref = min(images, key=lambda i: (i["level"] if i["level"] is not None else 1e9))
    log(f"registering {len(images)} levels onto L{ref['level']}")
    payload = register_levels(images, ref["path"], log=log)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    log(f"  wrote {out_path}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=paths.in_out("configs", "eclipse.json"))
    ap.add_argument("--levels-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    register_dataset(cfg, args.levels_dir, args.out)


if __name__ == "__main__":
    main()
