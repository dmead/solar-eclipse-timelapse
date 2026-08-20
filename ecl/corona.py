"""Shared corona geometry: log gradient, ring scoring, Moon finding.

Used by every stage of the corona chain. The tuning differs from
`tl_centres`, which solves a similar-looking problem on single video frames:

- The log floor is a fixed tiny epsilon, not the frame median. These are stacked
  exposures with a real sky pedestal, not single noisy frames where the log of
  near-zero sky blows up.
- The ring only has to be 60% inside the frame, against 95% there. Levels are
  shifted by up to 227 px to register, so a disc genuinely can sit partly off.
- The seed searches radius as well as position, because the exposure ladder is
  measured before any radius is known.

FINDING THE MOON

Earlier versions cast rays outward and fitted a circle to where the brightness
crossed a threshold. The radius that produced was excellent — 565.8 px on every
level, to a tenth of a pixel — but the centre was not: tracking it through
totality gave apparent frame motion of 66, 3.2, 12, 2.7 and 0.82 px/s on
successive levels, and the Moon cannot change speed like that. A least-squares
circle through crossings is only as good as the crossings, and prominences, a
lopsided inner corona and saturation bias them asymmetrically, which moves the
centre while leaving the radius about right.

This scores candidate centres directly: the limb is the strongest edge in the
frame at every exposure, so the true centre is the one whose circle of radius R
lies along the most edge. Scoring on the gradient of LOG brightness makes that
exposure-invariant — it asks where brightness multiplies fastest, not where it
exceeds a level — and searching the whole frame hierarchically means an
asymmetric feature cannot drag the answer off, only fail to improve the score.
"""

import math

import numpy as np

RING_SAMPLES = 360
COARSE_STEP = 16
REFINE_STEPS = [8, 4, 2, 1]

# Radius search bracket around the seed, when no radius is imposed.
R_LO_FRAC = 0.55
R_HI_FRAC = 1.70
R_SEARCH_STEPS = 24

__all__ = ["log_gradient", "search_centre", "seed_disc", "measure_moon"]


def _jsround(a):
    """JavaScript Math.round — half away from zero, not banker's."""
    return np.floor(np.asarray(a, dtype=np.float64) + 0.5).astype(np.int64)


def log_gradient(g, tiny=1e-9):
    """Gradient magnitude of log brightness.

    Log first: the corona spans orders of magnitude, so a plain gradient is
    dominated by the bright inner region and the limb's signature would depend on
    exposure. In log space the limb is a step of roughly constant height at every
    exposure level, which is what makes one scoring function work across the
    whole ladder.
    """
    L = np.log(np.maximum(g.astype(np.float32), np.float32(tiny)))
    out = np.zeros_like(L)
    out[1:-1, 1:-1] = np.hypot(L[1:-1, 2:] - L[1:-1, :-2],
                               L[2:, 1:-1] - L[:-2, 1:-1])
    return out


def _ring_scores(grad, cxs, cys, dxk, dyk, min_n):
    """Mean gradient around a ring, for many integer candidate centres at once."""
    h, w = grad.shape
    X = np.asarray(cxs)[:, None] + dxk[None, :]
    Y = np.asarray(cys)[:, None] + dyk[None, :]
    ok = (X >= 1) & (Y >= 1) & (X < w - 1) & (Y < h - 1)
    v = np.where(ok, grad[np.clip(Y, 0, h - 1), np.clip(X, 0, w - 1)], 0.0)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = v.sum(axis=1) / np.maximum(n, 1)
    return np.where(n >= min_n, s, -np.inf)


def search_centre(grad, R, log=None, chunk=8192):
    """Hierarchical centre search at a known radius.

    A coarse sweep of the whole frame, then successively finer local
    refinements, then a sub-pixel parabola through the best cell.
    """
    h, w = grad.shape
    a = 2.0 * np.pi * np.arange(RING_SAMPLES) / RING_SAMPLES
    dxk, dyk = _jsround(np.cos(a) * R), _jsround(np.sin(a) * R)
    # Require most of the ring inside the frame, or a circle hanging off the
    # edge can win on a handful of bright samples.
    min_n = RING_SAMPLES * 0.6

    gx = np.arange(0, w, COARSE_STEP)
    gy = np.arange(0, h, COARSE_STEP)
    YY, XX = np.meshgrid(gy, gx, indexing="ij")
    fx, fy = XX.reshape(-1), YY.reshape(-1)
    bx, by, bs = w // 2, h // 2, -np.inf
    best_j, best_s = -1, -np.inf
    for o in range(0, fx.size, chunk):
        sc = _ring_scores(grad, fx[o:o + chunk], fy[o:o + chunk], dxk, dyk, min_n)
        j = int(np.argmax(sc))
        if sc[j] > best_s:
            best_s, best_j = float(sc[j]), o + j
    if best_j >= 0:
        bx, by, bs = int(fx[best_j]), int(fy[best_j]), best_s

    # Sequential on purpose: the original updates the centre inside the loop, so
    # later probes are taken around the new best rather than the old one.
    for i, step in enumerate(REFINE_STEPS):
        span = COARSE_STEP if i == 0 else REFINE_STEPS[i - 1]
        for dy in range(-span, span + 1, step):
            for dx in range(-span, span + 1, step):
                s = float(_ring_scores(grad, [bx + dx], [by + dy],
                                       dxk, dyk, min_n)[0])
                if s > bs:
                    bs, bx, by = s, bx + dx, by + dy

    probe = _ring_scores(grad, [bx - 1, bx + 1, bx, bx],
                         [by, by, by - 1, by + 1], dxk, dyk, min_n)
    sxm, sxp, sym, syp = (float(v) for v in probe)
    ddx, ddy = sxm + sxp - 2 * bs, sym + syp - 2 * bs
    ox = 0.5 * (sxm - sxp) / ddx if ddx < 0 else 0.0
    oy = 0.5 * (sym - syp) / ddy if ddy < 0 else 0.0
    if not abs(ox) <= 1:
        ox = 0.0
    if not abs(oy) <= 1:
        oy = 0.0
    if log:
        log(f"  ring search r={R:.1f} -> ({bx + ox:.2f}, {by + oy:.2f}) "
            f"score {bs:.3e}")
    return {"cx": bx + ox, "cy": by + oy, "score": bs}


def seed_disc(g, log=None, ds=32, ring_angles=32):
    """Matched filter: the dark disc inside a bright ring.

    Searches position and radius together on a heavily decimated image. This only
    has to bracket the radius for `search_centre`; the centre it returns is not
    used directly.
    """
    H, W = g.shape
    h, w = H // ds, W // ds
    small = g[:h * ds, :w * ds].reshape(h, ds, w, ds).mean(axis=(1, 3))

    a = 2.0 * np.pi * np.arange(ring_angles) / ring_angles
    ca, sa = np.cos(a), np.sin(a)

    best = {"cx": W / 2, "cy": H / 2, "r": min(W, H) / 4, "score": -np.inf}
    r_max = h // 2 - 2 if h < w else w // 2 - 2
    r_max = min(h, w) // 2 - 2
    for r in range(4, r_max + 1):
        r_in, r_out = r * 0.70, r * 1.15
        # Candidate centres are integers, so the sample offsets are identical
        # for all of them and can be rounded once.
        dxo, dyo = _jsround(ca * r_out), _jsround(sa * r_out)
        dxi, dyi = _jsround(ca * r_in), _jsround(sa * r_in)
        cys, cxs = np.arange(r + 1, h - r - 1), np.arange(r + 1, w - r - 1)
        if cys.size == 0 or cxs.size == 0:
            continue
        YY, XX = np.meshgrid(cys, cxs, indexing="ij")

        ring = np.zeros(YY.shape, np.float64)
        disc = np.zeros(YY.shape, np.float64)
        bad = np.zeros(YY.shape, bool)
        for k in range(ring_angles):
            yo, xo = YY + dyo[k], XX + dxo[k]
            # The original abandons a candidate whose OUTER sample leaves the
            # decimated frame; the loop bounds do not guarantee it, since
            # r_out > r.
            bad |= (xo < 0) | (yo < 0) | (xo >= w) | (yo >= h)
            ring += small[np.clip(yo, 0, h - 1), np.clip(xo, 0, w - 1)]
            disc += small[np.clip(YY + dyi[k], 0, h - 1),
                          np.clip(XX + dxi[k], 0, w - 1)]
        score = np.where(bad, -np.inf, (ring - disc) / ring_angles)
        j = int(np.argmax(score))
        iy, ix = divmod(j, score.shape[1])
        if score[iy, ix] > best["score"]:
            best = {"cx": float((cxs[ix] + 0.5) * ds),
                    "cy": float((cys[iy] + 0.5) * ds),
                    "r": float(r * ds), "score": float(score[iy, ix])}

    if log:
        log(f"  seed disc centre ({best['cx']:.0f}, {best['cy']:.0f}) "
            f"r~{best['r']:.0f} px")
    return best


def measure_moon(g, log=print, fixed_r=0.0):
    """Locate the Moon's centre and limb radius in one level's green plane."""
    seed = seed_disc(g, log)
    grad = log_gradient(g)

    if fixed_r and fixed_r > 0:
        best = search_centre(grad, fixed_r, log)
        best["r"] = float(fixed_r)
        if log:
            log(f"  radius imposed at {fixed_r:.1f} px")
    else:
        # Scan radii around the seed, taking the centre search's best score at
        # each. The limb ring outscores everything else, so the peak over radius
        # is the true disc.
        best = {"score": -np.inf, "cx": 0.0, "cy": 0.0, "r": 0.0}
        r_lo, r_hi = seed["r"] * R_LO_FRAC, seed["r"] * R_HI_FRAC
        for k in range(R_SEARCH_STEPS + 1):
            R = r_lo + (r_hi - r_lo) * k / R_SEARCH_STEPS
            c = search_centre(grad, R)
            if c["score"] > best["score"]:
                best = {"cx": c["cx"], "cy": c["cy"], "r": R,
                        "score": c["score"]}
        if log:
            log(f"  radius scan {r_lo:.0f}..{r_hi:.0f} -> {best['r']:.1f} px")
        # Polish the centre at the winning radius on the finest grid.
        c = search_centre(grad, best["r"], log)
        best["cx"], best["cy"], best["score"] = c["cx"], c["cy"], c["score"]

    if log:
        log(f"  moon centre ({best['cx']:.1f}, {best['cy']:.1f}) limb radius "
            f"{best['r']:.1f} px, ring score {best['score']:.3e}")
    return best
