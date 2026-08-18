"""CFA demosaic — 2x2 superpixel and full-resolution bilinear.

Two demosaics, because the pipeline genuinely wants both:

- `superpixel()` halves the resolution and is what the timelapse renders from
  (`pjsr/tl-frames.js:readSuperpixel`). It is exact — every output sample is a
  real measurement, no interpolation — and at 3.43 arcsec/px it undersamples by
  2x, which the README names as the single largest resolution loss in the
  pipeline. The drizzle pass is what buys that back.
- `bilinear_rggb()` keeps full resolution and is what anything fitting a limb or
  correlating fine coronal structure needs, since halving the sampling throws
  away exactly the detail those rely on.

lunation's SerReader offers superpixel-mono and single-plane extraction, but not
a three-plane superpixel and no full-resolution demosaic at all — it is a lunar
mono/LRGB pipeline. Both live here.
"""

import numpy as np

from .vendor.io.ser import CFA_LAYOUT

__all__ = ["superpixel", "superpixel_mono", "green_plane", "bilinear_rggb"]


def green_plane(raw, color_id=8, max_value=65535):
    """Just the half-resolution green plane, in [0, 1].

    Same result as `superpixel(...)[1]` without allocating R and B. The detection
    passes fit the disc on G alone over every frame in the sequence, so the two
    planes they discard are worth not building.
    """
    cfa = CFA_LAYOUT.get(color_id)
    if cfa is None:
        raise ValueError(f"colorId {color_id} is not a supported Bayer layout")
    a = np.asarray(raw)
    g1x, g1y = cfa["G1"]
    g2x, g2y = cfa["G2"]
    return ((a[g1y::2, g1x::2].astype(np.float32) + a[g2y::2, g2x::2])
            * np.float32(0.5 / max_value))


def superpixel(raw, color_id=8, max_value=65535):
    """2x2 superpixel demosaic -> three half-resolution planes in [0, 1].

    `raw` is one whole CFA frame as a 2D array. Each 2x2 cell becomes one output
    pixel: R and B from their single site, G from the mean of both green sites.
    Matches `tl-frames.js:readSuperpixel` (which hardcodes RGGB) generalised over
    the four Bayer orders via lunation's CFA table.
    """
    cfa = CFA_LAYOUT.get(color_id)
    if cfa is None:
        raise ValueError(f"colorId {color_id} is not a supported Bayer layout")

    a = np.asarray(raw)
    if a.ndim != 2:
        raise ValueError(f"expected a 2D CFA frame, got shape {a.shape}")
    h, w = a.shape
    if h % 2 or w % 2:
        raise ValueError(f"CFA frame must have even dimensions, got {w}x{h}")

    inv = np.float32(1.0 / max_value)
    rx, ry = cfa["R"]
    bx, by = cfa["B"]
    g1x, g1y = cfa["G1"]
    g2x, g2y = cfa["G2"]

    R = a[ry::2, rx::2].astype(np.float32) * inv
    B = a[by::2, bx::2].astype(np.float32) * inv
    G = (a[g1y::2, g1x::2].astype(np.float32) + a[g2y::2, g2x::2]) * (0.5 * inv)
    return R, G, B


def superpixel_mono(raw, max_value=65535):
    """2x2 cell average -> one half-resolution plane in [0, 1].

    Layout-independent: every 2x2 cell holds one R, one B and two G whatever the
    Bayer order, so the mean is the same four samples regardless.
    """
    a = np.asarray(raw)
    q = (a[0::2, 0::2].astype(np.float32) + a[0::2, 1::2]
         + a[1::2, 0::2] + a[1::2, 1::2])
    return q * np.float32(0.25 / max_value)


def bilinear_rggb(raw, width=None, height=None, max_value=None, color_id=8):
    """Bilinear CFA demosaic at full resolution, vectorised -> (3, H, W).

    Each output pixel takes its own colour directly when it sits on that colour's
    site, and the mean of the nearest same-colour sites otherwise. Borders clamp
    rather than mirror: mirroring folds the CFA phase and swaps colours.

    Output is float32; scaled to [0, 1] when `max_value` is given, raw ADU
    otherwise. Despite the name it follows the CFA table, so all four Bayer
    orders work; the default is colorId 8 (RGGB), which is what these captures
    are.

    Supersedes the hand-unrolled version that lived in `ser_to_fits.py`. That one
    was wrong at green sites — half of all pixels — mixing R and B into each
    other and, on odd rows, reading the green value into R. Verified against a
    synthetic flat-colour mosaic; see `tests/test_demosaic.py`. The neighbour
    offsets here are derived from the layout instead of written out per site,
    which is what makes them checkable.
    """
    cfa = CFA_LAYOUT.get(color_id)
    if cfa is None:
        raise ValueError(f"colorId {color_id} is not a supported Bayer layout")

    a = np.asarray(raw)
    if a.ndim == 1:
        if width is None or height is None:
            raise ValueError("flat input needs width and height")
        a = a.reshape(height, width)
    a = a.astype(np.float32)
    H, W = a.shape

    ap = np.pad(a, 1, mode="edge")

    def nbr(py, px, dy, dx):
        """The (dy, dx) neighbour of every pixel on the (py, px) sub-lattice."""
        r, c = py + 1 + dy, px + 1 + dx
        return ap[r:r + H:2, c:c + W:2]

    def mean(*terms):
        s = terms[0].copy()
        for t in terms[1:]:
            s += t
        return s / len(terms)

    out = np.zeros((3, H, W), np.float32)
    rx, ry = cfa["R"]
    bx, by = cfa["B"]

    for py in (0, 1):
        for px in (0, 1):
            here = nbr(py, px, 0, 0)
            cross = (nbr(py, px, -1, 0), nbr(py, px, 1, 0),
                     nbr(py, px, 0, -1), nbr(py, px, 0, 1))
            diag = (nbr(py, px, -1, -1), nbr(py, px, -1, 1),
                    nbr(py, px, 1, -1), nbr(py, px, 1, 1))
            horiz = (nbr(py, px, 0, -1), nbr(py, px, 0, 1))
            vert = (nbr(py, px, -1, 0), nbr(py, px, 1, 0))

            if (px, py) == (rx, ry):
                r, g, b = here, mean(*cross), mean(*diag)
            elif (px, py) == (bx, by):
                r, g, b = mean(*diag), mean(*cross), here
            else:
                # Green site: the other two colours lie along one axis each, and
                # which axis holds red depends on whether this row carries red.
                g = here
                r, b = (mean(*horiz), mean(*vert)) if py == ry \
                    else (mean(*vert), mean(*horiz))

            out[0, py::2, px::2] = r
            out[1, py::2, px::2] = g
            out[2, py::2, px::2] = b

    return out / np.float32(max_value) if max_value else out
