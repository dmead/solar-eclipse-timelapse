"""Demosaic correctness.

The flat-colour test is the one that matters. A CFA demosaic can look plausible
on real data while mixing channels — the bug it caught had R and B swapped at
every green site, which is half the pixels, and it survived in `ser_to_fits.py`
long enough to produce a set of FITS exports. On a frame where every R site holds
one constant, every G site another and every B site a third, a correct demosaic
must return those three constants everywhere in the interior no matter which site
the output pixel sits on. Nothing about that is subtle once it is written down.
"""

import numpy as np
import pytest

from ecl.demosaic import bilinear_rggb, green_plane, superpixel, superpixel_mono

# colorId -> the (x, y) site of R and B within the 2x2 cell.
LAYOUTS = {8: ((0, 0), (1, 1)), 9: ((1, 0), (0, 1)),
           10: ((0, 1), (1, 0)), 11: ((1, 1), (0, 0))}

R0, G0, B0 = 0.80, 0.50, 0.20


def _flat_mosaic(color_id, h=16, w=16):
    """A mosaic of three flat colours laid out for `color_id`."""
    (rx, ry), (bx, by) = LAYOUTS[color_id]
    a = np.full((h, w), G0, np.float32)
    a[ry::2, rx::2] = R0
    a[by::2, bx::2] = B0
    return a


@pytest.mark.parametrize("color_id", sorted(LAYOUTS))
def test_bilinear_reconstructs_flat_colours(color_id):
    a = _flat_mosaic(color_id)
    out = bilinear_rggb(a, max_value=1.0, color_id=color_id)
    inner = out[:, 2:-2, 2:-2]          # borders clamp, so skip them
    for c, want in enumerate((R0, G0, B0)):
        assert np.allclose(inner[c], want, atol=1e-5), (
            f"colorId {color_id} channel {'RGB'[c]}: "
            f"got {inner[c].min():.4f}..{inner[c].max():.4f}, want {want}")


@pytest.mark.parametrize("color_id", sorted(LAYOUTS))
def test_bilinear_preserves_known_sites(color_id):
    """A pixel sitting on its own colour's site must pass through untouched."""
    rng = np.random.default_rng(3)
    a = rng.random((32, 32)).astype(np.float32)
    (rx, ry), (bx, by) = LAYOUTS[color_id]
    out = bilinear_rggb(a, max_value=1.0, color_id=color_id)
    assert np.allclose(out[0, ry::2, rx::2], a[ry::2, rx::2], atol=1e-6)
    assert np.allclose(out[2, by::2, bx::2], a[by::2, bx::2], atol=1e-6)


@pytest.mark.parametrize("color_id", sorted(LAYOUTS))
def test_superpixel_reconstructs_flat_colours(color_id):
    a = _flat_mosaic(color_id)
    R, G, B = superpixel(a, color_id=color_id, max_value=1.0)
    assert np.allclose(R, R0, atol=1e-6)
    assert np.allclose(G, G0, atol=1e-6)
    assert np.allclose(B, B0, atol=1e-6)


def test_green_plane_matches_superpixel():
    rng = np.random.default_rng(11)
    a = (rng.random((64, 64)) * 65535).astype(np.uint16)
    assert np.allclose(green_plane(a), superpixel(a)[1], atol=0)


def test_superpixel_mono_is_the_cell_mean():
    """Layout-independent: every 2x2 cell holds one R, one B and two G whatever
    the Bayer order, so the mean is the same four samples regardless."""
    rng = np.random.default_rng(5)
    a = (rng.random((32, 32)) * 65535).astype(np.uint16)
    mono = superpixel_mono(a)
    for color_id in LAYOUTS:
        R, G, B = superpixel(a, color_id=color_id)
        assert np.allclose(mono, (R + 2 * G + B) / 4, atol=1e-6)


def test_rejects_unsupported_layout():
    a = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        bilinear_rggb(a, color_id=100)
    with pytest.raises(ValueError):
        superpixel(a, color_id=0)
