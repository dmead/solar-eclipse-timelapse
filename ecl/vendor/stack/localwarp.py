"""Local seeing-warp correction (alignment points, AS!3-style, sub-pixel).

Port of pjsr/ser-stack.js:229-324. Measured on TS-70 data: median 0.41px /
p90 0.78px residual after global alignment — correction tightens the PSF
rather than rescuing geometry. Overlapping tiles are phase-correlated
against the reference, translated by their local residual, and reassembled
with raised-cosine weights; a low-weight globally-shifted base layer
guarantees full coverage where tiles are gated off (sky, low signal).

Better-than extension (no PJSR counterpart, 2026-07-24): per-tile lucky
selection. With select_gamma > 0, each tile's blend weight is additionally
scaled by (local sharpness / reference-tile sharpness) ** gamma, and
apply_weighted() exposes the un-normalized (numerator, weight) pair so the
stacker can average ACROSS frames with those weights — the sharpest
moments of each region dominate regardless of which frame they came from
(AS!4-style per-AP selection, smooth-weight variant). apply() normalizes
per-frame and is unchanged.
"""

import numpy as np

from ..core.fftreg import PhaseCorrelator
from ..core.kernels import laplacian_sharpness
from ..core.warp import translate

BASE_WEIGHT = 0.05
TILE_GATE = 0.12  # tile mean below this fraction of ref mean = sky/shadow
Q_CLIP = (0.2, 5.0)  # relative-sharpness clamp: one tile never dominates


class LocalWarp:
    def __init__(self, ref: np.ndarray, tile_size: int = 256,
                 tile_step: int = 128, clamp_px: float = 3.0,
                 upsample: int = 20, select_gamma: float = 0.0):
        self.tile = tile_size
        self.step = tile_step
        self.clamp = clamp_px
        self.gamma = float(select_gamma)
        self.H, self.W = ref.shape
        self.tiles = []
        ref_mean = float(ref.mean())
        for y in range(0, self.H - tile_size + 1, tile_step):
            for x in range(0, self.W - tile_size + 1, tile_step):
                patch = ref[y : y + tile_size, x : x + tile_size]
                if float(patch.mean()) < TILE_GATE * ref_mean:
                    continue  # sky/deep shadow tile
                pc = PhaseCorrelator(use_gradient=False, upsample=upsample)
                pc.initialize(np.ascontiguousarray(patch))
                s_ref = (max(laplacian_sharpness(patch), 1e-12)
                         if self.gamma > 0 else 1.0)
                self.tiles.append((x, y, pc, s_ref))

        # raised-cosine blend weight
        i = (np.arange(tile_size, dtype=np.float32) + 0.5) / tile_size
        w1 = 0.5 - 0.5 * np.cos(2 * np.pi * i)
        self.weight = np.maximum(1e-4, np.outer(w1, w1)).astype(np.float32)

    def apply_weighted(self, img: np.ndarray, gdx: float,
                       gdy: float) -> tuple[np.ndarray, np.ndarray]:
        """Register img and return the UN-normalized (numerator, weight)
        pair. numerator = sum(tile * blend_w * q**gamma); dividing the
        cross-frame sums of both arrays yields the lucky-weighted stack."""
        # base layer: globally shifted frame at low weight (coverage guarantee)
        base = translate(img, -gdx, -gdy)
        out = base * BASE_WEIGHT
        wsum = np.full((self.H, self.W), BASE_WEIGHT, dtype=np.float32)

        gix, giy = round(gdx), round(gdy)
        gfx, gfy = gdx - gix, gdy - giy
        t = self.tile
        for x, y, pc, s_ref in self.tiles:
            # extract source patch at the globally-corrected position
            sx, sy = x + gix, y + giy
            if sx < 0 or sy < 0 or sx + t > self.W or sy + t > self.H:
                continue
            patch = np.ascontiguousarray(img[sy : sy + t, sx : sx + t])
            rx, ry = pc.evaluate(patch)
            # implausible local lock -> fall back to the global residual
            if np.hypot(rx - gfx, ry - gfy) > self.clamp:
                rx, ry = gfx, gfy
            w = self.weight
            if self.gamma > 0:
                q = laplacian_sharpness(patch) / s_ref
                w = w * float(np.clip(q, *Q_CLIP)) ** self.gamma
            shifted = translate(patch, -rx, -ry)
            out[y : y + t, x : x + t] += shifted * w
            wsum[y : y + t, x : x + t] += w
        return out, wsum

    def apply(self, img: np.ndarray, gdx: float, gdy: float) -> np.ndarray:
        """Return img registered to reference geometry (global + local)."""
        out, wsum = self.apply_weighted(img, gdx, gdy)
        return out / wsum

    def tile_qualities(self, img: np.ndarray, gdx: float,
                       gdy: float) -> np.ndarray:
        """Per-tile sharpness of img (laplacian, at the globally-corrected
        patch positions), aligned with self.tiles order. NaN where the
        patch falls outside the frame. Pass 1 of hard lucky selection."""
        gix, giy = round(gdx), round(gdy)
        t = self.tile
        q = np.full(len(self.tiles), np.nan, dtype=np.float32)
        for k, (x, y, _pc, _s) in enumerate(self.tiles):
            sx, sy = x + gix, y + giy
            if sx < 0 or sy < 0 or sx + t > self.W or sy + t > self.H:
                continue
            q[k] = laplacian_sharpness(
                np.ascontiguousarray(img[sy : sy + t, sx : sx + t]))
        return q

    def apply_masked(self, img: np.ndarray, gdx: float, gdy: float,
                     tile_mask: np.ndarray,
                     base_weight: float) -> tuple[np.ndarray, np.ndarray]:
        """Hard lucky selection: register and accumulate ONLY the tiles
        flagged in tile_mask (cosine-blended), plus a base layer at
        base_weight (callers scale it by the select fraction so the
        global-shift fallback cannot dilute the selection). Returns the
        un-normalized (numerator, weight) pair like apply_weighted."""
        base = translate(img, -gdx, -gdy)
        out = base * base_weight
        wsum = np.full((self.H, self.W), base_weight, dtype=np.float32)
        gix, giy = round(gdx), round(gdy)
        gfx, gfy = gdx - gix, gdy - giy
        t = self.tile
        for k, (x, y, pc, _s) in enumerate(self.tiles):
            if not tile_mask[k]:
                continue
            sx, sy = x + gix, y + giy
            if sx < 0 or sy < 0 or sx + t > self.W or sy + t > self.H:
                continue
            patch = np.ascontiguousarray(img[sy : sy + t, sx : sx + t])
            rx, ry = pc.evaluate(patch)
            if np.hypot(rx - gfx, ry - gfy) > self.clamp:
                rx, ry = gfx, gfy
            shifted = translate(patch, -rx, -ry)
            out[y : y + t, x : x + t] += shifted * self.weight
            wsum[y : y + t, x : x + t] += self.weight
        return out, wsum

    def shift_field(self, img: np.ndarray, gdx: float,
                    gdy: float) -> tuple[np.ndarray, np.ndarray]:
        """Dense per-pixel (dx, dy) displacement of img vs the reference —
        the same per-tile locks apply() uses, but returned as a smooth
        field instead of a resampled image, so the caller can fold the
        warp INTO a drizzle pixmap and resample exactly once. Content at
        (x, y) in img belongs at (x - dx, y - dy) in reference geometry.
        Gated tiles (sky/shadow) fall back to the global shift via the
        low-weight base layer, exactly like apply()."""
        dxs = np.full((self.H, self.W), gdx, dtype=np.float32) * BASE_WEIGHT
        dys = np.full((self.H, self.W), gdy, dtype=np.float32) * BASE_WEIGHT
        wsum = np.full((self.H, self.W), BASE_WEIGHT, dtype=np.float32)
        gix, giy = round(gdx), round(gdy)
        gfx, gfy = gdx - gix, gdy - giy
        t = self.tile
        for x, y, pc, _s_ref in self.tiles:
            sx, sy = x + gix, y + giy
            if sx < 0 or sy < 0 or sx + t > self.W or sy + t > self.H:
                continue
            patch = np.ascontiguousarray(img[sy : sy + t, sx : sx + t])
            rx, ry = pc.evaluate(patch)
            if np.hypot(rx - gfx, ry - gfy) > self.clamp:
                rx, ry = gfx, gfy
            # total displacement of img content in this tile vs reference,
            # accumulated at the INPUT pixel positions (sx, sy) — the
            # pixmap consumer indexes by input coordinates
            dxs[sy : sy + t, sx : sx + t] += (gix + rx) * self.weight
            dys[sy : sy + t, sx : sx + t] += (giy + ry) * self.weight
            wsum[sy : sy + t, sx : sx + t] += self.weight
        return dxs / wsum, dys / wsum
