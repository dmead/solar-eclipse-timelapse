"""Disk-cached aligned-frame cube + kappa-sigma rejection.

Ports the frame-plane cache and rejection pass of pjsr/ser-stack.js
(cache: 493-521; clip: 596-693). Planes are raw uint16, value =
round(clamp(v,0,1)*65534), sentinel 0xFFFF outside a frame's coverage.
The cube never materializes in RAM (worst case ~9-12 GB on disk); the
combine pass streams row-chunks through a memmap.

Two clip engines:
- "ported" (default): exact ser-stack.js math — mean-centered kappa bounds,
  no rejection where a pixel has <5 samples, and the never-lose-every-sample
  rule (a pixel that would reject all its samples keeps its prior stats).
- "astropy": astropy.stats.sigma_clip per chunk (mean-centered to match);
  same never-lose fallback. Kept for the M1 A/B.
"""

import os

import numpy as np

SENTINEL = np.uint16(0xFFFF)
SCALE = 65534.0


def encode_plane(plane: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """float32 plane -> uint16 with sentinel where invalid.

    Follows ser-stack.js:500-505: values < -0.5 (the -1 coverage fill) become
    the sentinel; everything else clamps to [0,1] and quantizes to 65534.
    """
    p = np.asarray(plane, dtype=np.float32)
    u = np.rint(np.clip(p, 0.0, 1.0) * SCALE).astype(np.uint16)
    invalid = p < -0.5 if valid is None else ~valid
    u[invalid] = SENTINEL
    return u


class FrameCube:
    def __init__(self, path: str, width: int, height: int):
        self.path = path
        self.width = width
        self.height = height
        self.plane_bytes = width * height * 2
        self.nplanes = 0

    def write_plane(self, index: int, plane: np.ndarray,
                    valid: np.ndarray | None = None) -> None:
        """Write one aligned plane at a deterministic offset (parallel-safe:
        each worker opens its own handle and owns disjoint indices)."""
        u = encode_plane(plane, valid)
        mode = "r+b" if os.path.exists(self.path) else "wb"
        with open(self.path, mode) as f:
            f.seek(index * self.plane_bytes)
            f.write(u.tobytes())
        self.nplanes = max(self.nplanes, index + 1)

    def open_writer(self):
        """Persistent single-process writer handle (append-order usage)."""
        return open(self.path, "wb")

    def combine(
        self,
        kappa: float,
        iters: int,
        nplanes: int | None = None,
        engine: str = "ported",
        chunk_rows: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kappa-sigma-clipped mean over the cube.

        Returns (mean float32 HxW — 0 where no samples, count int32 HxW).
        """
        n = self.nplanes if nplanes is None else nplanes
        cube = np.memmap(self.path, dtype=np.uint16, mode="r",
                         shape=(n, self.height, self.width))
        mean = np.zeros((self.height, self.width), dtype=np.float32)
        count = np.zeros((self.height, self.width), dtype=np.int32)
        for y0 in range(0, self.height, chunk_rows):
            y1 = min(y0 + chunk_rows, self.height)
            chunk = np.ascontiguousarray(cube[:, y0:y1, :])
            m, c = _clip_chunk(chunk, kappa, iters, engine)
            mean[y0:y1] = m
            count[y0:y1] = c
        del cube
        return mean, count

    def remove(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


def _clip_chunk(chunk_u16: np.ndarray, kappa: float, iters: int,
                engine: str) -> tuple[np.ndarray, np.ndarray]:
    valid0 = chunk_u16 != SENTINEL
    v = chunk_u16.astype(np.float32) / SCALE

    if engine == "astropy":
        from astropy.stats import sigma_clip

        data = np.ma.MaskedArray(v, mask=~valid0)
        clipped = sigma_clip(data, sigma=kappa, maxiters=iters,
                             cenfunc="mean", stdfunc="std", axis=0)
        cnt = (~clipped.mask).sum(axis=0).astype(np.int32)
        mean = np.ma.mean(clipped, axis=0).filled(0.0).astype(np.float32)
        # never lose every sample: fall back to the unclipped mean
        lost = (cnt == 0) & valid0.any(axis=0)
        if np.any(lost):
            fallback = np.ma.mean(data, axis=0).filled(0.0).astype(np.float32)
            mean[lost] = fallback[lost]
            cnt[lost] = valid0.sum(axis=0)[lost].astype(np.int32)
        return mean, cnt

    # --- ported engine: exact ser-stack.js:601-685 semantics, vectorized ---
    valid = valid0
    s = np.where(valid, v, 0.0).sum(axis=0, dtype=np.float64)
    q = np.where(valid, v * v, 0.0).sum(axis=0, dtype=np.float64)
    c = valid.sum(axis=0).astype(np.int64)
    for _ in range(iters):
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            sd = np.sqrt(np.maximum(q / np.maximum(c, 1) - mu * mu, 0.0))
        lo = np.where(c >= 5, mu - kappa * sd, -1.0)
        hi = np.where(c >= 5, mu + kappa * sd, 2.0)
        keep = valid & (v >= lo) & (v <= hi)
        ns = np.where(keep, v, 0.0).sum(axis=0, dtype=np.float64)
        nq = np.where(keep, v * v, 0.0).sum(axis=0, dtype=np.float64)
        nc = keep.sum(axis=0).astype(np.int64)
        # a pixel must never lose every sample — keep prior stats there
        lost = (nc == 0) & (c > 0)
        ns = np.where(lost, s, ns)
        nq = np.where(lost, q, nq)
        nc = np.where(lost, c, nc)
        keep = np.where(lost[None, :, :], valid, keep)
        s, q, c, valid = ns, nq, nc, keep
    mean = np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)
    return mean, c.astype(np.int32)
