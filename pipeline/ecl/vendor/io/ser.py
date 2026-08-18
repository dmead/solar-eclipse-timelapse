"""SER container I/O — port of the SerFile reader (pjsr/ser-stack.js:78-203)
and the trim writer's header handling (pjsr/ser-trim.js:144-168).

SER header is 178 bytes; the fields we use sit at fixed offsets:
colorId@18, littleEndian@22, width@26, height@30, depth@34, count@38.
The endianness flag is ignored (SharpCap and friends write little-endian
regardless); numpy '<' dtypes are explicit little-endian.

Frames are exposed the way the pipeline consumes them: "mono" (Bayer 2x2
superpixel at half resolution / RGB channel average) or a single "R"/"G"/"B"
plane at the same resolution, normalized float32 [0,1].
"""

import os
from dataclasses import dataclass

import numpy as np

HEADER_BYTES = 178

# CFA offsets per SER colorId: (dx, dy) of each site in the 2x2 cell
CFA_LAYOUT = {
    8: {"R": (0, 0), "G1": (1, 0), "G2": (0, 1), "B": (1, 1)},   # RGGB
    9: {"R": (1, 0), "G1": (0, 0), "G2": (1, 1), "B": (0, 1)},   # GRBG
    10: {"R": (0, 1), "G1": (0, 0), "G2": (1, 1), "B": (1, 0)},  # GBRG
    11: {"R": (1, 1), "G1": (1, 0), "G2": (0, 1), "B": (0, 0)},  # BGGR
}


@dataclass
class SerHeader:
    color_id: int
    raw_width: int
    raw_height: int
    depth: int
    frame_count: int

    @property
    def bayer(self) -> bool:
        return 8 <= self.color_id < 100

    @property
    def rgb(self) -> bool:
        return self.color_id >= 100

    @property
    def bytes_per_value(self) -> int:
        return 2 if self.depth == 16 else 1

    @property
    def values_per_pixel(self) -> int:
        return 3 if self.rgb else 1

    @property
    def frame_bytes(self) -> int:
        return (self.raw_width * self.raw_height
                * self.bytes_per_value * self.values_per_pixel)


def read_header(path: str) -> SerHeader:
    with open(path, "rb") as f:
        f.seek(18)
        a = np.frombuffer(f.read(24), dtype="<i4")
    h = SerHeader(color_id=int(a[0]), raw_width=int(a[2]),
                  raw_height=int(a[3]), depth=int(a[4]),
                  frame_count=int(a[5]))
    if h.depth not in (8, 16):
        raise ValueError(f"unsupported pixel depth: {h.depth}")
    return h


class SerReader:
    """channel: "mono" (default) or "R"|"G"|"B" — see module docstring."""

    def __init__(self, path: str, channel: str = "mono"):
        self.path = path
        self.channel = channel
        self.header = h = read_header(path)
        # truncated-file guard (ser-stack.js:106-113)
        usable = (os.path.getsize(path) - HEADER_BYTES) // h.frame_bytes
        self.truncated = usable < h.frame_count
        self.frame_count = min(h.frame_count, usable)
        self.width = h.raw_width >> 1 if h.bayer else h.raw_width
        self.height = h.raw_height >> 1 if h.bayer else h.raw_height
        dt = "<u2" if h.depth == 16 else "u1"
        shape = ((self.frame_count, h.raw_height, h.raw_width, 3) if h.rgb
                 else (self.frame_count, h.raw_height, h.raw_width))
        self._mm = np.memmap(path, dtype=dt, mode="r",
                             offset=HEADER_BYTES, shape=shape)

    def read(self, i: int) -> np.ndarray:
        """Frame i as float32 [0,1], (height x width)."""
        h = self.header
        maxv = 65535.0 if h.depth == 16 else 255.0
        raw = self._mm[i]
        ch = self.channel
        if h.bayer:
            if ch == "mono":
                # superpixel: average each 2x2 cell, half resolution
                q = (raw[0::2, 0::2].astype(np.float32)
                     + raw[0::2, 1::2] + raw[1::2, 0::2] + raw[1::2, 1::2])
                return q * (0.25 / maxv)
            cfa = CFA_LAYOUT.get(h.color_id, CFA_LAYOUT[8])
            if ch == "G":
                (ax, ay), (bx, by) = cfa["G1"], cfa["G2"]
                g = (raw[ay::2, ax::2].astype(np.float32)
                     + raw[by::2, bx::2])
                return g * (0.5 / maxv)
            ox, oy = cfa[ch]
            return raw[oy::2, ox::2].astype(np.float32) / maxv
        if h.rgb:
            if ch == "mono":
                return raw.sum(axis=2, dtype=np.float32) / (3.0 * maxv)
            idx = {"R": 0, "G": 1, "B": 2}[ch]
            if h.color_id == 101:  # BGR order
                idx = 2 - idx
            return raw[:, :, idx].astype(np.float32) / maxv
        return raw.astype(np.float32) / maxv

    def read_raw(self, i: int) -> np.ndarray:
        """Frame i as stored (uint8/uint16, native mosaic/interleaved)."""
        return np.asarray(self._mm[i])

    def close(self) -> None:
        self._mm = None


def write_trimmed(src_path: str, out_path: str, frame_indices: list[int],
                  x0: int, y0: int, crop_w: int, crop_h: int) -> None:
    """Write selected frames cropped to a new SER, patching width/height/count
    at offsets 26/30/38 (ser-trim.js:144-168). (x0, y0) must be even-aligned
    by the caller to preserve CFA phase."""
    h = read_header(src_path)
    dt = "<u2" if h.depth == 16 else "u1"
    shape = ((h.raw_height, h.raw_width, 3) if h.rgb
             else (h.raw_height, h.raw_width))
    with open(src_path, "rb") as f:
        header = bytearray(f.read(HEADER_BYTES))
    header[26:30] = np.int32(crop_w).tobytes()
    header[30:34] = np.int32(crop_h).tobytes()
    header[38:42] = np.int32(len(frame_indices)).tobytes()
    mm = np.memmap(src_path, dtype=dt, mode="r",
                   offset=HEADER_BYTES,
                   shape=(h.frame_count, *shape))
    with open(out_path, "wb") as out:
        out.write(bytes(header))
        for i in frame_indices:
            crop = np.ascontiguousarray(
                mm[i][y0 : y0 + crop_h, x0 : x0 + crop_w])
            out.write(crop.tobytes())
    del mm
