"""Minimal SER v3 reader — Python standard library only, no numpy.

Written for 460 GB of eclipse captures living on a slow drive, so everything here
is seek-based: nothing ever reads a whole frame unless you ask it to.

SER layout:
    0..178      header
    178..       frameCount * frameBytes of image data
    tail        frameCount * int64 UTC timestamps (.NET ticks), optional

The header's LittleEndian flag is read but deliberately ignored: SharpCap writes 0
there while emitting little-endian data, and honouring the flag scrambles every
sample (median 735 -> 32769 on these captures). pix-planetary's ser-stack.js takes
the same position — "SharpCap and friends write little-endian regardless".
"""

import array
import datetime
import struct

HEADER_BYTES = 178
_EPOCH = datetime.datetime(1, 1, 1)

# SER colorId -> human name. 8..19 are Bayer CFA, 100+ are interleaved RGB.
COLOR_IDS = {
    0: "MONO",
    8: "BAYER_RGGB", 9: "BAYER_GRBG", 10: "BAYER_GBRG", 11: "BAYER_BGGR",
    100: "RGB", 101: "BGR",
}


def ticks_to_utc(t):
    """.NET ticks (100 ns since 0001-01-01) -> datetime."""
    return _EPOCH + datetime.timedelta(microseconds=t / 10)


class SerFile:
    """Random-access reader over a SER capture.

    Use as a context manager. Frame indices are 0-based.
    """

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        head = self._fh.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise ValueError("%s: truncated SER header" % self.path)

        self.file_id = head[0:14].decode("ascii", "replace")
        (self.lu_id, self.color_id, self.little_endian,
         self.width, self.height, self.depth, self.frame_count) = struct.unpack("<7i", head[14:42])
        self.observer = head[42:82].decode("latin1").strip("\x00 ")
        self.instrument = head[82:122].decode("latin1").strip("\x00 ")
        self.telescope = head[122:162].decode("latin1").strip("\x00 ")
        self.dt_local, self.dt_utc = struct.unpack("<2q", head[162:178])
        self.raw_header = head

        self.bayer = 8 <= self.color_id < 100
        self.rgb = self.color_id >= 100
        self.planes = 3 if self.rgb else 1
        self.bytes_per_sample = 1 if self.depth <= 8 else 2
        self.row_bytes = self.width * self.planes * self.bytes_per_sample
        self.frame_bytes = self.row_bytes * self.height
        self.max_value = (1 << 16) - 1 if self.bytes_per_sample == 2 else 255

        self._trailer_off = HEADER_BYTES + self.frame_count * self.frame_bytes
        self._fh.seek(0, 2)
        self.file_size = self._fh.tell()
        self.has_trailer = self.file_size >= self._trailer_off + self.frame_count * 8
        self._ts_cache = None

    # -- lifecycle -------------------------------------------------------

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- timestamps ------------------------------------------------------

    def timestamp(self, index):
        """UTC datetime for one frame, from the trailer when present."""
        if not self.has_trailer:
            return ticks_to_utc(self.dt_utc)
        self._fh.seek(self._trailer_off + index * 8)
        return ticks_to_utc(struct.unpack("<q", self._fh.read(8))[0])

    def timestamps(self):
        """All frame timestamps as a list of datetimes (one seek, cached)."""
        if self._ts_cache is None:
            if not self.has_trailer:
                self._ts_cache = [ticks_to_utc(self.dt_utc)] * self.frame_count
            else:
                self._fh.seek(self._trailer_off)
                raw = array.array("q")
                raw.frombytes(self._fh.read(self.frame_count * 8))
                self._ts_cache = [ticks_to_utc(t) for t in raw]
        return self._ts_cache

    def duration(self):
        ts = self.timestamps()
        return (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else 0.0

    def fps(self):
        d = self.duration()
        return (self.frame_count - 1) / d if d > 0 else 0.0

    # -- pixel access ----------------------------------------------------

    def _offset(self, index, row=0):
        return HEADER_BYTES + index * self.frame_bytes + row * self.row_bytes

    def read_rows(self, index, rows):
        """Read the given row indices of one frame as a flat array of samples.

        This is the workhorse for light-curve sampling: 24 rows out of 2160 is
        ~180 KB instead of the full 16.6 MB, and the seeks stay inside one frame.
        """
        typecode = "H" if self.bytes_per_sample == 2 else "B"
        out = array.array(typecode)
        for r in rows:
            self._fh.seek(self._offset(index, r))
            chunk = array.array(typecode)
            chunk.frombytes(self._fh.read(self.row_bytes))
            out.extend(chunk)
        return out

    def read_frame(self, index):
        """Read one whole frame as a flat array of samples."""
        typecode = "H" if self.bytes_per_sample == 2 else "B"
        self._fh.seek(self._offset(index))
        out = array.array(typecode)
        out.frombytes(self._fh.read(self.frame_bytes))
        return out

    # -- stats -----------------------------------------------------------

    def sample_stats(self, index, rows):
        """Robust brightness statistics for one frame, from sampled rows.

        Returns a dict. `lit` is the fraction of samples well above the black
        floor, which separates a filtered crescent (tiny lit area) from an
        unfiltered corona (large diffuse glow) without either being hardcoded.
        """
        vals = sorted(self.read_rows(index, rows))
        n = len(vals)

        def pct(p):
            return vals[min(n - 1, int(n * p))]

        black = pct(0.05)
        floor = black + max(32, black >> 3)
        # count of samples > floor, via binary search on the sorted list
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if vals[mid] <= floor:
                lo = mid + 1
            else:
                hi = mid
        sat = self.max_value - (self.max_value >> 6)  # within ~1.5% of full scale
        lo2, hi2 = 0, n
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            if vals[mid] < sat:
                lo2 = mid + 1
            else:
                hi2 = mid
        return {
            "black": black,
            "med": pct(0.50),
            "p99": pct(0.99),
            "p999": pct(0.999),
            "max": vals[-1],
            "lit": (n - lo) / n,
            "satfrac": (n - lo2) / n,
        }


def describe(path):
    """One-line summary of a SER file, for logs and sanity checks."""
    with SerFile(path) as s:
        ts = s.timestamps()
        return ("%s %dx%d %db %s n=%d utc=%s-%s %.2fs %.2ffps" % (
            s.path, s.width, s.height, s.depth,
            COLOR_IDS.get(s.color_id, str(s.color_id)), s.frame_count,
            ts[0].strftime("%H:%M:%S"), ts[-1].strftime("%H:%M:%S"),
            s.duration(), s.fps()))
