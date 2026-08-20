"""SER access for the eclipse pipeline — pixels from lunation, times from serlib.

The two readers in play are complementary, not redundant:

- `lunation.io.ser.SerReader` is memmap-backed and fast, and knows all four Bayer
  orders — but it stops at the 178-byte header. It has no notion of the trailer.
- `serlib.SerFile` is seek-based stdlib with no numpy, and reads the per-frame UTC
  trailer, which is what every stage here actually uses for time. SharpCap's
  filenames are UTC-5 and its header TimeZone is wrong (EST, not the EDT in
  force); the trailer is the only correct clock in the file.

`EclipseSer` is the one object with both halves. Pixel reads go through the
memmap, timestamps and header metadata through serlib, and the reader is opened
lazily so metadata-only work (segmentation, planning) never pays for it.
"""

import os

import numpy as np

from . import demosaic
from .serlib import COLOR_IDS, SerFile, ticks_to_utc
from .vendor.io.ser import SerReader

__all__ = ["EclipseSer", "COLOR_IDS", "ticks_to_utc", "restage"]


def restage(cfg, data_dir, log=print):
    """Point a config's frames at a copy of the captures on another volume.

    The configs carry absolute `frames[].src` paths, so staging the 449 GB of
    captures somewhere faster would otherwise mean rewriting timelapse.json —
    which the four config passes rewrite in place anyway, making it exactly the
    file you do not want to edit for a transient reason. Every pass that reads
    SER takes this instead. Missing captures raise rather than silently render a
    short sequence.
    """
    if not data_dir:
        return cfg
    d = str(data_dir).rstrip("/\\")
    dbase = os.path.basename(d)
    for fr in cfg["frames"]:
        base = os.path.basename(str(fr["src"]).rstrip("/\\"))
        # A capture is normally a file or a subfolder inside the data root. The
        # one exception is a single flat run of images, where the capture IS the
        # root - there the basename joins the root to itself and the result does
        # not exist. Recognise that case rather than reporting the user's own
        # data directory as a missing capture.
        fr["src"] = d if base == dbase else f"{d}/{base}"
    missing = {fr["src"] for fr in cfg["frames"] if not os.path.exists(fr["src"])}
    if missing:
        raise SystemExit(f"--data-dir is missing {len(missing)} capture(s), "
                         f"e.g. {sorted(missing)[0]}")
    log(f"reading captures from {d}")
    return cfg


class EclipseSer:
    """Random-access SER reader carrying both pixels and correct frame times.

    Use as a context manager. Frame indices are 0-based and count from the start
    of this file — for a slice, that is the slice's own numbering, not the
    parent capture's.
    """

    def __init__(self, path, channel="mono"):
        self.path = str(path)
        self.meta = SerFile(self.path)
        self._channel = channel
        self._reader = None

    # -- lifecycle -------------------------------------------------------

    @property
    def reader(self):
        """The memmap reader, opened on first pixel access."""
        if self._reader is None:
            self._reader = SerReader(self.path, channel=self._channel)
        return self._reader

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.meta.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- geometry and metadata -------------------------------------------

    @property
    def frame_count(self):
        """Frames the data actually backs, not what the header claims.

        A capture interrupted mid-write leaves the two disagreeing; lunation's
        reader already applies the truncated-file guard, so prefer it once open.
        """
        if self._reader is not None:
            return self._reader.frame_count
        return min(self.meta.frame_count,
                   (self.meta.file_size - 178) // self.meta.frame_bytes)

    @property
    def raw_width(self):
        return self.meta.width

    @property
    def raw_height(self):
        return self.meta.height

    @property
    def color_id(self):
        return self.meta.color_id

    @property
    def bayer(self):
        return self.meta.bayer

    @property
    def max_value(self):
        return self.meta.max_value

    def describe(self):
        ts = self.timestamps()
        return (f"{self.path} {self.raw_width}x{self.raw_height} "
                f"{self.meta.depth}b {COLOR_IDS.get(self.color_id, self.color_id)} "
                f"n={self.frame_count} "
                f"utc={ts[0]:%H:%M:%S}-{ts[-1]:%H:%M:%S} "
                f"{self.duration():.2f}s {self.fps():.2f}fps")

    # -- time ------------------------------------------------------------

    def timestamp(self, index):
        return self.meta.timestamp(index)

    def timestamps(self):
        return self.meta.timestamps()

    def duration(self):
        return self.meta.duration()

    def fps(self):
        return self.meta.fps()

    # -- pixels ----------------------------------------------------------

    def raw(self, index):
        """Frame as stored: the undemosaiced CFA mosaic, native dtype."""
        return self.reader.read_raw(index)

    def mono(self, index):
        """Half-resolution 2x2 cell average in [0, 1]."""
        return demosaic.superpixel_mono(self.raw(index), self.max_value)

    def planes(self, index):
        """Half-resolution (R, G, B) superpixel planes in [0, 1]."""
        return demosaic.superpixel(self.raw(index), self.color_id, self.max_value)

    def green(self, index):
        """Half-resolution green plane in [0, 1] — the alignment channel.

        Alignment and every disc fit are measured on G throughout: it carries the
        most signal of the three, and measuring once and applying to all three
        channels is what stops the colours drifting apart by a fraction of a
        pixel. Computed without materialising R and B, since the detection passes
        run this over every frame in the sequence.
        """
        return demosaic.green_plane(self.raw(index), self.color_id, self.max_value)

    def rgb_full(self, index):
        """Full-resolution bilinear-demosaiced (3, H, W) in [0, 1]."""
        return demosaic.bilinear_rggb(self.raw(index), max_value=self.max_value)

    def frames(self, start=0, count=None, stride=1):
        """Iterate (index, raw frame) without holding more than one in memory."""
        n = self.frame_count if count is None else min(count, self.frame_count - start)
        for i in range(start, start + n, stride):
            yield i, self.raw(i)


def open_ser(path, channel="mono"):
    """Convenience opener mirroring `serlib.SerFile` usage."""
    return EclipseSer(path, channel=channel)
