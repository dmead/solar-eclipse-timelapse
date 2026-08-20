"""One reader API over SER files and directories of single images.

The pipeline was written against SER because that is what a planetary camera
writes, and twelve modules import `EclipseSer` directly. A DSLR or mirrorless
body produces something else entirely: a folder of CR2/NEF/TIFF/JPEG, or FITS
from an astro camera, one exposure per file, with the timestamp in EXIF or in a
FITS header rather than in a binary trailer.

`Source` is the shape both fit. A capture is "an ordered run of exposures from
one camera at one setting", whether that is a 4 GB SER or a folder of 300 CR2s,
and everything downstream only ever needs: how many frames, how big, how deep,
when was each taken, and give me the R/G/B planes of frame i.

WHAT A "PLANE" MEANS IS DELIBERATELY NOT THE SAME on both paths, and callers
must not assume it is.

  - A CFA sensor gives 2x2 superpixel planes at HALF the sensor dimensions. That
    is the existing behaviour and it costs no interpolation.
  - An already-demosaiced image gives full-resolution planes. There is no mosaic
    to reduce, and throwing away half the resolution to imitate the SER path
    would be destroying real data.

So `plane_scale` reports the ratio, and the geometry the pipeline measures - disc
radius above all - is always in PLANE pixels. Nothing downstream should convert
between the two on its own; ask the source.
"""

import os
import re

import numpy as np

from . import demosaic

__all__ = ["Source", "SerSource", "ImageSource", "open_source", "discover",
           "frame_stats", "IMAGE_EXTS"]

# Extensions treated as one-exposure-per-file. RAW formats are included only if
# rawpy is installed; the loader says so rather than failing at frame 400.
IMAGE_EXTS = {".fit", ".fits", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
              ".xisf",
              ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
FITS_EXTS = {".fit", ".fits", ".fts"}
# TIFF does NOT go through Pillow. Pillow opens a 16-bit RGB TIFF and hands back
# uint8 without saying so - eight bits of an astronomical exposure silently
# gone, which for a corona spanning three decades is most of the picture.
# tifffile is already a hard dependency and reads it as written.
TIFF_EXTS = {".tif", ".tiff"}
# XISF needs no optional dependency: `xisf` is a hard requirement already,
# because the corona stages write their intermediates in it. A folder of XISF
# is what you have if you came here from PixInsight, calibrated elsewhere, or
# exported from an earlier run of this pipeline.
XISF_EXTS = {".xisf"}
SER_EXTS = {".ser"}

_NUM = re.compile(r"(\d+)")


def _natural(name):
    """Sort key that orders IMG_9 before IMG_10."""
    return [int(t) if t.isdigit() else t.lower() for t in _NUM.split(name)]


def frame_stats(samples, max_value):
    """Robust brightness statistics for one frame, from a set of its pixels.

    This is the measurement the whole segmenter rests on, and it is defined
    HERE, once, because it used to live inside the SER reader - which is the
    reason segmentation only ever worked on SER files. Both readers feed it now
    and neither owns it.

    `lit` is the fraction of samples well above the black floor. That is what
    separates a filtered crescent (a tiny lit area in a black frame) from an
    unfiltered corona (a large diffuse glow) without hardcoding a brightness for
    either. `satfrac` is the fraction within ~1.5% of full scale, which is what
    catches a frame blown out by the filter coming off.

    EVERYTHING HERE IS A FRACTION OF FULL SCALE, in and out. The stats used to
    be raw sample values, which quietly made the whole segmenter 16-bit-only:
    downstream, `log2(max(v, 1))` floored every level at one ADU and a float
    frame normalised to [0, 1] has no value above 1, so its entire light curve
    flattened to zero and no exposure change was ever found. Ratios - which is
    all any caller wants - are identical either way.

    Thresholds are fractions for the same reason. The original added a floor of
    32 counts regardless of bit depth: ~0.05% of a 16-bit frame and 13% of an
    8-bit one. 1/2048 of full scale reproduces the 16-bit behaviour exactly and
    behaves sanely on 8-bit and float frames.
    """
    v = np.sort(np.asarray(samples).ravel())
    n = v.size
    if n == 0:
        raise ValueError("no samples")

    def pct(q):
        return float(v[min(n - 1, int(n * q))])

    black = pct(0.05)
    floor = black + max(max_value / 2048.0, black * 0.125)
    sat = max_value * (1.0 - 1.0 / 64.0)
    m = float(max_value) or 1.0
    return {
        "black": black / m,
        "med": pct(0.50) / m,
        "p99": pct(0.99) / m,
        "p999": pct(0.999) / m,
        "max": float(v[-1]) / m,
        "lit": float(n - np.searchsorted(v, floor, "right")) / n,
        "satfrac": float(n - np.searchsorted(v, sat, "left")) / n,
    }


class Source:
    """Common interface. Use as a context manager."""

    # -- identity -------------------------------------------------------
    name = ""
    path = ""

    @property
    def frame_count(self):
        raise NotImplementedError

    @property
    def width(self):
        """Sensor width, in SENSOR px."""
        raise NotImplementedError

    @property
    def height(self):
        raise NotImplementedError

    @property
    def max_value(self):
        raise NotImplementedError

    @property
    def plane_scale(self):
        """PLANE px per SENSOR px. 0.5 for a CFA superpixel, 1.0 otherwise."""
        raise NotImplementedError

    @property
    def is_cfa(self):
        return self.plane_scale != 1.0

    @property
    def cfa_pattern(self):
        """Bayer layout id, or None when the frames are already RGB.

        Only the full-resolution demosaic needs this. Everything else takes
        `planes` and never learns how they were made.
        """
        return None

    @property
    def depth(self):
        """Bits per sample as stored. Reported, never used to scale anything."""
        raise NotImplementedError

    # -- statistics -----------------------------------------------------
    def sample_stats(self, i, nrows):
        """`frame_stats` over `nrows` evenly spaced rows of frame i.

        The default reads the whole frame, because for one-image-per-file
        formats there is no cheaper option - a PNG or a RAW has to be decoded
        before any row of it can be looked at. `SerSource` overrides it with a
        genuine partial read, which is where the saving that makes a light curve
        affordable over hundreds of gigabytes actually comes from.
        """
        a = np.asarray(self.raw(i))
        rows = self._row_indices(a.shape[0], nrows)
        return frame_stats(a[rows], self.max_value)

    @staticmethod
    def _row_indices(height, nrows):
        return [r * height // nrows for r in range(nrows)]

    # -- pixels ---------------------------------------------------------
    def planes(self, i):
        """(R, G, B) in [0, 1] at plane resolution."""
        raise NotImplementedError

    def green(self, i):
        """Green plane in [0, 1] — what every fit and alignment measures on."""
        return self.planes(i)[1]

    def raw(self, i):
        """Frame as stored, native dtype, undemosaiced where that applies."""
        raise NotImplementedError

    # -- time -----------------------------------------------------------
    def timestamps(self):
        raise NotImplementedError

    def fps(self):
        ts = self.timestamps()
        if len(ts) < 2:
            return 0.0
        span = (ts[-1] - ts[0]).total_seconds()
        return (len(ts) - 1) / span if span > 0 else 0.0

    # Aliases so a Source drops straight into code written against EclipseSer.
    @property
    def raw_width(self):
        return self.width

    @property
    def raw_height(self):
        return self.height

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def describe(self):
        kind = "CFA" if self.is_cfa else "RGB"
        return (f"{self.name} {self.width}x{self.height} {kind} "
                f"n={self.frame_count} {self.fps():.2f}fps")


class SerSource(Source):
    """A SER capture, via the existing reader."""

    def __init__(self, path):
        from .serio import EclipseSer

        self.path = str(path)
        self.name = os.path.basename(self.path)
        self._ser = EclipseSer(self.path)

    @property
    def frame_count(self):
        return self._ser.frame_count

    @property
    def width(self):
        return self._ser.raw_width

    @property
    def height(self):
        return self._ser.raw_height

    @property
    def max_value(self):
        return self._ser.max_value

    @property
    def plane_scale(self):
        # colour_id 0 is mono; anything else here is a Bayer mosaic.
        return 0.5 if self._ser.color_id else 1.0

    @property
    def cfa_pattern(self):
        return self._ser.color_id or None

    @property
    def depth(self):
        return self._ser.meta.depth

    def sample_stats(self, i, nrows):
        """24 rows out of 2160 is ~180 KB instead of 16.6 MB, and the seeks all
        stay inside one frame. Over 29000 frames that is the difference between
        a light curve costing minutes and costing an afternoon."""
        rows = self._row_indices(self._ser.meta.height, nrows)
        vals = self._ser.meta.read_rows(i, rows)
        return frame_stats(np.frombuffer(vals, dtype=vals.typecode),
                           self.max_value)

    def planes(self, i):
        if self.plane_scale == 1.0:
            g = self._ser.raw(i).astype(np.float32) / self.max_value
            return g, g, g
        return self._ser.planes(i)

    def green(self, i):
        if self.plane_scale == 1.0:
            return self._ser.raw(i).astype(np.float32) / self.max_value
        return self._ser.green(i)

    def raw(self, i):
        return self._ser.raw(i)

    def timestamps(self):
        return self._ser.timestamps()

    def fps(self):
        return self._ser.fps()

    def close(self):
        self._ser.close()


class ImageSource(Source):
    """A directory of single exposures, one file per frame.

    Files are ordered NATURALLY, not lexically: a camera writing IMG_0009 then
    IMG_0010 sorts wrongly under a plain string sort exactly once every power of
    ten, and the resulting frame order is a silent corruption rather than an
    error.
    """

    def __init__(self, path, files=None):
        self.path = str(path)
        self.name = os.path.basename(self.path.rstrip("/\\"))
        if files is None:
            files = [f for f in os.listdir(self.path)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        self.files = sorted(files, key=_natural)
        if not self.files:
            raise ValueError(f"no images in {self.path}")
        self._meta = None
        self._cache = (None, None)

    # -- lazy probe of the first frame ----------------------------------
    def _probe(self):
        if self._meta is None:
            a = self._read(0)
            mono = a.ndim == 2
            self._meta = {
                "h": a.shape[0], "w": a.shape[1], "mono": mono,
                "max": 65535.0 if a.dtype == np.uint16 else
                       (255.0 if a.dtype == np.uint8 else 1.0),
                "dtype": a.dtype,
            }
        return self._meta

    def _read(self, i):
        """Raw array for frame i, cached one deep: every caller asks for the
        green plane and then often the full set, and a RAW decode is expensive
        enough that doing it twice is the difference between minutes and hours."""
        if self._cache[0] == i:
            return self._cache[1]
        p = os.path.join(self.path, self.files[i])
        ext = os.path.splitext(p)[1].lower()
        if ext in RAW_EXTS:
            try:
                import rawpy
            except ImportError as e:
                raise RuntimeError(
                    f"{ext} needs rawpy: pip install rawpy") from e
            with rawpy.imread(p) as r:
                a = r.postprocess(output_bps=16, no_auto_bright=True,
                                  use_camera_wb=True, gamma=(1, 1))
        elif ext in TIFF_EXTS:
            import tifffile

            a = tifffile.imread(p)
            if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
                a = np.moveaxis(a, 0, -1)      # planes-first
        elif ext in XISF_EXTS:
            from .vendor.io.xisf_io import read_xisf

            a = read_xisf(p)          # float32 in [0, 1], (H,W) or (H,W,3)
        elif ext in FITS_EXTS:
            try:
                from astropy.io import fits
            except ImportError as e:
                raise RuntimeError(
                    "FITS needs astropy: pip install astropy") from e
            with fits.open(p, memmap=False) as hdul:
                d = next(h.data for h in hdul if h.data is not None)
            a = np.squeeze(d)
            if a.ndim == 3 and a.shape[0] in (3, 4):     # planes-first
                a = np.moveaxis(a, 0, -1)
        else:
            from PIL import Image

            with Image.open(p) as im:
                a = np.asarray(im)
        self._cache = (i, a)
        return a

    @property
    def frame_count(self):
        return len(self.files)

    @property
    def width(self):
        return self._probe()["w"]

    @property
    def height(self):
        return self._probe()["h"]

    @property
    def max_value(self):
        return self._probe()["max"]

    @property
    def plane_scale(self):
        return 1.0            # already demosaiced, or genuinely mono

    @property
    def depth(self):
        m = self._probe()["max"]
        return 8 if m == 255.0 else (16 if m == 65535.0 else 32)

    def planes(self, i):
        a = self._read(i).astype(np.float32) / self._probe()["max"]
        if a.ndim == 2:
            return a, a, a
        return a[:, :, 0], a[:, :, 1], a[:, :, 2]

    def green(self, i):
        a = self._read(i)
        m = self._probe()["max"]
        if a.ndim == 2:
            return a.astype(np.float32) / m
        return a[:, :, 1].astype(np.float32) / m

    def raw(self, i):
        return self._read(i)

    def timestamps(self):
        """Capture time per frame, from metadata where there is any.

        Falls back to file mtime, which is usually the write time and therefore
        close enough to order and pace frames, but is NOT a clock: it survives a
        copy on Windows and not always elsewhere. Anything depending on absolute
        UTC should check `has_real_times` first.
        """
        import datetime as _dt

        if getattr(self, "_ts", None) is not None:
            return self._ts
        out, real = [], True
        for f in self.files:
            p = os.path.join(self.path, f)
            t = _exif_time(p) or _fits_time(p) or _xisf_time(p)
            if t is None:
                real = False
                t = _dt.datetime.utcfromtimestamp(os.path.getmtime(p))
            out.append(t)
        self.has_real_times = real
        self._ts = out
        return out

    def close(self):
        self._cache = (None, None)


def _exif_time(path):
    import datetime as _dt
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return None
            tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
            s = tags.get("DateTimeOriginal") or tags.get("DateTime")
            if s:
                return _dt.datetime.strptime(str(s), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def _fits_time(path):
    import datetime as _dt
    if os.path.splitext(path)[1].lower() not in FITS_EXTS:
        return None
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            for h in hdul:
                s = h.header.get("DATE-OBS")
                if s:
                    return _dt.datetime.fromisoformat(str(s).replace("Z", ""))
    except Exception:
        return None
    return None


def _xisf_time(path):
    """Capture time from an XISF header.

    Two places carry it and both are common: the XISF property
    `Observation:Time:Start`, and a FITS `DATE-OBS` keyword carried through by
    whatever wrote the file. Try the native property first.
    """
    import datetime as _dt
    if os.path.splitext(path)[1].lower() not in XISF_EXTS:
        return None
    try:
        from xisf import XISF

        meta = XISF(path).get_images_metadata()
        if not meta:
            return None
        m = meta[0]
        prop = m.get("XISFProperties", {}).get("Observation:Time:Start")
        if prop and prop.get("value") is not None:
            v = prop["value"]
            if isinstance(v, _dt.datetime):
                return v.replace(tzinfo=None)
            return _dt.datetime.fromisoformat(str(v).replace("Z", ""))
        kw = m.get("FITSKeywords", {}).get("DATE-OBS")
        if kw:
            return _dt.datetime.fromisoformat(
                str(kw[0]["value"]).strip().strip("'").replace("Z", ""))
    except Exception:
        return None
    return None


def open_source(path):
    """Open a SER file or an image directory as a Source."""
    p = str(path)
    if os.path.isdir(p):
        return ImageSource(p)
    if os.path.splitext(p)[1].lower() in SER_EXTS:
        return SerSource(p)
    raise ValueError(f"not a capture: {p}")


def discover(data_dir):
    """Captures under `data_dir`, in time order where that can be known.

    Three layouts are accepted, because all three are what people actually have:
    SER files side by side, one subdirectory of images per capture, and a single
    flat directory of images that IS one capture.
    """
    data_dir = str(data_dir)
    if not os.path.isdir(data_dir):
        raise ValueError(f"not a directory: {data_dir}")
    ents = sorted(os.listdir(data_dir), key=_natural)

    sers = [os.path.join(data_dir, e) for e in ents
            if os.path.splitext(e)[1].lower() in SER_EXTS]
    if sers:
        return sers

    subs = []
    for e in ents:
        d = os.path.join(data_dir, e)
        if os.path.isdir(d) and any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTS
                for f in os.listdir(d)):
            subs.append(d)
    if subs:
        return subs

    if any(os.path.splitext(e)[1].lower() in IMAGE_EXTS for e in ents):
        return [data_dir]
    return []
