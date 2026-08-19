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
           "IMAGE_EXTS"]

# Extensions treated as one-exposure-per-file. RAW formats are included only if
# rawpy is installed; the loader says so rather than failing at frame 400.
IMAGE_EXTS = {".fit", ".fits", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
              ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
SER_EXTS = {".ser"}

_NUM = re.compile(r"(\d+)")


def _natural(name):
    """Sort key that orders IMG_9 before IMG_10."""
    return [int(t) if t.isdigit() else t.lower() for t in _NUM.split(name)]


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
        elif ext in (".fit", ".fits", ".fts"):
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
            t = _exif_time(p) or _fits_time(p)
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
    if os.path.splitext(path)[1].lower() not in (".fit", ".fits", ".fts"):
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
