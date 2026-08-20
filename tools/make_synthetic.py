"""Synthesise a small eclipse shoot, so the pipeline can be run without 460 GB.

    python tools/make_synthetic.py /tmp/eclipse/data
    python -m ecl.run /tmp/eclipse/data

Writes three captures as folders of 16-bit TIFF — partial phases, second contact,
totality at three exposures, third contact, partial phases — with file mtimes
spaced at the frame rate. It exercises every pass: the segmenter has a filter
change and an exposure ladder to find, the disc fits have a crescent and a
corona, and the panel planner has a sunspot, cusps and three prominences.

WHY THE BRIGHTNESS RATIOS MATTER. The first version of this drew each frame
straight into [0, 1], which made the corona about twice darker than the
photosphere instead of a thousand times. The exposure normalisation then
computed gains in the thousands and every rendered frame came out at the
highlight ceiling — a picture that looks like a bug in the renderer and is not.
So a scene here is a RADIANCE map, unbounded and exposure-free, and a frame is
that radiance times an exposure, clipped. Which is what a camera does.

This is a geometry and bookkeeping fixture, not a simulation. Do not read
anything physical out of the rendered video beyond "the pieces line up".
"""

import argparse
import datetime as dt
import os

import numpy as np

H, W = 360, 480
CX, CY = 240.0, 180.0
RSUN = 48.0
FPS = 10.0

# Scene radiance, relative to the photosphere at 1.0.
PHOTOSPHERE = 1.0
CORONA_AT_LIMB = 1.2e-3
CHROMOSPHERE = 6.0e-3
SKY_DAY = 3.0e-3          # scattered daylight, during the filtered phases
SKY_TOTALITY = 4.0e-5     # dusk
FILTER = 1e-5             # white-light filter transmission

# Sensor bias (black level) and read noise, in output units. A real camera never
# reports zero for zero light; without an offset the totality sky here lands at
# half a count, quantises to 0-or-1, and the per-segment gain then amplifies
# that dither into salt-and-pepper across the whole frame.
BIAS = 0.030
READ_NOISE = 0.004

# Exposures the operator chose. Filtered, then the filter comes off with that
# exposure still set (blown), then a ladder ridden down by hand.
EXP_FILTERED = 0.55 / (PHOTOSPHERE * FILTER)
EXP_SHORT, EXP_MID, EXP_LONG = 120.0, 300.0, 750.0

yy, xx = np.mgrid[0:H, 0:W]


def radiance(moon_x, filtered):
    """Scene radiance in photosphere units. No exposure, no clipping."""
    rs = np.hypot(xx - CX, yy - CY)
    rm = np.hypot(xx - moon_x, yy - CY)
    sun = rs <= RSUN
    moon = rm <= RSUN * 1.03

    a = np.full((H, W), SKY_DAY if filtered else SKY_TOTALITY)
    # Corona, falling off steeply with height above the limb.
    a += (CORONA_AT_LIMB * np.exp(-(rm - RSUN * 1.03) / (0.55 * RSUN))
          * (rm > RSUN * 1.03))
    # Prominences: three arcs standing on the limb, not a uniform ring — the
    # panel planner picks discrete maxima and a ring gives it nothing to pick.
    th = np.arctan2(yy - CY, xx - moon_x)
    ring = (rm > RSUN * 1.02) & (rm < RSUN * 1.09)
    for t0, width, amp in ((0.6, 0.16, 1.0), (2.4, 0.10, 0.7), (-1.9, 0.13, 0.85)):
        d = np.abs(np.arctan2(np.sin(th - t0), np.cos(th - t0)))
        a = a + CHROMOSPHERE * amp * np.exp(-(d / width) ** 2) * ring
    # Photosphere wherever the Moon is not, with limb darkening and a sunspot.
    mu = np.sqrt(np.clip(1.0 - (rs / RSUN) ** 2, 0, 1))
    disc = PHOTOSPHERE * (0.4 + 0.6 * mu)
    spot = np.hypot(xx - (CX + 17), yy - (CY - 11)) < 4.5
    disc = np.where(spot, disc * 0.55, disc)
    a = np.where(sun & ~moon, disc, a)
    # The filter sits in front of the whole scene, sky included. Attenuating
    # only the disc leaves a daylight sky 165x over full scale, so every partial
    # frame comes out uniformly white and the segmenter calls the lot "blown".
    return a * FILTER if filtered else a


def frame(rng, moon_x, exposure, filtered):
    """One RGB frame in [0, 1]: radiance x exposure, clipped, plus read noise."""
    a = radiance(moon_x, filtered) * exposure
    a = a + BIAS + rng.normal(0, READ_NOISE, a.shape)
    v = np.clip(a, 0, 1)
    # Halpha makes the chromosphere red, which is what tells the panel planner
    # a prominence from the white corona behind it.
    rm = np.hypot(xx - moon_x, yy - CY)
    red = 1.0 + 0.9 * ((rm > RSUN * 1.02) & (rm < RSUN * 1.09) & (not filtered))
    return np.clip(np.stack([v * red, v * 0.98, v * 0.93], -1), 0, 1)


def save(path, rgb, fmt):
    """Write one frame.

    16-bit TIFF is the default because 8 bits cannot hold this subject. The
    corona needs a gain of several hundred to be visible at all, and at 8 bits
    the quantisation step is 1/255 - multiplied up, it renders as salt-and-
    pepper over the whole frame. That is the format's floor showing, not the
    renderer misbehaving, and it is exactly why nobody shoots totality in JPEG.
    """
    if fmt == "png":
        from PIL import Image

        Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)
    elif fmt == "xisf":
        from ecl.vendor.io.xisf_io import write_xisf

        write_xisf(path, rgb.astype(np.float32))
    else:
        import tifffile

        tifffile.imwrite(path, (rgb * 65535.0).astype(np.uint16))


def write(dirname, frames, t0, fmt="tif"):
    os.makedirs(dirname, exist_ok=True)
    for i, a in enumerate(frames):
        p = os.path.join(dirname, "IMG_%d.%s" % (i + 1, fmt))
        save(p, a, fmt)
        # Image sequences carry no capture clock, so the pipeline falls back to
        # mtime. Space them at the frame rate or every pass that measures a rate
        # sees one instant.
        t = (t0 + dt.timedelta(seconds=i / FPS)).timestamp()
        os.utime(p, (t, t))
    print("  %-10s %4d frames" % (os.path.basename(dirname), len(frames)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="eclipse/data",
                    help="directory to write the captures into")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format", choices=("tif", "png", "xisf"), default="tif",
                    help="frame format; tif is 16-bit and the default")
    args = ap.parse_args(argv)
    fmt = args.format

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)
    base = dt.datetime(2024, 4, 8, 18, 50, 0)
    print("writing to", os.path.abspath(root))

    # Partials: the Moon advances across the disc, filter on.
    a = [frame(rng, CX - 150 + 1.6 * i, EXP_FILTERED, True) for i in range(60)]
    write(os.path.join(root, "18_50_00"), a, base, fmt)

    # C2, the exposure ladder, C3. The blown frames are the filter coming off
    # with the photosphere exposure still set, which is how the pipeline
    # brackets totality physically rather than by a brightness threshold.
    b = [frame(rng, CX + 3, EXP_FILTERED, False) for _ in range(12)]
    b += [frame(rng, CX + 4 + 0.02 * i, EXP_SHORT, False) for i in range(40)]
    b += [frame(rng, CX + 5 + 0.02 * i, EXP_MID, False) for i in range(40)]
    b += [frame(rng, CX + 6 + 0.02 * i, EXP_LONG, False) for i in range(40)]
    b += [frame(rng, CX - 3, EXP_FILTERED, False) for _ in range(12)]
    write(os.path.join(root, "18_51_00"), b,
          base + dt.timedelta(seconds=70), fmt)

    # Partials again, filter back on.
    c = [frame(rng, CX + 6 + 1.6 * i, EXP_FILTERED, True) for i in range(60)]
    write(os.path.join(root, "18_54_00"), c,
          base + dt.timedelta(seconds=70 + len(b) / FPS + 20), fmt)

    print("now run:  python -m ecl.run %s" % os.path.abspath(root))


if __name__ == "__main__":
    main()
