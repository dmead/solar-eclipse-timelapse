"""Export selected SER frames as debayered FITS for Umbra.

Umbra expects calibrated, debayered FITS and groups exposures by a FITS keyword.
This session never recorded the totality exposure times — the operator rode the
exposure by hand and SharpCap only wrote a settings file for one early capture —
so EXPTIME is synthesised from the measured background level of each segment,
which is linear in exposure time. Umbra only needs it to group frames and order
them, and a linear proxy does both correctly.

Run with the Umbra venv's Python, which has numpy and astropy:

    D:/projects/umbra/venv/Scripts/python.exe ser_to_fits.py --out <dir>
"""

import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecl.demosaic import bilinear_rggb  # noqa: E402
from serlib import SerFile  # noqa: E402

# Full-resolution image scale: the Sun's limb was fitted at 558 px radius against
# a true angular radius of 959 arcsec on 2024-04-08.
IMAGE_SCALE = 959.0 / 558.0

# Frames to export per segment, spread evenly across it.
PER_SEGMENT = 6


def debayer_rggb(raw, W, H):
    """Bilinear RGGB demosaic at full resolution.

    Full resolution rather than 2x2 superpixel: fitting the Moon's limb and
    correlating fine coronal structure both need the sampling that halving the
    resolution would throw away.

    NOTE: the hand-unrolled implementation that used to live here was wrong at
    green sites — half of all pixels. It mixed R and B into each other on even
    rows and read the green value into R on odd rows, so every FITS this script
    exported before 2026-08-15 has bad colour on half its pixels. Anything
    derived from those exports should be regenerated. The shared implementation
    derives its neighbour offsets from the CFA layout instead, and is checked
    against a synthetic flat-colour mosaic in `tests/test_demosaic.py`.
    """
    return bilinear_rggb(raw, width=W, height=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="S:/solar-eclipse/Sun")
    ap.add_argument("--segments", default="S:/solar-eclipse/out/segments.json")
    ap.add_argument("--lightcurve", default="S:/solar-eclipse/out/lightcurve.json")
    ap.add_argument("--out", default="S:/solar-eclipse/umbra/fits")
    ap.add_argument("--per-segment", type=int, default=PER_SEGMENT)
    ap.add_argument("--levels", default=None,
                    help="comma-separated level names to keep, e.g. L0,L1,L2")
    ap.add_argument("--max-satfrac", type=float, default=1.0,
                    help="skip segments whose median saturated fraction exceeds this")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    man = json.load(open(args.segments))
    lc = json.load(open(args.lightcurve))
    pts = {f["name"]: {p["i"]: p for p in f["points"]} for f in lc["files"]}

    jobs = []
    for f in man["files"]:
        for s in f["segments"]:
            if s["state"] != "unfiltered" or s["kind"] != "stable":
                continue
            inside = [p for p in pts[f["name"]].values()
                      if s["start"] <= p["i"] < s["start"] + s["count"]]
            if not inside:
                continue
            black = sorted(p["black"] for p in inside)[len(inside) // 2]
            level = max(s["med"] - black, 1)
            satfrac = sorted(p["satfrac"] for p in inside)[len(inside) // 2]
            jobs.append({"file": f["name"], "start": s["start"], "count": s["count"],
                         "level": level, "name": s.get("level_name", "L?"),
                         "satfrac": satfrac})

    # Keep only true totality. L3 sits before second contact and is still
    # photosphere; L4 is the exposure ramp and clips 17% of the frame. Both
    # wreck the HDR fit - L3's stack came out 25% blank because it is far enough
    # from the anchors in time that the interpolated shift pushes it off frame.
    if args.levels:
        keep = set(args.levels.split(","))
        jobs = [j for j in jobs if j["name"] in keep]
    jobs = [j for j in jobs if j["satfrac"] <= args.max_satfrac]

    # EXPTIME proxy in milliseconds, anchored so the shortest segment reads 1 ms.
    lo = min(j["level"] for j in jobs)
    for j in jobs:
        j["exptime"] = round(j["level"] / lo, 4)

    print("%-14s %-6s %-5s %-8s %s" % ("file", "start", "lvl", "EXPTIME", "frames"))
    total = 0
    for j in sorted(jobs, key=lambda j: j["exptime"]):
        path = os.path.join(args.data, j["file"])
        with SerFile(path) as ser:
            n = min(args.per_segment, j["count"])
            step = max(1, j["count"] // n)
            idx = [j["start"] + k * step for k in range(n)]
            idx = [i for i in idx if i < j["start"] + j["count"]]
            ts = ser.timestamps()
            for i in idx:
                raw = np.frombuffer(ser.read_frame(i), dtype=np.uint16)
                # Normalise to [0,1]. Umbra's HDR thresholds are expressed in
                # that range (0.001 to 0.8 in the shipped example), so handing it
                # raw ADU would put every pixel above the high threshold and the
                # composition would reject the whole frame.
                rgb = debayer_rggb(raw, ser.width, ser.height) / 65535.0
                hdr = fits.Header()
                hdr["EXPTIME"] = j["exptime"]
                hdr["DATE-OBS"] = ts[i].isoformat()
                hdr["INSTRUME"] = "ZWO ASI585MC"
                hdr["XPIXSZ"] = IMAGE_SCALE
                hdr["SRCFILE"] = j["file"]
                hdr["SRCFRAME"] = i
                hdr["LEVEL"] = j["name"]
                out = os.path.join(
                    args.out, "%s_f%05d.fits" % (j["file"].replace(".ser", ""), i))
                fits.PrimaryHDU(data=rgb.astype(np.float32), header=hdr).writeto(
                    out, overwrite=True)
                total += 1
        print("%-14s %-6d %-5s %-8.3f %d" % (j["file"], j["start"], j["name"],
                                             j["exptime"], len(idx)))
    print("\n%d FITS -> %s   image_scale %.3f arcsec/px" % (total, args.out, IMAGE_SCALE))


if __name__ == "__main__":
    main()
