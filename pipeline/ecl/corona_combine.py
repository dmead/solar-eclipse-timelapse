"""Assemble one exposure level's three channel stacks into linear colour, and
locate the Moon — port of corona-combine.js.

The stacker extracts one CFA channel per run, so each channel arrives as a
separate mono XISF. Because the Bayer extraction halves resolution and the
drizzle puts the factor of two back, the channels land at the sensor's native
3840x2160 and combine without any resampling.

The Moon's centre and radius go to a sidecar: every later stage needs them, for
registration between exposure levels and for the radial profile that makes the
outer corona visible.
"""

import argparse
import json
import os
import time

import numpy as np

from .corona import measure_moon
from .imgio import read_xisf, stack_channels, write_xisf
from . import paths

__all__ = ["combine_level"]


def combine_level(paths, out_path, fixed_r=0.0, log=print):
    """`paths` is (R, G, B) mono XISF. Writes the colour XISF and `_moon.json`."""
    t0 = time.time()
    planes = []
    for c, p in enumerate(paths):
        a = read_xisf(p)
        if a.ndim != 2:
            raise ValueError(f"{p}: expected a mono stack, got shape {a.shape}")
        if c == 0:
            H, W = a.shape
        elif a.shape != (H, W):
            raise ValueError(f"channel geometry mismatch: {p} is "
                             f"{a.shape[1]}x{a.shape[0]}, expected {W}x{H}")
        planes.append(a)
    log(f"combine {W}x{H}")

    # The Moon is measured on G: it carries the most signal and is the channel
    # the registration between levels also works on.
    moon = measure_moon(planes[1], log=log, fixed_r=fixed_r)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_xisf(out_path, stack_channels(*planes))

    side = os.path.splitext(out_path)[0] + "_moon.json"
    payload = {"image": out_path, "width": int(W), "height": int(H),
               "cx": moon["cx"], "cy": moon["cy"], "radius": moon["r"],
               "score": moon["score"]}
    with open(side, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    log(f"  saved {out_path} in {time.time() - t0:.0f} s")
    return payload


def combine_dataset(cfg, levels_dir=None, fixed_r=0.0, only=None, log=print):
    """Combine every segment in an eclipse config into a level image."""
    levels_dir = levels_dir or f"{cfg['outDir']}/levels"
    os.makedirs(levels_dir, exist_ok=True)
    out = []
    for seg in cfg["segments"]:
        if only and seg["id"] not in only:
            continue
        paths = [f"{cfg['stackDir']}/{seg['id']}_{c}.xisf" for c in "RGB"]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            log(f"  {seg['id']}: missing {len(missing)} channel stack(s), skipping")
            continue
        log(f"[{seg['id']}]")
        out.append(combine_level(paths, f"{levels_dir}/{seg['id']}.xisf",
                                 fixed_r=fixed_r, log=log))
    return out


def radius_consensus(payloads, log=print):
    """Median limb radius across levels.

    The radius is a property of the Moon, not of the exposure, so the levels must
    agree. `run-corona.mjs` used this to impose one radius on a re-run: a level
    whose own scan lands elsewhere has been misled by clipping, and taking the
    consensus is what stopped one level measuring a different disc.
    """
    rs = np.array([p["radius"] for p in payloads], dtype=float)
    med = float(np.median(rs))
    spread = float(rs.max() - rs.min())
    log(f"  radius consensus {med:.1f} px across {len(rs)} levels "
        f"(spread {spread:.1f} px)")
    return med


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=paths.in_out("configs", "eclipse.json"))
    ap.add_argument("--levels-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated segment ids")
    ap.add_argument("--fixed-radius", type=float, default=0.0)
    ap.add_argument("--consensus", action="store_true",
                    help="re-run every level with the median radius imposed")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    only = args.only.split(",") if args.only else None

    out = combine_dataset(cfg, args.levels_dir, args.fixed_radius, only)
    if args.consensus and len(out) > 1:
        med = radius_consensus(out)
        print(f"re-running with radius {med:.1f} px imposed")
        out = combine_dataset(cfg, args.levels_dir, med, only)
    for p in out:
        print(f"{os.path.basename(p['image'])}: centre ({p['cx']:.1f}, "
              f"{p['cy']:.1f}) r={p['radius']:.1f}")


if __name__ == "__main__":
    main()
