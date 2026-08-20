"""Look at the data and the machine, then choose the runtime parameters.

Nothing in the pipeline should carry a number that only makes sense for the
camera it was written on. This dataset is a 3840x2160 colour planetary camera at
prime focus, where the Sun's disc is 279 px in the superpixel plane; a full-frame
body at a different focal length can put it anywhere from 60 px to 2000. Every
geometric constant downstream is therefore expressed as a FRACTION OF THE SOLAR
RADIUS and resolved here, and the radius itself is measured rather than assumed.

Three things are surveyed:

  the data     format, sensor size, bit depth, CFA or already-demosaiced, how
               many captures and frames, cadence, and the disc radius
  the machine  cores and free memory, which set the worker count - a full-frame
               sensor at drizzle 2 needs about 4.6 GB per worker against 0.4 GB
               here, so a count that is right on one machine will thrash on
               another
  the plan     drizzle factor and output window, derived from the two above

THE RADIUS IS MEASURED FROM AREA, not from a limb fit. A limb fit is more precise
and needs a centre, a radius prior and a tuned edge criterion - all of which are
what this survey exists to produce. Bright-area radius needs none of them: the
Sun is the only large bright object in the frame, so sqrt(A/pi) over the largest
bright component is exact for an uneclipsed disc and simply an underestimate for
an eclipsed one. Taking the MAXIMUM across sampled frames finds the least
eclipsed frame in the set and reports that.
"""

import argparse
import json
import math
import os

import numpy as np

from .source import discover, open_source

__all__ = ["survey", "measure_radius", "plan_runtime"]

# Fraction of a frame's own peak counted as "disc" when measuring area.
#
# Half the peak, so it sits far below the photosphere and far above both the
# corona and any sky gradient. A filtered solar disc is nearly flat-topped, so
# the exact level barely moves the area; on an unfiltered totality frame nothing
# reaches it and the frame is skipped, which is correct - there is no disc to
# measure there.
DISC_LEVEL = 0.5

# Smallest bright component worth believing, as a fraction of the frame.
MIN_DISC_FRAC = 1e-4

# Area over bounding box for something round enough to be the disc. A full
# circle is pi/4 = 0.785; the Moon takes a bite out of that, so the bound is
# well below it and only rejects genuinely ragged regions.
MIN_FILL = 0.35

# Fine-grid disc radius the drizzle factor aims for. Below this the sampling is
# the limit on detail and upsampling buys real resolution back; above it the
# optics are, and drizzling only multiplies the pixel count.
TARGET_FINE_R = 500.0
MAX_DRIZZLE = 2

# Working copies of the fine grid a render holds at once, per channel. Measured
# on this pipeline: the accumulator, the upsampled frame, the translated copy and
# the output crop, plus headroom for the FFT.
WORK_COPIES = 5.0

# Never plan to use more than this share of free memory.
MEM_SAFETY = 0.6


def measure_radius(src, samples=9, log=None):
    """Disc radius in PLANE px, from the least-eclipsed frame sampled.

    THE BRIGHT REGION MUST NOT TOUCH THE FRAME EDGE. Without that test this
    measured 799 px on a dataset whose disc is 279: the frames where the solar
    filter comes off are more than half saturated, so nearly the whole frame
    clears the threshold and the "disc" it finds is the picture. A correctly
    framed solar disc never reaches the border, and a blown frame always does, so
    one test removes the whole failure mode without assuming anything about
    composition.

    Circularity is checked for the same reason from the other side: a bright
    region that is not round is not the disc, whatever its area.
    """
    import cv2

    n = src.frame_count
    if n == 0:
        return 0.0
    idx = sorted({int(round(k * (n - 1) / max(samples - 1, 1)))
                  for k in range(samples)})
    best = 0.0
    for i in idx:
        g = src.green(i)
        pk = float(g.max())
        if pk <= 0:
            continue
        m = (g >= DISC_LEVEL * pk).astype(np.uint8)
        nl, _lab, st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        h, w = m.shape
        for k in range(1, nl):
            x, y, bw, bh, area = (st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP],
                                  st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT],
                                  st[k, cv2.CC_STAT_AREA])
            if area < MIN_DISC_FRAC * g.size:
                continue
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue                      # runs off the frame: not the disc
            r = math.sqrt(area / math.pi)
            # A disc fills pi/4 of its bounding box; allow for it being clipped
            # by the Moon, which lowers the fill but never raises it.
            fill = area / max(bw * bh, 1)
            if not (MIN_FILL <= fill <= 1.0):
                continue
            best = max(best, r)
    if log:
        log(f"    disc radius {best:.1f} plane px from {len(idx)} frames")
    return best


def plan_runtime(plane_w, plane_h, r_plane, cores=None, free_bytes=None):
    """Drizzle factor, worker count and output window for this data and machine."""
    import psutil

    cores = cores or _physical_cores()
    free_bytes = free_bytes or psutil.virtual_memory().available

    drizzle = 1
    if r_plane > 0:
        drizzle = max(1, min(MAX_DRIZZLE, int(round(TARGET_FINE_R / r_plane))))

    per_worker = plane_w * plane_h * (drizzle ** 2) * 4 * 3 * WORK_COPIES
    by_mem = max(1, int(free_bytes * MEM_SAFETY // max(per_worker, 1)))
    workers = max(1, min(cores, by_mem))
    return {
        "drizzle": drizzle,
        "workers": workers,
        "workers_by_cores": cores,
        "workers_by_memory": by_mem,
        "bytes_per_worker": int(per_worker),
        "free_bytes": int(free_bytes),
    }


def _physical_cores():
    try:
        from .affinity import core_groups
        n = len(core_groups())
        if n:
            return n
    except Exception:
        pass
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        return os.cpu_count() or 4


def survey(data_dir, samples=9, log=print):
    """Everything the pipeline needs to know before it is configured."""
    caps = discover(data_dir)
    if not caps:
        raise SystemExit(
            f"no captures found in {data_dir}\n"
            f"  expected .ser files, or image files "
            f"(.fits/.tif/.png/.jpg/.cr2/...), either directly in that folder "
            f"or one folder per capture")

    log(f"surveying {len(caps)} capture(s) in {data_dir}")
    files, radii = [], []
    plane_w = plane_h = 0
    kinds, scales = set(), set()
    for p in caps:
        with open_source(p) as s:
            ts = []
            real = True
            try:
                ts = s.timestamps()
                real = getattr(s, "has_real_times", True)
            except Exception:
                real = False
            r = measure_radius(s, samples)
            pw = int(s.width * s.plane_scale)
            ph = int(s.height * s.plane_scale)
            plane_w, plane_h = max(plane_w, pw), max(plane_h, ph)
            kinds.add("cfa" if s.is_cfa else "rgb")
            scales.add(s.plane_scale)
            if r > 0:
                radii.append(r)
            files.append({
                "name": s.name, "path": str(p),
                "frames": s.frame_count,
                "sensor": [s.width, s.height],
                "plane": [pw, ph],
                "plane_scale": s.plane_scale,
                "cfa": s.is_cfa,
                "max_value": s.max_value,
                "fps": round(s.fps(), 4),
                "radius_plane_px": round(r, 2),
                "t0_utc": ts[0].isoformat() if ts else None,
                "t1_utc": ts[-1].isoformat() if ts else None,
                "real_times": bool(real),
            })
            log(f"  {s.describe()}  r={r:.1f}")

    if len(kinds) > 1:
        raise SystemExit(
            f"mixed capture kinds {sorted(kinds)} - the pipeline needs one "
            f"geometry throughout; split them into separate runs")

    # The radius that matters is the LARGEST seen: the least-eclipsed frame in
    # the whole set is the only one showing the full disc.
    r_plane = max(radii) if radii else 0.0
    if r_plane <= 0:
        raise SystemExit(
            "no frame in this data has a measurable solar disc - every frame "
            "sampled was either blank or entirely eclipsed. Set "
            "geometry.radius_plane_px in the config by hand.")

    plan = plan_runtime(plane_w, plane_h, r_plane)
    total = sum(f["frames"] for f in files)
    out = {
        "data_dir": str(data_dir),
        "captures": len(caps),
        "total_frames": total,
        "kind": kinds.pop(),
        "plane_scale": scales.pop() if len(scales) == 1 else None,
        "plane": [plane_w, plane_h],
        "radius_plane_px": round(r_plane, 2),
        "runtime": plan,
        "files": files,
    }
    log(f"  -> {total} frames, plane {plane_w}x{plane_h}, "
        f"disc radius {r_plane:.1f} px")
    log(f"  -> drizzle x{plan['drizzle']}, {plan['workers']} workers "
        f"({plan['workers_by_cores']} cores, "
        f"{plan['workers_by_memory']} by memory at "
        f"{plan['bytes_per_worker'] / 1e9:.2f} GB each, "
        f"{plan['free_bytes'] / 1e9:.1f} GB free)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", default=None,
                    help="where to write survey.json (default: <data_dir>/../out)")
    ap.add_argument("--samples", type=int, default=9)
    args = ap.parse_args(argv)

    s = survey(args.data_dir, args.samples)
    out = args.out or os.path.join(os.path.dirname(str(args.data_dir).rstrip("/\\")),
                                   "out")
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "survey.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
