"""Export individual SER frames as colour images — port of ser-frames.js.

Baily's beads and the diamond ring evolve in well under a second, so unlike every
other product here they must NOT be stacked: stacking averages the beads into a
smooth arc. This pulls single frames out instead.

Demosaic is bilinear at full resolution rather than 2x2 superpixel — beads are
small high-contrast features and halving the resolution to avoid interpolation
costs more than the interpolation does.

Two modes:
  "preview"  downsampled autostretched 8-bit PNGs, for choosing frames by eye
  "tif"      full-resolution 16-bit TIFFs of the frames worth keeping

The preview stretch is triage only and never a deliverable.
"""

import argparse
import os
import time

import numpy as np
from .vendor.core.warp import resample

from .demosaic import bilinear_rggb
from .imgio import write_png
from .serio import EclipseSer

PREVIEW_WIDTH = 960
PREVIEW_GAMMA = 0.45

__all__ = ["export_frames"]


def _write_tiff16(path, rgb):
    """16-bit RGB TIFF. lunation's write_tiff32 is float32, which is right for an
    intermediate but not for a frame meant to be opened in an editor."""
    import tifffile

    a = np.clip(rgb, 0.0, 1.0)
    tifffile.imwrite(path, (a * 65535.0 + 0.5).astype(np.uint16), photometric="rgb")


def export_frames(src, start, count, stride=1, out_dir=".", prefix="frame",
                  mode="preview", log=print):
    """Export frames [start, start+count) of `src`, every `stride`-th one."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    written = []

    with EclipseSer(src) as ser:
        if not ser.bayer:
            raise ValueError(f"{src}: colorId {ser.color_id} is not a Bayer layout")
        count = min(count, ser.frame_count - start)
        log(f"frames {src}")
        log(f"  {ser.raw_width}x{ser.raw_height} colorId={ser.color_id} -> {mode}, "
            f"frames [{start}..{start + count - 1}] stride {stride}")

        for k in range(0, count, stride):
            fi = start + k
            rgb = bilinear_rggb(ser.raw(fi), max_value=ser.max_value,
                                color_id=ser.color_id)
            rgb = np.moveaxis(rgb, 0, -1)          # (3,H,W) -> (H,W,3)

            if mode == "preview":
                # Autostretch for triage: rescale to full range, lift with a
                # gamma, then downsample.
                lo, hi = float(rgb.min()), float(rgb.max())
                if hi > lo:
                    rgb = (rgb - lo) / (hi - lo)
                rgb = rgb ** np.float32(PREVIEW_GAMMA)
                rgb = resample(rgb, PREVIEW_WIDTH / ser.raw_width, "mitchell")
                path = f"{out_dir}/{prefix}_{fi:05d}.png"
                write_png(path, np.clip(rgb, 0.0, 1.0), bit_depth=8)
            else:
                path = f"{out_dir}/{prefix}_{fi:05d}.tif"
                _write_tiff16(path, rgb)

            written.append(path)
            if len(written) % 20 == 0:
                log(f"  wrote {len(written)} frames")

    log(f"  {len(written)} frames -> {out_dir} in {time.time() - t0:.0f} s")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="frame")
    ap.add_argument("--mode", default="preview", choices=["preview", "tif"])
    args = ap.parse_args(argv)

    export_frames(args.src, args.start, args.count, args.stride,
                  args.out_dir, args.prefix, args.mode)


if __name__ == "__main__":
    main()
