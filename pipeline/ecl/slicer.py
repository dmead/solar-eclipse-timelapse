"""Extract a contiguous frame range from a SER capture — port of ser-slice.js.

An eclipse SER holds several exposure states in one file: the filter comes off
mid-capture and the exposure is then ridden down by hand. The stacker consumes a
whole file, so the segments found by Stage A have to be split out first.

Unlike lunation's `io.ser.write_trimmed` this does NOT quality-rank or ROI-crop —
both of those assume a lit lunar disc and are wrong for a corona — and it keeps
the trailer, which `write_trimmed` discards. It is a straight byte-range copy
preserving colorId, bit depth and geometry (so CFA phase is trivially intact)
plus the matching slice of trailer timestamps.

Standard library only, and streamed in bounded chunks: these files are up to
22 GB and the output must be byte-identical to what ser-slice.js produced.
"""

import os
import struct

HEADER_BYTES = 178
CHUNK_TARGET = 64 * 1024 * 1024  # cap RAM per read regardless of frame size

__all__ = ["slice_ser"]


def slice_ser(src, dst, start, count, log=print):
    """Copy frames [start, start+count) of `src` into a new SER at `dst`.

    Returns a report dict. `count` is truncated (with a warning) when the range
    runs past the frames the file actually contains.
    """
    start, count = int(start), int(count)
    if start < 0 or count <= 0:
        raise ValueError(f"bad range: start={start} count={count}")

    with open(src, "rb") as f:
        header = bytearray(f.read(HEADER_BYTES))
        if len(header) < HEADER_BYTES:
            raise ValueError(f"{src}: truncated SER header")
        color_id, _little_endian, W, H, depth, frame_count = struct.unpack(
            "<6i", header[18:42])

        planes = 3 if color_id >= 100 else 1
        bpp = 2 if depth > 8 else 1
        frame_bytes = W * H * planes * bpp
        in_size = os.path.getsize(src)

        # A capture interrupted mid-write has a frameCount its data does not
        # back; trust the bytes on disk, not the header.
        usable = min(frame_count, (in_size - HEADER_BYTES) // frame_bytes)
        trailer_off = HEADER_BYTES + frame_count * frame_bytes
        has_trailer = in_size >= trailer_off + frame_count * 8

        log(f"slice {src}")
        log(f"  source {W}x{H} {depth}b colorId={color_id} "
            f"frames={frame_count} usable={usable} trailer={has_trailer}")

        if start >= usable:
            raise ValueError(f"start {start} beyond usable frame count {usable}")
        if start + count > usable:
            log(f"  WARNING: range truncated to usable frames "
                f"({count} -> {usable - start})")
            count = usable - start

        struct.pack_into("<i", header, 38, count)  # rewrite frameCount

        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        with open(dst, "wb") as out:
            out.write(header)

            chunk_frames = max(1, CHUNK_TARGET // frame_bytes)
            done = 0
            while done < count:
                k = min(chunk_frames, count - done)
                f.seek(HEADER_BYTES + (start + done) * frame_bytes)
                out.write(f.read(k * frame_bytes))
                done += k
                if done % (chunk_frames * 8) < chunk_frames:
                    log(f"  copied {done}/{count} frames")

            # The trailer is authoritative for frame times everywhere downstream,
            # so the matching slice of it is carried over.
            if has_trailer:
                f.seek(trailer_off + start * 8)
                out.write(f.read(count * 8))
            else:
                log("  note: source had no trailer — slice has none either")

    # The header's DateTime fields still describe the original capture start.
    # Deliberate: every consumer reads frame times from the trailer.
    out_size = HEADER_BYTES + count * frame_bytes + (count * 8 if has_trailer else 0)
    log(f"  wrote {count} frames [{start}..{start + count - 1}] "
        f"{out_size / 1e9:.2f} GB")
    return {"src": src, "dst": dst, "start": start, "count": count,
            "width": W, "height": H, "depth": depth, "colorId": color_id,
            "hasTrailer": has_trailer, "bytes": out_size}
