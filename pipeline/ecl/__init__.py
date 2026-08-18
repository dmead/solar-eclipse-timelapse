"""Solar-eclipse processing — the PixInsight-free implementation.

The numeric spine is lunation (`lunation.core`, `lunation.io`, `lunation.stack`),
the validated Python port of pix-planetary's PJSR. This package holds what is
specific to a total solar eclipse and has no lunar analogue: SER trailer
timestamps, full-resolution demosaic, exposure-bracket HDR, radial corona
flattening, and the Sun-fixed timelapse.
"""

__all__ = ["demosaic", "imgio", "serio", "slicer"]
