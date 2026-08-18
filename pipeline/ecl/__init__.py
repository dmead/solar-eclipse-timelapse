"""Solar-eclipse processing — the PixInsight-free implementation.

The numeric spine is vendored in `ecl.vendor` — the parts of the lunation
project this pipeline actually uses, so a checkout installs from PyPI alone.
This package holds what is specific to a total solar eclipse and has no
lunar analogue: SER trailer timestamps, full-resolution demosaic, exposure-bracket HDR, radial corona
flattening, and the Sun-fixed timelapse.
"""

__all__ = ["demosaic", "imgio", "serio", "slicer"]
