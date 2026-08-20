"""Solar-eclipse processing — the PixInsight-free implementation.

The numeric spine is vendored in `ecl.vendor` — the parts of the lunation
project this pipeline actually uses, so a checkout installs from PyPI alone.
This package holds what is specific to a total solar eclipse and has no lunar
analogue: SER trailer timestamps, full-resolution demosaic, exposure-bracket
HDR, radial corona flattening, and the Sun-fixed timelapse.

Copyright (C) 2024-2026 Daniel Mead.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for details: <https://www.gnu.org/licenses/>.
"""

__version__ = "0.1.0"

__all__ = ["demosaic", "imgio", "run", "segment", "serio", "slicer", "source"]
