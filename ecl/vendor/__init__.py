"""Vendored numerics from `lunation`, so this project has no sibling-repo dependency.

WHAT THIS IS. `lunation` is a lunar-imaging pipeline that already contained the
PixInsight-free numeric spine this project needed: FFT phase correlation,
sub-pixel warp, kappa-sigma frame combination, the drizzle stacker, SER memmap
reads and XISF I/O. Depending on it as an editable local checkout worked on the
machine it was written on and nowhere else - a clone of this repository would
install from PyPI and then fail on the first import.

These eleven files are copied unmodified, in lunation's own package layout. That
is deliberate on both counts:

  - UNMODIFIED means the vendored copy can be diffed against upstream to see
    exactly what has drifted. Once you start "just tidying" vendored code you no
    longer know whether a difference is a fix or an accident.
  - THE LAYOUT IS PRESERVED because every internal import in these files is
    relative (`from ..core.fftreg import PhaseCorrelator`). Mirroring
    core/io/stack/finish means not one line inside them needs rewriting, so
    re-syncing is a file copy rather than a patch.

Only what is actually imported is here. lunation is much larger; this is the
transitive closure of what `ecl` uses, nothing else.

    core/warp        resample, translate - the drizzle kernel
    core/kernels     Laplacian sharpness, gradient magnitude
    core/fftreg      PhaseCorrelator - both the ported and skimage engines
    core/framecube   kappa-sigma rejection and combination
    io/ser           SerReader (memmap), CFA_LAYOUT
    io/images        read_image, write_png, write_tiff32
    io/xisf_io       read_xisf, write_xisf
    finish/primitives  CIELab conversions for the corona stretch
    stack/logutil    job logging used by the stacker
    stack/localwarp  the field-warp aligner
    stack/stacker    the drizzle stacker

PROVENANCE: lunation @ baf089b, 2026-07-17, plus that tree's uncommitted
2026-07-24 drizzle changes - `drizzleEngine="ported"` by default, adopted after
the STScI square kernel at pixfrac 1 measured 2.6x more Nyquist-band grid noise.
Anyone re-syncing from a clean upstream checkout will NOT get those changes and
will silently take older numerics; check `drizzleEngine` before trusting a
refresh.

Everything these files import - numpy, scipy, opencv, scikit-image, tifffile,
xisf - is on PyPI and declared in pyproject.toml.

LICENCE. lunation and this project have the same author and copyright holder,
and both are GPL-3.0-or-later. So these files carry the licence of the tree they
sit in and there is no third-party term to honour here - but that is a fact
about who wrote them, not a general property of vendored code. Anything copied
in later from someone else needs its own licence recorded in this docstring and
its notice kept alongside the file.

Copyright (C) 2024-2026 Daniel Mead. See the LICENSE file at the repository
root, or <https://www.gnu.org/licenses/>.
"""
