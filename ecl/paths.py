"""Default locations, when the caller did not say.

Every module in here used to carry one machine's drive letters as its argparse
defaults - `S:/solar-eclipse/out`, `S:/solar-eclipse/Sun`. That works exactly
once, on the machine it was written on, and it quietly outlives the layout: two
of those defaults pointed at a capture directory that had already moved to
another drive, and nothing said so because the supported entry point passes its
paths explicitly and never reads them.

So the defaults come from the environment, and fall back to the working
directory rather than to somebody's disk:

    ECLIPSE_OUT     where survey.json, configs/ and diag/ live   (else ./out)
    ECLIPSE_DATA    the captures                                 (else ./data)

`python -m ecl.run <data-dir>` is unaffected - it computes both from the data
directory it is given and passes them down. These are only for running a stage
by hand.
"""

import os

__all__ = ["ENV_OUT", "ENV_DATA", "out_dir", "data_dir", "in_out"]

ENV_OUT = "ECLIPSE_OUT"
ENV_DATA = "ECLIPSE_DATA"


def out_dir():
    """Output root: $ECLIPSE_OUT, else ./out."""
    return os.environ.get(ENV_OUT) or os.path.join(os.getcwd(), "out")


def data_dir():
    """Capture root: $ECLIPSE_DATA, else ./data."""
    return os.environ.get(ENV_DATA) or os.path.join(os.getcwd(), "data")


def in_out(*parts):
    """A path under the output root, resolved when it is asked for."""
    return os.path.join(out_dir(), *parts)
