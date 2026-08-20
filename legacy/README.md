# The PixInsight and Node implementation

This is the original pipeline that produced the 2024-04-08 results, kept on this
branch and deliberately not on `main`. **It is a record. Do not try to run it.**

It needs, and none of these are on `main` any more:

- **PixInsight 1.9.4**, at `C:/Program Files/PixInsight/bin/PixInsight.exe`, plus
  a licence for it.
- **`pix-planetary`**, a second repository, at the literal path
  `D:/projects/pix-planetary`. `ser-stack.js`, `lib/fftalign.jsh`,
  `xisf-preview.js` and `pi-lock.mjs` are called where they sit there.
- **Node 24 by absolute path.** `node` from PATH was v10 on that machine and
  died on the first `import` having run zero lines.
- One machine's drive letters, hardcoded: `S:/solar-eclipse/out`,
  `S:/solar-eclipse/swap`, `Z:/eclipse-work`, `D:/Temp/pi-launch.lock`.

## Why it was replaced

Not because it was wrong — it was not, and every measurement it made is written
up in [`../docs/NOTES.md`](../docs/NOTES.md). It was replaced because none of the
four requirements above can be met by anybody else, which made the project
impossible to hand over or even to reinstall. The Python pipeline on `main` is a
port of this one, depends on nothing but PyPI, and carries the vendored numerics
in `ecl/vendor/` for exactly that reason.

## What is here

| | |
|---|---|
| `scripts/*.mjs` | the Node drivers — slice, stack, corona, beads, timelapse |
| `pjsr/*.js` | the PixInsight-side scripts each driver launches |
| `render_detached.ps1` | ran a long render outside the session that started it |
| `ser_to_fits.py` | one-off SER-to-FITS export, for looking at frames elsewhere |
| `centroid_rescue.py` | one-off re-measure of centroids on a single capture |

## Where each part went

| here | on `main` |
|---|---|
| `scan_ser.py` (was at the repo root) | `ecl/segment.py`, rewritten to read any format |
| `pjsr/ser-slice.js`, `ser-frames.js` | `ecl/slicer.py`, `ecl/serio.py` |
| `pjsr/corona-*.js` | `ecl/corona_*.py` |
| `pjsr/tl-centres.js` | `ecl/tl_centres.py` |
| `pjsr/tl-frames.js` | `ecl/tl_render.py`, `ecl/font5x7.py`, `ecl/demosaic.py` |
| `pjsr/tl-corona-track.js` | `ecl/tl_track.py` |
| `scripts/run-*.mjs` | `ecl/run.py`, one pass per module |
| `scripts/gen-eclipse-config.mjs` | `ecl/params.py`, `ecl/survey.py` |
| `scripts/encode-deliverables.mjs` | `ecl/encode.py` |

The PJSR hazards these scripts were written around — a regex ending in an escaped
slash truncating a line at the preprocessor, `File.readTextFile` not existing,
scripts dying at load with exit 0 and no output, the shared launch lock — are
documented in [`../docs/NOTES.md`](../docs/NOTES.md) under Environment. They cost
real days and are worth reading before touching any PJSR, here or elsewhere.
