# 2024-04-08 Total Solar Eclipse — Processing Pipeline

Processing for ~460 GB of SER captures of the 2024-04-08 total solar eclipse, shot
from the Port of Cleveland on Lake Erie with a ZWO ASI585MC.

`pix-planetary` is used as a library, by absolute path — `ser-stack.js`, its
`lib/fftalign.jsh`, `xisf-preview.js` and `pi-lock.mjs` are called where they sit
in `D:/projects/pix-planetary`, never copied. Improvements there land here.

## The data

| | |
|---|---|
| Camera | ZWO ASI585MC, 3840×2160 RAW16, Bayer RGGB, 2.9 µm |
| Capture | SharpCap 4.1, ~23.3 fps (USB-bandwidth limited at full res) |
| Optics | TS-Optics 70 mm APO + William Optics 0.8× reducer — **350 mm effective, f/5** |
| Mount | Sky-Watcher EQ6-R Pro |
| Image scale | **1.72 arcsec/sensor px** — solar disc 1116 px, i.e. `rSun = 279.03` superpixels |

The focal length was **measured before the optics were known**, and the two agree:
the fitted solar disc is 1116 px where a bare 440 mm scope would give 1408 px,
implying **349 mm** against the reducer's nominal 350 mm. Two independent facts
agreeing that closely is the strongest check available that the disc fits are right
in absolute terms and not merely self-consistent.

**Sampling.** The sensor is at about Nyquist for the seeing actually present —
1.72 arcsec/px critically samples a 3.4 arcsec FWHM, against 1.5-1.9 arcsec of
diffraction for a 60-80 mm aperture and 3-5 arcsec of realistic daytime lakefront
seeing. The 2×2 superpixel demosaic used by `pjsr/tl-frames.js` then halves that
again to 3.43 arcsec/px, which undersamples by 2× and is the single largest
resolution loss in the pipeline. Recovering it costs one render pass; drizzle would
buy less and cost far more (see `STATE.md`).
| `Sun/` | 22 files, 18:44:19 → 19:30:11 UTC, ~446 GB |
| `Capture/` | duplicates of the first two `Sun/` files, plus the only settings file |

**Filenames are UTC−5.** SharpCap's `TimeZone` was set to EST, not the EDT actually
in force. Add 1 h for Cleveland local time, 5 h for UTC. The SER trailer timestamps
are correct UTC and are what every stage actually uses.

**The SER endianness flag is ignored.** SharpCap writes `LittleEndian = 0` while
emitting little-endian data; honouring the flag scrambles every sample (median 735
becomes 32769). `ser-stack.js` takes the same position.

**Third contact was not captured.** SharpCap recorded 60 s then sat idle ~37 s
flushing 22 GB to disk. `14_16_14` ends 19:17:14 and `14_17_51` opens already
blown, so C3 fell in the gap. Second contact survived. About 96 s of the ~230 s of
totality is in dead gaps.

## Reference frames — the Moon is moving

The Moon crosses the corona at **~0.27 px/s** at this image scale. Nothing rigid
can register both: the corona is fixed to the Sun, the occulting limb to the Moon.

| Interval | Moon-vs-corona slip |
|---|---|
| One 60 s capture | ~16 px |
| Across the exposure ladder (~214 s) | ~58 px |

The solar radius is ~504 px, so the ladder slip is ~11% of a radius — enough to
double the streamers, or the limb, depending on which one the alignment happens to
lock onto. Worse, the lock is not consistent across levels: short exposures are
dominated by the sharp limb, long ones by the broad corona.

Handled in three places:

1. **Stack spans are capped** (`MAX_CORONA_STACK_S`, 15 s) so the residual inside
   any one stack is ~4 px rather than ~16.
2. **The rate is measured, not assumed** — `corona-drift.js` cross-correlates the
   outer-corona annulus between two same-level stacks with a long time baseline.
   It correlates the *radially flattened* image; without that the smooth falloff
   correlates with itself and every candidate shift scores the same.
3. **The output is defined as the Moon frame at one instant** — the shortest
   exposure's capture time, which is the level carrying the limb, chromosphere and
   prominences. Other levels are Moon-registered and then have the measured drift
   taken back out, so their *corona* lands where the corona was at that instant. A
   level whose Moon has moved more than a few pixels is excluded from inside
   1.25 R, where its displaced limb would print as a crescent.

Baily's beads are unaffected: they are single frames, never stacked.

## Stages

| Stage | Entry point | Output |
|---|---|---|
| A — measure & segment | `scan_ser.py` | `lightcurve.json`, `segments.json`, `segments.txt` |
| — plan | `scripts/gen-eclipse-config.mjs` | `configs/eclipse.json` |
| B/C — slice & stack | `scripts/run-eclipse.mjs` | `slices/*.ser`, `stacks/*.xisf` |
| D — corona | `scripts/run-corona.mjs` | `final/corona_hdr.xisf`, `final/corona_flat.xisf` |
| E — beads | `scripts/run-beads.mjs` | `beads/**` |
| F — timelapse | `gen_timelapse.py` + `scripts/run-timelapse.mjs` | `tl/seq_*.png`, `final/timelapse.mp4` |

### A — segmentation

A single eclipse SER holds several exposure states: the filter comes off mid-file
and the exposure is then ridden by hand. `scan_ser.py` samples every 20th frame
(24 rows each, ~180 KB instead of 16.6 MB) and finds the state changes.

Nothing is tuned to this dataset. Change points are steps in log₂(median) larger
than both a floor and several times the series' own typical step, so a uniform
capture yields exactly one segment. Boundaries are then refined to the exact frame
by binary search. It independently located the filter removal at frame **821 of
`14_13_00`, 19:13:35.2 UTC**, and found a five-level exposure ladder.

Totality is bracketed physically rather than by a brightness threshold: pulling the
filter with the exposure still set for the photosphere blows the frame out, and the
Sun reappearing at C3 blows it out again. Everything between the first and last
blowout is unfiltered.

### D — HDR corona

`corona-combine.js` assembles the per-channel stacks into linear colour and
measures the Moon (centroid of the brightest 1%, which is the near-circular inner
corona; limb radius from the peak of the radial profile).

`corona-hdr.js` puts the levels on a common scale measured from their own overlap,
blends so each pixel comes from the longest exposure that has not clipped there,
then divides out the radial brightness profile — the corona falls off close to
exponentially with radius, so no global stretch shows inner and outer detail at
once. A fraction of the profile is put back so the result still reads as a corona.

`HDRComposition` is deliberately not used; the merge is a handful of arithmetic
decisions and doing them explicitly keeps them inspectable.

### F — holding the Sun still

The first cut framed on the centroid of the lit region, which bounces: the
centroid of a crescent is not the centre of the disc it was cut from, it slides
further off as the Moon covers more of the Sun, and it jumps when the filter comes
off and the bright region becomes a corona.

`pjsr/tl-centres.js` finds the Sun's limb instead, by the same method
`analyzeDisk()` in pix-planetary's `gif-frames.js` uses for lunar phases — a Kasa
circle fit with a radius prior, iterated with a shrinking inlier tolerance, over
limb points from a row scan. A lunar terminator and an eclipsing Moon pose the
same problem: a circle constrained by a partial arc, with a second boundary trying
to pollute the fit. The prior is what rejects it. Their note that thin crescents
*need* the prior applies here verbatim.

Measured Sun limb radius: **558 px full-res**, independently consistent with the
550 px implied by the Moon radius from the corona work.

Two failures worth recording:

- **Bootstrapping from the bounding-box centre only works on a nearly full disc.**
  For anything else that seed sits *on* the crescent, about a radius from the true
  centre, so no limb point lands within tolerance and the fit dies on its first
  pass. The radius is therefore measured once on the least-eclipsed frame and held.
- **`|∇log|` explodes in empty sky.** The values there are noise about zero, and
  the log of a ratio of two small noisy numbers is large, so blank sky outscored
  the real limb: the totality search pinned itself against the frame edge and
  reported the Moon at x=1918 of 1920 when it was at x=755. That single bug made
  162 totality frames look like they had a Sun clipped by the sensor. Fixed by
  flooring at the frame's own median before the log, requiring nearly the whole
  ring to be inside the frame, and seeding from the dark-disc-in-bright-ring
  matched filter that `corona-combine.js` already uses.

`smooth_track.py` then models the track rather than following it. Within a capture
the mount drifts smoothly, so a robust line through the accepted fits describes it
to well under a pixel; between captures the mount was nudged by hand, so each file
is fitted independently. Segments are fitted per file **and per state** — one line
through both sides of the filter change extrapolated the totality fits back across
the crescent frames and produced a 1198 px drift across a 60 s capture. Segments
whose fitted drift rate exceeds 4x the median across all segments are treated as
bad fits and held constant.

**Totality is placed on a bare line, and nothing may perturb it.** The ring search
that finds the Moon during totality scatters ~7 px with a lag-1 autocorrelation of
only +0.2 to +0.5 — noise, not motion — so the residual is not tracked there and
the frames sit exactly on the fitted line. That makes the totality track
mathematically smooth, and it means any per-frame correction added afterwards can
only *break* smoothness rather than improve it.

That is what the per-level detection-bias correction was doing. It subtracts each
constant-exposure run's mean residual, and it earns its place where the model
FOLLOWS the detections — there the bias reaches the render as a nudge at every
exposure change. On a bare line there is no per-level offset to inherit, so all it
contributed was a step: 8.2 px going into the prominence level and 11.7 px inside
`14_14_36`, single frames, in a 1120 px window where the normal motion is 0.72 px
per frame. It is now gated on the residual actually being tracked, and totality's
worst step falls to 0.71 px.

The cost is absolute placement — each level's frames sit up to 8 px from where its
own detections put them. In a Sun-fixed window a viewer sees changes, not offsets,
and over three and a half minutes the line is the better estimate anyway.

### F — the zoomed panels

`gen_insets.py` places up to four magnified squares in the corners. Nothing is
hand-placed: every panel follows a feature computed per frame from geometry
already measured, and carries the name of what it follows — sunspot, lunar limb,
upper cusp, lower cusp, prominence.

The count is not fixed. A feature is emitted only while its subject exists, which
is the whole point of naming them: the sunspot passes behind the Moon, the cusps
exist only while the limbs actually intersect, and a totality exposure level may
show fewer than four prominences worth a box. Padding the list to four with
duplicates — the first version did — puts a labelled box on nothing.

Three things here are easy to get subtly wrong:

- **The Moon's centre is fitted from the terminator it casts**, not taken from
  `drift.json`. That file's sign convention is the shift needed to ALIGN two
  frames, not the direction the Moon travelled; using it put the Moon down-left
  when the picture plainly shows it up-right and threw every cusp into empty sky.
  Fitting the lit/unlit boundary needs no ephemeris and reports a residual (7.1 px).
- **Prominences are found in `R - k*G`, over an annulus reaching well INSIDE the
  Moon's radius.** Plain brightness finds the corona, which is white; subtracting
  a scaled G cancels it and leaves Halpha. And the inner bound must reach to about
  0.90 R, because prominences stand on the SUN's limb and the Sun is the smaller
  disc (1919" against 2010"). At 0.97 the search was discarding a prominence that
  was the third strongest feature in its frame.
- **They are detected per exposure LEVEL.** Totality was not shot at one exposure;
  a prominence that stands clear in a short frame is buried in coronal glare in a
  long one. One detection held across all of totality points panels at whatever
  was visible in a different exposure.

Corners are assigned by minimising total leader-line length, which also guarantees
the leaders never cross — a minimum-length matching cannot contain a crossing pair,
since swapping two crossing assignments is strictly shorter. A 9-frame hold stops
the layout flickering, but it is abandoned outright whenever the feature LIST
changes rather than moves: a held permutation maps slot number to corner, and each
level's prominences are re-ranked, so carrying one across a level boundary sends
the panels pointing across each other.

Labels are rasterised from a 5x7 bitmap font inside `tl-frames.js` — PJSR's
`Graphics`/`Bitmap` are not reachable from a script that works on plain
Float32Array planes. They sit outside the panel, on a plate that multiplies the
background rather than filling it, so it stays invisible against black sky and
still carries white text over the photosphere.

### Framing, and what it costs

Holding the Sun fixed costs field, because the Sun wandered 556 x 284 px over the
45 minutes. A window that never reaches past the sensor would be barely wider than
the disc. Totality is the exception: the Sun sat within x 713..876, y 437..483
then, so a window up to 1425 x 874 keeps every totality frame whole.

Square framing is far more efficient than 16:9 for a round subject — 1.29 solar
radii in every direction costs 9.7% of frames, where 16:9 gives the same vertical
field only by dropping 40%. Current output is **1440x1440, 1.29 R**, dropping 142
partial-phase frames (two whole captures, `14_05_08` and `14_27_34`, where the Sun
sat near a corner) and keeping totality complete at 162/162.

```bash
python smooth_track.py --window 720x720 --drop-padded     # half-res px
```

### Field rotation — measured? no. bounded? yes.

The scope was on a German equatorial mount that was not well polar aligned, which
rotates the field slowly. At roughly `misalignment x 15 deg/hr`, the 46-minute
sequence accumulates ~0.2 deg per degree of misalignment — **6 to 11 px at the
frame periphery** across the whole timelapse, and **under a pixel within totality**.
So the corona stacks, the HDR merge and the drift measurement are unaffected; only
the timelapse could show it.

Two approaches were considered and one was tried:

- **Sunspot correlation in polar coordinates — attempted, failed.** Two of three
  frame pairs had no usable overlap (frames either side of totality have nearly
  complementary uncovered regions), and the third returned -27 deg/hr at a
  correlation of 0.17, which is not physically possible for an equatorial mount.
  It is also confounded in principle: sunspots rotate with the Sun at ~0.55 deg/hr
  at the equator, the same order as the effect being measured.
- **Straightness of the Moon's path — rejected before implementing.** The apparent
  path is *not* straight at Cleveland's latitude. Earth's rotation carries the
  observer at 465 cos(41.5 deg) = 348 m/s, about 960 km over the sequence, giving a
  parallax swing of ~516 arcsec against the Moon's ~1400 arcsec of geocentric
  motion relative to the Sun. The observer's velocity vector also turns ~11.5 deg
  in that time, so the path curves — by roughly 15 px, against the ~6 px the field
  rotation would contribute. The systematic is larger than the signal.

What would work: an ephemeris residual for the exact site and times, or a joint fit
of rigid field rotation *and* solar differential rotation to several sunspots at
different disc positions, which have distinguishable displacement fields.

### F — exposure normalization

Normalization is per *segment*, not per frame. Normalizing every frame to constant
brightness would erase the eclipse — the sky genuinely darkens, and that is the
thing worth watching. Only the operator's own exposure changes are removed: a
stable segment is an interval of constant camera settings and gets one gain, while
a transition segment is the operator actively riding the exposure and gets a
per-frame gain that cancels the ride.

Gains chain across boundaries via background level, and **only** across boundaries.
Background level is not a pure exposure proxy — it also falls as less sunlight is
scattered — so it is trustworthy only over the fraction of a second spanning a
boundary. Comparing whole-segment averages mistakes the sky darkening for an
exposure change and drives the gain up until the crescent blows out.

Filtered and unfiltered runs are anchored independently. When the filter comes off
the subject changes from photosphere to corona and no gain relates the two; that
boundary is a hard cut, as it was in person.

### F — the prominence level is resampled, not repeated

One exposure level gets special treatment, chosen by measurement rather than by
name. Every stable totality segment is normalized to the same rendered brightness,
so the one needing the most gain is the one that was shot shortest — here 20x
clear of the rest, and it is the short-exposure prominence run right after second
contact. At the light curve's own cadence it was twelve video frames, 0.4 s, and
the best prominence footage in the set went past before there was anything to see.

It is resampled to one video frame per RAW frame — 211 unique frames, padded to
360 for 12 s of screen time. Two details make that work rather than merely last
longer:

- The run stops a full drizzle group short of the segment end, so every frame
  gets the same stack depth. Otherwise the last twenty frames get progressively
  shorter groups and the picture visibly grows noisier on the way out.
- Groups OVERLAP, sharing 19 of 20 raw frames. The ordinary rule sizes a group by
  the gap to the next video frame, which for dense frames is one — it would undo
  the drizzle exactly where it matters most. Drizzle never needed disjoint groups.

Roughly seven frames in ten are therefore shown twice, and it does not read as
judder because there is almost nothing to judder: **the Moon advances 0.006 px
between consecutive raw frames**, so neighbours differ only by seeing and noise,
and each is already a mean of twenty.

Anything measured over "neighbouring samples" has to use the ORIGINAL cadence once
this exists. Dense frames sit one raw frame apart instead of twenty, so their
frame-to-frame steps are twenty times smaller for no physical reason; letting them
into `smooth_track.py`'s follow heuristic dragged its median step under the
threshold and sent the whole of totality down the followed path, cutting 135 good
frames as "thrashing".

## Running it

```bash
NODE="C:/Users/dan/AppData/Local/Microsoft/WinGet/Packages/OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe/node-v24.18.1-win-x64/node.exe"

python scan_ser.py --data S:/solar-eclipse/Sun --out S:/solar-eclipse/out
"$NODE" scripts/gen-eclipse-config.mjs
"$NODE" scripts/run-eclipse.mjs            # slice + stack (hours)
"$NODE" scripts/run-corona.mjs             # drift, HDR, flatten
"$NODE" scripts/run-beads.mjs              # contact-window previews
python gen_timelapse.py                    # then centres, track, insets, render
"$NODE" scripts/run-centres.mjs            # required whenever frames were ADDED
python smooth_track.py --out S:/solar-eclipse/out --window 1120x840 --drop-padded --require-disc
python gen_insets.py --out S:/solar-eclipse/out
"$NODE" scripts/run-timelapse.mjs --workers 12
```

The timelapse is four passes that each rewrite `configs/timelapse.json` in place,
and they are strictly ordered. `smooth_track.py` joins detections on (file, index),
so a frame that `run-centres.mjs` never saw has no centre and gets dropped — it
warns when that happens.

`scan_ser.py` caches `lightcurve.json`; delete it to force a rescan. `run-eclipse.mjs`
leaves an existing slice alone when its size matches the requested frame range, so
re-running to redo one channel does not re-cut 22 GB off a slow drive.

## Environment

- **Node: never `node` from PATH** (v10.9.0, dies on the first `import` having run
  zero lines). Use the winget v24 binary by full path, above. The version folder
  moves with upgrades — glob `node-v24.*-win-x64`.
- **PixInsight** 1.9.4. PJSR needs `#engine v8`; under it `#include <pjsr/*.jsh>`
  is silently broken, so these scripts use no angle-bracket includes. A missing
  PJSR API kills a script with exit 0 and no output, which is why every script here
  creates its log file before doing anything else — "no log" means "died on line
  one", not "never ran".
- `console.writeln` is not captured in headless stdout. File logs are the record.
- **A regex ending in an escaped slash is a silent killer.** `path.replace( /^.*\//, "" )`
  ends `\/` immediately followed by the closing delimiter, so the source contains
  `//`. The PJSR preprocessor strips that as a line comment, truncating the line
  and leaving an unterminated expression — the script then dies at load with
  exit 0 and no log, indistinguishable from one that ran and did nothing. Use
  `p.substring( p.lastIndexOf( "/" ) + 1 )` instead. This cost two scripts and a
  long bisect; `pjsr/check-one.js` here does not catch it, because `eval` never
  runs the preprocessor.
- `File.readTextFile` does not exist (despite appearing in pix-planetary's
  `check-compile.js`). Use `File.readFile( p ).utf8ToString()`, as `ser-stack.js`
  does. Same silent-death signature.
- `pjsr/check-one.js` reports load errors for a single script:
  `-r="…/check-one.js,<script>,<report>"`. It catches ordinary syntax errors, not
  preprocessor damage — for that, bisect by building the file up function by
  function until the log stops appearing.
- **All PixInsight launches go through `withPiLaunchLock()`**, shared with
  pix-planetary. Two instances starting in the same second race the same instance
  slot and hang at 0 CPU.
- **`D:` was at 0 bytes free** during development, which broke tooling that writes
  temp files there. Set `CLAUDE_CODE_TMPDIR` somewhere with room. The shared launch
  lock still lives on `D:` (`D:/Temp/pi-launch.lock`) and needs a few spare bytes.
- Python is stdlib-only here — no numpy required.
- Scratch, slices, stacks and PI swap all live on `S:`. `pix-planetary` lists `S:`
  as a slow drive; this was a deliberate choice because `D:` is full. Expect slower
  stacking I/O than a lunar session.

## How the registration was made to work

Fitting the Moon independently in each level does not survive a 60x exposure
range. The radius was always excellent - 565.8 px on every level, to a tenth of a
pixel - but the centre was not. Tracking it through totality gave apparent frame
motion of 66, 3.2, 12, 2.7 and 0.82 px/s on successive levels, which the Moon
cannot do, and one level landed 838 px out. The merged image had two Moons.

Three changes fixed it:

1. **`corona-register.js` matches levels against each other** instead of
   differencing two independent fits. Both frames are reduced to the gradient
   magnitude of log brightness, which makes a 60x exposure difference comparable,
   and matched by normalized cross-correlation over a coarse-to-fine pyramid.
2. **`corona-combine.js` scores candidate centres** rather than least-squares
   fitting a circle to threshold crossings. Crossings are biased asymmetrically by
   prominences, a lopsided inner corona and saturation, which moves the centre
   while leaving the radius about right; a global ring-response search can only
   fail to improve, not be dragged.
3. **`corona-drift.js` got a tighter annulus and a physics gate.** Reaching out to
   3.2 R it measured 0.81 px/s - three times the ceiling - because that far out the
   corona is faint and the correlation locked onto the sensor's flat field, which
   does not move and so reported the Moon's whole frame motion as differential.
   Pulled in to [1.1, 2.0] R it measures **0.279 px/s**, against 0.29 px/s
   predicted from the lunar synodic rate at the fitted Moon radius - agreement to
   4%, from the data alone. Anything above the ceiling is now rejected and the
   caller falls back to rigid registration.

The exposure ladder also came out monotonic for the first time once registration
was right (12.9x, 1.48x, 2.10x, 2.31x between successive levels); before, one
level measured as *darker* than the one below it, because the overlap being
compared was misaligned.

A useful thing this surfaced: **the mount was nudged between captures.** Frame
motion is 1.85, 2.74 and 0.82 px/s across successive file boundaries, which is not
physical for a tracked mount - but the differential rate is constant, exactly as it
should be, because mount motion affects Moon and corona equally and cancels. Only
correlation-measured, per-pair registration copes with this; anything assuming a
constant frame rate of motion would not.

### The second dark lobe, and why a segment can be "stable" and still wrong

The first clean merge still had a large dark lobe beside the Moon, about the same
apparent size as the Moon itself. It was not an optical ghost - a single raw frame
from mid-totality is clean - it was **`14_13_00` f931+120**, which spans
19:13:40-19:13:45 UTC. **Second contact was ~19:13:46.** That segment predates
totality: the photosphere was still visible, and the level image is a huge blown
blob sitting off to one side of the Moon.

Stage A had labelled it `stable`, correctly by its own definition - the brightness
plateaued for five seconds while the exposure was being ridden down. Constant
brightness is not the same as usable, so the config gate now also tests **saturated
fraction**. During totality only the chromosphere and prominences clip, a thin ring
at the limb: at this Moon radius a generous 20 px annulus is ~0.9% of the frame.
The four good levels measure 0.004-0.41%; the two rejected ones measure 8.7% and
17.3%. `14_14_36` f133+82 fails the same test for the opposite reason - not
pre-contact, just exposed long enough to clip the entire inner corona, which is
also why it had no limb left to register on (NCC 0.399).

Dropping both left every remaining level correlating at NCC 0.62-0.72.

### Output is trimmed to common coverage

Levels are shifted by up to 227 px to register, so each runs off the sensor on one
side. The boundary where a level stops contributing is a step in the merge - the
pixels beyond it are built from fewer levels - and it printed as a bright band down
two edges. The merge now trims to the rectangle every level covers (here
3631x2037, losing 209x123 px) and the radial profile is measured on the trimmed
frame so it never integrates across the seam.

Two smaller fixes in the same area: `translateImage` leaves zero outside the source
instead of replicating the edge pixel (a replicated bright border passed the
signal test and got blended in as real data), and the radial profile is now held
constant beyond the last radius with at least 2000 contributing pixels, since
radii that only clip a frame corner have no meaningful median.

### Still open

The flattened corona is geometrically correct but tonally flat - it needs a real
stretch (asinh or a strong MTF) before it is presentable. The chromosphere sits
near 1.0 and the outer corona near 0.001, so the preview's rescale-and-gamma
crushes everything interesting. That is tone mapping, not registration.

Everything upstream is verified: segmentation, slicing (byte-identical to source),
stacking (sharp limb, ~2% sigma rejection), and colour (prominences render red, so
CFA phase is right).

## Possible improvement

Chunking a long segment currently re-slices it from the raw capture. An optional
`firstFrame`/`frameCount` in `ser-stack.js` would let chunks be stacked straight
out of one slice and would save a re-read per chunk — but it means editing shared
pix-planetary code, so it was left alone.
