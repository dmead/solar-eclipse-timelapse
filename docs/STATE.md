# Engineering log — read this first when resuming

*State as of 2026-08-20. Entries below are in reverse chronological order and
are kept as written; where one names a file that has since moved, the rebuild
commands at the end of this document are the current truth.*

## 2026-08-20 — drizzle 1 against drizzle 2, measured on the features

Now that `render.drizzle` reaches the workers it can be varied, so the claim in
NOTES — that the 2x2 superpixel demosaic undersamples by 2x and "recovering it
costs one render pass" — is finally testable. Same published config, insets
re-planned at each drizzle (panel 225 px at 1, 450 at 2), then rendered with the
panels stripped, because they are drawn content at a different size in each
render and any sharpness metric over them measures boxes and label plates.

**Drizzle 2 does recover detail, and it is visible on the features.** Measured as
the share of power above the drizzle-1 Nyquist, drizzle 2 native against drizzle
1 upscaled with Lanczos — a resampler cannot invent signal above the source
Nyquist, so anything up there is real:

| patch | 25-50% | 50-75% | 75-100% | above Nyquist |
|---|---|---|---|---|
| prominence, 128 px | 1.18x | 1.60x | **2.14x** | 1.81x |
| sunspot, 128 px | 1.10x | 1.29x | **1.41x** | 1.34x |

The monotonic rise with frequency is the part that matters. Noise would sit flat
across the bands and interpolation ringing would favour drizzle 1; an advantage
that grows as the structure gets finer is what recovered detail looks like. At
4:1 the prominence knot is cleanly separated in the drizzled version and mushy in
the interpolated one.

**WHERE YOU MEASURE DECIDES THE ANSWER.** The first pass at this reported no
benefit, from 1024x1024 crops in which 99.999% of the power sat below a quarter
of Nyquist: the corona's smooth gradient swamped the very signal being looked
for, and the supra-Nyquist energy came out at 1e-6 of the total. Moving to
128 px patches on the sunspot and the prominences — the only genuinely fine
structure in this data, and already located per frame by the panel planner —
raised it a hundredfold and turned a null into a clear result. The features were
`insets` all along; nothing needed detecting.

At MATCHED resolution the two agree to 2% (mean |grad| on the limb annulus,
drizzle 2 area-averaged down to 1200x900), so drizzle is not degrading anything
either. The cost is 4x the pixels and about 1.7x the render time — 1.0 min
against 0.6 for 24 frames.

One caveat left standing: this is measured on 8-bit stretched output, so it is a
lower bound. Quantisation at 1/255 and the highlight shoulder both work against
fine low-contrast detail, and the sampling argument in NOTES is about linear data
at the sensor. The real margin is at least this large.

Three failed measurements preceded the working one, all for the same reason: they
depended on locating the limb, and a disc centre estimated from a rendered frame
disagreed by 21 px between the two renders — enough to smear an edge metric into
a 93 px "edge width" and then a 0.1 px one. The frequency-domain test needs no
centre.

## 2026-08-20 — full re-run on the real data, against the published one

The whole pipeline re-run from zero on the 449 GB, into a fresh `out-verify` so
the state behind the published video was untouched. What it settles:

**The rewrite is faithful.** `ecl.segment` — new reader, new statistic, new
units — reproduces `scan_ser.py` exactly: 22 captures, 42 segments, every
boundary, state, kind and exposure level identical, and it re-found the filter
removal at frame 821 of `14_13_00` and the five-level ladder on its own. Medians
differ only by the unit change (490 ADU against 0.0074769, and 490/65535 =
0.0074769). `tl_centres` measured the sun limb at 279.0 px plane / 558 px
sensor, the same number recorded above.

**The drizzle bug was real and cost this data nothing.** Both halves are
measured, not argued:

- Real: the same config asking for `drizzle = 1` rendered 2400x1800 before the
  fix and 1200x900 after it.
- Harmless here: holding the PUBLISHED config fixed and re-rendering 48 frames
  with the current code — spans at stack depth 3 and 20, inside totality where
  the group-shift bound applies — gives **48 of 48 byte-identical** frames. The
  one input that actually differed, `MAX_GROUP_SHIFT_PX` 4.0 against 4.172,
  rejects nothing at either value on groups measured at ~1 px.

So no documentation image needed regenerating: `docs/media/preview.gif`
re-encoded from the published frames comes back with the same md5.

Holding the config fixed is the only way to attribute a difference to the code.
A plain re-run does NOT reproduce the published video, and should not be read as
a regression when it does not: `ecl.run` passes no framing flags, so the window
is chosen automatically at 1180x880 keeping 2299 frames, where the published cut
used `--window 1120x840 --drop-padded --require-disc` and kept 2228. 27.8% of
frames then reach past a sensor edge — by at most 24 px of 2360, onto sky that
renders as exact zero either side of the boundary, so it is invisible. The
auto window shrinks precisely to keep that true.

Also confirmed on the way: `--resume` picked up a render stopped at frame 1800
and finished the remaining 499 with no gap and no seam.

## 2026-08-20 — the render pool never read the config

`tune()` writes seventeen module globals, and whether a worker sees them depends
entirely on the pool's start method: only `fork` inherits module state.

| platform | start method | inherits tuned globals |
|---|---|---|
| Windows, macOS | `spawn` | no |
| Linux, Python <= 3.13 | `fork` | yes |
| Linux, Python >= 3.14 | `forkserver` | **no** |

So it was a Windows-only bug when it was written and is a Linux one now too:
3.14 moved Unix off `fork`, and this project supports 3.11 upward. Every worker
re-imported `tl_render` and got the built-in defaults. The
parent tuned itself, logged the tuned values, and then handed all the rendering
to processes that had never read `eclipse.toml`. `--workers 1` renders in-process
and always behaved, which is why a knob that appeared to do nothing at 24
workers would start working the moment you tried to reproduce it serially.

**Cost to the 2024-04-08 render: essentially nil, by luck.** Measured by diffing
a fresh import against a tuned one, exactly one setting differed —
`MAX_GROUP_SHIFT_PX`, 4.0 px used against 4.172 asked for. The groups run at
~1 px with zero rejections against a 4 px bound, so neither number rejects
anything. Everything else matched because this data's survey picks drizzle 2 and
the module default IS 2.

**Cost to anyone else: real.** A well-sampled disc is the documented case where
the survey drops `render.drizzle` to 1; those runs rendered at 2 regardless —
four times the pixels and four times the memory per worker, against a survey
line saying it had chosen 1. And every setting in `[render]`, plus the panel
exposure settings, did nothing at all.

Found by setting `drizzle = 1` for the alignment comparison and getting
2400x1800 frames out of it. Confirmed after the fix: the same config renders
1200x900.

Verified on Ubuntu 24.04 the same day — a clean clone, `pip install .`, the
synthetic fixture and the whole pipeline to an mp4 in 44 s, 49 tests green. The
fixture is byte-identical across the two platforms (same md5) and segmentation
agrees to five decimals, so a cross-platform difference is a finding rather than
expected drift.

`tuned_state()` / `apply_tuned()` now pass the parent's state through the pool
initializer. `tests/test_worker_state.py` reads the `global` declarations out of
`tune()` and fails if `TUNED` does not name the same set, so adding a setting
and forgetting the workers breaks the suite instead of silently disabling it.

## Honest assessment of the timelapse

**No PIPP-style per-clip stabilization was ever done.** The timelapse centres each
frame on a per-frame geometric fit of the Sun's limb, smoothed into a track
(shared drift rate, per-capture intercept, smoothed residual, 1 px/frame slew
limit). There is no frame-to-frame registration anywhere in this path.

That distinction is not pedantry. PIPP-style object centring, and whole-frame
cross-correlation, both lock onto the brightest content — which during the partials
is the crescent. The crescent's centroid is not the Sun's centre and slides around
the limb as the Moon advances, about 750 px over 45 minutes. Either technique would
stabilize the wrong thing. That was the original bouncing bug.

**Fixed this session:** the disc fit was fed limb points from a row scan that
captured the Sun's limb *and* the Moon's limb. On a thin crescent the two arcs sit
a few pixels apart, inside the inlier tolerance, so the radius prior could not
separate them — and the resulting bias depended on crescent geometry, so it was
different in every capture. Measured in the rendered frames, the Sun sat up to
76 px off centre and the offset jumped at every capture boundary.
`sunLimbPoints()` in `pjsr/tl-centres.js` now marches outward along rays and takes
the OUTERMOST falling edge in log brightness: past the Sun's limb there is only
sky, so that edge is the photosphere and nothing else.

**Fixed, second pass — fixed-radius centre fit.** A three-parameter circle fit
needs a good spread of arc to pin the centre; as the crescent thins, the usable
arc drops well under half the limb and radius trades off against centre along the
direction perpendicular to it. The crescent's orientation differs per capture, so
the bias did too — that was the 10–30 px step at capture boundaries.
`fixedRadiusFit()` holds the radius at the measured 279 px (half-res, 558 full)
and solves only the centre: each limb point says "the centre lies R inward along
my own ray", and the answer is their mean. Two unknowns, no degenerate direction.

Measured independently in the rendered frames:

| | before | after |
|---|---|---|
| Sun centre swing, x | 173 px | 18.5 px |
| Sun centre swing, y | 178 px | 10.1 px |

**All twelve captures from `14_08_22` onward sit within ±0.6 px of frame centre.**
The residual is confined to the six earliest captures (~8–10 px in x, consistent
sign), where the disc is fullest and most saturated.

**Caveat on that residual — it may be a measurement artifact, not real placement
error.** The validator reads the rendered 8-bit PNG *after* the 0.65 gamma, while
the detector works on half-res linear data. Gamma moves where the "falling edge"
sits relative to the true limb, and the shift depends on saturation. The bias
appearing only on the brightest discs fits that explanation. Settling it needs a
validator that reads the linear SER directly — do that before chasing it further.

## The scope was touched at the filter change

Removing the solar film both shifts the pointing and leaves the mount ringing, and
the frames during the settle are the worst case for the detector: the exposure is
still ramping, so they are saturated *and* moving. In `14_13_00` the first two
unfiltered detections land 148 px and 11 px from where the capture settles.
Smoothing cannot rescue that — it just spreads a wrong position over the
neighbours — and a dissolve cannot either, because the frame underneath is
genuinely in the wrong place.

`smooth_track.py` now drops any frame whose own detection disagrees with the
settled model by more than **6× that capture's own median scatter**, floored at
8 px. Adaptive because the two detectors have very different precision: the circle
fit on a filtered disc is good to about a pixel, the ring search on the corona
scatters several. A flat 6 px tolerance threw away 89 good totality frames; the
adaptive one keeps totality complete at 147/147 and drops 116 frames concentrated
exactly where the scope was disturbed — `14_13_00` ×7 (film off), `14_17_51` ×9
(film back on), plus `14_05_08` and `14_11_23`, two captures with already-known
fit failures.

Largest remaining within-capture step: **4.81 px**.

## Wobble through totality — following noise instead of motion

The residual tracker exists because the mount genuinely wobbles and a straight
line cannot follow it. But during totality the detector switches from a circle fit
on the photosphere to a ring search on the corona, and that is far less precise:
**7 px RMS against ~1 px**, with lag-1 autocorrelation of only +0.2 to +0.5
against +0.6 to +0.75 on filtered captures. Low correlation at high amplitude
means noise, not motion — and the tracker was following it, running at its slew
limit the entire time just to keep up. That was the totality wobble.

Residual tracking is now gated on lag-1 (`MIN_LAG1 = 0.55`). Where a series looks
like noise, the straight drift line is used instead: following noise is strictly
worse than ignoring it. Totality is only 4.9 s of video, over which real mount
wobble is sub-pixel anyway.

| totality model step | before | after |
|---|---|---|
| median | 1.48 px | 0.65 px |
| p90 | 1.63 px | 1.63 px |
| max | 38.72 px | 5.47 px |

The gate is data-driven, not a special case for totality — any capture whose
residual is dominated by noise gets the same treatment.

## Highlight protection on the filtered frames

The gain maps each segment's p99 to a fixed target, but p99 is not the peak: on a
nearly full disc the photosphere covers enough of the frame that p99 sits INSIDE
it, so the brighter centre landed above 1.0 and hard-clipped. **13% of the first
frame was pinned at 254+** while the source frames clip nothing - the flat white
disc was manufactured in the renderer, not captured.

`cropGain()` now applies a tanh shoulder: linear below `SHOULDER_KNEE` (0.60),
compressing asymptotically above it toward `SHOULDER_CEIL` (0.965). The ceiling
being below 1.0 is the important part - a first attempt asymptoting TO white still
rendered 255 for anything far enough over the knee, and left 9% of the fullest
disc pinned.

| frame | >=254 before | after | peak |
|---|---|---|---|
| seq 0 (13_44_19) | 13.150% | 0.000% | 249 |
| seq 120 (13_58_40) | 2.254% | 0.000% | 247 |
| seq 500 (14_11_23) | 0.791% | 0.000% | 249 |

Nothing clips anywhere now, and limb darkening plus the sunspots near disc centre
keep their gradient instead of being flattened into a plate.

## Corona tone mapping — `pjsr/corona-stretch.js` (new)

Produces `out/final/corona_final.xisf`. Four steps: sky-pedestal subtraction from
an annulus at 3.2–3.9 R, white balance by equalising channel medians over real
corona at 1.15–2.10 R, a luminance-driven asinh stretch (one gain for all three
channels, so colour ratios survive where a per-channel curve would bleach the
inner corona), the Moon's interior taken to black with a feathered edge, then
two-scale local contrast.

**Input must be `corona_hdr.xisf`, NOT `corona_flat.xisf`.** I tried the flattened
input twice and both times got a false-colour disaster — the flatten already
divides out the radial falloff, so the outer field sits at the same level as the
inner corona and asinh saturates all of it, including the residual inside the
Moon. Flatten and stretch are two answers to the same problem; they do not stack.
Structure is instead raised by local contrast at r=110 (streamers) and r=20 (polar
plumes), which lifts detail without altering the brightness envelope.

`pjsr/xisf-export.js` (new) writes a PNG with no tone changes — needed because
pix-planetary's `xisf-preview.js` rescales and applies its own 0.6 gamma, which
re-stretches an already-stretched image and shows you something the file does not
contain.

**Still not good enough.** The current result has a correct black Moon, neutral
colour, and faint streamers visible upper-right and lower-left, but it is hazy and
the streamers are weak. Two known faults:

- A green ring hugs the limb, from the chromosphere's channel balance in the
  shortest exposure. The white balance is measured further out at 1.15–2.10 R and
  does not correct it.
- Overall it still reads flat. The likely cause is that the inner corona sits at
  `INNER_TARGET` = 0.72 and the local contrast then lifts the mid-field toward it,
  compressing the difference. Worth trying a lower `INNER_TARGET` (~0.55) with a
  higher `LC2_AMOUNT`, or a third contrast scale around r=300.

Tunables at the top of the file: `ASINH_BETA` (120), `INNER_TARGET` (0.72),
`LC_RADIUS`/`LC_AMOUNT` (20/0.30), `LC2_RADIUS`/`LC2_AMOUNT` (110/0.85).

Note `PROFILE_RESTORE` in `corona-hdr.js` was changed 0.22 → 0.45 during this
work. That only affects `corona_flat.xisf`, which is now a diagnostic product
rather than an input to anything.

```bash
PI="C:/Program Files/PixInsight/bin/PixInsight.exe"
"$PI" -n --automation-mode --no-splash -r="S:/solar-eclipse/pipeline/pjsr/corona-stretch.js,S:/solar-eclipse/out/final/corona_hdr.xisf,S:/solar-eclipse/out/final/corona_final.xisf,S:/solar-eclipse/out/final/corona_flat_moon.json,S:/solar-eclipse/out/logs/corona_stretch.log" --force-exit
```

## Umbra trial (2026-08-13)

Cloned to `D:/projects/umbra`, venv at `D:/projects/umbra/venv` built on the uv
Python 3.12.13 (`D:/uv-pythons/cpython-3.12.13-windows-x86_64-none`) — the pinned
deps have no 3.14 wheels, and 3.14 is the system default. Config at
`D:/projects/umbra/config.yaml`. Ran registration → integration → hdr end to end.

Feed: `pipeline/ser_to_fits.py` exports totality frames as full-res bilinear-
debayered FITS normalised to [0,1] (umbra's HDR thresholds are in that range;
raw ADU would put every pixel above the high threshold). Grouped by a `LEVEL`
keyword rather than `EXPTIME`, because the totality exposure times were never
recorded and the synthetic EXPTIME proxy differs slightly between two segments
shot at the same setting, which would split a level in two.

**Umbra measured the field rotation I failed to get three times: 0.038 deg over
132 s = 1.04 deg/hr.** That sits exactly in the range predicted for a few degrees
of polar misalignment. Its sun-registration solves rotation and the Sun-Moon
offset jointly across anchor frames, which is why it succeeds where correlating
sunspots or the lunar path did not.

**Auto anchor selection failed and must be set explicitly.** With
`anchor_filenames: null` it converged on a Sun-Moon separation of ~960 px, which
is partial-eclipse geometry — during totality the discs are nearly concentric —
and produced a ghost disc plus a cone artifact. With the two L1 frames named
explicitly (same exposure level, 132 s apart, the widest same-setting baseline in
the data) the delta drops to a correct 16-51 px.

**It independently confirmed the L3 finding.** HDR composition failed with "the
ray at 95 deg holds no sample to fit the brightness on" until L3 was removed: its
stack is 25% blank, because it sits far enough in time from the anchors that the
interpolated shift pushes it off frame. L3 is the pre-second-contact segment this
pipeline already excludes (19:13:40-45, C2 ~19:13:46). Two independent methods now
agree it is not corona.

Current output `S:/solar-eclipse/umbra/hdr/hdr.fits` has correct geometry — one
Moon, concentric, no ghost — but two faults remain:

- The Moon's interior is a grey cone rather than black. The polar brightness
  equalization fits an affine relation that varies with angle around the Moon, and
  inside the disc it is extrapolating with nothing to fit against.
- A border of blank pixels (3.1%) from the registration shifts, needing a crop to
  common coverage as `corona-hdr.js` already does.

### Second run — true-totality frames only

`ser_to_fits.py` grew `--levels` and `--max-satfrac`. Re-ran with
`--levels L0,L1,L2 --max-satfrac 0.02 --per-segment 12` = 60 frames, all inside
the clean window between C2 (19:13:46) and C3 (~19:17:36), excluding L3
(pre-contact photosphere) and L4 (17.3% saturated, the exposure ramp).

Everything ran clean: registration 62/62, integration and HDR both exit 0, no
threshold widening needed. Anchor rotation **0.034–0.057 deg over 137 s**,
consistent with the first run's 1.04 deg/hr.

**Anchor filenames are frame-index dependent** — changing `--per-segment` renames
the files and registration dies with a `KeyError` on the missing anchor. Re-derive
them after any re-export (widest same-level time baseline; currently
`14_14_36_f00338` + `14_16_14_f01265`, 137.2 s apart).

**The cone inside the Moon persists** and is the blocking defect. It is not a
registration failure — the limb is sharp and concentric, and the corona outside is
clean and symmetric with streamers visible left and lower-right. It is the polar
brightness equalization extrapolating inside the disc, where there is no corona to
fit against. Options, untried: raise `low_threshold` so disc pixels are rejected
outright rather than fitted; or take Umbra's registered/integrated stacks and do
the merge and Moon blanking with `corona-hdr.js` + `corona-stretch.js`, which
already handle the disc correctly.

That last option is the recommendation: **Umbra's registration is better than
mine, its HDR composition is not.** The two combine well.

## Sun-frame totality, and the 2 R crop

**The detector measures the Moon during totality, but the video should hold the
SUN still** — the corona is what the viewer is watching, and it is attached to the
Sun. The two centres coincide at greatest eclipse and separate to at most
(Rmoon − Rsun), about 9 px at half resolution, at each contact; in between the
offset runs very nearly linearly with time. So the Moon track is converted to a
Sun track by subtracting an offset that is zero at mid-totality and accumulates at
the differential rate `corona-drift.js` measured. Mid-totality is taken as the
midpoint of the observed unfiltered span — C3 fell in a recording gap, so that is
a few seconds off the true instant, worth about a pixel.

`smooth_track.py` now computes an absolute UTC per frame (from `segments.json`)
to anchor that offset.

**2 R is not achievable in both axes.** A square window reaching 2 R needs a
558 px half-width, and the Sun sits closer than that to a sensor edge for most of
the session — a 1:1 crop at 2 R drops **100%** of frames. The vertical limit is
hard: the Sun's y-track runs 352–623, so 558 px of clearance below the top edge
does not exist for much of the sequence.

Current cut is **3:2 at 1100x740 half-res = 2200x1480**, which is **1.97 R wide**
(the requested 2 R, to within a rounding step) and 1.33 R tall, costing 15% of
frames and keeping all 147 totality frames. Alternatives measured:

| aspect | window | field | dropped |
|---|---|---|---|
| 3:2 | 1100x740 | 1.97 x 1.33 R | 15.2% |
| 4:3 | 1100x820 | 1.97 x 1.47 R | 30.9% |
| 16:9 | 1100x620 | 1.97 x 1.11 R | 13.3% |
| 1:1 | 1080x1080 | 1.94 x 1.94 R | 100% |

Totality step after the Sun-frame conversion: median 0.64 px, p90 1.72 px,
max 5.29 px.

## Reducing dropped frames

Three independent sources, addressed separately.

1. **Blown frames (54)** - clipped source data, unrecoverable, correctly dropped.

2. **Untrusted position (was 117, now 46).** The trust check exists to catch the
   few frames where the scope was knocked. But when MOST of a capture fails it,
   the detector is at fault, not the frames - and dropping the lot throws away
   good footage for a fault of mine. 14_05_08 lost all 71 frames that way. A
   capture failing more than MOSTLY_FAILED (0.5) of its frames is now kept whole
   and placed on the shared drift line, giving up sub-pixel residual tracking for
   that capture in exchange for not leaving a two-second hole.

3. **Window past the sensor (was 197, now ~139).** The strict rule dropped any
   frame whose window reached past an edge, however slightly - and a quarter of
   them reached over by under 10 px on a 1100 px window. `--pad-tolerance` sets
   how much black edge is acceptable before a frame is dropped.

| tolerance | % of width | frames kept | dropped |
|---|---|---|---|
| 0 px | 0% | 1096 | 15.2% |
| 30 px | 2.7% | 1210 | 11.3% |
| **60 px** | **5.5%** | **1225** | **10.2%** |
| 100 px | 9.1% | 1287 | 5.6% |

Shipping at 60 px: **1225 frames, 40.8 s**, up from 1096/36.5 s. In practice only
129 frames (10.5%) show any black at all and the worst is 38 px, 3.5% of the
width, in a dark corner.

Note 14_05_08 is recovered by the trust change and then lost again to padding -
the Sun genuinely sat near a sensor corner throughout that capture, so it cannot
be framed at this window size. That is a real constraint of the data, not a
processing choice.

## Framing criterion: the disc, not the window

This is a timelapse, not a stack. A black wedge in the corner of the field is
cosmetic; a frame where the SENSOR cut into the disc is missing subject and cannot
be used. `--require-disc` drops only on the latter, letting the window hang off
the sensor as far as it likes with the renderer padding.

**Nothing is dropped for framing under that rule - 0 of 1364.** The disc was
always whole on the sensor; every earlier framing drop was about the window, not
the data. Final cut is 1364 frames, 45.5 s, at 2240x1680 = **2.01 R wide,
1.51 R tall**, with 37% of frames showing some black at a corner (worst 175 px,
16% of the width).

Sequence of framing rules and what each kept, all at 147/147 totality:

| rule | frames | length |
|---|---|---|
| window inside sensor | 1096 | 36.5 s |
| window, 60 px tolerance | 1225 | 40.8 s |
| **disc inside sensor** | **1364** | **45.5 s** |

## Totality wobble, second pass

`14_14_36` was stepping 1.56 px/frame against 0.63 for the two totality captures
either side of it. Its lag-1 passed the noise gate, so residual tracking was on -
but it is a ring-search capture, and that detector scatters ~7 px against ~1 px
for the circle fit. Even when lag-1 passes, following a series that noisy over
five seconds of video injects more error than it removes.

Residual tracking is now refused outright for ring-detected captures. All three
totality captures now step at an identical **0.631 px/frame median**. The single
5.69 px outlier in 14_14_36 is a dropped-blown-frame gap, which the dissolve
covers.

## The remaining totality wobble was BRIGHTNESS, not motion

Worth recording because it was misdiagnosed twice. With positions perfectly smooth
at 0.631 px/frame everywhere, the corona still pulsed through 14_14_36. The cause
was the gain chain: rendered brightness ran 0.31 to 0.44 through that capture
against 0.67 for the segment before it and 0.80 for the one after.

The chained gains are correct for the partials, where the subject genuinely
changes brightness and only the operator's exposure steps should be removed.
**Totality is the opposite case**: the corona's real brightness is essentially
constant over the four minutes it is visible, so ANY change between segments is
the operator riding the exposure and all of it should go. Each stable unfiltered
segment is now anchored to the target directly rather than through the chain,
which would be wrong for the filtered phases and is not applied there.

Transitions inside totality also needed a ceiling. The exposure proxy moves fast
through a ramp and the estimate overshoots at the end of one: single frames were
landing at 3.5 to 4.9 against a 0.78 target, a white flash one frame long.
`TRANSITION_CEILING` (1.15) caps them on the frame's own p99.

| rendered brightness, 149 totality frames | before | after |
|---|---|---|
| median | - | 0.780 |
| p10 - p90 | - | 0.656 - 0.825 |
| max | 4.86 | 1.78 |

The 1.78 maximum is a single frame just after C2 whose chromosphere is genuinely
much brighter than its segment median; the highlight shoulder absorbs it.

## Render speed: parallel shards, not a different disk

Moving the output to another drive does not help. Measured on this machine:

| | throughput |
|---|---|
| S: SER read | 1035 MB/s (0.016 s/frame) |
| S: PNG write | 4170 MB/s (~0 s/file) |
| Z: PNG write | 3264 MB/s - SLOWER than S: |

Against 0.45 s/frame of demosaic, crop and PNG encode, I/O is a rounding error.
The render is CPU-bound inside PJSR and the only lever is concurrency.

`run-timelapse.mjs --workers N` (default 4) splits the frame list into CONTIGUOUS
blocks and runs one PixInsight per block through the shared launch lock. Blocks
rather than round-robin because the renderer cross-dissolves across sequence gaps,
and a round-robin shard sees a gap at every frame; blocks keep the dissolve right
everywhere except the n-1 seams. Each shard writes its frames' ORIGINAL sequence
numbers, so the shards jointly produce one continuous seq_%05d run.

**9.7 min -> 3.1 min at 4 workers**, output verified contiguous 0..1292 with no
zero-length files (which is what a shard collision would look like).

## An unplaceable capture is worse than a gap

14_05_08 was recovered by the mostly-failed rule and then made the video lurch at
the 10 s mark: its detections scatter over 595 px in x and 444 px in y, so the
median used to place it is the median of noise, and it entered 35 px off. Every
other filtered capture sits at 0.1-0.8 px median.

The mostly-failed rule now also checks that the detections AGREE with each other
(`UNPLACEABLE_SPREAD_PX` = 60). A capture that fails the trust check but is
self-consistent is kept - the threshold was at fault. One that scatters is
genuinely unplaceable and is dropped. Same principle as the scope-touch frames:
a short gap beats a confidently wrong position.

## Totality is ONE track, not three

The Moon jumped 33 px in y between 14_14_36 and 14_16_14, where the physical drift
over that 38 s gap accounts for about 5 px. Cause: every capture was given its own
intercept. That is right everywhere else, because the mount was nudged between
captures - but NOT during totality, where the scope was touched to take the filter
off and again to put it back and not in between.

Verified before adopting: a single line through all 149 totality detections leaves
mean residuals of (-2.6, 3.4) and (1.1, -1.4) px on the two long captures. They
genuinely are one continuous track. Totality is now one group keyed
`("TOTALITY", "unfiltered")` and fits its own rate from its own 149 samples, which
measures it better than the session mount drift plus a modelled lunar term.

Three bugs surfaced while making that change, all mine, all worth remembering:

1. **The unplaceable test judged raw spread.** A long group covers a lot of ground
   legitimately - unified totality spans 210 px of real Moon motion - so it was
   declared unplaceable and ALL of totality was dropped. Spread is now measured
   about the fitted track.
2. **A `None` residual was counted as zero scatter.** `None` means the detection
   was too far out to measure, which is a failure, not agreement - it brought
   14_05_08 back with "spread 0 px" and restored the 10 s lurch. Only measurable
   residuals count now, and a group with under a quarter of them is unplaceable.
3. **The model was indexed on per-capture frame number while the line was fitted
   against absolute time.** Frame index restarts at zero each capture, so the
   track reset at every boundary - a 49 px step. The unified group now evaluates
   on continuous seconds; per-capture groups keep the index, where it is
   equivalent and keeps the shared drift slope in its native units.

Remaining totality boundary steps are 31 px, which is 0.82 px/s x the 38 s
recording gap: the Sun genuinely moved that far on the sensor while SharpCap was
flushing to disk. The crop follows it, so the Sun stays put and the Moon advances,
which is the eclipse happening rather than an artifact.

## 14_05_08 - RESOLVED, and it was not a detector bug

The scope was being physically handled during this capture. Both the circle fit
AND an independent brightness centroid agree the Sun moves hundreds of pixels
through it, in steps of **408 px in 0.86 s** followed by 80 px steps. The frames
are perfectly good - which is exactly why it looked like a detector fault.

`centroid_rescue.py` places the capture from the brightness centroid. A centroid
is NOT the disc centre for a crescent - that was the original bouncing bug - but
the offset is a smooth function of how thin the crescent is, so it is measured on
the neighbouring captures where the circle fit works and interpolated in time
across the broken one. Measured offsets: (-169.0, +112.8) px on 14_03_31 and
(-190.2, +117.0) px on 14_06_45, drifting smoothly as the crescent thins.

`smooth_track.py` then FOLLOWS such a capture rather than modelling it: positions
come straight from the detections with a 5-sample median against per-frame noise,
no line and no slew limit, because the motion is real and a shared-drift model
cannot represent it. Since the window is cropped around the Sun, tracking the
lurches puts the Sun in the same place every frame and the video comes out smooth
- the sensor moved, the subject did not.

71 frames recovered, 2.4 s of partial phase restored.

Standing rule note: pix-planetary forbids PIPP. No PIPP was used - only the
centroid idea, computed directly from the raw SER. The rule against PIPP OUTPUT
as input data is untouched.

## Per-exposure detection bias inside totality

A small nudge remained INSIDE totality, at an exposure change rather than a
capture boundary. The ring search locates the Moon's limb from a brightness
gradient, and where that gradient crosses threshold depends on exposure - so the
same Moon is reported a few pixels away at a different level. Measured inside
14_14_36 the two levels disagreed by **6.6 px in y**. Because the model is a
single smooth line it inherits that as a per-level offset, and the subject nudges
whenever the operator changed exposure.

The bias belongs to the detector, not to the sky, so the mean residual of each
constant-exposure run is measured and removed (keyed on the frame's gain, which is
exactly the constant-exposure identifier). Runs shorter than `LEVEL_BIAS_MIN_N`
(4) are left alone because their mean is noise, and corrections are capped at
`LEVEL_BIAS_MAX_PX` (12) so a bad run cannot displace the frame.

Corrections applied: (+4.68, -3.86), (-5.69, +4.84), (-1.48, +1.43),
(+0.83, -0.61) px. Within-capture level disagreement **6.6 px -> 3.2 px**.

The 21.5 px spread that remains across the whole of totality is the Moon genuinely
crossing the Sun and should be there.

## Camera shake when the film went back on

14_17_51 rings for a few seconds after the solar film was refitted: real steps of
5-27 px decaying to 1-3 px, made worse by the crescent being at its thinnest just
after third contact, where arc coverage drops to 60 deg and single fits are
unreliable. The 1 px/frame slew limit could not follow real motion that fast, so
the subject shook.

The `follow directly` treatment written for 14_05_08 is now a general rule rather
than a special case. A capture is FOLLOWED rather than modelled when it departs
from its own drift line by more than `FOLLOW_EXCURSION_PX` while still stepping
smoothly frame to frame - a handled or ringing scope moves far but continuously,
whereas noise gives large steps AND a large excursion.

Two calibration mistakes on the way, both worth remembering:

- Measuring excursion in ABSOLUTE position flagged twelve captures including
  entirely normal ones, because ordinary mount drift covers ~40 px over a 60 s
  capture. It has to be measured about the drift line.
- The threshold was then set from the data rather than guessed. Residual excursion
  per capture separates cleanly: **480, 142, 39, 37, 22, 20 px** for the captures
  where the scope was handled or ringing, against **4.8 px and below** for every
  ordinary one. 15 px sits in that gap and catches exactly the six.

Followed captures use a 7-sample median, wider than the 5 used elsewhere, because
the fits are poor exactly where the motion is largest.

14_17_51 now tracks its detections to a median of 0.32 px, and all 69 frames are
kept.

## Thrashing captures: keep the longest steady stretch

Following a capture works when the scope moves far but CONTINUOUSLY - a bump that
settles, a slow reposition. It fails when the scope is being actively waved.
14_05_08 swings **250 to 430 px between consecutive video frames**, repeatedly.
Each individual frame is a fine picture, which is why it kept looking recoverable,
but no tracker makes that sequence smooth: where the median lags a swing, the
subject lurches with it.

Cutting just the thrashing frames was not enough - the surviving fragments are
separated by the repositioning itself, so the crop still jumped 250-430 px between
them and stitching them was no smoother than leaving the thrash in. A followed
capture now keeps only its LONGEST continuous steady stretch, giving one piece
with a dissolve at each end.

| capture | kept | cut |
|---|---|---|
| 14_05_08 | 16 / 71 | 55 |
| 14_11_23 | 42 / 69 | 27 |
| 14_01_54, 14_17_51, 14_27_34, 14_29_11 | all | 0 |

Largest within-capture model step is now **27.7 px**, in 14_13_00 across the
blown-frame gap at the filter change, where a dissolve covers it. It was 429 px.

This is the honest limit for 14_05_08: about half a second of it is usable as
smooth footage, not the full 2.4 s.

Still true for 14_05_08, which is centroid-placed and whose positions really are
noisy. **Superseded for every other filtered capture** by the section below - the
premise that a shaking capture cannot be recovered was wrong wherever the fit is
precise enough to follow.

## Partial phases: stabilise per frame, do not model

The whole modelled-track apparatus was the wrong tool for the partials, and this
is the single largest improvement to the video.

The clip-to-clip placement was already good, because it uses each capture's own
measured disc centre. Run that same measurement one level finer - between FRAMES
of a capture rather than between captures - and the shake cancels too. Each
filtered frame is now centred on its own fit. Nothing is modelled, nothing is
smoothed, no slew limit: whatever the mount did during the capture is removed
rather than averaged.

This works because of a 30:1 margin that is worth re-measuring rather than
assuming, if the detector ever changes:

| | frame-to-frame fit scatter | real mount wander |
|---|---|---|
| ordinary filtered capture | 0.15-0.37 px median, <0.9 px p90 | 1-6 px |
| 14_17_51 (film back on) | 0.61 px median | 19.9 px |

A modelled track *cannot* do this. `MAX_RESID_STEP = 1.0 px/sample` exists
precisely to stop the model chasing fast motion, so it leaves the shake in by
design. That limit, and `smooth_series()`, now apply to totality only.

The one thing a per-frame follow must not do is follow a fit that has **snapped to
a different solution**. As the crescent thins the disc fit can be confident and
wrong: 14_11_23 reports arc 140-170 deg, 120 inliers and rms 1.7 while jumping
40-140 px per frame through its last 40%. Departure from the capture's own robust
drift line separates the cases cleanly, which is where `FOLLOW_LINE_TOL_PX = 40`
comes from:

| | departure from drift line |
|---|---|
| ordinary captures | 0.5 - 5.9 px |
| 14_17_51, genuinely ringing | 18.4 px |
| 14_11_23, snapped fits | 145.7 px |

Untrustworthy frames are **cut, not guessed at**, and that now costs nothing in
continuity: every surviving frame is centred on the Sun, so a cut leaves a hiccup
in time and *no jump at all* in position. This is the opposite of what happens
when a modelled track drops a frame, and it is why the fragment-stitching problem
described above does not apply here.

Kept: 43/69 on 14_11_23, 60/69 on 14_17_51, 37/42 on 14_13_00, all frames on every
other filtered capture. 1311 frames, 43.7 s, totality 148/148.

**Bug found and fixed while doing this**: `run-timelapse.mjs` globs `seq_*.png` to
encode rather than reading the config, so leftovers from a longer previous render
were silently appended - a 1311-frame render came out as a 1322-frame video whose
last 11 frames belonged to a different cut. The frame directory is now cleared
before rendering.

## Drizzle-stacked totality (2026-08-13)

The video now renders on a **2x grid — 2240x1680**, the sensor's own sampling. The
148 totality frames are drizzled from 20-frame groups; everything else is
resampled onto the same grid so the sequence has one geometry.

**Why prominences and not the corona.** Prominences are Halpha, so they sit almost
entirely in R, and R is one sample per 2x2 CFA site: **3.43 arcsec/px**, half the
linear resolution of luminance and the worst-sampled signal in the capture. Note
this is NOT a demosaic loss — superpixel passes R through intact, because R really
is that sparse. Only drizzle recovers it, and only with sub-pixel dither.

**The dither was free.** Across a 20-frame group the pointing moves **0.89 px**,
landing the frames in 8-9 of the 16 sub-pixel cells. The unaligned mount supplied
exactly what drizzle needs.

**Alignment trial** (`scripts/align-trial.mjs`, one prominence group, R channel):

| aligner | edge/noise |
|---|---|
| FFT phase correlation | 1561.8 |
| `alignOnGradient: true` | 1571.2 |

0.6% apart — it makes no difference, because at 0.89 px of excursion there is
nothing for a better aligner to fix. FFT is kept. Noise falls as **√N exactly**
(x2.31 / x3.26 / x4.51 at 5 / 10 / 20 frames against √N = 2.24 / 3.16 / 4.47),
which is the real evidence the aligner works. Frame selection buys nothing here —
quality spread across a group is only ~3%, so all 20 are used, not
`bestFraction 0.4`.

**Group sizing** is in `gen_timelapse.py` (`STACK_MAX = 20`): the frames between
one video frame and the next, bounded by the SEGMENT end so a group can never span
an exposure change.

**Partials are stacked too, since 2026-08-15.** They were left out at first on the
grounds that a hard white-light limb is not where the detail is and that it means
reading all 446 GB rather than the totality slices. Neither holds up: the panels
magnify a sunspot to 3x and a cusp to as much as 7x, and at those magnifications an
unstacked frame is being enlarged past its own noise, with nothing under it but a
bicubic interpolation of the superpixel grid. The read is not the problem either -
1164 filtered frames at 20 raw frames each is 386 GB, about six minutes off S:. The
cost is CPU in the renderer: **1225 of 1245 filtered frames now carry a full
20-frame stack, and the render goes from ~40 min to roughly 2 h at 12 workers.**

Twenty filtered frames still sit at `stack = 1` because they are the last sample of
their segment and `seg_end - index` leaves no room. Those will be visibly noisier
than their neighbours. Extending a group BACKWARD would fix it but decouples the
stack from the frame the disc track was measured on - the reference is the group's
first frame - so it is left alone and recorded here instead.

**Costs:** ~10 s/frame for a stacked frame against 0.45 s unstacked; PNGs double in
size. Sharding is contiguous, so the shard holding totality dominates wall time.

**Observed, not fixed:** the corona in the rendered totality frames has a green
cast. It predates this change (gain is applied equally to all three channels, and
superpixel G is unchanged), and it is the same imbalance as the green limb ring in
`corona-stretch.js`. Worth a white balance pass on the unfiltered frames.

## Three requests, DONE (2026-08-14)

**1. The first totality level now holds for 12 s.** It is the gain-27.31 run in
`14_13_00`, the short-exposure prominence footage right after C2, and it was
twelve video frames — 0.4 s. `gen_timelapse.py` now finds it by MEASUREMENT (the
stable totality segment needing the most gain is by definition the one shot
shortest; here it is 20x clear of the rest) and resamples it to one video frame
per raw frame: **211 unique frames padded to 360 = 12.0 s**.

Two bounds on the dense run, both load-bearing:

- It stops `STACK_MAX` raw frames short of the segment end. Groups start at their
  own frame and run forward, so without this the last twenty frames would get
  progressively shorter stacks and the picture would visibly grow noisier on the
  way out.
- Groups OVERLAP — consecutive frames share 19 of 20 raw frames. The old
  gap-to-the-next-video-frame rule would have handed dense frames a group of ONE,
  undoing the drizzle exactly where it matters most. Nothing about drizzle needs
  disjoint groups; tiling was only ever a way to use each raw frame once.

About seven frames in ten are shown twice, which is normally judder and here is
invisible: the Moon advances **0.006 px between consecutive raw frames**, so
neighbours differ only by seeing and noise, and each is already a mean of twenty.

**2. Panels are labelled and a panel with no subject is dropped.** `gen_insets.py`
emits between 1 and 4 features, each carrying its own corner and name; the old
`pts.extend(pts[:2])` padding is gone. Gating: the sunspot only while it is not
behind the Moon, the lunar limb and both cusps only while the discs actually
overlap, a prominence only where that level's own search found one. Result over
the video: **3 panels on 950 frames, 4 on 710**.

`tl-frames.js` rasterises a 5x7 bitmap font itself — PJSR's `Graphics`/`Bitmap`
are not reachable from a script that works on plain Float32Array planes. Labels
sit OUTSIDE the panel (below a top one, above a bottom one) on a plate that
MULTIPLIES the background by 0.25 rather than filling it, so it stays invisible
against black sky and still carries white text over the photosphere.

**3. Prominences are detected per EXPOSURE LEVEL.** They were detected once at
mid-totality and held for all of totality, which pointed panels at whatever had
been visible in a *different* exposure. Now each `(file, gain)` run is measured on
its own middle frame, and peaks below `PROM_MIN_SNR` are not emitted at all.

## Three bugs found while doing it, each by LOOKING

1. **`PROM_R_INNER = 0.97` was throwing away real prominences.** The missing one
   in the user's screenshot was the THIRD STRONGEST feature in its frame
   (snr 38.6) and was excluded on geometry, not brightness: it sits at **0.939 R**.
   Prominences stand on the SUN's limb and the Sun is the smaller disc — 1919"
   against 2010", ratio 0.955 — so one is already inside 0.96 R before the ~7 px
   residual in the fitted Moon track is counted. Now 0.90–1.25. Widening is free:
   behind the Moon there is no Halpha, so the lunar disc cannot win an argmax on
   `R - k*G`.
2. **The anti-flicker hold was carrying a layout across exposure levels.** A held
   permutation maps SLOT to corner, and each level's prominences are re-detected
   and re-ranked, so slot 1 means a different prominence either side of a
   boundary. On `14_14_36` f340 the held layout cost 2377 px of leader line
   against 1875 for that frame's own optimum — two lines crossing the corona. The
   hold is now abandoned whenever the feature LIST changes rather than moves.
   Measured after: 3 of 1661 frames sub-optimal, worst excess 1 px. (Minimising
   total length also guarantees no crossings — a minimum-length matching cannot
   contain a crossing pair.)
3. **"Leading" and "trailing" cusp were not measurable.** The cusps sit
   symmetrically either side of the Sun-Moon line and the Moon travels very nearly
   along it, so their projections onto the direction of travel are equal to first
   order: **12.7 px median separation, 3.4 px at worst, against 7.1 px of residual
   in the Moon track**. The choice was noise and it quietly swapped the two horns
   once. They are now **upper** and **lower**, which are 466 px apart at the median,
   never closer than 247, never ambiguous — and checkable by the viewer, which
   leading/trailing is not without a sky orientation that nothing here solves for.

## Totality stability: the level-bias correction was the only thing moving

Asked whether totality could be steadier. It could, and the cause was a fix
fighting a fix.

The per-level detection-bias correction subtracts each constant-exposure run's
mean residual. It was written for a model that FOLLOWS the detections, where the
bias reaches the render as a nudge at every exposure change. But totality is
placed on a bare straight line — `tracked` is false, because ring-search scatter
fails the lag-1 test — and rx = ry = 0. A line has no per-level offset to inherit,
so every frame already sat on a perfectly smooth path, and adding a constant that
changes at level boundaries could only INTRODUCE a step.

It did. `bias` is now gated on `tracked`:

| worst single-frame step | before | after |
|---|---|---|
| `14_13_00` unfiltered | 8.22 px | **0.71 px** |
| `14_14_36` | 11.68 px | 6.47 px |
| `14_16_14` | 0.72 px | 0.72 px |

`14_14_36`'s remaining 6.47 px is not a jump: it is a genuine 180-raw-frame
(7.7 s) hole where blown frames were dropped, and the renderer already
cross-dissolves it (`GAP_FRAMES = 60`).

What is given up is absolute placement — each level's frames sit up to 8 px from
where its own detections put them. Invisible in a Sun-fixed window, where a viewer
sees changes and not offsets, and the line is the better estimate anyway of a Moon
that genuinely moved in a straight line over these three and a half minutes.

## One regression caught, and what it says about the follow heuristic

Adding dense frames sent **the whole of totality down the "followed directly"
path**, which then cut 135 frames as thrashing — every frame of `14_14_36` and
`14_16_14`. The heuristic asks whether the pointing steps smoothly by taking the
MEDIAN frame-to-frame step, and dense frames sit one raw frame apart instead of
twenty, so their steps are twenty times smaller for no physical reason.

Any statistic over "neighbouring samples" now needs the original cadence. Three
places were fixed: the follow heuristic, the robust line fit, and the level-bias
ensemble (which is now the mean of the per-level MEANS, one vote per level, rather
than the mean of all samples).

## Tracked zoom panels — `gen_insets.py` (2026-08-14)

Every frame is RENDERED on the 2x fine grid, because the panels magnify past 1:1
and the sequence needs one geometry throughout. Only totality is drizzle-STACKED,
though — the partials are bicubic upsamples of the superpixel grid and gain no
real detail. (An earlier version of this section claimed all frames drizzle. They
do not; see the deliberate decision under "Drizzle-stacked totality" below. The
cost of changing it is ~10 s/frame against 0.45 s for 1164 partial frames.)

Panels follow features computed per frame, never hand-placed: the auto-detected
sunspot held in Sun coordinates, the Moon's deepest incursion, and both cusps
(circle-circle intersections); through totality, the strongest Halpha prominences
in that exposure level. Corners are assigned by minimising total leader-line
length, with a 9-frame hold so panels do not flicker.

Four bugs, each found by LOOKING rather than by a metric:

1. **drift.json's sign is not the Moon's direction.** Using it put the Moon
   down-left when the picture shows it up-right, throwing every cusp into empty
   sky. Its convention is the shift needed to ALIGN two frames. In
   `smooth_track.py` the same term is worth <=9 px so the error was invisible
   there; here it was worth 240 px. `gen_insets.py` now FITS the Moon from the
   terminator instead (Kasa on the lit/unlit boundary), which needs no ephemeris
   and reports a residual — 7.5 px over 12 frames.
2. **A box mean cannot normalise limb darkening.** It removes a constant; limb
   darkening is a gradient, so the darkest pixel was just the outermost one. A
   radial profile removes it exactly.
3. **Masking the Moon geometrically is not enough** for the sunspot search — the
   model is good to a few px and a few px at the terminator is photosphere vs
   shadow. Find the photosphere by BRIGHTNESS.
4. **Prominences are not the brightest limb points** — the corona is. Search
   `R - k*G`, which cancels anything white and leaves Halpha. And search the
   annulus in 2-D, not a single ring: a prominence is an arch, so its peak is not
   at one radius.

Panels are composited AFTER the dissolve, with `prevOut` holding a pre-panel copy.
Cross-fading them turns the box outlines into doubled ghost lines.

## Totality stabilised per frame, from the corona itself (2026-08-14)

`pjsr/tl-corona-track.js` + `scripts/run-corona-track.mjs` (1.2 min) measure every
totality frame by phase correlation against one reference per capture, writing
`out/diag/corona_track.json`. `smooth_track.py` places totality from that instead
of from the modelled line.

**Why the line was not enough.** It removes drift; it cannot remove the mount's
wobble. Measured against the correlation, the line sat **0.74 px RMS** from where
the picture says the frame is — and at 3x inside a zoom panel that is ~4.4 px of
shimmer in all four panels at once, which is what "jittery in all spots" looks like.

**Why correlation works where the ring search does not.** The frame is not a smooth
blob: it carries the lunar limb, a hard edge right round the disc, with prominences
on it. That is the same signal the drizzle already uses to stack 20 raw frames
inside a group — this measures it BETWEEN groups, which nothing was doing.

**Its noise floor is measurable, and that mattered.** Inside the densely resampled
level consecutive frames are 0.043 s apart, over which the Moon moves 0.005 px, so
what is left is almost pure measurement: **0.39 px**. Raw, that is worse than the
disc fit stabilising the partials, so the track is smoothed over a window measured
in SECONDS (`CORONA_SMOOTH_S = 1.5`) rather than samples — totality is sampled at
0.86 s over most of it and 0.043 s through the held level, and a fixed tap count
would barely touch one and erase the other.

| high-frequency content (px) | before | after | steadiest partials |
|---|---|---|---|
| `14_13_00` held level | — | **0.005** | 0.146–0.200 |
| `14_14_36` | 0.74 RMS placement error | **0.137** | |
| `14_16_14` | 0.77 RMS placement error | **0.150** | |

Each capture is anchored on its MEAN against the line, not its first frame, so one
bad line value cannot displace a whole capture; the line still supplies the
relationship BETWEEN captures, which is right because the scope was not touched
during totality. A correlation-placed frame is also no longer dropped for
disagreeing with its own ring detection — its placement does not depend on it.

The per-level bias correction is now gated on `tracked`. On a bare line there is no
per-level offset to inherit, so it could only INTRODUCE a step, and it did: 8.2 px
into the prominence level, 11.7 px inside `14_14_36`.

## Cusps are read off the PICTURE, not intersected from two circles (2026-08-14)

`find_cusps()` walks the Sun's limb and takes the two ends of the lit run. The old
geometric `cusps()` was exact arithmetic on inexact inputs and worst exactly where
it was used: two circles that nearly touch have an intersection that slides a long
way along the limb for a small error in either centre or either radius, and both
radii are estimates while the Moon's centre came from a fitted track with ~7 px of
residual. It put the box ~19 superpixel px off the visible horn. The Sun's limb is
the best-measured thing in the frame (fixed-radius fit, a few tenths of a pixel),
so walking it needs no Moon centre and no lunar radius. `cusps()` is deleted.

Three details, and the first was worth more than the other two together:

- **Scan a BAND of radii and keep the brightest per angle — never one circle**
  (`CUSP_R_LO/HI/STEPS` = 0.93–1.01 R, 17 steps). One circle works while the
  crescent is fat and fails where it matters: near second contact the lit band is a
  few px wide radially and does not sit at a fixed fraction of r_sun, so a single
  circle clips it obliquely, the brightness falls before the horn ends, and every
  box lands SHORT. Not a small bias — the dominant NOISE source too.

  | per-capture scatter about the fit | one circle | band |
  |---|---|---|
  | typical capture | 2–5 px | **0.2–1.0 px** |
  | `14_13_00` (thinnest crescent) | 8.0 px | **1.9 px** |
  | `14_17_51` | 5.0 px | **1.3 px** |

- **The crossing is taken LOW** (`CUSP_EDGE_FRAC = 0.08`), for the same reason
  `tl-centres.js` takes its limb points low: the horn TAPERS, so the falloff is
  gradual and a half-height crossing sits inside the visible tip.
- **Cusps carry their own zoom** (`CUSP_ZOOM = 5.0` against 3.0), per-inset in the
  config and honoured by `drawInsets`. A horn is a needle; at the zoom that suits a
  sunspot group it is a thin sliver crossing one corner.

**Then fitted per capture, per cusp, as a quadratic** (`fit_cusp_track`). Even at
0.2–1.9 px, magnified fivefold that is 30 px of visible wobble, and the first cut
had the tip leaving its own box. Each cusp's angle about the Sun's centre is smooth
and slowly curving, so fitting and regenerating is smooth by construction. The two
are tracked SEPARATELY by their stable upper/lower identity, not as a bisector plus
half-separation: the pair passes through diametrically opposite when the centres
are sqrt(rMoon²-rSun²) = 86 px apart, which happens partway through this eclipse,
and a bisector flips half a turn exactly there.

**A whole-eclipse geometric fit was tried and abandoned.** Solving the Moon's radius
and track against all 2328 measurements at once ought to beat per-capture fits. It
did not: unbounded the radius ran to 279814 px (a clipped arccos is FLAT, so past
some radius every frame lands in the flat region and the solver loses its
gradient), and even bounded the prediction started **123 px** from measurements
that agree with each other to 3 px. Do not retry without first explaining that
123 px.

**"Leading" and "trailing" cusp were not measurable** and are now upper/lower. The
cusps sit symmetrically either side of the Sun-Moon line and the Moon travels very
nearly along it, so projections onto the direction of travel are equal to first
order: 12.7 px median separation, 3.4 px at worst, against 7.1 px of residual in
the Moon track. It quietly swapped the two horns once. In image y they are 466 px
apart at the median and never closer than 247.

## Caution: the validation tooling has been wrong FIVE times

Two more, both this session:

- **A second-difference "shimmer" metric** reported `14_14_36` three times worse
  than its neighbours. A second difference assumes EVEN SAMPLING, and that capture
  has a 180-frame hole where blown frames were dropped; the two triples straddling
  it compare a 0.86 s step with a 7.7 s one. Every other triple reads 0.03–0.65,
  identical to `14_16_14`. Restricting to equally spaced triples: 0.531 → 0.137.
- **The theory tested first was wrong and the measurement said so.** `14_14_36` was
  supposed to correlate badly because its reference is a transition frame at half
  the bulk exposure. Fixing that moved it 0.539 → 0.531, i.e. nothing.

## Dissolves: leave the panels alone. Two fixes were worse than the flaw

The panels describe the new capture, but for the length of a dissolve the picture
is mostly the old one — the Moon advances 7 to 136 px across a gap — so a box
reaches its new place up to three frames (0.1 s) before the subject does. Two
attempts to close that, both reverted:

1. **Slide the boxes from old positions to new, on the dissolve weight.** Much
   worse. A box partway between two eclipse phases points at neither, and
   `drawInsets` samples the PANEL at that same position, so every zoom showed the
   wrong patch of sky while drifting. Three boxes doing that at a boundary reads as
   the panels freaking out.
2. **Fade the whole annotation in on the dissolve weight** (25/50/75/100%). Fixed
   the timing honestly and drew nothing untrue, but a panel whose brightness ramps
   over three frames reads as the corners flashing.

**Current behaviour: constant opacity, correct position, every frame.** The 0.1 s
of lead is the smallest of the three artifacts and the only one that never draws
anything false. `drawInsets` takes no alpha; there is no `prevInsets`.

If it is ever worth another attempt, the one untried idea is to FREEZE the panels
through the dissolve — keep a copy of the previous frame's rendered corner
rectangles and blit those until the blend finishes, so the annotation follows the
old picture until the old picture is gone and then updates once. That costs one
more full-frame copy per frame (the same size as `prevOut`) and still ends in a
single discrete switch, which may itself read as a pop. Do not attempt a fourth
variation without watching the boundary first.

## The cusp box now scales with the horn (2026-08-14)

A fixed box suits a needle and not a wedge, and the horn is both at different times:
near second contact the limbs cross almost tangentially and it is a hair, near
first and last contact they cross at a wide angle and it is a blunt corner. With
one box size the fat horns were not in the frame enough.

**Measured, not modelled, and free.** The band scan in `find_cusps` already samples
17 radii per angle, so counting how many clear the threshold gives the crescent's
radial THICKNESS at that angle. Walking inward from a horn until that reaches half
the band gives the arc distance over which the horn opens up. Thickness itself
would saturate — the band is only 0.08 R deep and a fat crescent fills it
everywhere — but the DISTANCE to reach a given thickness does not.

That length is the span the box needs, so `half = CUSP_BOX_K * median(opening)`,
clamped to `CUSP_HALF_MIN/MAX` (14–46 superpixel px), and both the per-inset zoom
and the inward shift follow from it. One value per CAPTURE: both horns are the same
shape by symmetry, and the wedge angle changes far too slowly to re-decide per
frame.

Measured across the eclipse, and it tracks the phase cleanly:

| capture | horn opens over | half-box | zoom |
|---|---|---|---|
| `13_52_58` (fat) | 12 px | 16 px | 6.6x |
| `14_01_54` | 19 px | 25 px | 4.1x |
| `14_08_22` | 48 px | 46 px | 2.3x |
| `14_13_00` (needle) | 369 px | 46 px (capped) | 2.3x |

Ten captures near totality hit the cap, which is correct: a 369 px needle cannot
fit a panel at any useful magnification, so it shows the tip and a good length of
horn instead.

## Panel layout is a property of the CLIP, not the frame (2026-08-14)

The corner assignment used to be re-chosen per frame with a 9-frame hold. That
switched panels between corners in the middle of a capture, and worst at the very
start: the first two runs are only 28 and 34 frames long and are **eight and a half
minutes apart**, so the geometry jumps between them and the best assignment jumps
with it. A layout that changes while a clip is playing reads as the boxes swapping
targets rapidly.

The layout is now chosen once from the clip's MEDIAN feature positions and held for
every frame of it, so it can only change where a change is already expected and
hidden — at a capture boundary, under the dissolve. Keyed on the label list as well
as the file, so a capture that loses a feature partway (the sunspot going behind
the Moon inside `14_01_54`) gets a fresh layout for the frames after it rather than
one built around a feature that is no longer there. `HOLD_FRAMES` is gone.

**That was not enough, and neither was weighting continuity.** A `LAYOUT_MARGIN`
that charged a swap 20% of total leader length still left panels trading, because
near-ties were never the real cause. What actually moved them was a feature
LEAVING: when the sunspot goes behind the Moon partway through `14_01_54` it frees
the upper-left corner, the unconstrained optimum pulls everyone else into it, and
two panels jump mid-clip. Nothing about the remaining features had changed — only
the vacancy.

**Corners are now INHERITED, and a panel never moves unless its own subject is
new.** Each feature takes the corner of the nearest feature of the same name in the
previous clip; only features with no predecessor are placed, into whatever corners
remain free, by shortest leader. Inheriting by nearest POSITION rather than by rank
matters for the prominences, which are re-detected and re-ranked at every exposure
level — by rank, slot 2 of one level is a different prominence from slot 2 of the
next and the panels would shuffle although the picture had barely changed. The
memory is cleared when the state changes, because totality has an entirely
different cast and sits either side of the largest gap in the video.

Result: **3 placement events in the whole video** — the opening clip, the totality
prominences, and the post-totality partials. Measured on the config, the upper cusp
holds corner 2 for every pre-totality capture and corner 0 for every post-totality
one; the lower cusp holds corner 3 throughout. The one change is across totality,
where the crescent genuinely flips to the other side of the disc.

Identity was checked at the same time and is not the problem: the upper/lower
labels never swap horns, with a worst-case y-separation of 196 px.

## Superseded: inheritance is gone, the layout is a ring (2026-08-18)

Everything above about INHERITED corners is history. Inheritance was solving the
right problem — panels must not shuffle mid-clip — with a mechanism that caused a
worse one. A corner held while its subject moves stops matching that subject's
position, and once two panels are out of order their leaders CROSS: 294 frames,
12% of the video, every one of them a four-prominence frame, with the topmost
subject wired to the lower-right panel.

A swap pass was written to untangle them and then thrown away. Swapping a crossing
pair is strictly shorter by the triangle inequality, so it terminates and it works
— but it repairs a symptom. Removing the cause is better and turns out to be
simpler.

**The features and the slots are both rings about the same centre**: the features
on the limb, the slots around the frame. Sort each by angle, walk both in the same
direction, match in order.

I claimed that cannot cross. **It can, and this data contains the counterexample**
— see the section below. What the cyclic match actually bought was a cheap way to
avoid most crossings, which held on this cut by luck rather than by construction.

That leaves only WHICH slots, and this is where the packing was hiding. Minimising
total leader length always bunches, because the shortest leaders are the ones that
never leave the crowded side: four prominences on the left of the disc took the
three left slots and stacked their panels down one edge with the rest of the frame
empty. The cost now subtracts `SPREAD_WEIGHT` px per radian of the TIGHTEST
angular gap between the chosen slots. Swept, not guessed:

| weight | packed frames | min gap |
|---|---|---|
| 0 | 1157 | 37 deg |
| 400 | 1086 | 37 deg |
| 1000 | 660 | 74 deg |
| **2000** | **342** | **74 deg** |
| 8000 | 342 | 74 deg |

It saturates at 2000.

**Eight slots, not six.** A panel may not sit on the disc — one at bottom centre
once covered the bead chain it was magnifying — and the disc is round where the
frame is not, leaving 616 px clear at the sides against 316 above and below. I
first read that as "the top and bottom do not fit" and stopped at six. The right
question was what size DOES fit: shrinking the panel from 420 to **292 px, 70% of
the edge**, buys the entire perimeter instead of three slots a side. The panel is
sized from the disc radius at layout time, so a data set whose disc leaves no room
falls back to six on its own. `panels.use_top_bottom = false` keeps the larger
panels deliberately.

Measured on that config (2393 frames, 2400x1800, panel 292 px):

```
crossing leaders      : 294 -> 0 frames
tightest gap median   :  37 -> 90 deg      (even, for four panels)
slot usage            : bottom 1766, top 1657, right 900, left 872,
                        UL 808, LL 720, UR 466, LR 378
```

**And that is the cut that got sent back**, for three faults the numbers above do
not show. See the next section.

## Corners, clearance and memory (2026-08-19)

Rewarding even angular spacing and nothing else went wrong in three ways at once,
and the measurements that looked good were measuring the wrong things.

**The top and bottom panels touched the Sun.** Not nearly — exactly. The
shrink-to-fit rule took the free space above the disc and subtracted the margin,
which puts the panel edge on the limb by construction:

```
slot        gap to disc
corners        238 px
left/right     150 px
top/bottom       0 px      <- and these were the two most used
```

**Spacing prefers the tightest slots.** Four slots at top, bottom, left and right
are 90° apart; four corners are 74°/106°. So the arrangement that scores best on
spread is exactly the one that puts panels where the frame has least room, and 69%
of panels ended up outside a corner.

**Nothing carried a layout forward.** Every clip re-decided from scratch, so a
boundary could move every panel at once. Only 8 arrangements over the video, which
is why this did not show up in a count — it is not how OFTEN they change but how
MUCH changes each time.

Three rules, in the order they matter:

1. **Clearance is a hard rule, not a fitted panel size.** Each slot is turned into
   the rectangle the renderer will draw, its gap to the limb is measured, and
   anything under `panels.min_clear_r` (0.15 R) is not offered. Top and bottom
   drop out of a 4:3 frame; a wider frame or a smaller disc keeps them. The panel
   goes back to its configured 420 px — 2.1× the area of the 292 px it had been
   shrunk to.
2. **Clearance is also the corner preference.** `panels.clear_weight` pays leader
   length per px of clearance, so the roomiest slot wins. Corners are not named
   anywhere: in a rectangular frame they are simply the roomiest, and preferring
   room keeps doing something sensible on a frame shape where that stops holding.
3. **`panels.continuity_px` charges for moving a panel.** This is inheritance
   again — removed earlier because holding a slot while its subject moved crossed
   the leaders, and safe now only because crossings are priced directly.

### The cyclic-order claim was wrong

Matching two rings in cyclic order is **not** crossing-free in general. Clip
`14_14_36` is the counterexample: four prominences at −159°, −73°, +119° and
+162°, all on the limb at r = 285, assigned UR / LR / LL / UL. The slot angles are
a rotation of their own sorted order, so the matching is cyclic — and two leaders
cross anyway, because each sweeps more than a quarter turn about the centre and
they sweep by different amounts. Order preservation only forbids crossings for
segments that do not wind around the centre.

That layout scored 0 crossings on the delivered cut purely because the search
happened to pick a different rotation. Adding the clearance and continuity terms
changed the pick and **138 frames crossed** — a latent fault, not a new one.

So the restriction to cyclic candidates is gone. Every injective assignment is
scored — 360 of them for four features in six slots, which costs nothing — and
crossings are counted on the clip medians and priced like any other term. **14 of
the 26 clip layouts are now non-cyclic**, so the restriction had been excluding
the best crossing-free answers as well as admitting bad ones.

### Measured

```
                        before    after
crossing leaders             0        0     (before was luck; see above)
worst panel-disc gap      0 px    86 px     = 0.29 R
panels in a corner         31%      95%
panel size               292 px   420 px    2.1x the area
arrangement changes          8        6
median stable stretch    8.3 s   11.4 s
panels that move            10        4
leader length p90        565 px  946 px     <- the cost of corners
```

The leaders get longer, and that is the trade the corners buy: a corner is the
furthest point in the frame from a feature on the limb. They pass behind the disc
rather than over it, so the extra length is drawn on corona, not on the Moon.

### The jump that was not a layout problem at all

Chasing "too many jumps" through the layout found only 10 slot changes in 2393
frames, so most of the chaos had to be something else. It was: **panel 4 in
14_14_36 moved 561 px in one frame** - right limb to left limb, the width of the
Sun - inside a single clip, with its slot unchanged the whole time.

Prominences were detected per exposure LEVEL (file plus gain) and returned ranked
by strength, four kept. At gain 1.23 a fainter prominence on the left became
detectable, outranked the right-limb one, and took its place in the list; the
panel pointing at rank 4 followed it across the disc.

`PROM_LEVEL_MIN` was an earlier fix for exactly this, and it missed: it required 8
frames before a level detected its own set, and the level that caused this has
119. Frame count was never the right axis. A capture is two seconds, its levels
are exposure brackets of one moment, and the prominences do not move between them
- so each capture is detected once now, on the level with the most frames behind
it, and the threshold is deleted rather than retuned.

```
largest in-clip box step   561 px -> 5.3 px
p99 in-clip box step       2.0 px -> 2.0 px   (unchanged)
```

Worth remembering as a measurement lesson: counting how often the layout changed
said everything was fine. The thing the eye called a jump never touched the
layout.

## A resume shard is not a contiguous run (2026-08-19)

Reported as "at about 5 seconds the limb of the Moon takes a step or jitters
backwards", and it was neither the track nor the data. It was a ghost of a
different frame, put there by `--resume`.

`--resume` drops the frames already on disk and hands what is left to the
sharder, which slices it into equal contiguous pieces — contiguous **in the
compacted list**, which is not contiguous in the video. A kill leaves a gap at the
end of every one of the 24 shards, so the survivors are many separate runs: the
last resume had 301 frames in **17 runs**, sliced into 24 shards. Most shards
straddled a join.

The renderer's own discontinuity test then fires on the join — different file, or
an index gap over `GAP_FRAMES` — and starts a cross-dissolve against the
previously rendered frame, which at a join belongs somewhere else in the video.

Measured against a clean single-shard render of the same frames:

```
seq 177   shipped 229   clean 210   +11.7%
seq 178   shipped 225   clean 213    +7.3%
seq 179   shipped 235   clean 235     0.0%
```

A ghost decaying over exactly the dissolve length. A blended-in copy of another
frame's limb is precisely what "steps backwards" looks like.

Boundaries now land at every discontinuity as well as at every step, so a shard is
always a contiguous run. Jobs can outnumber the workers; the pool queues them. A
full render has no discontinuities and is unchanged.

**The measurement lesson is the sharper half of this.** Everything I had been
checking said the video was fine: frame order monotonic, sky pedestal flat to four
decimal places, raw data times gain moving 0.9% across that seam. All true, all
irrelevant — the fault was introduced downstream of every one of them. It took
re-rendering the same frames a second way and diffing the two to see it, and that
comparison is the one to reach for first next time.

### And the fix for it lost the dissolves it should have kept

Making shards contiguous stopped the wrong cross-dissolve. It did not give the
right one back. A shard that starts mid-video has no previous frame, so
`prev_out` is None and a genuine cut at its first frame renders as a **hard cut**
instead of a cross-fade. Measured at third contact against the same frames
rendered in one piece:

```
seq   1861  1862  1863  1864  1865
one piece  237   196   152    98    17     the dissolve
sharded    237    15    14    16    17     a cut
```

True in principle all along — any shard boundary landing on a cut lost its
dissolve — but a full render puts 24 boundaries in predictable places while a
resume puts one per surviving run, in arbitrary ones. Three landed on cuts.

A shard now renders the `dissolve` frames preceding its span as **warm-up**: the
same work, no output. They come from the UNCOMPACTED frame list, because the
frames before a resume span in the compacted one belong to the previous run —
which is the ghost again, arriving by the back door. Three frames per shard,
about 2% of a resume, and verified identical to the one-piece render.

The single-worker path is folded into the same code while fixing this. It
rendered `cfg["frames"]` straight through, which under `--resume` meant straight
through every join: the same ghost, sitting in the path most likely to be used to
reproduce a bug by hand.

**This is the third distinct fault in the same twenty lines**, and the pattern is
worth naming. Each fix was correct about the thing it fixed and silent about a
neighbouring assumption it broke — contiguity fixed the ghost and broke the
dissolve; the dissolve warm-up would have reintroduced the ghost had it read the
compacted list. Sharding a sequential process is not a local change, and the
check that catches it every time is the same one: render the same frames a second
way and diff.

### The segment boundary lands after the exposure change

Genuinely in the data: the raw photosphere jumps **27% at seq 179** (index 940–952
of `14_00_16`). The camera's exposure changes at raw index 948 and the segment
boundary sits at 953, so that segment's last group averaged eight frames at one
level with five at +71% and was then given the pre-change gain — a +10% flash.

Fixed by making the group enforce its own invariant rather than trusting the
boundary: a raw frame whose mean level disagrees with the group's reference by
more than `render.group_level_tol` ends the group. It dropped exactly the five
frames measured, and `seq 178 → 179` went from +10.3% to +0.5%.

The boundary itself is still five frames late. `refine_boundary` binary-searches
for where the brightness crosses the MIDPOINT of the transition, so frames that
have already begun changing stay in the previous segment by construction. Nothing
downstream depends on it any more, but it is worth knowing.

## The exposure proxy was the sky, not the Sun (2026-08-19)

A whole-video scan for brightness steps over 4% found six, and after the resume
ghost and the segment-tail flash were fixed, three were still real: −5.7% and
+9.4% at 12.83/12.90 s, and −12.1% at 13.27 s. All three inside one capture.

`level()`, the exposure proxy behind every segment gain, is the **sky background**
— compared only across boundaries, because background also falls as the eclipse
deepens for reasons that have nothing to do with the camera, and chained from
segment to segment from an anchor. Chaining accumulates whatever error each
comparison carries. Across the four boundaries of `14_00_16` it accumulates
**15.5%**, which is why the level after that boundary sat 8% high even once the
stacking was fixed.

That scheme is not replaced. It is corrected where the correction is unarguable:
**every residual step was inside a capture, and a capture is about two seconds.**
The Sun's photosphere does not change brightness in two seconds, so anything that
moves within one is the camera or the sky, and flattening it is a correction
rather than a falsification. Between captures nothing changes — the chained gains
still set each capture's level, and the cross-dissolve still hides the
minute-long gaps.

### Two things this got wrong first

**Aim at the rendered level, not the raw level.** Correcting the raw photosphere
to a constant cancels the segment gain wherever a capture genuinely contains an
exposure change. `14_06_45` has one, and the first attempt turned its +13.6% step
into **−39.2%** by undoing the gain that was handling it correctly. The target is
level × gain.

**The threshold has to follow the crescent down.** Thresholding at half a fixed
percentile is the photosphere only while the crescent is larger than that
percentile. Below it the threshold lands on sky and the answer silently becomes a
sky level: `14_11_23` measured **0.025 where the answer was 0.73**. Not an error,
just a wrong number, which is worse. The threshold now sits halfway between the
sky and the frame's peak. There is a test asserting the old rule fails on a small
crescent, so the reason for the design sits next to the design.

And where the crescent cannot be measured at all, the correction is **held** from
the nearest frame that could be, never reset to 1 — resetting puts a seam exactly
where measurable meets unmeasurable.

### Measured

```
                          spread within the capture
                          before     after
18 of 20 captures         11-103%     0.0%
14_08_22                   48.9%      5.5%   (59 of 71 measurable)
14_19_28                   37.5%     37.5%   (23 of 71 - crescent blown)
14_13_00, 14_17_51           -          -    left alone, no crescent to measure

seq 384 -> 385   -5.9%  ->  0.0%
seq 386 -> 387  +13.6%  ->  0.0%
seq 397 -> 398  -18.1%  ->  0.0%
```

`14_13_00` and `14_17_51` sit either side of totality, where the crescent is a
saturated sliver; `14_19_28` is third contact re-emerging, where most frames are
clipped in the raw. No gain recovers a clipped crescent, so those are left alone
rather than corrected on a measurement that is not there.

## Still open, and why

**The bead dwell repeats 4.5x.** Arithmetic, not a bug: the small-bead window is
67 raw frames — 2.87 s — and the dwell asks for 10 s, so each frame is held 4 or 5
times, about 6.7 fps. The cadence is already as even as it can be (`5454545…`, 32
fives against 35 fours), so there is nothing to smooth. The only real choices are
a shorter `dwell.beads_s` (4.5 s gives a 2x hold) or interpolating intermediate
frames, which would invent the bead geometry the dwell exists to show.

**The filter-off ramp steps 6.5% at 25.80 s and 4.6% at 30.27 s.** Both are inside
the deliberate resolve, where the gain goes from 4.19 to 27.31 — a few percent
inside a 600% ramp is not worth chasing, and the segment either side is saturated
so there is nothing better to measure against.

**Corona tone mapping.** Starlet multiscale was tried and lost 26–44% of the
baseline gradient. Not pursued.

## Panels that belong to the picture (2026-08-19)

Three reports, three different causes, all of them a constant set too high.

### The panels were nine times darker than the frame

`panel_gain` reduced LINEARLY. To bring a highlight ten times over the ceiling
down to the target it brought the corona around it down ten times too: at seq
1000, corona inside a panel rendered at **24 of 255** against **227** for the same
corona just outside it. The panel stopped reading as a magnifier and started
reading as a different photograph.

The step is now taken in log space, `panels.expose_strength`. Swept on the bead
panel and looked at:

```
strength   1.00   0.75   0.55   0.40   0.25   0.00
median       25     43     67     95    133    226
mean |dx|  2.29   2.56   2.72   2.88   3.07   2.63
clipped       0      0      0      0      0      0   % over 250
```

Two things here would have been got wrong by reasoning instead of measuring.
Local contrast **rises** as the panel brightens, because the display gamma
expands the mid-tones — the full reduction was buying darkness and losing detail.
And **nothing clips at any setting**: the highlight shoulder was already absorbing
the peak, so the linear reduction was not protecting anything. 0.55 is where the
corona reads and the chromosphere is still a crisp coloured line; by 0.25 the rim
blooms.

### Leaders stopped at the Moon and pointed at nothing

The span of a leader inside the disc was skipped, so the line read as passing
BEHIND the Moon — true to the scene, and fatal to the annotation. The lunar-limb
panel points at a feature on the disc's own edge, so nearly all of its leader was
inside the circle; what survived was two strokes ending in blank sky. Being
honest about an occlusion is worth less than being readable, and the Moon is the
one part of the frame with nothing to lose behind a hairline. Kept as
`panels.occlude_leaders` for a subject whose disc carries detail.

### A panel held a corner its subject had walked away from

`CONTINUITY_PX` is a flat 2000 px paid to leave a panel where it is — right while
its subject is near, wrong once it is not. At seq 1705 a prominence on the LEFT
limb was served from the LOWER-RIGHT corner with a 1027 px leader, because the
free `left` slot saved only 679 px against a 2000 px charge to move.

Capped now: a panel keeps its slot unless that costs more than
`panels.continuity_max_extra` px of extra leader.

```
cap           none    600    400    250
leader p90     946    945    887    664
across disc   1831   1794   1540   1222
panels moved     6      5      3      3
```

**250 costs nothing in stability — it moves FEWER panels than no cap at all**,
because a panel dragged out of position drags its neighbours with it.

`SPREAD_WEIGHT` was doing more than stopping bunching. It rewards the tightest
angular gap between the chosen SLOTS, which says nothing about where the subjects
are, so it refused an edge slot standing beside its own subject whenever two
corners were free: `left` between UL and LL scores 37 degrees against 74 for four
corners, a 1292 px penalty at 2000. Re-swept with the cap in place, 500 is the
turning point.

```
worst "a free slot would have saved"   680 px -> 393 px
leader p90                             946 px -> 664 px
panels crossing >0.3R of disc            2285 -> 1238
left slot used                            342 -> 1530 frames
crossing leaders                             0 -> 0
panels that move                             6 -> 3
```

## Why a bigger frame does not fix the edge slots

Asked directly, and worth recording because the answer is a hard limit rather
than a tuning choice. A full-size top panel needs, measured out from the disc
centre:

```
R 292 + clearance 44 + panel 210 + margin 12 = 558 half-res px
the sensor's half-height is                    540
```

It misses by **18 px with the Sun perfectly centred**, before any drift. The disc
is 54% of the plane's short axis, leaving 248 px above and below where a full
panel needs 266. Growing the window does not buy room, it pads black:

| outH | top panel | frames padded |
|---|---|---|
| 900 (now) | 102 px | 25% |
| 1000 | 152 px | 91% |
| 1116 | 210 px (full) | 100% |

Sideways there IS room — the plane is 1920 wide against 1200 used — but widening
makes it worse: the corners move away from the disc so every leader grows by half
the extra width, and padding reaches 28% at 1400.

Eight slots with SMALLER edge panels was tried too. A top panel must be 204 px
against 420 to clear, and the optimizer then puts almost everything on the edges
(bottom 2123, left 1592, top 1463) because they are nearest the limb — so most
panels in the video become the small ones. Shorter leaders bought by shrinking
every panel is not the trade.

**What would unlock it is a smaller disc relative to the sensor** — a shorter
focal length at capture, or more sensor rows. The code already handles that
without changes: `min_clear_r` measures each slot and offers it when it fits, so
a wider field would bring all eight back at full size on its own.

## Caution: the validation tooling has been wrong TWICE — now three times

`pjsr/sharpness.js` reported edge contrast FALLING 5.6% with stacking, and the
images plainly show the stack is better. The metric measured noise in blank sky
and concluded noise contributed 0.003% of the gradient energy at the rim — but the
rim is the brightest part of the frame, so its photon noise is far higher than
sky's. The metric was counting noise-driven gradient along the rim as edge
contrast, and stacking removed it.

Use `pjsr/crop-export.js` and LOOK at the result. It exports a crop centred on the
brightest structure with a shared normalisation across files, so brightness and
sharpness differences between them are real.

## Caution: the validation tooling has been wrong twice

Do not trust automated limb measurements without checking them against a frame
whose answer is known independently.

- An early check compared the model against its own detections. Both carried the
  same bias, so it reported 1–4 px error when the truth was 173 px.
- A later check thresholded on brightness, and the glow around the disc inflated
  the radius asymmetrically — it reported offsets of −86 and −71 px where a
  three-point circle through the outer limb gives about −31 and −11.

Fit the limb; do not threshold the brightness.

## The fix worth doing next

Chain absolute placement across captures instead of fitting each independently.
Register each capture's reference frame to the previous capture's reference frame
**on the Sun's limb alone**, and let the per-capture intercept follow that chain.
That removes capture-dependent bias by construction, which is exactly what the
residual 10–30 px is. This is the legitimate form of "stabilize each clip" for this
data. Totality is the natural break in the chain, and a small step there is
expected.

## Also still open

- **Field rotation** — uncorrected. Bounded at ~1° total, ~12 px at the frame
  corner, under a pixel within totality (so the corona stacks and HDR are
  unaffected). Three measurement attempts failed: long-baseline sunspot
  correlation (no overlap across totality), Moon-path straightness (topocentric
  parallax curves it ~15 px against a ~6 px signal), adjacent-capture correlation
  (dominated by the moving lunar limb, returned an impossible +146°/hr).
- **HDR corona** — geometry correct, tone mapping not. Needs an asinh or MTF
  stretch; the preview's rescale-and-gamma crushes everything between the
  chromosphere at ~1.0 and the outer corona at ~0.001.
- **Render speed — the old "CPU-bound, 1035 MB/s" note was WRONG.** `S:` is a
  `TOSHIBA HDWR11A`, a **spinning SATA HDD**, measured at **117 MB/s** single-stream
  sequential. No SATA HDD does 1035 MB/s; that figure must have come from a cached
  read, and it steered this project away from I/O for months. Beware measuring disk
  throughput on this machine at all — it has 192 GB of RAM, so the page cache
  swallows whole 20 GB captures and a naive benchmark returns 2657 MB/s.

  It matters now that every frame drizzles: a render reads **551 GB** (1660 frames
  x 20 raw frames x 16.6 MB). At 117 MB/s that is a 78 min floor even if the reads
  were perfectly sequential, and they are not — with 12 workers the disk delivered
  only **48 MB/s** at queue depth 3.8, because twelve readers turn streaming into
  seeking on one head. Concurrency costs about 2.4x here.

  Changed 2026-08-15: frame output and the PixInsight swap moved to `D:` (NVMe —
  `gen_timelapse.py --frames-dir` now defaults there, and `run-timelapse.mjs`
  finally parses `--swap`, which it had accepted in its defaults but never read),
  and workers dropped from 12 to 4 so the reads stay closer to sequential.
  **Result: 122 min for a fully stacked render**, and 4 workers beating 12 is now
  MEASURED, not assumed. Two disjoint 200-frame blocks of early partials, so
  neither run could be served from the other's page cache:

  | workers | data | time | rate | effective read |
  |---|---|---|---|---|
  | 12 | 63.4 GB | 21.7 min | 9.2 f/min | 48.7 MB/s |
  | 4 | 63.3 GB | 14.9 min | 13.4 f/min | **70.9 MB/s** |

  **4 workers is 1.46x faster** — about an hour saved on a full render. Note the
  read rate FALLS as workers rise: the mechanism is contention for one head, not
  CPU. Both are still under the 117 MB/s single-stream figure, so even 4 costs some
  sequentiality and 2 may be better again; untested, and CPU would eventually bind.

  The 12-worker default was correct when only totality was stacked and the job
  really was CPU-bound. It became wrong the moment every frame started reading 20
  raw frames. Re-measure this after any change to STACK_MAX or to how much of the
  video is stacked.

  The encode is the evidence the stacking is real: same CRF, **99.0 MB -> 39.1 MB**.
  A 2.5x bitrate drop at identical settings is noise leaving the picture. All the other large volumes (`E:`, `G:`, `H:`) are HDDs too; the only
  SSDs are `C:`/`D:`/`Z:` with ~208 GB free between them against 449 GB of SER, so
  the source data cannot simply move.

  Two further levers if it is still too slow: stage each capture onto `D:` before
  rendering it and delete after (turns 12 seeking streams into one sequential HDD
  read plus NVMe random reads; ~20 GB per capture, several fit at once), or drop
  `STACK_MAX` to 10 for the partials, which halves both the I/O and the CPU for a
  root-2 cost in noise (3.2x reduction instead of 4.5x).

## PixInsight is gone; the pipeline is Python (2026-08-15 to 08-18)

Every stage now runs under `pipeline/.venv` against `ecl/`, with `lunation`
(`D:/projects/lunation`) installed editable as the numeric spine. PixInsight was
only ever an FFT, a bicubic resample and file I/O here — there is not one
`executeOn` in the old PJSR — so the port is a like-for-like swap with two
measured improvements. `pjsr/` and the `run-*.mjs` drivers are still on disk and
still work; nothing has been deleted.

The stacker is better than the one it replaces: same 35/87 frames, same reference
frame, r=0.9993, and a **23% sharper limb** (10–90 width 2.00 px against 2.60)
with less undershoot. That is registration accuracy — upsampled-DFT at ~0.01 px
against PI's 3×3 peak at ~0.5 px — not the interpolation kernel, which moves
detail by 0.9%.

**A real colour bug fell out of it.** `ser_to_fits.debayer_rggb` swapped R and B
at green sites — half of every frame. Every FITS exported before 2026-08-15 has
bad colour. It is now a one-line delegation to `ecl.demosaic.bilinear_rggb`,
which derives neighbour offsets from the CFA table and is tested against a
synthetic mosaic in all four Bayer orders.

## Totality is now four explicit dwells, not one cadence

Totality was 24.4 s and spent in the wrong places: 12.0 s on the prominence
level, 8.0 s on the resolve, and **4.5 s on the corona itself** — while 2,583 raw
frames of corona sat unused, because the light curve samples one frame in twenty
and the corona segments were emitted at that cadence. The classic totality view
was the shortest thing in totality.

| phase | screen time | source | factor |
|---|---|---|---|
| resolve (filter off → ring fades) | 9.0 s | 239 raw | 1.1x |
| Baily's beads | 10.0 s | 67 raw | 4.5x |
| prominence level | 12.0 s | 144 raw | 2.5x |
| corona | 9.7 s | 291 raw | **1.0x, no repeats** |

The corona dwell has enough footage that every video frame is a different
exposure. The bead dwell does not — 2.87 s of beads were recorded — so it repeats
each frame ~4.5x and visibly steps. That is the footage limit, not a setting;
`BEADS_S` trades dwell length against stutter and nothing else will.

## The bead window has to be MEASURED, and saturation cannot do it

The diamond ring and the beads are both clipped photosphere and the clipped area
falls smoothly through both — 56,453 px at f1090 to 5 px at f1300 with no step
anywhere. satfrac reads 0.00025 at f1160 and 0.00018 at f1180, on either side of
the moment the ring breaks up. Splitting there put the whole dwell on the ring.

**Shape separates them.** The ring is ONE connected blob; beads are several with
lunar ridges between, so the discriminator is the fraction of clipped area in the
largest connected component:

    raw    area  blobs  largest  largest%
    1090  56453      1    56413      100    one solid blob — diamond ring
    1168    419      3      391       93
    1204    136      2       72       53    broken up — beads
    1240     69      2       18       26    small beads, ridges between
    1300      5      0        1       20    gone

`ecl/beadwindow.py` measures this and writes `diag/beads.json`. Connected
components need no disc centre, which is what lets it run BEFORE `tl_centres` —
and it must, because `gen_timelapse` decides screen time and runs first of all.
Measured window: **14_13_00 f1189–1255**.

Two things this forces. The window sits INSIDE the prominence level, nowhere near
the resolve that ends at f1170 — nothing about the exposure changes there, so no
segment boundary marks it and none ever will; screen time is therefore allocated
to raw frame ranges, with each segment split into before/window/after. And those
frames would otherwise inherit the prominence level's overlapping 20-frame
drizzle groups, averaging the beads into the arc the dwell exists to show
breaking up, so `bead` frames are forced to `stack: 1`.

## Panels: six slots, exposed for themselves, and leaders behind the disc

**Six slots**, not four: the corners plus the LEFT and RIGHT edge midpoints.
There is no top or bottom midpoint — the frame is 2240×1680 and the disc 1168
across, so there is 536 px of clearance beside it and only 256 above and below
against a 420 px panel. Rendered, those two sat 164 px over the disc and one
covered the bead chain it pointed at. The two that fit shorten leaders by 8% at
the median and **26% at p90**, which is the awkward-leader case.

**Leaders pass behind the disc.** No corner assignment can avoid crossing it:
every frame with two or more panels has NO crossing-free option, because the
features sit ON the limb while the slots are outside it. The worst leader drew
1123 px of line across a disc 1168 px wide. `_Canvas.line` now skips the span
inside `cfg["discR"]`.

**Each panel is exposed for its own subject**, never brighter than the frame
gain, measured across all three channels so a panel is not white-balanced by
accident. Panels used to inherit a gain chosen for the corona — 27x around the
contacts while pointed at photosphere — and rendered as flat white squares.

**The bead box is centred on the config's disc centre, not `moon_of`.** Through
totality the detector measures the Moon directly, so cx/cy IS the Moon;
`moon_of` adds the terminator track, fitted on ten partial-phase frames at 7.2 px
residual and extrapolated, and sits a steady 12 px off. The annulus is only
0.12 r_moon thick, so 12 px puts it inside the limb on one side and outside on
the other, and as the chain shortened the misplaced ring clipped part of it out of
the centroid: the box walked 640 → 607 while the beads held at 643 → 649. One box
size and a degree-2 track for the whole run, as the cusps already do — sizing per
frame made the zoom pump 1.167 → 2.100.

## Worker count: the cliff was two bugs, not the hardware

The old default of 4 came from a measurement where 12 workers ran twenty times
slower than one. That was a spinning disc having its readahead shredded AND
`scipy.fft` unpinned, so every worker fanned its FFT across all 32 cores. Both
are fixed. Re-measured on an i9-14900K (24 physical cores, 192 GB), captures
staged on Z::

    workers      8      12      16      20      24      32
    s/frame  1.235   1.029   0.891   0.820   0.780   0.749

Monotonic throughout. The I/O-heavy case was checked separately on the corona
dwell — 291 distinct raw frames, nothing re-read, halves swapped between arms —
at 1.305 s/frame against 0.796, and cold-against-cold alone is 1.71x. `--workers`
now defaults to the physical core count. A full 2393-frame render went 50.9 min
to **26.8 min**. RE-MEASURE ON A SPINNING DISC before trusting this elsewhere.

Pinning workers to performance cores (`--affinity`, `ecl/affinity.py`) was
measured at **+0.6%** and is not worth using — Windows already places them well,
and the flag declines above 8 workers anyway.

## Two tools worth knowing about

`python -m ecl.progress --frames <dir> [--watch]` reads the frames on disk rather
than the job's output, so it works on a render started any way and survives
restarts. Rate is measured over the last 120 frames, not since the start, and the
ETA follows the slowest shard divided among shards STILL RUNNING — the bead dwell
is all `stack=1` and finishes early, and counting those idle workers as active
put the ETA out by 40%.

`tl_render --resume` skips frames already written as COMPLETE PNGs (checked by
the IEND marker, one seek each), so a killed render continues instead of starting
over. Used once already: 2055 skipped, 338 rendered in 4.8 min against 28 for a
restart.

## Still open

- **The step into the resolve is 6.24x** on one frame — the filter physically
  coming off, a 15x step in the footage that the gain-jump dissolve softens. It is
  the largest brightness event in the video.
- **Post-bead alignment.** Reported as possibly wrong; could not be reproduced.
  Frame placement through the dense block steps 0.016 px median, and the drizzle
  groups run at full depth with ZERO rejections at ~1 px against a 4 px bound.
  Both mechanisms measure healthy, so if it is visible it is something else.
- **Corona tone mapping** — still the open item from 2026-08-13. Starlet
  multiscale was tried and failed: four bias schedules all landed at 56–74% of
  baseline `|grad|`. sd and spread improved, which is why it looked promising;
  `|grad|` is the metric that answers "is structure missing" and it fell 44%.

## Current outputs

Three cuts from one render, via `ecl/encode.py`. 2393 frames, 79.8 s at 30 fps,
all 22 captures.

- `timelapse.mp4` — 2240×1680, CRF 17, ~126 MB.
- `timelapse_instagram.mp4` — **1080×810**, High@4.0, capped bitrate, closed 2 s
  GOP, ~19 MB. Sized so Instagram does not re-encode it: anything wider than 1080
  goes through their scaler, which is much worse than ffmpeg's, and thin white
  leader lines on black sky are exactly the content that shows it. 4:3 sits inside
  the 4:5–1.91:1 range the feed accepts, so it uploads whole rather than cropped.
- `timelapse_preview.mp4` — 960 wide, CRF 26, ~3 MB.
- `out/final/corona_hdr.xisf`, `corona_flat.xisf` — 3631×2037 merged corona.
- `out/beads/` — 7 contact windows; the C2 diamond ring is in `14_13_00_f1051`.

Working copies are on `Z:/eclipse-work/` (frames `tl_py/`, videos `final/`);
`out/final/` still holds the 2026-08-15 videos.

## Rebuild commands

The whole thing is one command now, and it starts from zero:

```bash
python -m ecl.run D:/eclipse/data
```

The per-pass form, for re-running one stage. Note that every pass is a MODULE:
`gen_timelapse.py`, `smooth_track.py`, `gen_insets.py` and `serlib.py` used to
sit beside the package as loose scripts and are now `ecl.gen_timelapse`,
`ecl.smooth_track`, `ecl.gen_insets` and `ecl.serlib`. `scan_ser.py` is gone
entirely, replaced by `ecl.segment`, which does the same job through
`ecl.source` and therefore works on image sequences as well as SER.

```bash
PY="D:/projects/solar-eclipse-timelapse/.venv/Scripts/python.exe"
OUT="S:/solar-eclipse/out"
DATA="Z:/solar-eclipse/Sun"          # staged copy; S: is a spinning disc
cd D:/projects/solar-eclipse-timelapse

"$PY" -m ecl.segment    --out $OUT --data $DATA        # segments.json
"$PY" -m ecl.beadwindow --out $OUT --data $DATA        # diag/beads.json
"$PY" -m ecl.gen_timelapse --out $OUT                  # frames + gains + dwells
"$PY" -m ecl.tl_centres --data-dir $DATA --out $OUT/diag/centres.json
"$PY" -m ecl.tl_track   --data-dir $DATA --out $OUT/diag/corona_track.json
"$PY" -m ecl.smooth_track --out $OUT --window 1120x840 --drop-padded --require-disc
"$PY" -m ecl.gen_insets --out $OUT --data $DATA --zoom 4.0
"$PY" -m ecl.tl_render --data-dir $DATA --out-dir Z:/eclipse-work/tl_py
"$PY" -m ecl.encode --frames Z:/eclipse-work/tl_py --out-dir Z:/eclipse-work/final
```

**`out/configs/timelapse.json` is the artifact that feeds the render, and it is
written by these passes in order.** Reverting sources alone does not change a
video until the pipeline is re-run — and conversely a good config survives a
source revert, which is why a render can be correct while the code that produced
it is not. If in doubt about what a video contains, read the config, not the
scripts.

**The passes are strictly ordered and each rewrites `configs/timelapse.json` in
place.** Start from `gen_timelapse` every time. Re-running a later pass against a
config an earlier one already rewrote is the single most common way to break this
— it has happened twice: once running `smooth_track` over its own output, once
with two chains racing each other on the same file.

**`tl_centres` is not optional when frames were added.** `smooth_track.py` joins
detections on (file, index), so a frame the last detection pass never saw has no
centre and is dropped. It prints a warning — believe it.

`--workers` now defaults to the physical core count; see the note above before
changing it on other storage.
