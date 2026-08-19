"""Runtime parameters: scale-free defaults, an editable file, and a resolver.

Every geometric constant in this pipeline was a pixel count measured on one
camera. `PROM_MIN_SEP_PX = 90` means "about a third of a solar radius" and
nothing else - on a full-frame body at a different focal length it is either the
whole disc or a rounding error. The same is true of the panel box sizes, the
alignment shift bound, the trust thresholds in the track fit and the bead areas.

So constants live here as FRACTIONS OF THE SOLAR RADIUS, and are turned into
pixels against the radius the survey measured. Lengths scale as r, areas as r^2,
and anything genuinely dimensionless (gammas, gains, ratios, seconds) is carried
through unchanged.

Precedence, lowest to highest:

    DEFAULTS below  ->  eclipse.toml in the output dir  ->  explicit overrides

The file is written on first run with every value in it and a comment on each,
so tuning is editing a documented file rather than reading source. It is never
overwritten afterwards: a run that silently reset the user's settings would be
worse than one that failed.

    from ecl.params import load
    P = load(out_dir)
    P["render"]["drizzle"]        # resolved value
    P.px("panels.bead_half_r")    # fraction of R -> pixels
"""

import copy
import json
import os

__all__ = ["DEFAULTS", "Params", "load", "write_template"]

# ---------------------------------------------------------------------------
# Defaults. Keys ending _r are fractions of the solar RADIUS; _r2 are fractions
# of its squared area; _s are seconds; everything else is dimensionless.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "geometry": {
        # 0 = use the surveyed radius. Set this to force it.
        "radius_plane_px": 0.0,
        # Output window HALF-width in solar radii: the window comes out
        # 2 x this x radius across. 2.2 holds the corona to where this
        # dataset's signal ends. It shrinks toward min_half_r if too much of
        # the sequence would need padding.
        "output_half_r": 2.2,
        "min_half_r": 1.6,
        "max_padded_fraction": 0.35,
        # Width over height. 4:3 rather than 16:9 because the subject is round:
        # a wide window buys corona at the sides that a short one loses above and
        # below, and the corona does not care which way it is measured. Every
        # render before this was made with an explicit 1120x840, which is 4:3;
        # the auto-sizer had 16/9 hardcoded and silently produced a window
        # 2.19 R wide by 1.22 R tall instead of 2.01 by 1.51.
        "aspect": 1.3333333,
        # Window dimensions are rounded to a multiple of this many pixels.
        "size_step": 20,
    },
    "render": {
        "drizzle": 0,          # 0 = surveyed
        "workers": 0,          # 0 = surveyed
        "gamma": 0.65,
        "shoulder_knee": 0.60,
        "shoulder_ceil": 0.965,
        "pedestal_pct": 1.0,
        "pedestal_frac": 1.0,
        "gain_jump": 3.0,
        "gap_frames": 60,
        "dissolve": 3,
        # Fractional change in a raw frame's mean level that drops it from its
        # stacking group. Guards against a segment boundary that lands a few
        # frames after the exposure actually changed. Within a group the level
        # moves 0.5%; the smallest exposure step between segments here is 39%.
        "group_level_tol": 0.08,
        # Input level the highlight curve maps to the ceiling. The old tanh
        # shoulder ran out at v=1.57 while the totality limb reaches v=23.5, so
        # 8% of it rendered as one flat grey. The corona is at v=0.03, far below
        # the knee, so widening this cannot darken it.
        "shoulder_max": 24.0,
        # Largest intra-group alignment shift believed, in radii. 4 px at r=279.
        "max_group_shift_r": 0.0143,
        "engine": "ported",
        "fps": 30,
    },
    "panels": {
        # Panel edge as a fraction of the output window's SHORT side; 420 px in
        # a 1680 px tall frame.
        "size_frac": 0.25,
        "zoom": 4.0,
        # Smallest gap between a panel and the disc edge, in radii. A slot that
        # cannot make this gap is not offered at all, which is how the top and
        # bottom slots come and go: in a 4:3 window with the disc at 2.2 R there
        # is 616 px clear at the sides and 316 above, so a 420 px panel stands
        # off the sides and would sit ON the limb top and bottom. A wider frame,
        # or a smaller disc in it, gets all eight.
        "min_clear_r": 0.15,
        # Leader length, in px, worth paying per px of extra clearance. This is
        # what "favour the corners" means numerically - a corner is simply the
        # roomiest slot in any rectangular frame, so preferring room prefers
        # corners without naming them, and still does the right thing on a frame
        # shape where that stops being true.
        "clear_weight": 2.0,
        # Leader length, in px, worth paying to leave a panel where it already
        # is. Panels moving mid-video reads as chaos however good each individual
        # arrangement is, so this is deliberately far larger than any leader.
        "continuity_px": 2000.0,
        # Extra leader, in px, that keeping a panel in place may cost before it
        # is made to move. Without a cap a panel keeps a corner its subject has
        # walked away from.
        "continuity_max_extra": 250.0,
        # Leader px worth paying per radian of extra spacing between panels.
        "spread_weight": 500.0,
        "expose": True,
        "expose_pct": 99.5,
        "expose_target": 0.85,
        # How far a panel is exposed toward its own subject, 0 to 1, in log
        # space. 1.0 exposes fully and renders the corona inside a panel far
        # darker than the same corona outside it; 0.0 keeps the frame's exposure
        # and blows the subject out.
        "expose_strength": 0.55,
        # Hide the part of a leader that crosses the Moon. Off: the Moon is dark,
        # nothing is lost behind the line, and the lunar-limb leader is almost
        # entirely inside the disc - occluded, it points at nothing.
        "occlude_leaders": False,
        # Feature box half-widths, in radii (was 40-46 and 45-80 px at r=279).
        "cusp_half_r": [0.143, 0.165],
        "bead_half_r": [0.161, 0.287],
        "prom_sep_r": 0.323,
        "prom_r_inner": 0.90,
        "prom_r_outer": 1.25,
        "prom_min_snr": 8.0,
        # A prominence pick sitting on a long run of chromosphere is boxed to the
        # whole run rather than to a fixed size, and picks sharing a run merge
        # into one panel. Measured, not assumed: at seq 1200 the arc ran 65 deg
        # and an isolated prominence 5 deg.
        "arc_snr": 5.0,
        "arc_gap_deg": 3.0,
        "arc_min_deg": 12.0,
        "arc_pad_r": 0.06,
        "arc_min_zoom": 1.5,
        # Share of the smaller source box that may lie inside another before the
        # two panels are treated as showing the same thing.
        "overlap_max": 0.35,
        "spot_max_r": 0.90,
        "spot_ring_r": 0.072,
        "moon_fit_max_rms_r": 0.043,
    },
    "beads": {
        # Clipped area worth calling a bead, as a fraction of r^2. 40 px at
        # r=279 is 40/279^2.
        "min_area_r2": 5.14e-4,
        "min_blob_r2": 1.03e-4,
        "sat": 0.90,
        "max_blob_frac": 0.70,
        "max_thick_r": 0.0215,
        "arc_max_deg": 90.0,
        "arc_min_deg": 1.5,
        "annulus": [0.93, 1.05],
        "near_frames": 10,
        "run_gap": 4,
    },
    "dwell": {
        "resolve_s": 9.0,
        "beads_s": 10.0,
        "prominence_s": 12.0,
        "corona_s": 10.0,
    },
    "track": {
        # All in radii; the px values were measured at r=279.
        "trust_floor_r": 0.0287,
        "max_residual_r": 0.0896,
        "unplaceable_spread_r": 0.215,
        "follow_excursion_r": 0.0538,
        "follow_step_r": 0.0287,
        "follow_line_tol_r": 0.143,
        "thrash_r": 0.0538,
        "level_bias_max_r": 0.043,
        "trust_k": 6.0,
        "corona_smooth_s": 1.5,
    },
    "select": {
        "target_filtered": 0.88,
        "target_unfiltered": 0.78,
        "max_satfrac_filtered": 0.01,
        "max_satfrac_unfiltered": 0.02,
        "transition_ceiling": 1.15,
        "stack_max": 20,
        # Smallest crescent, in r^2, whose level can be told from the sky well
        # enough to normalize a frame on. 2000 px at r=292.
        "min_crescent_r2": 0.0235,
        # Above this the crescent is clipped in the raw data and its level says
        # nothing about the exposure.
        "crescent_sat": 0.97,
        # Largest rescale the within-capture flattening may apply. Beyond this
        # the measurement is wrong rather than the gain.
        "flatten_max": 2.0,
    },
}

_COMMENTS = {
    "geometry.radius_plane_px": "0 = use the radius measured by the survey",
    "geometry.output_half_r": "output window half-width in solar radii (window = 2 x this x radius)",
    "geometry.min_half_r": "never shrink the window below this half-width",
    "geometry.aspect": "output width / height; 1.3333 = 4:3, 1.7778 = 16:9",
    "render.drizzle": "0 = auto (aims for a 500 px fine-grid disc radius)",
    "render.workers": "0 = auto (physical cores, capped by free memory)",
    "render.max_group_shift_r": "largest believable intra-group shift, in radii",
    "render.engine": "'ported' or 'skimage' phase correlator",
    "render.group_level_tol": "level change that drops a raw frame from its stacking group",
    "render.shoulder_max": "input level mapped to the highlight ceiling",
    "select.min_crescent_r2": "smallest crescent, in r^2, that can be normalized on",
    "select.crescent_sat": "crescent level above which the raw data is clipped",
    "select.flatten_max": "largest within-capture brightness correction allowed",
    "panels.size_frac": "panel edge as a fraction of the output short side",
    "panels.min_clear_r": "smallest panel-to-disc gap, in radii; slots tighter than this are dropped",
    "panels.expose_strength": "0 = frame exposure, 1 = fully exposed for the panel subject",
    "panels.occlude_leaders": "hide the span of a leader that crosses the Moon",
    "panels.clear_weight": "leader px paid per px of extra clearance - the corner preference",
    "panels.continuity_px": "leader px paid to keep a panel in the slot it already holds",
    "panels.continuity_max_extra": "extra leader px that keeping a panel may cost",
    "panels.spread_weight": "leader px paid per radian of spacing between panels",
    "panels.cusp_half_r": "[min, max] cusp box half-width, in radii",
    "panels.bead_half_r": "[min, max] bead box half-width, in radii",
    "panels.prom_sep_r": "minimum separation between prominence picks, in radii",
    "panels.arc_snr": "sigmas over the annulus at which the limb counts as lit",
    "panels.arc_min_deg": "shortest run that is an arc rather than a point",
    "panels.arc_min_zoom": "least magnification a panel may drop to holding an arc",
    "beads.min_area_r2": "smallest clipped area called a bead, as a fraction of r^2",
    "beads.max_blob_frac": "above this the clipped region is one blob: diamond ring, not beads",
    "dwell.resolve_s": "screen seconds for the filter-off sequence",
    "dwell.beads_s": "screen seconds held on Baily's beads",
    "dwell.prominence_s": "screen seconds on the short prominence exposure",
    "dwell.corona_s": "screen seconds on the corona proper",
    "track.trust_floor_r": "detection/model disagreement tolerated, in radii",
    "select.max_satfrac_unfiltered": "clipped fraction above which a totality frame is dropped",
}

CONFIG_NAME = "eclipse.toml"


class Params:
    """Resolved parameters. Index by section, or by dotted path via px()/get()."""

    def __init__(self, data, radius_px):
        self._d = data
        self.radius_px = float(radius_px)

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def get(self, path, default=None):
        cur = self._d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def px(self, path, default=None):
        """A radius fraction as pixels. Lists map elementwise."""
        v = self.get(path, default)
        if isinstance(v, (list, tuple)):
            return [x * self.radius_px for x in v]
        return v * self.radius_px

    def area(self, path, default=None):
        """An r^2 fraction as pixels squared."""
        return self.get(path, default) * self.radius_px ** 2

    def as_dict(self):
        return copy.deepcopy(self._d)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_template(path, data=None):
    """Write a fully-populated, commented config. Never overwrites."""
    if os.path.exists(path):
        return False
    data = data or DEFAULTS
    lines = [
        "# Eclipse pipeline configuration.",
        "#",
        "# Written once, then yours to edit - re-running never overwrites it.",
        "# Keys ending _r are fractions of the SOLAR RADIUS and _r2 of its square,",
        "# so they carry across sensors and focal lengths unchanged. Keys ending",
        "# _s are seconds. Delete any key to fall back to the built-in default.",
        "",
    ]
    for sect, vals in data.items():
        lines.append(f"[{sect}]")
        for k, v in vals.items():
            c = _COMMENTS.get(f"{sect}.{k}")
            if c:
                lines.append(f"# {c}")
            lines.append(f"{k} = {_fmt(v)}")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def read_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        try:
            import tomllib
            return tomllib.load(f)
        except ImportError:
            import tomli
            return tomli.load(f)


def load(out_dir, survey_data=None, overrides=None, create=True, log=None):
    """DEFAULTS <- eclipse.toml <- overrides, resolved against the survey."""
    path = os.path.join(out_dir, CONFIG_NAME)
    if create and write_template(path) and log:
        log(f"wrote {path} - edit it to tune; it will not be overwritten")
    data = _merge(DEFAULTS, read_config(path))
    data = _merge(data, overrides)

    if survey_data is None:
        sp = os.path.join(out_dir, "survey.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                survey_data = json.load(f)

    r = data["geometry"].get("radius_plane_px") or 0.0
    if not r and survey_data:
        r = survey_data.get("radius_plane_px", 0.0)
    if not r:
        raise SystemExit(
            "no solar radius: run `python -m ecl.survey <data-dir>` first, or "
            f"set geometry.radius_plane_px in {path}")

    # Anything left at 0 in [render] means "take the surveyed value".
    if survey_data:
        rt = survey_data.get("runtime", {})
        for k in ("drizzle", "workers"):
            if not data["render"].get(k):
                data["render"][k] = rt.get(k) or 0

    return Params(data, r)
