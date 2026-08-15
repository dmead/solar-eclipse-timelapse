#!/usr/bin/env node
/*
 * gen-eclipse-config.mjs — turn Stage A's measured segments into a run config.
 *
 * Mirrors pix-planetary's gen-configs.mjs convention: the config states facts
 * about the data (which frames, which file, which exposure level) and carries no
 * hand-tuned per-segment overrides. Stacking parameters come from role defaults
 * only, so re-running Stage A on different data produces a valid config with no
 * edits.
 *
 *   node scripts/gen-eclipse-config.mjs [--segments <segments.json>] [--out <dir>]
 */

import fs from 'fs';
import path from 'path';

const OUT_DIR = 'S:/solar-eclipse/out';

// Role defaults. Corona keeps far more frames than a lunar stack: the Laplacian
// sharpness metric measures mostly noise on a smooth corona, so rejecting 90% of
// frames throws away signal rather than blur. LocalWarp stays off — it assumes a
// rigid surface, and warping streamers is how you invent detail that was not there.
const ROLE_DEFAULTS = {
  corona: {
    bestFraction: 0.40, minFrames: 25, maxFrames: 600,
    alignOnGradient: false, localAlign: false,
    drizzle: 2, drizzleMargin: 16,
  },
  partial: {
    bestFraction: 0.10, minFrames: 25, maxFrames: 500,
    alignOnGradient: false, localAlign: false,
    drizzle: 2, drizzleMargin: 16,
  },
};

// A stack needs enough frames to be worth the I/O; below this a segment is still
// reported (and still usable for single-frame stills) but is not queued.
const MIN_STACKABLE_FRAMES = 60;

/*
 * Largest saturated fraction a genuine corona exposure can have.
 *
 * During totality the only things that clip are the chromosphere and the
 * prominences, which occupy a thin ring at the limb: at the ~576 px Moon radius
 * of this session a generous 20 px annulus is 2*pi*576*20 / (3840*2160), about
 * 0.9% of the frame. Much more than that means saturation has spread well beyond
 * any plausible chromospheric ring, which happens in exactly two ways - the
 * segment predates second contact and still contains photosphere, or the exposure
 * is long enough to clip the whole inner corona.
 *
 * Both wreck the merge. A pre-C2 segment contributes a huge blown blob offset
 * from the Moon (it put a second dark lobe in the first HDR), and neither kind
 * has a usable limb to register on or an unclipped overlap to fit an exposure
 * ratio against. Measured here: the four good levels sit at 0.004-0.41%, the two
 * bad ones at 8.7% and 17.3%.
 */
const MAX_CORONA_SATFRAC = 0.02;

// Longest span allowed in a single corona stack. The Moon crosses the corona at
// ~0.27 px/s at this image scale, and a stack registers rigidly, so a 60 s stack
// smears the Moon against the corona by ~16 px whichever of the two the
// alignment locks onto. Splitting caps that residual; the chunks are recombined
// afterwards with the measured drift taken out.
const MAX_CORONA_STACK_S = 15;

function parseArgs(argv) {
  const args = {
    segments: `${OUT_DIR}/segments.json`, out: OUT_DIR,
    includePartials: false, maxStackSeconds: MAX_CORONA_STACK_S,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--segments') args.segments = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else if (argv[i] === '--include-partials') args.includePartials = true;
    else if (argv[i] === '--max-stack-seconds') args.maxStackSeconds = parseFloat(argv[++i]);
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

/* Split a frame range into as few equal chunks as keep each within maxSeconds. */
function chunkRange(start, count, fps, maxSeconds) {
  const parts = Math.max(1, Math.ceil(count / fps / maxSeconds));
  const per = Math.floor(count / parts);
  const out = [];
  for (let k = 0; k < parts; k++) {
    const s = start + k * per;
    out.push({ start: s, count: k === parts - 1 ? start + count - s : per });
  }
  return out;
}

function main() {
  const args = parseArgs(process.argv);
  const man = JSON.parse(fs.readFileSync(args.segments, 'utf8'));
  const outDir = args.out.replace(/\\/g, '/');

  const segments = [];
  const beads = [];
  const overexposed = [];
  let skipped = 0;

  // Capture geometry is uniform across a session; the runner uses it to tell an
  // up-to-date slice from one cut for a different frame range.
  const g0 = man.files[0];
  const geometry = {
    width: g0.width, height: g0.height, depth: g0.depth, color_id: g0.color_id,
  };

  for (const f of man.files) {
    const base = f.name.replace(/\.ser$/i, '');
    for (const s of f.segments) {
      const tag = `${base}_f${s.start}`;

      // Frames where the exposure is being ridden, and the blown-out frames at
      // the filter change, belong to no stable state. They are useless to a stack
      // and are exactly where the diamond ring and Baily's beads live.
      if (s.state === 'unfiltered' && s.kind !== 'stable') {
        beads.push({
          id: tag, src: f.path, start: s.start, count: s.count,
          kind: s.kind, seconds: +s.seconds.toFixed(2), med: s.med,
        });
        continue;
      }
      if (s.kind !== 'stable') continue;

      const role = s.state === 'unfiltered' ? 'corona' : 'partial';
      if (role === 'partial' && !args.includePartials) continue;
      if (s.count < MIN_STACKABLE_FRAMES) { skipped++; continue; }
      if (role === 'corona' && s.satfrac > MAX_CORONA_SATFRAC) {
        overexposed.push({ id: tag, satfrac: s.satfrac, med: s.med });
        continue;
      }

      // Python isoformat carries no zone; the times are UTC and only differences
      // are ever used, so pin it explicitly rather than let Date assume local.
      const fileT0 = Date.parse(`${f.t0_utc}Z`) / 1000;
      const chunks = role === 'corona'
        ? chunkRange(s.start, s.count, f.fps, args.maxStackSeconds)
        : [{ start: s.start, count: s.count }];

      chunks.forEach((c, ci) => {
        const id = chunks.length > 1 ? `${base}_f${c.start}` : tag;
        segments.push({
          id,
          src: f.path,
          start: c.start,
          count: c.count,
          role,
          level: s.level ?? null,
          level_name: s.level_name ?? null,
          seconds: +(c.count / f.fps).toFixed(2),
          // Midpoint of the chunk, in seconds — the time base the drift
          // correction interpolates against.
          t_mid: +(fileT0 + (c.start + c.count / 2) / f.fps).toFixed(3),
          chunk: chunks.length > 1 ? ci : null,
          of: chunks.length > 1 ? chunks.length : null,
          med: s.med,
          p99: s.p99,
          slice: `${outDir}/slices/${id}.ser`,
          ...ROLE_DEFAULTS[role],
        });
      });
    }
  }

  const cfg = {
    name: '2024-04-08_eclipse',
    outDir,
    sliceDir: `${outDir}/slices`,
    stackDir: `${outDir}/stacks`,
    swapDir: 'S:/solar-eclipse/swap',
    geometry,
    // OSC captures are processed as colour via CFA channel extraction — one
    // stack per channel off the same slice, combined in Stage D.
    channels: ['R', 'G', 'B'],
    concurrency: 2,
    segments,
    beads,
  };

  fs.mkdirSync(path.join(outDir, 'configs'), { recursive: true });
  const dest = path.join(outDir, 'configs', 'eclipse.json').replace(/\\/g, '/');
  fs.writeFileSync(dest, JSON.stringify(cfg, null, 1));

  const byRole = {};
  for (const s of segments) byRole[s.role] = (byRole[s.role] || 0) + 1;
  console.log(`${dest}`);
  console.log(`  segments: ${JSON.stringify(byRole)}  beads windows: ${beads.length}`
    + (skipped ? `  (skipped ${skipped} short)` : ''));
  for (const o of overexposed)
    console.log(`  rejected ${o.id}: ${(o.satfrac * 100).toFixed(2)}% saturated `
      + `(limit ${(MAX_CORONA_SATFRAC * 100).toFixed(0)}%) — photosphere or clipped inner corona`);
  console.log(`  stack jobs: ${segments.length * cfg.channels.length}`);
}

main();
