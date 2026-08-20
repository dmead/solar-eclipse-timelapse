#!/usr/bin/env node
/*
 * run-corona.mjs — Stage D: per-level colour images, then the HDR corona.
 *
 *   1. corona-combine.js  R/G/B stacks of one segment -> linear colour + Moon fix
 *   2. corona-hdr.js      all levels -> merged HDR + radially flattened version
 *   3. xisf-preview.js    PNGs for inspection (pix-planetary, used as a library)
 *
 *   node scripts/run-corona.mjs [--config <eclipse.json>] [--skip-combine]
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';

const COMBINE = path.join(ROOT, 'pjsr', 'corona-combine.js').replace(/\\/g, '/');
const REGISTER = path.join(ROOT, 'pjsr', 'corona-register.js').replace(/\\/g, '/');
const DRIFT = path.join(ROOT, 'pjsr', 'corona-drift.js').replace(/\\/g, '/');
const HDR = path.join(ROOT, 'pjsr', 'corona-hdr.js').replace(/\\/g, '/');
const PREVIEW = `${PIX_PLANETARY}/pjsr/xisf-preview.js`;

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/eclipse.json',
    skipCombine: false, noDrift: false,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--skip-combine') args.skipCombine = true;
    else if (argv[i] === '--no-drift') args.noDrift = true;
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

/*
 * Pick the pair to measure differential drift from: same exposure level (so the
 * two images are directly comparable) and the longest time baseline available,
 * which is what makes the small per-second rate measurable at all.
 */
function pickDriftPair(images) {
  let best = null;
  const byLevel = new Map();
  for (const im of images) {
    if (!byLevel.has(im.level)) byLevel.set(im.level, []);
    byLevel.get(im.level).push(im);
  }
  for (const group of byLevel.values()) {
    if (group.length < 2) continue;
    const sorted = group.slice().sort((a, b) => a.t - b.t);
    const a = sorted[0];
    const b = sorted[sorted.length - 1];
    const dt = Math.abs(b.t - a.t);
    if (!best || dt > best.dt) best = { a, b, dt };
  }
  return best;
}

function runPi(call, logPath, swapDir, okMarker) {
  const started = Date.now();
  return withPiLaunchLock(async () => spawn(
    PIX,
    ['-n', '--automation-mode', '--no-splash', `-r=${call}`, '--force-exit'],
    {
      env: {
        ...process.env,
        TMP: swapDir.replace(/\//g, '\\'),
        TEMP: swapDir.replace(/\//g, '\\'),
      },
      stdio: 'ignore',
    },
  )).then((child) => new Promise((resolve) => {
    child.on('exit', (code) => {
      const elapsed = ((Date.now() - started) / 1000).toFixed(0);
      let ok = false;
      let tail = '(no log)';
      try {
        const lines = fs.readFileSync(logPath, 'utf8').trim().split('\n');
        ok = okMarker === null || lines.some((l) => l.includes(okMarker));
        tail = lines.slice(-3).join('\n    ');
      } catch { /* absent log is the diagnosis */ }
      resolve({ ok, code, elapsed, tail });
    });
  }));
}

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const swapDir = cfg.swapDir || 'S:/solar-eclipse/swap';
  const levelDir = `${cfg.outDir}/levels`;
  const finalDir = `${cfg.outDir}/final`;
  const logDir = `${cfg.outDir}/logs`;
  for (const d of [levelDir, finalDir, logDir]) fs.mkdirSync(d, { recursive: true });

  const segments = cfg.segments.filter((s) => s.role === 'corona');

  // ---- 1. combine channels per segment ----
  const combine = async (seg, fixedR) => {
    const out = `${levelDir}/${seg.id}.xisf`;
    const chans = ['R', 'G', 'B'].map((c) => `${cfg.stackDir}/${seg.id}_${c}.xisf`);
    const missing = chans.filter((p) => !fs.existsSync(p));
    if (missing.length) {
      console.log(`  [${seg.id}] SKIPPED — missing ${missing.length} channel stack(s)`);
      return false;
    }
    const log = `${logDir}/${seg.id}_combine.log`;
    const call = `${COMBINE},${chans.join(',')},${out},${log}`
      + (fixedR ? `,${fixedR.toFixed(3)}` : '');
    const r = await runPi(call, log, swapDir, '=== COMBINE OK ===');
    console.log(`  [${seg.id}] combine ${r.ok ? 'OK' : 'FAILED'} in ${r.elapsed}s`
      + `${fixedR ? ' (r imposed)' : ''}\n    ${r.tail}`);
    if (!r.ok) throw new Error(`combine failed for ${seg.id} — see ${log}`);
    return true;
  };

  const done = [];
  for (const seg of segments) {
    if (args.skipCombine && fs.existsSync(`${levelDir}/${seg.id}.xisf`)) continue;
    if (await combine(seg, 0)) done.push(seg);
  }

  /*
   * The Moon's angular size is fixed over the few minutes of totality, so every
   * level must measure the same radius. Levels that disagree had their limb fit
   * pulled off by saturation or a lopsided inner corona; redo those with the
   * radius the well-behaved levels agree on, which leaves only the centre free
   * and is far better conditioned.
   */
  const radii = [];
  for (const seg of segments) {
    const side = `${levelDir}/${seg.id}_moon.json`;
    if (fs.existsSync(side)) radii.push(JSON.parse(fs.readFileSync(side, 'utf8')).radius);
  }
  if (radii.length >= 3) {
    const med = radii.slice().sort((a, b) => a - b)[radii.length >> 1];
    const bad = segments.filter((seg) => {
      const side = `${levelDir}/${seg.id}_moon.json`;
      if (!fs.existsSync(side)) return false;
      const r = JSON.parse(fs.readFileSync(side, 'utf8')).radius;
      return Math.abs(r - med) / med > 0.02;
    });
    if (bad.length) {
      console.log(`radius consensus ${med.toFixed(1)} px; refitting `
        + `${bad.map((s) => s.id).join(', ')}`);
      for (const seg of bad) await combine(seg, med);
    } else {
      console.log(`radius consensus ${med.toFixed(1)} px across all levels`);
    }
  }

  // ---- 2. HDR merge over the exposure ladder ----
  const images = [];
  for (const seg of segments) {
    const img = `${levelDir}/${seg.id}.xisf`;
    const side = `${levelDir}/${seg.id}_moon.json`;
    if (!fs.existsSync(img) || !fs.existsSync(side)) continue;
    const moon = JSON.parse(fs.readFileSync(side, 'utf8'));
    images.push({
      path: img, level: seg.level, t: seg.t_mid,
      cx: moon.cx, cy: moon.cy, r: moon.radius,
    });
  }
  if (images.length < 2) throw new Error(`need at least 2 levels, have ${images.length}`);

  /* ---- 2a. register the levels against each other ----
   * The shortest exposure is the reference: it is the level that carries the
   * limb, chromosphere and prominences, so it is the one whose Moon should end
   * up sharp. Every other level is matched to it by correlation rather than by
   * differencing two independent Moon fits, which is what put a second Moon in
   * the merged image. */
  const ref = images.reduce((a, b) => (a.level <= b.level ? a : b));
  const regCfgPath = `${cfg.outDir}/configs/corona_register.json`;
  const regOut = `${finalDir}/registration.json`;
  fs.writeFileSync(regCfgPath, JSON.stringify({
    images: images.map((i) => ({ path: i.path, level: i.level, t: i.t })),
    refPath: ref.path,
    out: regOut,
  }, null, 1));
  const rlog = `${logDir}/corona_register.log`;
  console.log(`registering ${images.length} levels onto L${ref.level}`);
  const rr = await runPi(`${REGISTER},${regCfgPath},${rlog}`, rlog, swapDir,
    '=== REGISTER OK ===');
  console.log(`  register ${rr.ok ? 'OK' : 'FAILED'} in ${rr.elapsed}s\n    ${rr.tail}`);
  if (!rr.ok) throw new Error(`registration failed — see ${rlog}`);
  const reg = JSON.parse(fs.readFileSync(regOut, 'utf8'));
  const byPath = new Map(reg.shifts.map((s) => [s.path, s]));
  for (const im of images) {
    const s = byPath.get(im.path);
    if (s) { im.dx = s.dx; im.dy = s.dy; im.ncc = s.ncc; }
  }

  // ---- 2b. measure how fast the Moon crosses the corona ----
  let drift = { x: 0, y: 0 };
  const pair = args.noDrift ? null : pickDriftPair(images);
  if (!pair) {
    console.log('drift: no same-level pair available — falling back to rigid registration');
  } else {
    const t0 = Math.min(...images.map((i) => i.t));
    const driftCfgPath = `${cfg.outDir}/configs/corona_drift.json`;
    const driftOut = `${finalDir}/drift.json`;
    const driftSpec = (im) => ({
      path: im.path, cx: im.cx, cy: im.cy, r: im.r, t: im.t - t0,
      dx: im.dx, dy: im.dy,
    });
    fs.writeFileSync(driftCfgPath, JSON.stringify({
      a: driftSpec(pair.a), b: driftSpec(pair.b), out: driftOut,
    }, null, 1));
    const dlog = `${logDir}/corona_drift.log`;
    console.log(`drift: level ${pair.a.level}, ${pair.dt.toFixed(1)}s baseline`);
    const dr = await runPi(`${DRIFT},${driftCfgPath},${dlog}`, dlog, swapDir, '=== DRIFT OK ===');
    console.log(`  drift ${dr.ok ? 'OK' : 'FAILED'} in ${dr.elapsed}s\n    ${dr.tail}`);
    if (dr.ok) {
      const d = JSON.parse(fs.readFileSync(driftOut, 'utf8'));
      drift = { x: d.driftPxPerSec.x, y: d.driftPxPerSec.y };
    } else {
      console.log('  continuing with rigid registration');
    }
  }

  const tRef = images.reduce((a, b) => (a.level <= b.level ? a : b)).t;
  for (const im of images) im.t -= tRef;

  const hdrCfg = {
    images,
    drift,
    out: `${finalDir}/corona_hdr.xisf`,
    outFlat: `${finalDir}/corona_flat.xisf`,
  };
  const hdrCfgPath = `${cfg.outDir}/configs/corona_hdr.json`;
  fs.writeFileSync(hdrCfgPath, JSON.stringify(hdrCfg, null, 1));

  console.log(`HDR over ${images.length} level image(s)`);
  const hlog = `${logDir}/corona_hdr.log`;
  const hr = await runPi(`${HDR},${hdrCfgPath},${hlog}`, hlog, swapDir, '=== HDR OK ===');
  console.log(`  hdr ${hr.ok ? 'OK' : 'FAILED'} in ${hr.elapsed}s\n    ${hr.tail}`);
  if (!hr.ok) throw new Error(`hdr failed — see ${hlog}`);

  // ---- 3. previews ----
  for (const [name, src] of [['corona_hdr', hdrCfg.out], ['corona_flat', hdrCfg.outFlat]]) {
    const plog = `${logDir}/${name}_preview.log`;
    await runPi(`${PREVIEW},${src},${cfg.outDir}/preview/${name}_full.png,`
      + `${cfg.outDir}/preview/${name}_crop.png,1200`, plog, swapDir, null);
  }
  console.log(`previews -> ${cfg.outDir}/preview/`);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
