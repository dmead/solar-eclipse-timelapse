#!/usr/bin/env node
/*
 * align-trial.mjs — which alignment gives the sharpest drizzled prominence group?
 *
 * Before drizzle-stacking the timelapse we need to know whether a ~20-frame group
 * registers well enough to gain anything, and which aligner to use inside it. The
 * per-frame disc fit that stabilises the video is good to 0.15-0.37 superpixel,
 * which is 0.3-0.7 SENSOR px - marginal for placing samples on a 2x drizzle grid,
 * where sub-0.2 px is wanted. Phase correlation should do better, but on this
 * subject it is unproven: the plan flagged it as a risk because fftalign.jsh was
 * tuned on lunar surfaces, and the prominence frames are a thin bright rim on
 * black.
 *
 * So: cut one group, stack it both ways, and measure.
 *
 *   node scripts/align-trial.mjs [--frames 20] [--start 60]
 *
 * Prominences are Halpha, so the R channel is the one that matters and the only
 * one trialled. A single unstacked frame is drizzled alongside as the control -
 * without it a "sharper" stack could just be the drizzle grid, not the alignment.
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';
const SLICE = path.join(ROOT, 'pjsr', 'ser-slice.js').replace(/\\/g, '/');
const STACK = `${PIX_PLANETARY}/pjsr/ser-stack.js`;
const SWAP = 'S:/solar-eclipse/swap';

// The short-exposure prominence run in 14_13_00, already sliced by Stage B.
const SRC = 'S:/solar-eclipse/out/slices/14_13_00_f1170.ser';
const WORK = 'S:/solar-eclipse/out/aligntrial';

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function args() {
  const a = { frames: 20, start: 60 };
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--frames') a.frames = parseInt(process.argv[++i], 10);
    else if (process.argv[i] === '--start') a.start = parseInt(process.argv[++i], 10);
    else throw new Error(`unknown arg: ${process.argv[i]}`);
  }
  return a;
}

function runPi(call, log, marker) {
  return withPiLaunchLock(async () => spawn(
    PIX, ['-n', '--automation-mode', '--no-splash', `-r=${call}`, '--force-exit'],
    { env: { ...process.env, TMP: SWAP.replace(/\//g, '\\'), TEMP: SWAP.replace(/\//g, '\\') },
      stdio: 'ignore' },
  )).then((c) => new Promise((r) => c.on('exit', r)))
    .then(() => {
      let t = '';
      try { t = fs.readFileSync(log, 'utf8'); } catch { /* absent log is the diagnosis */ }
      if (!t.includes(marker)) throw new Error(`FAILED - see ${log}`);
      return t;
    });
}

async function main() {
  const a = args();
  fs.mkdirSync(WORK, { recursive: true });
  const slice = `${WORK}/group.ser`;
  const one = `${WORK}/single.ser`;

  console.log(`slicing ${a.frames} frames from ${a.start}`);
  await runPi(`${SLICE},${SRC},${slice},${a.start},${a.frames},${WORK}/slice.log`,
    `${WORK}/slice.log`, '=== SLICE OK ===');
  await runPi(`${SLICE},${SRC},${one},${a.start},1,${WORK}/slice1.log`,
    `${WORK}/slice1.log`, '=== SLICE OK ===');

  /*
   * bestFraction 1.0, not the usual 0.4. A timelapse group is only 20 frames and
   * they are 0.86 s apart in a video whose cadence is fixed - there is no budget
   * to throw 60% of them away, and the question being asked is what the aligner
   * does with the frames as they come.
   */
  const trials = [
    { tag: 'fft', alignOnGradient: false },
    { tag: 'grad', alignOnGradient: true },
  ];
  for (const t of trials) {
    const cfg = {
      ser: slice, channel: 'R', out: `${WORK}/${t.tag}.xisf`,
      log: `${WORK}/${t.tag}.log`, report: `${WORK}/${t.tag}.json`,
      bestFraction: 1.0, minFrames: 2, maxFrames: a.frames,
      alignOnGradient: t.alignOnGradient, localAlign: false,
      drizzle: 2, drizzleMargin: 16,
    };
    const p = `${WORK}/${t.tag}_cfg.json`;
    fs.writeFileSync(p, JSON.stringify(cfg));
    console.log(`stacking: alignOnGradient=${t.alignOnGradient}`);
    await runPi(`${STACK},${p}`, cfg.log, '=== STACK OK');
  }

  // Control: the same drizzle grid, one frame, no alignment involved at all.
  const cfg1 = {
    ser: one, channel: 'R', out: `${WORK}/single.xisf`,
    log: `${WORK}/single.log`, report: `${WORK}/single.json`,
    bestFraction: 1.0, minFrames: 1, maxFrames: 1,
    alignOnGradient: false, localAlign: false, drizzle: 2, drizzleMargin: 16,
  };
  fs.writeFileSync(`${WORK}/single_cfg.json`, JSON.stringify(cfg1));
  console.log('stacking: single-frame control');
  await runPi(`${STACK},${WORK}/single_cfg.json`, cfg1.log, '=== STACK OK');

  console.log(`\noutputs in ${WORK}`);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
