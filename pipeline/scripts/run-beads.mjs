#!/usr/bin/env node
/*
 * run-beads.mjs — Stage E: single frames from the contact windows.
 *
 * Baily's beads and the diamond ring are the one product here that must not be
 * stacked. They change shape in a fraction of a second, so averaging even a
 * second of frames turns the beads into a smooth arc. Stage A already isolated
 * these windows: they are the unfiltered frames that belong to no stable exposure
 * state, because the filter was coming off and the exposure was being ridden.
 *
 * Two passes. `preview` renders small autostretched PNGs of every window so the
 * frames worth keeping can be chosen by eye; `tif` then re-renders chosen frames
 * at full resolution in 16 bits.
 *
 *   node scripts/run-beads.mjs [--config <eclipse.json>] [--max-per-window 40]
 *   node scripts/run-beads.mjs --tif <file.ser>:<start>:<count>
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';
const FRAMES = path.join(ROOT, 'pjsr', 'ser-frames.js').replace(/\\/g, '/');

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/eclipse.json',
    maxPerWindow: 40, tif: null,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--max-per-window') args.maxPerWindow = parseInt(argv[++i], 10);
    else if (argv[i] === '--tif') args.tif = argv[++i];
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

function runPi(call, logPath, swapDir) {
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
        ok = lines.some((l) => l.includes('=== FRAMES OK ==='));
        tail = lines.slice(-2).join('\n    ');
      } catch { /* absent log is the diagnosis */ }
      resolve({ ok, code, elapsed, tail });
    });
  }));
}

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const swapDir = cfg.swapDir || 'S:/solar-eclipse/swap';
  const logDir = `${cfg.outDir}/logs`;
  fs.mkdirSync(logDir, { recursive: true });

  // Full-resolution re-render of a chosen range.
  if (args.tif) {
    const [src, start, count] = args.tif.split(':');
    const outDir = `${cfg.outDir}/beads/tif`;
    fs.mkdirSync(outDir, { recursive: true });
    const prefix = path.basename(src).replace(/\.ser$/i, '');
    const log = `${logDir}/beads_tif.log`;
    const r = await runPi(
      `${FRAMES},${src},${start},${count},1,${outDir},${prefix},tif,${log}`,
      log, swapDir);
    console.log(`tif ${r.ok ? 'OK' : 'FAILED'} in ${r.elapsed}s\n    ${r.tail}`);
    if (!r.ok) process.exit(1);
    return;
  }

  const windows = cfg.beads || [];
  if (!windows.length) throw new Error('no contact windows in config');

  console.log(`${windows.length} contact window(s)`);
  for (const w of windows) {
    const stride = Math.max(1, Math.ceil(w.count / args.maxPerWindow));
    const outDir = `${cfg.outDir}/beads/${w.id}`;
    fs.mkdirSync(outDir, { recursive: true });
    const log = `${logDir}/${w.id}_beads.log`;
    console.log(`  [${w.id}] ${w.kind} ${w.seconds}s, ${w.count} frames, stride ${stride}`);
    const r = await runPi(
      `${FRAMES},${w.src},${w.start},${w.count},${stride},${outDir},${w.id},preview,${log}`,
      log, swapDir);
    console.log(`  [${w.id}] ${r.ok ? 'OK' : 'FAILED'} in ${r.elapsed}s`
      + (r.ok ? '' : `\n    ${r.tail}`));
  }
  console.log(`previews -> ${cfg.outDir}/beads/`);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
