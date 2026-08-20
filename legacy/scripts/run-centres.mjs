#!/usr/bin/env node
/*
 * run-centres.mjs — Stage F pass 1: find the Sun in every planned frame.
 *
 * pjsr/tl-centres.js reads the frame list gen_timelapse.py wrote and measures a
 * disc centre for each one, which smooth_track.py then joins on (file, index).
 * It has to be re-run whenever the frame list gains frames the last run never
 * saw — otherwise those frames have no detection and smooth_track drops them.
 *
 *   node scripts/run-centres.mjs [--config <timelapse.json>]
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';
const JS = path.join(ROOT, 'pjsr', 'tl-centres.js').replace(/\\/g, '/');

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/timelapse.json',
    out: 'S:/solar-eclipse/out/diag/centres.json',
    swap: 'S:/solar-eclipse/swap',
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const logDir = `${path.dirname(cfg.outDir)}/logs`;
  fs.mkdirSync(logDir, { recursive: true });
  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  const log = `${logDir}/centres.log`;

  console.log(`detecting ${cfg.frames.length} frames -> ${args.out}`);
  const started = Date.now();

  const child = await withPiLaunchLock(async () => spawn(
    PIX,
    ['-n', '--automation-mode', '--no-splash',
      `-r=${JS},${args.config},${args.out},${log}`, '--force-exit'],
    {
      env: {
        ...process.env,
        TMP: args.swap.replace(/\//g, '\\'),
        TEMP: args.swap.replace(/\//g, '\\'),
      },
      stdio: 'ignore',
    },
  ));
  const code = await new Promise((r) => child.on('exit', r));

  let ok = false;
  try {
    ok = fs.readFileSync(log, 'utf8').includes('=== CENTRES OK ===');
  } catch { /* absent log is the diagnosis */ }
  console.log(`  ${ok ? 'OK' : `FAILED (exit ${code}) - see ${log}`}`
    + ` in ${((Date.now() - started) / 60000).toFixed(1)} min`);
  if (!ok) throw new Error('detection failed');
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
