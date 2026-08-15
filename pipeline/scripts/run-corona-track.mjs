#!/usr/bin/env node
/*
 * run-corona-track.mjs — measure totality pointing by correlating the corona.
 *
 * Writes out/diag/corona_track.json, which smooth_track.py picks up to stabilise
 * totality per frame instead of placing it on a modelled line. Run it after
 * gen_timelapse.py and before smooth_track.py.
 *
 *   node scripts/run-corona-track.mjs [--config <timelapse.json>]
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';
const JS = path.join(ROOT, 'pjsr', 'tl-corona-track.js').replace(/\\/g, '/');

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/timelapse.json',
    out: 'S:/solar-eclipse/out/diag/corona_track.json',
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
  const log = `${logDir}/corona_track.log`;

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
    ok = fs.readFileSync(log, 'utf8').includes('=== CORONA TRACK OK ===');
  } catch { /* absent log is the diagnosis */ }
  console.log(`  ${ok ? 'OK' : `FAILED (exit ${code}) - see ${log}`}`
    + ` in ${((Date.now() - started) / 60000).toFixed(1)} min`);
  if (!ok) throw new Error('corona track failed');
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
