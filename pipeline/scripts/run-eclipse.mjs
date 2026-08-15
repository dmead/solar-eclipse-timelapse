#!/usr/bin/env node
/*
 * run-eclipse.mjs — slice measured segments out of the raw captures, then stack
 * each one per CFA channel.
 *
 * Two phases, because an eclipse SER holds several exposure states:
 *   1. ser-slice.js (this repo)      raw SER + frame range -> one-state SER
 *   2. ser-stack.js (pix-planetary)  slice + channel       -> drizzled XISF stack
 *
 * pix-planetary is used as a library, by absolute path — ser-stack.js and its
 * relative `#include "lib/fftalign.jsh"` are not copied here, so improvements to
 * the stacker land in this pipeline automatically.
 *
 *   node scripts/run-eclipse.mjs [--config <eclipse.json>] [--only id,id]
 *                                [--channels R,G] [--concurrency 2]
 *                                [--slice-only|--stack-only] [--dry]
 *
 * Node: must be the winget v24 binary by full path. `node` on PATH is v10 and
 * dies on the first import having run zero lines.
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';

const SLICE_SCRIPT = path.join(ROOT, 'pjsr', 'ser-slice.js').replace(/\\/g, '/');
const STACK_SCRIPT = `${PIX_PLANETARY}/pjsr/ser-stack.js`;

// The launch mutex is shared with pix-planetary on purpose: two PixInsight
// instances starting in the same second race the same instance slot and hang at
// 0 CPU, and that race does not care which repo launched them.
const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/eclipse.json',
    only: null, channels: null, concurrency: null,
    sliceOnly: false, stackOnly: false, dry: false,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--only') args.only = argv[++i].split(',');
    else if (argv[i] === '--channels') args.channels = argv[++i].split(',');
    else if (argv[i] === '--concurrency') args.concurrency = parseInt(argv[++i], 10);
    else if (argv[i] === '--slice-only') args.sliceOnly = true;
    else if (argv[i] === '--stack-only') args.stackOnly = true;
    else if (argv[i] === '--dry') args.dry = true;
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

function runPi(scriptArgs, logPath, swapDir, okMarker) {
  const started = Date.now();
  return withPiLaunchLock(async () => spawn(
    PIX,
    ['-n', '--automation-mode', '--no-splash', `-r=${scriptArgs}`, '--force-exit'],
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
      let tail = '(no log — script died before its first write)';
      try {
        const lines = fs.readFileSync(logPath, 'utf8').trim().split('\n');
        ok = lines.some((l) => l.includes(okMarker));
        tail = lines.slice(-2).join('\n    ');
      } catch { /* absent log is itself the diagnosis */ }
      resolve({ ok, code, elapsed, tail });
    });
  }));
}

/* Slices are reused across the per-channel stacks, so an existing slice of the
 * expected size is left alone — re-slicing 22 GB off a slow drive to redo one
 * channel would dominate the run. */
function expectedSliceBytes(seg, cfg) {
  const g = cfg.geometry;
  if (!g) return null;
  const bpp = g.depth > 8 ? 2 : 1;
  const planes = g.color_id >= 100 ? 3 : 1;
  return 178 + seg.count * (g.width * g.height * planes * bpp) + seg.count * 8;
}

function sliceIsCurrent(seg, cfg) {
  if (!fs.existsSync(seg.slice)) return false;
  const want = expectedSliceBytes(seg, cfg);
  // Without known geometry, presence is all we can check; with it, a size
  // mismatch means the slice was cut for a different frame range.
  return want === null ? true : fs.statSync(seg.slice).size === want;
}

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const swapDir = (cfg.swapDir || 'S:/solar-eclipse/swap').replace(/\\/g, '/');
  const channels = args.channels || cfg.channels || ['R', 'G', 'B'];
  const concurrency = args.concurrency ?? cfg.concurrency ?? 2;

  let segments = cfg.segments;
  if (args.only) segments = segments.filter((s) => args.only.includes(s.id));
  if (!segments.length) throw new Error('no segments selected');

  for (const d of [cfg.sliceDir, cfg.stackDir, swapDir, `${cfg.outDir}/logs`])
    fs.mkdirSync(d, { recursive: true });

  console.log(`${cfg.name}: ${segments.length} segment(s) x ${channels.length} channel(s), `
    + `concurrency ${concurrency}`);

  // ---- phase 1: slice ----
  if (!args.stackOnly) {
    const todo = segments.filter((s) => !sliceIsCurrent(s, cfg));
    console.log(`slice: ${todo.length} to cut, ${segments.length - todo.length} already present`);
    for (const seg of todo) {
      const log = `${cfg.outDir}/logs/${seg.id}_slice.log`;
      const call = `${SLICE_SCRIPT},${seg.src},${seg.slice},${seg.start},${seg.count},${log}`;
      if (args.dry) { console.log(`  [dry] ${seg.id} f${seg.start}+${seg.count}`); continue; }
      console.log(`  [${seg.id}] slicing f${seg.start}+${seg.count} from ${path.basename(seg.src)}`);
      const r = await runPi(call, log, swapDir, '=== SLICE OK ===');
      console.log(`  [${seg.id}] ${r.ok ? 'OK' : 'FAILED'} in ${r.elapsed}s (exit ${r.code})`
        + (r.ok ? '' : `\n    ${r.tail}`));
      if (!r.ok) throw new Error(`slice failed for ${seg.id} — see ${log}`);
    }
  }
  if (args.sliceOnly) { console.log('slice-only: done'); return; }

  // ---- phase 2: stack, one job per segment x channel ----
  const jobs = [];
  for (const seg of segments)
    for (const ch of channels) {
      const id = `${seg.id}_${ch}`;
      const jobCfg = {
        ser: seg.slice,
        channel: ch,
        out: `${cfg.stackDir}/${id}.xisf`,
        log: `${cfg.outDir}/logs/${id}_stack.log`,
        report: `${cfg.stackDir}/${id}.json`,
        bestFraction: seg.bestFraction,
        minFrames: seg.minFrames,
        maxFrames: seg.maxFrames,
        alignOnGradient: seg.alignOnGradient,
        localAlign: seg.localAlign,
        drizzle: seg.drizzle,
        drizzleMargin: seg.drizzleMargin,
      };
      const cfgPath = `${cfg.outDir}/configs/${id}.json`;
      if (!args.dry) fs.writeFileSync(cfgPath, JSON.stringify(jobCfg, null, 2));
      jobs.push({ id, cfgPath, logPath: jobCfg.log });
    }

  if (args.dry) {
    for (const j of jobs) console.log(`  [dry] stack ${j.id}`);
    return;
  }

  const results = [];
  let next = 0;
  async function worker() {
    while (next < jobs.length) {
      const j = jobs[next++];
      console.log(`  [${j.id}] stacking`);
      const r = await runPi(`${STACK_SCRIPT},${j.cfgPath}`, j.logPath, swapDir, '=== STACK OK');
      console.log(`  [${j.id}] ${r.ok ? 'OK' : 'FAILED'} in ${r.elapsed}s (exit ${r.code})`
        + (r.ok ? '' : `\n    ${r.tail}`));
      results.push({ id: j.id, ...r });
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, worker));

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} stack jobs OK`);
  if (failed.length) {
    console.log(`failed: ${failed.map((r) => r.id).join(', ')}`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
