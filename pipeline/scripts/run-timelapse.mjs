#!/usr/bin/env node
/*
 * run-timelapse.mjs — Stage F: render the planned frames, then encode.
 *
 * gen_timelapse.py has already chosen which frames to use and what gain each
 * needs; this renders them and hands the sequence to ffmpeg.
 *
 *   node scripts/run-timelapse.mjs [--config <timelapse.json>] [--encode-only]
 *
 * ffmpeg has no glob support in this build and palettegen has crashed here
 * before, so the frames are a numbered seq_%05d.png sequence encoded straight to
 * H.264.
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PIX = 'C:/Program Files/PixInsight/bin/PixInsight.exe';
const PIX_PLANETARY = 'D:/projects/pix-planetary';
const TL = path.join(ROOT, 'pjsr', 'tl-frames.js').replace(/\\/g, '/');

const { withPiLaunchLock } = await import(
  pathToFileURL(`${PIX_PLANETARY}/scripts/pi-lock.mjs`).href);

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/timelapse.json',
    out: 'S:/solar-eclipse/out/final/timelapse.mp4',
    swap: 'S:/solar-eclipse/swap',
    encodeOnly: false,
    workers: 4,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else if (argv[i] === '--swap') args.swap = argv[++i];
    else if (argv[i] === '--encode-only') args.encodeOnly = true;
    else if (argv[i] === '--workers') args.workers = parseInt(argv[++i], 10);
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

function run(cmd, cmdArgs, env) {
  return new Promise((resolve) => {
    const child = spawn(cmd, cmdArgs, { env: env ?? process.env, stdio: 'ignore' });
    child.on('exit', (code) => resolve(code));
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const logDir = path.dirname(cfg.outDir) + '/logs';
  fs.mkdirSync(logDir, { recursive: true });
  fs.mkdirSync(path.dirname(args.out), { recursive: true });

  if (!args.encodeOnly) {
    /*
     * Clear the frame directory first. The encoder globs seq_*.png rather than
     * reading the config, so leftovers from a longer previous render are silently
     * appended to the video - a 1311-frame render encoded as 1322 frames, the
     * last 11 belonging to a different cut.
     */
    try {
      for (const f of fs.readdirSync(cfg.outDir))
        if (/^seq_\d+\.png$/.test(f)) fs.unlinkSync(path.join(cfg.outDir, f));
    } catch { /* first run: the directory does not exist yet */ }

    /*
     * Render in parallel shards.
     *
     * The render is CPU-bound inside PJSR, not I/O-bound. Measured on this
     * machine: S: reads SER at 1035 MB/s and writes PNGs at 4170 MB/s, against
     * 0.45 s/frame of demosaic, crop and PNG encode. Moving the output to another
     * disk therefore buys nothing — Z: benchmarks SLOWER than S:, at 3264 MB/s —
     * and the only real lever is running several PixInsight instances at once.
     *
     * Shards are CONTIGUOUS blocks, not round-robin. The renderer cross-dissolves
     * across gaps in the sequence, and a round-robin shard sees a gap at every
     * frame, so it would dissolve continuously. Contiguous blocks keep the
     * dissolve correct everywhere except the n-1 block seams, and they also open
     * each SER file in far fewer separate passes.
     *
     * Each shard keeps its frames' ORIGINAL sequence numbers, so the shards
     * jointly write one continuous seq_%05d run and ffmpeg sees exactly what it
     * would have from a single pass.
     */
    const n = Math.max(1, Math.min(args.workers, cfg.frames.length));
    console.log(`rendering ${cfg.frames.length} frames -> ${cfg.outDir}`
      + ` (${n} worker${n > 1 ? 's' : ''})`);
    const started = Date.now();

    const cfgDir = path.dirname(args.config);
    const per = Math.ceil(cfg.frames.length / n);
    const shards = [];
    for (let k = 0; k < n; k++) {
      const sub = { ...cfg, frames: [] };
      for (let i = k * per; i < Math.min((k + 1) * per, cfg.frames.length); i++)
        sub.frames.push({ ...cfg.frames[i], seq: i });
      if (!sub.frames.length) continue;
      const p = path.join(cfgDir, `tl_shard${k}.json`).replace(/\\/g, '/');
      fs.writeFileSync(p, JSON.stringify(sub));
      shards.push({ cfg: p, log: `${logDir}/timelapse_${k}.log` });
    }

    const codes = await Promise.all(shards.map((sh) =>
      withPiLaunchLock(async () => spawn(
        PIX,
        ['-n', '--automation-mode', '--no-splash', `-r=${TL},${sh.cfg},${sh.log}`,
          '--force-exit'],
        {
          env: {
            ...process.env,
            TMP: args.swap.replace(/\//g, '\\'),
            TEMP: args.swap.replace(/\//g, '\\'),
          },
          stdio: 'ignore',
        },
      )).then((child) => new Promise((r) => child.on('exit', r)))));

    let ok = true;
    shards.forEach((sh, k) => {
      let good = false;
      try {
        good = fs.readFileSync(sh.log, 'utf8').includes('=== TIMELAPSE OK ===');
      } catch { /* absent log is the diagnosis */ }
      if (!good) {
        ok = false;
        console.log(`  shard ${k} FAILED (exit ${codes[k]}) - see ${sh.log}`);
      }
    });
    console.log(`  render ${ok ? 'OK' : 'FAILED'} in `
      + `${((Date.now() - started) / 60000).toFixed(1)} min`);
    if (!ok) throw new Error('render failed');
  }

  const n = fs.readdirSync(cfg.outDir).filter((f) => /^seq_\d+\.png$/.test(f)).length;
  if (!n) throw new Error(`no frames in ${cfg.outDir}`);
  console.log(`encoding ${n} frames at ${cfg.fps} fps`);

  const code = await run('ffmpeg', [
    '-y', '-framerate', String(cfg.fps),
    '-i', `${cfg.outDir}/seq_%05d.png`,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    args.out,
  ]);
  if (code !== 0) throw new Error(`ffmpeg failed with exit ${code}`);
  const mb = (fs.statSync(args.out).size / 1e6).toFixed(1);
  console.log(`${args.out} (${mb} MB)`);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
