#!/usr/bin/env node
/*
 * encode-deliverables.mjs — one render, three cuts.
 *
 *   full      2240x1680, CRF 17. The archive copy, no scale filter at all.
 *   instagram 1080x810,  CRF 18, H.264 High @ 4.0. Sized so Instagram does not
 *             re-encode it: anything wider than 1080 goes through their scaler,
 *             which is much worse than ffmpeg's, and a thin white leader line on
 *             a black sky is exactly the content that shows it. 4:3 sits inside
 *             the 4:5 to 1.91:1 range the feed accepts, so the frame is delivered
 *             whole rather than cropped.
 *   preview   960 wide, CRF 26. Small enough to hand round for review.
 *
 * All three carry +faststart and yuv420p, and none carry audio.
 *
 *   node scripts/encode-deliverables.mjs [--config <timelapse.json>]
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

function parseArgs(argv) {
  const args = {
    config: 'S:/solar-eclipse/out/configs/timelapse.json',
    outDir: 'S:/solar-eclipse/out/final',
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--config') args.config = argv[++i];
    else if (argv[i] === '--out-dir') args.outDir = argv[++i];
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  return args;
}

function run(cmd, cmdArgs) {
  return new Promise((resolve) => {
    const child = spawn(cmd, cmdArgs, { stdio: 'ignore' });
    child.on('exit', resolve);
  });
}

const CUTS = [
  {
    name: 'timelapse.mp4',
    vf: null,
    extra: ['-crf', '17', '-preset', 'slow'],
  },
  {
    name: 'timelapse_instagram.mp4',
    vf: 'scale=1080:-2:flags=lanczos',
    extra: ['-crf', '18', '-preset', 'slow',
      '-profile:v', 'high', '-level', '4.0',
      // A capped bitrate and a closed 2 s GOP keep it inside what Instagram's
      // pipeline passes through untouched.
      '-maxrate', '8M', '-bufsize', '16M', '-g', '60', '-keyint_min', '60',
      '-x264-params', 'scenecut=0:open-gop=0'],
  },
  {
    name: 'timelapse_preview.mp4',
    vf: 'scale=960:-2:flags=lanczos',
    extra: ['-crf', '26', '-preset', 'slow'],
  },
];

async function main() {
  const args = parseArgs(process.argv);
  const cfg = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  const n = fs.readdirSync(cfg.outDir).filter((f) => /^seq_\d+\.png$/.test(f)).length;
  if (!n) throw new Error(`no frames in ${cfg.outDir}`);
  fs.mkdirSync(args.outDir, { recursive: true });
  console.log(`${n} frames at ${cfg.fps} fps -> ${CUTS.length} cuts`);

  for (const cut of CUTS) {
    const dest = path.join(args.outDir, cut.name).replace(/\\/g, '/');
    const ff = ['-y', '-framerate', String(cfg.fps),
      '-i', `${cfg.outDir}/seq_%05d.png`];
    if (cut.vf) ff.push('-vf', cut.vf);
    ff.push('-an', '-c:v', 'libx264', ...cut.extra,
      '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dest);
    const code = await run('ffmpeg', ff);
    if (code !== 0) throw new Error(`ffmpeg failed on ${cut.name} (exit ${code})`);
    const mb = (fs.statSync(dest).size / 1e6).toFixed(1);
    console.log(`  ${cut.name.padEnd(26)} ${mb} MB`);
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
