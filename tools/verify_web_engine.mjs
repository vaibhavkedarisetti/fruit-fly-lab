/*
 * Headless harness: runs a scenario through the browser LIF engine in Node and
 * writes the resulting spike counts, so tools/verify_web_engine.py can compare
 * them against the Python engine.
 *
 * Usage: node tools/verify_web_engine.mjs <scenario.json> <out.json>
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { LIFEngine, decodeConnectome } from '../web/js/engine.js';

const [, , scenarioPath, outPath] = process.argv;
const scenario = JSON.parse(fs.readFileSync(scenarioPath, 'utf8'));

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dataDir = path.join(root, 'web', 'data');

function readAB(p) {
  const b = fs.readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

const meta = JSON.parse(fs.readFileSync(path.join(dataDir, 'meta.json'), 'utf8'));
const cn = decodeConnectome(readAB(path.join(dataDir, 'connectome.bin')), meta.n, meta.nnz);

const t0 = Date.now();
const eng = new LIFEngine(cn, undefined, scenario.seed);

if (scenario.silence && scenario.silence.length) eng.silence(scenario.silence, true);
if (scenario.inject_g) {
  for (const [i, val] of scenario.inject_g) eng.g[i] = val;
}
if (scenario.poisson_idx && scenario.poisson_idx.length) {
  eng.setPoisson(scenario.poisson_idx, scenario.poisson_rate);
}

eng.run(scenario.duration_ms);
const wall = (Date.now() - t0) / 1000;

const counts = Array.from(eng.spikeCounts);
const active = counts.reduce((a, c) => a + (c > 0 ? 1 : 0), 0);
const total = counts.reduce((a, c) => a + c, 0);

fs.writeFileSync(outPath, JSON.stringify({
  n: meta.n,
  nnz: meta.nnz,
  total_spikes: total,
  active_neurons: active,
  spike_counts: counts,
  wall_s: wall,
  steps: eng.stepCount,
}));

console.error(`[js] ${meta.n} neurons, ${meta.nnz} connections | ` +
  `${scenario.duration_ms} ms in ${wall.toFixed(2)} s | ` +
  `${total} spikes across ${active} neurons`);
