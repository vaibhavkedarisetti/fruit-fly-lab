/*
 * Simulation worker.
 *
 * The whole-brain simulation runs about ten times slower than real time, so it
 * cannot share a thread with the interface. This worker owns the Session and
 * posts telemetry back; the page only draws.
 */
'use strict';

import { decodeConnectome } from './engine.js';
import { Session } from './sim.js';

let session = null;
let meta = null;
let running = false;
let simMsPerTick = 2.0;
let neurons = null;
let timer = null;

// `type` is the message tag; payload keys must never shadow it.
function post(type, payload, transfer) {
  const msg = { ...payload, type };
  transfer ? self.postMessage(msg, transfer) : self.postMessage(msg);
}

// Data lives at <site>/data/, but this module is served from <site>/js/, so
// relative fetches must be resolved against the module URL, not the worker CWD.
const asset = (name) => new URL(`../data/${name}`, import.meta.url).href;

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) {
    post('progress', { label, loaded: 0, total: 0 });
    return await res.arrayBuffer();
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post('progress', { label, loaded, total });
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

/** neurons.bin: int64 root_id, float32 pos*3, uint16 type, uint8 class, uint8 side, int8 sign */
function decodeNeurons(buf, n) {
  let o = 0;
  const root = new BigInt64Array(buf, o, n); o += n * 8;
  const pos = new Float32Array(buf, o, n * 3); o += n * 12;
  const type = new Uint16Array(buf, o, n); o += n * 2;
  const cls = new Uint8Array(buf, o, n); o += n;
  const side = new Uint8Array(buf, o, n); o += n;
  const sign = new Int8Array(buf, o, n);
  return { root, pos, type, cls, side, sign };
}

async function boot() {
  try {
    post('status', { message: 'loading annotations' });
    meta = await (await fetch(asset('meta.json'))).json();

    post('status', { message: 'loading connectome' });
    const cbuf = await fetchWithProgress(asset('connectome.bin'), 'connectome');
    post('status', { message: 'loading neurons' });
    const nbuf = await fetchWithProgress(asset('neurons.bin'), 'neurons');

    post('status', { message: 'decoding connectivity' });
    const cn = decodeConnectome(cbuf, meta.n, meta.nnz);
    neurons = decodeNeurons(nbuf, meta.n);

    post('status', { message: 'building network' });
    session = new Session(cn, meta, 0);

    // hand the page what it needs to draw, transferring the buffers
    const posCopy = neurons.pos.slice();
    const clsCopy = neurons.cls.slice();
    const typeCopy = neurons.type.slice();
    post('ready', {
      meta: {
        dataset: meta.dataset, manifest: meta.manifest, n: meta.n, nnz: meta.nnz,
        cell_types: meta.cell_types, super_classes: meta.super_classes,
        sides: meta.sides, modalities: meta.modalities.map(m => ({ ...m, idx: [] })),
        dn_commands: meta.dn_commands, lif: meta.lif,
        lesionable: Object.keys(meta.lesionable),
        dnp01_root_ids: meta.dnp01_root_ids,
      },
      pos: posCopy, cls: clsCopy, typeCodes: typeCopy,
    }, [posCopy.buffer, clsCopy.buffer, typeCopy.buffer]);
  } catch (e) {
    post('error', { message: String(e && e.message || e) });
  }
}

function loop() {
  if (!running || !session) return;
  const t0 = performance.now();
  const frames = session.advance(simMsPerTick);
  const wall = performance.now() - t0;
  if (frames.length) {
    const f = frames[frames.length - 1];
    f.wall_ms = Math.round(wall * 100) / 100;
    f.realtime_factor = (f.t_ms / 1000) / Math.max(1e-6, totalWall / 1000);
    totalWall += wall;
    // sparse list of currently active neurons for the 3D view
    const ws = session.windowSum, act = [];
    for (let i = 0; i < ws.length && act.length < 3000; i++) if (ws[i]) act.push(i);
    f.active_idx = act;
    f.raster = session.raster.slice(-1500);
    post('frame', { frame: f });
  }
  timer = setTimeout(loop, 0);
}

let totalWall = 0;

/**
 * Real connectivity of a cell type, aggregated by partner type. Reads the same
 * CSR the simulation runs on, so what the inspector shows is what the model uses.
 */
function circuitOf(cellType, top) {
  const tcode = meta.cell_types.indexOf(cellType);
  if (tcode < 0) return { error: `no cell type "${cellType}" in ${meta.dataset}` };
  const cells = [];
  for (let i = 0; i < meta.n; i++) if (neurons.type[i] === tcode) cells.push(i);
  if (!cells.length) return { error: `no neurons of type "${cellType}"` };

  const isTarget = new Uint8Array(meta.n);
  for (const i of cells) isTarget[i] = 1;
  const cn = session.engine;
  const outSyn = new Map(), inSyn = new Map();

  const bump = (map, code, syn, signed) => {
    let e = map.get(code);
    if (!e) { e = { syn: 0, n: 0, signed: 0 }; map.set(code, e); }
    e.syn += Math.abs(syn); e.n += 1; e.signed += signed;
  };

  for (const i of cells) {                       // outputs: this type's own rows
    for (let k = cn.indptr[i]; k < cn.indptr[i + 1]; k++) {
      const w = cn.w[k] / meta.lif.w_syn;
      bump(outSyn, neurons.type[cn.indices[k]], w, w);
    }
  }
  for (let i = 0; i < meta.n; i++) {             // inputs: one scan of the graph
    const a = cn.indptr[i], b = cn.indptr[i + 1];
    if (a === b) continue;
    const srcType = neurons.type[i];
    for (let k = a; k < b; k++) {
      if (isTarget[cn.indices[k]]) {
        const w = cn.w[k] / meta.lif.w_syn;
        bump(inSyn, srcType, w, w);
      }
    }
  }

  const fmt = (map) => [...map.entries()]
    .map(([code, e]) => ({ type: meta.cell_types[code] || '(unannotated)',
                           synapses: Math.round(e.syn), connections: e.n,
                           sign: e.signed > 0 ? 'excitatory' : 'inhibitory' }))
    .sort((x, y) => y.synapses - x.synapses).slice(0, top);

  const sides = {};
  for (const i of cells) {
    const s = meta.sides[neurons.side[i]] || '?';
    sides[s] = (sides[s] || 0) + 1;
  }
  return {
    cell_type: cellType, n_cells: cells.length, sides,
    root_ids: cells.slice(0, 20).map(i => neurons.root[i].toString()),
    inputs: fmt(inSyn), outputs: fmt(outSyn), dataset: meta.dataset,
  };
}

self.onmessage = (ev) => {
  const m = ev.data;
  try {
    switch (m.cmd) {
      case 'boot': boot(); break;
      case 'play':
        if (!running) { running = true; loop(); }
        post('state', { running: true });
        break;
      case 'pause':
        running = false; if (timer) clearTimeout(timer);
        post('state', { running: false });
        break;
      case 'reset':
        running = false; if (timer) clearTimeout(timer);
        totalWall = 0;
        session.reset(m.seed ?? 0);
        post('state', { running: false, reset: true });
        break;
      case 'speed':
        simMsPerTick = Math.max(0.1, Math.min(20, m.value));
        post('state', { simMsPerTick });
        break;
      case 'looming': {
        const s = session.addLooming(m.opts || {});
        post('stimulus', { ok: true, kind: 'looming', state: s.state(session.engine.tMs) });
        break;
      }
      case 'modality': {
        try {
          const r = session.addModality(m.key, m.intensity ?? 1, m.duration_ms ?? 300);
          post('stimulus', { ok: true, kind: m.key, n: r.n, label: r.label,
                             citation: r.citation });
        } catch (e) {
          post('stimulus', { ok: false, notModeled: !!e.notModeled,
                             label: e.label, reason: e.message });
        }
        break;
      }
      case 'food': {
        // Composite environment: odour + tarsal contact + proboscis sugar.
        const parts = ['odor_vinegar', 'touch_leg_taste', 'taste_sugar'];
        let n = 0; const labels = [];
        for (const k of parts) {
          const r = session.addModality(k, m.intensity ?? 1, m.duration_ms ?? 600);
          n += r.n; labels.push(r.label);
        }
        post('stimulus', { ok: true, kind: 'food', n, label: labels.join(' + ') });
        break;
      }
      case 'clear': session.clearStimuli(); post('stimulus', { ok: true, cleared: true }); break;
      case 'silence': {
        const idx = meta.lesionable[m.cellType] || [];
        session.engine.silence(idx, m.on);
        post('lesion', { ok: true, cellType: m.cellType, n: idx.length, on: m.on });
        break;
      }
      case 'replay': post('replay', { frames: session.history.slice(-6000) }); break;
      case 'circuit': post('circuit', circuitOf(m.cellType, m.top ?? 25)); break;
      case 'lookup': {
        const i = m.idx;
        post('lookup', {
          idx: i,
          root_id: neurons.root[i].toString(),
          cell_type: meta.cell_types[neurons.type[i]],
          super_class: meta.super_classes[neurons.cls[i]],
          side: meta.sides[neurons.side[i]],
          sign: neurons.sign[i],
          spikes: session.engine.spikeCounts[i],
        });
        break;
      }
    }
  } catch (e) {
    post('error', { message: String(e && e.message || e) });
  }
};
