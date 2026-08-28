/*
 * Whole-brain leaky integrate-and-fire engine, browser port.
 *
 * PROVENANCE
 *   A. REAL DATA       connectivity and synapse counts, FlyWire FAFB v783
 *   B. PUBLISHED MODEL every equation and constant below
 *                      (Shiu et al. 2024, Nature 634:210-219)
 *   D. OUR ENGINEERING this file
 *
 * This is a line-for-line port of simulation/engine/lif_engine.py. It exists so
 * the laboratory can run without a persistent server process, not to simplify
 * anything: it reads all 139,255 neurons and all 3,732,460 real connections.
 *
 * tools/verify_web_engine.py runs the same seed and stimulus through both this
 * engine and the Python one and asserts the spike trains are identical, which
 * is why both use the same mulberry32 PRNG rather than their native ones.
 *
 * Exact integration (Brian2 method='linear'), a = 1/t_mbr, b = 1/tau:
 *   g(t+dt) = g(t) * exp(-b*dt)
 *   v(t+dt) = v_0 + (v(t)-v_0)*exp(-a*dt) + a*g(t)*(exp(-b*dt)-exp(-a*dt))/(a-b)
 */
'use strict';

/** Deterministic PRNG, identical to the Python reference in tools/prng.py. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Shiu et al. 2024 parameters. Units: ms and mV. */
export const LIF = {
  v_0: -52.0, v_rst: -52.0, v_th: -45.0,
  t_mbr: 20.0, tau: 5.0, t_rfc: 2.2, t_dly: 1.8,
  w_syn: 0.275, r_poi: 150.0, f_poi: 250.0, dt: 0.1,
};

export class LIFEngine {
  /**
   * @param {Object} cn  {n, indptr:Int32Array, indices:Int32Array, weights:Int16Array}
   *                     indices must already be delta-decoded to absolute.
   */
  constructor(cn, params = LIF, seed = 0) {
    this.p = { ...params };
    this.n = cn.n;
    this.indptr = cn.indptr;
    this.indices = cn.indices;

    // B: weight in mV = signed synapse count * w_syn
    this.w = new Float32Array(cn.weights.length);
    for (let i = 0; i < cn.weights.length; i++) this.w[i] = cn.weights[i] * this.p.w_syn;

    const a = 1.0 / this.p.t_mbr, b = 1.0 / this.p.tau, dt = this.p.dt;
    this.ev = Math.exp(-a * dt);
    this.eg = Math.exp(-b * dt);
    this.kg = a * (this.eg - this.ev) / (a - b);
    this.v0rest = this.p.v_0 * (1.0 - this.ev);
    // The Python engine holds these as float32 scalars and rounds after every
    // operation. Math.fround reproduces that bit for bit, which is what lets
    // tools/verify_web_engine.py compare the two exactly.
    this.ev32 = Math.fround(this.ev);
    this.eg32 = Math.fround(this.eg);
    this.kg32 = Math.fround(this.kg);
    this.v0rest32 = Math.fround(this.v0rest);
    this.vth32 = Math.fround(this.p.v_th);

    this.D = Math.round(this.p.t_dly / this.p.dt);          // 18 steps
    this.R = Math.round(this.p.t_rfc / this.p.dt);          // 22 steps
    this.poiW = this.p.w_syn * this.p.f_poi;                // 68.75 mV

    this.seed = seed;
    this.reset();
  }

  reset() {
    const n = this.n;
    this.v = new Float32Array(n).fill(this.p.v_0);
    this.g = new Float32Array(n);
    this.rfcLeft = new Int32Array(n);
    this.rfcLen = new Int32Array(n).fill(this.R);
    this.ring = new Float32Array((this.D + 1) * n);
    this.acc = new Float64Array(n);      // per-step float64 accumulator
    this.accTouched = new Int32Array(1024);
    this.slot = 0;
    this.tmpRef = new Int32Array(1024);
    this.spikeCounts = new Int32Array(n);
    this.silenced = new Uint8Array(n);
    this.stepCount = 0;
    this.tMs = 0;
    this.rand = mulberry32(this.seed);
    this.poiIdx = new Int32Array(0);
    this.poiP = new Float64Array(0);
    this.lastSpikes = new Int32Array(0);
  }

  /**
   * Poisson drive, matching brian2.PoissonInput(target_var='v', N=1,
   * weight=w_syn*f_poi). Targets lose their refractory period, as in the
   * reference implementation.
   */
  setPoisson(indices, ratesHz) {
    for (let k = 0; k < this.poiIdx.length; k++) this.rfcLen[this.poiIdx[k]] = this.R;
    const m = indices.length;
    this.poiIdx = Int32Array.from(indices);
    this.poiP = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const r = Array.isArray(ratesHz) || ArrayBuffer.isView(ratesHz) ? ratesHz[k] : ratesHz;
      this.poiP[k] = Math.max(0, Math.min(1, r * this.p.dt * 1e-3));
      this.rfcLen[this.poiIdx[k]] = 0;
      this.rfcLeft[this.poiIdx[k]] = 0;
    }
  }

  clearPoisson() {
    for (let k = 0; k < this.poiIdx.length; k++) this.rfcLen[this.poiIdx[k]] = this.R;
    this.poiIdx = new Int32Array(0);
    this.poiP = new Float64Array(0);
  }

  /** Silence neurons by removing their output (Shiu et al. `silence()`). */
  silence(indices, on = true) {
    for (const i of indices) this.silenced[i] = on ? 1 : 0;
  }

  /** Advance one dt. Returns an Int32Array of neurons that spiked. */
  step() {
    const n = this.n, v = this.v, g = this.g;
    const rfcLeft = this.rfcLeft, rfcLen = this.rfcLen;
    const base = this.slot * n, ring = this.ring;

    // 1. deliver synaptic input scheduled for this step
    for (let i = 0; i < n; i++) {
      const p = ring[base + i];
      if (p !== 0) { g[i] += p; ring[base + i] = 0; }
    }

    // 2. exact linear integration; refractory neurons are held, not integrated.
    //    Rounded to float32 after each operation, matching numpy.
    const ev = this.ev32, eg = this.eg32, kg = this.kg32, v0rest = this.v0rest32;
    const fr = Math.fround;
    for (let i = 0; i < n; i++) {
      if (rfcLeft[i] !== 0) { rfcLeft[i] -= 1; continue; }
      const gi = g[i];
      const tmp = fr(gi * kg);
      let vi = fr(v[i] * ev);
      vi = fr(vi + v0rest);
      v[i] = fr(vi + tmp);
      g[i] = fr(gi * eg);
    }

    // 3. external Poisson drive (adds directly to v, as in the reference)
    const pi = this.poiIdx, pp = this.poiP, pw = this.poiW;
    for (let k = 0; k < pi.length; k++) {
      if (this.rand() < pp[k]) v[pi[k]] += pw;
    }

    // 4. threshold and reset. Refractory neurons sit at v_rst (< v_th) and
    //    Poisson targets are never refractory, so no extra mask is needed.
    const vth = this.vth32, vrst = this.p.v_rst;
    let count = 0;
    let spikes = this.tmpRef;
    for (let i = 0; i < n; i++) {
      if (v[i] > vth) {
        if (count === spikes.length) {
          const bigger = new Int32Array(spikes.length * 2);
          bigger.set(spikes); spikes = this.tmpRef = bigger;
        }
        spikes[count++] = i;
      }
    }

    if (count) {
      const tgt = ((this.slot + this.D) % (this.D + 1)) * n;
      const indptr = this.indptr, indices = this.indices, w = this.w;
      const sc = this.spikeCounts, sil = this.silenced, acc = this.acc;
      let nTouched = 0;
      let touched = this.accTouched;
      for (let s = 0; s < count; s++) {
        const i = spikes[s];
        v[i] = vrst;
        g[i] = 0;                       // reference reset: g = 0
        rfcLeft[i] = rfcLen[i];
        sc[i] += 1;
        if (sil[i]) continue;           // 5. schedule outgoing events (delayed)
        const a = indptr[i], b = indptr[i + 1];
        for (let k = a; k < b; k++) {
          const j = indices[k];
          if (acc[j] === 0) {
            if (nTouched === touched.length) {
              const bigger = new Int32Array(touched.length * 2);
              bigger.set(touched); touched = this.accTouched = bigger;
            }
            touched[nTouched++] = j;
          }
          acc[j] += w[k];               // summed in float64, as np.bincount does
        }
      }
      // one float32 rounding when the total lands in the ring, as numpy does
      for (let t = 0; t < nTouched; t++) {
        const j = touched[t];
        ring[tgt + j] = Math.fround(ring[tgt + j] + acc[j]);
        acc[j] = 0;
      }
    }

    this.slot = (this.slot + 1) % (this.D + 1);
    this.stepCount += 1;
    this.tMs = this.stepCount * this.p.dt;
    this.lastSpikes = spikes.subarray(0, count);
    return this.lastSpikes;
  }

  /** Run for a duration in ms. Returns total spike count. */
  run(durationMs) {
    const steps = Math.round(durationMs / this.p.dt);
    let total = 0;
    for (let s = 0; s < steps; s++) total += this.step().length;
    return total;
  }
}

/**
 * Decode the exported connectome.bin.
 * Layout: int32 indptr[n+1], int32 indices_delta[nnz], int16 weights[nnz].
 */
export function decodeConnectome(buffer, n, nnz) {
  const indptr = new Int32Array(buffer, 0, n + 1);
  const deltaBytes = (n + 1) * 4;
  const delta = new Int32Array(buffer, deltaBytes, nnz);
  const weights = new Int16Array(buffer, deltaBytes + nnz * 4, nnz);

  // undo the per-row delta coding
  const indices = new Int32Array(nnz);
  indices.set(delta);
  for (let r = 0; r < n; r++) {
    const a = indptr[r], b = indptr[r + 1];
    let acc = 0;
    for (let k = a; k < b; k++) { acc += indices[k]; indices[k] = acc; }
  }
  return { n, nnz, indptr, indices, weights };
}
