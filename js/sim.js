/*
 * Sensory encoders, descending-neuron readout, body model and session loop.
 * Direct ports of the Python modules of the same names; see those files for the
 * full provenance notes and citations.
 *
 *   simulation/stimuli/looming.py   -> LoomingStimulus       (A: exact geometry)
 *   simulation/stimuli/pulse.py     -> PulseStimulus         (C/D)
 *   brain/sensory/encoders.py       -> LoomingEncoder, PopulationEncoder (B + C)
 *   brain/motor/descending.py       -> DescendingReadout     (A + B + C)
 *   fly/body/fly_body.py            -> FlyBody               (C/D)
 *   simulation/engine/session.py    -> Session               (D)
 *
 * No behaviour is decided here. The body never sees the stimulus.
 */
'use strict';

import { LIFEngine, LIF } from './engine.js';

/* ───────────────────────── stimuli ───────────────────────── */

/** A: the angular expansion of an approaching object is exact geometry. */
export class LoomingStimulus {
  constructor(o = {}) {
    this.azimuth_deg = o.azimuth_deg ?? 0;
    this.elevation_deg = o.elevation_deg ?? 0;
    this.half_size_mm = o.half_size_mm ?? 5;
    this.speed_mm_s = o.speed_mm_s ?? 250;
    this.start_distance_mm = o.start_distance_mm ?? 50;
    this.max_half_angle_deg = o.max_half_angle_deg ?? 80;
    this.t_start_ms = o.t_start_ms ?? 0;
  }
  get lOverV() {
    return this.speed_mm_s === 0 ? Infinity
      : 1000 * this.half_size_mm / Math.abs(this.speed_mm_s);
  }
  get collisionTimeMs() {
    return this.speed_mm_s <= 0 ? Infinity
      : this.t_start_ms + 1000 * this.start_distance_mm / this.speed_mm_s;
  }
  distanceMm(t) {
    return this.start_distance_mm - this.speed_mm_s * Math.max(0, t - this.t_start_ms) / 1000;
  }
  halfAngleDeg(t) {
    if (t < this.t_start_ms) return 0;
    const d = this.distanceMm(t);
    if (d <= 0) return this.max_half_angle_deg;
    return Math.min(Math.atan(this.half_size_mm / d) * 180 / Math.PI,
                    this.max_half_angle_deg);
  }
  /** dtheta/dt in deg/s, closed form: l*v / (d^2 + l^2). */
  expansionRateDegS(t) {
    if (t < this.t_start_ms) return 0;
    if (this.halfAngleDeg(t) >= this.max_half_angle_deg) return 0;
    const d = this.distanceMm(t);
    if (d <= 0) return 0;
    const l = this.half_size_mm, v = this.speed_mm_s;
    return (l * v / (d * d + l * l)) * 180 / Math.PI;
  }
  state(t) {
    return {
      kind: 'looming', t_ms: t,
      azimuth_deg: this.azimuth_deg, elevation_deg: this.elevation_deg,
      distance_mm: Math.max(0, this.distanceMm(t)),
      half_angle_deg: this.halfAngleDeg(t),
      expansion_rate_deg_s: this.expansionRateDegS(t),
      l_over_v_ms: this.lOverV, collision_time_ms: this.collisionTimeMs,
      active: t >= this.t_start_ms,
    };
  }
}

export class PulseStimulus {
  constructor(o = {}) {
    this.modality_key = o.modality_key ?? 'unknown';
    this.intensity = o.intensity ?? 1;
    this.t_start_ms = o.t_start_ms ?? 0;
    this.duration_ms = o.duration_ms ?? Infinity;
    this.rise_ms = o.rise_ms ?? 5;
    this.fall_ms = o.fall_ms ?? 20;
  }
  level(t) {
    if (t < this.t_start_ms) return 0;
    const dt = t - this.t_start_ms;
    let ramp = (dt < this.rise_ms && this.rise_ms > 0) ? dt / this.rise_ms : 1;
    if (dt > this.duration_ms) {
      if (this.fall_ms <= 0) return 0;
      ramp = Math.min(ramp, Math.max(0, 1 - (dt - this.duration_ms) / this.fall_ms));
    }
    return Math.max(0, Math.min(1, ramp)) * this.intensity;
  }
  state(t) {
    const l = this.level(t);
    return { kind: 'pulse', t_ms: t, modality: this.modality_key,
             intensity: this.intensity, level: l, active: l > 0,
             t_start_ms: this.t_start_ms,
             duration_ms: this.duration_ms === Infinity ? null : this.duration_ms };
  }
}

export function angularDistanceDeg(az1, el1, az2, el2) {
  const R = Math.PI / 180;
  const a1 = az1 * R, e1 = el1 * R, a2 = az2 * R, e2 = el2 * R;
  const c = Math.sin(e1) * Math.sin(e2) + Math.cos(e1) * Math.cos(e2) * Math.cos(a1 - a2);
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}

/* ───────────────────────── encoders ──────────────────────── */

/**
 * B: LC4 encodes angular velocity, LPLC2 angular size gated by outward motion.
 * Receptive fields come from real FlyWire column assignments (see meta.json).
 */
export class LoomingEncoder {
  constructor(meta) {
    const t = meta.looming_tuning;
    this.tn = t;
    const idx = [], az = [], el = [], sig = [], isLC4 = [], side = [];
    for (const type of ['LC4', 'LPLC2']) {
      const rf = meta.receptive_fields[type];
      for (let k = 0; k < rf.idx.length; k++) {
        idx.push(rf.idx[k]); az.push(rf.az[k]); el.push(rf.el[k]);
        sig.push(Math.max(5, Math.min(60, rf.r[k])));
        isLC4.push(type === 'LC4' ? 1 : 0);
        side.push(rf.side[k]);
      }
    }
    this.indices = Int32Array.from(idx);
    this.az = Float64Array.from(az);
    this.el = Float64Array.from(el);
    this.sigma = Float64Array.from(sig);
    this.isLC4 = Uint8Array.from(isLC4);
    this.side = side;
    this.rates = new Float64Array(idx.length);
  }
  ratesHz(t, stim) {
    const st = stim.state(t), r = this.rates;
    r.fill(0);
    if (!st.active || st.half_angle_deg <= 0) return r;
    const theta = st.half_angle_deg;
    const dth = Math.max(0, st.expansion_rate_deg_s);
    const tn = this.tn;
    const expGate = dth / (dth + tn.lplc2_expansion_gate_deg_s);
    const lc4 = tn.lc4_max_hz * dth / (dth + tn.lc4_half_vel_deg_s);
    const lp = tn.lplc2_max_hz * theta / (theta + tn.lplc2_half_size_deg) * expGate;
    for (let i = 0; i < r.length; i++) {
      const d = angularDistanceDeg(st.azimuth_deg, st.elevation_deg, this.az[i], this.el[i]);
      const edge = Math.max(0, d - theta);
      const gate = Math.exp(-(edge * edge) / (2 * this.sigma[i] * this.sigma[i]));
      r[i] = Math.max(0, (this.isLC4[i] ? lc4 : lp) * gate);
    }
    return r;
  }
}

/** Drives a whole real population at a rate proportional to intensity. */
export class PopulationEncoder {
  constructor(modality) {
    this.modality = modality;
    this.indices = Int32Array.from(modality.idx);
    this.rates = new Float64Array(this.indices.length);
  }
  ratesHz(t, stim) {
    this.rates.fill(this.modality.max_rate_hz * stim.level(t));
    return this.rates;
  }
}

/* ───────────────────── descending readout ────────────────── */

export class DescendingReadout {
  constructor(meta) {
    this.meta = meta;
    this.half = meta.channel_half_max_hz;
    this.commands = meta.dn_commands;
    this.tracked = meta.dn_tracked;
    this.proboscis = Int32Array.from(meta.proboscis_motor_idx);
    this.channels_list = [...new Set(this.commands.map(c => c.channel))].sort();
  }
  _hz(idxList, ws, windowMs) {
    if (!idxList || !idxList.length) return 0;
    let s = 0;
    for (const i of idxList) s += ws[i];
    return s / idxList.length / (windowMs * 1e-3);
  }
  rates(ws, windowMs) {
    const out = {};
    for (const [k, idx] of Object.entries(this.tracked)) {
      if (idx.length) out[k] = this._hz(idx, ws, windowMs);
    }
    return out;
  }
  channels(ws, windowMs) {
    const acc = {}, lt = [], rt = [];
    for (const ch of this.channels_list) acc[ch] = [];
    for (const cmd of this.commands) {
      for (const side of ['left', 'right']) {
        const idx = this.tracked[`${cmd.cell_type}_${side}`];
        if (!idx || !idx.length) continue;
        const hz = this._hz(idx, ws, windowMs);
        const act = hz / (hz + this.half);
        acc[cmd.channel].push(act);
        if (cmd.channel === 'turn') (side === 'left' ? lt : rt).push(act);
      }
    }
    const res = {};
    for (const [ch, v] of Object.entries(acc)) res[ch] = v.length ? Math.max(...v) : 0;
    const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    res.turn_bias = mean(rt) - mean(lt);
    return res;
  }
  escapeLaterality(ws) {
    const tot = { left: 0, right: 0 };
    for (const cmd of this.commands) {
      if (!cmd.channel.startsWith('escape')) continue;
      for (const side of ['left', 'right']) {
        const idx = this.tracked[`${cmd.cell_type}_${side}`];
        if (idx && idx.length) {
          let s = 0; for (const i of idx) s += ws[i];
          tot[side] += s / idx.length;
        }
      }
    }
    const s = tot.left + tot.right;
    return s === 0 ? 0 : (tot.right - tot.left) / s;
  }
  proboscisDrive(ws, windowMs) {
    if (!this.proboscis.length) return 0;
    const hz = this._hz(this.proboscis, ws, windowMs);
    return hz / (hz + this.half);
  }
}

/* ─────────────────────────── body ────────────────────────── */

const ESCAPE_THRESHOLD = 0.35, LONG_MODE_THRESHOLD = 0.30;
const FREEZE_THRESHOLD = 0.30, BACKWARD_THRESHOLD = 0.30, PROBOSCIS_THRESHOLD = 0.15;
const GF_TAKEOFF_LATENCY_MS = 5.0, LONG_MODE_PREP_MS = 200.0;
const JUMP_SPEED_MM_S = 750.0, MAX_WALK_SPEED_MM_S = 25.0;
const MAX_TURN_RATE_DEG_S = 400.0, BACKWARD_SPEED_MM_S = 8.0, GRAVITY = 9810.0;

export class FlyBody {
  constructor() { this.reset(); }
  reset() {
    this.s = { x_mm: 0, y_mm: 0, z_mm: 0, heading_deg: 0, speed_mm_s: 0,
               turn_rate_deg_s: 0, wing_angle_deg: 0, proboscis_extension: 0,
               leg_extension: 0, airborne: false, vz_mm_s: 0,
               behaviour: 'resting', escape_mode: '' };
    this.armedAt = null; this.mode = null; this.complete = false; this.events = [];
  }
  update(dtMs, ch, tMs, laterality = 0, proboscisDrive = 0) {
    const s = this.s, dt = dtMs / 1000;
    const takeoff = ch.escape_takeoff ?? 0, longMode = ch.escape_long_mode ?? 0;
    const freeze = ch.stop_freeze ?? 0, backward = ch.backward_walk ?? 0;
    const turn = ch.turn_bias ?? 0;

    if (this.complete && takeoff < ESCAPE_THRESHOLD && longMode < LONG_MODE_THRESHOLD) {
      this.complete = false;
    }
    if (this.armedAt === null && !s.airborne && !this.complete) {
      if (takeoff >= ESCAPE_THRESHOLD) {
        this.armedAt = tMs; this.mode = 'short';
        this._log(tMs, 'GF (DNp01) escape command', takeoff);
      } else if (longMode >= LONG_MODE_THRESHOLD) {
        this.armedAt = tMs; this.mode = 'long';
        this._log(tMs, 'long-mode escape command (DNp02/04/11)', longMode);
      }
    }
    if (this.armedAt !== null && !s.airborne) {
      const el = tMs - this.armedAt;
      s.escape_mode = this.mode;
      if (this.mode === 'short') {
        s.behaviour = 'escape (short mode, GF-driven)';
        s.leg_extension = Math.min(1, el / GF_TAKEOFF_LATENCY_MS);
        if (el >= GF_TAKEOFF_LATENCY_MS) this._takeoff(laterality, false, tMs);
      } else {
        s.behaviour = 'escape (long mode, preparing)';
        s.wing_angle_deg = 90 * Math.min(1, el / LONG_MODE_PREP_MS);
        s.leg_extension = Math.min(1, el / LONG_MODE_PREP_MS);
        if (el >= LONG_MODE_PREP_MS) this._takeoff(laterality, true, tMs);
      }
    }

    if (s.airborne) {
      s.vz_mm_s -= GRAVITY * dt;
      s.z_mm += s.vz_mm_s * dt;
      if (s.z_mm <= 0) {
        s.z_mm = 0; s.vz_mm_s = 0; s.airborne = false; s.speed_mm_s = 0;
        s.behaviour = 'landed'; s.leg_extension = 0;
        this.armedAt = null; this.mode = null; this.complete = true;
        this._log(tMs, 'landed', 0);
      }
      this._translate(dt);
      return s;
    }

    if (this.armedAt === null) {
      if (freeze >= FREEZE_THRESHOLD) {
        s.speed_mm_s = 0; s.turn_rate_deg_s = 0; s.behaviour = 'freezing (DNp09)';
      } else if (backward >= BACKWARD_THRESHOLD) {
        s.speed_mm_s = -BACKWARD_SPEED_MM_S * backward;
        s.behaviour = 'walking backward (MDN)';
      } else if (Math.abs(turn) > 0.02) {
        s.turn_rate_deg_s = -MAX_TURN_RATE_DEG_S * turn;
        s.speed_mm_s = MAX_WALK_SPEED_MM_S * Math.min(1, Math.abs(turn) * 2);
        s.behaviour = 'turning ' + (turn > 0 ? 'right' : 'left');
      } else {
        s.speed_mm_s *= 0.9; s.turn_rate_deg_s = 0;
        if (Math.abs(s.speed_mm_s) < 0.5) { s.speed_mm_s = 0; s.behaviour = 'resting'; }
      }
      const target = proboscisDrive >= PROBOSCIS_THRESHOLD ? 1 : 0;
      s.proboscis_extension += (target - s.proboscis_extension) * Math.min(1, dtMs / 40);
      if (s.proboscis_extension > 0.5 && s.behaviour === 'resting') {
        s.behaviour = 'proboscis extension (feeding)';
      }
      s.wing_angle_deg *= 0.92;
    }
    s.heading_deg = (s.heading_deg + s.turn_rate_deg_s * dt + 360) % 360;
    this._translate(dt);
    return s;
  }
  _takeoff(lat, directed, tMs) {
    const s = this.s;
    s.airborne = true; s.vz_mm_s = JUMP_SPEED_MM_S * 0.6;
    s.speed_mm_s = JUMP_SPEED_MM_S; s.wing_angle_deg = 90;
    if (directed) s.heading_deg = (s.heading_deg - 90 * lat + 360) % 360;
    s.behaviour = `airborne (${this.mode}-mode takeoff)`;
    this._log(tMs, `takeoff (${this.mode} mode)`, 1);
  }
  _translate(dt) {
    const s = this.s, r = s.heading_deg * Math.PI / 180;
    s.x_mm += Math.cos(r) * s.speed_mm_s * dt;
    s.y_mm += Math.sin(r) * s.speed_mm_s * dt;
  }
  _log(t, event, strength) {
    this.events.push({ t_ms: Math.round(t * 10) / 10, event,
                       strength: Math.round(strength * 1000) / 1000 });
  }
  asDict() { return { ...this.s, events: this.events.slice(-12) }; }
}

/* ────────────────────────── session ──────────────────────── */

export class Session {
  static RATE_UPDATE_MS = 1.0;

  constructor(cn, meta, seed = 0, windowMs = 50) {
    this.meta = meta;
    this.n = meta.n;
    this.engine = new LIFEngine(cn, LIF, seed);
    this.readout = new DescendingReadout(meta);
    this.body = new FlyBody();
    this.windowMs = windowMs;
    this.windowFrames = Math.round(windowMs / Session.RATE_UPDATE_MS);
    this.encoders = [];
    this.loomingEncoder = new LoomingEncoder(meta);

    this.windowSum = new Int32Array(this.n);
    this.ringFrames = [];
    this.regionCode = Int32Array.from(meta.neuropil_code);
    this.regionNames = meta.neuropils;
    this.regionSizes = new Int32Array(this.regionNames.length);
    for (const c of this.regionCode) this.regionSizes[c]++;
    this.watch = new Uint8Array(this.n);
    for (const i of meta.watch_idx) this.watch[i] = 1;
    this.raster = [];
    this.history = [];
  }

  reset(seed = 0) {
    this.engine.seed = seed;
    this.engine.reset();
    this.body.reset();
    this.windowSum.fill(0);
    this.ringFrames = [];
    this.raster = [];
    this.history = [];
    this.encoders = [];
  }

  clearStimuli() { this.encoders = []; this.engine.clearPoisson(); }

  addLooming(opts) {
    const stim = new LoomingStimulus({ ...opts, t_start_ms: this.engine.tMs + (opts.delay_ms ?? 0) });
    this.encoders.push({ enc: this.loomingEncoder, stim });
    return stim;
  }

  addModality(key, intensity = 1, durationMs = 300) {
    const m = this.meta.modalities.find(x => x.key === key);
    if (!m) throw new Error(`unknown modality ${key}`);
    if (!m.supported) { const e = new Error(m.unsupported_reason); e.notModeled = true; e.label = m.label; throw e; }
    const stim = new PulseStimulus({ modality_key: key, intensity,
                                     t_start_ms: this.engine.tMs, duration_ms: durationMs });
    this.encoders.push({ enc: new PopulationEncoder(m), stim });
    return { stim, n: m.idx.length, label: m.label, citation: m.citation };
  }

  _refreshRates() {
    if (!this.encoders.length) return;
    const best = new Map();
    for (const { enc, stim } of this.encoders) {
      const r = enc.ratesHz(this.engine.tMs, stim);
      for (let k = 0; k < enc.indices.length; k++) {
        const i = enc.indices[k];
        if (!best.has(i) || best.get(i) < r[k]) best.set(i, r[k]);
      }
    }
    const idx = new Int32Array(best.size), rates = new Float64Array(best.size);
    let k = 0;
    for (const [i, r] of best) { idx[k] = i; rates[k] = r; k++; }
    this.engine.setPoisson(idx, rates);
  }

  /** Advance `durationMs`, returning one telemetry frame per RATE_UPDATE_MS. */
  advance(durationMs) {
    const frames = [];
    const blocks = Math.max(1, Math.round(durationMs / Session.RATE_UPDATE_MS));
    const steps = Math.round(Session.RATE_UPDATE_MS / this.engine.p.dt);

    for (let b = 0; b < blocks; b++) {
      this._refreshRates();
      const frameSpikes = [];
      for (let s = 0; s < steps; s++) {
        const spk = this.engine.step();
        for (let k = 0; k < spk.length; k++) frameSpikes.push(spk[k]);
      }
      for (const i of frameSpikes) this.windowSum[i]++;
      this.ringFrames.push(frameSpikes);
      while (this.ringFrames.length > this.windowFrames) {
        for (const i of this.ringFrames.shift()) this.windowSum[i]--;
      }
      const t = this.engine.tMs;
      for (const i of frameSpikes) {
        if (this.watch[i]) this.raster.push([Math.round(t * 10) / 10, i]);
      }
      if (this.raster.length > 20000) this.raster.splice(0, 5000);
      frames.push(this._telemetry(frameSpikes));
    }
    return frames;
  }

  _telemetry(spk) {
    const ws = this.windowSum, win = this.windowMs;
    const channels = this.readout.channels(ws, win);
    const lat = this.readout.escapeLaterality(ws);
    const prob = this.readout.proboscisDrive(ws, win);
    this.body.update(Session.RATE_UPDATE_MS, channels, this.engine.tMs, lat, prob);

    let active = 0, totalWin = 0;
    const regionTot = new Float64Array(this.regionNames.length);
    for (let i = 0; i < this.n; i++) {
      const c = ws[i];
      if (c > 0) { active++; totalWin += c; regionTot[this.regionCode[i]] += c; }
    }
    const regions = {};
    for (let r = 0; r < regionTot.length; r++) {
      const hz = regionTot[r] / Math.max(1, this.regionSizes[r]) / (win * 1e-3);
      if (hz > 0.01) regions[this.regionNames[r]] = hz;
    }
    let totalSpikes = 0;
    const sc = this.engine.spikeCounts;
    for (let i = 0; i < this.n; i++) totalSpikes += sc[i];

    const frame = {
      t_ms: Math.round(this.engine.tMs * 1000) / 1000,
      n_spikes: spk.length,
      active_neurons: active,
      total_spikes: totalSpikes,
      mean_rate_hz: totalWin / this.n / (win * 1e-3),
      regions,
      dn_rates: this.readout.rates(ws, win),
      channels, proboscis_drive: prob, escape_laterality: lat,
      stimuli: this.encoders.map(e => e.stim.state(this.engine.tMs)),
      body: this.body.asDict(),
    };
    this.history.push({
      t_ms: frame.t_ms, channels, dn_rates: frame.dn_rates,
      active_neurons: active,
      body: { x_mm: frame.body.x_mm, y_mm: frame.body.y_mm, z_mm: frame.body.z_mm,
              heading_deg: frame.body.heading_deg, behaviour: frame.body.behaviour,
              proboscis_extension: frame.body.proboscis_extension,
              wing_angle_deg: frame.body.wing_angle_deg, airborne: frame.body.airborne },
    });
    if (this.history.length > 20000) this.history.splice(0, 5000);
    return frame;
  }
}
