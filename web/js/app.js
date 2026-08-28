/* Fruit Fly Laboratory — browser build.
   Display and controls only. The simulation runs in js/worker.js. */
'use strict';

const $ = (s) => document.querySelector(s);

const CLASS_COLOR = {
  optic:              [ 70, 120, 190],
  central:            [110, 118, 140],
  sensory:            [ 90, 200, 160],
  visual_projection:  [130, 200, 255],
  ascending:          [160, 140, 220],
  descending:         [255, 120, 140],
  sensory_ascending:  [ 90, 190, 190],
  visual_centrifugal: [ 90, 140, 200],
  motor:              [255, 190,  90],
  endocrine:          [200, 120, 200],
};

const state = {
  meta: null, latest: null, classes: [], pos: null, codes: null, types: null, n: 0,
  active: new Set(), view: '3d', rot: { x: -0.25, y: 0.6 }, zoom: 1.0,
  drag: null, silenced: new Set(), stimText: 'no stimulus',
  replay: null, replayIdx: 0, running: false, scale: 1,
};

const worker = new Worker('js/worker.js', { type: 'module' });
const send = (cmd, extra = {}) => worker.postMessage({ cmd, ...extra });

/* ─────────────────────── worker messages ──────────────────── */
worker.onmessage = (ev) => {
  const m = ev.data;
  switch (m.type) {
    case 'status': $('#boot-note').textContent = m.message; break;
    case 'progress': {
      const pct = m.total ? (m.loaded / m.total) * 100 : 0;
      $('#boot-bar').style.width = pct + '%';
      $('#boot-note').textContent = m.total
        ? `${m.label}: ${(m.loaded / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB`
        : `${m.label}: loading…`;
      break;
    }
    case 'ready': onReady(m); break;
    case 'frame': onFrame(m.frame); break;
    case 'state':
      if ('running' in m) { state.running = m.running; $('#live-dot').classList.toggle('on', m.running); }
      if (m.reset) { state.active.clear(); state.latest = null; }
      break;
    case 'stimulus': onStimulus(m); break;
    case 'lesion':
      toast(`${m.cellType}: ${m.n} real neurons ${m.on ? 'silenced' : 'restored'}`, 2600);
      break;
    case 'replay': onReplayData(m.frames); break;
    case 'circuit': renderCircuit(m); break;
    case 'error':
      $('#boot-err').hidden = false;
      $('#boot-err').textContent = m.message;
      $('#boot-note').textContent = 'failed';
      console.error(m.message);
      break;
  }
};

function onReady(m) {
  state.meta = m.meta;
  state.n = m.meta.n;
  state.classes = m.meta.super_classes;
  state.pos = new Float32Array(m.pos);
  state.codes = new Uint8Array(m.cls);
  state.types = new Uint16Array(m.typeCodes);

  let mx = 0;
  for (let i = 0; i < state.pos.length; i++) mx = Math.max(mx, Math.abs(state.pos[i]));
  state.scale = mx || 1;

  const man = m.meta.manifest;
  $('#dataset-label').textContent =
    `${m.meta.dataset} · ${man.n_neurons.toLocaleString()} neurons · ${man.n_synapses.toLocaleString()} synapses`;
  $('#brain-count').textContent =
    `${man.n_neurons.toLocaleString()} real neurons · ${man.n_neuron_pairs.toLocaleString()} connections`;
  $('#brain-legend').innerHTML = state.classes.map(c => {
    const col = CLASS_COLOR[c] || [140, 140, 140];
    return `<span style="color:rgb(${col})">■</span> ${c}`;
  }).join('&nbsp; ');

  buildModalities(m.meta.modalities);
  buildLesions(m.meta.lesionable);
  wireControls();
  $('#boot').hidden = true;
  requestAnimationFrame(draw);
  send('circuit', { cellType: 'DNp01' });
}

function onStimulus(m) {
  if (m.cleared) { state.stimText = 'no stimulus'; return; }
  if (!m.ok) {
    if (m.notModeled) toast(`<b>Not currently modeled.</b><br>${m.reason}`, 9000);
    else toast(m.reason || 'stimulus failed', 5000);
    return;
  }
  if (m.kind === 'looming') {
    state.stimText = `looming object, azimuth ${$('#rock-side').value}° → LC4 + LPLC2`;
  } else if (m.kind === 'food') {
    state.stimText = `food placed → ${m.n} real chemosensory neurons`;
    toast(`Food placed. Driving <b>${m.n}</b> real FlyWire chemosensory neurons: ${m.label}`, 5200);
  } else {
    state.stimText = `${m.label} → ${m.n} real neurons`;
  }
  play();
}

/* ────────────────────── stimulus buttons ──────────────────── */
function buildModalities(modalities) {
  const host = $('#modality-list');
  const groups = {};
  for (const m of modalities) {
    if (m.key === 'looming') continue;
    (groups[m.group] ||= []).push(m);
  }
  host.innerHTML = '';
  for (const [g, items] of Object.entries(groups)) {
    const div = document.createElement('div');
    div.className = 'mgroup';
    div.innerHTML = `<h4>${g}</h4>`;
    for (const m of items) {
      const b = document.createElement('button');
      b.className = 'mbtn' + (m.supported ? '' : ' off');
      b.innerHTML = `<span>${m.label}</span><span class="n">${
        m.supported ? m.n_neurons + ' cells' : 'not modeled'}</span>`;
      b.title = m.supported
        ? `${m.description}\n\nDrives ${m.n_neurons} real FlyWire neurons.\nSource: ${m.citation}`
        : `Not currently modeled.\n\n${m.unsupported_reason}`;
      b.onclick = () => {
        if (!m.supported) { toast(`<b>Not currently modeled.</b><br>${m.unsupported_reason}`, 9000); return; }
        send('modality', { key: m.key, intensity: 1.0, duration_ms: 300 });
      };
      div.appendChild(b);
    }
    host.appendChild(div);
  }
}

function buildLesions(list) {
  const host = $('#lesion-list');
  host.innerHTML = '';
  for (const t of list) {
    const b = document.createElement('button');
    b.className = 'lesion'; b.textContent = t;
    b.onclick = () => {
      const on = !state.silenced.has(t);
      on ? state.silenced.add(t) : state.silenced.delete(t);
      b.classList.toggle('on', on);
      send('silence', { cellType: t, on });
    };
    host.appendChild(b);
  }
}

/* ───────────────────────── controls ───────────────────────── */
function play() { send('play'); }
function pause() { send('pause'); }

function wireControls() {
  $('#btn-play').onclick = play;
  $('#btn-pause').onclick = () => { if (state.replay) closeReplay(); pause(); };
  $('#btn-reset').onclick = () => {
    closeReplay();
    send('reset');
    state.stimText = 'no stimulus';
    state.silenced.clear();
    document.querySelectorAll('.lesion').forEach(e => e.classList.remove('on'));
  };
  $('#btn-clear').onclick = () => send('clear');
  $('#btn-rock').onclick = () => send('looming', {
    opts: {
      azimuth_deg: +$('#rock-side').value,
      speed_mm_s: +$('#rock-speed').value,
      half_size_mm: +$('#rock-size').value,
      start_distance_mm: 50,
    },
  });
  $('#btn-food').onclick = () => send('food', { intensity: 1.0, duration_ms: 600 });
  $('#btn-replay').onclick = openReplay;
  $('#btn-live').onclick = closeReplay;
  $('#replay-scrub').oninput = (e) => { state.replayIdx = +e.target.value; showReplayFrame(); };

  for (const [id, out] of [['rock-speed', 'rock-speed-v'], ['rock-size', 'rock-size-v']]) {
    $('#' + id).oninput = (e) => { $('#' + out).textContent = e.target.value; };
  }
  $('#btn-prov').onclick = showProvenance;
  $('#prov-close').onclick = () => { $('#prov-modal').hidden = true; };
  $('#circuit-go').onclick = () => send('circuit', { cellType: $('#circuit-q').value.trim() });
  $('#circuit-q').onkeydown = (e) => { if (e.key === 'Enter') $('#circuit-go').click(); };

  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      state.view = t.dataset.view;
    };
  });

  const bc = $('#brain');
  bc.onmousedown = (e) => { state.drag = { x: e.clientX, y: e.clientY }; };
  window.onmouseup = () => { state.drag = null; };
  window.onmousemove = (e) => {
    if (!state.drag || state.view !== '3d') return;
    state.rot.y += (e.clientX - state.drag.x) * 0.006;
    state.rot.x += (e.clientY - state.drag.y) * 0.006;
    state.drag = { x: e.clientX, y: e.clientY };
  };
  bc.onwheel = (e) => {
    e.preventDefault();
    state.zoom = Math.max(0.35, Math.min(6, state.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
  };
}

/* ───────────────────────── telemetry ──────────────────────── */
function onFrame(f) {
  state.latest = f;
  if (state.replay) return;
  state.active = new Set(f.active_idx || []);
  $('#s-time').textContent = f.t_ms.toFixed(1) + ' ms';
  $('#s-active').textContent = f.active_neurons.toLocaleString();
  $('#s-spikes').textContent = f.total_spikes.toLocaleString();
  $('#s-rt').textContent = f.realtime_factor
    ? (f.realtime_factor < 1 ? `${(1 / f.realtime_factor).toFixed(1)}× slower` : 'real time') : '—';

  const b = f.body || {};
  const beh = $('#behaviour');
  beh.textContent = b.behaviour || 'resting';
  beh.classList.toggle('escape', /escape|airborne/.test(b.behaviour || ''));
  $('#body-info').textContent =
    `heading ${(b.heading_deg || 0).toFixed(0)}°  z ${(b.z_mm || 0).toFixed(1)} mm` +
    (b.proboscis_extension > 0.05 ? `  proboscis ${(b.proboscis_extension * 100).toFixed(0)}%` : '');

  const st = (f.stimuli || [])[0];
  $('#stim-info').textContent = st
    ? (st.kind === 'looming'
        ? `looming  θ=${st.half_angle_deg.toFixed(1)}°  dθ/dt=${st.expansion_rate_deg_s.toFixed(0)}°/s  d=${st.distance_mm.toFixed(0)} mm`
        : `${st.modality}  level ${(st.level * 100).toFixed(0)}%`)
    : state.stimText;

  renderChannels(f.channels || {}, f.proboscis_drive || 0);
  renderDN(f.dn_rates || {});
  renderRegions(f.regions || {});
}

function renderChannels(ch, prob) {
  const rows = [
    ['Escape takeoff (Giant Fibre DNp01)', ch.escape_takeoff || 0],
    ['Escape, long mode (DNp02/04/11)', ch.escape_long_mode || 0],
    ['Stop / freeze (DNp09)', ch.stop_freeze || 0],
    ['Backward walk (MDN)', ch.backward_walk || 0],
    ['Turn (DNa01/DNa02)', Math.abs(ch.turn_bias || 0)],
    ['Proboscis extension (motor neurons)', prob],
  ];
  $('#channels').innerHTML = rows.map(([n, v]) => {
    const cls = v > 0.6 ? 'hot' : v > 0.3 ? 'warm' : '';
    return `<div class="ch"><div class="ch-top"><span>${n}</span><span>${v.toFixed(2)}</span></div>
      <div class="bar"><i class="${cls}" style="width:${Math.min(100, v * 100)}%"></i></div></div>`;
  }).join('');
}

function renderDN(rates) {
  const ent = Object.entries(rates).filter(([, v]) => v > 0.05)
    .sort((a, b) => b[1] - a[1]).slice(0, 14);
  $('#dn-rates').innerHTML = ent.length ? ent.map(([k, v]) =>
    `<div class="dnrow"><span class="nm">${k}</span>
     <span class="bar"><i class="${v > 120 ? 'hot' : v > 50 ? 'warm' : ''}"
       style="width:${Math.min(100, v / 2.5)}%"></i></span>
     <span class="v">${v.toFixed(0)} Hz</span></div>`).join('')
    : '<div class="hint">No descending neuron is firing.</div>';
}

function renderRegions(regions) {
  const ent = Object.entries(regions).sort((a, b) => b[1] - a[1]).slice(0, 14);
  const max = ent.length ? ent[0][1] : 1;
  $('#regions').innerHTML = ent.length ? ent.map(([k, v]) =>
    `<div class="dnrow"><span class="nm">${k}</span>
     <span class="bar"><i style="width:${(v / max) * 100}%"></i></span>
     <span class="v">${v.toFixed(1)}</span></div>`).join('')
    : '<div class="hint">Brain is quiet.</div>';
}

/* ─────────────────── experiment replay ────────────────────── */
function openReplay() { pause(); send('replay'); }

function onReplayData(frames) {
  if (!frames || !frames.length) { toast('Nothing recorded yet — run an experiment first.'); return; }
  state.replay = frames;
  state.replayIdx = 0;
  const sc = $('#replay-scrub');
  sc.max = frames.length - 1; sc.value = 0; sc.disabled = false;
  $('#btn-live').hidden = false;
  $('#replay-t').classList.add('on');
  document.querySelector('.arena-card').classList.add('replaying');
  showReplayFrame();
}

function closeReplay() {
  if (!state.replay) return;
  state.replay = null;
  const sc = $('#replay-scrub');
  sc.disabled = true; sc.value = 0; sc.max = 0;
  $('#btn-live').hidden = true;
  $('#replay-t').classList.remove('on');
  $('#replay-t').textContent = 'live';
  document.querySelector('.arena-card').classList.remove('replaying');
}

function showReplayFrame() {
  const f = state.replay && state.replay[state.replayIdx];
  if (!f) return;
  $('#replay-t').textContent = `${f.t_ms.toFixed(0)} ms`;
  const beh = $('#behaviour');
  beh.textContent = f.body.behaviour || 'resting';
  beh.classList.toggle('escape', /escape|airborne/.test(f.body.behaviour || ''));
  $('#body-info').textContent =
    `heading ${(f.body.heading_deg || 0).toFixed(0)}°  z ${(f.body.z_mm || 0).toFixed(1)} mm`;
  $('#s-time').textContent = f.t_ms.toFixed(1) + ' ms';
  $('#s-active').textContent = (f.active_neurons || 0).toLocaleString();
  renderChannels(f.channels || {}, 0);
  renderDN(f.dn_rates || {});
}

/* ────────────────────────── drawing ───────────────────────── */
function fit(cv) {
  const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(r.width * d) || cv.height !== Math.round(r.height * d)) {
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
  }
  return cv.getContext('2d');
}

function draw() {
  drawArena();
  state.view === '3d' ? drawBrain3D() : drawRaster();
  requestAnimationFrame(draw);
}

function drawArena() {
  const cv = $('#arena'), g = fit(cv), W = cv.width, H = cv.height;
  g.fillStyle = '#0a0d13'; g.fillRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, PX = Math.min(W, H) / 130;

  g.strokeStyle = '#161d2a'; g.lineWidth = 1;
  for (let r = 10; r <= 60; r += 10) { g.beginPath(); g.arc(cx, cy, r * PX, 0, 7); g.stroke(); }
  g.strokeStyle = '#131a26';
  for (let a = 0; a < 360; a += 45) {
    const t = a * Math.PI / 180;
    g.beginPath(); g.moveTo(cx, cy);
    g.lineTo(cx + Math.cos(t) * 62 * PX, cy + Math.sin(t) * 62 * PX); g.stroke();
  }

  const rp = state.replay ? state.replay[state.replayIdx] : null;
  const f = rp || state.latest;
  const b = (f && f.body) || { heading_deg: 0, z_mm: 0, wing_angle_deg: 0 };
  const st = rp ? null : (f && (f.stimuli || [])[0]);

  if (st && st.kind === 'looming' && st.active) {
    const az = (st.azimuth_deg - 90) * Math.PI / 180;
    const d = Math.max(0, st.distance_mm);
    const ox = cx + Math.cos(az) * d * PX, oy = cy + Math.sin(az) * d * PX;
    const rr = Math.max(3, st.half_angle_deg * 0.55 * PX);
    const grd = g.createRadialGradient(ox, oy, 0, ox, oy, rr * 2.2);
    grd.addColorStop(0, 'rgba(255,107,129,.55)');
    grd.addColorStop(1, 'rgba(255,107,129,0)');
    g.fillStyle = grd; g.beginPath(); g.arc(ox, oy, rr * 2.2, 0, 7); g.fill();
    g.fillStyle = '#e8536b'; g.beginPath(); g.arc(ox, oy, rr, 0, 7); g.fill();
    g.strokeStyle = 'rgba(232,83,107,.35)'; g.setLineDash([4, 5]);
    g.beginPath(); g.moveTo(ox, oy); g.lineTo(cx, cy); g.stroke(); g.setLineDash([]);
  }
  drawFly(g, cx, cy, PX, b);
}

function drawFly(g, cx, cy, PX, b) {
  const scale = PX * 1.5 * (1 + (b.z_mm || 0) / 22);
  g.save(); g.translate(cx, cy); g.rotate((b.heading_deg || 0) * Math.PI / 180);
  if ((b.z_mm || 0) > 0.2) {
    g.save(); g.scale(1, .45); g.fillStyle = 'rgba(0,0,0,.45)';
    g.beginPath(); g.arc(0, b.z_mm * 1.6, 3.4 * PX, 0, 7); g.fill(); g.restore();
  }
  const wing = (b.wing_angle_deg || 0) / 90;
  g.fillStyle = 'rgba(180,215,255,.30)';
  for (const s of [-1, 1]) {
    g.save(); g.rotate(s * (0.5 + wing * 0.7));
    g.beginPath(); g.ellipse(-0.6 * scale, 0, 2.1 * scale, 0.62 * scale, 0, 0, 7);
    g.fill(); g.restore();
  }
  g.fillStyle = '#2c3444';
  g.beginPath(); g.ellipse(-1.5 * scale, 0, 1.75 * scale, 1.0 * scale, 0, 0, 7); g.fill();
  g.fillStyle = '#3a4457';
  g.beginPath(); g.ellipse(0, 0, 1.25 * scale, 1.05 * scale, 0, 0, 7); g.fill();
  g.fillStyle = '#4a5568';
  g.beginPath(); g.arc(1.5 * scale, 0, 0.85 * scale, 0, 7); g.fill();
  g.fillStyle = '#d9455e';
  for (const s of [-1, 1]) {
    g.beginPath(); g.ellipse(1.65 * scale, s * 0.55 * scale, 0.55 * scale, 0.42 * scale, 0, 0, 7); g.fill();
  }
  if ((b.proboscis_extension || 0) > 0.04) {
    g.strokeStyle = '#ffb454'; g.lineWidth = 0.34 * scale; g.lineCap = 'round';
    g.beginPath(); g.moveTo(2.2 * scale, 0);
    g.lineTo((2.2 + 1.5 * b.proboscis_extension) * scale, 0); g.stroke();
  }
  g.strokeStyle = '#39435a'; g.lineWidth = Math.max(1, 0.17 * scale);
  const legOut = 1 + (b.leg_extension || 0) * 0.9;
  for (const s of [-1, 1]) for (const [ax, ay] of [[0.9, 0.8], [0, 1], [-1, 0.85]]) {
    g.beginPath(); g.moveTo(ax * scale, s * ay * 0.7 * scale);
    g.lineTo(ax * scale * 1.2, s * ay * 1.9 * scale * legOut); g.stroke();
  }
  g.restore();
}

function drawBrain3D() {
  const cv = $('#brain'), g = fit(cv), W = cv.width, H = cv.height;
  if (!state.pos) { g.fillStyle = '#0a0d13'; g.fillRect(0, 0, W, H); return; }

  const img = g.createImageData(W, H), px = img.data;
  for (let i = 0; i < px.length; i += 4) { px[i] = 10; px[i + 1] = 13; px[i + 2] = 19; px[i + 3] = 255; }

  const cy_ = Math.cos(state.rot.y), sy = Math.sin(state.rot.y);
  const cx_ = Math.cos(state.rot.x), sx = Math.sin(state.rot.x);
  const S = Math.min(W, H) * 0.42 * state.zoom / state.scale;
  const ox = W / 2, oy = H / 2;
  const pos = state.pos, codes = state.codes, cls = state.classes, act = state.active;

  const hot = [];
  for (let i = 0; i < state.n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    const x1 = x * cy_ + z * sy;
    const z1 = -x * sy + z * cy_;
    const y1 = y * cx_ - z1 * sx;
    const sxp = (ox + x1 * S) | 0, syp = (oy + y1 * S) | 0;
    if (sxp < 0 || syp < 0 || sxp >= W || syp >= H) continue;
    if (act.has(i)) { hot.push(sxp, syp, codes[i]); continue; }
    const c = CLASS_COLOR[cls[codes[i]]] || [130, 130, 130];
    const o = (syp * W + sxp) * 4;
    px[o] = Math.max(px[o], c[0] * 0.30);
    px[o + 1] = Math.max(px[o + 1], c[1] * 0.30);
    px[o + 2] = Math.max(px[o + 2], c[2] * 0.30);
  }
  g.putImageData(img, 0, 0);
  for (let k = 0; k < hot.length; k += 3) {
    const c = CLASS_COLOR[cls[hot[k + 2]]] || [255, 255, 255];
    g.fillStyle = `rgb(${Math.min(255, c[0] + 90)},${Math.min(255, c[1] + 90)},${Math.min(255, c[2] + 90)})`;
    g.fillRect(hot[k] - 1, hot[k + 1] - 1, 3, 3);
  }
  g.fillStyle = '#5d6b82'; g.font = `${11 * (window.devicePixelRatio || 1)}px ui-monospace,monospace`;
  g.fillText(`${state.active.size.toLocaleString()} active of ${state.n.toLocaleString()}`, 10, 18);
}

function drawRaster() {
  const cv = $('#brain'), g = fit(cv), W = cv.width, H = cv.height;
  g.fillStyle = '#0a0d13'; g.fillRect(0, 0, W, H);
  const f = state.latest;
  if (!f || !f.raster || !f.raster.length) {
    g.fillStyle = '#5d6b82'; g.font = '13px system-ui';
    g.fillText('No spikes yet — deliver a stimulus.', 16, 26); return;
  }
  const pts = f.raster, t1 = f.t_ms, t0 = t1 - 250;
  g.strokeStyle = '#161d2a'; g.lineWidth = 1;
  for (let k = 0; k <= 5; k++) {
    const x = (k / 5) * W; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
    g.fillStyle = '#3c4a60'; g.font = `${10 * (window.devicePixelRatio || 1)}px ui-monospace`;
    g.fillText(`${(t0 + (k / 5) * 250).toFixed(0)} ms`, x + 4, H - 6);
  }
  for (const [t, i] of pts) {
    const x = ((t - t0) / 250) * W;
    if (x < 0) continue;
    const y = (1 - (i / state.n)) * (H - 20);
    const c = CLASS_COLOR[state.classes[state.codes[i]]] || [200, 200, 200];
    g.fillStyle = `rgb(${c})`;
    g.fillRect(x, y, 2, 2);
  }
  g.fillStyle = '#8b98ad'; g.font = `${11 * (window.devicePixelRatio || 1)}px ui-monospace`;
  g.fillText('spike raster — sensory / visual projection / descending', 10, 18);
}

/* ─────────────────── circuit inspector ────────────────────── */
function renderCircuit(d) {
  const host = $('#circuit');
  if (d.error) { host.innerHTML = `<div class="hint">${d.error}</div>`; return; }
  const tbl = (rows) => rows.length ? `<table class="ctable">
      <tr><th>type</th><th>syn</th><th>cells</th></tr>
      ${rows.map(r => `<tr><td class="${r.sign === 'excitatory' ? 'exc' : 'inh'}">${r.type}</td>
        <td>${r.synapses.toLocaleString()}</td><td>${r.connections}</td></tr>`).join('')}
    </table>` : '<div class="hint">none</div>';
  host.innerHTML =
    `<div class="ctitle">${d.cell_type} — ${d.n_cells} real neurons
      (${Object.entries(d.sides).map(([k, v]) => `${v} ${k}`).join(', ')})</div>
     <div class="hint">root id ${d.root_ids[0]} ·
       <a href="https://codex.flywire.ai/app/cell_details?root_id=${d.root_ids[0]}"
          target="_blank" rel="noopener" style="color:var(--info)">verify in Codex ↗</a></div>
     <h2>Inputs</h2>${tbl(d.inputs)}
     <h2>Outputs</h2>${tbl(d.outputs)}`;
}

/* ───────────────────────── provenance ─────────────────────── */
function showProvenance() {
  $('#prov-modal').hidden = false;
  const meta = state.meta;
  const m = meta.manifest;
  const cmds = (meta.dn_commands || []).map(c =>
    `<li><b>${c.cell_type}</b> → ${c.behaviour}<br>
      <span class="muted">${c.citation} · doi:${c.doi}</span></li>`).join('');
  $('#prov-content').innerHTML = `
    <p><span class="tag A">A REAL DATA</span><span class="tag B">B PUBLISHED MODEL</span>
       <span class="tag C">C APPROXIMATION</span><span class="tag D">D OUR CODE</span></p>

    <h3><span class="tag A">A</span> Connectome</h3>
    <pre>dataset      ${m.dataset} v${m.version}
source       ${m.source_url}
neurons      ${m.n_neurons.toLocaleString()}
connections  ${m.n_neuron_pairs.toLocaleString()} neuron pairs
synapses     ${m.n_synapses.toLocaleString()}
excitatory   ${m.excitatory_neurons.toLocaleString()}
inhibitory   ${m.inhibitory_neurons.toLocaleString()}
unknown sign ${m.unknown_sign_neurons.toLocaleString()}
built        ${m.built_utc}</pre>

    <h3><span class="tag B">B</span> Neuron model</h3>
    <p>${m.model_reference}</p>
    <pre>${JSON.stringify(meta.lif, null, 1)}</pre>

    <h3><span class="tag B">B</span> Descending neuron → behaviour</h3>
    <ul>${cmds}</ul>

    <h3><span class="tag D">D</span> This build</h3>
    <p>The simulation runs entirely in your browser, in a Web Worker. The engine
       is a port of the Python one and is verified against it: the same seed and
       stimulus produce identical spike counts for all
       ${m.n_neurons.toLocaleString()} neurons
       (<code>tools/verify_web_engine.py</code>).</p>

    <h3><span class="tag C">C</span> Known limits</h3>
    <ul>
      <li>Leg and wing motor neurons are in the ventral nerve cord and are absent
          from FlyWire FAFB v783. The simulation ends at the descending neurons.</li>
      <li>FlyWire predicts six neurotransmitters; histamine is absent, so
          photoreceptor output sign cannot be modelled correctly and light is
          disabled rather than faked.</li>
      <li>Gap junctions are not in the dataset, so the Giant Fibre's electrical
          synapses onto motor neurons are not represented.</li>
      <li>The body is a kinematic approximation, not a biomechanical model.</li>
    </ul>`;
}

/* ───────────────────────────── toast ──────────────────────── */
let toastT;
function toast(html, ms = 3200) {
  const t = $('#toast'); t.innerHTML = html; t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, ms);
}

send('boot');
