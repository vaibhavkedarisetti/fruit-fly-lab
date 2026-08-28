"""
Fruit Fly Laboratory -- web server.

Runs the whole-brain simulation in a background thread and streams telemetry to
the browser over a WebSocket. The browser is a display and a control panel; it
contains no biology.

Run:  python -m visualization.server
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from brain.neurons.registry import load_connectome
from brain.sensory.modalities import BY_KEY, census
from simulation.engine.session import Session

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Fruit Fly Laboratory")

# --------------------------------------------------------------------------- #
# Simulation runner
# --------------------------------------------------------------------------- #


class Runner:
    """Owns the Session and advances it on a background thread."""

    def __init__(self):
        self.connectome = load_connectome()
        self.session = Session(self.connectome, seed=0)
        self.lock = threading.Lock()
        self.running = False
        self.sim_ms_per_tick = 2.0
        self.latest = None
        self.thread = None
        self._stop = threading.Event()
        self._wall_start = None
        self._replay = []          # recorded frames of the current experiment

    # ------------------------------------------------------------------ loop
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            with self.lock:
                frames = self.session.advance(self.sim_ms_per_tick)
            if frames:
                f = frames[-1]
                f["wall_elapsed_s"] = round(time.perf_counter() - (self._wall_start or t0), 2)
                f["realtime_factor"] = round(
                    (f["t_ms"] / 1000.0) / max(1e-6, f["wall_elapsed_s"]), 3)
                self.latest = f
                self._replay.append({
                    "t_ms": f["t_ms"], "channels": f["channels"],
                    "body": {k: f["body"][k] for k in
                             ("x_mm", "y_mm", "z_mm", "heading_deg",
                              "behaviour", "proboscis_extension",
                              "wing_angle_deg", "airborne")},
                    "active_neurons": f["active_neurons"],
                    "dn_rates": f["dn_rates"],
                })
                if len(self._replay) > 20000:
                    del self._replay[:5000]

    # --------------------------------------------------------------- controls
    def play(self):
        with self.lock:
            self.session.paused = False
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        self.running = True

    def pause(self):
        self.running = False

    def reset(self, seed: int = 0):
        self.running = False
        time.sleep(0.05)
        with self.lock:
            self.session.reset(seed=seed)
            self.session.body.reset()
        self.latest = None
        self._replay = []
        self._wall_start = None

    # Composite: what a fly actually encounters when food is placed in front of
    # it. Each component drives a real, separately cited FlyWire population;
    # bundling them is an ENVIRONMENT description (category C/D), not a
    # behavioural rule. The response still comes entirely from the connectome.
    FOOD_COMPONENTS = ("odor_vinegar", "touch_leg_taste", "taste_sugar")

    def apply_stimulus(self, kind: str, params: dict) -> dict:
        if kind == "food":
            out = []
            for key in self.FOOD_COMPONENTS:
                r = self.apply_stimulus(key, params)
                out.append({"component": key, **r})
            n = sum(x.get("n_neurons", 0) for x in out)
            return {"ok": True, "composite": "food", "components": out,
                    "n_neurons": n,
                    "note": ("food odour + tarsal contact chemosensation + "
                             "proboscis sugar, driven simultaneously")}
        with self.lock:
            if kind == "looming":
                s = self.session.add_looming(
                    azimuth_deg=float(params.get("azimuth_deg", 45.0)),
                    elevation_deg=float(params.get("elevation_deg", 0.0)),
                    half_size_mm=float(params.get("half_size_mm", 5.0)),
                    speed_mm_s=float(params.get("speed_mm_s", 250.0)),
                    start_distance_mm=float(params.get("start_distance_mm", 50.0)),
                )
                return {"ok": True, "stimulus": s.state(self.session.engine.t_ms)}
            m = BY_KEY.get(kind)
            if m is None:
                return {"ok": False, "error": "unknown stimulus %r" % kind}
            if not m.supported:
                return {"ok": False, "not_modeled": True,
                        "label": m.label, "reason": m.unsupported_reason}
            s = self.session.add_modality(
                kind, intensity=float(params.get("intensity", 1.0)),
                duration_ms=float(params.get("duration_ms", 300.0)))
            from brain.sensory.modalities import resolve_neurons
            return {"ok": True, "n_neurons": int(len(resolve_neurons(m, self.connectome))),
                    "label": m.label, "citation": m.citation,
                    "stimulus": s.state(self.session.engine.t_ms)}

    def clear_stimuli(self):
        with self.lock:
            self.session.clear_stimuli()

    def silence(self, cell_type: str, on: bool) -> dict:
        with self.lock:
            c = self.connectome
            cells = c.by_cell_type(cell_type)
            if cells.empty:
                return {"ok": False, "error": "no neurons of type %r" % cell_type}
            idx = cells["idx"].to_numpy()
            if on:
                self.session.engine.silence(idx)
            else:
                self.session.engine._silenced[idx] = False
            return {"ok": True, "cell_type": cell_type,
                    "n": int(len(idx)), "silenced": on}


RUNNER: Runner = None


@app.on_event("startup")
def _startup():
    global RUNNER
    RUNNER = Runner()
    RUNNER.start()
    print("[server] connectome loaded: %s" % RUNNER.connectome)


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #


@app.get("/api/provenance")
def api_provenance():
    c = RUNNER.connectome
    return {
        "dataset": c.dataset,
        "manifest": c.manifest,
        "checksums_file": str(config.CHECKSUM_FILE),
        "session": RUNNER.session.provenance,
        "body": RUNNER.session.body.provenance,
    }


@app.get("/api/modalities")
def api_modalities():
    return {"modalities": census(RUNNER.connectome)}


@app.get("/api/state")
def api_state():
    return {
        "running": RUNNER.running,
        "t_ms": RUNNER.session.engine.t_ms,
        "sim_ms_per_tick": RUNNER.sim_ms_per_tick,
        "latest": RUNNER.latest,
    }


@app.post("/api/play")
def api_play():
    RUNNER.play()
    return {"running": True}


@app.post("/api/pause")
def api_pause():
    RUNNER.pause()
    return {"running": False}


@app.post("/api/reset")
def api_reset():
    RUNNER.reset()
    return {"ok": True, "t_ms": 0.0}


@app.post("/api/speed/{value}")
def api_speed(value: float):
    RUNNER.sim_ms_per_tick = max(0.1, min(20.0, float(value)))
    return {"sim_ms_per_tick": RUNNER.sim_ms_per_tick}


@app.post("/api/stimulus/{kind}")
async def api_stimulus(kind: str, payload: dict = None):
    res = RUNNER.apply_stimulus(kind, payload or {})
    if res.get("ok"):
        RUNNER.play()
    return res


@app.post("/api/stimulus/clear")
def api_clear():
    RUNNER.clear_stimuli()
    return {"ok": True}


@app.post("/api/silence/{cell_type}/{on}")
def api_silence(cell_type: str, on: int):
    return RUNNER.silence(cell_type, bool(int(on)))


@app.get("/api/replay")
def api_replay():
    return {"n": len(RUNNER._replay), "frames": RUNNER._replay[-4000:]}


@app.get("/api/raster")
def api_raster(last_ms: float = 300.0):
    with RUNNER.lock:
        pts = RUNNER.session.raster(last_ms)
    c = RUNNER.connectome
    sc = c.neurons["super_class"].astype(str).to_numpy()
    return {"points": [[t, i, sc[i]] for t, i in pts[-4000:]]}


@app.get("/api/circuit/{cell_type}")
def api_circuit(cell_type: str, top: int = 25):
    """Real connectivity of a cell type, for the circuit inspector."""
    c = RUNNER.connectome
    cells = c.by_cell_type(cell_type)
    if cells.empty:
        return {"error": "no neurons of type %r in %s" % (cell_type, c.dataset)}
    idx = cells["idx"].to_numpy()
    ptypes = c.neurons["primary_type"].astype(str).to_numpy()

    def agg(mat, axis_idx):
        import pandas as pd
        co = mat.tocoo()
        d = pd.DataFrame({"t": ptypes[axis_idx(co)], "s": np.abs(co.data),
                          "w": co.data})
        g = d.groupby("t").agg(syn=("s", "sum"), n=("s", "size"),
                               signed=("w", "sum")).reset_index()
        g = g.sort_values("syn", ascending=False).head(top)
        return [{"type": r.t, "synapses": int(r.syn), "connections": int(r.n),
                 "sign": "excitatory" if r.signed > 0 else "inhibitory"}
                for r in g.itertuples()]

    return {
        "cell_type": cell_type,
        "n_cells": int(len(cells)),
        "root_ids": [int(x) for x in cells["root_id"].head(20)],
        "sides": cells["side"].astype(str).value_counts().to_dict(),
        "inputs": agg(c.w[:, idx], lambda co: co.row),
        "outputs": agg(c.w[idx, :], lambda co: co.col),
        "dataset": c.dataset,
    }


@app.get("/api/neurons/positions")
def api_positions():
    """Binary Float32 xyz positions + Uint8 super-class codes for the 3D view."""
    c = RUNNER.connectome
    n = c.neurons
    xyz = n[["pos_x_nm", "pos_y_nm", "pos_z_nm"]].to_numpy(dtype=np.float64)
    xyz = np.nan_to_num(xyz) / 1000.0                      # nm -> um
    centre = np.nanmean(xyz, axis=0)
    xyz = (xyz - centre).astype(np.float32)

    classes = sorted(n["super_class"].astype(str).unique())
    code = {k: i for i, k in enumerate(classes)}
    codes = n["super_class"].astype(str).map(code).to_numpy().astype(np.uint8)

    header = json.dumps({"n": int(len(n)), "classes": classes}).encode()
    # Pad so the Float32 block starts on a 4-byte boundary (typed-array rule).
    header += b" " * ((-(len(header) + 4)) % 4)
    body = struct.pack("<I", len(header)) + header + xyz.tobytes() + codes.tobytes()
    return Response(content=body, media_type="application/octet-stream")


@app.get("/api/neurons/lookup")
def api_lookup(root_id: int):
    c = RUNNER.connectome
    try:
        i = c.idx(root_id)
    except KeyError:
        return {"error": "root_id %d is not in %s" % (root_id, c.dataset)}
    r = c.neurons.iloc[i]
    return {
        "root_id": int(r["root_id"]), "idx": int(i),
        "cell_type": str(r["primary_type"]), "super_class": str(r["super_class"]),
        "side": str(r["side"]), "nt": str(r["nt_resolved"]),
        "neuropil": str(r["primary_neuropil"]),
        "spikes": int(RUNNER.session.engine.spike_counts[i]),
        "codex_url": "https://codex.flywire.ai/app/cell_details?root_id=%d" % r["root_id"],
    }


# --------------------------------------------------------------------------- #
# WebSocket telemetry
# --------------------------------------------------------------------------- #


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    last_t = -1.0
    try:
        while True:
            f = RUNNER.latest
            if f is not None and f["t_ms"] != last_t:
                last_t = f["t_ms"]
                with RUNNER.lock:
                    raster = RUNNER.session.raster(250.0)
                    active = np.flatnonzero(
                        RUNNER.session.recorder.window_sum).astype(np.int32)
                payload = dict(f)
                payload["raster"] = raster[-1500:]
                payload["active_idx"] = active[
                    np.linspace(0, len(active) - 1, min(len(active), 3000)).astype(int)
                ].tolist() if len(active) else []
                await sock.send_text(json.dumps(payload, default=float))
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        return


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


def main():
    import uvicorn
    print("Fruit Fly Laboratory -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
