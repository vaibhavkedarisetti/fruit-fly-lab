"""
Prove the browser LIF engine and the Python LIF engine are the same simulation.

Both are driven with the identical mulberry32 PRNG (tools/prng.py and the copy
in web/js/engine.js), so the comparison is exact rather than statistical: every
one of the 139,255 neurons must have the same spike count.

Three scenarios are checked:
  1. deterministic  -- no Poisson at all; an initial charge on LC4/LPLC2
                       propagates through the connectome. Tests the integrator,
                       the 1.8 ms delay ring, threshold, reset, and propagation.
  2. poisson        -- the real looming drive at 150 Hz. Adds the PRNG path.
  3. lesion         -- same, with LC4 and LPLC2 silenced. Tests silencing.

Run:  python -m tools.verify_web_engine
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import config
from brain.neurons.registry import load_connectome
from simulation.engine.lif_engine import LIFEngine
from tools.prng import Mulberry32

NODE_HARNESS = config.PROJECT_ROOT / "tools" / "verify_web_engine.mjs"


def run_python(c, scenario):
    e = LIFEngine(c, seed=scenario["seed"], dtype=np.float32)
    e.rng = Mulberry32(scenario["seed"])          # match the JS PRNG exactly
    if scenario.get("silence"):
        e.silence(np.array(scenario["silence"], dtype=np.int64))
    for i, val in scenario.get("inject_g", []):
        e.g[i] = val
    if scenario.get("poisson_idx"):
        e.set_poisson(np.array(scenario["poisson_idx"], dtype=np.int64),
                      scenario["poisson_rate"])
    t0 = time.perf_counter()
    e.run(scenario["duration_ms"])
    return e.spike_counts.copy(), time.perf_counter() - t0


def run_js(scenario):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "scenario.json"
        op = Path(td) / "out.json"
        sp.write_text(json.dumps(scenario))
        r = subprocess.run(
            ["node", str(NODE_HARNESS), str(sp), str(op)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("node harness failed:\n%s" % (r.stderr or r.stdout))
        sys.stdout.write(r.stderr)
        d = json.loads(op.read_text())
    return np.array(d["spike_counts"], dtype=np.int32), d["wall_s"]


def compare(name, c, scenario):
    print("\n=== %s ===" % name)
    py, py_t = run_python(c, scenario)
    js, js_t = run_js(scenario)

    same = np.array_equal(py, js)
    print("  python : %6d spikes across %5d neurons   (%.2f s)"
          % (py.sum(), (py > 0).sum(), py_t))
    print("  browser: %6d spikes across %5d neurons   (%.2f s)"
          % (js.sum(), (js > 0).sum(), js_t))

    if int(py.sum()) == 0:
        print("  WARNING: scenario produced no spikes; this comparison is vacuous")
        same = False
    if same:
        print("  MATCH  : spike counts identical for all %d neurons" % len(py))
    else:
        diff = np.flatnonzero(py != js)
        print("  MISMATCH on %d neurons; first 10 indices %s"
              % (len(diff), diff[:10].tolist()))
        for i in diff[:5]:
            print("      idx %6d  python %4d  browser %4d" % (i, py[i], js[i]))
    return same, py, js, py_t, js_t


def main():
    if not (config.PROJECT_ROOT / "web" / "data" / "connectome.bin").exists():
        print("web/data not exported. Run: python -m tools.export_web_connectome")
        return 1

    c = load_connectome()
    lc4 = c.by_cell_type("LC4")["idx"].to_numpy()
    lplc2 = c.by_cell_type("LPLC2")["idx"].to_numpy()
    drive = np.concatenate([lc4, lplc2]).tolist()

    scenarios = [
        ("deterministic (no PRNG involved)", {
            # A large charge on 60 real LC4/LPLC2 cells, enough to make them
            # fire and drive the network. Verified non-vacuous below.
            "seed": 3, "duration_ms": 120.0,
            "inject_g": [[int(i), 1600.0] for i in drive[:60]],
            "poisson_idx": [], "poisson_rate": 0.0, "silence": [],
        }),
        ("poisson looming drive, 150 Hz", {
            "seed": 7, "duration_ms": 120.0,
            "poisson_idx": drive, "poisson_rate": 150.0, "silence": [],
        }),
        ("looming drive with LC4+LPLC2 silenced", {
            "seed": 7, "duration_ms": 120.0,
            "poisson_idx": drive, "poisson_rate": 150.0, "silence": drive,
        }),
    ]

    results = []
    for name, sc in scenarios:
        results.append(compare(name, c, sc))

    ok = all(r[0] for r in results)
    print("\n" + "=" * 68)
    if ok:
        print("PASS: the browser engine reproduces the Python engine exactly.")
        print("      Same connectome, same equations, same spikes.")
    else:
        print("FAIL: the two engines diverge. Do not deploy.")
    print("=" * 68)

    speed = [(r[3], r[4]) for r in results]
    print("wall-clock, python vs browser: " +
          ", ".join("%.2fs / %.2fs" % s for s in speed))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
