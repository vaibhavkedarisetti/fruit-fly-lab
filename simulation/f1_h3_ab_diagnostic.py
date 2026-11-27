"""In-memory F1/H3 preservation diagnostic for the recorded v783 protocols.

This module compares the installed F1/H3 conditional correction with the
otherwise-identical baseline path.  It does not edit the engine, topology,
synaptic signs, or published parameters.  The off arm only clears the existing
per-instance gate after its normal eligibility calculation.

Run with::

    python -m simulation.f1_h3_ab_diagnostic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain.neuron_models.lif import DEFAULT
from brain.neurons.registry import load_connectome
from simulation.benchmark_sugar_mn9 import (
    DURATION_MS as SUGAR_DURATION_MS,
    MN9_ROOT_ID,
    SEED as SUGAR_SEED,
    STIMULUS_RATE_HZ as SUGAR_RATE_HZ,
    benchmark_population,
)
from simulation.diagnostics.l5_ignition import connected_odor_vinegar_candidates
from simulation.diagnostics.stability import select_odor_vinegar
from simulation.engine.lif_engine import LIFEngine


L4 = {"seed": 0, "rate_hz": 1.0, "n_driven": 258, "duration_ms": 1500.0,
      "measurement_start_ms": 1000.0, "measurement_end_ms": 1500.0, "post_ms": 100.0}
L5 = {"seed": 0, "rate_hz": 20.0, "duration_ms": 1500.0,
      "measurement_start_ms": 1000.0, "measurement_end_ms": 1500.0, "post_ms": 100.0}


def _engine(connectome, driven, rate_hz, duration_ms, seed, f1_h3_enabled):
    engine = LIFEngine(connectome, seed=seed)
    if len(driven):
        engine.set_poisson(driven, rate_hz)
    engine.set_run_duration(duration_ms)
    eligible = bool(engine._f1_h3_active)
    # The control is deliberately per-instance and exists only for this run.
    # It does not alter source, parameters, signs, or graph data.
    engine._f1_h3_active = bool(f1_h3_enabled and eligible)
    return engine, eligible


def _sugar(connectome, driven, f1_h3_enabled):
    engine, eligible = _engine(
        connectome, driven, SUGAR_RATE_HZ, SUGAR_DURATION_MS, SUGAR_SEED,
        f1_h3_enabled,
    )
    for _ in range(int(round(SUGAR_DURATION_MS / DEFAULT.dt))):
        engine.step()
    mn9 = connectome.idx(MN9_ROOT_ID)
    return {
        "eligible": eligible,
        "active": bool(engine._f1_h3_active),
        "mn9_spike_count": int(engine.spike_counts[mn9]),
        "mn9_firing_rate_hz": float(engine.spike_counts[mn9] / (SUGAR_DURATION_MS * 1e-3)),
        "total_spikes": int(engine.spike_counts.sum()),
        "active_neurons": int(np.count_nonzero(engine.spike_counts)),
    }


def _stability(connectome, driven, spec, f1_h3_enabled):
    engine, eligible = _engine(
        connectome, driven, spec["rate_hz"], spec["duration_ms"], spec["seed"],
        f1_h3_enabled,
    )
    start = int(round(spec["measurement_start_ms"] / DEFAULT.dt))
    end = int(round(spec["measurement_end_ms"] / DEFAULT.dt))
    late_spikes = 0
    late_active = np.zeros(connectome.n, dtype=bool)
    for step in range(int(round(spec["duration_ms"] / DEFAULT.dt))):
        spikes = engine.step()
        if start <= step < end:
            late_spikes += int(spikes.size)
            late_active[spikes] = True
    engine.clear_poisson()
    post_spikes = sum(
        int(engine.step().size)
        for _ in range(int(round(spec["post_ms"] / DEFAULT.dt)))
    )
    late_duration_s = (spec["measurement_end_ms"] - spec["measurement_start_ms"]) * 1e-3
    post_duration_s = spec["post_ms"] * 1e-3
    return {
        "eligible": eligible,
        "active": bool(engine._f1_h3_active),
        "measurement_spikes": late_spikes,
        "brain_rate_hz": late_spikes / late_duration_s / connectome.n,
        "active_fraction": float(np.count_nonzero(late_active) / connectome.n),
        "post_drive_spikes": post_spikes,
        "post_drive_rate_hz": post_spikes / post_duration_s / connectome.n,
    }


def _compare(on, off):
    rows = {}
    for key, value in on.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        delta = float(value) - float(off[key])
        rows[key] = {
            "f1_h3_on": value,
            "f1_h3_off": off[key],
            "on_minus_off": delta,
            "relative_to_off": (delta / abs(float(off[key]))) if off[key] else None,
        }
    return rows


def _pair(fn, *args):
    on = fn(*args, True)
    off = fn(*args, False)
    return {"f1_h3_on": on, "f1_h3_off": off, "comparison": _compare(on, off)}


def main() -> int:
    connectome = load_connectome()
    roots, missing = benchmark_population(connectome)
    sugar_driven = connectome.indices(roots)
    l4_driven = select_odor_vinegar(connectome, L4["n_driven"])
    l5_driven = connected_odor_vinegar_candidates(connectome)
    result = {
        "purpose": "in-memory F1/H3 preservation comparison; no model changes",
        "dataset": connectome.dataset,
        "sugar": {
            "root_ids": list(roots), "missing_v783_roots": list(missing),
            "seed": SUGAR_SEED, "duration_ms": SUGAR_DURATION_MS,
            "rate_hz": SUGAR_RATE_HZ,
            "rest": _pair(_sugar, connectome, np.empty(0, dtype=np.int64)),
            "stimulated": _pair(_sugar, connectome, sugar_driven),
        },
        "l4": {"configuration": L4, "result": _pair(_stability, connectome, l4_driven, L4)},
        "l5": {
            "configuration": L5,
            "k1": _pair(_stability, connectome, l5_driven[:1], L5 | {"k": 1}),
            "k2": _pair(_stability, connectome, l5_driven[:2], L5 | {"k": 2}),
        },
    }
    out = Path("simulation/outputs/f1_h3_ab_diagnostic.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
