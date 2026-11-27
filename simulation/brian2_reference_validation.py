"""Independent Brian2 validation of the v783 LIFEngine.

Run only from the isolated Brian2 environment:

    .\\.brian2-venv\\Scripts\\python.exe -m simulation.brian2_reference_validation

This is a comparison harness. It does not alter model parameters, topology,
or signs. Brian2 is configured directly from the published reference model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import brian2
import numpy as np
from brian2 import Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, ms, mV

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


def _build_brian2(connectome, driven: np.ndarray, rate_hz: float):
    """Literal published-model construction over the real signed v783 graph."""
    p = DEFAULT
    brian2.start_scope()
    brian2.defaultclock.dt = p.dt * ms
    equations = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    """
    neurons = NeuronGroup(
        connectome.n, model=equations, method="linear", threshold="v > v_th",
        reset="v = v_rst; g = 0*mV", refractory="rfc",
        namespace={"v_0": p.v_0*mV, "v_rst": p.v_rst*mV, "v_th": p.v_th*mV,
                   "t_mbr": p.t_mbr*ms, "tau": p.tau*ms},
    )
    neurons.v = p.v_0*mV
    neurons.g = 0*mV
    neurons.rfc = p.t_rfc*ms

    coo = connectome.w.tocoo()
    synapses = Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=p.t_dly*ms)
    synapses.connect(i=coo.row.astype(np.int32), j=coo.col.astype(np.int32))
    synapses.w = coo.data.astype(np.float64) * p.w_syn * mV

    poisson_inputs = []
    for index in np.asarray(driven, dtype=np.int64):
        poisson_inputs.append(PoissonInput(
            neurons[int(index)], "v", N=1, rate=rate_hz*Hz,
            weight=(p.w_syn*p.f_poi)*mV,
        ))
        neurons[int(index)].rfc = 0*ms

    monitor = SpikeMonitor(neurons)
    return Network(neurons, synapses, monitor, *poisson_inputs), monitor, poisson_inputs


def _summary(spike_indices: np.ndarray, mn9_index: int, duration_ms: float) -> dict:
    counts = np.bincount(spike_indices, minlength=mn9_index + 1) if spike_indices.size else np.zeros(mn9_index + 1, dtype=np.int64)
    mn9_spikes = int(counts[mn9_index])
    return {
        "mn9_spike_count": mn9_spikes,
        "mn9_firing_rate_hz": mn9_spikes / (duration_ms * 1e-3),
        "total_spikes": int(spike_indices.size),
        "active_neurons": int(np.unique(spike_indices).size),
    }


def _lif_run(connectome, driven: np.ndarray, rate_hz: float, duration_ms: float, seed: int) -> dict:
    engine = LIFEngine(connectome, seed=seed)
    if driven.size:
        engine.set_poisson(driven, rate_hz)
    engine.set_run_duration(duration_ms)
    for _ in range(int(round(duration_ms / DEFAULT.dt))):
        engine.step()
    indices = np.repeat(np.arange(connectome.n), engine.spike_counts)
    return _summary(indices, connectome.idx(MN9_ROOT_ID), duration_ms) | {
        "f1_h3_active": bool(engine._f1_h3_active), "engine": engine,
    }


def _brian_run(connectome, driven: np.ndarray, rate_hz: float, duration_ms: float, seed: int) -> dict:
    brian2.seed(seed)
    net, monitor, _ = _build_brian2(connectome, driven, rate_hz)
    net.run(duration_ms*ms)
    indices = np.asarray(monitor.i, dtype=np.int64)
    return _summary(indices, connectome.idx(MN9_ROOT_ID), duration_ms)


def _window_summary(indices, times_ms, start_ms, end_ms, n_neurons):
    mask = (times_ms >= start_ms) & (times_ms < end_ms)
    selected = indices[mask]
    duration_s = (end_ms - start_ms) * 1e-3
    return {
        "measurement_spikes": int(selected.size),
        "brain_rate_hz": float(selected.size / duration_s / n_neurons),
        "active_fraction": float(np.unique(selected).size / n_neurons) if selected.size else 0.0,
    }


def _stability_lif(connectome, driven, spec):
    engine = LIFEngine(connectome, seed=spec["seed"])
    engine.set_poisson(driven, spec["rate_hz"])
    engine.set_run_duration(spec["duration_ms"])
    times, indices = [], []
    for step in range(int(round(spec["duration_ms"] / DEFAULT.dt))):
        spk = engine.step()
        if spk.size:
            times.append(np.full(spk.size, step * DEFAULT.dt))
            indices.append(spk)
    t = np.concatenate(times) if times else np.empty(0)
    i = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    result = _window_summary(i, t, spec["measurement_start_ms"], spec["measurement_end_ms"], connectome.n)
    engine.clear_poisson()
    post = sum(int(engine.step().size) for _ in range(int(round(spec["post_ms"] / DEFAULT.dt))))
    result.update({"post_drive_spikes": post, "post_drive_rate_hz": post / (spec["post_ms"]*1e-3) / connectome.n,
                   "f1_h3_active": bool(engine._f1_h3_active)})
    return result


def _stability_brian(connectome, driven, spec):
    brian2.seed(spec["seed"])
    net, monitor, poisson_inputs = _build_brian2(connectome, driven, spec["rate_hz"])
    net.run(spec["duration_ms"]*ms)
    t = np.asarray(monitor.t/ms)
    i = np.asarray(monitor.i, dtype=np.int64)
    result = _window_summary(i, t, spec["measurement_start_ms"], spec["measurement_end_ms"], connectome.n)
    net.remove(*poisson_inputs)
    net.run(spec["post_ms"]*ms)
    t2 = np.asarray(monitor.t/ms)
    result.update({"post_drive_spikes": int(np.count_nonzero(t2 >= spec["duration_ms"])),
                   "post_drive_rate_hz": float(np.count_nonzero(t2 >= spec["duration_ms"]) / (spec["post_ms"]*1e-3) / connectome.n)})
    return result


def _comparison(lif: dict, brian: dict, metrics: tuple[str, ...]) -> dict:
    rows = {}
    for metric in metrics:
        left, right = float(lif[metric]), float(brian[metric])
        absolute = abs(left - right)
        rows[metric] = {"lif_engine": left, "brian2": right, "absolute_difference": absolute,
                        "relative_difference": absolute / abs(right) if right else (0.0 if absolute == 0 else None)}
    return rows


def main() -> int:
    brian2.prefs.codegen.target = "numpy"
    connectome = load_connectome()
    roots, missing = benchmark_population(connectome)
    sugar_indices = connectome.indices(roots)
    started = time.perf_counter()
    sugar_rest_lif = _lif_run(connectome, np.empty(0, dtype=np.int64), SUGAR_RATE_HZ, SUGAR_DURATION_MS, SUGAR_SEED)
    sugar_rest_brian = _brian_run(connectome, np.empty(0, dtype=np.int64), SUGAR_RATE_HZ, SUGAR_DURATION_MS, SUGAR_SEED)
    sugar_lif = _lif_run(connectome, sugar_indices, SUGAR_RATE_HZ, SUGAR_DURATION_MS, SUGAR_SEED)
    sugar_brian = _brian_run(connectome, sugar_indices, SUGAR_RATE_HZ, SUGAR_DURATION_MS, SUGAR_SEED)

    l4_driven = select_odor_vinegar(connectome, L4["n_driven"])
    l4_lif = _stability_lif(connectome, l4_driven, L4)
    l4_brian = _stability_brian(connectome, l4_driven, L4)
    l5_driven = connected_odor_vinegar_candidates(connectome)
    l5 = {}
    for k in (1, 2):
        spec = L5 | {"k": k}
        l5[str(k)] = {"lif_engine": _stability_lif(connectome, l5_driven[:k], spec),
                      "brian2": _stability_brian(connectome, l5_driven[:k], spec)}

    for section in (l4_lif, l4_brian):
        section["sustained_by_operational_rule"] = bool(section["brain_rate_hz"] >= 0.5 and section["post_drive_rate_hz"] >= 0.5)
    for item in l5.values():
        for section in item.values():
            section["sustained_by_operational_rule"] = bool(section["brain_rate_hz"] >= 0.5 and section["post_drive_rate_hz"] >= 0.5)

    result = {
        "brian2_version": brian2.__version__, "brian2_codegen_target": "numpy",
        "connectome": {"dataset": connectome.dataset, "n_neurons": connectome.n, "n_connections": int(connectome.w.nnz), "n_synapses": int(np.abs(connectome.w.data).sum())},
        "parameters": DEFAULT.as_dict(), "sugar": {"stimulated_root_ids": list(roots), "missing_v783_roots": list(missing), "stimulated_indices": [int(x) for x in sugar_indices], "seed": SUGAR_SEED, "duration_ms": SUGAR_DURATION_MS, "rate_hz": SUGAR_RATE_HZ,
            "rest": {"lif_engine": {k: v for k, v in sugar_rest_lif.items() if k not in {"engine"}}, "brian2": sugar_rest_brian},
            "stimulated": {"lif_engine": {k: v for k, v in sugar_lif.items() if k not in {"engine"}}, "brian2": sugar_brian},
            "comparison": _comparison(sugar_lif, sugar_brian, ("mn9_spike_count", "mn9_firing_rate_hz", "total_spikes", "active_neurons"))},
        "l4": {"configuration": L4, "driven_indices": [int(x) for x in l4_driven], "lif_engine": l4_lif, "brian2": l4_brian,
               "comparison": _comparison(l4_lif, l4_brian, ("measurement_spikes", "brain_rate_hz", "active_fraction", "post_drive_spikes", "post_drive_rate_hz"))},
        "l5": {"configuration": L5, "candidate_indices": [int(x) for x in l5_driven], "conditions": l5},
        "wall_time_s": time.perf_counter() - started,
    }
    for item in result["l5"]["conditions"].values():
        item["comparison"] = _comparison(item["lif_engine"], item["brian2"], ("measurement_spikes", "brain_rate_hz", "active_fraction", "post_drive_spikes", "post_drive_rate_hz"))
    out = Path("simulation/outputs/brian2_reference_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
