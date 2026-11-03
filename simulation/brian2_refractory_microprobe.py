"""Deterministic Brian2 probe for a synaptic event during refractoriness.

The first step makes a target neuron spike.  A source event then arrives while
the target is refractory.  The probe records the target conductance after that
arrival in Brian2 and in both existing LIFEngine gate states.  It does not
change any model source or parameter.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import brian2
import numpy as np
import scipy.sparse as sp
from brian2 import SpikeGeneratorGroup, StateMonitor, Synapses, ms, mV

from brain.neuron_models.lif import DEFAULT
from simulation.engine.lif_engine import LIFEngine


EVENT_WEIGHT_MV = 3.0


def _engine_result(f1_h3_enabled: bool) -> dict:
    # A zero-edge two-neuron graph is sufficient because the delivery is placed
    # directly in the existing delay ring after a deterministic spike.
    connectome = SimpleNamespace(n=2, w=sp.csr_matrix((2, 2), dtype=np.float64))
    engine = LIFEngine(connectome, dtype=np.float64)
    target = 0
    engine.v[target] = -44.0  # deterministic threshold crossing on step one
    first = engine.step()
    assert np.array_equal(first, np.array([target]))
    assert engine.rfc_left[target] > 0
    engine._f1_h3_active = f1_h3_enabled
    engine._ring[engine._slot, target] = EVENT_WEIGHT_MV
    second = engine.step()
    return {
        "first_spikes": first.tolist(),
        "second_spikes": second.tolist(),
        "refractory_after_arrival": int(engine.rfc_left[target]),
        "g_after_arrival_mV": float(engine.g[target]),
    }


def _brian2_result(initial_v_mV: float) -> dict:
    brian2.start_scope()
    brian2.defaultclock.dt = DEFAULT.dt * ms
    equations = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    """
    target = brian2.NeuronGroup(
        1, equations, method="linear", threshold="v > v_th",
        reset="v = v_rst; g = 0*mV", refractory=DEFAULT.t_rfc * ms,
        namespace={"v_0": DEFAULT.v_0*mV, "v_rst": DEFAULT.v_rst*mV,
                   "v_th": DEFAULT.v_th*mV, "t_mbr": DEFAULT.t_mbr*ms,
                   "tau": DEFAULT.tau*ms},
    )
    target.v = initial_v_mV*mV
    source = SpikeGeneratorGroup(1, indices=[0], times=[DEFAULT.dt*ms])
    synapse = Synapses(source, target, on_pre="g += weight")
    synapse.connect()
    synapse.namespace["weight"] = EVENT_WEIGHT_MV*mV
    monitor = StateMonitor(target, ("g",), record=True, when="end")
    spikes = brian2.SpikeMonitor(target)
    net = brian2.Network(target, source, synapse, monitor, spikes)
    # Three steps guarantee the 0.1 ms source event has been delivered and
    # sampled at an end-of-step monitor slot.
    net.run(3 * DEFAULT.dt * ms)
    return {
        "spike_times_ms": [float(t/ms) for t in spikes.t],
        "g_after_arrival_mV": float(monitor.g[0, -1] / mV),
    }


def main() -> int:
    brian2.prefs.codegen.target = "numpy"
    result = {
        "brian2_version": brian2.__version__,
        "event_weight_mV": EVENT_WEIGHT_MV,
        "brian2_refractory_target": _brian2_result(-44.0),
        "brian2_nonrefractory_delivery_control": _brian2_result(DEFAULT.v_0),
        "lif_engine_f1_h3_off": _engine_result(False),
        "lif_engine_f1_h3_on": _engine_result(True),
    }
    assert result["brian2_nonrefractory_delivery_control"]["g_after_arrival_mV"] > 0.0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
