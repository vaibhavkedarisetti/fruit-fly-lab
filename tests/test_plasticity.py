import numpy as np

from brain.neurons.registry import load_connectome
from simulation.engine.lif_engine import LIFEngine


def test_plasticity_changes_existing_weights_only():
    connectome = load_connectome()

    # Use a small real FlyWire subnetwork.
    idx = connectome.by_cell_types(
        ["LPLC2", "LC4", "DNp01"]
    )["idx"].to_numpy()[:150]

    small = connectome.subgraph(idx)

    engine = LIFEngine(
        small,
        seed=42,
        dtype=np.float32,
    )

    # Plasticity must initially be disabled.
    assert engine.plasticity_enabled is False

    # Record the real anatomical topology.
    original_indptr = engine.indptr.copy()
    original_indices = engine.indices.copy()

    # Record runtime synaptic weights.
    original_weights = engine.wdata.copy()

    # Enable learning.
    engine.enable_plasticity(
        learning_rate=0.001,
        trace_tau_ms=500.0,
    )

    assert engine.plasticity_enabled is True

    # Drive some real neurons so that actual spikes occur.
    lplc2 = small.by_cell_type(
        "LPLC2"
    )["idx"].to_numpy()

    engine.set_poisson(
        lplc2,
        300.0,
    )

    # Run the real LIF network.
    engine.run(100.0)

    # We need some activity before reward can affect eligibility.
    assert engine.spike_counts.sum() > 0

    # Apply positive reward.
    engine.apply_reward(1.0)

    learned_weights = engine.wdata.copy()

    # At least one existing synaptic weight should change.
    assert not np.array_equal(
        original_weights,
        learned_weights,
    )

    # The anatomical topology MUST remain identical.
    assert np.array_equal(
        original_indptr,
        engine.indptr,
    )

    assert np.array_equal(
        original_indices,
        engine.indices,
    )

    # Number of connections remains identical.
    assert engine.wdata.size == original_weights.size