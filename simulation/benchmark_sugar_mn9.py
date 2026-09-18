"""Reproducible full-v783 sugar GRN -> MN9 benchmark.

The stimulus roots are copied verbatim from the Figure 1D/3A list in the
published reference notebook.  That list has 21 v630 roots; one
(720575940620900446) is absent from the repository's real v783 connectome, so
the benchmark uses the remaining 20 roots.  It deliberately does not use the
broader ``functional_group('sugar_grn')`` convenience selector.

Run from the repository root:

    python -m simulation.benchmark_sugar_mn9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain.neuron_models.lif import DEFAULT
from brain.neurons.registry import load_connectome
from simulation.engine.lif_engine import LIFEngine


# Published reference notebook Figure 1D / Figure 3A (v630):
# https://github.com/philshiu/Drosophila_brain_model/blob/main/figures.ipynb
REFERENCE_SUGAR_GRN_ROOT_IDS_V630 = (
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
)
MN9_ROOT_ID = 720575940660219265
STIMULUS_RATE_HZ = 150.0
DURATION_MS = 1000.0
SEED = 0


def benchmark_population(connectome):
    """Return the published roots that persist in the loaded v783 graph."""
    present = tuple(
        root_id for root_id in REFERENCE_SUGAR_GRN_ROOT_IDS_V630
        if root_id in connectome._id2idx
    )
    missing = tuple(
        root_id for root_id in REFERENCE_SUGAR_GRN_ROOT_IDS_V630
        if root_id not in connectome._id2idx
    )
    if len(present) != 20:
        raise RuntimeError(
            "Expected exactly 20 reference sugar GRNs in v783; got "
            f"{len(present)} present and {len(missing)} missing."
        )
    return present, missing


def run_condition(connectome, stimulated_indices: np.ndarray) -> dict:
    """Run one full-network trial without retaining a large spike-train list."""
    engine = LIFEngine(connectome, seed=SEED)
    if stimulated_indices.size:
        engine.set_poisson(stimulated_indices, STIMULUS_RATE_HZ)
    engine.set_run_duration(DURATION_MS)
    for _ in range(int(round(DURATION_MS / DEFAULT.dt))):
        engine.step()

    mn9_index = connectome.idx(MN9_ROOT_ID)
    total_spikes = int(engine.spike_counts.sum())
    return {
        "mn9_spike_count": int(engine.spike_counts[mn9_index]),
        "mn9_firing_rate_hz": float(engine.spike_counts[mn9_index] / (DURATION_MS * 1e-3)),
        "total_network_spikes": total_spikes,
        "active_neuron_count": int(np.count_nonzero(engine.spike_counts)),
        "f1_h3_active": bool(engine._f1_h3_active),
    }


def benchmark() -> dict:
    connectome = load_connectome()
    roots, missing = benchmark_population(connectome)
    indices = connectome.indices(roots)
    mn9_index = connectome.idx(MN9_ROOT_ID)
    rest = run_condition(connectome, np.empty(0, dtype=np.int64))
    stimulated = run_condition(connectome, indices)
    return {
        "benchmark": "published sugar GRN -> MN9, v783 persistence subset",
        "population_source": "published reference figures.ipynb Figure 1D / Figure 3A",
        "reference_population_root_ids_v630": list(REFERENCE_SUGAR_GRN_ROOT_IDS_V630),
        "stimulated_root_ids": list(roots),
        "roots_missing_from_v783": list(missing),
        "stimulated_global_indices": [int(i) for i in indices],
        "stimulated_neuron_count": len(roots),
        "stimulus_rate_hz": STIMULUS_RATE_HZ,
        "duration_ms": DURATION_MS,
        "seed": SEED,
        "mn9_root_id": MN9_ROOT_ID,
        "mn9_global_index": mn9_index,
        "connectome": {
            "dataset": connectome.dataset,
            "n_neurons": connectome.n,
            "n_connections": int(connectome.w.nnz),
            "n_synapses": int(np.abs(connectome.w.data).sum()),
        },
        "parameters": DEFAULT.as_dict(),
        "rest": rest,
        "stimulated": stimulated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("simulation/outputs/sugar_mn9_benchmark_seed0.json"),
        help="JSON result path (default: %(default)s)",
    )
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
