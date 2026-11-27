"""
Phase 2C — Bilateral neural symmetry diagnostic.

Purpose
-------
Find out whether the original left/right drift comes from:

1. stochastic differences in bilateral Poisson sensory input, or
2. a persistent asymmetry in the real FlyWire network/readout.

Two conditions are tested:

A. INDEPENDENT:
   Left and right ORNs receive independent Poisson events at the same rate.

B. MATCHED:
   Left and right ORNs receive the same random event pattern, while their
   spikes still enter their own real FlyWire neurons and propagate through the
   real connectome.

Nothing in the brain, readout, or body is altered for the test.
Left/right body steering is ignored. No reward or plasticity is used.

The important output is the neural left/right imbalance:
    L, R, R-L

and whether that imbalance remains when the sensory spike patterns are matched.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from brain.neurons.registry import load_connectome
from brain.motor.descending import DescendingReadout
from brain.sensory.modalities import BY_KEY
from fly.body.fly_body import FlyBody
from simulation.engine.lif_engine import LIFEngine
from simulation.environment.world import World


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SIMULATION_CHUNK_MS = 100.0
TEST_DURATION_MS = 5_000.0
TRIALS = 3

ODOR_MODALITY_KEY = "odor_vinegar"
SENSORY_DRIVE_FRACTION = 0.50

BALANCED_SIDE_COUNT = 129
START_X_FRACTION = 0.50
START_Y_FRACTION = 0.50
START_HEADING_DEG = 0.0

OUTPUT_CSV = Path("bilateral_symmetry_diagnostic.csv")


# ---------------------------------------------------------------------------
# Matched random source
# ---------------------------------------------------------------------------

class PairedRNG:
    """Return identical uniforms to left and right ORN positions.

    The experiment's Poisson target list is ordered:
        [all left indices, all right indices]

    Therefore the same 129 random values are returned for both halves.
    LIFEngine only calls rng.random() for Poisson stimulation, so this
    isolates bilateral sensory spike randomness without changing the engine.
    """

    def __init__(self, seed: int, side_count: int):
        self._rng = np.random.default_rng(seed)
        self.side_count = int(side_count)

    def random(self, size=None):
        if size != 2 * self.side_count:
            raise ValueError(
                f"PairedRNG expected size {2 * self.side_count}, got {size}"
            )
        u = self._rng.random(self.side_count)
        return np.concatenate((u, u))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_body(body: FlyBody, world: World) -> None:
    body.state.x_mm = world.width * START_X_FRACTION
    body.state.y_mm = world.height * START_Y_FRACTION
    body.state.z_mm = 0.0
    body.state.heading_deg = START_HEADING_DEG
    body.state.speed_mm_s = 0.0
    body.state.turn_rate_deg_s = 0.0
    body.state.airborne = False
    body.state.vz_mm_s = 0.0
    body.state.behaviour = "resting"
    body.state.escape_mode = ""
    body.state.wing_angle_deg = 0.0
    body.state.leg_extension = 0.0
    body.state.proboscis_extension = 0.0


def resolve_odor_indices(connectome, modality):
    cells = connectome.by_cell_types(modality.cell_types)

    left_idx = cells[
        cells["side"].astype(str) == "left"
    ]["idx"].to_numpy(dtype=np.int64)

    right_idx = cells[
        cells["side"].astype(str) == "right"
    ]["idx"].to_numpy(dtype=np.int64)

    n = min(BALANCED_SIDE_COUNT, left_idx.size, right_idx.size)

    left_idx = np.sort(left_idx)[:n]
    right_idx = np.sort(right_idx)[:n]

    return left_idx, right_idx


def one_trial(
    connectome,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    sensory_rate_hz: float,
    condition: str,
    seed: int,
):
    """Run one fresh whole-brain trial and return telemetry."""
    engine = LIFEngine(connectome, seed=seed)

    if condition == "independent":
        engine.rng = np.random.default_rng(seed)
    elif condition == "matched":
        engine.rng = PairedRNG(seed, left_idx.size)
    else:
        raise ValueError(condition)

    readout = DescendingReadout(connectome)
    body = FlyBody()
    world = World()
    reset_body(body, world)

    all_odor_idx = np.concatenate((left_idx, right_idx))
    rates = np.full(
        all_odor_idx.size,
        sensory_rate_hz,
        dtype=np.float64,
    )

    rows = []

    simulation_time_ms = 0.0

    while simulation_time_ms < TEST_DURATION_MS:
        simulation_time_ms += SIMULATION_CHUNK_MS

        engine.set_poisson(all_odor_idx, rates)
        engine.run(SIMULATION_CHUNK_MS)

        channels = readout.channels(
            engine.spike_counts,
            SIMULATION_CHUNK_MS,
        )

        forward = float(channels.get("forward", 0.0))
        backward = float(channels.get("backward", 0.0))
        left = float(channels.get("left", 0.0))
        right = float(channels.get("right", 0.0))
        bias = right - left

        body_channels = dict(channels)
        body_channels["forward"] = forward
        body_channels["backward"] = backward
        body_channels["left"] = 0.0
        body_channels["right"] = 0.0
        body_channels["turn_bias"] = 0.0
        body_channels["turn"] = 0.0

        body.update(
            dt_ms=SIMULATION_CHUNK_MS,
            channels=body_channels,
            t_ms=simulation_time_ms,
            escape_laterality=readout.escape_laterality(engine.spike_counts),
            proboscis_drive=readout.proboscis_drive(
                engine.spike_counts,
                SIMULATION_CHUNK_MS,
            ),
        )

        rows.append(
            {
                "time_ms": simulation_time_ms,
                "forward": forward,
                "backward": backward,
                "left": left,
                "right": right,
                "bias_R_minus_L": bias,
            }
        )

    engine.clear_poisson()

    arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}

    return {
        "condition": condition,
        "seed": seed,
        "mean_forward": float(np.mean(arr["forward"])),
        "mean_backward": float(np.mean(arr["backward"])),
        "mean_left": float(np.mean(arr["left"])),
        "mean_right": float(np.mean(arr["right"])),
        "mean_bias": float(np.mean(arr["bias_R_minus_L"])),
        "mean_abs_bias": float(np.mean(np.abs(arr["bias_R_minus_L"]))),
        "max_abs_bias": float(np.max(np.abs(arr["bias_R_minus_L"]))),
        "final_bias": float(arr["bias_R_minus_L"][-1]),
        "x_displacement_mm": float(body.state.x_mm - world.width * START_X_FRACTION),
        "rows": rows,
    }


def main() -> int:
    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — BILATERAL SYMMETRY DIAGNOSTIC")
    print("=" * 78)
    print()

    print("Question:")
    print("  Why does a symmetric sensory condition sometimes produce")
    print("  a left/right motor imbalance?")
    print()
    print("Condition A: independent bilateral Poisson spikes")
    print("Condition B: exactly matched bilateral spike patterns")
    print()
    print("Real connectome: YES")
    print("Reward: DISABLED")
    print("Plasticity: DISABLED")
    print("Body left/right steering: IGNORED")
    print()

    print("Loading FlyWire v783 connectome...")
    connectome = load_connectome()
    print(f"  neurons     : {connectome.n:,}")
    print(f"  connections : {connectome.w.nnz:,}")
    print()

    modality = BY_KEY[ODOR_MODALITY_KEY]
    left_idx, right_idx = resolve_odor_indices(connectome, modality)

    sensory_rate_hz = modality.max_rate_hz * SENSORY_DRIVE_FRACTION

    print("Balanced real ORN population:")
    print(f"  left  : {left_idx.size}")
    print(f"  right : {right_idx.size}")
    print(f"  rate  : {sensory_rate_hz:.2f} Hz/neuron")
    print()

    all_results = []

    # Use identical seeds across conditions and fresh engines.
    for trial in range(1, TRIALS + 1):
        seed = 1000 + trial

        print("-" * 78)
        print(f"TRIAL {trial}/{TRIALS}  seed={seed}")
        print("-" * 78)

        for condition in ("independent", "matched"):
            print(f"Running {condition.upper()}...")

            result = one_trial(
                connectome,
                left_idx,
                right_idx,
                sensory_rate_hz,
                condition,
                seed,
            )
            all_results.append(result)

            print(
                f"  forward={result['mean_forward']:.3f} "
                f"backward={result['mean_backward']:.3f} "
                f"L={result['mean_left']:.3f} "
                f"R={result['mean_right']:.3f} "
                f"R-L={result['mean_bias']:+.3f} "
                f"|R-L|={result['mean_abs_bias']:.3f} "
                f"final={result['final_bias']:+.3f}"
            )

        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    independent = [
        r for r in all_results if r["condition"] == "independent"
    ]
    matched = [
        r for r in all_results if r["condition"] == "matched"
    ]

    independent_abs = float(np.mean([r["mean_abs_bias"] for r in independent]))
    matched_abs = float(np.mean([r["mean_abs_bias"] for r in matched]))

    independent_signed = float(np.mean([r["mean_bias"] for r in independent]))
    matched_signed = float(np.mean([r["mean_bias"] for r in matched]))

    ratio = (
        matched_abs / independent_abs
        if independent_abs > 1e-12
        else float("nan")
    )

    out_path = Path.cwd() / OUTPUT_CSV

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "mean_forward",
                "mean_backward",
                "mean_left",
                "mean_right",
                "mean_bias",
                "mean_abs_bias",
                "max_abs_bias",
                "final_bias",
                "x_displacement_mm",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                k: r[k]
                for k in writer.fieldnames
            })

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    print(
        f"Independent mean |R-L| : {independent_abs:.4f}"
    )
    print(
        f"Matched mean |R-L|     : {matched_abs:.4f}"
    )
    print(
        f"Matched / independent  : {ratio:.3f}"
    )
    print()
    print(
        f"Independent signed bias : {independent_signed:+.4f}"
    )
    print(
        f"Matched signed bias     : {matched_signed:+.4f}"
    )
    print()
    print(f"CSV saved to: {out_path}")
    print()

    if np.isfinite(ratio):
        if ratio < 0.5:
            print(
                "INTERPRETATION: much of the lateral imbalance disappears "
                "when bilateral sensory spikes are matched."
            )
            print(
                "Likely primary source: stochastic sensory input."
            )
        elif ratio > 0.8:
            print(
                "INTERPRETATION: lateral imbalance largely persists "
                "despite matched sensory spikes."
            )
            print(
                "Likely source: downstream connectome dynamics and/or readout."
            )
        else:
            print(
                "INTERPRETATION: mixed result; both sensory randomness "
                "and downstream asymmetry may contribute."
            )

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
