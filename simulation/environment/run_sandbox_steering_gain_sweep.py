"""
Phase 2E — Separated locomotion/steering diagnostic.

Goal
----
Test the separated engineering interface: bilateral DNa activity provides
locomotion magnitude while its left/right difference provides steering.

This is NOT a navigation experiment.

The brain still receives only real FlyWire olfactory sensory input.
The body receives four explicit motor channels. The forward command is
re-derived from bilateral DNa activity using the mean rather than the previous
max rule. Legacy turn/turn_bias fields are intentionally omitted.

Motor channels used:
    forward  -> mean bilateral DNa activity × 4
    backward -> MDN readout
    left/right -> DNa lateral imbalance
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from brain.neurons.registry import load_connectome
from brain.motor.descending import DescendingReadout
from brain.sensory.modalities import BY_KEY
from fly.body.fly_body import FlyBody
from simulation.engine.lif_engine import LIFEngine
from simulation.environment.viewer import WorldViewer
from simulation.environment.world import World


# ============================================================================
# PARAMETERS
# ============================================================================

SIMULATION_CHUNK_MS = 100.0
TEST_DURATION_MS = 10_000.0

# Use the real supported olfactory modality.
ODOR_MODALITY_KEY = "odor_vinegar"

# Fixed sensory intensity for this diagnostic.
SENSORY_DRIVE_FRACTION = 0.50

# Deterministic body start.
START_X_FRACTION = 0.50
START_Y_FRACTION = 0.50
START_HEADING_DEG = 0.0

# Use equal numbers of real ORNs on the two sides to avoid the
# 137-vs-129 population-count confound seen earlier.
BALANCE_ODOR_POPULATION = True
BALANCED_SIDE_COUNT = 129


# ============================================================================
# BODY RESET
# ============================================================================

# Experiment configuration
START_X_MM = 500.0
START_Y_MM = 350.0


def reset_body(
    body: FlyBody,
    world: World,
) -> None:
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


# ============================================================================
# REAL FLYWIRE ODOR POPULATION
# ============================================================================

def resolve_odor_indices(
    connectome,
    modality,
) -> tuple[np.ndarray, np.ndarray]:
    cells = connectome.by_cell_types(
        modality.cell_types
    )

    left_idx = (
        cells[
            cells["side"].astype(str) == "left"
        ]["idx"]
        .to_numpy(dtype=np.int64)
    )

    right_idx = (
        cells[
            cells["side"].astype(str) == "right"
        ]["idx"]
        .to_numpy(dtype=np.int64)
    )

    if BALANCE_ODOR_POPULATION:
        n = min(
            BALANCED_SIDE_COUNT,
            left_idx.size,
            right_idx.size,
        )
        left_idx = np.sort(left_idx)[:n]
        right_idx = np.sort(right_idx)[:n]

    return left_idx, right_idx


# ============================================================================
# MAIN
# ============================================================================

STEERING_GAINS = (1.0, 0.5, 0.25)


def run_one_gain(connectome, left_idx, right_idx, sensory_rate_hz, steering_gain, seed):
    """Run one clean four-channel trial at a specified steering gain."""
    engine = LIFEngine(connectome, seed=seed)
    readout = DescendingReadout(connectome)

    all_odor_idx = np.concatenate((left_idx, right_idx))
    rates = np.full(all_odor_idx.size, sensory_rate_hz, dtype=np.float64)

    body = FlyBody()
    body.state.x_mm = START_X_MM
    body.state.y_mm = START_Y_MM
    body.state.heading_deg = START_HEADING_DEG

    simulation_time_ms = 0.0
    samples = []

    while simulation_time_ms < TEST_DURATION_MS:
        simulation_time_ms += SIMULATION_CHUNK_MS

        engine.set_poisson(all_odor_idx, rates)
        engine.run(SIMULATION_CHUNK_MS)

        channels = readout.channels(
            engine.spike_counts,
            SIMULATION_CHUNK_MS,
        )

        forward = float(channels.get("forward", channels.get("locomotion_drive", 0.0)))
        backward = float(channels.get("backward", 0.0))
        left_raw = float(channels.get("left", 0.0))
        right_raw = float(channels.get("right", 0.0))

        # Only the steering authority is scaled.
        left_cmd = np.clip(left_raw * steering_gain, 0.0, 1.0)
        right_cmd = np.clip(right_raw * steering_gain, 0.0, 1.0)

        body_channels = {
            "forward": forward,
            "backward": backward,
            "left": float(left_cmd),
            "right": float(right_cmd),
        }

        body.update(
            SIMULATION_CHUNK_MS,
            body_channels,
            simulation_time_ms,
        )

        samples.append(
            (
                simulation_time_ms,
                forward,
                backward,
                left_raw,
                right_raw,
                right_raw - left_raw,
                body.state.x_mm,
                body.state.y_mm,
                body.state.heading_deg,
            )
        )

    engine.clear_poisson()

    arr = np.asarray(samples, dtype=float)
    start_x = START_X_MM
    start_y = START_Y_MM
    dx = float(arr[-1, 6] - start_x)
    dy = float(arr[-1, 7] - start_y)
    distance = float(np.hypot(dx, dy))

    return {
        "gain": steering_gain,
        "final_x": float(arr[-1, 6]),
        "final_y": float(arr[-1, 7]),
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "final_heading": float(arr[-1, 8]),
        "mean_forward": float(np.mean(arr[:, 1])),
        "max_forward": float(np.max(arr[:, 1])),
        "mean_backward": float(np.mean(arr[:, 2])),
        "max_backward": float(np.max(arr[:, 2])),
        "mean_abs_turn_bias": float(np.mean(np.abs(arr[:, 5]))),
    }


def main() -> int:
    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — STEERING GAIN SWEEP")
    print("=" * 78)
    print()
    print("Goal:")
    print("  Determine whether the remaining curvature is mainly caused by")
    print("  excessive engineering steering gain.")
    print()
    print("Fixed:")
    print("  - real FlyWire v783 connectome")
    print("  - symmetric 75 Hz bilateral ORN input")
    print("  - separated forward and steering channels")
    print("  - no navigation")
    print("  - no reward")
    print("  - no runtime plasticity")
    print("  - no legacy turn/turn_bias channel")
    print()
    print("Steering gains:", ", ".join(map(str, STEERING_GAINS)))
    print("Paired neural seed: 3000 (same stochastic sensory/brain seed for every gain)")
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

    results = []

    for i, gain in enumerate(STEERING_GAINS, start=1):
        print("-" * 78)
        print(f"GAIN TEST {i}/{len(STEERING_GAINS)}  steering_gain={gain:.2f}")
        print("-" * 78)

        result = run_one_gain(
            connectome,
            left_idx,
            right_idx,
            sensory_rate_hz,
            gain,
            seed=3000,
        )
        results.append(result)

        print(f"  mean forward      : {result['mean_forward']:.3f}")
        print(f"  mean backward     : {result['mean_backward']:.3f}")
        print(f"  mean |turn bias|  : {result['mean_abs_turn_bias']:.3f}")
        print(f"  final X           : {result['final_x']:.2f} mm")
        print(f"  final Y           : {result['final_y']:.2f} mm")
        print(f"  displacement      : {result['distance']:.2f} mm")
        print(f"  final heading     : {result['final_heading']:.2f} deg")
        print()

    print("=" * 78)
    print("STEERING GAIN SUMMARY")
    print("=" * 78)
    print()
    def signed_heading_error_deg(heading_deg: float) -> float:
        return ((heading_deg + 180.0) % 360.0) - 180.0

    for r in results:
        r["final_heading_error_deg"] = signed_heading_error_deg(r["final_heading"])
        r["abs_final_heading_error_deg"] = abs(r["final_heading_error_deg"])

    print(
        f"{'Gain':>8} {'Mean|bias|':>12} {'Mean Fwd':>10} "
        f"{'Distance':>12} {'Final heading':>15} {'Heading err':>13}"
    )
    for r in results:
        print(
            f"{r['gain']:>8.2f} {r['mean_abs_turn_bias']:>12.3f} "
            f"{r['mean_forward']:>10.3f} {r['distance']:>12.2f} "
            f"{r['final_heading']:>15.2f} {r['final_heading_error_deg']:>13.2f}"
        )


    best = min(results, key=lambda r: r["abs_final_heading_error_deg"])

    print()
    print(
        f"Smallest final heading error: gain={best['gain']:.2f} "
        f"({best['final_heading_error_deg']:+.2f} deg from initial heading)"
    )
    print()
    print("Interpretation:")
    print("  If heading magnitude falls strongly as gain falls while forward")
    print("  remains high, the remaining curvature is primarily a motor-interface")
    print("  scaling issue.")
    print("  If heading stays similar despite lower gains, the lateral signal")
    print("  itself needs deeper investigation.")
    print()

    out_path = Path.cwd() / "steering_gain_sweep.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV saved to: {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
