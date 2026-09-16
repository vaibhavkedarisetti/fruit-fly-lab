"""
Phase 2B — Forward/backward-only real-brain motor test.

Goal
----
Test whether the real FlyWire brain + descending-neuron readout can produce
forward and/or backward locomotor commands, while deliberately ignoring
left/right steering.

This is NOT a navigation experiment.

The brain still receives only real FlyWire olfactory sensory input.
The body receives only the four-command motor interface, but the left/right
commands are explicitly suppressed at the body boundary for this test.

Motor channels used:
    forward  -> body forward translation
    backward -> body backward translation

Motor channels ignored:
    left
    right
"""

from __future__ import annotations

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

def main() -> int:
    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — FORWARD/BACKWARD-ONLY MOTOR TEST")
    print("=" * 78)
    print()

    print("Goal:")
    print("  Test real-brain forward/backward motor output.")
    print("  Left/right steering is intentionally ignored.")
    print()
    print("Disabled:")
    print("  - navigation")
    print("  - reward")
    print("  - runtime plasticity")
    print("  - left/right body steering")
    print()

    # ========================================================================
    # CONNECTOME
    # ========================================================================

    print("Loading FlyWire v783 connectome...")

    connectome = load_connectome()

    print(
        f"Loaded {connectome.n:,} real FlyWire neurons."
    )
    print(
        f"Real connections: {connectome.w.nnz:,}"
    )
    print()

    # ========================================================================
    # REAL BRAIN
    # ========================================================================

    print("Creating whole-brain LIF engine...")

    engine = LIFEngine(
        connectome
    )

    print("LIF engine ready.")
    print("Runtime plasticity: DISABLED")
    print()

    # ========================================================================
    # REAL DESCENDING READOUT
    # ========================================================================

    readout = DescendingReadout(
        connectome
    )

    print("Descending-neuron motor readout ready.")
    print(
        f"  modelled DN types: {len(readout.commands)}"
    )
    print()

    # ========================================================================
    # BODY + WORLD
    # ========================================================================

    body = FlyBody()
    world = World()
    viewer = WorldViewer(world)

    reset_body(
        body,
        world,
    )

    print(
        f"Sandbox display: "
        f"{world.width:.0f} x {world.height:.0f} mm"
    )
    print(
        f"Start position: "
        f"({body.state.x_mm:.1f}, {body.state.y_mm:.1f})"
    )
    print(
        f"Start heading: {body.state.heading_deg:.1f} deg"
    )
    print()

    # ========================================================================
    # REAL OLFACTORY POPULATION
    # ========================================================================

    odor_modality = BY_KEY[
        ODOR_MODALITY_KEY
    ]

    left_idx, right_idx = resolve_odor_indices(
        connectome,
        odor_modality,
    )

    all_odor_idx = np.concatenate(
        [
            left_idx,
            right_idx,
        ]
    )

    sensory_rate_hz = (
        odor_modality.max_rate_hz
        * SENSORY_DRIVE_FRACTION
    )

    rates = np.full(
        all_odor_idx.size,
        sensory_rate_hz,
        dtype=np.float64,
    )

    print("Real FlyWire odor neurons:")
    print(
        f"  left  = {left_idx.size}"
    )
    print(
        f"  right = {right_idx.size}"
    )
    print(
        f"  total = {all_odor_idx.size}"
    )
    print(
        f"  rate  = {sensory_rate_hz:.2f} Hz per neuron"
    )
    print()

    # ========================================================================
    # TELEMETRY
    # ========================================================================

    simulation_time_ms = 0.0

    start_x = body.state.x_mm
    min_x = start_x
    max_x = start_x

    max_speed = 0.0

    forward_samples: list[float] = []
    backward_samples: list[float] = []

    # ========================================================================
    # START
    # ========================================================================

    print("=" * 78)
    print("FORWARD/BACKWARD-ONLY TEST RUNNING")
    print("=" * 78)
    print()

    print(
        "The real FlyWire brain receives fixed symmetric odor drive."
    )
    print(
        "Only forward and backward motor commands are allowed to move the body."
    )
    print(
        "Left and right commands from the neural readout are ignored."
    )
    print()

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    while (
        viewer.running
        and simulation_time_ms < TEST_DURATION_MS
    ):
        viewer.process_events()

        if not viewer.running:
            break

        simulation_time_ms += SIMULATION_CHUNK_MS

        # --------------------------------------------------------------------
        # 1. REAL OLFACTORY INPUT
        # --------------------------------------------------------------------

        engine.set_poisson(
            all_odor_idx,
            rates,
        )

        # --------------------------------------------------------------------
        # 2. REAL WHOLE-BRAIN SIMULATION
        # --------------------------------------------------------------------

        engine.run(
            SIMULATION_CHUNK_MS
        )

        # --------------------------------------------------------------------
        # 3. REAL DESCENDING-NEURON MOTOR READOUT
        # --------------------------------------------------------------------

        spike_counts = engine.spike_counts

        channels = readout.channels(
            spike_counts,
            SIMULATION_CHUNK_MS,
        )

        escape_laterality = (
            readout.escape_laterality(
                spike_counts
            )
        )

        proboscis_drive = (
            readout.proboscis_drive(
                spike_counts,
                SIMULATION_CHUNK_MS,
            )
        )

        forward = float(
            channels.get("forward", 0.0)
        )

        backward = float(
            channels.get("backward", 0.0)
        )

        left_neural = float(
            channels.get("left", 0.0)
        )

        right_neural = float(
            channels.get("right", 0.0)
        )

        # --------------------------------------------------------------------
        # 4. FOUR-CHANNEL BODY INTERFACE
        #
        # Forward/backward are preserved.
        # Left/right are deliberately zeroed for this experiment.
        # --------------------------------------------------------------------

        body_channels = dict(channels)
        body_channels["forward"] = forward
        body_channels["backward"] = backward
        body_channels["left"] = 0.0
        body_channels["right"] = 0.0

        # Keep legacy fields internally consistent with the four-channel test.
        body_channels["turn_bias"] = 0.0
        body_channels["turn"] = 0.0

        body.update(
            dt_ms=SIMULATION_CHUNK_MS,
            channels=body_channels,
            t_ms=simulation_time_ms,
            escape_laterality=escape_laterality,
            proboscis_drive=proboscis_drive,
        )

        # --------------------------------------------------------------------
        # 5. BODY BOUNDARY
        # --------------------------------------------------------------------

        body.state.x_mm = max(
            0.0,
            min(
                world.width,
                body.state.x_mm,
            ),
        )

        body.state.y_mm = max(
            0.0,
            min(
                world.height,
                body.state.y_mm,
            ),
        )

        min_x = min(
            min_x,
            body.state.x_mm,
        )

        max_x = max(
            max_x,
            body.state.x_mm,
        )

        max_speed = max(
            max_speed,
            abs(body.state.speed_mm_s),
        )

        forward_samples.append(forward)
        backward_samples.append(backward)

        # --------------------------------------------------------------------
        # 6. TELEMETRY
        # --------------------------------------------------------------------

        print(
            f"t={simulation_time_ms:6.0f}ms "
            f"forward={forward:.3f} "
            f"backward={backward:.3f} "
            f"raw_left={left_neural:.3f} "
            f"raw_right={right_neural:.3f} "
            f"speed={body.state.speed_mm_s:7.2f} "
            f"x={body.state.x_mm:7.2f} "
            f"heading={body.state.heading_deg:7.2f} "
            f"behaviour={body.state.behaviour}"
        )

        viewer.render(
            fly_x=body.state.x_mm,
            fly_y=body.state.y_mm,
            heading=body.state.heading_deg,
            time_ms=simulation_time_ms,
        )

    # ========================================================================
    # CLEANUP
    # ========================================================================

    engine.clear_poisson()
    viewer.close()

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    x_displacement = (
        body.state.x_mm - start_x
    )

    mean_forward = (
        float(np.mean(forward_samples))
        if forward_samples
        else 0.0
    )

    mean_backward = (
        float(np.mean(backward_samples))
        if backward_samples
        else 0.0
    )

    max_forward = (
        float(np.max(forward_samples))
        if forward_samples
        else 0.0
    )

    max_backward = (
        float(np.max(backward_samples))
        if backward_samples
        else 0.0
    )

    print()
    print("=" * 78)
    print("FORWARD/BACKWARD-ONLY TEST COMPLETE")
    print("=" * 78)
    print()

    print(
        f"Simulation time       : "
        f"{simulation_time_ms:.0f} ms"
    )

    print(
        f"Start X               : "
        f"{start_x:.2f} mm"
    )

    print(
        f"Final X               : "
        f"{body.state.x_mm:.2f} mm"
    )

    print(
        f"X displacement        : "
        f"{x_displacement:+.2f} mm"
    )

    print(
        f"X range               : "
        f"{min_x:.2f} -> {max_x:.2f} mm"
    )

    print(
        f"Maximum |speed|       : "
        f"{max_speed:.2f} mm/s"
    )

    print(
        f"Mean forward command  : "
        f"{mean_forward:.3f}"
    )

    print(
        f"Max forward command   : "
        f"{max_forward:.3f}"
    )

    print(
        f"Mean backward command : "
        f"{mean_backward:.3f}"
    )

    print(
        f"Max backward command  : "
        f"{max_backward:.3f}"
    )

    print(
        f"Final heading         : "
        f"{body.state.heading_deg:.2f} deg"
    )

    print()

    if x_displacement > 1.0:
        print("RESULT: FORWARD TRANSLATION DETECTED")
    elif x_displacement < -1.0:
        print("RESULT: BACKWARD TRANSLATION DETECTED")
    else:
        print("RESULT: NO NET X TRANSLATION")

    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
