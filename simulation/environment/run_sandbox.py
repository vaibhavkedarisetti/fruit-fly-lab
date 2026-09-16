"""
Phase 1 — Digital Drosophila walking test.

Purpose
-------
Prove that the real FlyWire v783 brain can drive translational walking
through the descending-neuron motor readout and FlyBody.

Phase 1 intentionally excludes:
    - exit location
    - distance/direction to exit
    - odor-gradient navigation
    - reward
    - runtime plasticity
    - episode success/failure logic

The brain receives a symmetric, fixed sensory drive on the real
FlyWire vinegar ORNs. This is a diagnostic motor stimulus, not a
navigation signal.

Pipeline
--------
        FIXED SYMMETRIC SENSOR DRIVE
                    |
                    v
             REAL FlyWire ORNs
                    |
                    v
             REAL FlyWire v783
                    |
                    v
          REAL descending neurons
                    |
                    v
              motor readout
                    |
                    v
              FlyBody / VNC
                    |
                    v
              visual viewer

Research constraint
-------------------
The neural system receives no exit coordinates, distance, direction,
reward, or navigation target.

The FlyBody receives only neural motor outputs. It never receives
environmental information.
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
# PHASE 1 PARAMETERS
# ============================================================================

# One whole-brain/body update.
SIMULATION_CHUNK_MS = 100.0

# Run length for the walking test.
PHASE1_DURATION_MS = 10_000.0

# Fixed symmetric sensory drive.
#
# Both antenna-side ORN populations receive the same firing rate.
# There is therefore no left/right odor gradient and no navigation cue.
SENSORY_DRIVE_FRACTION = 0.50

# Deterministic initial body state.
START_X_FRACTION = 0.20
START_Y_FRACTION = 0.50
START_HEADING_DEG = 0.0

# Threshold used only for the Phase 1 diagnostic report.
TRANSLATION_THRESHOLD_MM = 1.0


# ============================================================================
# BODY RESET
# ============================================================================

def reset_body(
    body: FlyBody,
    world: World,
) -> None:
    """
    Reset the body to a deterministic starting state.

    No environment information is passed to the body.
    """

    body.state.x_mm = (
        world.width
        * START_X_FRACTION
    )

    body.state.y_mm = (
        world.height
        * START_Y_FRACTION
    )

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
# MAIN
# ============================================================================

def main() -> int:

    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — PHASE 1 WALKING TEST")
    print("=" * 78)
    print()

    print("Phase 1 goal:")
    print("  Verify real FlyWire brain -> descending neurons -> body translation.")
    print()
    print("Disabled in Phase 1:")
    print("  - exit / target location")
    print("  - odor gradient")
    print("  - navigation")
    print("  - reward")
    print("  - runtime plasticity")
    print()

    # ========================================================================
    # LOAD CONNECTOME
    # ========================================================================

    print("Loading FlyWire v783 connectome...")

    connectome = load_connectome()

    print(
        f"Loaded {connectome.n:,} real FlyWire neurons."
    )

    print(
        f"Real connections: "
        f"{connectome.w.nnz:,}"
    )

    print()

    # ========================================================================
    # CREATE WHOLE-BRAIN LIF ENGINE
    # ========================================================================

    print("Creating whole-brain LIF engine...")

    engine = LIFEngine(
        connectome
    )

    print("LIF engine ready.")
    print()

    # ========================================================================
    # IMPORTANT: NO PLASTICITY IN PHASE 1
    # ========================================================================

    print("Runtime plasticity: DISABLED")
    print()

    # ========================================================================
    # MOTOR READOUT
    # ========================================================================

    readout = DescendingReadout(
        connectome
    )

    print("Descending-neuron motor readout ready.")

    print(
        f"  modelled DN types: "
        f"{len(readout.commands)}"
    )

    if readout.missing:
        print(
            "  missing DN types: "
            + ", ".join(
                readout.missing
            )
        )

    print()

    # ========================================================================
    # BODY
    # ========================================================================

    body = FlyBody()

    # ========================================================================
    # WORLD
    # ========================================================================

    world = World()

    print(
        f"Sandbox display: "
        f"{world.width:.0f} x "
        f"{world.height:.0f} mm"
    )

    print()

    # ========================================================================
    # VIEWER
    # ========================================================================

    viewer = WorldViewer(
        world
    )

    # ========================================================================
    # REAL OLFACTORY POPULATION
    # ========================================================================

    odor_modality = BY_KEY[
        "odor_vinegar"
    ]

    odor_cells = connectome.by_cell_types(
        odor_modality.cell_types
    )

    left_idx = (
        odor_cells[
            odor_cells["side"]
            .astype(str)
            == "left"
        ]["idx"]
        .to_numpy(
            dtype=np.int64
        )
    )

    right_idx = (
        odor_cells[
            odor_cells["side"]
            .astype(str)
            == "right"
        ]["idx"]
        .to_numpy(
            dtype=np.int64
        )
    )

    all_odor_idx = np.concatenate(
        [
            left_idx,
            right_idx,
        ]
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
        f"  cell types = "
        f"{'/'.join(odor_modality.cell_types)}"
    )

    print()

    # ========================================================================
    # FIXED SYMMETRIC SENSOR DRIVE
    # ========================================================================

    sensory_rate_hz = (
        odor_modality.max_rate_hz
        * SENSORY_DRIVE_FRACTION
    )

    rates = np.full(
        all_odor_idx.size,
        sensory_rate_hz,
        dtype=np.float64,
    )

    print("Phase 1 sensory input:")
    print(
        f"  left rate  = {sensory_rate_hz:.2f} Hz"
    )
    print(
        f"  right rate = {sensory_rate_hz:.2f} Hz"
    )
    print(
        "  bilateral difference = 0.00 Hz"
    )
    print()

    # ========================================================================
    # INITIAL BODY STATE
    # ========================================================================

    reset_body(
        body,
        world,
    )

    start_x = body.state.x_mm
    start_y = body.state.y_mm
    start_heading = body.state.heading_deg

    # ========================================================================
    # TELEMETRY ACCUMULATORS
    # ========================================================================

    simulation_time_ms = 0.0

    min_x = start_x
    max_x = start_x
    min_y = start_y
    max_y = start_y

    max_speed = 0.0
    locomotion_samples: list[float] = []
    turn_samples: list[float] = []
    bias_samples: list[float] = []

    # ========================================================================
    # START
    # ========================================================================

    print("=" * 78)
    print("PHASE 1 WALKING TEST RUNNING")
    print("=" * 78)
    print()

    print(
        "The fly starts facing +X with identical left/right sensory input."
    )
    print(
        "Any sustained translation must therefore come from the neural "
        "motor pathway."
    )
    print()

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    while (
        viewer.running
        and simulation_time_ms < PHASE1_DURATION_MS
    ):

        # --------------------------------------------------------------------
        # PROCESS WINDOW EVENTS
        # --------------------------------------------------------------------

        viewer.process_events()

        if not viewer.running:
            break

        # --------------------------------------------------------------------
        # ADVANCE TIME
        # --------------------------------------------------------------------

        simulation_time_ms += SIMULATION_CHUNK_MS

        # ====================================================================
        # 1. FIXED SYMMETRIC SENSOR -> REAL FLYWIRE ORNs
        # ====================================================================

        engine.set_poisson(
            all_odor_idx,
            rates,
        )

        # ====================================================================
        # 2. ADVANCE WHOLE BRAIN
        # ====================================================================

        engine.run(
            SIMULATION_CHUNK_MS
        )

        # ====================================================================
        # 3. READ REAL DESCENDING NEURONS
        # ====================================================================

        spike_counts = (
            engine.spike_counts
        )

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

        # ====================================================================
        # 4. REAL DN ACTIVITY -> BODY
        # ====================================================================

        body.update(
            dt_ms=SIMULATION_CHUNK_MS,
            channels=channels,
            t_ms=simulation_time_ms,
            escape_laterality=escape_laterality,
            proboscis_drive=proboscis_drive,
        )

        # ====================================================================
        # 5. DISPLAY BOUNDARY ONLY
        # ====================================================================

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

        # ====================================================================
        # 6. TELEMETRY
        # ====================================================================

        min_x = min(
            min_x,
            body.state.x_mm,
        )

        max_x = max(
            max_x,
            body.state.x_mm,
        )

        min_y = min(
            min_y,
            body.state.y_mm,
        )

        max_y = max(
            max_y,
            body.state.y_mm,
        )

        max_speed = max(
            max_speed,
            body.state.speed_mm_s,
        )

        # Four independent motor channels from the descending-neuron readout.
        forward = float(channels.get("forward", 0.0))
        backward = float(channels.get("backward", 0.0))
        left = float(channels.get("left", 0.0))
        right = float(channels.get("right", 0.0))

        locomotion_drive = float(
            channels.get(
                "locomotion_drive",
                forward,
            )
        )

        turn = float(
            channels.get(
                "turn",
                0.0,
            )
        )

        turn_bias = float(
            channels.get(
                "turn_bias",
                0.0,
            )
        )

        # Expected net angular command from the independent
        # left/right body motor channels.
        left_turn_rate = left * 400.0
        right_turn_rate = right * 400.0
        net_turn_rate = left_turn_rate - right_turn_rate


        locomotion_samples.append(
            locomotion_drive
        )

        turn_samples.append(
            turn
        )

        bias_samples.append(
            turn_bias
        )

        # Actual angular velocity currently applied by FlyBody.
        actual_turn_rate = float(
            body.state.turn_rate_deg_s
        )

        print(
            f"t={simulation_time_ms:6.0f}ms "
            f"forward={forward:.3f} "
            f"backward={backward:.3f} "
            f"left={left:.3f} "
            f"right={right:.3f} "
            f"net_turn={net_turn_rate:+6.2f}deg/s "
            f"actual_turn={actual_turn_rate:+7.2f}deg/s "
            f"turn={turn:.3f} "
            f"loc={locomotion_drive:.3f} "
            f"bias={turn_bias:+.3f} "
            f"speed={body.state.speed_mm_s:6.2f} "
            f"pos=({body.state.x_mm:7.1f},"
            f"{body.state.y_mm:7.1f}) "
            f"heading={body.state.heading_deg:7.2f} "
            f"behaviour={body.state.behaviour}"
        )

        # ====================================================================
        # 7. RENDER
        # ====================================================================

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
    # FINAL PHASE 1 REPORT
    # ========================================================================

    displacement_x = (
        body.state.x_mm
        - start_x
    )

    displacement_y = (
        body.state.y_mm
        - start_y
    )

    displacement = float(
        np.hypot(
            displacement_x,
            displacement_y,
        )
    )

    heading_change = (
        body.state.heading_deg
        - start_heading
    )

    mean_locomotion = (
        float(np.mean(locomotion_samples))
        if locomotion_samples
        else 0.0
    )

    mean_turn = (
        float(np.mean(turn_samples))
        if turn_samples
        else 0.0
    )

    mean_bias = (
        float(np.mean(bias_samples))
        if bias_samples
        else 0.0
    )

    print()
    print("=" * 78)
    print("PHASE 1 WALKING TEST COMPLETE")
    print("=" * 78)
    print()

    print(
        f"Simulation time       : "
        f"{simulation_time_ms:.0f} ms"
    )

    print(
        f"Start position        : "
        f"({start_x:.1f}, {start_y:.1f})"
    )

    print(
        f"Final position        : "
        f"({body.state.x_mm:.1f}, {body.state.y_mm:.1f})"
    )

    print(
        f"Net displacement      : "
        f"{displacement:.2f} mm"
    )

    print(
        f"X displacement        : "
        f"{displacement_x:+.2f} mm"
    )

    print(
        f"Y displacement        : "
        f"{displacement_y:+.2f} mm"
    )

    print(
        f"Maximum speed         : "
        f"{max_speed:.2f} mm/s"
    )

    print(
        f"Mean locomotion drive : "
        f"{mean_locomotion:.3f}"
    )

    print(
        f"Mean turn activity    : "
        f"{mean_turn:.3f}"
    )

    print(
        f"Mean turn bias        : "
        f"{mean_bias:+.3f}"
    )

    print(
        f"Heading change        : "
        f"{heading_change:+.2f} deg"
    )

    print()

    print(
        f"X range               : "
        f"{min_x:.1f} -> {max_x:.1f} mm"
    )

    print(
        f"Y range               : "
        f"{min_y:.1f} -> {max_y:.1f} mm"
    )

    print()

    if displacement >= TRANSLATION_THRESHOLD_MM:
        print(
            "RESULT: TRANSLATION DETECTED"
        )
        print(
            "Phase 1 motor pathway produced measurable body movement."
        )
    else:
        print(
            "RESULT: NO MEANINGFUL TRANSLATION"
        )
        print(
            "Phase 1 motor pathway still needs debugging."
        )

    print()
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
