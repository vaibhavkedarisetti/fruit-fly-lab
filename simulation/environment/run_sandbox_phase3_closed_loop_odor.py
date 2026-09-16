"""
Phase 2A — Closed-loop odor navigation with the real FlyWire brain.

Purpose
-------
Replace the Phase 1 fixed 75 Hz sensory drive with a spatial odor field.

Pipeline
--------
    FOOD-ODOR SOURCE IN THE ENVIRONMENT
                 |
                 v
        left/right antenna sampling
                 |
                 v
       local odor concentration
                 |
                 v
          real FlyWire ORNs
                 |
                 v
          real FlyWire v783
                 |
                 v
       descending-neuron readout
                 |
                 +-------------------------------+
                 |                               |
              forward                         backward
                 |                               |
                 +-------------+-----------------+
                               |
                         left / right
                               |
                               v
                            FlyBody
                               |
                               v
                         new position
                               |
                               +----> sample odor again

Research constraints
--------------------
The brain receives ONLY ORN indices and their firing rates.

The brain does NOT receive:
    - food_x
    - food_y
    - distance_to_food
    - direction_to_food

FlyBody receives ONLY descending-neuron motor outputs.

This phase intentionally has:
    - NO reward
    - NO runtime plasticity
    - NO navigation rule
    - NO target-direction injection
    - NO learning

Four-command motor interface
----------------------------
    forward  -> +X/body-forward translation
    backward -> -X/body-backward translation
    left     -> counter-clockwise heading change
    right    -> clockwise heading change

The four-command interface is an engineering interface at the
brain -> kinematic-body boundary. The existing descending-neuron
readout remains the source of the motor signals.
"""

from __future__ import annotations

import math
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
# PHASE 2A PARAMETERS
# ============================================================================

SIMULATION_CHUNK_MS = 100.0
PHASE2_DURATION_MS = 10_000.0

# The existing supported olfactory modality in this project.
ODOR_MODALITY_KEY = "odor_vinegar"

# Environment-only odor-field parameters.
#
# This exponential distance falloff is an engineering odor-field model,
# not a measured Drosophila plume model.
ODOR_FALLOFF_MM = 250.0

# Distance from the body centre to each virtual antenna.
ANTENNA_OFFSET_MM = 60.0

# Food/odor source position is defined only in the environment.
FOOD_X_FRACTION = 0.80
FOOD_Y_FRACTION = 0.50

# Deterministic start.
START_X_FRACTION = 0.20
START_Y_FRACTION = 0.50
START_HEADING_DEG = 0.0

# IMPORTANT:
# The available real odor-neuron population is slightly asymmetric in count
# in the dataset. To remove that anatomical-count confound from this
# navigation experiment, use the same number of REAL neurons on each side.
# No synthetic neurons are created; we deterministically select an equal
# number from the real FlyWire population.
BALANCE_ODOR_POPULATION = True

# Maximum number of real neurons used on each side when balancing.
BALANCED_SIDE_COUNT = 129

# Validated engineering steering scale from the paired gain sweep.
# Forward remains unchanged; only left/right steering authority is scaled.
STEERING_GAIN = 0.25


# ============================================================================
# ENVIRONMENT MODEL
# ============================================================================

def concentration_from_distance(
    distance_mm: float,
) -> float:
    """
    Convert local distance from the food-odor source to concentration.

    Environment only.

    The result is the only odor quantity passed toward the brain.
    Distance itself is never passed into the neural simulation.
    """
    distance_mm = max(0.0, float(distance_mm))

    return float(
        math.exp(
            -distance_mm / ODOR_FALLOFF_MM
        )
    )


def antenna_positions(
    body: FlyBody,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return left and right virtual antenna positions."""
    heading_rad = math.radians(body.state.heading_deg)

    left = (
        body.state.x_mm
        + ANTENNA_OFFSET_MM * math.cos(
            heading_rad + math.pi / 2.0
        ),
        body.state.y_mm
        + ANTENNA_OFFSET_MM * math.sin(
            heading_rad + math.pi / 2.0
        ),
    )

    right = (
        body.state.x_mm
        + ANTENNA_OFFSET_MM * math.cos(
            heading_rad - math.pi / 2.0
        ),
        body.state.y_mm
        + ANTENNA_OFFSET_MM * math.sin(
            heading_rad - math.pi / 2.0
        ),
    )

    return left, right


def sample_food_odor(
    body: FlyBody,
    food_x_mm: float,
    food_y_mm: float,
) -> tuple[float, float]:
    """
    Sample odor independently at the two antenna positions.
    """
    (left_x, left_y), (right_x, right_y) = antenna_positions(body)

    left_distance = math.hypot(
        left_x - food_x_mm,
        left_y - food_y_mm,
    )

    right_distance = math.hypot(
        right_x - food_x_mm,
        right_y - food_y_mm,
    )

    return (
        concentration_from_distance(left_distance),
        concentration_from_distance(right_distance),
    )


def concentration_to_rates(
    concentration_left: float,
    concentration_right: float,
    max_rate_hz: float,
) -> tuple[float, float]:
    """
    Map local odor concentration to per-neuron ORN Poisson rates.

    This uses the project's existing linear intensity -> firing-rate
    abstraction. No extra bilateral "turn command" is injected here.
    """
    drive_left = float(
        np.clip(concentration_left, 0.0, 1.0)
    )
    drive_right = float(
        np.clip(concentration_right, 0.0, 1.0)
    )

    return (
        max_rate_hz * drive_left,
        max_rate_hz * drive_right,
    )


# ============================================================================
# BODY RESET
# ============================================================================

def reset_body(
    body: FlyBody,
    world: World,
) -> None:
    """Reset the body without passing any environment state to it."""
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

def resolve_balanced_odor_indices(
    connectome,
    odor_modality,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve real FlyWire odor neurons and optionally equalize left/right count.

    Equalization is only a control against the v783 dataset's unequal number
    of reconstructed left/right odor cells. It does not create synthetic cells.
    """
    odor_cells = connectome.by_cell_types(
        odor_modality.cell_types
    )

    left_idx = (
        odor_cells[
            odor_cells["side"].astype(str) == "left"
        ]["idx"]
        .to_numpy(dtype=np.int64)
    )

    right_idx = (
        odor_cells[
            odor_cells["side"].astype(str) == "right"
        ]["idx"]
        .to_numpy(dtype=np.int64)
    )

    if not BALANCE_ODOR_POPULATION:
        return left_idx, right_idx

    side_count = min(
        BALANCED_SIDE_COUNT,
        left_idx.size,
        right_idx.size,
    )

    # Deterministic selection: no random neurons are introduced.
    left_idx = np.sort(left_idx)[:side_count]
    right_idx = np.sort(right_idx)[:side_count]

    return left_idx, right_idx


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — PHASE 3 CLOSED-LOOP ODOR NAVIGATION TEST")
    print("=" * 78)
    print()

    print("Phase 2A goal:")
    print("  Replace fixed sensory drive with a spatial food-odor field.")
    print("  Preserve the real FlyWire brain and four-command motor interface.")
    print()

    print("Disabled in Phase 2A:")
    print("  - reward")
    print("  - runtime plasticity")
    print("  - target-direction input")
    print("  - explicit navigation controller")
    print(f"  - steering gain = {STEERING_GAIN:.2f}")
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
        f"Real connections: {connectome.w.nnz:,}"
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
        f"  modelled DN types: {len(readout.commands)}"
    )

    if readout.missing:
        print(
            "  missing DN types: "
            + ", ".join(readout.missing)
        )

    print()

    # ========================================================================
    # BODY + WORLD + VIEWER
    # ========================================================================

    body = FlyBody()
    world = World()
    viewer = WorldViewer(world)

    print(
        f"Sandbox display: "
        f"{world.width:.0f} x {world.height:.0f} mm"
    )

    # Food odor source exists only in the environment.
    food_x_mm = world.width * FOOD_X_FRACTION
    food_y_mm = world.height * FOOD_Y_FRACTION

    print(
        f"Food-odor source (environment only): "
        f"({food_x_mm:.1f}, {food_y_mm:.1f}) mm"
    )
    print()

    # ========================================================================
    # REAL OLFACTORY POPULATION
    # ========================================================================

    odor_modality = BY_KEY[
        ODOR_MODALITY_KEY
    ]

    left_idx, right_idx = resolve_balanced_odor_indices(
        connectome,
        odor_modality,
    )

    all_odor_idx = np.concatenate(
        [
            left_idx,
            right_idx,
        ]
    )

    print("Real FlyWire odor neurons used:")
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
        f"  source cell types = "
        f"{'/'.join(odor_modality.cell_types)}"
    )

    if BALANCE_ODOR_POPULATION:
        print(
            "  bilateral population balancing = ENABLED"
        )
    else:
        print(
            "  bilateral population balancing = DISABLED"
        )

    print()

    # ========================================================================
    # INITIAL BODY STATE
    # ========================================================================

    reset_body(
        body,
        world,
    )

    print("Initial body state:")
    print(
        f"  position = "
        f"({body.state.x_mm:.1f}, {body.state.y_mm:.1f}) mm"
    )
    print(
        f"  heading  = {body.state.heading_deg:.1f} deg"
    )
    print()

    # ========================================================================
    # TELEMETRY
    # ========================================================================

    simulation_time_ms = 0.0

    max_speed = 0.0
    start_x = body.state.x_mm
    start_y = body.state.y_mm
    start_heading = body.state.heading_deg

    # ========================================================================
    # START
    # ========================================================================

    print("=" * 78)
    print("PHASE 3 CLOSED-LOOP ODOR NAVIGATION TEST RUNNING")
    print("=" * 78)
    print()

    print(
        "The environment knows where the food odor source is."
    )
    print(
        "The brain receives only left/right odor-driven ORN rates."
    )
    print(
        "The FlyBody receives only descending-neuron motor outputs."
    )
    print()

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    while (
        viewer.running
        and simulation_time_ms < PHASE2_DURATION_MS
    ):
        viewer.process_events()

        if not viewer.running:
            break

        simulation_time_ms += SIMULATION_CHUNK_MS

        # --------------------------------------------------------------------
        # 1. ENVIRONMENT -> TWO ANTENNA CONCENTRATIONS
        # --------------------------------------------------------------------

        (
            concentration_left,
            concentration_right,
        ) = sample_food_odor(
            body,
            food_x_mm,
            food_y_mm,
        )

        # --------------------------------------------------------------------
        # 2. ANTENNA CONCENTRATIONS -> REAL FLYWIRE ORN RATES
        # --------------------------------------------------------------------

        (
            rate_left,
            rate_right,
        ) = concentration_to_rates(
            concentration_left,
            concentration_right,
            odor_modality.max_rate_hz,
        )

        rates = np.concatenate(
            [
                np.full(
                    left_idx.size,
                    rate_left,
                    dtype=np.float64,
                ),
                np.full(
                    right_idx.size,
                    rate_right,
                    dtype=np.float64,
                ),
            ]
        )

        # Only real ORN indices + firing rates enter the brain.
        engine.set_poisson(
            all_odor_idx,
            rates,
        )

        # --------------------------------------------------------------------
        # 3. ADVANCE WHOLE FLYWIRE BRAIN
        # --------------------------------------------------------------------

        engine.run(
            SIMULATION_CHUNK_MS
        )

        # --------------------------------------------------------------------
        # 4. READ REAL DESCENDING NEURON MOTOR OUTPUT
        # --------------------------------------------------------------------

        spike_counts = engine.spike_counts

        channels = readout.channels(
            spike_counts,
            SIMULATION_CHUNK_MS,
        )

        escape_laterality = readout.escape_laterality(
            spike_counts
        )

        proboscis_drive = readout.proboscis_drive(
            spike_counts,
            SIMULATION_CHUNK_MS,
        )

        # --------------------------------------------------------------------
        # 5. FOUR MOTOR COMMANDS -> FLY BODY
        # --------------------------------------------------------------------

        # Apply the validated steering scale to the explicit four-channel
        # interface. Forward/backward are unchanged.
        body_channels = dict(channels)
        body_channels["forward"] = float(
            np.clip(
                body_channels.get("forward", 0.0),
                0.0,
                1.0,
            )
        )
        body_channels["backward"] = float(
            np.clip(
                body_channels.get("backward", 0.0),
                0.0,
                1.0,
            )
        )
        body_channels["left"] = float(
            np.clip(
                body_channels.get("left", 0.0) * STEERING_GAIN,
                0.0,
                1.0,
            )
        )
        body_channels["right"] = float(
            np.clip(
                body_channels.get("right", 0.0) * STEERING_GAIN,
                0.0,
                1.0,
            )
        )

        # Do not pass legacy turn/turn_bias into this closed-loop experiment.
        body_channels.pop("turn", None)
        body_channels.pop("turn_bias", None)

        body.update(
            dt_ms=SIMULATION_CHUNK_MS,
            channels=body_channels,
            t_ms=simulation_time_ms,
            escape_laterality=escape_laterality,
            proboscis_drive=proboscis_drive,
        )

        # --------------------------------------------------------------------
        # 6. KEEP BODY INSIDE SANDBOX
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

        max_speed = max(
            max_speed,
            body.state.speed_mm_s,
        )

        # --------------------------------------------------------------------
        # 7. FOUR-CHANNEL + ODOR TELEMETRY
        # --------------------------------------------------------------------

        forward = float(
            channels.get("forward", 0.0)
        )

        backward = float(
            channels.get("backward", 0.0)
        )

        left = float(
            channels.get("left", 0.0)
        )

        right = float(
            channels.get("right", 0.0)
        )

        applied_left = float(body_channels.get("left", 0.0))
        applied_right = float(body_channels.get("right", 0.0))

        print(
            f"t={simulation_time_ms:6.0f}ms "
            f"conc_L={concentration_left:.4f} "
            f"conc_R={concentration_right:.4f} "
            f"odor_diff="
            f"{concentration_left - concentration_right:+.4f} "
            f"forward={forward:.3f} "
            f"backward={backward:.3f} "
            f"raw_L={left:.3f} "
            f"raw_R={right:.3f} "
            f"applied_L={applied_left:.3f} "
            f"applied_R={applied_right:.3f} "
            f"speed={body.state.speed_mm_s:6.2f} "
            f"pos=({body.state.x_mm:7.1f},"
            f"{body.state.y_mm:7.1f}) "
            f"heading={body.state.heading_deg:7.2f} "
            f"behaviour={body.state.behaviour}"
        )

        # --------------------------------------------------------------------
        # 8. RENDER
        # --------------------------------------------------------------------

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

    displacement_x = (
        body.state.x_mm - start_x
    )

    displacement_y = (
        body.state.y_mm - start_y
    )

    displacement = float(
        np.hypot(
            displacement_x,
            displacement_y,
        )
    )

    heading_change = (
        body.state.heading_deg - start_heading
    )

    final_distance_to_food = math.hypot(
        body.state.x_mm - food_x_mm,
        body.state.y_mm - food_y_mm,
    )

    print()
    print("=" * 78)
    print("PHASE 3 CLOSED-LOOP ODOR NAVIGATION TEST COMPLETE")
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
        f"Final distance to food:"
        f" {final_distance_to_food:.2f} mm"
    )

    print(
        f"Heading change        : "
        f"{heading_change:+.2f} deg"
    )

    print()

    print(
        "NOTE: Final distance is an environment-side diagnostic only."
    )
    print(
        "It was NOT supplied to the neural simulation or FlyBody."
    )
    print()

    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
