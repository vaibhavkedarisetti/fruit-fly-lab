"""
Phase 3B — TRUE SEPARATED FORWARD / READOUT AUDIT

Scientific question
-------------------
In the shipped Phase 3 spatial-odor experiment, was reported forward
locomotion a genuine bilateral locomotion command, or was part of it an
engineering re-labelling of unilateral DNa01/DNa02 steering activity?

This diagnostic is deliberately non-invasive:
- real FlyWire FAFB v783 connectome is preserved;
- brain/motor/descending.py is NOT modified;
- simulation/engine/lif_engine.py is NOT modified;
- fly/body/fly_body.py is NOT modified;
- existing sandbox scripts are NOT modified.

The diagnostic fixes its own measurement window locally by snapshotting the
cumulative engine.spike_counts before and after each 100 ms chunk.

Five arms
---------
0. Open-loop symmetric sensory null, correctly windowed.
1. Shipped Phase 3 closed-loop geometry, correctly windowed, OLD max rule.
2. Same shipped geometry, correctly windowed, MEAN rule (descriptive only).
3. Shipped geometry, intentionally CUMULATIVE counts + OLD max rule
   (labelled READOUT_ARTIFACT; reproduces the old measurement defect).
4. Biological antenna separation (0.35 mm) with a fly-scale odor field
   (engineering environmental abstraction chosen to rise above the noise
   floor), correctly windowed, OLD max rule.

Important interpretation
------------------------
The max-vs-mean forward difference is an engineering/readout observable.
DNa01/DNa02 themselves are one reconstructed neuron per side in this dataset;
their side-to-side difference is therefore NOT called population lateralization.

The diagnostic does not claim that mean(L,R) is a biological forward command.
It only quantifies how much the shipped max rule can attribute to unilateral
activity, before clipping.

No navigation controller, target direction, food bearing, or food distance
is injected into the brain.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from brain.neurons.registry import load_connectome
from brain.motor.descending import (
    BACKWARD_GAIN,
    CHANNEL_HALF_MAX_HZ,
    FORWARD_GAIN,
    TURN_GAIN,
    DescendingReadout,
)
from brain.sensory.modalities import BY_KEY
from fly.body.fly_body import FlyBody
from simulation.engine.lif_engine import LIFEngine


# ============================================================================
# CONSTANTS
# ============================================================================

SEED = 3000
SIMULATION_CHUNK_MS = 100.0
DURATION_MS = 10_000.0

ODOR_MODALITY_KEY = "odor_vinegar"
PEAK_ODOR_RATE_HZ = 150.0

# Shipped Phase 3 configuration.
SHIPPED_ODOR_FALLOFF_MM = 250.0
SHIPPED_ANTENNA_HALF_SEP_MM = 60.0
SHIPPED_ANTENNA_FORWARD_MM = 0.0

# Biological-scale antenna separation and a fly-scale environmental field.
BIO_ANTENNA_HALF_SEP_MM = 0.35
BIO_ANTENNA_FORWARD_MM = 0.5
BIO_ODOR_FALLOFF_MM = 3.0

# Engineered steering authority from the earlier open-loop gain sweep.
STEERING_GAIN = 0.25

# Body/sandbox coordinates.
WORLD_WIDTH_MM = 1000.0
WORLD_HEIGHT_MM = 700.0
START_X_MM = 200.0
START_Y_MM = 350.0
START_HEADING_DEG = 0.0
FOOD_X_MM = 800.0
FOOD_Y_MM = 350.0

# Real ORNs, balanced deterministically.
BALANCED_SIDE_COUNT = 129

EPS = 1e-12

REAL_CONNECTOME = "REAL_CONNECTOME"
ENGINEERING_ABSTRACTION = "ENGINEERING_ABSTRACTION"
READOUT_ARTIFACT = "READOUT_ARTIFACT"
CATEGORIES = {
    REAL_CONNECTOME,
    ENGINEERING_ABSTRACTION,
    READOUT_ARTIFACT,
}


# ============================================================================
# PURE HELPERS
# ============================================================================

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def forward_max_preclip(left: float, right: float) -> float:
    return FORWARD_GAIN * max(left, right)


def forward_mean_preclip(left: float, right: float) -> float:
    return FORWARD_GAIN * 0.5 * (left + right)


def forward_max(left: float, right: float) -> float:
    return clip01(forward_max_preclip(left, right))


def forward_mean(left: float, right: float) -> float:
    return clip01(forward_mean_preclip(left, right))


def unilaterality(left: float, right: float) -> float | None:
    total = left + right
    if total <= EPS:
        return None
    return abs(left - right) / total


def contamination_fraction(u: float | None) -> float | None:
    if u is None:
        return None
    return u / (1.0 + u)


def odor_concentration(px: float, py: float, food_xy: tuple[float, float],
                       peak_rate_hz: float, falloff_mm: float) -> float:
    d = math.hypot(px - food_xy[0], py - food_xy[1])
    return peak_rate_hz * math.exp(-d / falloff_mm)


def antenna_positions(
    x_mm: float,
    y_mm: float,
    heading_deg: float,
    half_sep_mm: float,
    forward_mm: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    h = math.radians(heading_deg)
    fx, fy = math.cos(h), math.sin(h)
    lx, ly = -math.sin(h), math.cos(h)

    bx = x_mm + fx * forward_mm
    by = y_mm + fy * forward_mm

    left_xy = (bx + lx * half_sep_mm, by + ly * half_sep_mm)
    right_xy = (bx - lx * half_sep_mm, by - ly * half_sep_mm)
    return left_xy, right_xy


def poisson_relative_sd(n_neurons: int, rate_hz: float,
                        window_ms: float, between_sides: bool = False) -> float:
    expected = n_neurons * rate_hz * (window_ms * 1e-3)
    if expected <= 0.0:
        return float("inf")
    if between_sides:
        return math.sqrt(2.0) / math.sqrt(expected)
    return 1.0 / math.sqrt(expected)


def rate_quantum_hz(n_neurons: int, window_ms: float) -> float:
    if n_neurons <= 0 or window_ms <= 0:
        return float("inf")
    return 1000.0 / (window_ms * n_neurons)


def sensory_report(
    body_x: float,
    body_y: float,
    heading_deg: float,
    half_sep_mm: float,
    forward_mm: float,
    falloff_mm: float,
    peak_rate_hz: float,
    n_orns: int,
) -> dict:
    left_xy, right_xy = antenna_positions(
        body_x,
        body_y,
        heading_deg,
        half_sep_mm,
        forward_mm,
    )

    conc_left = odor_concentration(
        *left_xy, (FOOD_X_MM, FOOD_Y_MM), peak_rate_hz, falloff_mm
    )
    conc_right = odor_concentration(
        *right_xy, (FOOD_X_MM, FOOD_Y_MM), peak_rate_hz, falloff_mm
    )

    center_dist = math.hypot(
        body_x - FOOD_X_MM,
        body_y - FOOD_Y_MM,
    )
    local_rate = peak_rate_hz * math.exp(-center_dist / falloff_mm)
    gradient_hz_mm = local_rate / falloff_mm

    # Radial-gradient prediction for a small bilateral span.
    food_bearing = math.atan2(
        FOOD_Y_MM - body_y,
        FOOD_X_MM - body_x,
    )
    theta = food_bearing - math.radians(heading_deg)
    predicted_delta = abs(
        gradient_hz_mm
        * (2.0 * half_sep_mm)
        * math.sin(theta)
    )

    exact_delta = abs(conc_left - conc_right)
    contrast_frac = exact_delta / max(local_rate, EPS)
    noise_floor = poisson_relative_sd(
        n_orns,
        max(local_rate, EPS),
        SIMULATION_CHUNK_MS,
        between_sides=True,
    )
    contrast_over_noise = (
        contrast_frac / noise_floor if math.isfinite(noise_floor) else 0.0
    )

    return {
        "conc_left": conc_left,
        "conc_right": conc_right,
        "local_rate_hz": local_rate,
        "gradient_hz_mm": gradient_hz_mm,
        "predicted_bilateral_delta_hz": predicted_delta,
        "bilateral_delta_hz": exact_delta,
        "contrast_frac": contrast_frac,
        "noise_floor_frac": noise_floor,
        "contrast_over_noise": contrast_over_noise,
        "sensory_interpretable": contrast_over_noise >= 1.0,
    }


def distance_to_food(x_mm: float, y_mm: float) -> float:
    return math.hypot(x_mm - FOOD_X_MM, y_mm - FOOD_Y_MM)


def bearing_to_food_deg(x_mm: float, y_mm: float, heading_deg: float) -> float:
    bearing = math.degrees(math.atan2(
        FOOD_Y_MM - y_mm,
        FOOD_X_MM - x_mm,
    ))
    error = (bearing - heading_deg + 180.0) % 360.0 - 180.0
    return error


# ============================================================================
# WINDOWED SPIKE COUNTS
# ============================================================================

class SpikeWindow:
    """Local snapshot/difference adapter for cumulative LIF spike counts."""

    def __init__(self) -> None:
        self._snapshot: np.ndarray | None = None

    def begin(self, engine: LIFEngine) -> None:
        self._snapshot = engine.spike_counts.astype(np.int64).copy()

    def end(self, engine: LIFEngine) -> np.ndarray:
        if self._snapshot is None:
            raise RuntimeError("SpikeWindow.end() called before begin().")
        current = engine.spike_counts.astype(np.int64)
        delta = current - self._snapshot
        if np.any(delta < 0):
            raise RuntimeError("Cumulative spike counter decreased between snapshots.")
        self._snapshot = current.copy()
        return delta


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ArmConfig:
    arm_id: int
    name: str
    counting_mode: str
    driving_rule: str
    closed_loop: bool
    falloff_mm: float
    half_sep_mm: float
    forward_mm: float
    peak_rate_hz: float
    category: str
    description: str


ARMS = (
    ArmConfig(
        0,
        "OPEN_LOOP_NULL",
        "WIN",
        "MAX",
        False,
        SHIPPED_ODOR_FALLOFF_MM,
        SHIPPED_ANTENNA_HALF_SEP_MM,
        SHIPPED_ANTENNA_FORWARD_MM,
        PEAK_ODOR_RATE_HZ,
        REAL_CONNECTOME,
        "Fixed symmetric ORN drive; estimates U from network/Poisson activity alone.",
    ),
    ArmConfig(
        1,
        "SHIPPED_PHASE3_MAX",
        "WIN",
        "MAX",
        True,
        SHIPPED_ODOR_FALLOFF_MM,
        SHIPPED_ANTENNA_HALF_SEP_MM,
        SHIPPED_ANTENNA_FORWARD_MM,
        PEAK_ODOR_RATE_HZ,
        ENGINEERING_ABSTRACTION,
        "Primary controlled measurement of the shipped Phase 3 configuration.",
    ),
    ArmConfig(
        2,
        "SHIPPED_PHASE3_MEAN",
        "WIN",
        "MEAN",
        True,
        SHIPPED_ODOR_FALLOFF_MM,
        SHIPPED_ANTENNA_HALF_SEP_MM,
        SHIPPED_ANTENNA_FORWARD_MM,
        PEAK_ODOR_RATE_HZ,
        ENGINEERING_ABSTRACTION,
        "Descriptive cross-run comparison using the mean forward rule.",
    ),
    ArmConfig(
        3,
        "SHIPPED_PHASE3_CUM_ARTIFACT",
        "CUM",
        "MAX",
        True,
        SHIPPED_ODOR_FALLOFF_MM,
        SHIPPED_ANTENNA_HALF_SEP_MM,
        SHIPPED_ANTENNA_FORWARD_MM,
        PEAK_ODOR_RATE_HZ,
        READOUT_ARTIFACT,
        "Intentional reproduction of cumulative-count measurement defect.",
    ),
    ArmConfig(
        4,
        "BIO_ANTENNA_ENGINEERED_FIELD",
        "WIN",
        "MAX",
        True,
        BIO_ODOR_FALLOFF_MM,
        BIO_ANTENNA_HALF_SEP_MM,
        BIO_ANTENNA_FORWARD_MM,
        PEAK_ODOR_RATE_HZ,
        ENGINEERING_ABSTRACTION,
        "Biological antenna spacing with a fly-scale engineering odor field.",
    ),
)


# ============================================================================
# REPORTING
# ============================================================================

def report(name: str, value, category: str, note: str = "") -> None:
    if category not in CATEGORIES:
        raise ValueError(f"Invalid provenance category: {category}")
    suffix = f"  [{note}]" if note else ""
    if isinstance(value, float):
        print(f"{name:<42} {value:.6g}  [{category}]{suffix}")
    else:
        print(f"{name:<42} {value}  [{category}]{suffix}")


# ============================================================================
# CONNECTOME / POPULATION HELPERS
# ============================================================================

def resolve_odor_indices(connectome, modality) -> tuple[np.ndarray, np.ndarray]:
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


def dn_type_counts(readout: DescendingReadout) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ctype in ("DNa01", "DNa02", "MDN"):
        out[ctype] = {}
        for side in ("left", "right"):
            out[ctype][side] = int(
                readout.tracked.get(
                    (ctype, side),
                    np.empty(0, dtype=np.int64),
                ).size
            )
    return out


# ============================================================================
# ARM EXECUTION
# ============================================================================

def run_arm(
    connectome,
    modality,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    arm: ArmConfig,
) -> list[dict]:
    engine = LIFEngine(connectome, seed=SEED)
    readout = DescendingReadout(connectome)
    body = FlyBody()

    body.state.x_mm = START_X_MM
    body.state.y_mm = START_Y_MM
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

    all_odor_idx = np.concatenate([left_idx, right_idx])
    symmetric_rate_hz = modality.max_rate_hz * 0.5
    fixed_rates = np.full(
        all_odor_idx.size,
        symmetric_rate_hz,
        dtype=np.float64,
    )

    snapshot = SpikeWindow()
    rows: list[dict] = []
    simulation_time_ms = 0.0

    # For arm 0, fixed symmetric input; for all others, environment-driven input.
    while simulation_time_ms < DURATION_MS:
        simulation_time_ms += SIMULATION_CHUNK_MS

        if arm.closed_loop:
            odor = sensory_report(
                body.state.x_mm,
                body.state.y_mm,
                body.state.heading_deg,
                arm.half_sep_mm,
                arm.forward_mm,
                arm.falloff_mm,
                arm.peak_rate_hz,
                left_idx.size,
            )
            left_rate = float(odor["conc_left"])
            right_rate = float(odor["conc_right"])
        else:
            odor = sensory_report(
                body.state.x_mm,
                body.state.y_mm,
                body.state.heading_deg,
                arm.half_sep_mm,
                arm.forward_mm,
                arm.falloff_mm,
                arm.peak_rate_hz,
                left_idx.size,
            )
            left_rate = symmetric_rate_hz
            right_rate = symmetric_rate_hz

        rates = np.concatenate([
            np.full(left_idx.size, left_rate, dtype=np.float64),
            np.full(right_idx.size, right_rate, dtype=np.float64),
        ])

        engine.set_poisson(all_odor_idx, rates)

        # Correct measurement: snapshot exactly one 100 ms window.
        if arm.counting_mode == "WIN":
            snapshot.begin(engine)

        engine.run(SIMULATION_CHUNK_MS)

        if arm.counting_mode == "WIN":
            window_counts = snapshot.end(engine)
        elif arm.counting_mode == "CUM":
            window_counts = engine.spike_counts.copy()
        else:
            raise ValueError(arm.counting_mode)

        channels = readout.channels(
            window_counts,
            SIMULATION_CHUNK_MS,
        )

        # Recompute the shipped forward rule and proposed mean rule locally.
        left = float(channels.get("left", 0.0))
        right = float(channels.get("right", 0.0))
        fwd_max_pre = forward_max_preclip(left, right)
        fwd_mean_pre = forward_mean_preclip(left, right)
        fwd_max = clip01(fwd_max_pre)
        fwd_mean = clip01(fwd_mean_pre)

        # Audit the actual shared readout.
        shared_forward = float(
            channels.get("forward", channels.get("locomotion_drive", 0.0))
        )
        if abs(shared_forward - fwd_max) > 1e-12:
            raise AssertionError(
                f"Readout drift: shared forward={shared_forward} "
                f"but rederived max={fwd_max}"
            )

        if arm.driving_rule == "MAX":
            forward_used = fwd_max
        else:
            forward_used = fwd_mean

        backward = float(channels.get("backward", 0.0))
        body_channels = {
            "forward": forward_used,
            "backward": backward,
            "left": clip01(left * STEERING_GAIN),
            "right": clip01(right * STEERING_GAIN),
        }

        # Never pass legacy turn/turn_bias.
        escape_laterality = readout.escape_laterality(window_counts)
        proboscis_drive = readout.proboscis_drive(
            window_counts,
            SIMULATION_CHUNK_MS,
        )

        if arm.closed_loop:
            body.update(
                dt_ms=SIMULATION_CHUNK_MS,
                channels=body_channels,
                t_ms=simulation_time_ms,
                escape_laterality=escape_laterality,
                proboscis_drive=proboscis_drive,
            )
            body.state.x_mm = max(0.0, min(WORLD_WIDTH_MM, body.state.x_mm))
            body.state.y_mm = max(0.0, min(WORLD_HEIGHT_MM, body.state.y_mm))

        u = unilaterality(left, right)
        contamination = contamination_fraction(u)

        rows.append(
            {
                "arm": arm.arm_id,
                "arm_name": arm.name,
                "category": arm.category,
                "counting_mode": arm.counting_mode,
                "driving_rule": arm.driving_rule,
                "seed": SEED,
                "t_ms": simulation_time_ms,
                "window_ms": SIMULATION_CHUNK_MS,
                "n_orns_per_side": left_idx.size,

                "DNa01_left_n": dn_type_counts(readout)["DNa01"]["left"],
                "DNa01_right_n": dn_type_counts(readout)["DNa01"]["right"],
                "DNa02_left_n": dn_type_counts(readout)["DNa02"]["left"],
                "DNa02_right_n": dn_type_counts(readout)["DNa02"]["right"],
                "MDN_left_n": dn_type_counts(readout)["MDN"]["left"],
                "MDN_right_n": dn_type_counts(readout)["MDN"]["right"],

                "DNa01_left_spikes": int(window_counts[readout.tracked[("DNa01", "left")]].sum()),
                "DNa01_right_spikes": int(window_counts[readout.tracked[("DNa01", "right")]].sum()),
                "DNa02_left_spikes": int(window_counts[readout.tracked[("DNa02", "left")]].sum()),
                "DNa02_right_spikes": int(window_counts[readout.tracked[("DNa02", "right")]].sum()),
                "MDN_left_spikes": int(window_counts[readout.tracked[("MDN", "left")]].sum()),
                "MDN_right_spikes": int(window_counts[readout.tracked[("MDN", "right")]].sum()),

                "L": left,
                "R": right,
                "u_preclip": u,
                "u_defined": u is not None,
                "contamination_preclip": contamination,

                "fwd_max_preclip": fwd_max_pre,
                "fwd_mean_preclip": fwd_mean_pre,
                "fwd_max": fwd_max,
                "fwd_mean": fwd_mean,
                "fwd_max_clipped": fwd_max_pre >= 1.0,
                "fwd_mean_clipped": fwd_mean_pre >= 1.0,
                "both_clipped": fwd_max_pre >= 1.0 and fwd_mean_pre >= 1.0,
                "comparison_uninformative": fwd_max_pre >= 1.0 and fwd_mean_pre >= 1.0,
                "delta_preclip": fwd_max_pre - fwd_mean_pre,
                "delta_postclip": fwd_max - fwd_mean,

                "conc_left": odor["conc_left"],
                "conc_right": odor["conc_right"],
                "local_rate_hz": odor["local_rate_hz"],
                "gradient_hz_mm": odor["gradient_hz_mm"],
                "predicted_bilateral_delta_hz": odor["predicted_bilateral_delta_hz"],
                "bilateral_delta_hz": odor["bilateral_delta_hz"],
                "contrast_frac": odor["contrast_frac"],
                "noise_floor_frac": odor["noise_floor_frac"],
                "contrast_over_noise": odor["contrast_over_noise"],
                "sensory_interpretable": odor["sensory_interpretable"],

                "speed_mm_s": float(body.state.speed_mm_s),
                "x_mm": float(body.state.x_mm),
                "y_mm": float(body.state.y_mm),
                "heading_deg": float(body.state.heading_deg),
                "behaviour": body.state.behaviour,
                "distance_to_food_mm": distance_to_food(body.state.x_mm, body.state.y_mm),
                "bearing_to_food_deg": bearing_to_food_deg(
                    body.state.x_mm,
                    body.state.y_mm,
                    body.state.heading_deg,
                ),
            }
        )

    engine.clear_poisson()
    return rows


# ============================================================================
# SUMMARY / VALIDATION
# ============================================================================

def as_array(rows: list[dict], key: str) -> np.ndarray:
    vals = [r[key] for r in rows if r[key] is not None]
    return np.asarray(vals, dtype=float)


def summarize_arm(rows: list[dict], arm: ArmConfig) -> dict:
    u = as_array(rows, "u_preclip")
    contamination = as_array(rows, "contamination_preclip")
    valid = np.asarray(
        [r["u_defined"] for r in rows],
        dtype=bool,
    )
    informative = np.asarray(
        [not r["comparison_uninformative"] for r in rows],
        dtype=bool,
    )
    sensory_ok = np.asarray(
        [r["sensory_interpretable"] for r in rows],
        dtype=bool,
    )

    informative_fraction = float(np.mean(informative))
    sensory_fraction = float(np.mean(sensory_ok))

    if u.size:
        u_stats = {
            "mean": float(np.mean(u)),
            "median": float(np.median(u)),
            "p10": float(np.percentile(u, 10)),
            "p90": float(np.percentile(u, 90)),
            "min": float(np.min(u)),
            "max": float(np.max(u)),
        }
    else:
        u_stats = {
            k: float("nan")
            for k in ("mean", "median", "p10", "p90", "min", "max")
        }

    informative_contamination = np.asarray(
        [
            r["contamination_preclip"]
            for r in rows
            if r["contamination_preclip"] is not None
            and not r["comparison_uninformative"]
        ],
        dtype=float,
    )

    median_contamination = (
        float(np.median(informative_contamination))
        if informative_contamination.size
        else float("nan")
    )

    if informative_fraction < 0.5:
        verdict = "INDETERMINATE"
    elif arm.closed_loop and sensory_fraction < 0.5:
        verdict = "INDETERMINATE"
    elif not informative_contamination.size:
        verdict = "INDETERMINATE"
    elif median_contamination < 0.05:
        verdict = "BILATERAL-DOMINANT"
    elif median_contamination <= 0.25:
        verdict = "MIXED"
    else:
        verdict = "UNILATERAL-DOMINANT"

    return {
        "arm": arm.arm_id,
        "arm_name": arm.name,
        "category": arm.category,
        "u_stats": u_stats,
        "median_contamination": median_contamination,
        "clip_fraction_max": float(np.mean([r["fwd_max_clipped"] for r in rows])),
        "clip_fraction_mean": float(np.mean([r["fwd_mean_clipped"] for r in rows])),
        "both_clip_fraction": float(np.mean([r["both_clipped"] for r in rows])),
        "informative_fraction": informative_fraction,
        "sensory_fraction": sensory_fraction,
        "mean_forward_max": float(np.mean(as_array(rows, "fwd_max"))),
        "mean_forward_mean": float(np.mean(as_array(rows, "fwd_mean"))),
        "mean_speed": float(np.mean(as_array(rows, "speed_mm_s"))),
        "final_x": float(rows[-1]["x_mm"]),
        "final_y": float(rows[-1]["y_mm"]),
        "final_heading": float(rows[-1]["heading_deg"]),
        "final_distance": float(rows[-1]["distance_to_food_mm"]),
        "min_distance": float(np.min(as_array(rows, "distance_to_food_mm"))),
        "verdict": verdict,
        "valid_u_windows": int(np.sum(valid)),
        "n_windows": len(rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_drift(rows: list[dict], arm: ArmConfig) -> None:
    sums = as_array(rows, "L") + as_array(rows, "R")
    t = as_array(rows, "t_ms")
    if len(rows) < 2:
        slope = float("nan")
    else:
        slope = float(np.polyfit(t, sums, 1)[0])

    n = max(1, len(sums) // 10)
    first_clip = float(np.mean([r["both_clipped"] for r in rows[:n]]))
    last_clip = float(np.mean([r["both_clipped"] for r in rows[-n:]]))

    report(
        f"{arm.name} L+R drift slope",
        slope,
        arm.category,
        "readout-window diagnostic",
    )
    report(
        f"{arm.name} both-clipped first decile",
        first_clip,
        arm.category,
    )
    report(
        f"{arm.name} both-clipped last decile",
        last_clip,
        arm.category,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    print()
    print("=" * 88)
    print("DIGITAL DROSOPHILA — PHASE 3B FORWARD READOUT AUDIT")
    print("=" * 88)
    print()
    print("Scientific question:")
    print("  Was Phase 3 forward locomotion supported by bilateral DNa activity,")
    print("  or did the max(L,R) engineering readout re-label unilateral steering?")
    print()

    print("Loading FlyWire v783 connectome...")
    connectome = load_connectome()
    report(
        "connectome neurons",
        connectome.n,
        REAL_CONNECTOME,
    )
    report(
        "connectome connections",
        connectome.w.nnz,
        REAL_CONNECTOME,
    )
    print()

    modality = BY_KEY[ODOR_MODALITY_KEY]
    left_idx, right_idx = resolve_odor_indices(connectome, modality)

    report(
        "real ORNs per side",
        left_idx.size,
        REAL_CONNECTOME,
        "deterministically balanced",
    )
    report(
        "odor rate at source",
        PEAK_ODOR_RATE_HZ,
        ENGINEERING_ABSTRACTION,
    )

    readout_probe_engine = LIFEngine(connectome, seed=SEED)
    readout_probe = DescendingReadout(connectome)
    dn_counts = dn_type_counts(readout_probe)
    report("DNa01 left neuron count", dn_counts["DNa01"]["left"], REAL_CONNECTOME)
    report("DNa01 right neuron count", dn_counts["DNa01"]["right"], REAL_CONNECTOME)
    report("DNa02 left neuron count", dn_counts["DNa02"]["left"], REAL_CONNECTOME)
    report("DNa02 right neuron count", dn_counts["DNa02"]["right"], REAL_CONNECTOME)
    report("MDN left neuron count", dn_counts["MDN"]["left"], REAL_CONNECTOME)
    report("MDN right neuron count", dn_counts["MDN"]["right"], REAL_CONNECTOME)
    del readout_probe_engine

    report(
        "measurement window ms",
        SIMULATION_CHUNK_MS,
        ENGINEERING_ABSTRACTION,
    )
    report(
        "DNa01/DNa02 rate quantum Hz",
        rate_quantum_hz(1, SIMULATION_CHUNK_MS),
        ENGINEERING_ABSTRACTION,
        "one reconstructed neuron per side",
    )
    report(
        "MDN rate quantum Hz",
        rate_quantum_hz(2, SIMULATION_CHUNK_MS),
        ENGINEERING_ABSTRACTION,
        "two reconstructed neurons per side",
    )
    print()

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for arm in ARMS:
        print()
        print("-" * 88)
        print(f"ARM {arm.arm_id}: {arm.name}")
        print("-" * 88)
        print(arm.description)

        if arm.closed_loop:
            start_sens = sensory_report(
                START_X_MM,
                START_Y_MM,
                START_HEADING_DEG,
                arm.half_sep_mm,
                arm.forward_mm,
                arm.falloff_mm,
                arm.peak_rate_hz,
                left_idx.size,
            )
            report(
                "start sensory contrast fraction",
                start_sens["contrast_frac"],
                ENGINEERING_ABSTRACTION,
            )
            report(
                "start Poisson difference noise floor",
                start_sens["noise_floor_frac"],
                ENGINEERING_ABSTRACTION,
            )
            report(
                "start contrast / noise ratio",
                start_sens["contrast_over_noise"],
                ENGINEERING_ABSTRACTION,
            )

            if arm.half_sep_mm == BIO_ANTENNA_HALF_SEP_MM:
                report(
                    "antenna bilateral span mm",
                    2.0 * arm.half_sep_mm,
                    ENGINEERING_ABSTRACTION,
                    "biological-scale geometry",
                )
            else:
                report(
                    "antenna bilateral span mm",
                    2.0 * arm.half_sep_mm,
                    ENGINEERING_ABSTRACTION,
                    "SHIPPED Phase 3 geometry",
                )

        rows = run_arm(
            connectome,
            modality,
            left_idx,
            right_idx,
            arm,
        )
        all_rows.extend(rows)

        summary = summarize_arm(rows, arm)
        summaries.append(summary)

        report(
            "median pre-clip unilaterality U",
            summary["u_stats"]["median"],
            arm.category,
        )
        report(
            "median contamination U/(1+U)",
            summary["median_contamination"],
            arm.category,
        )
        report(
            "max-rule clip fraction",
            summary["clip_fraction_max"],
            arm.category,
        )
        report(
            "mean-rule clip fraction",
            summary["clip_fraction_mean"],
            arm.category,
        )
        report(
            "both-rule clip fraction",
            summary["both_clip_fraction"],
            arm.category,
        )
        report(
            "informative-window fraction",
            summary["informative_fraction"],
            arm.category,
        )
        report(
            "sensory-interpretable window fraction",
            summary["sensory_fraction"],
            ENGINEERING_ABSTRACTION,
        )
        report(
            "mean forward (max rule)",
            summary["mean_forward_max"],
            arm.category,
        )
        report(
            "mean forward (mean rule)",
            summary["mean_forward_mean"],
            arm.category,
        )
        report(
            "final distance to food mm",
            summary["final_distance"],
            ENGINEERING_ABSTRACTION if arm.closed_loop else arm.category,
        )
        report(
            "minimum distance to food mm",
            summary["min_distance"],
            ENGINEERING_ABSTRACTION if arm.closed_loop else arm.category,
        )
        report(
            "final heading deg",
            summary["final_heading"],
            ENGINEERING_ABSTRACTION,
        )
        report(
            "verdict",
            summary["verdict"],
            arm.category,
        )

        if arm.counting_mode in ("WIN", "CUM"):
            print_drift(rows, arm)

        # CSV for each arm as well as combined output.
        arm_path = Path.cwd() / f"phase3b_{arm.name.lower()}.csv"
        write_csv(arm_path, rows)
        print(f"wrote {arm_path} ({len(rows)} rows)")

    # Combined CSV.
    combined_path = Path.cwd() / "phase3b_forward_readout_diagnostic.csv"
    write_csv(combined_path, all_rows)

    print()
    print("=" * 88)
    print("PHASE 3B SUMMARY")
    print("=" * 88)
    print()

    for s in summaries:
        print(
            f"{s['arm_name']:<32} "
            f"median_U={s['u_stats']['median']:.4f}  "
            f"contam={s['median_contamination']:.4f}  "
            f"maxclip={s['clip_fraction_max']:.3f}  "
            f"meanclip={s['clip_fraction_mean']:.3f}  "
            f"informative={s['informative_fraction']:.3f}  "
            f"verdict={s['verdict']}"
        )

    print()
    print("ALGEBRAIC IDENTITY — NOT EVIDENCE")
    print(
        "Pre-clip: max(L,R)*G - mean(L,R)*G/2 = G*|L-R|/2 = 2*|L-R|."
    )
    print(
        "A correlation between this delta and U would therefore be algebraically"
    )
    print("determined and is NOT used as evidence.")

    print()
    print("IMPORTANT FINDINGS")
    print(
        "1. DNa01/DNa02 side differences are REAL_CONNECTOME single-cell"
    )
    print("   observations, not population-level lateralization.")
    print(
        "2. The cumulative-count arm is READOUT_ARTIFACT and is never used"
    )
    print("   as evidence of neural firing.")
    print(
        "3. Closed-loop behavioral comparisons between max and mean runs are"
    )
    print("   DESCRIPTIVE ONLY because changing the drive changes the trajectory")
    print("   and therefore the future sensory input.")
    print(
        "4. Biological-scale odor geometry must beat the computed Poisson noise"
    )
    print("   floor before a chemotaxis result is interpreted.")
    print()
    print(f"Combined CSV: {combined_path}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    sys.exit(main())
