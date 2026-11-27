"""
Phase 3C — BILATERAL DNa + BIO-SCALE ODOR DIAGNOSTIC

Purpose
-------
Resolve the two remaining problems from Phase 3B without changing the shared
connectome, LIF engine, DescendingReadout, or FlyBody modules:

1. A 100 ms window gives a one-neuron DNa side a 10 Hz rate quantum.  Here we
   retain 100 ms control windows, but aggregate ten of them into 1 s diagnostic
   windows for bilateral DNa statistics.

2. The previous biological-scale odor arm started ~600 mm from a 3 mm decay
   length source, making odor effectively zero.  Here the odor source is placed
   6 mm away and laterally offset from the fly, while keeping the antenna span
   biologically small (0.70 mm).

Design principles
-----------------
- Real FlyWire FAFB v783 connectome is unchanged.
- 100 ms engine/body control windows are unchanged.
- DNa bilateral diagnostic uses 1 s aggregated spike counts.
- A matched-rate symmetric null is run at the same initial local odor rate
  as the bio-scale arm, rather than at an arbitrary 75 Hz drive.
- The bio arm uses real odor-vinegar ORNs with spatially asymmetric rates.
- No target bearing, target direction, reward, memory, or hand-coded
  navigation controller is injected into the brain.
- The body is an engineering downstream model; behavioral conclusions are
  labelled ENGINEERING_ABSTRACTION.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain.neurons.registry import load_connectome
from brain.motor.descending import DescendingReadout, FORWARD_GAIN
from brain.sensory.modalities import BY_KEY
from fly.body.fly_body import FlyBody
from simulation.engine.lif_engine import LIFEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTROL_WINDOW_MS = 100.0
DIAGNOSTIC_WINDOW_MS = 1000.0
DURATION_MS = 10_000.0
CONTROLS_PER_DIAGNOSTIC = int(DIAGNOSTIC_WINDOW_MS / CONTROL_WINDOW_MS)

BALANCED_SIDE_COUNT = 129
PEAK_ODOR_RATE_HZ = 150.0

# Bio-scale odor geometry. Start = 6 mm from source, source is lateral to
# heading so the two antennas initially receive different odor rates.
BIO_START_X_MM = 200.0
BIO_START_Y_MM = 350.0
BIO_START_HEADING_DEG = 0.0
BIO_SOURCE_X_MM = 200.0
BIO_SOURCE_Y_MM = 356.0
BIO_ODOR_FALLOFF_MM = 3.0
BIO_ANTENNA_HALF_SEP_MM = 0.35
BIO_ANTENNA_FORWARD_MM = 0.5

# Keep the engineering steering authority from Phase 3.
STEERING_GAIN = 0.25

# Use multiple seeds so the null is an empirical distribution rather than one
# stochastic realization.
NULL_SEEDS = tuple(range(3000, 3010))
BIO_SEEDS = tuple(range(3000, 3006))

EPS = 1e-12

REAL_CONNECTOME = "REAL_CONNECTOME"
ENGINEERING_ABSTRACTION = "ENGINEERING_ABSTRACTION"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def odor_concentration(
    px: float,
    py: float,
    source_xy: tuple[float, float],
    peak_rate_hz: float,
    falloff_mm: float,
) -> float:
    d = math.hypot(px - source_xy[0], py - source_xy[1])
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


def sensory_report(
    x_mm: float,
    y_mm: float,
    heading_deg: float,
) -> dict[str, float]:
    left_xy, right_xy = antenna_positions(
        x_mm,
        y_mm,
        heading_deg,
        BIO_ANTENNA_HALF_SEP_MM,
        BIO_ANTENNA_FORWARD_MM,
    )

    left = odor_concentration(
        *left_xy,
        (BIO_SOURCE_X_MM, BIO_SOURCE_Y_MM),
        PEAK_ODOR_RATE_HZ,
        BIO_ODOR_FALLOFF_MM,
    )
    right = odor_concentration(
        *right_xy,
        (BIO_SOURCE_X_MM, BIO_SOURCE_Y_MM),
        PEAK_ODOR_RATE_HZ,
        BIO_ODOR_FALLOFF_MM,
    )

    cx = x_mm + math.cos(math.radians(heading_deg)) * BIO_ANTENNA_FORWARD_MM
    cy = y_mm + math.sin(math.radians(heading_deg)) * BIO_ANTENNA_FORWARD_MM
    center_distance = math.hypot(cx - BIO_SOURCE_X_MM, cy - BIO_SOURCE_Y_MM)
    local = PEAK_ODOR_RATE_HZ * math.exp(-center_distance / BIO_ODOR_FALLOFF_MM)

    contrast = abs(left - right) / max(local, EPS)
    expected_events_per_side = 129 * max(local, 0.0) * (CONTROL_WINDOW_MS / 1000.0)
    if expected_events_per_side <= 0.0:
        noise = float("inf")
    else:
        # Relative SD of a difference of two independent Poisson counts.
        noise = math.sqrt(2.0 / expected_events_per_side)

    return {
        "left_rate_hz": left,
        "right_rate_hz": right,
        "mean_rate_hz": 0.5 * (left + right),
        "local_rate_hz": local,
        "contrast_fraction": contrast,
        "noise_floor_fraction": noise,
        "contrast_noise_ratio": (
            contrast / noise if math.isfinite(noise) and noise > 0 else 0.0
        ),
    }


def signed_bias(left_hz: float, right_hz: float) -> float | None:
    total = left_hz + right_hz
    if total <= EPS:
        return None
    return (right_hz - left_hz) / total


def unilaterality(left_hz: float, right_hz: float) -> float | None:
    total = left_hz + right_hz
    if total <= EPS:
        return None
    return abs(left_hz - right_hz) / total


def contamination(u: float | None) -> float | None:
    if u is None:
        return None
    return u / (1.0 + u)


class SpikeWindow:
    """Snapshot/difference adapter for cumulative LIF spike counters."""

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
            raise RuntimeError("Cumulative spike count decreased unexpectedly.")
        self._snapshot = current.copy()
        return delta


@dataclass
class DNWindow:
    left: float
    right: float
    u: float | None
    bias: float | None


def resolve_or_path(connectome):
    modality = BY_KEY["odor_vinegar"]
    cells = connectome.by_cell_types(modality.cell_types)

    left_idx = cells[cells["side"].astype(str) == "left"]["idx"].to_numpy(dtype=np.int64)
    right_idx = cells[cells["side"].astype(str) == "right"]["idx"].to_numpy(dtype=np.int64)

    n = min(BALANCED_SIDE_COUNT, left_idx.size, right_idx.size)
    left_idx = np.sort(left_idx)[:n]
    right_idx = np.sort(right_idx)[:n]
    return modality, left_idx, right_idx


def tracked_count(readout: DescendingReadout, ctype: str, side: str) -> int:
    return int(readout.tracked.get((ctype, side), np.empty(0, dtype=np.int64)).size)


def readout_lr(readout: DescendingReadout, counts: np.ndarray, window_ms: float) -> tuple[float, float]:
    ch = readout.channels(counts, window_ms)
    return float(ch.get("left", 0.0)), float(ch.get("right", 0.0))


def save_csv(path: Path, rows: list[dict]) -> None:
    """Write heterogeneous diagnostic rows without dropping columns.

    Different experiment arms intentionally record different metadata.  The
    original implementation used the first row's keys as the CSV schema,
    which caused DictWriter to fail as soon as a later arm added fields such
    as x_mm, odor_left_hz, or heading_deg.
    """
    if not rows:
        return

    # Build one stable schema containing every column present in every row.
    # Sorting makes the output deterministic across runs.
    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Part A: matched-rate null
# ---------------------------------------------------------------------------

def run_matched_rate_null(connectome, modality, left_idx, right_idx):
    """Empirical null at the bio arm's initial mean odor rate."""
    start_report = sensory_report(
        BIO_START_X_MM,
        BIO_START_Y_MM,
        BIO_START_HEADING_DEG,
    )
    matched_rate = start_report["mean_rate_hz"]

    all_idx = np.concatenate([left_idx, right_idx])
    rows: list[dict] = []

    for seed in NULL_SEEDS:
        engine = LIFEngine(connectome, seed=seed)
        readout = DescendingReadout(connectome)
        engine.set_poisson(
            all_idx,
            np.full(all_idx.size, matched_rate, dtype=np.float64),
        )

        short_window = SpikeWindow()
        one_second: deque[np.ndarray] = deque(maxlen=CONTROLS_PER_DIAGNOSTIC)
        elapsed = 0.0

        while elapsed < DURATION_MS:
            elapsed += CONTROL_WINDOW_MS
            short_window.begin(engine)
            engine.run(CONTROL_WINDOW_MS)
            delta = short_window.end(engine)
            one_second.append(delta)

            if len(one_second) == CONTROLS_PER_DIAGNOSTIC:
                agg = np.sum(np.stack(one_second, axis=0), axis=0)
                left, right = readout_lr(readout, agg, DIAGNOSTIC_WINDOW_MS)
                u = unilaterality(left, right)
                bias = signed_bias(left, right)

                rows.append(
                    {
                        "kind": "NULL",
                        "seed": seed,
                        "t_ms": elapsed,
                        "window_ms": DIAGNOSTIC_WINDOW_MS,
                        "matched_rate_hz": matched_rate,
                        "DNa_left_hz": left,
                        "DNa_right_hz": right,
                        "U": u,
                        "signed_bias": bias,
                    }
                )

    return rows, start_report


# ---------------------------------------------------------------------------
# Part B: bio-scale closed-loop arm
# ---------------------------------------------------------------------------

def run_bio_arm(connectome, modality, left_idx, right_idx):
    all_idx = np.concatenate([left_idx, right_idx])
    rows: list[dict] = []

    for seed in BIO_SEEDS:
        engine = LIFEngine(connectome, seed=seed)
        readout = DescendingReadout(connectome)
        body = FlyBody()
        body.state.x_mm = BIO_START_X_MM
        body.state.y_mm = BIO_START_Y_MM
        body.state.z_mm = 0.0
        body.state.heading_deg = BIO_START_HEADING_DEG
        body.state.speed_mm_s = 0.0
        body.state.turn_rate_deg_s = 0.0
        body.state.airborne = False
        body.state.vz_mm_s = 0.0
        body.state.behaviour = "resting"
        body.state.escape_mode = ""
        body.state.wing_angle_deg = 0.0
        body.state.leg_extension = 0.0
        body.state.proboscis_extension = 0.0

        short_window = SpikeWindow()
        one_second: deque[np.ndarray] = deque(maxlen=CONTROLS_PER_DIAGNOSTIC)
        elapsed = 0.0

        while elapsed < DURATION_MS:
            elapsed += CONTROL_WINDOW_MS
            odor = sensory_report(
                body.state.x_mm,
                body.state.y_mm,
                body.state.heading_deg,
            )

            rates = np.concatenate([
                np.full(left_idx.size, odor["left_rate_hz"], dtype=np.float64),
                np.full(right_idx.size, odor["right_rate_hz"], dtype=np.float64),
            ])
            engine.set_poisson(all_idx, rates)

            short_window.begin(engine)
            engine.run(CONTROL_WINDOW_MS)
            delta = short_window.end(engine)
            one_second.append(delta)

            # Real 100 ms body control. Mean forward is used here specifically
            # to remove max(L,R) unilateral-steering contamination from this
            # sensory diagnostic. This is still an engineering abstraction.
            left100, right100 = readout_lr(readout, delta, CONTROL_WINDOW_MS)
            forward = clip01(0.5 * (left100 + right100) * FORWARD_GAIN)
            backward = float(readout.channels(delta, CONTROL_WINDOW_MS).get("backward", 0.0))
            channels = {
                "forward": forward,
                "backward": backward,
                "left": clip01(left100 * STEERING_GAIN),
                "right": clip01(right100 * STEERING_GAIN),
            }
            body.update(
                dt_ms=CONTROL_WINDOW_MS,
                channels=channels,
                t_ms=elapsed,
                escape_laterality=readout.escape_laterality(delta),
                proboscis_drive=readout.proboscis_drive(delta, CONTROL_WINDOW_MS),
            )

            if len(one_second) == CONTROLS_PER_DIAGNOSTIC:
                agg = np.sum(np.stack(one_second, axis=0), axis=0)
                left1s, right1s = readout_lr(readout, agg, DIAGNOSTIC_WINDOW_MS)
                u1s = unilaterality(left1s, right1s)
                b1s = signed_bias(left1s, right1s)

                odor_now = sensory_report(
                    body.state.x_mm,
                    body.state.y_mm,
                    body.state.heading_deg,
                )

                rows.append(
                    {
                        "kind": "BIO",
                        "seed": seed,
                        "t_ms": elapsed,
                        "window_ms": DIAGNOSTIC_WINDOW_MS,
                        "local_rate_hz": odor_now["local_rate_hz"],
                        "odor_left_hz": odor_now["left_rate_hz"],
                        "odor_right_hz": odor_now["right_rate_hz"],
                        "odor_contrast_fraction": odor_now["contrast_fraction"],
                        "odor_noise_floor_fraction": odor_now["noise_floor_fraction"],
                        "odor_contrast_noise_ratio": odor_now["contrast_noise_ratio"],
                        "DNa_left_hz": left1s,
                        "DNa_right_hz": right1s,
                        "U": u1s,
                        "signed_bias": b1s,
                        "x_mm": body.state.x_mm,
                        "y_mm": body.state.y_mm,
                        "heading_deg": body.state.heading_deg,
                        "speed_mm_s": body.state.speed_mm_s,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 96)
    print("DIGITAL DROSOPHILA — PHASE 3C BILATERAL + BIO-GRADIENT DIAGNOSTIC")
    print("=" * 96)
    print()
    print("Loading FlyWire v783 connectome...")
    connectome = load_connectome()
    print(f"connectome neurons                         {connectome.n}  [{REAL_CONNECTOME}]")
    print(f"connectome connections                     {connectome.w.nnz}  [{REAL_CONNECTOME}]")

    modality, left_idx, right_idx = resolve_or_path(connectome)
    print(f"real ORNs per side                         {left_idx.size}  [{REAL_CONNECTOME}]")
    print(f"control window ms                          {CONTROL_WINDOW_MS:g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"diagnostic window ms                       {DIAGNOSTIC_WINDOW_MS:g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"DNa one-neuron rate quantum Hz            {1000.0 / DIAGNOSTIC_WINDOW_MS:g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"bio antenna bilateral span mm              {2 * BIO_ANTENNA_HALF_SEP_MM:g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"bio odor source distance mm                {math.hypot(BIO_SOURCE_X_MM-BIO_START_X_MM, BIO_SOURCE_Y_MM-BIO_START_Y_MM):g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"bio odor falloff mm                        {BIO_ODOR_FALLOFF_MM:g}  [{ENGINEERING_ABSTRACTION}]")
    print(f"null seeds                                 {len(NULL_SEEDS)}  [{ENGINEERING_ABSTRACTION}]")
    print(f"bio seeds                                   {len(BIO_SEEDS)}  [{ENGINEERING_ABSTRACTION}]")

    initial = sensory_report(BIO_START_X_MM, BIO_START_Y_MM, BIO_START_HEADING_DEG)
    print()
    print("Initial bio-scale sensory field")
    print(f"  left rate Hz                            {initial['left_rate_hz']:.6g}")
    print(f"  right rate Hz                           {initial['right_rate_hz']:.6g}")
    print(f"  mean rate Hz                            {initial['mean_rate_hz']:.6g}")
    print(f"  contrast fraction                       {initial['contrast_fraction']:.6g}")
    print(f"  Poisson difference noise floor          {initial['noise_floor_fraction']:.6g}")
    print(f"  contrast/noise ratio                    {initial['contrast_noise_ratio']:.6g}")

    print("\n" + "-" * 96)
    print("PART A: MATCHED-RATE EMPIRICAL NULL")
    print("-" * 96)
    null_rows, null_initial = run_matched_rate_null(connectome, modality, left_idx, right_idx)
    null_abs_bias = np.array([abs(r["signed_bias"]) for r in null_rows if r["signed_bias"] is not None], dtype=float)
    null_u = np.array([r["U"] for r in null_rows if r["U"] is not None], dtype=float)

    print(f"null rows                                  {len(null_rows)}")
    print(f"null median |signed bias|                 {np.median(null_abs_bias):.6g}")
    print(f"null 95th pct |signed bias|               {np.quantile(null_abs_bias, 0.95):.6g}")
    print(f"null median U                             {np.median(null_u):.6g}")
    print(f"null 95th pct U                           {np.quantile(null_u, 0.95):.6g}")

    print("\n" + "-" * 96)
    print("PART B: BIO-SCALE CLOSED LOOP")
    print("-" * 96)
    bio_rows = run_bio_arm(connectome, modality, left_idx, right_idx)
    bio_abs_bias = np.array([abs(r["signed_bias"]) for r in bio_rows if r["signed_bias"] is not None], dtype=float)
    bio_u = np.array([r["U"] for r in bio_rows if r["U"] is not None], dtype=float)
    bio_contrast_ratio = np.array([r["odor_contrast_noise_ratio"] for r in bio_rows], dtype=float)

    exceed_bias = float(np.mean(bio_abs_bias > np.quantile(null_abs_bias, 0.95))) if bio_abs_bias.size else float("nan")
    exceed_u = float(np.mean(bio_u > np.quantile(null_u, 0.95))) if bio_u.size else float("nan")
    interpretable = float(np.mean(bio_contrast_ratio >= 1.0)) if bio_contrast_ratio.size else 0.0

    print(f"bio rows                                   {len(bio_rows)}")
    print(f"bio median |signed bias|                  {np.median(bio_abs_bias) if bio_abs_bias.size else float('nan'):.6g}")
    print(f"bio median U                               {np.median(bio_u) if bio_u.size else float('nan'):.6g}")
    print(f"fraction above null 95% |bias|            {exceed_bias:.6g}")
    print(f"fraction above null 95% U                  {exceed_u:.6g}")
    print(f"sensory-interpretable fraction             {interpretable:.6g}")

    # Conservative diagnostic verdict: this is not a claim of biological
    # chemotaxis; it only states whether the neural side imbalance escapes the
    # matched-rate null while sensory contrast is measurable.
    if interpretable < 0.5:
        verdict = "INDETERMINATE — sensory contrast below noise too often"
    elif exceed_bias >= 0.5 or exceed_u >= 0.5:
        verdict = "DNa ASYMMETRY ABOVE MATCHED NULL — investigate causality next"
    else:
        verdict = "NO CLEAR DNa ASYMMETRY ABOVE MATCHED NULL"
    print(f"diagnostic verdict                         {verdict}")

    root = Path.cwd()
    save_csv(root / "phase3c_matched_rate_null.csv", null_rows)
    save_csv(root / "phase3c_bio_gradient.csv", bio_rows)

    combined = []
    combined.extend({"dataset": "MATCHED_NULL", **r} for r in null_rows)
    combined.extend({"dataset": "BIO", **r} for r in bio_rows)
    save_csv(root / "phase3c_bilateral_gradient_diagnostic.csv", combined)

    print()
    print("Wrote:")
    print(root / "phase3c_matched_rate_null.csv")
    print(root / "phase3c_bio_gradient.csv")
    print(root / "phase3c_bilateral_gradient_diagnostic.csv")
    print("=" * 96)


if __name__ == "__main__":
    main()
