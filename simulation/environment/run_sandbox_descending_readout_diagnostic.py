"""
Phase 2D — Descending readout diagnostic.

Purpose
-------
Determine whether the current lateral bias is already present in DNa01/DNa02
or is introduced by the engineering forward/left/right conversion.

The test uses exactly matched bilateral sensory spikes so any residual
lateral bias must arise after the sensory spike-generation step.

Nothing in the real brain is altered. No reward or plasticity is used.
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

OUTPUT_CSV = Path("descending_readout_diagnostic.csv")


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
    """Run one matched/independent trial and expose the DN populations."""
    engine = LIFEngine(connectome, seed=seed)

    if condition == "independent":
        engine.rng = np.random.default_rng(seed)
    elif condition == "matched":
        engine.rng = PairedRNG(seed, left_idx.size)
    else:
        raise ValueError(condition)

    readout = DescendingReadout(connectome)

    all_odor_idx = np.concatenate((left_idx, right_idx))
    rates = np.full(all_odor_idx.size, sensory_rate_hz, dtype=np.float64)

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
        dn_rates = readout.rates(
            engine.spike_counts,
            SIMULATION_CHUNK_MS,
        )

        # Raw descending-neuron population rates.
        d1_l = dn_rates.get("DNa01_left", 0.0)
        d1_r = dn_rates.get("DNa01_right", 0.0)
        d2_l = dn_rates.get("DNa02_left", 0.0)
        d2_r = dn_rates.get("DNa02_right", 0.0)
        mdn_l = dn_rates.get("MDN_left", 0.0)
        mdn_r = dn_rates.get("MDN_right", 0.0)

        # The current four-command readout defines turn-side activations
        # from DNa01/DNa02 and derives the engineering forward command
        # from bilateral steering activity. Reconstruct that relationship
        # directly so we can test whether "forward" is mathematically
        # coupled to left/right by the readout itself.
        def act(hz: float) -> float:
            return hz / (hz + 60.0) if hz > 0.0 else 0.0

        d1_l_a = act(d1_l)
        d1_r_a = act(d1_r)
        d2_l_a = act(d2_l)
        d2_r_a = act(d2_r)

        raw_left_turn = float(np.mean([d1_l_a, d2_l_a]))
        raw_right_turn = float(np.mean([d1_r_a, d2_r_a]))
        raw_turn_activity = max(raw_left_turn, raw_right_turn)

        # Current engineering interface.
        reconstructed_forward = float(
            np.clip(raw_turn_activity * 4.0, 0.0, 1.0)
        )

        readout_forward = float(channels.get("forward", channels.get("locomotion_drive", 0.0)))
        readout_left = float(channels.get("left", 0.0))
        readout_right = float(channels.get("right", 0.0))

        rows.append(
            {
                "time_ms": simulation_time_ms,
                "forward": readout_forward,
                "backward": float(channels.get("backward", 0.0)),
                "left_channel": readout_left,
                "right_channel": readout_right,
                "DNa01_left_hz": d1_l,
                "DNa01_right_hz": d1_r,
                "DNa02_left_hz": d2_l,
                "DNa02_right_hz": d2_r,
                "MDN_left_hz": mdn_l,
                "MDN_right_hz": mdn_r,
                "DNa_left_activation": raw_left_turn,
                "DNa_right_activation": raw_right_turn,
                "DNa_turn_activity": raw_turn_activity,
                "reconstructed_forward": reconstructed_forward,
                "readout_forward_minus_reconstructed": readout_forward - reconstructed_forward,
            }
        )

    engine.clear_poisson()

    fields = list(rows[0].keys())
    arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in fields}

    return {
        "condition": condition,
        "seed": seed,
        "rows": rows,
        "mean_forward": float(np.mean(arr["forward"])),
        "mean_backward": float(np.mean(arr["backward"])),
        "mean_left_channel": float(np.mean(arr["left_channel"])),
        "mean_right_channel": float(np.mean(arr["right_channel"])),
        "mean_dna_left": float(np.mean(arr["DNa_left_activation"])),
        "mean_dna_right": float(np.mean(arr["DNa_right_activation"])),
        "mean_dna_bias": float(
            np.mean(arr["DNa_right_activation"] - arr["DNa_left_activation"])
        ),
        "mean_abs_dna_bias": float(
            np.mean(np.abs(arr["DNa_right_activation"] - arr["DNa_left_activation"]))
        ),
        "mean_forward_reconstruction_error": float(
            np.mean(np.abs(arr["readout_forward_minus_reconstructed"]))
        ),
        "mean_dna01_left_hz": float(np.mean(arr["DNa01_left_hz"])),
        "mean_dna01_right_hz": float(np.mean(arr["DNa01_right_hz"])),
        "mean_dna02_left_hz": float(np.mean(arr["DNa02_left_hz"])),
        "mean_dna02_right_hz": float(np.mean(arr["DNa02_right_hz"])),
        "mean_mdn_left_hz": float(np.mean(arr["MDN_left_hz"])),
        "mean_mdn_right_hz": float(np.mean(arr["MDN_right_hz"])),
    }

def main() -> int:
    print()
    print("=" * 78)
    print("DIGITAL DROSOPHILA — DESCENDING READOUT DIAGNOSTIC")
    print("=" * 78)
    print()
    print("Question:")
    print("  Is lateral bias already present in the real descending neurons,")
    print("  or is the current forward/left/right readout creating the coupling?")
    print()
    print("Real connectome: YES")
    print("Reward: DISABLED")
    print("Plasticity: DISABLED")
    print("Sensory condition: bilateral 75 Hz")
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

    # The key test is matched input because the previous experiment showed
    # the lateral bias persists even when left/right sensory spikes are identical.
    all_results = []

    for trial in range(1, TRIALS + 1):
        seed = 2000 + trial
        print("-" * 78)
        print(f"TRIAL {trial}/{TRIALS}  MATCHED INPUT  seed={seed}")
        print("-" * 78)

        result = one_trial(
            connectome,
            left_idx,
            right_idx,
            sensory_rate_hz,
            "matched",
            seed,
        )
        all_results.append(result)

        print(
            f"  DNa01  L={result['mean_dna01_left_hz']:.2f} Hz"
            f"  R={result['mean_dna01_right_hz']:.2f} Hz"
        )
        print(
            f"  DNa02  L={result['mean_dna02_left_hz']:.2f} Hz"
            f"  R={result['mean_dna02_right_hz']:.2f} Hz"
        )
        print(
            f"  DNa activation  L={result['mean_dna_left']:.3f}"
            f"  R={result['mean_dna_right']:.3f}"
            f"  bias(R-L)={result['mean_dna_bias']:+.3f}"
        )
        print(
            f"  readout  forward={result['mean_forward']:.3f}"
            f"  left={result['mean_left_channel']:.3f}"
            f"  right={result['mean_right_channel']:.3f}"
        )
        print(
            f"  reconstruction error"
            f"  |forward - max(DNa-L,DNa-R)*4|"
            f" = {result['mean_forward_reconstruction_error']:.6f}"
        )
        print(
            f"  MDN     L={result['mean_mdn_left_hz']:.2f} Hz"
            f"  R={result['mean_mdn_right_hz']:.2f} Hz"
        )
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    mean_dna_bias = float(np.mean([r["mean_dna_bias"] for r in all_results]))
    mean_abs_dna_bias = float(np.mean([r["mean_abs_dna_bias"] for r in all_results]))
    mean_reconstruction_error = float(
        np.mean([r["mean_forward_reconstruction_error"] for r in all_results])
    )

    print(f"Mean DNa signed bias (R-L) : {mean_dna_bias:+.4f}")
    print(f"Mean DNa |R-L|             : {mean_abs_dna_bias:.4f}")
    print(f"Forward reconstruction err  : {mean_reconstruction_error:.6f}")
    print()

    print("IMPORTANT:")
    print("  The current engineering forward signal is derived from the")
    print("  same DNa01/DNa02 steering activity used for left/right.")
    print("  Therefore forward and lateral steering are structurally coupled")
    print("  by the current readout. This is an engineering interface,")
    print("  not evidence that a biological forward command has been identified.")
    print()

    if mean_reconstruction_error < 1e-6:
        print(
            "RESULT: the current forward channel is mathematically explained "
            "by the DNa01/DNa02 left/right steering activity."
        )
        print(
            "NEXT: separate the engineering forward interface from the "
            "left/right steering channels rather than trying to correct "
            "the body's turning."
        )

    out_path = Path.cwd() / OUTPUT_CSV
    fieldnames = [
        "trial",
        "seed",
        "mean_dna01_left_hz",
        "mean_dna01_right_hz",
        "mean_dna02_left_hz",
        "mean_dna02_right_hz",
        "mean_dna_left",
        "mean_dna_right",
        "mean_dna_bias",
        "mean_abs_dna_bias",
        "mean_forward",
        "mean_left_channel",
        "mean_right_channel",
        "mean_forward_reconstruction_error",
        "mean_mdn_left_hz",
        "mean_mdn_right_hz",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(all_results, start=1):
            writer.writerow(
                {
                    "trial": i,
                    "seed": r["seed"],
                    **{k: r[k] for k in fieldnames if k not in {"trial", "seed"}},
                }
            )

    print(f"CSV saved to: {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
