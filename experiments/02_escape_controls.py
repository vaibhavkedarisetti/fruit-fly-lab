"""
EXPERIMENT 2 -- Controls for the looming -> escape pathway.

The point of this experiment is falsifiability. If the Giant Fibre responded the
same way to every stimulus, or kept responding after its real presynaptic
partners were removed, the "escape response" would be an artefact of our encoder
rather than a property of the connectome.

Conditions
----------
  looming            object approaching  (expect strong DNp01 response)
  receding           same object retreating (expect little; LPLC2 is
                     outward-motion selective, LC4 is expansion-driven)
  static             object at fixed distance (expect none)
  looming, -LC4      looming with the 104 real LC4 neurons silenced
  looming, -LPLC2    looming with the 210 real LPLC2 neurons silenced
  looming, -both     looming with both silenced -- the pathway is cut, so any
                     remaining DNp01 activity would be spurious

Silencing follows Shiu et al. (2024): a silenced neuron makes no output.

Run:  python -m experiments.02_escape_controls
"""
from __future__ import annotations

import json
import sys

import numpy as np

import config
from brain.neurons.registry import load_connectome
from brain.sensory.encoders import LoomingEncoder
from brain.sensory.retinotopy import load_retinotopy
from simulation.engine.session import Session
from simulation.stimuli.looming import LoomingStimulus

DURATION_MS = 240.0
AZIMUTH_DEG = 45.0


def make_stimulus(kind: str) -> LoomingStimulus:
    """All conditions use the same object; only its motion differs."""
    common = dict(azimuth_deg=AZIMUTH_DEG, elevation_deg=0.0,
                  half_size_mm=5.0, t_start_ms=20.0)
    if kind == "looming":
        return LoomingStimulus(speed_mm_s=250.0, start_distance_mm=50.0, **common)
    if kind == "receding":
        # Starts at the same angular size and retreats at the same speed.
        return LoomingStimulus(speed_mm_s=-250.0, start_distance_mm=50.0, **common)
    if kind == "static":
        return LoomingStimulus(speed_mm_s=0.0, start_distance_mm=50.0, **common)
    raise ValueError(kind)


def run_condition(c, retino, kind, silence_types=(), seed=0):
    sess = Session(c, seed=seed)
    enc = LoomingEncoder(c, retino)
    sess.add_stimulus(enc, make_stimulus(kind))

    for t in silence_types:
        sess.engine.silence(c.by_cell_type(t)["idx"].to_numpy())

    gf = c.by_cell_type("DNp01")["idx"].to_numpy()
    lc4 = c.by_cell_type("LC4")["idx"].to_numpy()
    lplc2 = c.by_cell_type("LPLC2")["idx"].to_numpy()

    peak_gf_hz = 0.0
    trace = []
    for _ in range(int(DURATION_MS / 5)):
        sess.advance(5.0)
        ws = sess.recorder.window_sum
        hz = float(ws[gf].sum() / len(gf) / (sess.window_ms * 1e-3))
        peak_gf_hz = max(peak_gf_hz, hz)
        trace.append({"t_ms": sess.engine.t_ms, "gf_hz": round(hz, 1)})

    sc = sess.engine.spike_counts
    return {
        "condition": kind + ("" if not silence_types
                             else ", -" + "/-".join(silence_types)),
        "silenced": list(silence_types),
        "lc4_spikes": int(sc[lc4].sum()),
        "lplc2_spikes": int(sc[lplc2].sum()),
        "dnp01_spikes": int(sc[gf].sum()),
        "peak_dnp01_hz": round(peak_gf_hz, 1),
        "neurons_active": int((sc > 0).sum()),
        "total_spikes": int(sc.sum()),
        "trace": trace,
    }


def main():
    c = load_connectome()
    retino = load_retinotopy(c)

    conditions = [
        ("looming", ()),
        ("receding", ()),
        ("static", ()),
        ("looming", ("LC4",)),
        ("looming", ("LPLC2",)),
        ("looming", ("LC4", "LPLC2")),
    ]

    print("Escape-pathway controls on %s" % c.dataset)
    print("Object: half-size 5 mm at azimuth %+.0f deg; %.0f ms per condition\n"
          % (AZIMUTH_DEG, DURATION_MS))
    header = ("%-24s %10s %10s %10s %12s %10s"
              % ("condition", "LC4 spk", "LPLC2 spk", "DNp01 spk",
                 "peak DNp01", "active"))
    print(header)
    print("-" * len(header))

    results = []
    for kind, sil in conditions:
        r = run_condition(c, retino, kind, sil)
        results.append(r)
        print("%-24s %10d %10d %10d %10.0f Hz %10d"
              % (r["condition"], r["lc4_spikes"], r["lplc2_spikes"],
                 r["dnp01_spikes"], r["peak_dnp01_hz"], r["neurons_active"]))

    loom = results[0]
    cut = results[-1]
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("Looming peak DNp01 rate            : %.0f Hz" % loom["peak_dnp01_hz"])
    print("Receding peak DNp01 rate           : %.0f Hz" % results[1]["peak_dnp01_hz"])
    print("Static peak DNp01 rate             : %.0f Hz" % results[2]["peak_dnp01_hz"])
    print("Looming with LC4+LPLC2 silenced    : %.0f Hz" % cut["peak_dnp01_hz"])
    if cut["dnp01_spikes"] == 0 and loom["dnp01_spikes"] > 0:
        print("\nPASS: cutting the two real presynaptic populations abolishes the")
        print("      Giant Fibre response. The response is carried by the FlyWire")
        print("      connectome, not by the stimulus encoder.")
    else:
        print("\nNOTE: DNp01 still active with the pathway cut (%d spikes)."
              % cut["dnp01_spikes"])

    out = config.OUTPUT_DIR / "experiment_02_escape_controls.json"
    out.write_text(json.dumps({"dataset": c.dataset, "results": results},
                              indent=2, default=float))
    print("\nSaved: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
