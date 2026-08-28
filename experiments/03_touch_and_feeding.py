"""
EXPERIMENT 3 -- Touch by body region, and feeding.

Two things are demonstrated:

1. FEEDING (reproducing the reference paper's headline result).
   Sugar gustatory receptor neurons drive the proboscis motor neurons;
   bitter gustatory receptor neurons do not. Both populations are identified by
   real FlyWire community labels -- the sugar GRNs were labelled by Philip Shiu
   (Kristin Scott Lab), an author of the reference model.

2. TOUCH BY BODY REGION.
   The user asked for HEAD / THORAX / ABDOMEN / LEG. Only HEAD is answerable:
   head bristle mechanosensory neurons are in the brain dataset. Thoracic,
   abdominal and leg mechanosensory neurons project to the ventral nerve cord,
   which FlyWire FAFB v783 does not contain. Those are reported as
   "Not currently modeled" rather than faked.

Run:  python -m experiments.03_touch_and_feeding
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

import config
from brain.neurons.labels import functional_group
from brain.neurons.registry import load_connectome
from brain.sensory.modalities import BY_KEY, resolve_neurons
from simulation.engine.session import Session

DURATION_MS = 400.0


def _group_idx(c, name):
    return np.array([c.idx(r) for r in functional_group(name)
                     if int(r) in c._id2idx], dtype=np.int64)


def run_modality(c, key, readouts, seed=0):
    """Deliver one stimulus and report what the real readout neurons did."""
    m = BY_KEY[key]
    if not m.supported:
        return {"key": key, "label": m.label, "supported": False,
                "reason": m.unsupported_reason}

    sess = Session(c, seed=seed)
    sess.add_modality(key, intensity=1.0, duration_ms=DURATION_MS)
    n_driven = len(resolve_neurons(m, c))

    peak = {k: 0.0 for k in readouts}
    for _ in range(int(DURATION_MS / 5)):
        sess.advance(5.0)
        ws = sess.recorder.window_sum
        for k, idx in readouts.items():
            if len(idx):
                hz = float(ws[idx].sum() / len(idx) / (sess.window_ms * 1e-3))
                peak[k] = max(peak[k], hz)

    sc = sess.engine.spike_counts
    return {
        "key": key, "label": m.label, "supported": True,
        "n_driven": n_driven,
        "neurons_active": int((sc > 0).sum()),
        "total_spikes": int(sc.sum()),
        "peak_hz": {k: round(v, 1) for k, v in peak.items()},
        "readout_spikes": {k: int(sc[idx].sum()) for k, idx in readouts.items()},
        "behaviour": sess.history[-1]["body"]["behaviour"],
        "proboscis": round(sess.history[-1]["body"]["proboscis_extension"], 2),
        "citation": m.citation,
    }


def main():
    c = load_connectome()

    readouts = {
        "proboscis_MN": _group_idx(c, "proboscis_motor"),
        "descending": c.neurons[c.neurons["super_class"].astype(str)
                                == "descending"]["idx"].to_numpy(),
        "DNp01": c.by_cell_type("DNp01")["idx"].to_numpy(),
    }
    print("Readout populations (all real FlyWire neurons):")
    for k, v in readouts.items():
        print("  %-14s %5d neurons" % (k, len(v)))
    print()

    conditions = ["taste_sugar", "taste_bitter", "touch_head", "touch_leg_taste",
                  "wind", "sound", "odor_vinegar", "odor_geosmin", "heat", "cold",
                  "touch_thorax", "touch_abdomen", "touch_leg"]

    header = "%-30s %7s %8s %11s %10s %9s" % (
        "stimulus", "driven", "active", "prob.MN Hz", "DNp01 Hz", "prob.ext")
    print(header)
    print("-" * len(header))

    results = []
    for key in conditions:
        r = run_modality(c, key, readouts)
        results.append(r)
        if not r["supported"]:
            print("%-30s %7s %8s %11s %10s %9s"
                  % (r["label"][:30], "--", "--", "--", "--", "NOT MODELED"))
            continue
        print("%-30s %7d %8d %11.1f %10.1f %9.2f"
              % (r["label"][:30], r["n_driven"], r["neurons_active"],
                 r["peak_hz"]["proboscis_MN"], r["peak_hz"]["DNp01"],
                 r["proboscis"]))

    print("\n" + "=" * 78)
    print("NOT CURRENTLY MODELED")
    print("=" * 78)
    for r in results:
        if not r["supported"]:
            print("  %s" % r["label"])
            print("    %s\n" % r["reason"])

    sugar = next(r for r in results if r["key"] == "taste_sugar")
    bitter = next(r for r in results if r["key"] == "taste_bitter")
    print("=" * 78)
    print("FEEDING RESULT (compare Shiu et al. 2024)")
    print("=" * 78)
    print("  sugar  -> proboscis motor neurons peak at %.1f Hz, extension %.2f"
          % (sugar["peak_hz"]["proboscis_MN"], sugar["proboscis"]))
    print("  bitter -> proboscis motor neurons peak at %.1f Hz, extension %.2f"
          % (bitter["peak_hz"]["proboscis_MN"], bitter["proboscis"]))
    if sugar["peak_hz"]["proboscis_MN"] > 5 and bitter["readout_spikes"]["proboscis_MN"] == 0:
        print("\n  PASS: sugar evokes proboscis motor output and bitter does not,")
        print("        reproducing the published behaviour of the reference model.")

    out = config.OUTPUT_DIR / "experiment_03_touch_and_feeding.json"
    out.write_text(json.dumps({"dataset": c.dataset, "results": results},
                              indent=2, default=float))
    print("\nSaved: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
