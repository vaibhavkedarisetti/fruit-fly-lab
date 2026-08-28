"""
EXPERIMENT 1 -- Visual looming -> escape response.

Pipeline (no step is a hand-written behavioural rule):

    approaching object (exact geometry)
      -> LC4 / LPLC2 firing rates via published tuning + FlyWire-derived
         receptive fields
      -> Poisson drive onto the REAL LC4 (104) and LPLC2 (210) neurons
      -> whole-brain LIF simulation over 139,255 real neurons and
         3,732,460 real connections
      -> DNp01 (Giant Fibre) and other real descending neurons
      -> motor channel readout

Run:  python -m experiments.01_looming_escape
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


def run(azimuth_deg=45.0, l_over_v_ms=20.0, duration_ms=280.0,
        t_start_ms=20.0, seed=0, quiet=False):
    c = load_connectome()
    retino = load_retinotopy(c)

    sess = Session(c, seed=seed)
    enc = LoomingEncoder(c, retino)

    # l/v = 20 ms: half_size 5 mm at 250 mm/s
    stim = LoomingStimulus(
        azimuth_deg=azimuth_deg, elevation_deg=0.0,
        half_size_mm=5.0, speed_mm_s=5.0 / (l_over_v_ms / 1000.0),
        start_distance_mm=50.0, t_start_ms=t_start_ms,
    )
    sess.add_stimulus(enc, stim)

    watch = {t: c.by_cell_type(t)["idx"].to_numpy()
             for t in ("LC4", "LPLC2", "DNp01", "DNp02", "DNp04", "DNp11", "DNp09")}
    prev = {k: 0 for k in watch}
    rows = []

    if not quiet:
        print("Stimulus: azimuth %+.0f deg, l/|v| = %.0f ms, collision at t = %.0f ms"
              % (azimuth_deg, stim.l_over_v_ms, stim.collision_time_ms))
        print("Driving %d LC4 + %d LPLC2 real FlyWire neurons\n"
              % (enc.provenance["drives"]["LC4"], enc.provenance["drives"]["LPLC2"]))
        print("   t_ms   theta   dtheta/dt |    LC4   LPLC2  |  DNp01  DNp02  DNp04  DNp11 | escape")
        print("   " + "-" * 88)

    for _ in range(int(duration_ms / 5)):
        sess.advance(5.0)
        t = sess.engine.t_ms
        cur = {k: int(sess.engine.spike_counts[v].sum()) for k, v in watch.items()}
        d = {k: cur[k] - prev[k] for k in cur}
        prev = cur
        st = stim.state(t)
        ch = sess.readout.channels(sess.recorder.window_sum, sess.window_ms)
        rows.append({"t_ms": t, **st, **{("d_" + k): d[k] for k in d}, **ch})

        if not quiet and (int(t) % 20 == 0 or d["DNp01"] > 0):
            print("  %5.0f  %5.1f   %9.0f | %6d %6d  | %6d %6d %6d %6d | %.2f"
                  % (t, st["half_angle_deg"], st["expansion_rate_deg_s"],
                     d["LC4"], d["LPLC2"], d["DNp01"], d["DNp02"],
                     d["DNp04"], d["DNp11"], ch.get("escape_takeoff", 0.0)))

    return sess, stim, rows


def main():
    sess, stim, rows = run()
    c = sess.c

    gf = c.by_cell_type("DNp01")
    gf_idx = gf["idx"].to_numpy()
    gf_spikes = int(sess.engine.spike_counts[gf_idx].sum())
    first = next((r for r in rows if r["d_DNp01"] > 0), None)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print("Giant Fibre (DNp01) total spikes : %d" % gf_spikes)
    if first:
        print("First GF spike at                : t = %.0f ms "
              "(%.0f ms before collision, theta = %.1f deg)"
              % (first["t_ms"], stim.collision_time_ms - first["t_ms"],
                 first["half_angle_deg"]))
    print("Neurons that fired at least once : %d of %d"
          % (int((sess.engine.spike_counts > 0).sum()), c.n))
    print("Total spikes                     : %d" % int(sess.engine.spike_counts.sum()))

    out = config.OUTPUT_DIR / "experiment_01_looming_escape.json"
    out.write_text(json.dumps({
        "experiment": "visual looming -> escape",
        "dataset": c.dataset,
        "stimulus": stim.state(stim.collision_time_ms),
        "provenance": sess.provenance,
        "timeseries": rows,
    }, indent=2, default=float))
    print("\nSaved: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
