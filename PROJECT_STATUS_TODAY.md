# Project Status — 2026-09-18

## Starting state

- Git commit: `284af7c` (`Establish whole-brain stability baseline`)
- Branch: `wholebrain-network-stability`
- F3: not installed in the active engine; experimental candidate files were present, untracked.
- F4: installed in the active engine as a conditional spike-frequency-adaptation current.
- F5: not installed in the active engine; experimental candidate files were present, untracked.
- F1/H3: installed as a conditional full-v783 correction in `LIFEngine.set_run_duration()` and `step()`; preserved.
- Test baseline: `python -m pytest tests/test_lif_engine.py -q` reported 11 passed, 1 failed. The failure was the existing H5 refractory Poisson regression test.

## Changes made today

| File | Before | After | Reason | Provenance |
| --- | --- | --- | --- | --- |
| `simulation/engine/lif_engine.py` | F4 adaptation state/current was conditionally active on full-v783 F1/H3 runs. | Removed all 22 F4-specific inserted lines; no F3/F4/F5 mechanism remains active. | F4 was a failed experimental mechanism, outside the published model. | Engineering removal |
| `simulation/engine/lif_engine.py` | Thresholding allowed a refractory Poisson-driven neuron to emit a spike. | Thresholding now requires `rfc_left == 0`. | Matches the literal reference/Brian2 refractory semantics; fixes H5. | Bug fix / reference conformance |
| `simulation/benchmark_sugar_mn9.py` | No exact-population, reproducible full-v783 benchmark. | Added JSON-producing 20-GRN sugar-to-MN9 benchmark. | Separate the published sensorimotor benchmark from broad label-based convenience groups. | Published/reference protocol plus engineering interface |
| `tests/test_sugar_mn9_benchmark.py` | No regression check for the benchmark population identity. | Verifies 21 published v630 roots map to exactly 20 v783 roots and validates MN9 index. | Prevent accidental population drift. | Bug-prevention / data identity |

F1/H3 was not added or altered today. It remains active only for a full-v783, >=500 ms, connected-Poisson condition. Its independent Brian2 validation remains unresolved because Brian2 is not installed in this environment.

## Tests

| Test command | Result | Notes |
| --- | --- | --- |
| `python -m pytest tests/test_lif_engine.py -q` (before edit) | 11 passed, 1 failed | H5: refractory Poisson target incorrectly spiked. |
| `python -m pytest tests/test_lif_engine.py -q` (after edit) | 12 passed | H5 now matches the literal reference. |
| `python -m pytest tests/test_lif_engine.py tests/test_sugar_mn9_benchmark.py -q` | 13 passed | One non-failing pytest cache warning (`.pytest_cache` path conflict). |
| `python -m simulation.benchmark_sugar_mn9 --output simulation/outputs/sugar_mn9_benchmark_seed0.json` | passed | Full 139,255-neuron rest and stimulated trials; JSON written. |
| `python -c "import brian2; ..."` | unavailable | `ModuleNotFoundError: No module named 'brian2'`; no external Brian2 rerun was possible. |

## Sugar -> MN9 benchmark

- Published population source: `figures.ipynb`, Figure 1D / Figure 3A, in `philshiu/Drosophila_brain_model`.
- The reference list contains 21 v630 labellar sugar-GRN roots. `720575940620900446` is absent from this v783 graph; the remaining exact 20 roots were stimulated:
  `720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538, 720575940639198653, 720575940617937543, 720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570`.
- Local/global indices: `69416, 98045, 123346, 124845, 29414, 101067, 110984, 51345, 49815, 130314, 127443, 28958, 127169, 33003, 108934, 111872, 14903, 91001, 92717, 12549`.
- Stimulus: 150 Hz, 1000 ms, seed 0; real FlyWire FAFB v783, 139,255 neurons, 3,732,460 pairs, 50,666,648 signed synapses.
- MN9: root `720575940660219265`, global index `138947`.
- Rest: MN9 spikes 0; MN9 0 Hz; total network spikes 0; active neurons 0.
- Stimulated: MN9 spikes 120; MN9 120 Hz; total network spikes 21,818; active neurons 560.
- Interpretation: the clean-rest and 120 Hz MN9 response reproduce the published reference regime (the published notebook uses the same 150 Hz protocol; published discussion reports a roughly 120–150 Hz response regime). No parameter was tuned.

Machine-readable result: `simulation/outputs/sugar_mn9_benchmark_seed0.json` (ignored generated output).

## Whole-brain stability

- Baseline/reference-conforming result: the prior `stability_L4_F1ctx.csv` records sustained stimulus-independent activity under the documented L4 drive: at 1 Hz input, brain rate was about 2.32–2.36 Hz, active fraction about 0.051–0.052, and post-drive decay fraction about 0.99–1.02.
- Whether S is satisfied: no. The existing L4/L5 results retain the attractor; F3, F4, and F5 previously failed their success criterion and are not active.
- Evidence: recorded L4 F1-context results and prior independent-Brian2 cross-check artifacts in the working tree. This task did not rerun a new whole-brain sweep.
- Limitations: this is a model-regime limitation/diagnostic finding, not evidence to change the real topology, signs, or published LIF parameters. A fresh independent Brian2 check could not run here because Brian2 is absent.

## What succeeded

- Removed the active F4 experimental adaptation mechanism without disturbing F1/H3.
- Fixed and tested the refractory threshold mismatch against the repository's literal published-model reference.
- Reconstructed the exact v783 persistence subset of the published sugar-GRN population.
- Reproduced a clean-rest, sugar-driven MN9 response at 120 Hz on the full real v783 connectome.

## What did not succeed

- An independent full-v783 Brian2 execution was not performed because Brian2 is not installed.
- The previous whole-brain sustained-drive criterion remains unsatisfied; no new F3/F4/F5 tuning was attempted.

## Unresolved

- F1/H3 is preserved as the current conditional correction, but still needs a clean independent Brian2 full-v783 comparison under the exact benchmark and L4/L5 protocols.
- The repository contains many pre-existing, untracked experimental scripts, backups, CSVs, data, and results. They were left untouched and are not part of this checkpoint.
- The full suite was invoked; the focused engine and benchmark suite is the confirmed green verification recorded above.

## Recommended next step

Install Brian2 in an isolated environment and run a one-seed, exact 20-GRN full-v783 reference comparison against `simulation/benchmark_sugar_mn9.py`, recording MN9 spikes and total spikes without changing model parameters.
