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

---

## Brian2 validation addendum — 2026-11-03

### Environment and reproducibility

- Isolated environment: `.brian2-venv` (Python 3.12), leaving the project interpreter unchanged.
- Brian2: `2.7.1`; NumPy `1.26.4`; SciPy `1.14.1`; code generation target `numpy`.
- Command: `.\\.brian2-venv\\Scripts\\python.exe -m simulation.brian2_reference_validation`.
- Runner: `simulation/brian2_reference_validation.py`; machine-readable output: `simulation/outputs/brian2_reference_validation.json`.
- Brian2 model construction uses the same real v783 CSR graph, signed per-pair synapse counts, published constants, `method="linear"`, 1.8-ms synaptic delay, and one `PoissonInput(target_var="v", N=1)` per driven neuron.

### Exact sugar -> MN9 comparison

The exact 20-root v783 persistence subset of the published 21-root v630 population was used, at 150 Hz for 1000 ms with seed 0. The one non-persistent v630 root remains `720575940620900446`; MN9 is `720575940660219265` (index 138947).

| Metric | LIFEngine | Brian2 | Absolute difference | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| Rest MN9 spikes | 0 | 0 | 0 | 0% |
| Rest total spikes | 0 | 0 | 0 | 0% |
| Stimulated MN9 spikes | 120 | 131 | 11 | 8.40% |
| Stimulated MN9 Hz | 120 | 131 | 11 | 8.40% |
| Stimulated total spikes | 21,818 | 21,955 | 137 | 0.62% |
| Stimulated active neurons | 560 | 559 | 1 | 0.18% |

The rest control is exactly silent in both implementations. Exact individual spike trains are not expected from identical integer seeds alone, because NumPy's `Generator` and Brian2 use different random-number streams for Poisson events. The close aggregate match supports a numerical/stochastic difference rather than a population, sign, delay, or gross implementation mismatch.

### Whole-brain Brian2 cross-check

All rows use the existing v783 selections and duration/window definitions (1500-ms drive, 1000--1500-ms late window, then 100-ms post-drive probe).

| Condition | LIF late Hz | Brian2 late Hz | LIF post-drive Hz | Brian2 post-drive Hz | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| L4, seed 0, 258 odor ORNs, 1 Hz | 2.3562 | 2.3138 | 2.3651 | 2.4411 | sustained in both |
| L5, seed 0, k=1, 20 Hz | 0.000158 | 0.000115 | 0 | 0 | quiet in both |
| L5, seed 0, k=2, 20 Hz | 2.3230 | 2.3678 | 2.3020 | 2.3631 | sustained in both |

The largest whole-brain aggregate discrepancy is about 3.1% (L4 post-drive rate); the qualitative states and the L5 k=1 -> k=2 transition agree exactly.

### Interpretation

- Classification: **D — shared reference-model regime behavior**, with small expected stochastic/numerical differences.
- The sustained attractor is independently reproduced by published-model Brian2. It is not an optimized-engine-only defect.
- F1/H3 was active in the LIFEngine long full-v783 conditions but is absent from the Brian2 model. Brian2 nevertheless reproduces the sugar response, L4 attractor, and L5 threshold transition. F1/H3 is therefore not demonstrated necessary for those qualitative outcomes; it remains preserved pending a dedicated removal/preservation comparison.
- No model parameter, topology, or synaptic sign was changed for this validation.

### Engineering repair after validation

- Restored the pre-existing, opt-in legacy reward-learning API required by `tests/test_plasticity.py`. It remains disabled by default and does not participate in the published/reference path or Brian2 runs.
- Restored web-test dependency availability; the focused suite command `python -m pytest tests/test_plasticity.py tests/test_lif_engine.py tests/test_sugar_mn9_benchmark.py tests/test_sensory_and_body.py -q` passed: **40 passed** (two FastAPI deprecation warnings only).

### Exact next step

Run a dedicated, preservation-safe F1/H3 A/B diagnostic against the same recorded protocols before considering any alteration to that existing conditional correction. This is a diagnostic only; it must not alter topology, signs, published LIF parameters, or introduce a mechanism.

### Checkpoint

- Completed checkpoint: `096892a` (`Validate v783 engine against Brian2`).
- The original checkpoint ID `11eeb1e` was superseded when this documentation completion was amended into the checkpoint.

### Full-suite completion

- The subsequent full `python -m pytest -q` run completed after the focused run. Pytest collected 92 node IDs and its `lastfailed` cache is empty.
- The test runtime is dominated by the repository's real full-brain integration and preservation tests; no test was removed, weakened, or skipped.

## F1/H3 preservation A/B diagnostic — 2026-11-03

Ran `python -m simulation.f1_h3_ab_diagnostic` with the same v783 populations,
durations, rates, and seed 0 used for the Brian2 validation. The diagnostic
constructs two fresh engines per condition and, only for the in-memory control
arm, clears the already-calculated F1/H3 gate. It changes no source behavior,
published parameter, topology, synaptic sign, or mechanism. The machine-readable
result is the ignored artifact `simulation/outputs/f1_h3_ab_diagnostic.json`.

| condition / metric | F1/H3 on | F1/H3 off | on minus off |
| --- | ---: | ---: | ---: |
| sugar -> MN9 spikes / Hz | 120 / 120 | 186 / 186 | -66 / -66 |
| sugar total spikes | 21,818 | 28,566 | -6,748 (-23.6%) |
| sugar active neurons | 560 | 624 | -64 (-10.3%) |
| L4 late brain rate | 2.356 Hz | 4.114 Hz | -1.758 Hz (-42.7%) |
| L4 post-drive rate | 2.365 Hz | 4.124 Hz | -1.759 Hz (-42.6%) |
| L5 k=1 post-drive rate | 0 Hz | 0 Hz | 0 Hz |
| L5 k=2 late brain rate | 2.323 Hz | 4.117 Hz | -1.794 Hz (-43.6%) |
| L5 k=2 post-drive rate | 2.302 Hz | 4.101 Hz | -1.799 Hz (-43.9%) |

Interpretation: F1/H3 is consequential in the currently installed engine: it
suppresses rather than removes the shared reference-model attractor, while
preserving the L5 k=1 quiet / k=2 sustained transition. It is an engineering
correction absent from the published Brian2 reference, not evidence that the
published model itself is stable. Do not remove or alter it without explicit
approval and a separate scientific decision.

Reference-alignment consequence: compared with the Brian2 validation above,
the F1/H3-on engine is the matching arm (sugar MN9 120 versus Brian2 131;
L4 late rate 2.356 versus 2.314 Hz; L5 k=2 late rate 2.323 versus 2.368 Hz).
The off arm is substantially farther away (186 MN9 spikes, 4.114 Hz L4, and
4.117 Hz L5 k=2). Thus the prior statement that F1/H3 was not demonstrated
necessary applies only to the coarse qualitative outcomes; this A/B result
demonstrates it is necessary for the optimized engine's close quantitative
agreement with the Brian2 reference under these protocols.

### Deterministic refractory-arrival microprobe

Ran `.\\.brian2-venv\\Scripts\\python.exe -m simulation.brian2_refractory_microprobe`.
The probe forces a target spike and then delivers a 3 mV synaptic event while
that target is refractory. A non-refractory Brian2 delivery control retained
nonzero conductance, confirming the event was delivered. Brian2 2.7.1 retained
**0 mV** for the refractory target; the LIFEngine retained 3 mV with F1/H3 off
and 0 mV with F1/H3 on. This isolates the correction from stochastic
whole-brain effects and confirms the on-arm as the Brian2-compatible behavior.
`tests/test_lif_engine.py::test_f1_h3_discards_arrival_to_refractory_target` is
the fast regression for it.
