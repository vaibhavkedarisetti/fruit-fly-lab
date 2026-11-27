# Technical validation report: optimized FlyWire v783 LIF engine versus Brian2

## Purpose

This report records an independent-reference validation of Fruit Fly
Laboratory's optimized whole-brain engine. It is a software-conformance report,
not a claim of a new biological mechanism or a complete behavioural model.

## Reference and environment

- Reference model: Shiu et al. (2024) Brian2 LIF implementation.
- Dataset: FlyWire FAFB v783; 139,255 neurons, 3,732,460 directed pairs, and
  50,666,648 signed synapses.
- Independent environment: Python 3.12, Brian2 2.7.1, NumPy 1.26.4, and
  SciPy 1.14.1.
- Brian2 construction: literal linear LIF integration, published 1.8 ms delay,
  signed synapse-count weights, and per-neuron `PoissonInput` to membrane
  voltage.

## Reproduce

```powershell
.\.brian2-venv\Scripts\python.exe -m simulation.brian2_reference_validation
.\.brian2-venv\Scripts\python.exe -m simulation.brian2_refractory_microprobe
python -m pytest tests/test_lif_engine.py -q
```

The first command writes `simulation/outputs/brian2_reference_validation.json`.
It requires a locally built FlyWire v783 connectome and does not change the
model, data, or topology.

## Exact sugar GRN → MN9 benchmark

Protocol: exact 20-root v783 persistence subset of the published 21-root v630
sugar-GRN population; 150 Hz, 1,000 ms, seed 0; MN9 root
`720575940660219265`.

| Metric | Optimized engine | Brian2 | Absolute difference |
| --- | ---: | ---: | ---: |
| Rest MN9 spikes | 0 | 0 | 0 |
| Stimulated MN9 spikes | 120 | 131 | 11 |
| Stimulated MN9 rate | 120 Hz | 131 Hz | 11 Hz |
| Total stimulated spikes | 21,818 | 21,955 | 137 (0.62%) |
| Active neurons | 560 | 559 | 1 (0.18%) |

The rest condition is exactly silent in both implementations. Event-by-event
identity is not expected from a shared integer seed because Brian2 and NumPy use
different Poisson random-number streams. The close aggregate agreement rules
out a gross population, sign, delay, or engine mismatch for this protocol.

## Whole-brain stability cross-check

Each condition uses the recorded v783 population, seed 0, 1,500 ms drive,
1,000–1,500 ms late window, and a 100 ms post-drive probe.

| Condition | Engine late Hz | Brian2 late Hz | Engine post-drive Hz | Brian2 post-drive Hz |
| --- | ---: | ---: | ---: | ---: |
| L4: 258 odor ORNs at 1 Hz | 2.3562 | 2.3138 | 2.3651 | 2.4411 |
| L5 k=1 at 20 Hz | 0.000158 | 0.000115 | 0 | 0 |
| L5 k=2 at 20 Hz | 2.3230 | 2.3678 | 2.3020 | 2.3631 |

The same L4 sustained state and the same L5 k=1 → k=2 transition occur in
both implementations. The remaining issue is therefore **shared
reference-model regime behaviour**, rather than an optimized-engine-only bug.

## Refractory synaptic-event semantics

The deterministic microprobe forces a target spike, then delivers a 3 mV
synaptic event during the target's refractory interval.

| Condition | Conductance retained after arrival |
| --- | ---: |
| Brian2 non-refractory control | 2.940596 mV |
| Brian2 refractory target | 0 mV |
| Optimized engine, F1/H3 off | 3.0 mV |
| Optimized engine, F1/H3 on | 0 mV |

This confirms the existing F1/H3 reference-alignment correction for the tested
semantics. `tests/test_lif_engine.py::test_f1_h3_discards_arrival_to_refractory_target`
protects the finding with a fast regression.

## Interpretation and limits

- The validation supports engine conformance under the documented protocols;
  it does not validate every possible stimulus or biological assumption.
- The L4/L5 sustained state is not silently altered because doing so would
  require a separately justified scientific model change.
- The model remains a point-neuron, chemical-synapse simulation of one female
  brain. Read [BIOLOGICAL_ASSUMPTIONS.md](../BIOLOGICAL_ASSUMPTIONS.md) before
  drawing biological conclusions.
