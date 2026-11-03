# Fruit Fly Laboratory v0.1.0

## First public, validated release

This release packages Fruit Fly Laboratory as a reproducible, GitHub-ready
whole-brain simulation project built on FlyWire FAFB v783 and the published
Shiu et al. LIF model.

## Highlights

- Full-v783 optimized-engine comparison with an independent Brian2 2.7.1
  implementation.
- Exact 20-root published sugar-GRN → MN9 benchmark with documented protocol,
  seed, parameters, and aggregate agreement.
- L4 and L5 whole-brain cross-check showing that the sustained-drive regime is
  shared reference-model behaviour, not an optimized-engine-only defect.
- Deterministic Brian2 refractory-arrival microprobe and a fast protected
  regression for the verified F1/H3 reference-alignment semantics.
- Guided notebooks for looming escape, exact sugar→MN9, and lesion controls.
- Public README, MIT code licence, FlyWire third-party-data notice, citation
  metadata, contribution guidance, and lightweight GitHub Actions checks.

## Verification performed for this release

```powershell
python -m compileall -q brain simulation fly visualization experiments
python -m pytest tests/test_lif_engine.py -q
.\.brian2-venv\Scripts\python.exe -m simulation.brian2_refractory_microprobe
```

The full real-v783 suite is also available with `python -m pytest -q`; it is
slower because it uses the full 139,255-neuron graph.

## Data and scientific scope

- This local derivative does not assert a new code licence because the upstream
  public repository did not include one. See
  [UPSTREAM_ATTRIBUTION_AND_LICENSING.md](../UPSTREAM_ATTRIBUTION_AND_LICENSING.md).
  FlyWire-derived assets and regenerated data retain their own CC BY-NC-SA 4.0
  terms; see [NOTICE.md](../NOTICE.md).
- This is not a complete digital fly or proof of cognition/consciousness.
- See [BIOLOGICAL_ASSUMPTIONS.md](../BIOLOGICAL_ASSUMPTIONS.md) and
  [TECHNICAL_VALIDATION_REPORT.md](TECHNICAL_VALIDATION_REPORT.md) before
  interpreting results.

## Known limitations

The project is a point-neuron chemical-synapse model of one adult female FAFB
brain. It does not model dendrites, electrical synapses, the ventral nerve
cord, spontaneous activity, or full biological sensory transduction.
