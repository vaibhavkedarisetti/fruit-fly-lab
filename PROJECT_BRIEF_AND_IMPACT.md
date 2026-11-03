# Fruit Fly Laboratory — project brief and impact plan

## One-sentence description

Fruit Fly Laboratory is an interactive, reproducible whole-brain simulation of
one adult female *Drosophila melanogaster*, built from the real FlyWire FAFB
v783 connectome and the published Shiu et al. leaky integrate-and-fire (LIF)
model.

It is a research and education tool, **not** a claim to simulate a fly's mind,
consciousness, or full real-world behaviour.

## What this project has accomplished

### Real data and reproducibility

- Builds its graph directly from the public FlyWire FAFB v783 source tables.
- Represents **139,255 real neurons**, **3,732,460 neuron pairs**, and
  **50,666,648 synapses**.
- Checks source-file hashes and validates neuron, synapse, class, and spatial
  counts, rather than accepting a preprocessed graph without provenance.
- Keeps FlyWire root IDs and annotations available for inspection.

### Published-model implementation

- Implements the Shiu et al. 2024 whole-brain LIF equations, signed synapses,
  Poisson-drive convention, 1.8 ms delay, and published constants.
- Uses a sparse, event-driven Python engine so the whole graph can be run in an
  interactive sensory-to-body loop.
- Supports reproducible experiments, including a fixed seed and exact sugar
  gustatory-neuron to MN9 benchmark.

### Independent validation completed in this repository

- Created an isolated Brian2 2.7.1 environment so the production Python
  environment remains unchanged.
- Compared the optimized engine with a literal Brian2 reference over the full
  v783 graph. The exact sugar-to-MN9 benchmark agreed closely in aggregate:
  120 versus 131 MN9 spikes, 21,818 versus 21,955 total spikes, and 560 versus
  559 active neurons. Independent Poisson random-number streams prevent
  spike-for-spike identity at a shared numeric seed.
- Confirmed independently that the L4 sustained-drive regime and L5 k=1/k=2
  transition occur in Brian2 too. This classifies that regime as shared
  reference-model behaviour, not an optimized-engine-only defect.
- Ran a deterministic Brian2 refractory-arrival microprobe. Brian2 retained
  0 mV for a synaptic event arriving at a refractory target, while the engine
  matches only when its existing F1/H3 reference-alignment correction is on.
  A non-refractory control retained 2.940596 mV. This is now protected by a
  fast regression test.

See [PROJECT_STATUS_TODAY.md](PROJECT_STATUS_TODAY.md) for commands, raw
numbers, and the full interpretation.

### Usable experience

- Provides sensory inputs, circuit-level activity, descending-neuron readout,
  and a kinematic fly body.
- Starts locally through the visualization server. A live smoke test loaded the
  real v783 graph and returned HTTP 200 from the application root.
- Separates data, published assumptions, project approximations, and software
  engineering in [BIOLOGICAL_ASSUMPTIONS.md](BIOLOGICAL_ASSUMPTIONS.md).

## Why this can help other people

| Audience | Practical value |
| --- | --- |
| Connectomics researchers | A reproducible executable view of the v783 graph, with a direct route from primary files to signed sparse connectivity. |
| Computational neuroscientists | A fast implementation plus a Brian2 reference harness for testing changes, numerical semantics, or alternate backends. |
| Students and educators | A concrete way to see how a stimulus can propagate through identified neurons and connections instead of being represented as a black-box behavioural rule. |
| Software/ML engineers | A real large sparse directed graph with a documented simulator, fixtures, provenance checks, and benchmark protocols. |
| Scientific communicators | An inspectable demonstration of what a connectome enables—and what it does not establish about behaviour or cognition. |

## Honest boundaries

This project should not be presented as a complete digital fly. In particular:

- FAFB is one adult female brain; it does not contain the ventral nerve cord,
  so it cannot directly simulate the leg and wing motor circuits.
- It is a point-neuron LIF model: no compartments, gap junctions, realistic
  synaptic diversity, spontaneous activity, adaptation, or neuromodulatory
  state.
- Neurotransmitter labels are predictions, not direct measurements.
- Sensory encoding and the mapping from descending activity to the displayed
  body are modelling choices; they are documented rather than hidden.
- The shared whole-brain sustained-drive regime is a result/limitation of the
  published reference-model regime. It must not be “fixed” by silently changing
  FlyWire topology, synaptic signs, or published parameters.

Read [BIOLOGICAL_ASSUMPTIONS.md](BIOLOGICAL_ASSUMPTIONS.md) before using an
output as a biological claim.

## Relationship to existing public work

The project is in an active ecosystem; it is neither the first FlyWire viewer
nor the only whole-brain fly simulator.

| Project | Relationship to this project |
| --- | --- |
| [vaibhavkedarisetti/fruit-fly-lab](https://github.com/vaibhavkedarisetti/fruit-fly-lab) | This is the publicly indexed repository with the same project name and description: an interactive v783, 139,255-neuron Shiu-model simulation. Treat it as the public home/lineage of this work rather than a competing unrelated project. |
| [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | The authoritative Brian2 implementation accompanying Shiu et al. It is the scientific reference model, not an interactive optimized v783 application. This repository now supplies a dedicated independent comparison harness. |
| [snedea/flybrain](https://github.com/snedea/flybrain) | A closely related interactive browser LIF simulation of FlyWire v783. It demonstrates that browser-facing whole-brain fly experiences are an active area; its published description reports 139,255 neurons and 2.7M connections. |
| [kornel77777/fly-brain-sim](https://github.com/kornel77777/fly-brain-sim) | A related v783 data/visualization project focused on a reproducible DuckDB data build and 3D viewer rather than this project’s validated LIF sensory-to-body loop. |
| [FlyWire tools and Codex](https://flywire.ai/apps) | The upstream ecosystem for graph exploration, annotations, proofreading, and data access—not a replacement for a runnable dynamics model. |

The useful distinction to communicate is: **this project combines verified
primary-data provenance, a runnable whole-brain LIF engine, interactive
stimulus-to-readout experiments, and an explicit Brian2 validation trail.**

## How to make the project useful to others

### Near-term: make it easy to trust and run

1. Restore a general project README as the main entry point; keep experimental
   F1/F3 notes in separate documents. Include a 5-minute quick start, system
   requirements, and a screenshot/GIF.
2. Add a repository license for the code after choosing one. Keep the FlyWire
   CC BY-NC-SA 4.0 data terms and required citations separate; do not imply the
   data are covered by the code license.
3. Publish a small “validation results” page that links to the Brian2 commands,
   benchmark JSON, F1/H3 microprobe, and the exact commits that produced them.
4. Add continuous integration for the fast tests and code formatting. Keep the
   full v783 integration suite as a scheduled/manual job because it is expensive.
5. Tag a release and archive its source/metadata in a DOI-capable service such
   as Zenodo, after checking the data licence and repository ownership.

### Research and teaching directions

1. Provide ready-made notebooks/tutorials for sugar-to-MN9, looming-to-Giant
   Fibre, lesion controls, and “why the rest brain is silent.”
2. Add an experiment template that records dataset version, root IDs, seed,
   LIF parameters, stimulus protocol, commit hash, and outputs automatically.
3. Invite domain experts to review sensory encoders and descending-neuron
   behaviour mappings; label accepted changes with their evidence level.
4. Use the simulator for *hypothesis generation*, then compare proposed circuit
   explanations with FlyWire connectivity and published experiments. Do not
   describe unvalidated output as a biological prediction.
5. If expanding to leg or wing output, create a separate, versioned BANC/v888
   pipeline. Do not silently merge it into FAFB v783.

## Safe public-facing wording

> Fruit Fly Laboratory is an open, connectome-grounded simulation platform.
> It runs a published whole-brain spiking model over the FlyWire FAFB v783
> connectivity graph and makes every major data source and modelling assumption
> inspectable. It is useful for reproducible computational experiments,
> education, and engineering validation; it is not a claim of a conscious or
> behaviourally complete digital animal.

## Citation and acknowledgement

When sharing results, cite the FlyWire connectome and annotation papers plus
Shiu et al. 2024 for the model. Follow the data requirements documented in
[DATA_SOURCES.md](DATA_SOURCES.md), including the FlyWire CC BY-NC-SA 4.0
terms. Cite the exact repository commit and experiment configuration for any
result produced by this code.
