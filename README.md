# Fruit Fly Laboratory

[![Fast checks](https://github.com/vaibhavkedarisetti/fruit-fly-lab/actions/workflows/fast-checks.yml/badge.svg)](https://github.com/vaibhavkedarisetti/fruit-fly-lab/actions/workflows/fast-checks.yml)

An interactive, connectome-grounded simulation of an adult female
*Drosophila melanogaster* brain. It runs the published Shiu et al. 2024
leaky-integrate-and-fire (LIF) model over the real FlyWire FAFB v783 graph:

- 139,255 neurons
- 3,732,460 directed neuron pairs
- 50,666,648 synapses

The project is for reproducible computational experiments, education, and
software validation. It is **not** a complete digital fly, a model of
consciousness, or a substitute for biological experiments.

## What is here

- A sparse, event-driven Python LIF engine and a browser/Web Worker engine.
- A direct build pipeline from checksum-verified FlyWire v783 source files.
- Stimulus-to-circuit-to-body demonstrations, including looming escape and
  sugar-to-MN9 stimulation.
- Independent Brian2 validation for the exact sugar benchmark and representative
  whole-brain L4/L5 conditions.
- A deterministic Brian2 regression for refractory synaptic-event semantics.

See [PROJECT_BRIEF_AND_IMPACT.md](PROJECT_BRIEF_AND_IMPACT.md) for impact,
related work, and responsible ways to use the project. See
[BIOLOGICAL_ASSUMPTIONS.md](BIOLOGICAL_ASSUMPTIONS.md) before making biological
claims from a simulation result.

## Quick start

### 1. Install Python dependencies

Python 3.11+ is required. Python 3.12 is recommended for the optional Brian2
validation environment.

```powershell
python -m pip install -r requirements/requirements.txt
```

### 2. Obtain the source data

Download the FlyWire FAFB **v783** source tables from the
[FlyWire Codex download portal](https://codex.flywire.ai/api/download). The
source data are intentionally not included in this repository.

Set the location before building:

```powershell
$env:FLYWIRE_V783_DIR = "D:\path\to\FlyWire Brain Dataset (FAFB v783)"
python -m brain.connectivity.build_connectome
```

The build validates identity and source checksums. Full data requirements and
licensing are in [DATA_SOURCES.md](DATA_SOURCES.md).

### 3. Start the local laboratory

```powershell
python -m visualization.server
```

Open <http://127.0.0.1:8000>.

### 4. Run verification

```powershell
python -m pytest -q
python -m tools.verify_web_engine
```

The full suite uses the real graph and can take several minutes. For the fast
engine-level regression:

```powershell
python -m pytest tests/test_lif_engine.py -q
```

## Brian2 reference validation

Brian2 is installed separately so it cannot perturb the main environment:

```powershell
py -3.12 -m venv .brian2-venv
.\.brian2-venv\Scripts\python.exe -m pip install "numpy==1.26.4" "scipy==1.14.1" "pandas>=2.1" "brian2==2.7.1"
.\.brian2-venv\Scripts\python.exe -m simulation.brian2_reference_validation
.\.brian2-venv\Scripts\python.exe -m simulation.brian2_refractory_microprobe
```

The validation results, exact protocols, and interpretation are preserved in
[PROJECT_STATUS_TODAY.md](PROJECT_STATUS_TODAY.md).

## Tutorials and validation report

- [Guided notebooks](notebooks/README.md): looming escape, exact sugar→MN9,
  and lesion controls.
- [Technical validation report](docs/TECHNICAL_VALIDATION_REPORT.md): exact
  Brian2 comparison, whole-brain cross-check, and refractory microprobe.
- [Reviewer outreach draft](docs/REVIEWER_OUTREACH.md): a ready-to-personalize
  request for scientific feedback. It is a draft only; no reviewers are
  contacted automatically.

## Scientific scope and limitations

The simulation uses real connectivity and a published point-neuron model, but
it does not include dendrites, electrical synapses, synaptic plasticity,
spontaneous activity, a ventral nerve cord, or measured sensory transduction
for every modality. Sensory encoders and body readouts are documented modelling
choices. Please read [BIOLOGICAL_ASSUMPTIONS.md](BIOLOGICAL_ASSUMPTIONS.md).

The current L4/L5 sustained-drive regime is independently reproduced by the
Brian2 reference implementation. It is therefore a shared reference-model
regime, not evidence of a FlyWire-topology or optimized-engine defect.

## Data, citation, and licence

- Code in this repository is licensed under the [MIT License](LICENSE).
- Raw FlyWire source data are **not** included. The committed browser assets
  under `web/data/` are FlyWire-derived and retain the separate CC BY-NC-SA 4.0
  terms. Do not treat the MIT licence as applying to FlyWire data or derivative
  data products; see [NOTICE.md](NOTICE.md).
- Cite FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024) and Shiu et al.
  2024 when publishing work based on this repository. Details are in
  [CITATION.cff](CITATION.cff) and [DATA_SOURCES.md](DATA_SOURCES.md).

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md).
Changes to FlyWire topology, synaptic signs, published LIF parameters, or
biological mechanisms require explicit scientific evidence and review.

## Deployment

The browser build is static. See [DEPLOY.md](DEPLOY.md) for Vercel deployment
and [tools/verify_web_engine.py](tools/verify_web_engine.py) for the Python ↔
browser equivalence check.
