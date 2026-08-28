# Fruit Fly Laboratory

An interactive simulation of an adult female *Drosophila melanogaster* built on
the **real FlyWire FAFB v783 connectome** — 139,255 neurons, 3,732,460
connections, 50,666,648 synapses — running the **published whole-brain
leaky integrate-and-fire model** of Shiu et al. (2024, *Nature* 634:210–219).

Throw a rock at the fly and it escapes. The escape is not a rule anyone wrote:
it comes out of real LC4 and LPLC2 neurons driving the real Giant Fibre through
the real wiring diagram. Silence those 314 neurons and the escape disappears
entirely.

**No large language model is used anywhere in this project.** No transformer,
no chatbot, no agent, no RAG, no LLM-generated rules. The pipeline is
connectome → differential equations → spikes.

---

## The pipeline

```
looming object (exact geometry)
  → LC4 / LPLC2 firing rates      published tuning × FlyWire-derived receptive fields
  → Poisson drive onto 314 REAL FlyWire neurons
  → 139,255-neuron LIF simulation over 3,732,460 real connections
  → real descending neurons (DNp01 = Giant Fibre)
  → motor channels
  → digital fly body
```

---

## 1. Exact software to install

| Requirement | Version used |
|---|---|
| Python | 3.13.14 (3.11+ works) |
| numpy | 2.4.2 |
| scipy | 1.16.2 |
| pandas | 3.0.1 |
| fastapi | 0.110.1 |
| uvicorn | 0.25.0 |
| websockets | 16.0 |
| pytest | 9.0.2 |

Optional: `brian2`, to cross-check against the original reference
implementation. Not needed to run the laboratory.

**GPU acceleration is not implemented.** The CPU engine is event-driven — each
step touches only the CSR rows of neurons that actually spiked — so its cost
scales with spike count, not with the 3.7M edges. At the activity levels these
experiments produce (hundreds to a few thousand active neurons) the bottleneck
is the dense 139,255-element membrane update, and a GPU port would be a modest
win rather than a large one. Shipping an untested CUDA path would risk exactly
the kind of silent numerical divergence the test suite exists to prevent. If you
want one, `simulation/engine/lif_engine.py` is written against a small array
interface, and `tests/test_lif_engine.py::test_matches_literal_reference_implementation`
is the equivalence test any new backend must pass.

## 2. Exact commands to install dependencies

```bash
python -m pip install -r requirements/requirements.txt
```

Optional extras:

```bash
python -m pip install -r requirements/requirements-dev.txt
```

## 3. Exact commands to obtain the real FlyWire data

The data is **already present** at `D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)`
and its SHA-256 checksums are recorded in `data/metadata/flywire_v783_checksums.txt`.

To obtain it from scratch:

1. Create a free FlyWire account and accept the data licence at
   <https://flywire.ai> (the download requires sign-in; there is no anonymous
   URL, so this step cannot be scripted).
2. Go to <https://codex.flywire.ai/api/download>, select **data version 783**,
   and download at minimum these files:

   ```
   neurons.csv.gz
   classification.csv.gz
   consolidated_cell_types.csv.gz
   connections_princeton.csv.gz
   coordinates.csv.gz
   column_assignment.csv.gz
   labels.csv.gz
   visual_neuron_types.csv.gz
   ```

3. Put them in one directory and point the project at it:

   ```bash
   export FLYWIRE_V783_DIR="/path/to/FlyWire Brain Dataset (FAFB v783)"
   ```

   On Windows PowerShell:

   ```powershell
   $env:FLYWIRE_V783_DIR = "D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)"
   ```

4. Verify the download matches what this project was built against:

   ```bash
   python -m pytest tests/test_data_provenance.py -q
   ```

5. Build the simulation-ready connectome (about 2 minutes):

   ```bash
   python -m brain.connectivity.build_connectome
   ```

   Expected output ends with:

   ```
   n_neurons        139255
   n_neuron_pairs  3732460
   n_synapses     50666648
   ```

## 4. Exact repository / files being used

| Purpose | Source |
|---|---|
| Connectome data | FlyWire FAFB **v783**, <https://codex.flywire.ai/api/download> |
| Neuron model | Shiu et al. 2024, [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9) |
| Reference code inspected | <https://github.com/philshiu/Drosophila_brain_model> — file `model.py` |
| Also inspected | <https://github.com/eonsystemspbc/fly-brain> |
| Not used | `snedea/flybrain`, `erojasoficial-byte/fly-brain` |

No third-party repository is vendored. `brain/neuron_models/lif.py` transcribes
the constants and equations from the authors' `model.py`, and
`tests/test_lif_engine.py` proves our engine is spike-for-spike identical to a
literal transcription of that Brian2 network. Full detail in
[`DATA_SOURCES.md`](DATA_SOURCES.md).

## 5. Hardware requirements

| | |
|---|---|
| CPU | Any x86-64. Developed on 16 cores; the engine is single-threaded. |
| RAM | **4 GB minimum, 8 GB comfortable.** The sparse connectome is ~45 MB in memory; the build step peaks around 3 GB while reading the 5.3M-row connectivity table. |
| Disk | 1.5 GB for the eight required source files. (The full Codex release including meshes is 17 GB; most of it is unused — see `DATA_SOURCES.md`.) |
| GPU | Not required. |

Performance: **≈1.0 ms wall-clock per 0.1 ms simulated step**, i.e. about
10× slower than real time for the whole 139,255-neuron brain. A 300 ms escape
trial takes roughly 3 seconds. Spike propagation is event-driven, so cost scales
with spike count rather than with the 3.7M edges.

## 6. First experiment to run

```bash
python -m experiments.02_escape_controls
```

This is the experiment that shows the escape is real. Expected output:

```
condition                   LC4 spk  LPLC2 spk  DNp01 spk   peak DNp01     active
---------------------------------------------------------------------------------
looming                         312        573         37        210 Hz       1042
receding                          0          0          0          0 Hz          0
static                            0          0          0          0 Hz          0
looming, -LC4                   312        573         34        150 Hz        779
looming, -LPLC2                 312        573         28        210 Hz        642
looming, -LC4/-LPLC2            312        573          0          0 Hz        182
```

The last row is the point: with the 104 real LC4 and 210 real LPLC2 neurons
silenced, those neurons still fire (312 and 573 spikes) but send nothing — and
the Giant Fibre goes to **exactly zero**.

Then the other two:

```bash
python -m experiments.01_looming_escape      # time course of one escape
python -m experiments.03_touch_and_feeding   # feeding, touch, and honest refusals
```

And the interactive laboratory. There are two builds, and they run the same
simulation:

**Browser build** (what is deployed). The whole simulation runs client-side in a
Web Worker; the server is only a static file host.

```bash
python -m tools.export_web_connectome
```

```bash
python -m http.server 8001 --directory web
```

**Python server build** (original). Streams telemetry over a WebSocket.

```bash
python -m visualization.server
```

Open <http://127.0.0.1:8000>. The controls are:

| Control | What it does |
|---|---|
| **🪨 THROW ROCK** | Looming object at a chosen azimuth, speed and size → 104 LC4 + 210 LPLC2 |
| **🍎 PLACE FOOD** | Food odour (266 ORNs) + tarsal contact chemosensation (71 GRNs) + proboscis sugar (23 GRNs), together |
| Stimulus list | 15 further modalities, each labelled with how many real neurons it drives; 4 shown as *not modeled* with the reason |
| Lesion buttons | Silence LC4 / LPLC2 / DNp01 / JO-A / JO-B and repeat the experiment |
| **⟲ Replay experiment** | Scrub the whole recorded run frame by frame |
| Run / Pause / Reset | Transport |
| 3D neurons / Spike raster | All 139,255 neurons at true anatomical positions, or the raster |
| Circuit inspector | Real inputs and outputs of any cell type, with a link to its Codex page |
| Provenance | Every component tagged A / B / C / D with citations |

## 6b. Deploying

The Python server is stateful — a background thread holding 139,255 neurons of
simulation state, streaming over a WebSocket. That cannot run on Vercel, which
has no persistent processes, no WebSockets and no cross-request state.

So the deployed build moves the simulation into the browser. `web/` is a static
site: it downloads the real connectome as a binary asset (~12 MB gzipped, and
Vercel serves it Brotli-compressed) and runs the identical LIF model in a Web
Worker. Nothing is simplified for the web — all 139,255 neurons and all
3,732,460 connections are present, and the engine is verified against the Python
one:

```bash
python -m tools.verify_web_engine
```

That runs the same seed and stimulus through both engines with a shared
deterministic PRNG and asserts the spike counts are identical for every neuron.
All three scenarios (deterministic propagation, Poisson looming drive, and
lesioned) must match exactly before deploying.

```bash
npx vercel --prod
```

`vercel.json` serves `web/` as a static site with immutable caching on the data
files. Browser requirements: a modern browser with module Web Workers
(Chrome/Edge 91+, Firefox 114+, Safari 15+), about 200 MB of tab memory, and the
one-time connectome download.

## 7. How to verify the simulation uses real FlyWire neurons and connections

```bash
python -m pytest tests/ -q
```

74 tests. The ones that matter for this question:

| Check | Test |
|---|---|
| Source files are byte-identical to the documented FlyWire release | `test_source_file_checksum` (SHA-256 of all 6 core files) |
| Exactly 139,255 neurons | `test_neuron_count_matches_published_v783` |
| Every root ID carries the FAFB `720575940` segmentation prefix | `test_all_root_ids_have_fafb_segmentation_prefix` |
| Exactly 50,666,648 synapses | `test_synapse_total_matches_published` |
| The published super-class census (77,873 optic, 1,305 descending, …) | `test_super_class_counts_match_v783` |
| The published neurotransmitter census | `test_neurotransmitter_census_matches_v783` |
| Real population sizes (LPLC2 = 210, LC4 = 104, DNp01 = 2) | `test_cell_type_population_sizes` |
| A fabricated root ID is *rejected*, not invented | `test_a_fabricated_root_id_is_rejected` |
| Neuron positions span a real fly brain (814 × 392 × 278 µm) | `test_neuron_positions_span_a_real_fly_brain` |
| Our engine == a literal transcription of the published Brian2 model | `test_matches_literal_reference_implementation` |

And the biological checks — facts discovered in wet labs that fall out of the
loaded wiring diagram, hard-coded nowhere:

| Known biology | Test |
|---|---|
| LC4 is the largest cell-type input to the Giant Fibre; LPLC2 is also top-5 | `test_lc4_and_lplc2_are_the_top_visual_inputs_to_the_giant_fibre` |
| The Giant Fibre also receives Johnston's-organ (JO-A/JO-B) input | `test_giant_fibre_receives_antennal_mechanosensory_input` |
| LPLC2 pools all four T4/T5 directional subtypes | `test_lplc2_receives_t4_t5_motion_input` |
| R1-6 photoreceptors target L1, L2, L3 (the lamina cartridge) | `test_photoreceptors_target_the_lamina_monopolar_cells` |
| Sugar GRNs drive proboscis motor neurons; bitter GRNs do not | `test_sugar_drives_proboscis_motor_neurons`, `test_bitter_does_not_drive_proboscis_motor_neurons` |
| Cutting LC4+LPLC2 abolishes the Giant Fibre response | `test_silencing_lc4_and_lplc2_abolishes_the_giant_fibre_response` |
| An unstimulated brain is completely silent | `test_an_unstimulated_brain_is_silent` |
| The body cannot move without descending activity | `test_body_does_nothing_without_descending_activity` |

**Verify a neuron by hand.** Every neuron in the UI's circuit inspector links to
its Codex page. For example the left Giant Fibre:
<https://codex.flywire.ai/app/cell_details?root_id=720575940622838154> —
compare the cell type, side, and partner list against what the app shows.

## 8. Known scientific limitations

Summarised here; the full treatment is in
[`BIOLOGICAL_ASSUMPTIONS.md`](BIOLOGICAL_ASSUMPTIONS.md).

1. **Brain only — no ventral nerve cord.** The leg and wing motor neurons are
   not in this dataset. The simulation ends at the descending neurons, which are
   genuinely the brain's only output. Proboscis motor neurons are the one
   exception; they are in the brain.
2. **Touch on thorax, abdomen and legs cannot be modelled** for the same reason,
   and is reported as "Not currently modeled" rather than faked.
3. **Histamine is missing from FlyWire's neurotransmitter vocabulary.**
   Photoreceptors are histaminergic and inhibitory; the dataset labels R1-6 as
   excitatory ACh. Driving light through photoreceptors would invert the sign of
   the whole early visual pathway, so it is **disabled**.
4. **No gap junctions.** The Giant Fibre's output onto motor neurons is largely
   electrical and is therefore absent.
5. **No spontaneous activity or inhibitory tone.** With no stimulus the network
   is silent. This makes the Giant Fibre fire at smaller angular sizes than a
   real fly would tolerate.
6. **Point neurons.** No dendritic computation, no plasticity, no adaptation, no
   neuromodulatory state (hunger, arousal, circadian phase).
7. **Neurotransmitters are predicted, not measured**, and 1,626 neurons (1.2%)
   have no resolvable sign and produce no output.
8. **Sensory transduction is modelled, not simulated.** The tuning curve shapes
   and their constants are ours; the neurons they drive are real. This is the
   honest weak point of the pipeline.
9. **Descending-neuron → behaviour assignments come from the literature**, not
   from the connectome, and cover only 8 of ~473 descending cell types.
10. **This is one fly.** A single adult female, imaged once.
11. **The body is kinematic**, not biomechanical.

**This simulation does not reproduce the subjective experience, awareness, or
consciousness of a fruit fly, and no result here should be described that way.**

---

## Project layout

```
fruit-fly-lab/
├── config.py                    dataset paths + identity; refuses to run on substituted data
├── data/
│   ├── derived/                 built connectome (CSR sparse) + neuron index
│   └── metadata/                SHA-256 checksums, build manifest
├── brain/
│   ├── connectivity/            build_connectome.py — real data -> sparse graph
│   ├── neurons/                 registry.py (queries), labels.py (FlyWire labels)
│   ├── neuron_models/           lif.py — the published LIF parameters
│   ├── sensory/                 retinotopy.py, encoders.py, modalities.py
│   └── motor/                   descending.py — DN readout + published behaviours
├── simulation/
│   ├── engine/                  lif_engine.py (the simulator), session.py (closed loop)
│   ├── stimuli/                 looming.py (exact geometry), pulse.py
│   └── outputs/                 experiment results (JSON)
├── fly/body/                    fly_body.py — kinematic body, downstream of all neurons
├── visualization/               server.py + static/ — Python server build
├── web/                         static browser build (this is what deploys)
│   ├── data/                    connectome + neurons as binary assets
│   └── js/                      engine.js (LIF port), sim.js, worker.js, app.js
├── tools/                       web export, shared PRNG, cross-engine verifier
├── experiments/                 01 looming escape, 02 controls, 03 touch & feeding
├── tests/                       74 tests
├── DATA_SOURCES.md
└── BIOLOGICAL_ASSUMPTIONS.md
```

## Citing

If you publish anything based on this, cite the data and the model, not this
code:

- Dorkenwald, S. et al. (2024) *Nature* **634**, 124–138. doi:10.1038/s41586-024-07558-y
- Schlegel, P. et al. (2024) *Nature* **634**, 139–152. doi:10.1038/s41586-024-07686-5
- Shiu, P.K. et al. (2024) *Nature* **634**, 210–219. doi:10.1038/s41586-024-07763-9

FlyWire data are CC BY-NC-SA 4.0.
