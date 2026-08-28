# DATA_SOURCES.md

Every dataset and model this project uses. Nothing else is loaded, and nothing
is ever silently substituted: `config.require_source_files()` refuses to run if
a source file is missing, and `tests/test_data_provenance.py` verifies the
SHA-256 of every file the pipeline reads.

---

## 1. Primary neural dataset

| field | value |
|---|---|
| **Dataset** | FlyWire FAFB — adult female *Drosophila melanogaster* whole-brain connectome |
| **Version** | `783` (public release) |
| **Specimen** | FAFB (Full Adult Fly Brain), a single adult female, serial-section TEM |
| **Download portal** | <https://codex.flywire.ai/api/download> |
| **Project site** | <https://flywire.ai> · <https://codex.flywire.ai> |
| **Neurons** | **139,255** (verified — matches the published count exactly) |
| **Connections** | 5,342,446 (pre, post, neuropil) rows → **3,732,460** unique neuron pairs |
| **Synapses** | **50,666,648** |
| **Local path** | `D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)` |
| **Date downloaded** | 2026-08-25 |
| **Date verified in this project** | 2026-08-28 |

### Papers to cite for the data

| Contribution | Citation | DOI |
|---|---|---|
| Connectome and annotations | Dorkenwald, S. *et al.* (2024). Neuronal wiring diagram of an adult brain. *Nature* **634**, 124–138. | [10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) |
| Cell-type annotation | Schlegel, P. *et al.* (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* **634**, 139–152. | [10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5) |
| Original FAFB EM volume | Zheng, Z. *et al.* (2018). A complete electron microscopy volume of the brain of adult *Drosophila melanogaster*. *Cell* **174**, 730–743. | [10.1016/j.cell.2018.06.019](https://doi.org/10.1016/j.cell.2018.06.019) |
| Synapse prediction (Buhmann) | Buhmann, J. *et al.* (2021). Automatic detection of synaptic partners in a whole-brain *Drosophila* EM dataset. *Nat Methods* **18**, 771–774. | [10.1038/s41592-021-01183-7](https://doi.org/10.1038/s41592-021-01183-7) |
| Neurotransmitter prediction | Eckstein, N. *et al.* (2024). Neurotransmitter classification from electron microscopy images. *Cell* **187**, 2574–2594. | [10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016) |
| Optic-lobe columns | Matsliah, A. *et al.* (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* **634**, 166–180. | [10.1038/s41586-024-07981-1](https://doi.org/10.1038/s41586-024-07981-1) |

### Files used by the pipeline (SHA-256 verified)

Checksums are stored in `data/metadata/flywire_v783_checksums.txt` and asserted
by `tests/test_data_provenance.py::test_source_file_checksum`.

| File | Bytes | SHA-256 | Used for |
|---|---:|---|---|
| `neurons.csv.gz` | 1,679,884 | `6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719` | root IDs, neurotransmitter predictions |
| `classification.csv.gz` | 934,402 | `e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663` | super-class, class, side, nerve |
| `consolidated_cell_types.csv.gz` | 901,707 | `8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d` | cell types (LC4, LPLC2, DNp01 …) |
| `connections_princeton.csv.gz` | 68,456,801 | `445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b` | **the connectivity used by the simulation** |
| `coordinates.csv.gz` | 5,314,546 | `14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05` | 3D neuron positions |
| `column_assignment.csv.gz` | 462,838 | `bdf4ce7f62cc63493d53eefad3816ff2dfd08b190e97b35a492e0e453df2f0f6` | retinotopic map / receptive fields |
| `labels.csv.gz` | 4,771,292 | `bdd4eafab2bfe30540256c84ea1513e4b1877c0c4cf03f919204b4eafae5868e` | functional labels (sugar GRN, proboscis MN) |
| `visual_neuron_types.csv.gz` | 631,701 | `4bcc6a2f98b86e6c3fb7eaddb49736f3d81ab65bda35da8f740641201a1e379f` | visual-system typing |
| `cell_stats.csv.gz` | 2,526,548 | `bd5879e1b5df964bea2f3ca5316348d4276ce2ccaac283f0e36583c04fbd3d8e` | morphometrics (reference only) |
| `names.csv.gz` | 1,181,576 | `e541ef9ef4b9e62d798f165ae76853d15b84174cf1dce95392f313a491055332` | display names (reference only) |
| `connectivity_tags.csv.gz` | 637,719 | `68c69cec13810fa543c600a5b9973d10718b449740e010834662be1dc1b8696c` | reference only |
| `neuropil_synapse_table.csv.gz` | 4,674,663 | `e525bdea7bc2fe585cf8ab8f9fc76bea5e29f2c3e5240588b6add518bcda5ab1` | reference only |
| `processed_labels.csv.gz` | 1,017,658 | `9feab030bd7f0f9f9909a56f4ed37f019fc19cc076180b494f3bff55bf1480d2` | reference only |
| `synapse_attachment_rates.csv.gz` | 3,257 | `2e412c5714c21aa4f6ebff74801f620d161594740aaee25c73f5bc18b0110447` | reference only |

### Files present but NOT used

| File | Bytes | Why not used |
|---|---:|---|
| `connections_princeton_no_threshold.csv.gz` | 275,679,780 | Unthresholded connectivity. Codex's default (and the reference model's) is the ≥5-synapse thresholded table; using the unthresholded one would add millions of single-synapse edges that are mostly detection noise. |
| `connections_buhmann_no_threshold.csv.gz` | 212,093,967 | Alternative synapse-detection pipeline. Mixing detectors is not valid. |
| `fafb_v783_princeton_synapse_table.csv.gz` | 2,695,106,039 | Per-synapse table. Not needed: the model uses synapse *counts* per connection. |
| `synapse_coordinates.csv.gz` | 316,819,225 | Individual synapse coordinates. Not used by a point-neuron model. |
| `sk_lod1_783_healed.zip` | 13,873,645,070 | Neuron skeleton meshes. The 3D view uses single anchor coordinates instead. Not checksummed (13.9 GB, unused). |

**Important detail discovered and verified during the build:** the `position`
column of `coordinates.csv.gz` is in **nanometres**, not in 4 × 4 × 40 nm voxels.
Read as nm, the neuron cloud spans 814 × 392 × 278 µm, which is a real adult
*Drosophila* brain. Read as voxels it would span 3604 × 1772 × 11165 µm. This is
asserted by `tests/test_data_provenance.py::test_neuron_positions_span_a_real_fly_brain`.

---

## 2. Computational model

| field | value |
|---|---|
| **Model** | Whole-brain leaky integrate-and-fire (LIF) |
| **Paper** | Shiu, P.K., Sterne, G.R., Spiller, N. *et al.* (2024). A leaky integrate-and-fire computational model based on the connectome of the entire adult *Drosophila* brain reveals insights into sensorimotor processing. *Nature* **634**, 210–219. |
| **DOI** | [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9) |
| **Preprint** | [10.1101/2023.05.02.539144](https://doi.org/10.1101/2023.05.02.539144) |
| **Reference code** | <https://github.com/philshiu/Drosophila_brain_model> (`model.py`) |
| **Date consulted** | 2026-08-28 |

All parameters in `brain/neuron_models/lif.py` are transcribed verbatim from
that `model.py`. `tests/test_lif_engine.py::test_matches_literal_reference_implementation`
proves our optimised engine produces spike-for-spike identical output to a
literal transcription of the reference Brian2 network.

Parameter sources cited inside the reference implementation:

| Constant | Value | Source |
|---|---|---|
| resting / reset potential | −52 mV | Kakaria & de Bivort 2017, [10.3389/fnbeh.2017.00008](https://doi.org/10.3389/fnbeh.2017.00008) |
| spike threshold | −45 mV | as above |
| membrane time constant | 20 ms | as above |
| synaptic time constant | 5 ms | Jürgensen et al. 2021, [10.1088/2634-4386/ac3ba6](https://doi.org/10.1088/2634-4386/ac3ba6) |
| refractory period | 2.2 ms | Lazar et al. 2021, [10.7554/eLife.62362](https://doi.org/10.7554/eLife.62362) |
| synaptic delay | 1.8 ms | Paul et al. 2015, [10.3389/fncel.2015.00029](https://doi.org/10.3389/fncel.2015.00029) |
| weight per synapse | 0.275 mV | free parameter fitted in Shiu et al. 2024 |

---

## 3. Reference implementations inspected

| Repository | What it is | What we took | Assessment |
|---|---|---|---|
| [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | The authors' own Brian2 code for Shiu et al. 2024 | **All LIF equations and constants**, the Poisson activation scheme, and the silencing semantics | Authoritative. Uses FlyWire v630 and v783. Its `Connectivity_783.parquet` is a pre-processed derivative; we build our own equivalent directly from the Codex release so the provenance chain is explicit. |
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) | Multi-backend re-implementation (Brian2, Brian2CUDA, PyTorch, NEST GPU, GeNN) benchmarking the same model | Confirmation of the v783 approach and backend design ideas | Real connectome, not synthetic. Oriented toward *benchmarking*, not interactive simulation, so it was not a usable base for a closed sensory→body loop. |
| [snedea/flybrain](https://github.com/snedea/flybrain), [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain) | Listed in the brief | Nothing | Not used. See the honesty note below. |

**Honesty note as requested.** We did not adopt any third-party repository as
the base. The two forks above were not needed once the authors' own reference
implementation was available, and building the connectome directly from the
Codex release is what lets us checksum every input and assert the published
139,255 / 50,666,648 figures in tests. Where a repository ships a
*pre-processed* connectivity file (as `philshiu` and `eonsystemspbc` both do),
that file is a derivative whose provenance you cannot verify from the file
itself; our `brain/connectivity/build_connectome.py` regenerates the equivalent
from checksummed primary files and records every transformation in
`data/metadata/build_manifest.json`.

---

## 4. Derived artefacts built by this project

Produced by `python -m brain.connectivity.build_connectome`.

| File | Contents |
|---|---|
| `data/derived/connectome_v783.npz` | Sparse CSR matrix of signed synapse counts (139,255 × 139,255, 3,732,460 non-zeros) plus the root-ID vector |
| `data/derived/neuron_index_v783.csv.gz` | One row per real neuron: root ID, cell type, class, side, neurotransmitter, resolved sign, primary neuropil, 3D position |
| `data/metadata/build_manifest.json` | Full build record: counts, sign convention, source directory, timestamp |
| `data/metadata/flywire_v783_checksums.txt` | SHA-256 of every source file |

Build manifest for the current build:

```
neurons                139,255
neuron pairs         3,732,460
synapses            50,666,648
excitatory neurons      96,514
inhibitory neurons      41,115
unknown sign             1,626
```

---

## 5. Dataset present but not yet used

| field | value |
|---|---|
| **Dataset** | BANC — Brain And Nerve Cord connectome |
| **Version** | `888` |
| **Local path** | `D:\Fruitfly\Brain and Nerve Cord (BANC v888)` |
| **Files** | `neurons.csv.gz` (2,881,728 B), `connections_princeton.csv.gz` (29,164,674 B) |
| **Date downloaded** | 2026-08-25 |
| **Status** | **Not loaded by any code in this project.** |

BANC includes the ventral nerve cord, and therefore the leg and wing motor
neurons that FAFB lacks. It is the natural way to extend the pipeline past the
descending neurons to real motor neurons. It is deliberately not wired in yet:
the brief specifies FlyWire FAFB v783 as the primary dataset, and mixing two
connectomes silently would be exactly the kind of substitution this document
exists to prevent. See `BIOLOGICAL_ASSUMPTIONS.md` §7.

---

## 6. Experimental literature used for stimulus and behaviour assignments

These papers do not supply data files. They justify *which real neurons* a
stimulus drives and *what behaviour* a descending neuron commands. Each is also
cited in the code (`brain/sensory/modalities.py`, `brain/motor/descending.py`)
and shown in the UI's provenance panel.

| Claim | Citation | DOI |
|---|---|---|
| LC4 encodes angular velocity, LPLC2 angular size; both drive the Giant Fibre | von Reyn et al. 2017, *Nat Neurosci* 20:1176 | [10.1038/nn.4600](https://doi.org/10.1038/nn.4600) |
| LPLC2 is selective for outward motion | Klapoetke et al. 2017, *Nature* 551:237 | [10.1038/nature24626](https://doi.org/10.1038/nature24626) |
| Giant Fibre escape: short-mode takeoff, ~5 ms | von Reyn et al. 2014, *Nat Neurosci* 17:962 | [10.1038/nn.3741](https://doi.org/10.1038/nn.3741) |
| Short vs long-mode escape kinematics | Card & Dickinson 2008, *J Exp Biol* 211:341 | [10.1242/jeb.012682](https://doi.org/10.1242/jeb.012682) |
| Johnston's organ subgroups: sound vs wind/gravity | Kamikouchi et al. 2009, *Nature* 458:165 | [10.1038/nature07843](https://doi.org/10.1038/nature07843) |
| Wind detection by Johnston's organ | Yorozu et al. 2009, *Nature* 458:201 | [10.1038/nature07843](https://doi.org/10.1038/nature07843) |
| Sugar and bitter GRN identification in FlyWire | Engert et al. 2022, *eLife* 11:e78110 | [10.7554/eLife.78110](https://doi.org/10.7554/eLife.78110) |
| Vinegar attraction glomeruli (DM1, VA2) | Semmelhack & Wang 2009, *Nature* 459:218 | [10.1038/nature07983](https://doi.org/10.1038/nature07983) |
| Geosmin avoidance via DA2 | Stensmyr et al. 2012, *Cell* 151:1345 | [10.1016/j.cell.2012.09.046](https://doi.org/10.1016/j.cell.2012.09.046) |
| cVA pheromone via DA1 | Kurtovic et al. 2007, *Nature* 446:542 | [10.1038/nature05672](https://doi.org/10.1038/nature05672) |
| CO2 avoidance | Suh et al. 2004, *Nature* 431:854 | [10.1038/nature02980](https://doi.org/10.1038/nature02980) |
| Cold/hot thermoreceptors → VP2 / VP3 | Gallio et al. 2011, *Cell* 144:614; Frank et al. 2015, *Cell Rep* 11:1345 | [10.1016/j.cell.2011.01.028](https://doi.org/10.1016/j.cell.2011.01.028) |
| Hygrosensory neurons | Enjin et al. 2016, *Curr Biol* 26:1352 | [10.1016/j.cub.2016.03.049](https://doi.org/10.1016/j.cub.2016.03.049) |
| Descending neuron anatomy and behaviour | Namiki et al. 2018, *eLife* 7:e34272 | [10.7554/eLife.34272](https://doi.org/10.7554/eLife.34272) |
| MDN drives backward walking | Bidaye et al. 2014, *Science* 344:97 | [10.1126/science.1249964](https://doi.org/10.1126/science.1249964) |
| DNp09 drives freezing | Zacarias et al. 2018, *Nat Commun* 9:3697 | [10.1038/s41467-018-05875-1](https://doi.org/10.1038/s41467-018-05875-1) |
| DNa01/DNa02 steering | Rayshubskiy et al. 2024, *Nature* 631:135 | [10.1038/s41586-024-07523-9](https://doi.org/10.1038/s41586-024-07523-9) |
| Head grooming driven by head bristles | Hampel et al. 2015, *eLife* 4:e08758 | [10.7554/eLife.08758](https://doi.org/10.7554/eLife.08758) |
| Tarsal gustatory neurons | Ledue et al. 2015, *Curr Biol* 25:1466 | [10.1016/j.cub.2015.03.020](https://doi.org/10.1016/j.cub.2015.03.020) |

---

## 7. Licensing

FlyWire data are released under **CC BY-NC-SA 4.0** and are free for
non-commercial use with attribution. See <https://codex.flywire.ai/about_flywire>
and the citation guidance at <https://flywire.ai/citing>. If you publish
anything derived from this project, cite Dorkenwald et al. 2024 and
Schlegel et al. 2024 for the data and Shiu et al. 2024 for the model.
