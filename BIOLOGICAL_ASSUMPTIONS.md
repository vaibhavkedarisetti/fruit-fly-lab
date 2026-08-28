# BIOLOGICAL_ASSUMPTIONS.md

What in this project is real, what is a published model, what is an
approximation, and what is just our code.

Read this before drawing any biological conclusion from the simulation.

---

## Provenance categories

Every component in the codebase is tagged with one of these, in its module
docstring and in the UI's provenance panel.

| Tag | Meaning |
|---|---|
| **A — Real data** | Comes from FlyWire FAFB v783 unmodified. Neuron identities, cell types, connectivity, synapse counts, neurotransmitter predictions, 3D positions, column assignments, community labels. |
| **B — Published model** | An assumption or measurement taken from a peer-reviewed paper, cited in place. The LIF equations and constants, the excitatory/inhibitory sign convention, sensory tuning properties, descending-neuron → behaviour assignments. |
| **C — Our approximation** | A modelling choice we made because the data does not determine it. Always documented, and where it matters, counted. |
| **D — Our engineering** | Code with no biological content: data loading, the simulation loop, the web server, the drawing routines. |

---

## 1. What the simulation actually is

```
looming object (exact geometry)                      A/D
  → LC4 + LPLC2 firing rates                          B + C  ← the only modelled step
  → Poisson drive onto 314 real FlyWire neurons       B
  → 139,255-neuron LIF network,                       A + B
    3,732,460 real connections, 50,666,648 synapses
  → real descending neurons (DNp01 …)                 A
  → motor channel activation                          B + C
  → kinematic body                                    C/D
```

The middle of that chain — by far the largest part — is real connectome and
published model. The two ends are where we had to model something.

**There is no rule anywhere that says "if looming then escape."** The proof is
in `experiments/02_escape_controls.py`: silencing the 104 real LC4 and 210 real
LPLC2 neurons drops the Giant Fibre from 37 spikes to **exactly zero** while the
stimulus is unchanged and LC4/LPLC2 themselves keep firing. The response travels
through the connectome or it does not happen at all.

---

## 2. The neuron model (category B)

Verbatim from Shiu et al. 2024 (`brain/neuron_models/lif.py`):

```
dv/dt = (v_0 - v + g) / t_mbr     (unless refractory)
dg/dt = -g / tau                  (unless refractory)
spike when v > v_th ; then v ← v_rst, g ← 0
presynaptic spike, after t_dly:  g ← g + w
w = sign(presynaptic neurotransmitter) × synapse_count × w_syn
```

with v_0 = v_rst = −52 mV, v_th = −45 mV, t_mbr = 20 ms, tau = 5 ms,
t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV.

**What this model does not have:**

- **No dendrites or compartments.** Every neuron is a point. A real *Drosophila*
  neuron performs substantial computation in its neurites; a Kenyon cell, an
  LPLC2 and a Giant Fibre are treated as electrically identical apart from their
  connections.
- **No synaptic plasticity, facilitation, depression, or adaptation.** Firing
  rates here do not decline with sustained input the way real sensory neurons do.
- **No neuromodulation.** Dopamine, serotonin and octopamine are treated as fast
  excitatory transmitters. In the real animal they act on slow metabotropic
  receptors and change the state of circuits rather than driving spikes. This
  affects 1,677 neurons (584 DA, 1,021 5-HT, 72 OA).
- **No gap junctions.** The dataset contains chemical synapses only. This
  matters acutely for escape: the Giant Fibre's output onto the tergotrochanteral
  motor neuron and the peripherally synapsing interneuron is largely
  **electrical**, and is therefore absent from the model.
- **No spontaneous activity.** With no stimulus, the network is completely
  silent (`tests/test_circuits.py::test_an_unstimulated_brain_is_silent`). Real
  brains have ongoing background firing and a balance of excitation and
  inhibition that this model lacks. A consequence is visible in the escape
  experiment: the Giant Fibre fires occasional spikes at small angular sizes
  where a real fly, sitting in a tonically inhibited state, would not.
- **A single global synaptic weight.** `w_syn = 0.275 mV` is one free parameter
  fitted across the whole brain. Real synaptic strengths vary by orders of
  magnitude between cell types.

---

## 3. Excitatory / inhibitory assignment (category B, with a C fallback)

Following Shiu et al.: **ACh, dopamine, octopamine, serotonin → excitatory;
GABA, glutamate → inhibitory.** Glutamate is inhibitory in *Drosophila* via the
GluClα chloride channel, which is the opposite of the vertebrate convention.

Neurotransmitters are **predicted from electron-microscopy images** (Eckstein et
al. 2024), not measured. They are wrong for some neurons. Two consequences we
observed directly:

- **Histamine is not in FlyWire's vocabulary.** The classifier predicts exactly
  six transmitters (ACh, GABA, Glu, DA, 5-HT, OA). Photoreceptors are
  histaminergic and their synapse onto lamina monopolar cells is **inhibitory
  and sign-inverting**. In this dataset R1-6 are labelled ACh, i.e. excitatory.
  **Driving photoreceptors would invert the sign of the entire early visual
  pathway, so we disabled it** rather than produce a plausible-looking wrong
  answer. The UI shows "Not currently modeled" with this reason.
- **DNp01 is annotated inconsistently.** The left Giant Fibre is predicted ACh,
  the right one GLUT. The Giant Fibre is cholinergic. One of these is a
  prediction error, which means the right Giant Fibre's chemical output has the
  wrong sign in the model. Its *inputs* — which is what our escape experiment
  measures — are unaffected.

**Our fallback (C):** 19,658 of 139,255 neurons (14.1%) have no neuron-level
neurotransmitter prediction. For those we take the synapse-count-weighted
majority transmitter across the neuron's own outgoing connections. This resolves
18,032 of them. The remaining **1,626 neurons (1.2%) have no sign and therefore
produce no output** in the simulation. All three counts are recorded in
`data/metadata/build_manifest.json` and shown in the UI provenance panel.

---

## 4. Connectivity (category A, with one C choice)

- We use `connections_princeton.csv.gz`, the **≥5-synapse thresholded** table.
  That is Codex's default and matches the reference model. Weaker connections
  are dominated by synapse-detection false positives. The unthresholded table is
  present on disk and deliberately unused.
- **We sum synapses across neuropils (C).** A neuron pair connected in two
  neuropils becomes one edge with the combined count. The model is a point-neuron
  model, so it cannot use the spatial separation anyway.
- **Proofreading is not uniform.** FlyWire's central brain is proofread to a high
  standard; parts of the optic lobes are less complete. Weakly connected or
  poorly reconstructed neurons will be under-represented.
- **This is one fly.** A single adult female. Individual variability, sexual
  dimorphism, and developmental variation are not represented. FAFB was also
  fixed and imaged, so this is a snapshot of one animal's wiring, not a species
  average.

---

## 5. Sensory encoding (category B + C) — the honest weak point

The connectome is a wiring diagram. It contains no phototransduction, no
odorant-receptor binding, no mechanotransduction. **Something has to turn a
physical stimulus into spikes, and that something is not the connectome.**

Shiu et al. handled this by driving chosen neurons with Poisson input at a fixed
rate — effectively simulating optogenetic activation. We do the same, and for
looming we additionally modulate the rate by published tuning and a receptive
field.

### Looming (`brain/sensory/encoders.py`)

| Element | Category | Notes |
|---|---|---|
| Which neurons (104 LC4, 210 LPLC2) | **A** | Real FlyWire cell types |
| Receptive-field centres and radii | **A→C** | Computed as the synapse-weighted mean visual direction of each cell's *column-assigned presynaptic partners*. The column assignments are real data; treating their centroid as a receptive-field centre is our inference. Mean radius came out at 14–15°, consistent with published LC/LPLC receptive fields, and lateralisation with a frontal binocular overlap fell out correctly — but these are **not measured receptive fields**. |
| Hex-lattice → visual angle mapping | **A→C** | Axis orientation was *measured* against real 3D anatomy (u = p + q/2 tracks the dorsoventral axis, R² = 0.98; the orthogonal axis tracks the anteroposterior axis, partial r = 0.95). The scaling onto a 175° × 160° field of view is a linear approximation; the real interommatidial angle varies from ~4.5° frontally to ~8° laterally. |
| LC4 ∝ angular velocity, LPLC2 ∝ angular size | **B** | von Reyn et al. 2017 |
| LPLC2 gated by outward motion | **B** | Klapoetke et al. 2017 |
| Saturating (Naka-Rushton) tuning shape and its half-maximum constants | **C** | The *shape* is a standard choice; the constants (300°/s for LC4, 25° for LPLC2) are **fitted by us to reproduce published qualitative behaviour, not measured**. Changing them changes when the escape triggers. |

### All other modalities (`brain/sensory/modalities.py`)

Intensity maps **linearly** to firing rate up to 150 Hz, with no receptor
adaptation and no spatial structure. Which neurons each stimulus drives is real
data plus a cited experimental identification; how hard it drives them is a
placeholder.

---

## 6. Motor output (category A + B + C)

**The simulation ends at the descending neurons, and that is a hard limit of the
dataset, not a shortcut.** FlyWire FAFB is a brain connectome. The motor neurons
that move legs and wings are in the ventral nerve cord. The 110 neurons FlyWire
labels `motor` innervate head structures. This is asserted by
`tests/test_circuits.py::test_no_leg_or_wing_motor_neurons_in_this_brain_dataset`.

One genuine exception: **proboscis motor neurons are in the brain**, so the
sugar-feeding experiment reaches a real motor neuron.

The descending-neuron → behaviour table in `brain/motor/descending.py` is
**category B**: it comes from optogenetic activation and silencing experiments,
each entry carrying its citation. It is not derivable from the connectome. Only
8 descending cell types have assignments; **the other ~465 descending types in
the dataset have none, and the UI reports them as "Not currently modeled."**

Mapping a firing rate to a 0–1 "command strength" with a half-maximum of 60 Hz,
and the thresholds at which the body acts, are **category C** — our choices.

---

## 7. What is not modelled at all

| Not modelled | Why |
|---|---|
| Touch on thorax, abdomen, legs | Those mechanosensory neurons project to the ventral nerve cord and are absent from FAFB v783. |
| Light / vision from photoreceptors | Histamine is missing from the neurotransmitter vocabulary, so the photoreceptor→lamina sign would be inverted (§3). |
| Leg and wing motor neurons, muscles | Ventral nerve cord (§6). |
| Gap junctions, including the Giant Fibre's output synapses | Not in the dataset (§2). |
| Neuromodulatory state, hunger, arousal, circadian phase | No model of internal state. A real fly's response to food depends heavily on satiety. |
| Learning and memory | No plasticity. The mushroom body is present and wired, but cannot learn here. |
| Flight aerodynamics, leg biomechanics | The body is kinematic (§8). |
| The ocelli, and most of the ~465 unassigned descending neuron types | No behavioural assignment we would stand behind. |

---

## 8. The body (category C/D)

`fly/body/fly_body.py` is **not** part of the neural simulation. It takes motor
channel activations and nothing else — it never sees the stimulus. If the neural
simulation produces no descending activity, the body does nothing
(`tests/test_sensory_and_body.py::test_body_does_nothing_without_descending_activity`).

Published kinematics are used where they exist: short-mode (Giant-Fibre) takeoff
~5 ms after the command with no preparatory wing raising, long-mode takeoff after
~200 ms of wing raising (Card & Dickinson 2008; von Reyn et al. 2014). Everything
else — walking speed, turn rate, the ballistic jump — is a plausible kinematic
approximation, not a biomechanical model.

---

## 9. Known quantitative discrepancies

Places where the simulation visibly departs from the real animal:

1. **Giant Fibre firing rate.** The real Giant Fibre is essentially all-or-none:
   it fires one or two spikes and triggers takeoff. Our model produces graded
   rates up to ~250 Hz. We read the escape command off a rate threshold, which
   is a reinterpretation, not a reproduction.
2. **Giant Fibre activity at small angular sizes.** Occasional Giant Fibre spikes
   occur at θ ≈ 3–10°, where a real fly would not escape. This follows from the
   absence of background inhibitory tone (§2).
3. **No escape-probability behaviour.** Real flies escape probabilistically and
   choose between short and long mode depending on stimulus dynamics
   (von Reyn et al. 2017). Our model is deterministic given a seed.
4. **Sensory populations are driven synchronously.** Every cell in a modality
   gets independent Poisson input at the same rate; real populations have
   heterogeneous thresholds and correlated noise.

---

## 10. What this simulation is not

It is a **circuit-level simulation of one connectome under one published neuron
model**. It reproduces some experimentally established input–output
relationships — LC4/LPLC2 → Giant Fibre, sugar → proboscis extension, bitter →
no proboscis extension — and it lets you cut those pathways and watch the
behaviour disappear.

It does **not** reproduce, and makes no claim to reproduce, the subjective
experience, awareness, consciousness, or inner life of a fruit fly. It is a
network of differential equations wired according to a photograph of one
animal's synapses. Nothing here bears on what it is like to be a fly, and
nothing here should be described that way.
