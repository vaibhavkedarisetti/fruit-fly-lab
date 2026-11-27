# Whole-Brain Network Stability Bugfix Design

## Overview

The whole-brain LIF simulation over FlyWire FAFB v783 (139,255 neurons, 3,732,460 connections,
50,666,648 synapses) ignites into a stimulus-independent attractor: a single connected ORN driven
at 20 Hz converts ~40 injected spikes into hundreds of thousands of brain spikes, and once the
plateau is reached a 150x change in drive moves whole-brain rate by 16%. No sensory information
survives, so every neural measurement taken in that state is uninterpretable.

This design does **not** choose a fix. It does five things:

1. **Fixes the stability criterion** (Section: Correctness Properties, criterion `S`) so that
   "fixed" becomes a decidable statement, with constants chosen so that both endpoints of
   Evidence E fail — the saturated state at `w_syn=0.275` and the silent state at `w_syn=0.0344`.
2. **Enumerates nine hypotheses** H1–H9 covering the root-cause fork of requirement 2.10, each
   with the cheapest probe that can falsify it and the verdict field that probe writes.
3. **Grounds every experiment in a measured cost model** so the programme fits the user's
   hardware. All timings below were measured in this session on the actual loaded connectome,
   not estimated.
4. **Specifies tractable estimators** for the two observables that have no affordable exact
   form: the per-spike branching ratio (2.15) and the per-step refractory invariant (2.12).
5. **Lays out six candidate fix paths** F1–F6 plus the terminal no-fix outcome F7, each with a
   provenance category, gated on which hypothesis survives. Uniform gain reduction is recorded
   as **already falsified as a sufficient fix** by Evidence E and may never be applied alone.

The engine is pure NumPy on the CPU. There is no CUDA path anywhere in `simulation/engine/`, so
the RTX 3050's 4 GB of VRAM is irrelevant to every budget in this document. All costs are
single-threaded CPU wall-clock; parallelism is available only by running independent conditions
as separate OS processes (6 physical cores measured).

---

## Glossary

- **Bug_Condition (C)** — a whole-brain drive configuration `X` (driven indices, per-neuron
  Poisson rate, duration, seed) on the FULL v783 connectome with `duration_ms >= 500` and at
  least one anatomically connected driven neuron. Defined formally in the requirements and
  restated below.
- **Property (P)** — the network's activity must carry stimulus magnitude: criterion `S` below.
- **Preservation** — every subgraph regime the 86 existing tests exercise (50 to ~4,000 neurons,
  short windows, the 120-neuron literal-reference comparison) must remain bit-identical.
- **F / F'** — the engine before / after the fix.
- **sigma (branching ratio)** — expected number of postsynaptic spikes causally produced by one
  presynaptic spike. Three distinct estimators are defined below: `sigma_rest`, `sigma_drive`,
  `sigma_transient`.
- **Generation** — one synaptic delay `t_dly = 1.8 ms = 18 steps`. The natural time unit for
  branching, **not** one integration step. A spike at step `t` can only influence spiking at
  step `t + 18`, so any consecutive-*step* ratio is meaningless.
- **Ignition** — self-sustaining activity that persists after drive removal.
- **Plateau** — the flat activity level reached after ignition; whole-brain rate 4.06–4.13 Hz
  measured this session, consistent with Evidence C/E's 4.09–4.25 Hz.
- **Refractory ceiling** — `1000 / t_rfc = 1000 / 2.2 = 454.5 Hz`, the absolute maximum rate of a
  non-Poisson neuron.
- **`LIFEngine.step`** — the hot loop in `simulation/engine/lif_engine.py`; nine numbered stages
  from delayed-input delivery to clock advance.
- **`_n_refractory`** — the engine's cached count of refractory neurons; gates the
  `if self._n_refractory:` branch that performs the refractory hold.
- **R1** — the in-repo literal reference `_literal_reference()` in `tests/test_lif_engine.py`.
- **R2** — a new independent sparse float64 reference specified in this design (see H1/R-TASK-2).
- **Provenance categories** — A: real FlyWire data. B: published Shiu et al. 2024 model.
  C: our engineering.

---

## Bug Details

### Bug Condition

The bug manifests whenever sustained Poisson drive is applied to any connected neuron of the full
v783 connectome for at least 500 ms. Whole-brain activity ignites to a fixed plateau whose level,
active fraction and per-hop profile are all essentially independent of the drive that caused it.
Which mechanism produces the ignition is undecided: it is either (i) a deviation in our engine
from the published model, or (ii) a genuine instability of the published parameterization under a
regime Shiu et al. never exercised. Requirement 2.10 forbids fixing anything before that fork is
decided by experiment.

**Formal Specification:**

```
FUNCTION isBugCondition(X)
  INPUT:  X of type WholeBrainDrive { driven_indices, rate_hz, duration_ms, seed }
  OUTPUT: boolean

  RETURN X.network      = FULL_CONNECTOME_V783
     AND X.duration_ms >= 500
     AND X.driven_indices IS NON-EMPTY
     AND EXISTS n IN X.driven_indices WHERE outDegree(n) > 0
END FUNCTION
```

`outDegree(n) > 0` is checked against the loaded connectome, which excludes the anatomically
isolated R[0] ORN (index 164). That exclusion is a real connectome fact (requirement 3.8), not an
exception to the bug.

### Examples

Measured this session against the loaded connectome, single ORN index 1574 (`odor_vinegar` left)
driven at 20 Hz for 2,000 ms, working tree, `dtype=float32`:

| seed | total brain spikes | active neurons | ignition onset | wall time |
|------|-------------------|----------------|----------------|-----------|
| 0    | 888,253           | 9,094          | 250–500 ms bin | 29.4 s    |
| 1    | 457,635           | 8,934          | 1000–1250 ms   | 17.0 s    |
| 2    | 871,195           | 9,056          | 500–750 ms bin | 28.4 s    |

- **Expected:** ~40 injected spikes into one ORN produce a bounded, drive-proportional response.
  **Actual:** 457,635–888,253 brain spikes, an amplification of roughly 11,400–22,200 spikes out
  per spike in.
- **Expected:** cumulative spike count is a stable observable across seeds.
  **Actual:** it varies 1.94x across seeds 0–2 while the active count varies only 1.8%
  (8,934–9,094) and the plateau rate is stable at 4.06–4.13 Hz. **Consequence for this design:**
  the recorded Evidence A figure of 579,293 spikes is one seed draw of a high-variance quantity;
  cumulative amplification requires >= 5 seeds and must be reported with its spread, whereas
  plateau rate and active fraction are low-variance and need few seeds. Ignition *latency* is the
  variance source — at 20 Hz only ~40 Poisson events land in 2 s, so when the first few land
  dominates the total.
- **Expected:** drive magnitude is recoverable from activity.
  **Actual:** 258 ORNs swept over 1.0/5.0/20.3/150.0 Hz give 4.09/4.15/4.25/4.76 Hz, a log-log
  gain of `ln(4.76/4.09)/ln(150) = 0.030` (Evidence C).
- **Expected:** relays have headroom to represent a stronger stimulus.
  **Actual:** hop1 sits at 230.09 Hz, `0.506` of the 454.5 Hz refractory ceiling (Evidence B).
- **Edge case (not the bug):** R[0] index 164 has zero out-degree and never propagates.
  A single-ORN probe seeded there correctly shows no ignition; that is anatomy, not health.

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**

- All 86 tests collected by `python -m pytest tests` must continue to pass, including
  `test_matches_literal_reference_implementation` (120-neuron LPLC2/DNp01 subgraph),
  `test_determinism` (150-neuron subgraph), `test_float32_and_float64_give_identical_spikes` and
  the ~4,000-neuron escape-subnetwork tests.
- Real FlyWire v783 connectivity exactly as loaded: 139,255 / 3,732,460 / 50,666,648, no topology
  edits, no sign changes, `_w_sign` preserved, no synapse crossing zero.
- Bilateral ORN convergence preserved **uncorrected**: 94.1% left / 96.3% right ORN synapses onto
  63 shared bilateral first-order targets, hop0 77.85% input contrast collapsing to
  hop1 2.11% / hop2 6.41% / hop3 8.93%. Real adult *Drosophila* anatomy (requirement 3.7).
- `engine.spike_counts` stays cumulative-since-`reset()`; any windowed measurement is an addition.
- No navigation controller, no target/food coordinates into the brain, `plasticity_enabled` false
  and `wdata` unmodified for the whole diagnostic.
- Existing provenance headers and citations in `brain/neuron_models/lif.py` and
  `simulation/engine/lif_engine.py`.

**Scope:**

All configurations where `isBugCondition(X)` is false must be **bit-identical** between F and F'.
Concretely that means every subgraph run, every run shorter than 500 ms, and every run on a
connectome that is not the full v783 graph. Any candidate fix that changes behaviour in those
regimes is rejected on preservation grounds regardless of how well it stabilises the full network.
This is a hard constraint on the *form* a fix may take: an unconditional change to the integration
kernel or to `w_syn` breaks it, whereas an activity-dependent mechanism that is inactive at the
low rates subgraph tests produce can satisfy it — but only if verified, not assumed.

The actual required correct behaviour is criterion `S` in Correctness Properties (Property 1).

---

## Hypothesized Root Cause

Nine hypotheses. **None is pre-committed.** Each carries the observable that decides it, the
ladder rung it sits on, and a verdict field to be filled by measurement. H1–H5 and H8 support
hypothesis (i) *engine deviation*; H6 and H9 support hypothesis (ii) *unstable published
parameterization in an untested regime*; H7 is a designated control (requirement 2.17) and is
**not** a prime suspect unless it fails.

### H1 — float32 integration drift at full scale (requirement 2.11)

`_ev = exp(-0.005) = 0.99501248`, `_eg = exp(-0.02) = 0.98019867`,
`_kg = a(_eg - _ev)/(a - b) = 0.00493794` with `a = 0.05`, `b = 0.2`, `dt = 0.1`. These are
scalars, so the *values* cost microseconds to check exactly. The open question is whether
rounding them to float32 and accumulating over 20,000 steps x 139,255 neurons biases activity
upward.

*Prediction if true:* the float32 trajectory shows a systematic positive bias in `v` relative to
float64, and the float64 run has a materially lower plateau rate or does not ignite.

*Probe:* L0 exact scalar check; then L7 paired float32/float64 full-scale run. Bit-identity will
break at some step through chaotic divergence, which is expected and not itself evidence — the
falsification metric is the *statistics* (plateau rate, active fraction, sigma), plus the step of
first divergence as a diagnostic. `test_float32_and_float64_give_identical_spikes` already proves
identity on a few-thousand-neuron subgraph, so H1 is only about scale.

*Cost:* L0 < 1 s. L7 float64 doubles the edge arrays (30 MB each) and roughly doubles step cost;
a 500 ms paired run is ~50 s.

*Verdict:* __TBD__

### H2 — `_n_refractory` can under-count, skipping the refractory hold (requirement 2.12)

If `_n_refractory` reaches 0 while any `rfc_left > 0`, the `if self._n_refractory:` guard skips
stage 2's hold and refractory neurons get integrated.

*Static reading of the current code suggests this is unlikely:* after the decrement,
`_n_refractory` is recomputed as `count_nonzero(rfc_left[ridx])` over exactly the set of
previously-nonzero entries, which is the correct post-decrement total; newly-refractory neurons
are then added in stage 7 and cannot already be in `ridx` because a refractory neuron's `v` is
held at `v_rst = -52 mV`, 7 mV below threshold, and so cannot spike. One path makes it
*transiently too high*: calling `set_poisson` mid-run zeroes `rfc_left` for new targets without
decrementing the counter. Too high is safe — `ridx = flatnonzero(rfc_left)` is recomputed
correctly and the counter self-heals on the next step.

**This reading is not the answer.** Requirement 2.12 demands a measured invariant check, and the
static argument is exactly the kind of reasoning the bug already defeated once. See
"Refractory invariant instrumentation" in Testing Strategy for the sound, affordable check.

*Verdict:* __TBD__

### H3 — refractory `g` accumulates undecayed (requirement 2.13)

In stage 1 `g += pending` runs for all neurons; the `g_hold` snapshot is taken *after* that add
and restored *after* `g *= _eg`. A refractory neuron therefore accumulates incoming drive across
its whole 2.2 ms refractory period with no decay.

*Important finding:* R1, the in-repo literal reference, does **exactly the same thing** — it adds
`ring[slot]` to `g` for all neurons and then skips the `g[i] *= eg` for refractory ones. So the
engine and R1 agree here, and `test_matches_literal_reference_implementation` passing is **not**
evidence that this is faithful to Brian2 or to Shiu et al.; R1 shares the assumption under test.
Whether Brian2's `dg/dt = -g/tau (unless refractory)` really means "events accumulate, decay is
suspended" must be settled by R-TASK-1, not by R1.

*Prediction if true and unfaithful:* mean extra `g` per refractory neuron per refractory period is
a material fraction of the 7.0 mV threshold gap.

*Probe:* instrumented full-scale run recording, per refractory episode, `g` at entry and `g` at
exit, versus a counterfactual decayed value. Report mean and max. Rung L1b.
*Cost:* two extra `n`-length float32 arrays and one masked subtraction per step, ~30 us/step,
~0.6 s over 20,000 steps.

*Verdict:* __TBD__

### H4 — Poisson drive forces spikes and driven neurons never go refractory (requirement 2.14)

`w_syn * f_poi = 68.75 mV` is added directly to `v` against a 7.0 mV gap — roughly a 10x forcing
margin, so every Poisson event is a guaranteed spike. Separately, `set_poisson` sets
`rfc_len[idx] = 0`, so a driven ORN has no refractory period at all and its effective maximum rate
is unbounded by `t_rfc` (at 150 Hz commanded drive, `p = 0.015` per step, so it can in principle
fire on consecutive 0.1 ms steps).

Both choices are asserted from the reference `default_params` but never verified against reference
*behaviour*. R1 hardcodes `rfc_len[exc_idx] = 0` too, so again R1 cannot arbitrate. Settled by
R-TASK-1.

*Verdict:* __TBD__

### H5 — threshold detection has no refractory guard (new; discovered while writing this design)

Stage 4 is `spk = np.flatnonzero(v > self._vth32)` with no `rfc_left == 0` condition. R1's
condition is `v[i] > p.v_th and rfc[i] == 0`. Today the two agree, because a refractory neuron's
`v` is pinned at `v_rst` and the only mechanism that can raise `v` outside the held path is the
stage-3 Poisson kick — which applies only to `_poi_idx`, whose `rfc_len` is forced to 0.

**The divergence is latent, and candidate fix F2 makes it live.** If the driven set is given back
its refractory period (the most likely form of an H4/R-TASK-1 correction), then a refractory
Poisson target receiving a 68.75 mV kick will spike in F but is blocked in R1 and in Brian2.
Any F2 variant must therefore add the `rfc_left == 0` guard in the same change, and that guard
must be shown not to alter any subgraph result.

*Probe:* targeted unit test — a subgraph with a Poisson target whose `rfc_len` is non-zero,
compared against R1. Rung L0, cost milliseconds. This is the cheapest probe in the whole design
and it should be written first.

*Verdict:* __TBD__

### H6 — the network is genuinely supercritical at the published parameterization (2.15, 2.16)

Structural quantities already known: 13.57 mean synapses per connection giving a 3.73 mV `g` step
at `w_syn = 0.275` against a 7.0 mV gap; mean out-degree 26.8; 62.33% excitatory connections.
Two or three coincident presynaptic spikes therefore suffice to cross threshold, and with ~17
excitatory targets per neuron the naive branching estimate is well above 1.

*Prediction if true:* `sigma_rest > 1` — a single forced spike on a quiescent network ignites it —
and `sigma_transient > 1` during the growth phase.

*Probe:* L2 avalanche battery (primary), L4 generation-binned growth (corroborating), L0 spectral
bound (cheap early falsifier).

*Verdict:* __TBD__

### H7 — E/I sign assignment error (requirement 2.17) — CONTROL, not a suspect

Confirm at full scale that no neuron carries sign 0 or an inverted sign under the Shiu convention
(ACH/DA/OCT/SER excitatory; GABA/GLUT inhibitory), and report 62.33%/37.67% per-connection,
61.45%/38.55% synapse-weighted, with `nt_resolved` counts ACH 94,182, GLUT 22,726, GABA 18,389,
SER 1,435, DA 791, OCT 106, zero unknown.

*Probe:* pure array operations on `wdata` and the neuron table. Rung L0, cost ~1 s.
Escalate to suspect status **only if this control fails**.

*Verdict:* __TBD__

### H8 — a stabilising mechanism present in the reference is missing from our transcription (2.18)

Our engine has no spike-frequency adaptation, no synaptic depression, no global gain control and
no per-neuron input normalisation. Requirement 2.18 requires recording, for each, whether it is
absent from Shiu et al. 2024 or absent only from our transcription. Settled by R-TASK-1.

*Verdict:* __TBD__

### H9 — sustained whole-brain drive is an untested regime of the published model (2.19)

The decisive evidence for hypothesis (ii). Requires knowing the neuron-set sizes and drive
durations the reference actually exercises. Settled by R-TASK-1.

*Verdict:* __TBD__

### R-TASK-1 — Reference-implementation research task (blocks H3, H4, H8, H9)

Requirements 2.11, 2.14, 2.18 and 2.19 cannot be answered from this repository. R1 is a
transcription written by us and shares every assumption under test. **This design records the
determination of reference behaviour as an explicit research task, not an assumption.**

Source: `https://github.com/philshiu/Drosophila_brain_model`, primarily `model.py`, plus
Shiu et al. 2024 Nature 634:210-219 (doi:10.1038/s41586-024-07763-9) Methods and any
supplementary code notes. Fetch `model.py` and the example notebooks; do not infer from the paper
text alone where code is available.

Questions to answer, each with the decision it feeds:

| # | Question | Feeds |
|---|----------|-------|
| Q-A | Does `PoissonInput` target `v` or `g`? What are `N`, `rate`, `weight`? | H4, F2 |
| Q-B | Is the stimulated population's refractory period suppressed, or does `refractory='t_rfc'` apply to driven neurons too? | H4, H5, F2 |
| Q-C | Does the Brian2 `(unless refractory)` clause on `dg/dt` accumulate events without decay, as R1 assumes? | H3 |
| Q-D | How many neurons are driven in the published experiments, and for how long per trial? How many trials? | H9, F7 |
| Q-E | Does the reference simulate all 139,255 neurons, or named subsets? | H9, F7 |
| Q-F | Does the reference contain adaptation, depression, normalisation or any global gain term? | H8, F3–F5 |
| Q-G | Does the reference apply drive continuously for the whole trial, or as a pulse? | H9, F2 |

Deliverable: a table of answers with the exact file, symbol and line, recorded in the tasks
document. If a question cannot be resolved from the public code, record it as unresolved and mark
every conclusion that depends on it as conditional. **Do not substitute a guess.**

### R-TASK-2 — Independent full-scale reference (blocks H1, and any claim of scale conformance)

Requirement 2.11 asks for conformance verification "at full scale". Two independent obstacles make
a literal comparison impossible as currently written:

- **Memory.** R1 builds `np.asarray(conn.w.todense())`. At full scale that is
  `139,255^2 x 8 B = 155 GB` (77 GB in float32) against 16.4 GB of RAM, of which 3.8 GB was
  available when measured. R1 is dense-bound to roughly 20,000 neurons (3.2 GB), and even that
  does not fit in the currently available headroom.
- **Time.** R1's inner loop is per-neuron Python. At full scale that is 139,255 iterations per
  step; a 2 s run is 2.8e9 iterations.

**Therefore:** write R2, a *new* float64 reference that keeps R1's deliberate independence from
the engine (no shared code, no shared constants beyond `LIFParams`) but uses CSR rather than dense
`W`, and vectorises only the parts that cannot change semantics. Validate R2 against R1 for
bit-identical spike trains on the existing 120-neuron LPLC2/DNp01 case, then run R2 at full scale
over a **short window only** (target 50 ms; calibrate the achievable window empirically before
committing to a number). Report the deviation of the engine's float32 trajectory from R2 as
requirement 2.11 asks, and state plainly that the comparison covers a 50 ms window and not a
full-length run.

---

## Correctness Properties

Constants below are **OUR ENGINEERING** (category C): they are a criterion we chose, not values
published by Shiu et al. Each is justified, and the pair of Evidence E endpoints that must fail is
shown to fail.

**Criterion `S` — all of S1–S4 must hold; the ignition threshold is reported alongside.**

Fixed measurement configuration: the 258-ORN driven set, `r_lo = 1.0 Hz`, `r_hi = 150.0 Hz`
(a 150x span, matching Evidence C so results are directly comparable), duration 1,500 ms,
measurement window 1,000–1,500 ms, >= 3 seeds.

- **S1 gain.** `gamma = ln(R(r_hi)/R(r_lo)) / ln(r_hi/r_lo) >= 0.5`, where `R` is whole-brain mean
  rate over the measurement window. *Justification:* `gamma = 1` is proportional rate coding;
  `gamma = 0.5` is a square-root law, a deliberately modest floor consistent with the compressive
  exponents typical of sensory coding. Over a 150x input range `gamma = 0.5` means >= 12x output
  change. Current state `gamma = 0.030` fails by a factor of 16.
- **S2 liveness at `r_hi` (makes silencing unsatisfiable, requirement 2.23).** Active fraction
  `>= 0.02` (>= 2,785 neurons) **and** `R(r_hi) >= 0.5 Hz` **and** the DNa01 pair fires
  `>= 1 spike/s`. Applied at `r_hi` rather than `r_lo`, because a healthy subcritical network is
  legitimately near-silent at 1 Hz drive.
- **S3 non-saturation at `r_hi`.** No hop-level mean rate exceeds `0.25 x 454.5 = 113.6 Hz`.
  Every reported hop rate is printed alongside the 454.5 Hz ceiling and its fraction thereof
  (requirement 2.7). Current hop1 = 230.09 Hz = 0.506 of ceiling, fails.
- **S4 branching.** `sigma_rest <= 1.0` **and** `sigma_drive <= 1.0`, estimated by B1 and B2 below.
- **Reported, not gated:** ignition threshold `k*` (requirement 2.16), the subcritical / critical /
  supercritical classification, `sigma_transient`, time-to-plateau, and the post-drive decay
  diagnostic D6. Each value carries its provenance category (requirement 2.22).

**Both Evidence E endpoints fail `S`, by different clauses (requirement 2.23):**

| state | S1 gain | S2 liveness | S3 saturation | S4 branching | verdict |
|-------|---------|-------------|---------------|--------------|---------|
| `w_syn = 0.275` (saturated) | FAIL 0.030 | pass (6.8% active, 4.19 Hz) | FAIL hop1 0.506 of ceiling | FAIL (single ORN ignites) | **fails** |
| `w_syn = 0.0344` (silent) | to be measured | FAIL 0.040 Hz < 0.5 Hz; DNa01 silent | pass (hop1 2.8 Hz) | pass | **fails** |

The silent endpoint's `gamma` was never measured — Evidence E swept `w_syn` at a single drive.
Requirement 2.23 requires demonstrating failure, so the two-point drive sweep must be re-run at
`w_syn = 0.0344` as part of the programme (2 extra sweep points, ~2 min).

---

Property 1: Bug Condition — Sensory drive must modulate whole-brain activity

_For any_ whole-brain drive configuration `X` where the bug condition holds (`isBugCondition(X)`
returns true), the fixed engine `F'` SHALL satisfy criterion `S`: a log-log input-output gain of at
least 0.5 measured over drive rates 1.0–150.0 Hz on the 258-ORN set, with a minimum active
fraction of 0.02 and minimum whole-brain rate of 0.5 Hz at the high end so that silencing cannot
satisfy it, no hop-level rate above a quarter of the 454.5 Hz refractory ceiling, and a measured
per-spike branching ratio not exceeding 1.0 in both the rest and driven states.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.22, 2.23**

Property 2: Preservation — Sub-threshold-scale behaviour is bit-identical

_For any_ configuration where the bug condition does NOT hold (`isBugCondition(X)` returns false)
— every subgraph of 50 to ~4,000 neurons, every run shorter than 500 ms, and the 120-neuron
literal-reference comparison — the fixed engine `F'` SHALL produce spike trains and
`spike_counts` bit-identical to `F`, preserving all 86 existing tests, the real FlyWire topology
and synaptic signs, the uncorrected bilateral ORN convergence, and the documented cumulative
semantics of `engine.spike_counts`.

**Validates: Requirements 3.1, 3.2, 3.5, 3.6, 3.7, 3.11, 3.12, 3.13**

Property 3: Branching ratio is measured from a non-stationary or quiescent state

_For any_ branching-ratio measurement reported under requirement 2.15, the estimator SHALL be
either the quiescent single-spike perturbation (B1), the paired first-generation excess in the
driven state (B2), or the generation-binned growth rate restricted to the ignition transient (B3);
and the system SHALL NOT report a consecutive-bin ratio taken at plateau, which is pinned at 1.0
by stationarity and carries no information about criticality.

**Validates: Requirements 2.15, 2.16, 2.22**

Property 4: Provenance is attached to every mechanism and every criterion value

_For any_ mechanism present in `F'` but absent from Shiu et al. 2024, and _for any_ constant in
criterion `S`, the system SHALL label it OUR ENGINEERING with its provenance category and SHALL
NOT present it as the published model; and _for any_ changed published constant SHALL record the
deviation with its justification.

**Validates: Requirements 2.18, 2.22, 3.3, 3.4**

---

## Fix Implementation

### Baseline prerequisite (requirements 2.20, 2.21) — must complete before any number is evidence

Measured this session. `HEAD = 26672e0`. `git status --porcelain` shows **five** modified tracked
files, not the four the requirements list:

| file | +/- | semantic for simulation output? |
|------|-----|-------------------------------|
| `simulation/engine/lif_engine.py` | +933 / -116 | **yes** — already verified behaviourally bit-identical to HEAD |
| `brain/motor/descending.py` | +483 / -121 | readout only, downstream of the brain |
| `fly/body/fly_body.py` | +802 / -147 | body only, downstream of the readout |
| `config.py` | +42 / -20 | **no** — see below |
| `data/metadata/build_manifest.json` | +2 / -2 | **no** — see below; not in the requirements' list of four |

Total 2,262 insertions / 406 deletions.

Inspecting the two small diffs collapses most of the audit surface:

- `config.py` is blank-line reformatting plus exactly one semantic change: `DEFAULT_FLYWIRE_DIR`
  moves from `D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)` to `PROJECT_ROOT/data/flywire_v783`.
  That changes only where source CSVs are read from. The connectome the engine consumes is the
  already-built `data/derived/connectome_v783.npz`, whose identity is asserted by
  `tests/test_data_provenance.py` checksums. So `config.py` cannot affect a simulation number.
- `build_manifest.json` changes `source_dir` and `built_utc` only — provenance strings read by
  `test_manifest_records_real_dataset`. Non-semantic for simulation.

**So the neural-stability numbers depend only on `lif_engine.py` (verified) plus the checksummed
`.npz`.** `descending.py` and `fly_body.py` are not in the neural simulation path at all; they
matter only for the downstream Phase 3B/3C claims that requirement 2.24 retracts. The 2,600-line
worry reduces to a bounded task.

Also load-bearing and **untracked**: the entire `simulation/environment/` directory (13 scripts,
including every `run_sandbox_*.py` that produced the reported CSVs), `tests/test_fly_body.py`,
`tests/test_plasticity.py`, `.kiro/`, `data/flywire_v783/`, and two root-level test files
`test_bilateral_odor.py` / `test_four_motor_commands.py`.

**Recommendation: COMMIT, do not revert, and do not merely pin.** Reasons:

1. **Revert is not viable.** `tests/test_plasticity.py` is untracked and exercises the
   plasticity API that exists only in the working-tree `lif_engine.py`; reverting the engine
   breaks a test that is part of the 86. And HEAD's `config.py` points at a `D:\` path that does
   not exist on this machine, so a HEAD tree cannot even load the data without an env override.
2. **Pinning cannot be verified or diffed.** Requirement 2.20 wants "the exact revision" every
   number was produced under, which requires a hash.
3. Commit the five modified files plus `simulation/environment/` and the two untracked
   `tests/*.py` as a single baseline commit on a branch. Keep the generated CSVs and
   `data/flywire_v783/` out (large source data, already checksummed).

**Also record:** the 86-test baseline is `python -m pytest tests` specifically. A bare
`python -m pytest` from the repository root does **not** terminate — measured: it was still
running at a 300 s timeout, because it collects the two root-level `test_*.py` files. Pin the
invocation in the baseline record.

For requirement 2.21, the remaining HEAD-vs-working-tree comparison covers `descending.py` and
`fly_body.py`. Method: `git worktree add ../fruit-fly-lab-head 26672e0`, set
`FLYWIRE_V783_DIR=<repo>/data/flywire_v783` in that worktree, copy or link `data/derived/`, then
swap **one** module at a time into the working tree, run the same diagnostic, and diff outputs.
One-at-a-time swapping is necessary because the modules are interdependent
(`fly_body <- descending <- engine`) and a whole-tree swap would not isolate the cause. Record
bit-identity per module. Expect `descending.py` / `fly_body.py` to differ — they were substantially
rewritten — and record that as the reason Phase 3B/3C numbers are not reproducible, which is
consistent with 2.24 rather than a new problem.

### Candidate fix paths — gated on hypothesis verdicts, none pre-committed

Ordered so that conformance restoration is attempted before any unpublished mechanism is added.

**F1 — Correct an engine deviation.** *Gate:* H1, H2, H3 or H5 survives its probe. *Provenance:*
restores category B; a bug fix, not a new mechanism. *Note:* H5's guard
(`v > v_th AND rfc_left == 0`) is a no-op today and must be added as part of F2 if F2 is chosen.

**F2 — Match the reference drive protocol.** *Gate:* R-TASK-1 answers Q-A/Q-B/Q-G differently from
what we do. Variants: restore refractoriness on driven neurons (`set_poisson` stops zeroing
`rfc_len`, and H5's guard is added in the same change); deliver drive to `g` instead of `v`;
deliver drive as a pulse rather than continuously. *Provenance:* category B restoration **if and
only if** R-TASK-1 shows the reference differs. If R-TASK-1 shows the reference does what we do,
then any change here is category C — our deviation from the published parameterization — and must
be labelled so (requirement 3.3).

**F3 — Divisive gain control / per-neuron input normalisation.** *Gate:* H6 survives and F1/F2 are
insufficient. *Provenance:* **OUR ENGINEERING**, absent from Shiu et al. 2024.

**F4 — Spike-frequency adaptation.** *Gate:* as F3. *Provenance:* **OUR ENGINEERING**.

**F5 — Short-term synaptic depression.** *Gate:* as F3. *Provenance:* **OUR ENGINEERING**.

**F6 — Uniform gain reduction (`w_syn` scaling). ALREADY FALSIFIED AS A SUFFICIENT FIX.**
Evidence E measured 0.2750 / 0.1375 / 0.0688 / 0.0344 giving 4.192 / 1.649 / 0.639 / 0.040 Hz and
found the network passes from saturated to silent with **no input-encoding regime in between**.
Requirement 2.23 makes an encoding regime mandatory, so F6 may never be applied alone. It is
retained only as a component of a composite, and any composite containing it must still pass S1.

*Why F4/F5 can plausibly succeed where F6 cannot — the mechanistic argument that must be tested,
not assumed.* F6 multiplies every synapse by a constant, scaling `sigma` uniformly. `sigma` crosses
1 at some `w_syn`, but near criticality a branching process has divergent susceptibility, so the
window of `w_syn` in which the network is both alive and non-saturating is vanishingly narrow —
which is exactly the saturated-to-silent jump Evidence E observed. F3–F5 are *activity-dependent*:
they make `sigma_effective(R)` a decreasing function of rate, which turns the operating point into
a stable fixed point with a wide dynamic range instead of a knife edge. That is the mechanism by
which an encoding regime can exist at all. Requirement 2.23 demands it be demonstrated by
measurement of `gamma`, not argued.

**F7 — Do not fix; re-scope. A legitimate terminal outcome.** *Gate:* H6 and H9 both survive, F1
and F2 find no deviation, and no reference-conforming configuration passes `S`. Then the honest
result is: the published parameterization does not support sustained whole-brain Poisson drive;
restrict all claims to the regime R-TASK-1 shows the reference exercises; retract Phase 1–3C
(2.24–2.27); and record any stabilisation offered as category C engineering that departs from the
published model. This outcome must remain on the table throughout — the requirements' honesty
constraints make "no fix, correctly characterised" preferable to an unattributable fix.

### Files expected to change

| file | change |
|------|--------|
| `simulation/engine/lif_engine.py` | conditional on F1/F2; instrumentation hooks (opt-in, default off) |
| `brain/neuron_models/lif.py` | only if a published constant changes — with deviation and justification recorded (3.3, 3.4) |
| `simulation/diagnostics/stability.py` (new) | observables D1–D8, estimators B1–B3, criterion `S` |
| `simulation/environment/run_stability_diagnostic.py` (new) | CLI driver, one condition per process |
| `tests/test_reference_conformance.py` (new) | R2 and the R1-vs-R2 equivalence check |
| `tests/test_whole_brain_stability.py` (new) | requirements 2.31, 2.32 |
| `pytest.ini` (new) | marker registration; no pytest config exists today |

Instrumentation must default to **off** so that Property 2's bit-identity is not disturbed, and
the off path must add no arithmetic to the hot loop.

---

## Testing Strategy

### Measured cost model

All figures measured this session, full v783 connectome, working tree, `float32`, single process:

- `load_connectome()` 1.78 s (`lru_cache`d, so amortised across conditions in one process);
  `LIFEngine(c)` construction 0.06 s.
- Quiescent: **0.187 ms/step**.
- Ignition transient (13.2 spikes/step): **0.746 ms/step**.
- Plateau (58.3 spikes/step): **1.997–2.072 ms/step**, mean 2.04.
- Linear fit, accurate to ~20% across all three regimes:

```
T_wall  ~=  0.19 ms x n_steps  +  0.032 ms x n_spikes
```

Checks: a 2 s ignited single-ORN run predicts 3.8 s + 28.4 s = 32.2 s; measured 29.4 s.
A 2 s run at 20,000 steps is therefore **~30 s**, matching the "tens of seconds" premise.

Cost scales with *spike count*, so ignited conditions are the expensive ones and quiescent probes
are nearly free. This is what makes the escalation ladder work.

Two hot-loop notes: `np.count_nonzero` on the 139,255-entry `rfc_left` costs **22.3 us**, while
`np.flatnonzero` on the same array costs **621 us**. The per-step `flatnonzero(rfc_left)` in the
refractory-hold branch is therefore a large share of the 2.04 ms plateau step — relevant to any
optimisation, not to correctness.

### Memory footprint

Per full-scale engine, arithmetic (`n = 139,255`, `nnz = 3,732,460`, float32):

| array | size |
|-------|------|
| `indices` (int32) | 14.93 MB |
| `indptr` (int64) | 1.11 MB |
| `wdata` | 14.93 MB |
| `eligibility` | 14.93 MB |
| `_w_sign` | 14.93 MB |
| `_w_abs_initial` | 14.93 MB |
| `_ring` (19 x 139,255) | 10.58 MB |
| per-neuron arrays (`v`, `g`, `_tmp`, traces, `rfc_*`, `spike_counts`, `_silenced`) | ~4.5 MB |
| **total** | **~91 MB** |

Note there are **four** edge-sized float32 arrays, not the three the requirements list:
`wdata`, `eligibility`, `_w_sign` and `_w_abs_initial`.

Measured process RSS: 232 MB after `load_connectome()`, 243 MB after engine construction. The
arithmetic total exceeds the RSS delta because `np.zeros` pages are not committed until touched
and `eligibility` is never touched with plasticity disabled. Budget **~300 MB per full-scale
engine process**. Two concurrent engines for an F-vs-F' comparison fit in under 700 MB.

Constraint to respect: **16.4 GB total but only 3.8 GB available** when measured. With 6 physical
cores and ~300 MB per process, 6 parallel conditions cost ~1.8 GB and fit. NumPy elementwise ops
and `bincount` are single-threaded, so parallelism must come from processes, not threads.

R1's dense `W` at full scale is 155 GB — infeasible. See R-TASK-2.

### Escalation ladder

Each rung is ordered by cost and can falsify a hypothesis before the next rung is paid for.
**Rungs L0–L2 cost under 6 minutes in total and can falsify H1, H2, H5, H6 and H7.**
Nothing above L2 should be run until L0–L2 have been read.

| rung | what | falsifies | cost |
|------|------|-----------|------|
| **L0** | Scalar coefficient check (`_ev`, `_eg`, `_kg`, float32 variants); structural stats; E/I sign control; H5 unit test on a subgraph; linearised spectral bound (ARPACK largest \|lambda\| of the threshold-normalised excitatory gain matrix, a few sparse matvecs) | H1 (values), H5, H7; early signal on H6 | **< 10 s** |
| **L1** | Full network, **no drive**, 200 ms — asserts zero spontaneous activity | any hypothesis predicting drive-free activity | **~2 s** + 1.8 s load |
| **L1b** | Instrumented 1 s ignited run: exact refractory invariant every step + H3 `g`-inflation accounting | H2, H3 | **~14 s** |
| **L2** | Avalanche battery: 200 single-spike perturbations from rest, capped | H6 (primary), gives `sigma_rest`, `k* = 1` check | **~4 min** (~1 min on 6 cores) |
| **L3** | Single-ORN ignition, 5 seeds x 2 s | reproduces Evidence A with a spread | **~2.5 min** |
| **L4** | Drive sweep: 4 rates x 3 seeds x 1.5 s, 258 ORNs; yields `gamma`, D1–D6, `sigma_transient` | measures S1, S2, S3 | **~4 min** |
| **L5** | Ignition threshold: bisection over `k` in {1,2,4,...,258}, 3 seeds | requirement 2.16 | **~6.5 min** |
| **L6** | Per-candidate-fix evaluation of full criterion `S` (= L2 + L4 + L5) | each of F1–F5 | **~15 min per candidate** |
| **L7** | R2 full-scale conformance over a 50 ms window; paired float32/float64 500 ms run | H1 at scale, requirement 2.11 | **calibrate first**; float32/64 pair ~50 s |

Whole programme: **~2.5–3 h single-process**, or **~45–60 min across 6 processes**, including
5 candidate fixes and the two Evidence E endpoints.

**Execution rule (non-negotiable).** Nothing above L2 is launched from a blocking tool call.
Every rung L3 and above is handed to the user as a command to run in their own terminal, writing
results to CSV, for example:

```powershell
python -m simulation.environment.run_stability_diagnostic --rung L4 --seeds 0,1,2 --out stability_L4.csv
```

The driver runs one condition per process, writes incrementally so a partial run is still usable,
and prints an ETA from the cost model above before doing any work.

### Branching-ratio estimation (requirement 2.15) — the tractable method

Exact causal attribution per spike across 3.73M edges is not affordable, and it is not even
well-defined for an LIF neuron whose threshold crossing is caused by the *sum* of several
presynaptic events. Three estimators, with an explicit primary.

**The generation, not the step, is the unit.** `t_dly = 1.8 ms = 18 steps`, so a spike at step `t`
can only influence spiking at step `t + 18`. Any consecutive-*step* ratio is meaningless.

**A plateaued network has branching ratio 1.0 by definition.** Stationarity forces
`spikes(t+18) = spikes(t)`. Measuring at plateau therefore cannot distinguish a critical network
from a supercritical one that has saturated against the refractory ceiling — which is precisely
the state the network is in. **Every branching measurement must come from a quiescent state or
from the ignition transient.**

**B1 — `sigma_rest`, quiescent single-spike perturbation. PRIMARY; this is the estimator
criterion S4 is tested against.**

Protocol: full network at rest (`v = v_0`, `g = 0`, no Poisson drive). Force exactly one spike in
neuron `i` by writing its outgoing events into the ring buffer directly, then run with no drive
until either no spike has occurred for `2 x 18 = 36` consecutive steps (avalanche extinct) or a
cap is reached (5,000 spikes or 500 ms, whichever first — "ignited/truncated"). Record avalanche
size `S_i` and generation-by-generation counts. Repeat over 200 seed neurons stratified by
out-degree, all with `outDegree > 0` (requirement 2.9, which also excludes R[0] index 164 and
reports the exclusion).

Estimator: for a subcritical Galton-Watson process `E[S] = 1/(1 - sigma)`, so
`sigma_rest = 1 - 1/mean(S)` over non-truncated avalanches, reported together with the truncation
fraction `P(ignition)`. If any avalanche truncates, `sigma_rest > 1` is established directly and
the point estimate is reported as a lower bound.

*Why this is the primary.* It measures exactly the quantity 2.15 names, from a state where the
measurement is meaningful; `sigma_rest <= 1.0` has the precise operational meaning "a single spike
cannot ignite the network", which is the negation of Evidence A; and it is cheap, because a
subcritical avalanche dies almost immediately (0.19 ms/step) and a supercritical one is capped.

*Stated limitation.* `sigma_rest` is measured at rest, where `v = -52 mV` sits 7 mV below
threshold. In the driven state `v` is distributed nearer threshold and `g` is elevated, so the
effective branching ratio is higher. `sigma_rest` is therefore a **lower bound** on the driven-state
ratio, and S4 requires B2 as well.

**B2 — `sigma_drive`, paired first-generation excess in the driven steady state. Also gated by S4.**

Run the identical driven configuration twice from the same seed. The engine is deterministic given
its seed, so the two runs are identical step for step. In the second run, at a fixed step `t0`
inside the measurement window, inject one extra forced spike. Measure the excess spike count in
generation 1 only, i.e. over steps `[t0+18, t0+36)`:
`sigma_drive = spikes_perturbed - spikes_unperturbed` over that window.

Restricting to the first generation is deliberate: two deterministic trajectories diverge
chaotically after a perturbation, so a difference measured over a long window reflects divergence
rather than branching. Average over >= 20 injection times and >= 3 seeds.
Cost: 2x a driven run, ~40 s per seed.

**B3 — `sigma_transient`, generation-binned growth rate. CORROBORATING; essentially free.**

Bin the per-step spike counts (already available as `len(spk)` per step) into 18-step generations
over the ignition transient only, then fit `ln n(k)` against generation index `k`;
`sigma_transient = exp(slope)`. Measured this session, the transient is well resolved: for seed 0
the 250–500 ms bin holds 33,107 spikes against 5 in the preceding bin and 142,425 in the following
one, so there are several clean generations of growth before saturation. Restrict the fit to
generations where activity is below 25% of the plateau, and report the fit window.

The external contribution must be netted out: at 20 Hz on one ORN the drive supplies ~20 spikes/s
against a plateau of ~583,000 spikes/s, so it is negligible at plateau but **not** negligible in
the first generations, where it is the only source. Start the fit at the first generation whose
count exceeds 10x the expected injected count in that generation.

**B4 — linearised spectral estimate. REJECTED as the primary; retained as a cheap L0 probe.**

Rejected because: the LIF map is not linear near threshold; the effective per-edge gain depends on
the momentary `v` distribution and on refractoriness, neither of which the weight matrix knows;
and the spectral radius of a signed, strongly non-normal 3.73M-edge operator gives a bound whose
looseness is unquantified. Retained at L0 as an *early falsifier only*: a few ARPACK iterations on
the threshold-normalised excitatory gain matrix cost seconds, and a value far above 1 is an early
corroboration of H6 that justifies paying for L2.

### Refractory invariant instrumentation (requirement 2.12) — the exact check is affordable

Requirement 2.12 asks for `_n_refractory == count_nonzero(rfc_left)` over the whole network at
every step of a full-scale run. Measured: `np.count_nonzero` on the 139,255-entry int32 `rfc_left`
costs **22.3 us**, so 20,000 steps cost **0.45 s** — about **1.1% overhead** on a ~41 s run.

**No sampling is needed.** Specify an exhaustive check behind an opt-in flag
(`LIFEngine(..., check_invariants=True)`, default `False`) that on every step asserts the identity
and, on violation, records the step index, `_n_refractory`, `count_nonzero(rfc_left)` and the
number of neurons wrongly integrated. The flag must be checked once outside the arithmetic so the
default path is untouched (Property 2).

**Detection guarantee:** exhaustive over every step of every instrumented run. If the invariant
holds for a 2 s ignited run at three seeds, H2 is falsified for that trajectory class with no
sampling gap — the strongest statement available short of a proof. It is *not* a proof for all
inputs; the design records that limit rather than overclaiming. A sampled check would have given a
detection probability instead of a guarantee, for a saving of 0.45 s; the exact check is strictly
better here and this is the only reason sampling is not used.

H3's `g`-inflation accounting rides on the same flag: two extra `n`-length float32 arrays holding
`g` at refractory entry and the counterfactual decayed value, differenced on exit, ~30 us/step.

### Observables

Every observable is reported with its provenance category (requirement 2.22).

| id | observable | requirement |
|----|-----------|-------------|
| D1 | brain spikes per injected spike, per seed, with spread | 2.1 |
| D2 | total activity vs injected spike count over the ORN-count sweep | 2.2 |
| D3 | `gamma` over 1.0–150.0 Hz | 2.3 |
| D4 | 250 ms-binned time course and time-to-plateau (first bin within 5% of the mean of the next three) | 2.4 |
| D5 | active fraction per condition | 2.5 |
| D6 | hop0–hop3 mean rate and population by BFS hop distance over the unsigned adjacency, each printed with its fraction of the 454.5 Hz ceiling | 2.6, 2.7 |
| D7 | seed exclusion report: out-degree of every candidate seed, which were excluded and why | 2.9 |
| D8 | post-drive decay: activity 100 ms after drive removal as a fraction of the driven level — separates a stimulus-locked response from an ignition plateau | 2.4 |

### Exploratory Bug Condition Checking

**Goal:** surface counterexamples on the **unfixed** engine, and confirm or refute the root-cause
analysis before any fix is written. If the surviving hypothesis set is empty or contradictory,
re-hypothesise rather than proceed.

**Test plan:** run L0–L5 against `F` and record every verdict field in H1–H9. These runs are
expected to *fail* criterion `S` — that is their purpose.

**Test cases:**

1. **H5 subgraph divergence** — Poisson target with non-zero `rfc_len`, engine vs R1
   (will diverge on unfixed code if the guard is genuinely missing; L0, milliseconds).
2. **Quiescent control** — full network, no drive, 200 ms: expect exactly zero spikes.
   (Should pass on unfixed code; a failure here would reframe the whole investigation.)
3. **Refractory invariant** — exhaustive check over a 1 s ignited run (may pass on unfixed code;
   the static reading in H2 predicts it will, and a pass falsifies H2).
4. **Avalanche battery** — 200 single-spike perturbations from rest (expect a non-zero truncation
   fraction on unfixed code, i.e. `sigma_rest > 1`).
5. **Drive sweep** — `gamma` over 1.0–150.0 Hz (expect ~0.030, failing S1 by 16x).
6. **Hop saturation** — hop1 vs the 454.5 Hz ceiling (expect ~0.506, failing S3).
7. **Seed sensitivity** — single ORN, 5 seeds (expect a ~2x spread in cumulative spikes with a
   stable plateau rate, as measured this session for seeds 0–2).

**Expected counterexamples:** single-spike avalanches that do not terminate; `gamma` far below
0.5; hop1 above a quarter of the refractory ceiling.
*Possible causes:* supercritical structural gain (H6); forced-spike Poisson drive without
refractoriness (H4); an untested drive regime of the published model (H9).

### Fix Checking

**Goal:** for all inputs where the bug condition holds, `F'` satisfies criterion `S`.

```
FOR ALL X WHERE isBugCondition(X) DO
  result := F'(X)
  ASSERT expectedBehavior(result)      // criterion S: S1 AND S2 AND S3 AND S4
END FOR
```

```
FUNCTION expectedBehavior(result)
  RETURN result.gamma            >= 0.5          // S1
     AND result.active_fraction  >= 0.02         // S2
     AND result.brain_rate_hz    >= 0.5          // S2
     AND result.dna01_rate_hz    >= 1.0          // S2
     AND result.max_hop_rate_hz  <= 113.6        // S3, 0.25 x 454.5
     AND result.sigma_rest       <= 1.0          // S4, estimator B1
     AND result.sigma_drive      <= 1.0          // S4, estimator B2
END FUNCTION
```

Realised as L6 per candidate fix. A candidate that lowers activity without reaching
`gamma >= 0.5` is **rejected**, per requirement 2.23 and the F6 falsification.

### Preservation Checking

**Goal:** for all inputs where the bug condition does not hold, `F'(X) = F(X)` bit-identically.

```
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
END FOR
```

**Testing approach.** Property-based testing is the right tool here: the preservation domain is
large (arbitrary subgraphs, sizes, drive sets, rates, durations, seeds) and the assertion is
exact equality, so generated cases are cheap to check and every generated case is a real test.
Manual unit tests would cover a handful of points in a space where the fix's activity-dependent
mechanisms are most likely to leak at the boundaries.

**Test plan.** Capture `F`'s behaviour on the unfixed engine first, as golden spike-count vectors
keyed by `(subgraph spec, drive spec, seed, duration, dtype)`, then assert `F'` reproduces them.

**Generator (Hypothesis or equivalent), all inside `NOT isBugCondition`:**
- subgraph size drawn from 50–4,000; index set drawn deterministically from the seed
- driven set 1–50 indices from the subgraph, rate 1–300 Hz
- duration 5–400 ms (strictly below the 500 ms bug-condition floor)
- `dtype` in {float32, float64}
- engine seed 0–1000

**Test cases:**

1. **Golden-vector preservation** — observe `spike_counts` on unfixed code across the generated
   domain, then assert bit-identity after the fix.
2. **Literal reference preservation** — `test_matches_literal_reference_implementation` continues
   to pass unchanged on the 120-neuron LPLC2/DNp01 subgraph.
3. **Determinism preservation** — the 150-neuron determinism test, unchanged.
4. **float32/float64 preservation** — `test_float32_and_float64_give_identical_spikes`, unchanged.
5. **Escape-subnetwork preservation** — the ~4,000-neuron tests, unchanged.
6. **Bilateral convergence preservation** — hop0 77.85% collapsing to hop1 2.11% / hop2 6.41% /
   hop3 8.93% is *reproduced*, not improved; the test asserts it stays within tolerance of the
   recorded values and is labelled REAL CONNECTOME (requirements 2.29, 2.30, 3.7).
7. **Instrumentation-off preservation** — runs with `check_invariants=False` are bit-identical to
   runs of the pre-instrumentation engine.
8. **Plasticity-untouched preservation** — `plasticity_enabled` stays false and `wdata` is
   unmodified across the whole diagnostic (requirement 3.10).

### Unit Tests

- Exact scalar values of `_ev`, `_eg`, `_kg` in float64 and float32 (requirement 2.11).
- H5: Poisson target with non-zero `rfc_len` does not spike while refractory, matching R1.
- Refractory invariant on a small subgraph, both instrumented and not.
- Avalanche extinction bookkeeping: a forced spike on an isolated neuron gives `S = 1`.
- Generation binning uses `delay_steps = 18`, and a synthetic branching process with known
  `sigma` is recovered by B3 within tolerance.
- E/I sign control: no zero or inverted signs; the published fractions and `nt_resolved` counts
  reproduce (requirement 2.17).
- Seed exclusion: index 164 has out-degree 0 and is excluded with a recorded reason
  (requirements 2.9, 3.8).

### Property-Based Tests

- **Preservation** (the generator above): `F(X) = F'(X)` for all `NOT isBugCondition(X)`.
- **Sign invariance:** for any weight-modifying operation, `sign(wdata)` equals `_w_sign` and no
  entry crosses zero (requirement 3.2).
- **Monotonicity of the gain:** for random ordered drive pairs `r1 < r2` in 1–150 Hz, whole-brain
  rate is non-decreasing. Post-fix only; expected to fail pre-fix, which is the point.
- **Branching-estimator soundness:** on synthetic Galton-Watson trees with known `sigma`, B1's
  `1 - 1/mean(S)` recovers `sigma` within tolerance, including near `sigma = 1` where variance is
  largest, so the truncation policy is validated independently of the connectome.

### Integration Tests

- **Whole-brain stability (requirement 2.31).** Full 139,255-neuron network: single-connected-seed
  ignition bound (B1 avalanche subset) plus `gamma` over a drive range, asserting criterion `S`.
- **Whole-brain specificity (requirement 2.32).** Whole-brain counterpart to
  `tests/test_circuits.py`: bitter GRN drive produces zero proboscis motor neuron spikes at full
  scale, where the runaway manifests.
- **Baseline reproducibility (requirements 2.20, 2.21).** Same revision, seed and parameters
  reproduce a recorded number exactly; the module-swap comparison for `descending.py` and
  `fly_body.py` runs as a script and records bit-identity per module.

### Test-suite runtime budget and markers

The suite today is 86 tests in 26 s via `python -m pytest tests`. There is no pytest
configuration file in the repository, so markers must be registered in a new `pytest.ini`.

**Default suite (`python -m pytest tests`) — budget +8 s, total ~34 s.** Everything from L0 and
L1 plus the small unit and property tests: scalar coefficients, H5 guard, E/I sign control,
structural stats, seed exclusion, quiescent 200 ms full-network run (~2 s), refractory invariant
over a 100 ms full-network window (~0.5 s), and a bounded preservation property run
(`max_examples=25`, subgraphs capped at 1,000 neurons). The 1.78 s connectome load is shared via
the existing module-scope fixture. Memory ~300 MB.

**Opt-in whole-brain suite — budget <= 12 min, not run by default.**

```ini
[pytest]
markers =
    wholebrain: full 139,255-neuron simulation; minutes, opt-in
    slow: over 10 s
addopts = -m "not wholebrain"
```

```powershell
python -m pytest tests -m wholebrain            # ~12 min
python -m pytest tests -m "not wholebrain"      # the ~34 s default
```

Contents and budgets: requirement 2.31 stability test — B1 with 20 seed neurons plus a two-point
drive sweep at 1.0 and 150.0 Hz for 1,500 ms, ~5 min; requirement 2.32 specificity test,
~1.5 min; the R2 conformance window, ~2 min once calibrated. Memory ceiling 1 GB, so it runs
inside the measured 3.8 GB of available RAM alongside the rest of the suite.

**Pre-fix state of these tests.** Both whole-brain tests will fail against `F` — that is their
purpose. Mark them `xfail(strict=True)` with a reason referencing this spec until a fix lands, so
the suite stays green while the failure remains recorded and an accidental pass is itself an error.
Requirement 2.31 explicitly prefers marking slow over omitting; requirement 3.5's "all 86 existing
tests pass" is unaffected because these are additions.

**Do not** put whole-brain tests in the default path. At 2.04 ms/step a single ignited 2 s
condition is 30 s, larger than the entire current suite.

---

## Established vs New (requirement 7 of the design brief)

**Already established — record, do not re-measure.** Evidence A–G in the requirements introduction
were measured directly against the loaded connectome. This design treats them as inputs:

- A: single connected ORN ignition; 3 of 4 tested ORNs ignite; R[0] index 164 is anatomically
  isolated.
- B: ORN-count and drive sweeps (123x injected spikes for +6.5%; 150x drive for +16%); per-hop
  clamp with hop1 at 230 Hz; 6.8–6.9% activation; plateau inside the first 500 ms bin.
- C: whole-brain rates 4.09 / 4.15 / 4.25 / 4.76 Hz over 1.0 / 5.0 / 20.3 / 150.0 Hz.
- D: bilateral contrast collapse and its 94.1% / 96.3% shared-target mechanism across 63 targets.
  **REAL CONNECTOME — must not be fixed** (requirements 2.29, 2.30, 3.7).
- E: `w_syn` sweep, saturated to silent with no encoding regime. **This falsifies F6 as a
  sufficient fix and that conclusion is not revisited.**
- F: HEAD vs working-tree `lif_engine.py` bit-identical.
- G: 86 tests in 26 s, all on subgraphs.
- Structural: 13.57 synapses per connection, out-degree 26.8, 62.33%/37.67% E/I,
  61.45%/38.55% synapse-weighted, `nt_resolved` counts, zero unknown.
- Phase 1–3C retractions (2.24–2.27) and the phase3b relationship (2.28, 3.15) are **recording
  tasks**, not experiments.

**Requires new measurement.** `sigma_rest`, `sigma_drive`, `sigma_transient`; the ignition
threshold `k*` and the criticality classification; the exhaustive refractory invariant; the H3
`g`-inflation magnitude; the full-scale float32-vs-float64 deviation; R2 and the full-scale
conformance window; R-TASK-1's seven answers; the E/I full-scale sign control; `gamma` at the
`w_syn = 0.0344` endpoint; the `descending.py` / `fly_body.py` HEAD comparison; and the seed
spread on every cumulative observable.

**Measured while writing this design, and recorded here as new evidence.** The cost model; the
memory footprint including the fourth edge-sized array `_w_abs_initial`; the ~2x seed spread in
single-ORN cumulative spikes against a stable plateau rate and active count; the `count_nonzero`
cost that makes the exhaustive invariant check affordable; the reduction of the baseline audit
surface to two non-simulation modules; the 155 GB dense-`W` obstacle to full-scale R1; that
`python -m pytest` from the repository root does not terminate; that R1 shares the H3 and H4
assumptions and therefore cannot arbitrate them; and H5.
