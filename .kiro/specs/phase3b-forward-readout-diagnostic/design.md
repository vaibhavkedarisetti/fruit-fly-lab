# Phase 3B Forward Readout Diagnostic — Bugfix Design

## Overview

The Phase 3B "TRUE SEPARATED FORWARD" diagnostic must answer Q0: in the Phase 3 spatial-odor
experiment, was the reported forward locomotion a genuine bilateral locomotion command, or an
arithmetic re-labelling of unilateral DNa01/DNa02 steering activity?

`simulation/environment/run_sandbox_phase3b_true_separated_forward.py` is **0 bytes on disk**.
There is no draft to repair. The defect clauses in `bugfix.md` describe a diagnostic design that
does not exist as code, so the fix is to **author** the diagnostic such that none of those defects
is present at first commit. `run_sandbox_separated_four_channel_clean.py` is the structural
template: it already demonstrates the correct connectome load, ORN resolution with 129-per-side
balancing, `set_poisson`, `engine.run(chunk)`, `readout.channels(counts, window)` and
`body.update(...)` call pattern.

The fix strategy has four parts.

1. **Non-invasive audit.** Nothing outside the new file changes. Both forward rules are recomputed
   inside the diagnostic from the per-side `left`/`right` activations that `DescendingReadout.channels`
   already returns. `brain/motor/descending.py`, `simulation/engine/lif_engine.py`,
   `fly/body/fly_body.py` and the nine existing sandbox scripts are not touched (3.7, 3.11).
2. **Correct measurement.** DN rates come from a per-window spike count obtained by
   snapshot-and-difference of the cumulative `engine.spike_counts`, over the same 100 ms window that
   is passed as `window_ms` (2.10).
3. **Falsifiable headline.** The reported quantity is the pre-clip unilaterality
   `U = |L - R| / (L + R)` and the contamination fraction `U / (1 + U)`, with per-rule clip fractions
   so a saturated window is never read as agreement between the two rules (2.14–2.17).
4. **Audit the shipped configuration.** The experiment under question is
   `run_sandbox_phase3_closed_loop_odor.py` as shipped: `ODOR_FALLOFF_MM = 250.0`,
   `ANTENNA_OFFSET_MM = 60.0`, `STEERING_GAIN = 0.25`, peak per-ORN rate 150 Hz
   (`odor_vinegar.max_rate_hz`), world 1000 × 700 mm, start (200, 350), food (800, 350). These are
   **not** the 400 mm falloff / 0.35 mm half-separation values named in the draft. The diagnostic
   audits the shipped numbers, because those are what produced the results under question, and it
   **computes** every contrast and noise-floor figure at runtime rather than hard-coding it.

`ANTENNA_OFFSET_MM = 60.0` on a fly whose body is ~3 mm long is surfaced as a separate finding with
its own measured consequence: it is a 171× inflation of the bilateral baseline, and it is the single
reason any bilateral odor contrast is visible at arena scale at all.

## Glossary

- **Bug_Condition (C)** — a diagnostic window, or a reported diagnostic quantity, that cannot
  discriminate a real connectome effect from an artifact of the measurement interface. Concretely: a
  DN rate measured over a window it was not counted over, a headline metric that is positive by
  construction, a between-rule comparison taken after clipping, or a reported number with no
  provenance category.
- **Property (P)** — the desired behaviour: every reported DN rate is counted over the window it is
  divided by; the headline quantity is pre-clip and can return an arbitrarily small value; every
  number carries exactly one provenance category.
- **Preservation** — the shipped behaviour of `DescendingReadout.channels`, the four motor gains,
  the cumulative semantics of `engine.spike_counts`, `FlyBody` kinematics, the FlyWire topology, and
  the byte-for-byte output of all nine existing sandbox scripts.
- **L, R** — the per-side DNa activations returned as `channels["left"]` and `channels["right"]`.
  Each is the mean over `{DNa01, DNa02}` on that side of `act = hz / (hz + CHANNEL_HALF_MAX_HZ)`
  with `CHANNEL_HALF_MAX_HZ = 60.0`. `TURN_GAIN = 1.0`, so `channels["left"] == L` exactly.
- **max rule (F, shipped)** — `forward = clip(max(L, R) * FORWARD_GAIN, 0, 1)`, `FORWARD_GAIN = 4.0`.
  This is what `descending.py` computes today and what the Phase 3 experiment used.
- **mean rule (F′, proposed)** — `forward = clip(0.5 * (L + R) * FORWARD_GAIN, 0, 1)`.
- **U(t)** — pre-clip unilaterality `|L - R| / (L + R)`. Undefined when `L + R == 0`.
- **Contamination fraction** — `U / (1 + U)`: the share of the pre-clip max-rule forward command
  attributable to bilateral imbalance rather than to bilateral drive. Because `U ∈ [0, 1]`, this
  quantity lies in **[0, 0.5]**; 0 means `L == R` (pure bilateral drive), 0.5 means one side is
  silent (pure re-labelling of unilateral activity).
- **SpikeWindow** — the diagnostic-local snapshot-and-difference helper that turns cumulative
  `engine.spike_counts` into a per-window count array.
- **Counting mode WIN / CUM** — per-window counts (correct) versus cumulative counts (the shipped
  defect, reproduced only as a labelled artifact arm).
- **Category** — the mandatory provenance label on every reported number: `REAL_CONNECTOME`,
  `ENGINEERING_ABSTRACTION`, or `READOUT_ARTIFACT`.

## Bug Details

### Bug Condition

The bug manifests when the Phase 3B diagnostic produces, or fails to produce, a measurement window.
Either the file does not exist and no window is produced at all (1.1–1.8: every adapter call in the
draft targets a signature the real modules do not have), or a window is produced whose DN rate was
divided by a window it was not counted over (1.10–1.13), whose headline metric is a monotone
function of the same `|L - R|` on both axes (1.14–1.15), whose between-rule delta was taken after
`np.clip` (1.16), whose readout window is too short to resolve a single reconstructed neuron
(1.18–1.20), whose sensory contrast is below the Poisson noise floor and unreported (1.21–1.23),
whose cross-run behavioural metrics are presented as an isolated interface effect (1.24–1.25), or
whose numbers carry no provenance category (1.26–1.27).

**Formal Specification:**

```
FUNCTION isBugCondition(input)
  INPUT:  input of type DiagnosticWindow
          { spike_count_array, count_interval_ms, window_ms,
            n_neurons_per_side, L, R, reported_metrics,
            sensory_contrast_frac, noise_floor_frac,
            driving_rule, counting_mode, categories }
  OUTPUT: boolean

  RETURN NOT diagnosticModuleExists()

         OR anyAdapterCallRejectedByRealAPI(input)

         OR input.count_interval_ms != input.window_ms

         OR headlineMetricIs("corr(forward_delta, unilaterality)", input.reported_metrics)

         OR betweenRuleDeltaTakenAfterClip(input.reported_metrics)

         OR NOT reports(input.reported_metrics, "U_preclip")
         OR NOT reports(input.reported_metrics, "contamination_preclip")
         OR NOT reports(input.reported_metrics, "clip_fraction_max_rule")
         OR NOT reports(input.reported_metrics, "clip_fraction_mean_rule")

         OR rateQuantumHz(input.n_neurons_per_side, input.window_ms) > 20.0

         OR NOT reports(input.reported_metrics, "sensory_contrast_frac")
         OR NOT reports(input.reported_metrics, "poisson_noise_floor_frac")
         OR (input.sensory_contrast_frac < input.noise_floor_frac
             AND NOT flagged(input.reported_metrics, "closed_loop_uninterpretable"))

         OR crossRunMetricsPresentedAsControlled(input.reported_metrics)

         OR EXISTS m IN input.reported_metrics
              WHERE input.categories[m] NOT IN
                    {REAL_CONNECTOME, ENGINEERING_ABSTRACTION, READOUT_ARTIFACT}
END FUNCTION
```

### Examples

All figures below are worked illustrations of what the diagnostic will compute at runtime from the
**shipped** Phase 3 configuration. None of them is hard-coded into the diagnostic.

- **No diagnostic at all.** `run_sandbox_phase3b_true_separated_forward.py` is 0 bytes. Expected:
  a module that runs to completion against the real modules and writes a CSV plus a categorised
  console report. Actual: nothing to run.

- **Cumulative counts inflate every DN rate.** `engine.spike_counts` is cumulative since `reset()`.
  At window 100 of a 10 s run the count in the numerator spans 10 000 ms while the denominator is
  100 ms, a 100× inflation. `act = hz / (hz + 60)` then sits at the ceiling for every tracked DN, and
  `forward = clip(max(L, R) * 4)` reaches the 1.0 clip as soon as mean per-side activation reaches
  0.25, i.e. 20 Hz — two spikes per neuron per 100 ms window. Expected: rates that are a function of
  neural activity only. Actual (shipped scripts): a forward channel pinned at 1.0 for essentially
  the whole run.

- **The "mean forward = 0.989, invariant across steering gains" observation has two independent
  non-neural explanations, and the diagnostic must separate them.** First, a mean of 0.989 over 100
  windows implies roughly 99 windows sitting exactly at the 1.0 clip — total saturation, which is
  what cumulative counting predicts. Second, the invariance across gains is not evidence of anything:
  `run_sandbox_steering_gain_sweep.py` drives the brain with a **fixed** symmetric sensory rate,
  builds a fresh `LIFEngine(connectome, seed=3000)` per gain, and applies `STEERING_GAINS =
  (1.0, 0.5, 0.25)` only to the `left`/`right` commands handed to the body. The body never feeds back
  into the sensory input, so the DN spike trace is **bit-identical** across the three gains and mean
  forward is identical by construction. Expected: the magnitude re-derived under per-window counting
  and the invariance attributed to the open-loop design. Actual: the pair reported as a neural result.

- **Headline metric is a tautology.** Pre-clip, `max(L,R)*4 - 0.5*(L+R)*4 = 2*|L - R|`, while
  `U = |L - R| / (L + R)`. With `(L + R)` roughly stable both axes are monotone in `|L - R|`, so the
  correlation is positive by construction and no dataset could produce a negative one. Expected: a
  metric with a reachable null. `U` and `U/(1+U)` both go to 0 exactly when `L == R`.

- **Clipping blinds the comparison.** With `FORWARD_GAIN = 4.0` the max rule clips at
  `max(L,R) ≥ 0.25` and the mean rule at `L + R ≥ 0.5`. Once both clip, the post-clip delta is
  exactly 0 and is indistinguishable from the two rules agreeing. Expected: per-rule clip fractions
  and an explicit "no information in this window" flag.

- **1 ms window quantizes a single neuron.** The loaded connectome has DNa01 = 1 left + 1 right,
  DNa02 = 1 left + 1 right, MDN = 2 left + 2 right. At `dt_ms = 1.0` a single-neuron rate is 0 or
  1000 Hz, so `act` is 0 or 0.943. At the 100 ms `SIMULATION_CHUNK_MS` convention the rate quantum is
  10 Hz and `act` takes values 0, 0.1429, 0.25, 0.3333, 0.4, 0.4545, … — coarse, but resolvable and
  reportable. Expected: a 50–100 ms window with the quantum stated.

- **Shipped antenna geometry inflates bilateral contrast 171×.** At the start position the distance
  to food is 600 mm, so the local per-ORN rate is `150 · exp(-600/250) = 13.61 Hz` and the radial
  gradient is `13.61/250 = 0.0544 Hz/mm`. The antenna span is `2 × 60 = 120 mm`, giving a maximum
  bilateral difference of `6.53 Hz`, i.e. **48.0 %** of the local rate. With a biological span of
  0.7 mm the same field gives `0.038 Hz`, i.e. **0.28 %**. Expected: a biologically plausible span
  (3.8), reported rather than tuned. Actual: a span 171× too large, which is the only reason a
  bilateral signal exists at all in the shipped experiment.

- **Below the Poisson noise floor with biological antennae.** Over a 100 ms window with 129 ORNs per
  side at 13.61 Hz, the expected per-side event count is 175.5, so the per-side relative SD is
  **7.55 %** and the relative SD of the between-side difference is **10.67 %**. The shipped 120 mm
  span sits 4.5× above that floor; a biological 0.7 mm span sits **38× below** it. Expected: contrast,
  floor and their ratio reported, with the closed-loop result flagged uninterpretable when the ratio
  is below 1.

- **Edge case, `L + R == 0`.** Both DNa sides silent for the window. `U` and the contamination
  fraction are undefined, not zero. Expected: the window is excluded from the distributions and
  counted separately as `n_windows_no_bilateral_drive`, and both forward rules are reported as 0.

- **Edge case, `min(L, R) == 0` with the other side non-zero.** `U = 1`, contamination `= 0.5`, the
  maximum the metric can take. Expected: reported as pure re-labelling, not as a failure.

## Expected Behavior

### Preservation Requirements

**Unchanged behaviours:**

- `brain/motor/descending.py` is not edited. `DescendingReadout.channels` keeps its signature,
  its return keys and its arithmetic, and `FORWARD_GAIN = 4.0`, `BACKWARD_GAIN = 1.0`,
  `TURN_GAIN = 1.0`, `CHANNEL_HALF_MAX_HZ = 60.0` keep their values for every caller (3.7).
- `simulation/engine/lif_engine.py` is not edited. `engine.spike_counts` stays cumulative since
  `reset()` and monotone non-decreasing, and `firing_rates_hz(window_ms)` keeps its current
  behaviour (3.11).
- `fly/body/fly_body.py` is not edited. `MIN_LOCOMOTION_DRIVE = 0.12`,
  `LOCOMOTION_PERSISTENCE_MS = 250.0`, the escape/freeze/backward/proboscis thresholds and the
  published escape latencies are untouched (3.12).
- The nine existing sandbox scripts and every file under `tests/` are not edited. Their outputs stay
  byte-identical for a given seed.
- The FlyWire FAFB v783 connectome is loaded as built: no topology edits, no added or removed
  connections, no synaptic sign changes (3.1).
- `plasticity_enabled` stays `False` and `engine.wdata` is never written (3.4).
- Only ORN indices and firing rates enter the brain. Food coordinates, bearing and distance stay in
  the environment, used for scoring only (3.3).
- `FlyBody` receives only motor channels, escape laterality and proboscis drive (3.6).
- The DNa lateral asymmetry is logged uncorrected; no rebalancing, offset or compensation is applied
  (3.5).
- No hand-written navigation controller exists anywhere in the loop, including the temporal-sampling
  observation described under Fix Implementation, which is measured and reported but never fed back
  (3.2).
- Determinism: same seed and configuration produce the same result, with deterministic
  `np.sort(...)[:129]` ORN selection and an explicitly seeded engine RNG (3.10).

**Scope:**

Every caller other than the new diagnostic is completely unaffected by this fix. This includes:

- all nine existing sandbox scripts, which continue to pass cumulative counts and continue to
  produce their current (inflated, now documented) numbers;
- the existing test suite under `tests/`;
- the web UI and any other importer of `DescendingReadout`, `LIFEngine` or `FlyBody`;
- the built connectome artifacts under `data/`.

The correct forward behaviour for buggy inputs is specified in Correctness Properties, Property 1.

## Hypothesized Root Cause

The bug is a design-level failure of the diagnostic, not a single wrong line. Seven causes are
hypothesised, and the exploratory phase confirms or refutes each.

1. **Adapter written from memory rather than from the shipped call sites.** Every one of 1.1–1.8 is
   a signature guess: `LIFEngine(seed=...)` without the positional connectome, `set_orn_drive`,
   `step(dt_ms)`, `DescendingReadout(engine)`, `channels()` with no arguments,
   `body.reset(x=..., y=...)`, `body.step(...)`, `body.x`. The working call pattern already exists in
   `run_sandbox_separated_four_channel_clean.py`. Root cause: the diagnostic did not start from a
   known-good call site.

2. **A cumulative counter paired with a window parameter, with no guard.** `engine.spike_counts` is
   documented as cumulative, yet `channels(spike_counts, window_ms)` invites a per-window array and
   silently accepts a cumulative one. Nothing in either API detects the mismatch, so the pattern
   propagated by copy-paste into all nine scripts. Root cause: an API pairing that is wrong by
   default and cannot fail loudly.

3. **A headline metric chosen for narrative shape rather than for a reachable null.** Correlating the
   between-rule delta against unilaterality correlates two monotone functions of the same `|L - R|`.
   Root cause: the algebraic identity `max(L,R) = (L+R)/2 + |L-R|/2` was used to motivate the
   question but not carried through to the metric.

4. **Comparison taken downstream of the saturating non-linearity.** `act = hz/(hz+60)` compresses,
   then `FORWARD_GAIN = 4.0` amplifies, then `np.clip(..., 0, 1)` truncates. Comparing the two rules
   post-clip destroys the very difference being measured. Root cause: comparing outputs instead of
   pre-clip arguments.

5. **Window sized from timestep intuition instead of from population size.** A 1 ms window is
   reasonable for `params.dt = 0.1 ms` integration and meaningless for a single reconstructed neuron.
   Root cause: DNa01/DNa02 having one neuron per side in v783 — a real connectome fact — was not
   propagated into the measurement design.

6. **Environment geometry scaled to the display world instead of to the fly, then compensated at the
   antenna.** The world is 1000 × 700 mm and the odor falloff is 250 mm, both arena-scale. A ~3 mm
   fly has an antenna span of order 0.3–0.4 mm, which yields 0.28 % contrast in that field. Setting
   `ANTENNA_OFFSET_MM = 60.0` restores a visible 48 % contrast by making the fly's head 120 mm wide.
   Root cause: a sensory-geometry mismatch closed by inflating the geometry rather than by rescaling
   the field, so the resulting steering signal is an artifact of the environment model.

7. **A closed loop reasoned about as if it were open.** The "both formulas evaluated on the SAME
   neural trace" claim is true of `run_sandbox_steering_gain_sweep.py`, which is open-loop with fixed
   sensory drive, and false of the Phase 3 closed loop, where the driving rule changes the trajectory,
   hence the sampled odor, hence the sensory input. Root cause: a valid claim carried across from a
   different experiment.

## Correctness Properties

Property 1: Bug Condition - Falsifiable, Window-Correct, Categorised Forward Audit

_For any_ diagnostic window where the bug condition holds (`isBugCondition` returns true), the fixed
diagnostic SHALL produce a window whose DN rates are computed from a spike count measured over
exactly the window passed as `window_ms`; whose headline quantities are the pre-clip
`U = |L - R| / (L + R)` and the contamination fraction `U / (1 + U)`, both computed before any
`clip`, with `L + R == 0` reported as undefined rather than zero; whose contamination fraction lies
in `[0, 0.5]`, equals 0 exactly when `L == R`, and equals 0.5 exactly when `min(L, R) == 0`; which
records a per-rule clip flag for the max rule and the mean rule and an explicit no-information flag
when both clipped; which records the per-side spike count, the window, the rate quantum and the
per-type per-side neuron count alongside every activation; which records the achieved sensory
contrast, the Poisson noise floor for the window and per-side population actually used, and their
ratio, flagging the closed-loop result uninterpretable when the ratio is below 1; and every number of
which carries exactly one provenance category from `{REAL_CONNECTOME, ENGINEERING_ABSTRACTION,
READOUT_ARTIFACT}`.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27**

Property 2: Preservation - Shared Modules and Existing Callers Unchanged

_For any_ input where the bug condition does NOT hold (`isBugCondition` returns false) — that is, any
call into the shared modules made by anything other than the new diagnostic — the fixed code SHALL
produce the same result as the original code, preserving `DescendingReadout.channels` outputs for
identical `(spike_counts, window_ms)` arguments, the values of `FORWARD_GAIN`, `BACKWARD_GAIN`,
`TURN_GAIN` and `CHANNEL_HALF_MAX_HZ`, the cumulative and monotone non-decreasing semantics of
`engine.spike_counts` across any number of `SpikeWindow` snapshots, `FlyBody` kinematics and
thresholds, the loaded FlyWire topology and synaptic signs, `plasticity_enabled == False` with
`wdata` unmodified, and the byte-identical output of the nine existing sandbox scripts at a fixed
seed.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12**

Property 3: Non-Invasive Re-Derivation Fidelity

_For any_ non-negative spike-count vector and positive window, the max rule recomputed inside the
diagnostic from `channels["left"]` and `channels["right"]` SHALL equal `channels["forward"]` exactly,
so that the audit is demonstrably auditing the shipped rule rather than a copy of it, and any future
drift in the shared `channels()` contract fails the diagnostic loudly instead of silently changing
the audited quantity.

**Validates: Requirements 2.6, 3.7**

Property 4: Windowed Counting Is Elapsed-Time Independent

_For any_ pair of windows drawn from the same run under a constant sensory drive, the per-window DN
rate produced by `SpikeWindow` SHALL have no systematic dependence on elapsed simulation time, while
the same quantity computed from cumulative counts SHALL grow monotonically with elapsed time; and the
per-window count SHALL be non-negative and equal to the difference of the cumulative counts at the
window boundaries.

**Validates: Requirements 2.10, 2.11, 2.12, 3.11**

## Fix Implementation

### Scope decision: shared fix versus documented finding

The cumulative-count defect affects nine shipped scripts. Two options:

- **Option A — fix the nine scripts now.** Introduce a shared windowed-count utility and migrate
  `run_sandbox.py`, `run_sandbox_phase2.py`, `run_sandbox_phase3_closed_loop_odor.py`,
  `run_sandbox_separated_four_channel_clean.py`, `run_sandbox_separated_locomotion_steering.py`,
  `run_sandbox_steering_gain_sweep.py`, `run_sandbox_bilateral_symmetry_diagnostic.py`,
  `run_sandbox_descending_readout_diagnostic.py` and `run_sandbox_forward_backward_only.py`.
  Upside: no inflated numbers remain in the tree. Downside: it changes the output of every shipped
  experiment, including results cited in `README.md` and `BIOLOGICAL_ASSUMPTIONS.md` and two
  committed CSVs; each script needs re-running and re-validating; and it merges a repair of nine
  experiments into a spec whose job is to answer Q0. The audit would be gated behind nine
  re-validations and would probably not land.

- **Option B — scope the fix to the new diagnostic plus a documented finding.** The new diagnostic
  measures correctly; the nine scripts are enumerated in its output as affected, with the mechanism
  and the direction of the error stated, and any prior quantity this spec relies on as evidence is
  re-derived inside the diagnostic under per-window counting (2.13). Upside: zero blast radius, the
  audit lands, and the finding is on record with a reproduction arm that demonstrates the mechanism.
  Downside: the nine scripts keep producing inflated numbers until a follow-up spec migrates them.

**Decision: Option B.** The tradeoff is deliberate — this spec is an audit, and an audit that
rewrites the nine artifacts it is auditing cannot be checked against them. The diagnostic includes a
labelled CUM arm that reproduces the defect, which is stronger evidence for a follow-up migration
spec than a silent rewrite would be. Requirement 3.11 holds under either option; Option B satisfies
it by construction.

### Scope decision: engine-level accessor versus diagnostic-local windowing

`engine.firing_rates_hz(window_ms)` has the same latent trap as the sandbox scripts: it divides
cumulative `spike_counts` by a caller-supplied window. The new measurement path must not route
through it.

**Decision: keep snapshot-and-difference local to the diagnostic.** An engine-level accessor would
have to hold a snapshot inside `LIFEngine`, adding mutable state to a class shared by nine scripts
and five test modules — precisely the risk 3.11 warns about. Worse, it would place a second,
differently-behaving rate accessor next to `firing_rates_hz`, so the next caller has two plausible
methods and no way to tell which is correct. Keeping the window local means the audit changes zero
lines outside the new file. The cost is that the helper is not immediately reusable by the other nine
scripts; that cost is accepted and recorded as part of the Option B finding, which names
`SpikeWindow` as the migration path.

Alternative considered and rejected: counting from the per-timestep index arrays returned by
`engine.run(duration_ms)`. It is exact and needs no snapshot, but it allocates 1000 arrays per 100 ms
chunk and a `bincount` over ~140 k neurons per chunk. The difference of two cumulative snapshots is
arithmetically identical and far cheaper.

### Changes required

**File:** `simulation/environment/run_sandbox_phase3b_true_separated_forward.py` (new, currently 0
bytes). This is the only file that changes, apart from `requirements/requirements-dev.txt` for the
PBT dependency and a new test module.

The script is headless — no `WorldViewer`, matching `run_sandbox_bilateral_symmetry_diagnostic.py`
and `run_sandbox_descending_readout_diagnostic.py`. That keeps it importable from tests with no
`pygame` dependency, and lets the pure helpers be property-tested without loading the connectome.
`World` is still imported for the 1000 × 700 mm geometry. Entry point: `python -m
simulation.environment.run_sandbox_phase3b_true_separated_forward`.

1. **Adapter layer against the real APIs (2.1–2.9).** Copy the call pattern from
   `run_sandbox_separated_four_channel_clean.py`: `connectome = load_connectome()`, then
   `LIFEngine(connectome, seed=SEED)` with the seed printed in the header;
   `DescendingReadout(connectome)`; ORN resolution via `BY_KEY["odor_vinegar"]` →
   `connectome.by_cell_types(modality.cell_types)` → split on the `side` column →
   `np.sort(idx)[:129]` per side, with the per-side counts actually used printed and no synthetic
   neurons created; `engine.set_poisson(all_idx, rates)` per window and `engine.clear_poisson()` on
   teardown; `engine.run(SIMULATION_CHUNK_MS)` with no argument ever passed to `step()`;
   `readout.channels(window_counts, SIMULATION_CHUNK_MS)`; body state read and written through
   `body.state.x_mm` / `y_mm` / `z_mm` / `heading_deg` / `speed_mm_s`; and
   `body.update(dt_ms=..., channels=..., t_ms=..., escape_laterality=..., proboscis_drive=...)` with
   `t_ms` the true accumulated simulation time so the 250 ms / 0.12 locomotion hysteresis behaves as
   it did in the Phase 3 run.

2. **`SpikeWindow` (2.10, 3.11).** A small class holding one `np.ndarray` snapshot.
   `begin(engine)` stores `engine.spike_counts.copy()`; `end(engine)` returns
   `engine.spike_counts.astype(np.int64) - snapshot`, asserting the result is non-negative so a
   forgotten `begin()` or an out-of-order call fails loudly rather than producing a plausible number.
   The engine is never mutated, so `engine.spike_counts` stays cumulative for every other caller.
   Both `escape_laterality` and `proboscis_drive` receive the same windowed array, for internal
   consistency.

3. **Both forward rules recomputed in the diagnostic (2.24, 3.7).** Read `L = channels["left"]` and
   `R = channels["right"]`, import `FORWARD_GAIN` and `CHANNEL_HALF_MAX_HZ` from
   `brain.motor.descending` rather than re-declaring them, and compute
   `fwd_max_preclip = FORWARD_GAIN * max(L, R)`, `fwd_mean_preclip = FORWARD_GAIN * 0.5 * (L + R)`,
   with `clip01` applied afterwards and separately. Assert
   `abs(clip01(fwd_max_preclip) - channels["forward"]) < 1e-12` every window (Property 3). The max
   rule drives the body in the primary arm, so the audited trajectory is the shipped one; the mean
   rule is shadowed from the same counts. `descending.py` is not modified.

4. **Headline metrics (2.14–2.17).** `unilaterality(L, R)` returns `None` when `L + R == 0`, else
   `abs(L - R) / (L + R)`. `contamination_fraction(u)` returns `u / (1 + u)`. Both are computed
   pre-clip. Per window, record `fwd_max_clipped`, `fwd_mean_clipped` and `both_clipped`; when
   `both_clipped` is true, mark the window `comparison_uninformative`. Report the full `U` time
   series in the CSV plus mean, median, p10, p90, min, max, and the count of undefined windows. The
   correlation `corr(delta, U)` is printed only under an `ALGEBRAICALLY DETERMINED — NOT EVIDENCE`
   heading that states the `max(L,R) = (L+R)/2 + |L-R|/2` identity.

5. **Pre-registered verdict thresholds (2.17).** Fixed in this design so they cannot be chosen after
   seeing the data, and stated on the `[0, 0.5]` contamination scale: median contamination `< 0.05`
   over informative windows → the observed forward locomotion was a genuine bilateral command;
   `0.05` to `0.25` → mixed, with the imbalance share quantified; `> 0.25` → predominantly a
   re-labelling of unilateral steering activity; `0.5` → purely unilateral. The verdict is reported
   as `INDETERMINATE` when the informative-window fraction is below 0.5 or when the closed-loop arm
   is flagged sensory-uninterpretable.

6. **Window sizing and quantization disclosure (2.18–2.20).** `SIMULATION_CHUNK_MS = 100.0` for both
   the readout window and the sensorimotor loop cadence. Print the window, the rate quantum
   (`1000 / window_ms / n_neurons` per type and side — 10 Hz for DNa01/DNa02, 5 Hz for MDN), and the
   per-type per-side neuron counts read from `readout.tracked[(ctype, side)].size`. Every reported
   activation is accompanied by its raw spike count and window. Any bilateral DNa asymmetry is
   labelled a comparison of two individual reconstructed neurons, never population lateralization,
   and is logged uncorrected.

7. **Sensory contrast and noise floor, computed not assumed (2.21–2.23).** Per window, from the
   environment side only: local per-ORN rate `peak · exp(-r/λ)`; radial gradient `rate/λ`; bilateral
   difference `gradient · 2 · half_sep · |sin θ|` where θ is the angle between heading and food
   bearing, plus the exact two-antenna difference actually delivered; contrast as a fraction of the
   local rate; Poisson relative SD per side `1/sqrt(N · rate · T)`; relative SD of the between-side
   difference `sqrt(2)/sqrt(N · rate · T)`; and the contrast-to-floor ratio. When the ratio is below
   1, the window and the run summary are flagged `closed_loop_uninterpretable`. The diagnostic also
   solves numerically for the falloff length that would lift a biological span above the floor at the
   sampled radius, and reports whether a feasible `(λ, r)` region exists — for the shipped geometry
   at `r = 600 mm` it does not, because shortening λ collapses the local rate faster than it steepens
   the gradient.

8. **The antenna-offset finding (3.8).** `ANTENNA_OFFSET_MM = 60.0` is reported as a distinct finding
   with its own measured consequence: the shipped span is `120 mm` against a biological span of
   ~`0.7 mm`, a `171×` inflation, and it moves the bilateral contrast from `0.28 %` (38× below the
   noise floor) to `48.0 %` (4.5× above it). The conclusion is stated plainly: any bilateral steering
   signal in the shipped Phase 3 experiment is an artifact of the environment model, categorised
   `ENGINEERING_ABSTRACTION`, and it is not evidence about the connectome. The antenna separation is
   never enlarged as a remedy; the biological value `ANTENNA_HALF_SEPARATION_MM = 0.35` is used in the
   bio-geometry arm and the field is rescaled instead, with the rescaling labelled an engineering
   abstraction of the environment.

9. **Temporal-sampling observation, measured but never fed back (3.2).** The diagnostic also reports
   what a temporal comparison would offer, because it bears directly on whether the closed loop is
   interpretable at all: at 25 mm/s the fly covers 2.5 mm per 100 ms window, giving 1.0 % contrast
   against a 7.55 % single-window floor, while a 1 s baseline gives 10 % contrast against a 2.39 %
   floor — 4.2× above. This is reported as a property of the environment model only. No temporal
   comparison, gradient memory or navigation rule is implemented or fed into the brain.

10. **Five arms, one seed (2.11, 2.13, 2.24, 2.25).** All arms use `SEED = 3000`, matching the gain
    sweep, and a fresh `LIFEngine` per arm.

    | Arm | Config | Counting | Driving rule | Role |
    |---|---|---|---|---|
    | 0 | open loop, fixed symmetric ORN drive | WIN | max | null distribution of `U` from network and Poisson noise alone |
    | 1 | shipped Phase 3 (λ=250, half-sep 60) | WIN | max | **primary controlled measurement**; mean rule shadowed from the same counts |
    | 2 | shipped Phase 3 | WIN | mean | cross-run, descriptive only |
    | 3 | shipped Phase 3 | CUM | max | labelled reproduction of the defect; re-derives the 0.989 observation |
    | 4 | biological half-sep 0.35, rescaled fly-scale field | WIN | max | interpretable closed-loop arm; mean rule shadowed |

    Arm 0 matters because without it any non-zero `U` looks meaningful; it bounds how much
    unilaterality arises from stochasticity alone. Arm 3 is the only place cumulative counts appear;
    every number it produces is categorised `READOUT_ARTIFACT` and it is never used as a measurement
    of neural activity, so 2.10 holds for all reported measurements. Arm 4's field parameters are
    chosen by the feasibility solve of change 7 — the illustrative point λ = 3 mm at r = 6 mm gives
    23.3 % contrast against an 8.74 % floor, a ratio of 2.67 — and are printed as an engineering
    abstraction. Within-run comparisons are labelled controlled; the arm-1-versus-arm-2 behavioural
    metrics (`end_dist`, `min_dist`, `heading_range`) are printed under a `DESCRIPTIVE ONLY —
    CONFOUNDED` heading stating that changing the driving rule changes the trajectory, hence the
    sampled odor, hence the sensory input. Arms are independently runnable so the default 10 s × 5
    cost can be trimmed via constants at module top.

11. **Category attribution enforced structurally (2.26–2.27).** A `report(name, value, category,
    note="")` function takes `category` as a required positional argument drawn from a three-member
    constant set and raises on anything else. Every printed metric goes through it, so a number
    cannot reach the output without a category. The report header states the seed, the connectome
    identity (`connectome.dataset`), the window, the per-side ORN counts, the per-type per-side DN
    counts, and the standing scope limit that FAFB v783 is a brain connectome with the VNC and
    leg/wing motor neurons absent, so nothing downstream of the DN stage is a biological claim (3.9).
    A fixed footer restates that a forward command derived from DNa01/DNa02 is an engineering
    interface at the brain-to-body boundary and is not a claim that DNa01/DNa02 are forward-walking
    neurons (2.27).

12. **The Option B finding block (2.13).** A dedicated section names all nine affected scripts,
    states the mechanism (cumulative numerator, per-chunk denominator), the direction (rates and
    channel magnitudes inflated, growing with elapsed time), the consequence (motor channels saturate
    as a function of wall-clock time), and the two independent non-neural explanations for the
    "mean forward = 0.989 invariant across three steering gains" claim: saturation under cumulative
    counting, and the fact that the gain sweep is open-loop with a fixed sensory drive and a fixed
    seed, so its DN trace is bit-identical across gains and the invariance carries no information.
    The re-derived per-window mean forward from arm 1 is printed next to the 0.989 figure.

13. **CSV output.** `phase3b_forward_readout_diagnostic.csv` in the working directory, following the
    existing diagnostic convention. One row per window per arm: `arm`, `config`, `counting_mode`,
    `driving_rule`, `seed`, `t_ms`, `window_ms`; per-type per-side spike counts and Hz for DNa01,
    DNa02, MDN; `L`, `R`, `u_preclip`, `contamination_preclip`, `u_defined`; `fwd_max_preclip`,
    `fwd_mean_preclip`, `fwd_max`, `fwd_mean`, `fwd_max_clipped`, `fwd_mean_clipped`, `both_clipped`,
    `delta_preclip`, `delta_postclip`; `conc_left`, `conc_right`, `local_rate_hz`, `gradient_hz_mm`,
    `bilateral_delta_hz`, `contrast_frac`, `noise_floor_frac`, `contrast_over_noise`,
    `sensory_interpretable`; `speed_mm_s`, `x_mm`, `y_mm`, `heading_deg`, `behaviour`,
    `distance_to_food_mm`, `bearing_to_food_deg`. The per-window `L + R` trace and clip flags are what
    2.11 is checked against: a drift statistic (slope of `L + R` against `t`, plus clip fraction in
    the first versus last decile) is computed for arms 1 and 3 and reported side by side.

14. **Pure helpers, separately testable.** `SpikeWindow`, `unilaterality`, `contamination_fraction`,
    `forward_max_preclip`, `forward_mean_preclip`, `clip01`, `odor_concentration`,
    `antenna_positions`, `poisson_relative_sd`, `rate_quantum_hz`, `sensory_contrast_report`,
    `required_falloff_mm` and `report` contain no I/O and no connectome dependency, so the property
    tests exercise them without a brain run.

## Testing Strategy

### Validation approach

Two phases. First, surface counterexamples on the unfixed state — which here means both the absent
diagnostic and the shipped measurement path that the diagnostic reproduces in arm 3. Then verify the
fix answers Q0 with a falsifiable metric and leaves every other caller untouched.

### Exploratory Bug Condition Checking

**Goal:** surface counterexamples that demonstrate the bug BEFORE implementing the fix, and confirm
or refute the seven hypothesised root causes. If a hypothesis is refuted, re-hypothesise before
writing the diagnostic.

**Test plan:** write short throwaway probes against the real modules and against the shipped scripts.
Run them on the unfixed state to observe the failures directly.

**Test cases:**

1. **Adapter signature probe** — call each draft signature from 1.1–1.8 against the real modules and
   record the exact exception (will fail on unfixed code; confirms cause 1).
2. **Cumulative growth probe** — construct `LIFEngine(connectome, seed=3000)`, apply a constant
   symmetric ORN drive, and print `readout.rates(engine.spike_counts, 100.0)` for 100 consecutive
   100 ms chunks. Expect monotone growth toward saturation (will fail a "rate is stationary under
   constant drive" assertion on unfixed code; confirms cause 2).
3. **Windowed comparison probe** — same run, printing both `channels(cumulative, 100)` and
   `channels(window_diff, 100)`. Expect divergence growing with elapsed time and a stationary
   windowed series (confirms cause 2 and the fix direction).
4. **Tautology probe** — over the recorded `(L, R)` series, compute `corr(delta_preclip, U)` and
   verify it is positive; then verify algebraically that `delta_preclip == 2·|L − R|` to floating
   point. A positive correlation with an exact algebraic identity confirms cause 3.
5. **Clip-blindness probe** — count windows where `both_clipped` is true and the post-clip delta is
   exactly 0 while the pre-clip delta is non-zero. Expect a large fraction under cumulative counting
   (confirms cause 4).
6. **Quantization probe** — print `readout.tracked[(ctype, side)].size` for DNa01, DNa02, MDN, and
   the distinct `act` values achievable at `window_ms = 1.0` versus `100.0`. Expect {0, 0.943} at
   1 ms (confirms cause 5).
7. **Sensory contrast probe** — compute local rate, gradient, bilateral difference and Poisson floor
   at the shipped start position for spans 120 mm and 0.7 mm. Expect 48.0 % versus 0.28 % against a
   10.67 % floor (confirms cause 6).
8. **Open-loop identity probe** — run `run_sandbox_steering_gain_sweep.py`'s `run_one_gain` at gains
   1.0, 0.5 and 0.25 with seed 3000 and compare the recorded forward series. Expect bit-identical
   series, which **refutes** any reading of the invariance as a neural or saturation result and
   confirms cause 7's family of errors. If the series differ, cause 7 needs re-hypothesising.
9. **Edge-case probe** — evaluate `U` and the contamination fraction at `L = R = 0`, at
   `L > 0, R = 0`, and at `L = R > 0`. Expect undefined, 0.5 and 0.0 respectively.

**Expected counterexamples:**

- Every draft adapter call raises; the module cannot execute at all.
- Under constant symmetric drive, cumulative-count DN rates grow ~100× across a 10 s run and the
  forward channel pins at 1.0 within the first few windows.
- Possible causes: cumulative numerator with per-chunk denominator; `act` saturating non-linearity
  followed by `FORWARD_GAIN = 4.0` and a hard clip; a 1 ms window on a one-neuron-per-side
  population; an arena-scale odor field compensated by a 171× antenna span.

### Fix Checking

**Goal:** verify that for all inputs where the bug condition holds, the fixed diagnostic produces the
expected behaviour.

**Pseudocode:**

```
FOR ALL input WHERE isBugCondition(input) DO
  result := phase3bDiagnostic(input)
  ASSERT expectedBehavior(result)
END FOR

WHERE expectedBehavior(result) =
      result.count_interval_ms = result.window_ms
  AND result.u_preclip IS DEFINED OR result.l_plus_r = 0
  AND result.contamination IN [0.0, 0.5]
  AND (result.l = result.r) IMPLIES result.contamination = 0.0
  AND (min(result.l, result.r) = 0 AND result.l_plus_r > 0)
        IMPLIES result.contamination = 0.5
  AND result.clip_flag_max IS PRESENT
  AND result.clip_flag_mean IS PRESENT
  AND result.both_clipped IMPLIES result.comparison_uninformative
  AND clip01(result.fwd_max_preclip) = result.channels_forward
  AND result.contrast_frac IS PRESENT
  AND result.noise_floor_frac IS PRESENT
  AND (result.contrast_frac < result.noise_floor_frac)
        IMPLIES result.flagged_uninterpretable
  AND FOR ALL m IN result.metrics:
        result.category[m] IN {REAL_CONNECTOME,
                              ENGINEERING_ABSTRACTION,
                              READOUT_ARTIFACT}
```

### Preservation Checking

**Goal:** verify that for all inputs where the bug condition does NOT hold, the fixed code produces
the same result as the original.

**Pseudocode:**

```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT channels_original(input.spike_counts, input.window_ms)
       = channels_fixed(input.spike_counts, input.window_ms)
  ASSERT (FORWARD_GAIN, BACKWARD_GAIN, TURN_GAIN, CHANNEL_HALF_MAX_HZ)
       = (4.0, 1.0, 1.0, 60.0)
  ASSERT spike_counts_after_SpikeWindow_use
       = spike_counts_without_SpikeWindow_use
  ASSERT isMonotoneNonDecreasing(engine.spike_counts)
  ASSERT engine.plasticity_enabled = FALSE
  ASSERT wdata_after = wdata_before
END FOR
```

**Testing approach:** property-based testing is recommended for preservation checking because it
generates many spike-count vectors across the input domain automatically, catches edge cases manual
tests miss (all-zero counts, a single spike, counts large enough to saturate every channel), and
gives a strong guarantee that behaviour is unchanged for all non-buggy inputs. `hypothesis` is not
currently installed; add it pinned to `requirements/requirements-dev.txt` (`hypothesis>=6.100,<7`).
The properties under test are pure functions of numpy arrays, so no connectome load is needed and the
tests run in milliseconds.

**Test plan:** observe behaviour on the unfixed code first for the non-bug inputs — record
`channels()` outputs for a fixed set of spike-count vectors and the current gain constants before any
work starts — then write property tests capturing exactly that behaviour.

**Test cases:**

1. **`channels()` invariance** — observe `channels(counts, window)` outputs on the unfixed tree for
   randomly generated non-negative count vectors, freeze them as a reference, then assert the fixed
   tree reproduces them exactly.
2. **Gain-constant invariance** — observe `FORWARD_GAIN`, `BACKWARD_GAIN`, `TURN_GAIN`,
   `CHANNEL_HALF_MAX_HZ` on the unfixed tree, then assert they are unchanged.
3. **Cumulative-semantics invariance** — observe that `engine.spike_counts` is cumulative and
   monotone on the unfixed tree, then assert that arbitrary interleaved `SpikeWindow.begin`/`end`
   calls leave it cumulative, monotone and identical to a run with no `SpikeWindow` at all.
4. **Existing-script invariance** — run the existing suite (`python -m pytest tests/ -q`) and confirm
   it passes unchanged; confirm no file outside the new diagnostic, the new test module and
   `requirements-dev.txt` is modified (`git status --short`).
5. **`FlyBody` invariance** — observe body kinematics and thresholds on the unfixed tree, then assert
   unchanged behaviour for a fixed channel sequence with a fixed `t_ms` sequence.
6. **Topology and plasticity invariance** — assert the loaded connectome's `nnz`, synaptic signs and
   `wdata` are unchanged after a full diagnostic arm, and that `plasticity_enabled` stayed `False`.

### Unit Tests

- `SpikeWindow`: per-window count equals the difference of cumulative snapshots; raises on a missing
  or out-of-order `begin()`; returns all zeros over a zero-length window.
- `unilaterality`: returns `None` at `L + R == 0`; `0.0` at `L == R > 0`; `1.0` at `min(L, R) == 0`
  with the other side positive.
- `contamination_fraction`: `0.0` at `U = 0`; `0.5` at `U = 1`; monotone increasing in `U`.
- Forward rules: `forward_max_preclip` and `forward_mean_preclip` agree exactly when `L == R`; their
  difference equals `2·|L − R|` at `FORWARD_GAIN = 4.0`; clip flags set exactly when the pre-clip
  value exceeds 1.0.
- `rate_quantum_hz`: 10 Hz for one neuron over 100 ms; 5 Hz for two; 1000 Hz for one over 1 ms.
- `poisson_relative_sd`: 7.55 % for 129 neurons at 13.61 Hz over 100 ms; the between-side difference
  figure is `sqrt(2)` times it.
- `odor_concentration` / `antenna_positions`: `exp(-600/250)` at the shipped start radius; zero
  bilateral difference when the food is straight ahead; maximum difference when the food is abeam.
- `sensory_contrast_report`: flags `uninterpretable` exactly when contrast is below the floor;
  returns 48.0 % for a 120 mm span and 0.28 % for a 0.7 mm span in the shipped field.
- `report`: raises when the category argument is absent or outside the three-member set.
- Verdict thresholds: `genuine` / `mixed` / `re-labelling` / `indeterminate` at contamination
  medians 0.02, 0.15, 0.40 and at an informative-window fraction of 0.3.

### Property-Based Tests

- **Property 1 exploration test** — generate arbitrary non-negative DN spike-count vectors and
  windows, drive the metric pipeline, and assert every clause of `expectedBehavior`: window
  consistency, `contamination ∈ [0, 0.5]`, the `L == R ⇒ 0` and `min(L, R) == 0 ⇒ 0.5` boundaries,
  clip flags present, the both-clipped implication, and a category on every emitted metric. Expected
  to fail on the unfixed state, since the pipeline does not exist.
- **Property 2 preservation test** — generate arbitrary non-negative count vectors and positive
  windows; assert `channels()` output matches the frozen reference; assert the gain constants; assert
  `engine.spike_counts` stays cumulative and monotone across arbitrary interleavings of `SpikeWindow`
  use.
- **Property 3 fidelity test** — generate arbitrary count vectors; assert
  `clip01(forward_max_preclip(L, R)) == channels["forward"]` to within 1e-12 for every generated
  input.
- **Property 4 windowing test** — generate arbitrary monotone cumulative count sequences; assert the
  windowed rate has no time trend while the cumulative rate is monotone increasing, and that every
  per-window count is non-negative and equals the boundary difference.
- Generate random `(λ, r, span, N, window)` tuples and assert `sensory_contrast_report` is
  scale-consistent: contrast equals `span/λ` to first order, the floor scales as
  `1/sqrt(N·rate·T)`, and the `uninterpretable` flag agrees with a direct comparison of the two.

### Integration Tests

- **Short-arm smoke run** — each of the five arms at a reduced duration (1 s, 10 windows) against the
  real connectome: completes without exception, writes a CSV with the full column set, and emits a
  header with the seed, connectome identity, per-side ORN counts and per-type per-side DN counts.
- **Full-flow determinism** — two identical runs at `SEED = 3000` produce identical CSVs (3.10).
- **Arm 1 versus arm 3 attribution** — the drift statistic on `L + R` is ≈ 0 for arm 1 and strongly
  positive for arm 3, and arm 3's clip fraction in the last decile is far above arm 1's. This is the
  direct check for 2.11 and 2.12, and the re-derivation for 2.13.
- **Arm 0 null versus arm 1** — the `U` distributions are reported side by side so the closed-loop
  unilaterality can be read against the stochastic null.
- **Arm 1 versus arm 4 sensory gate** — arm 1 is flagged `closed_loop_uninterpretable` under a
  biological antenna span while arm 4 is not, and the antenna-offset finding prints the 171× and
  38×-below-floor figures.
- **Cross-run confound labelling** — the arm-1-versus-arm-2 behavioural metrics appear only under the
  `DESCRIPTIVE ONLY — CONFOUNDED` heading, and the within-run paired comparison is the only one
  labelled controlled, naming which rule held the trajectory.
- **Constraint audit** — the run asserts no navigation controller, no target direction into the
  brain, no reward, no plasticity, and that the DNa asymmetry appears uncorrected in the output.
