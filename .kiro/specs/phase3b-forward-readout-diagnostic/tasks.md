# Implementation Plan

Scope guard for every task below: only
`simulation/environment/run_sandbox_phase3b_true_separated_forward.py` (currently 0 bytes),
`tests/test_phase3b_forward_readout_diagnostic.py` (new) and `requirements/requirements-dev.txt` may
be modified. `brain/motor/descending.py`, `simulation/engine/lif_engine.py`,
`fly/body/fly_body.py`, the nine existing sandbox scripts and every existing file under `tests/`
must not be edited. The check is `git status --short`.

Environment guard: the user is on Windows 11 / PowerShell and runs experiments manually. No
full-duration arm run is launched from a blocking tool call; long runs are handed over as commands.

---

- [ ] 1. Add the pinned property-testing dependency
  - Append `hypothesis>=6.100,<7` to `requirements/requirements-dev.txt`
  - This is done first because tasks 13, 14, 15.3 and 15.5 all depend on it
  - Confirm import works: `python -c "import hypothesis; print(hypothesis.__version__)"`
  - `requirements-dev.txt` is one of the three files this spec is allowed to touch; confirm with `git status --short`
  - _Requirements: 3.7, 3.11_

## Exploratory probes (run BEFORE the diagnostic is authored)

Probes 1–9 come from design section "Exploratory Bug Condition Checking". Each is a throwaway script
run against the real modules on the unfixed tree. Write each probe outside the repository tree (or
delete it once its result is recorded) so that `git status --short` stays limited to the three
allowed files. Record every probe's numeric output — the diagnostic is authored against these
recorded numbers, not against the illustrative figures in the design.

- [ ] 2. Probe 1 — adapter signature probe
  - **Property 1: Bug Condition** - Adapter Layer Does Not Match Any Real API
  - **CRITICAL**: every call here MUST raise on the unfixed tree - the exceptions confirm the bug exists
  - **DO NOT fix the modules being probed** - `lif_engine.py`, `descending.py`, `fly_body.py` are read-only for this spec
  - Call each draft signature from 1.1–1.8 against the real modules and record the exact exception type and message: `LIFEngine(seed=seed)`, `engine.set_orn_drive(...)`, `engine.step(dt_ms)`, `DescendingReadout(engine)`, `readout.channels()`, `body.reset(x=..., y=..., heading_deg=...)`, `body.x` / `body.y` / `body.heading_deg`, `body.step(channels, dt_ms)`
  - Record the real signature actually accepted in each case, read from `run_sandbox_separated_four_channel_clean.py`
  - **EXPECTED OUTCOME**: all eight calls raise; confirms hypothesised root cause 1 (adapter written from memory, not from a shipped call site)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

- [ ] 3. Probe 2 — cumulative growth probe
  - **Property 1: Bug Condition** - Cumulative Counts Inflate Every DN Rate
  - Build `LIFEngine(connectome, seed=3000)`, apply a constant symmetric ORN drive, and print `readout.rates(engine.spike_counts, 100.0)` for 100 consecutive 100 ms chunks
  - Assert "DN rate is stationary under constant drive" - this assertion MUST FAIL
  - Record the ratio between the window-100 rate and the window-1 rate, and the window index at which `forward` first reaches the 1.0 clip
  - **EXPECTED OUTCOME**: monotone growth toward saturation; confirms hypothesised root cause 2 (cumulative numerator paired with a per-chunk denominator, with no guard)
  - _Requirements: 1.10, 1.11, 2.10, 2.11_

- [ ] 4. Probe 3 — windowed comparison probe
  - **Property 1: Bug Condition** - Windowed And Cumulative Counting Diverge With Elapsed Time
  - On the same run, print both `channels(cumulative_counts, 100.0)` and `channels(window_diff_counts, 100.0)` per chunk
  - Record the divergence as a function of elapsed time and confirm the windowed series is stationary
  - **EXPECTED OUTCOME**: divergence grows with elapsed time; confirms root cause 2 and validates the snapshot-and-difference fix direction
  - _Requirements: 1.10, 2.10, 2.11, 2.12_

- [ ] 5. Probe 4 — tautology probe
  - **Property 1: Bug Condition** - Headline Metric Is Positive By Construction
  - Over the `(L, R)` series recorded in probe 2, compute `corr(delta_preclip, U)` and record its sign
  - Verify to floating point that `delta_preclip == 2 * abs(L - R)` at `FORWARD_GAIN = 4.0`
  - **EXPECTED OUTCOME**: correlation positive, identity exact; confirms hypothesised root cause 3 (metric chosen without a reachable null)
  - _Requirements: 1.14, 1.15, 2.14, 2.15_

- [ ] 6. Probe 5 — clip-blindness probe
  - **Property 1: Bug Condition** - Post-Clip Comparison Is Blind In The Saturated Regime
  - Count windows where both rules clipped and `delta_postclip == 0.0` while `delta_preclip != 0.0`
  - Record that fraction under cumulative counting and under windowed counting
  - **EXPECTED OUTCOME**: large fraction under cumulative counting; confirms hypothesised root cause 4 (comparison taken downstream of the saturating non-linearity)
  - _Requirements: 1.16, 2.16_

- [ ] 7. Probe 6 — quantization probe
  - **Property 1: Bug Condition** - One-Neuron-Per-Side Populations At A 1 ms Window
  - Print `readout.tracked[(ctype, side)].size` for DNa01, DNa02 and MDN, both sides, from the loaded connectome
  - Enumerate the distinct achievable `act` values at `window_ms = 1.0` versus `window_ms = 100.0`
  - **EXPECTED OUTCOME**: `{0.0, 0.943}` at 1 ms versus a 10 Hz quantum at 100 ms; confirms hypothesised root cause 5 (window sized from timestep intuition, not from population size)
  - Record the actual per-type per-side counts; the diagnostic reports these, it does not hard-code them
  - _Requirements: 1.18, 1.19, 1.20, 2.18, 2.19, 2.20_

- [ ] 8. Probe 7 — sensory contrast probe
  - **Property 1: Bug Condition** - Bilateral Contrast Against The Poisson Floor
  - At the shipped Phase 3 start position, compute local per-ORN rate, radial gradient, bilateral difference and Poisson relative SD for a 120 mm span (`ANTENNA_OFFSET_MM = 60.0`) and for a 0.7 mm biological span
  - Read `ODOR_FALLOFF_MM`, `ANTENNA_OFFSET_MM`, peak rate and world geometry from the shipped `run_sandbox_phase3_closed_loop_odor.py` / `odor_vinegar` modality rather than from the design prose
  - Record contrast, floor and their ratio for both spans
  - **EXPECTED OUTCOME**: the biological span sits far below the floor while the shipped span sits above it; confirms hypothesised root cause 6 (arena-scale field compensated at the antenna)
  - _Requirements: 1.21, 1.22, 1.23, 2.21, 2.22, 2.23_

- [ ] 9. Probe 8 — open-loop identity probe (GATE on hypothesised root cause 7)
  - **Property 1: Bug Condition** - Steering-Gain Invariance Is Open-Loop By Construction
  - Run `run_sandbox_steering_gain_sweep.py`'s `run_one_gain` at gains 1.0, 0.5 and 0.25 with seed 3000 and compare the recorded forward series across gains
  - Do not edit the script; import and call it, or run it as shipped and capture its output
  - **EXPECTED OUTCOME (a)**: series are bit-identical, which refutes any reading of "mean forward = 0.989 invariant across three steering gains" as a neural or saturation result, and confirms cause 7's family of errors
  - **EXPECTED OUTCOME (b)**: if the series differ, hypothesised root cause 7 is **REFUTED**. Stop. Do not proceed to task 12. Re-hypothesise the cause of the invariance, record the new hypothesis, and add the probes it implies to this list before any further task is started
  - Record the per-gain forward series and their exact equality/inequality
  - _Requirements: 1.12, 1.13, 1.24, 2.13, 2.24, 2.25_

- [ ] 10. Probe 9 — edge-case probe
  - **Property 1: Bug Condition** - Metric Boundaries At Zero And At Full Unilaterality
  - Evaluate `U` and the contamination fraction at `L = R = 0`, at `L > 0, R = 0`, and at `L = R > 0`
  - **EXPECTED OUTCOME**: undefined (not zero), 0.5, and 0.0 respectively; fixes the boundary contract the implementation must satisfy
  - _Requirements: 2.14, 2.17_

- [ ] 11. Consolidate probe results and re-check the seven hypothesised root causes
  - Write the recorded numbers from probes 1–9 into a findings note held with this spec
  - For each of the seven hypothesised root causes in design section "Hypothesized Root Cause", mark it CONFIRMED or REFUTED against the probe evidence
  - **GATE**: if any cause is refuted — in particular cause 7 via probe 8 — re-hypothesise before authoring the diagnostic, and revise design and this task list to match. Do not carry a refuted cause into implementation
  - Confirm `git status --short` lists no probe leftovers
  - _Requirements: 2.10, 2.14, 2.17, 2.18, 2.21, 2.24, 2.26_

## Preservation baselines (captured on the UNFIXED tree, before any new code lands)

- [ ] 12. Capture preservation baselines on the unfixed tree
  - **IMPORTANT**: observation-first. Record actual behaviour now; the property tests in task 14 assert exactly what is recorded here
  - Freeze reference outputs of `DescendingReadout.channels(counts, window_ms)` over a fixed, explicitly listed set of non-negative spike-count vectors — including all-zero, a single spike, and counts large enough to saturate every channel — at a fixed `window_ms`
  - Record the four gain constants as read from `brain.motor.descending`: `FORWARD_GAIN`, `BACKWARD_GAIN`, `TURN_GAIN`, `CHANNEL_HALF_MAX_HZ`
  - Record `FlyBody` kinematics and behaviour labels for a fixed channel sequence with a fixed `t_ms` sequence, covering the `LOCOMOTION_PERSISTENCE_MS = 250.0` / `MIN_LOCOMOTION_DRIVE = 0.12` hysteresis
  - Record that `engine.spike_counts` is cumulative and monotone non-decreasing across a short run
  - Record the loaded connectome's `nnz` and synaptic-sign summary, and that `plasticity_enabled` is `False`
  - Run `python -m pytest tests/ -q` and record a green baseline
  - Store the frozen references inside the new test module as literal data, so they cannot drift
  - _Requirements: 3.1, 3.4, 3.7, 3.10, 3.11, 3.12_

- [ ] 13. Write bug condition exploration test
  - **Property 1: Bug Condition** - Falsifiable, Window-Correct, Categorised Forward Audit
  - **CRITICAL**: This test MUST FAIL on the unfixed tree — the diagnostic module is 0 bytes, so there is nothing to import. Failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: this test encodes the expected behaviour; it validates the fix when it passes in task 15.11
  - **GOAL**: surface counterexamples demonstrating the bug
  - **Scoped PBT approach**: generate arbitrary non-negative DN spike-count vectors and positive windows with `hypothesis`; for the deterministic boundary cases from probe 9, scope the property to the concrete inputs `L = R = 0`, `min(L, R) = 0`, `L = R > 0` so those failures are reproducible
  - Drive the metric pipeline in `run_sandbox_phase3b_true_separated_forward.py` and assert every clause of `expectedBehavior` from design section "Fix Checking": `count_interval_ms == window_ms`; `u_preclip` defined unless `L + R == 0`; `contamination in [0.0, 0.5]`; `L == R` implies contamination `0.0`; `min(L, R) == 0` with `L + R > 0` implies contamination `0.5`; `clip_flag_max` and `clip_flag_mean` present; `both_clipped` implies `comparison_uninformative`; `clip01(fwd_max_preclip) == channels["forward"]`; `contrast_frac` and `noise_floor_frac` present; contrast below floor implies `flagged_uninterpretable`; every emitted metric carries a category from `{REAL_CONNECTOME, ENGINEERING_ABSTRACTION, READOUT_ARTIFACT}`
  - Run on the unfixed tree
  - **EXPECTED OUTCOME**: test FAILS with an import or attribute error (this is correct — it proves the bug exists)
  - Document the counterexamples and the failure mode
  - Mark complete when the test is written, run, and the failure is documented
  - _Requirements: 2.10, 2.14, 2.16, 2.17, 2.19, 2.21, 2.22, 2.26_

- [ ] 14. Write preservation property tests (BEFORE implementing the fix)
  - **Property 2: Preservation** - Shared Modules And Existing Callers Unchanged
  - **IMPORTANT**: follow the observation-first methodology using only the baselines recorded in task 12
  - Property test: for arbitrary generated non-negative count vectors and positive windows, `channels(counts, window_ms)` matches the frozen reference from task 12 exactly
  - Property test: the four gain constants equal `(4.0, 1.0, 1.0, 60.0)`
  - Property test: across arbitrary interleavings of `SpikeWindow.begin` / `end`, `engine.spike_counts` stays cumulative, monotone non-decreasing, and identical to a run with no `SpikeWindow` use at all (this clause can only be exercised once `SpikeWindow` exists in 15.1; until then mark it skipped, not deleted)
  - Test: `FlyBody` kinematics and behaviour labels match the recorded baseline for the fixed channel and `t_ms` sequence
  - Test: connectome `nnz`, synaptic signs and `wdata` unchanged, `plasticity_enabled` still `False`
  - Test: `python -m pytest tests/ -q` green, and `git status --short` shows only the three allowed files
  - Run on the unfixed tree
  - **EXPECTED OUTCOME**: tests PASS (confirms the baseline behaviour to preserve)
  - Mark complete when the tests are written, run, and passing on the unfixed tree
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12_

## Fix

- [ ] 15. Fix for the Phase 3B forward readout diagnostic — author the missing module

  - [ ] 15.1 Implement the pure helpers (no I/O, no connectome dependency)
    - Implement `SpikeWindow` (snapshot on `begin(engine)`, `end(engine)` returns the int64 difference, asserts non-negative so a missing or out-of-order `begin()` fails loudly, never mutates the engine), `unilaterality` (`None` when `L + R == 0`), `contamination_fraction`, `forward_max_preclip`, `forward_mean_preclip`, `clip01`, `odor_concentration`, `antenna_positions`, `poisson_relative_sd`, `rate_quantum_hz`, `sensory_contrast_report`, `required_falloff_mm`, and `report(name, value, category, note="")` with `category` a required positional argument drawn from a three-member constant set that raises on anything else
    - Import `FORWARD_GAIN` and `CHANNEL_HALF_MAX_HZ` from `brain.motor.descending`; do not re-declare them
    - Headless module — no `WorldViewer` import — so the helpers are importable and testable without `pygame` and without a brain load
    - Implemented before the connectome-dependent arms: these need no brain load and run in milliseconds
    - _Bug_Condition: isBugCondition(input) where the module does not exist and no window is produced_
    - _Expected_Behavior: expectedBehavior(result) — pre-clip U and contamination, boundary contract, mandatory category_
    - _Preservation: `descending.py` not edited; gains imported not copied; engine never mutated_
    - _Requirements: 2.10, 2.14, 2.16, 2.17, 2.18, 2.19, 2.21, 2.22, 2.26, 3.7, 3.11_

  - [ ] 15.2 Unit-test the pure helpers
    - `SpikeWindow`: per-window count equals the difference of cumulative snapshots; raises on missing or out-of-order `begin()`; all zeros over a zero-length window
    - `unilaterality`: `None` at `L + R == 0`; `0.0` at `L == R > 0`; `1.0` at `min(L, R) == 0` with the other side positive
    - `contamination_fraction`: `0.0` at `U = 0`; `0.5` at `U = 1`; monotone increasing in `U`
    - Forward rules: equal when `L == R`; difference equals `2 * abs(L - R)` at `FORWARD_GAIN = 4.0`; clip flags set exactly when the pre-clip value exceeds 1.0
    - `rate_quantum_hz`: 10 Hz for one neuron over 100 ms; 5 Hz for two; 1000 Hz for one over 1 ms
    - `poisson_relative_sd`: matches `1/sqrt(N*rate*T)`, and the between-side figure is `sqrt(2)` times it
    - `odor_concentration` / `antenna_positions`: matches `peak*exp(-r/lambda)` at the shipped start radius; zero bilateral difference with food straight ahead; maximum abeam
    - `sensory_contrast_report`: flags `uninterpretable` exactly when contrast is below the floor; reproduces the two span figures recorded in probe 7
    - `report`: raises when the category is absent or outside the three-member set
    - Verdict thresholds: `genuine` / `mixed` / `re-labelling` / `indeterminate` at contamination medians 0.02, 0.15, 0.40 and at an informative-window fraction of 0.3
    - _Requirements: 2.14, 2.16, 2.17, 2.18, 2.19, 2.21, 2.22, 2.26_

  - [ ] 15.3 Property-test the pure helpers
    - **Property 4: Windowed Counting Is Elapsed-Time Independent** - SpikeWindow Has No Time Trend
    - Generate arbitrary monotone cumulative count sequences; assert the windowed rate has no time trend while the cumulative rate is monotone increasing, and that every per-window count is non-negative and equals the boundary difference
    - Generate random `(lambda, r, span, N, window)` tuples; assert `sensory_contrast_report` is scale-consistent: contrast equals `span/lambda` to first order, the floor scales as `1/sqrt(N*rate*T)`, and the `uninterpretable` flag agrees with a direct comparison of the two
    - Runs without a connectome load, in milliseconds
    - _Requirements: 2.10, 2.11, 2.12, 2.21, 2.22, 3.11_

  - [ ] 15.4 Implement the adapter layer against the real APIs
    - `connectome = load_connectome()`, then `LIFEngine(connectome, seed=SEED)` with `SEED = 3000` printed in the header
    - `DescendingReadout(connectome)`
    - ORN resolution: `BY_KEY["odor_vinegar"]` -> `connectome.by_cell_types(modality.cell_types)` -> split on the `side` column -> deterministic `np.sort(idx)[:129]` per side; print the per-side counts actually used; create no synthetic neurons
    - `engine.set_poisson(all_idx, rates)` per window; `engine.clear_poisson()` on teardown
    - `engine.run(SIMULATION_CHUNK_MS)`; never pass an argument to `engine.step()`
    - `FlyBody.reset()` with no arguments; read and write position and heading through `body.state.x_mm` / `y_mm` / `z_mm` / `heading_deg` / `speed_mm_s`
    - `body.update(dt_ms=..., channels=..., t_ms=..., escape_laterality=..., proboscis_drive=...)` with `t_ms` the true accumulated simulation time, and the same `t_ms` sequence for both forward rules within a run
    - Use `run_sandbox_separated_four_channel_clean.py` as the known-good call-site reference; do not edit it
    - _Bug_Condition: isBugCondition(input) where anyAdapterCallRejectedByRealAPI(input)_
    - _Expected_Behavior: expectedBehavior(result) — a window is produced against the real modules_
    - _Preservation: only ORN indices and rates enter the brain; `FlyBody` receives only motor channels; no navigation controller_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.2, 3.3, 3.6_

  - [ ] 15.5 Wire per-window counting and assert re-derivation fidelity
    - Use `SpikeWindow` around each `engine.run(SIMULATION_CHUNK_MS)` so the numerator and the `window_ms` denominator describe the same interval; pass the same windowed array to `readout.channels`, `escape_laterality` and `proboscis_drive`
    - **Property 3: Non-Invasive Re-Derivation Fidelity** - Recomputed Max Rule Equals The Shipped Channel
    - Assert `abs(clip01(forward_max_preclip(L, R)) - channels["forward"]) < 1e-12` every window, so the audit demonstrably audits the shipped rule and any future drift in the shared contract fails loudly
    - Add the matching property test over arbitrary generated count vectors
    - Do not route through `engine.firing_rates_hz(window_ms)`; keep the windowing local to the diagnostic
    - _Bug_Condition: isBugCondition(input) where input.count_interval_ms != input.window_ms_
    - _Expected_Behavior: expectedBehavior(result) — count_interval_ms = window_ms, clip01(fwd_max_preclip) = channels_forward_
    - _Preservation: `engine.spike_counts` stays cumulative and monotone for all other callers_
    - _Requirements: 2.6, 2.10, 2.11, 2.12, 3.7, 3.11_

  - [ ] 15.6 Implement the headline metrics, clip flags and pre-registered verdict
    - Compute `U = abs(L - R) / (L + R)` and `contamination = U / (1 + U)` pre-clip; `U is None` when `L + R == 0`, with those windows excluded from the distributions and counted as `n_windows_no_bilateral_drive`
    - Per window record `fwd_max_clipped`, `fwd_mean_clipped`, `both_clipped`, and mark `comparison_uninformative` when `both_clipped`
    - Report the full `U` time series plus mean, median, p10, p90, min, max and the undefined-window count
    - Print `corr(delta_preclip, U)` only under an `ALGEBRAICALLY DETERMINED — NOT EVIDENCE` heading stating the `max(L,R) = (L+R)/2 + abs(L-R)/2` identity
    - Apply the pre-registered thresholds from design change 5 on the `[0, 0.5]` contamination scale: `< 0.05` genuine bilateral command; `0.05`–`0.25` mixed; `> 0.25` predominantly re-labelling; `0.5` purely unilateral; `INDETERMINATE` when the informative-window fraction is below 0.5 or the closed-loop arm is flagged sensory-uninterpretable
    - State the interpretation in both directions, so a small value is a real negative result
    - _Bug_Condition: isBugCondition(input) where the headline metric is corr(forward_delta, unilaterality) or the delta was taken after clip_
    - _Expected_Behavior: expectedBehavior(result) — contamination in [0, 0.5], L=R implies 0, min(L,R)=0 implies 0.5, clip flags present, both_clipped implies uninformative_
    - _Preservation: DNa asymmetry logged uncorrected, no rebalancing applied_
    - _Requirements: 2.14, 2.15, 2.16, 2.17, 3.5_

  - [ ] 15.7 Implement sensory contrast, noise floor, feasibility solve and the antenna finding
    - Per window compute from the environment side only: local per-ORN rate, radial gradient, the geometric bilateral difference `gradient * 2 * half_sep * abs(sin theta)` and the exact two-antenna difference actually delivered, contrast as a fraction of the local rate, per-side Poisson relative SD `1/sqrt(N*rate*T)`, the between-side figure `sqrt(2)/sqrt(N*rate*T)`, and the contrast-to-floor ratio
    - Flag the window and the run summary `closed_loop_uninterpretable` when the ratio is below 1
    - Solve numerically with `required_falloff_mm` for the falloff length that would lift a biological span above the floor at the sampled radius, and report whether a feasible `(lambda, r)` region exists
    - Report the `ANTENNA_OFFSET_MM = 60.0` finding with its measured consequence: shipped span versus biological span, the inflation factor, and the contrast on each side of the noise floor — all computed at runtime from the shipped configuration, not hard-coded
    - Remedy a below-floor contrast only by steepening or rescaling the field; never enlarge the antenna separation; use `ANTENNA_HALF_SEPARATION_MM = 0.35` in the bio-geometry arm and label the rescaling an engineering abstraction
    - Report the temporal-sampling observation as a property of the environment model only; implement no temporal comparison, gradient memory or navigation rule, and feed nothing back into the brain
    - _Bug_Condition: isBugCondition(input) where contrast/floor are unreported, or contrast < floor without the uninterpretable flag_
    - _Expected_Behavior: expectedBehavior(result) — contrast_frac and noise_floor_frac present, contrast below floor implies flagged_uninterpretable_
    - _Preservation: antenna separation kept biologically plausible and reported, not tuned; no navigation controller_
    - _Requirements: 2.21, 2.22, 2.23, 3.2, 3.8_

  - [ ] 15.8 Enforce category attribution, header and footer structurally
    - Route every printed metric through `report(name, value, category, note="")`; a number cannot reach the output without exactly one category from `{REAL_CONNECTOME, ENGINEERING_ABSTRACTION, READOUT_ARTIFACT}`
    - Header: seed, `connectome.dataset`, window, rate quantum, per-side ORN counts, per-type per-side DN counts read from `readout.tracked[(ctype, side)].size`, and the standing scope limit that FAFB v783 is a brain connectome with the VNC and leg/wing motor neurons absent
    - Every reported activation is accompanied by its raw spike count and window; any bilateral DNa asymmetry is labelled a comparison of individual reconstructed neurons, never population lateralization
    - Fixed footer: a forward command derived from DNa01/DNa02 is an engineering interface at the brain-to-body boundary, not a claim that DNa01/DNa02 are forward-walking neurons; convert no engineering abstraction into a biological claim
    - _Bug_Condition: isBugCondition(input) where a reported metric has no category in the three-member set_
    - _Expected_Behavior: expectedBehavior(result) — for all m in metrics, category[m] in the three-member set_
    - _Preservation: scope limit restated; DNa asymmetry uncorrected_
    - _Requirements: 2.19, 2.20, 2.26, 2.27, 3.5, 3.9_

  - [ ] 15.9 Implement the CSV writer
    - Write `phase3b_forward_readout_diagnostic.csv` in the working directory, one row per window per arm, with the full column set from design change 13: `arm`, `config`, `counting_mode`, `driving_rule`, `seed`, `t_ms`, `window_ms`; per-type per-side spike counts and Hz for DNa01, DNa02, MDN; `L`, `R`, `u_preclip`, `contamination_preclip`, `u_defined`; `fwd_max_preclip`, `fwd_mean_preclip`, `fwd_max`, `fwd_mean`, `fwd_max_clipped`, `fwd_mean_clipped`, `both_clipped`, `delta_preclip`, `delta_postclip`; `conc_left`, `conc_right`, `local_rate_hz`, `gradient_hz_mm`, `bilateral_delta_hz`, `contrast_frac`, `noise_floor_frac`, `contrast_over_noise`, `sensory_interpretable`; `speed_mm_s`, `x_mm`, `y_mm`, `heading_deg`, `behaviour`, `distance_to_food_mm`, `bearing_to_food_deg`
    - Follow the existing diagnostic CSV convention already used by the two shipped diagnostic scripts
    - _Requirements: 2.11, 2.14, 2.19, 2.21_

  - [ ] 15.10 Wire the five arms, each independently runnable
    - Arm 0 open loop / fixed symmetric ORN drive / WIN / max — stochastic null distribution of `U`
    - Arm 1 shipped Phase 3 / WIN / max — the primary controlled measurement, with the mean rule shadowed from the same counts
    - Arm 2 shipped Phase 3 / WIN / mean — cross-run, descriptive only
    - Arm 3 shipped Phase 3 / CUM / max — labelled reproduction of the defect; the only place cumulative counts appear, every number categorised `READOUT_ARTIFACT` and never used as a measurement of neural activity
    - Arm 4 biological half-separation 0.35 mm with a rescaled fly-scale field / WIN / max — the interpretable closed-loop arm, mean rule shadowed; field parameters taken from the 15.7 feasibility solve and printed as an engineering abstraction
    - `SEED = 3000` for all arms, a fresh `LIFEngine` per arm, arm selection and duration exposed as module-top constants so each arm can be run alone and trimmed
    - Label within-run paired comparisons controlled, naming which rule held the trajectory; print the arm-1-versus-arm-2 behavioural metrics (`end_dist`, `min_dist`, `heading_range`) only under a `DESCRIPTIVE ONLY — CONFOUNDED` heading stating that changing the driving rule changes the trajectory, hence the sampled odor, hence the sensory input
    - Entry point `python -m simulation.environment.run_sandbox_phase3b_true_separated_forward`
    - _Bug_Condition: isBugCondition(input) where cross-run metrics are presented as controlled_
    - _Expected_Behavior: expectedBehavior(result) — within-run paired comparison is the controlled measurement_
    - _Preservation: determinism at a fixed seed; deterministic ORN selection; no plasticity, no reward_
    - _Requirements: 2.11, 2.13, 2.24, 2.25, 3.4, 3.10_

  - [ ] 15.11 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Falsifiable, Window-Correct, Categorised Forward Audit
    - **IMPORTANT**: re-run the SAME test from task 13 — do NOT write a new test
    - The test from task 13 encodes the expected behaviour; when it passes, the expected behaviour is satisfied
    - Run: `python -m pytest tests/test_phase3b_forward_readout_diagnostic.py -q`
    - **EXPECTED OUTCOME**: test PASSES (confirms the bug is fixed)
    - _Requirements: 2.10, 2.14, 2.16, 2.17, 2.19, 2.21, 2.22, 2.26_

  - [ ] 15.12 Verify preservation tests still pass
    - **Property 2: Preservation** - Shared Modules And Existing Callers Unchanged
    - **IMPORTANT**: re-run the SAME tests from task 14 — do NOT write new tests; enable the `SpikeWindow` interleaving clause that was skipped in task 14
    - Run `python -m pytest tests/ -q` and confirm the whole suite is green, matching the task 12 baseline
    - Confirm `git status --short` lists only `simulation/environment/run_sandbox_phase3b_true_separated_forward.py`, `tests/test_phase3b_forward_readout_diagnostic.py` and `requirements/requirements-dev.txt`
    - **EXPECTED OUTCOME**: tests PASS (confirms no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12_

## Validation and reporting

- [ ] 16. Integration-test the arms at reduced duration
  - Run each of the five arms at 1 s / 10 windows against the real connectome, before any full-duration run
  - Assert: completes without exception; CSV written with the full column set; header emits seed, connectome identity, per-side ORN counts and per-type per-side DN counts
  - Assert determinism: two identical reduced runs at `SEED = 3000` produce identical CSVs
  - Constraint audit at reduced duration: no navigation controller, no target direction into the brain, no reward, no plasticity, and the DNa asymmetry present and uncorrected in the output
  - Keep each reduced run short enough to complete inside a normal command timeout; if any single arm does not, hand it over as a command instead of running it here
  - _Requirements: 2.1, 2.3, 2.5, 2.18, 2.19, 2.20, 3.2, 3.3, 3.4, 3.5, 3.10_

- [ ] 17. Full-duration run of all five arms
  - Hand these over for the user to run manually in PowerShell; do not launch them from a blocking call
  - `python -m simulation.environment.run_sandbox_phase3b_true_separated_forward` for the default 10 s x 5-arm run
  - Provide the per-arm invocation for running arms individually, and state the expected wall-clock cost per arm
  - Collect `phase3b_forward_readout_diagnostic.csv` and the console report from the user's run before proceeding
  - _Requirements: 2.1, 2.11, 2.18, 3.10_

- [ ] 18. Produce and validate the categorised report from the full run
  - Every number carries exactly one provenance category; verify no metric bypassed `report(...)`
  - Arm 1 versus arm 3 attribution: the drift statistic on `L + R` (slope against `t`, plus clip fraction in the first versus last decile) is about 0 for arm 1 and strongly positive for arm 3 — the direct check for 2.11 and 2.12
  - Arm 0 null versus arm 1: the `U` distributions printed side by side so closed-loop unilaterality reads against the stochastic null
  - Arm 1 versus arm 4 sensory gate: arm 1 flagged `closed_loop_uninterpretable` under a biological span while arm 4 is not, with the antenna-offset inflation and below-floor figures printed
  - Cross-run behavioural metrics appear only under `DESCRIPTIVE ONLY — CONFOUNDED`; the within-run paired comparison is the only one labelled controlled
  - State the Q0 verdict against the pre-registered thresholds, reporting the informative-window fraction and honouring `INDETERMINATE` when the gate conditions are met
  - Restate the scope limit in the header and the DNa01/DNa02 engineering-interface disclaimer in the footer
  - _Requirements: 2.11, 2.12, 2.14, 2.15, 2.16, 2.17, 2.19, 2.20, 2.21, 2.22, 2.24, 2.25, 2.26, 2.27, 3.9_

- [ ] 19. Author and validate the Option B finding block
  - Name all nine affected sandbox scripts explicitly: `run_sandbox.py`, `run_sandbox_phase2.py`, `run_sandbox_phase3_closed_loop_odor.py`, `run_sandbox_separated_four_channel_clean.py`, `run_sandbox_separated_locomotion_steering.py`, `run_sandbox_steering_gain_sweep.py`, `run_sandbox_bilateral_symmetry_diagnostic.py`, `run_sandbox_descending_readout_diagnostic.py`, `run_sandbox_forward_backward_only.py`
  - State the mechanism (cumulative numerator, per-chunk denominator), the direction (rates and channel magnitudes inflated, growing with elapsed time) and the consequence (motor channels saturate as a function of wall-clock time)
  - Give both independent non-neural explanations for "mean forward = 0.989 invariant across three steering gains": saturation under cumulative counting, and the open-loop fixed-drive fixed-seed design of the gain sweep whose DN trace is bit-identical across gains — citing the probe 8 result recorded in task 9
  - Print the re-derived per-window mean forward from arm 1 next to the 0.989 figure
  - Name `SpikeWindow` as the migration path for a follow-up spec, and state that none of the nine scripts is edited by this spec
  - _Requirements: 1.13, 2.13, 3.11_

- [ ] 20. Checkpoint - Ensure all tests pass
  - Run `python -m pytest tests/ -q` and confirm green
  - Confirm `git status --short` still lists only the three allowed files
  - Confirm every requirement clause in `bugfix.md` sections 2.x and 3.x is referenced by at least one completed task
  - Ensure all tests pass, ask the user if questions arise
