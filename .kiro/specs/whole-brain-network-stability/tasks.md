# Implementation Plan

Derived from `bugfix.md` (requirements) and `design.md` (hypotheses H1–H9, ladder L0–L7,
criterion `S`, estimators B1–B4, fix paths F1–F7).

## Standing constraints — apply to every task below

- **Platform.** Windows 11, PowerShell. Engine is pure NumPy on the CPU; there is no CUDA path in
  `simulation/engine/`, so GPU VRAM is irrelevant to every budget here.
- **Memory.** 16.4 GB installed but only ~3.8 GB available when measured; ~300 MB per full-scale
  engine process. 6 physical cores. **At most ~6 parallel conditions** (~1.8 GB). NumPy
  elementwise ops and `bincount` are single-threaded, so parallelism comes from processes, never
  threads.
- **Execution rule (non-negotiable).** Nothing above ladder rung **L2** may be launched from a
  blocking tool call. L3 and above are handed to the user as PowerShell commands that write
  incremental CSV and print an ETA first, computed from the measured cost model
  `T_wall ≈ 0.19 ms × n_steps + 0.032 ms × n_spikes`.
- **Seed spread.** Any cumulative observable requires **≥ 5 seeds with a reported spread**. The
  design measured a **1.94x** spread in single-ORN cumulative spikes across seeds 0–2 while
  plateau rate (4.06–4.13 Hz) and active fraction (8,934–9,094) were stable. **A single-seed
  cumulative figure is never reported as a result.**
- **Instrumentation.** `check_invariants` defaults to `False`, and the flag is checked **once
  outside the arithmetic** so the default hot loop is untouched and Property 2 bit-identity holds.
- **Branching ratio.** Never measured at plateau (stationarity pins it at 1.0). **B1** quiescent
  single-spike perturbation is the **primary** estimator that criterion **S4** is tested against;
  **B2** is restricted to **one generation** (18 steps); **B3** corroborates from the transient;
  **B4** spectral bound is an **L0 early falsifier only** and is never reported as `sigma`.
- **Untouchable.** Real FlyWire v783 connectivity exactly as loaded (139,255 / 3,732,460 /
  50,666,648), all 86 existing tests passing, bit-identical behaviour outside the bug condition,
  bilateral ORN convergence preserved **uncorrected**, no navigation controller, no reward and no
  plasticity anywhere in the diagnostic, and any non-Shiu mechanism labelled **OUR ENGINEERING**.

---

## Phase 0 — Baseline prerequisite (must complete before any number is evidence)

- [ ] 1. Establish the reproducible baseline by COMMIT
  - **Decision, from design "Baseline prerequisite": COMMIT — do not revert, do not merely pin.**
  - Reason 1: revert is not viable. `tests/test_plasticity.py` is untracked and exercises the
    plasticity API that exists only in the working-tree `lif_engine.py`, so reverting the engine
    breaks a test that is part of the 86; and HEAD's `config.py` points at
    `D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)`, a path that does not exist on this machine, so
    a HEAD tree cannot load data without an env override.
  - Reason 2: pinning cannot be verified or diffed. Requirement 2.20 wants "the exact revision",
    which requires a hash.
  - Scope note carried from design: the neural-stability numbers depend only on `lif_engine.py`
    (already verified bit-identical to HEAD) plus the checksummed
    `data/derived/connectome_v783.npz`. `config.py` is reformatting plus one non-semantic change
    (`DEFAULT_FLYWIRE_DIR` → `PROJECT_ROOT/data/flywire_v783`, source-CSV location only) and
    `data/metadata/build_manifest.json` changes only `source_dir` / `built_utc`.
  - _Requirements: 2.20, 2.21_

  - [ ] 1.1 Pin the test invocation and record the 86-test baseline
    - The baseline invocation is **`python -m pytest tests`** and nothing else.
    - Record explicitly that a bare **`python -m pytest` from the repository root does not
      terminate** — measured still running at a 300 s timeout, because it collects the two
      root-level `test_bilateral_odor.py` / `test_four_motor_commands.py` files.
    - Record the pass count and wall time (expected 86 tests in ~26 s).
    - _Requirements: 2.20, 3.5_

  - [ ] 1.2 Create the baseline branch and commit
    - Branch off `HEAD = 26672e0`; do not commit to `main`/`master`.
    - Commit the **five** modified tracked files: `simulation/engine/lif_engine.py`,
      `brain/motor/descending.py`, `fly/body/fly_body.py`, `config.py`,
      `data/metadata/build_manifest.json` (2,262 insertions / 406 deletions total — five files,
      not the four the requirements list).
    - Commit the untracked but load-bearing paths: the whole `simulation/environment/` directory
      (13 scripts, including every `run_sandbox_*.py` that produced the reported CSVs),
      `tests/test_fly_body.py`, `tests/test_plasticity.py`.
    - **Exclude** the generated root-level CSVs (`phase3b_*.csv`, `phase3c_*.csv`,
      `bilateral_symmetry_diagnostic.csv`, `descending_readout_diagnostic.csv`) and
      **exclude `data/flywire_v783/`** (large source data, already checksummed by
      `tests/test_data_provenance.py`).
    - Stage files by name; do not use `git add -A`.
    - _Requirements: 2.20_

  - [ ] 1.3 Record the baseline revision and run conditions
    - Write `.kiro/specs/whole-brain-network-stability/baseline.md` recording: the resulting
      commit hash, the pinned test invocation from 1.1, the connectome checksum, the seeds,
      `dtype`, and every parameter that any reported number was produced under.
    - Record that `descending.py` / `fly_body.py` are **not in the neural simulation path** and
      that their HEAD comparison is deferred to task 14 for exactly that reason.
    - Every stability number produced from here on cites this hash.
    - _Requirements: 2.20, 2.21_

---

## Phase 1 — Reference research (blocks H3, H4, H8, H9 and F2's provenance)

- [ ] 2. R-TASK-1 — read the reference implementation and answer Q-A through Q-G
  - **Ordering is mandatory: this task precedes every task whose conclusion depends on H3, H4, H8
    or H9, and precedes any F2 provenance claim.** Four hypotheses and the entire category-B/
    category-C labelling of F2 hang on it.
  - Source: `https://github.com/philshiu/Drosophila_brain_model`, primarily `model.py`, plus the
    example notebooks; Shiu et al. 2024 Nature 634:210-219 (doi:10.1038/s41586-024-07763-9)
    Methods. Read the code where code exists; do not infer from paper text alone.
  - **R1 (`_literal_reference()` in `tests/test_lif_engine.py`) cannot arbitrate any of these
    questions** — it is our own transcription and shares the H3 and H4 assumptions under test.
    `test_matches_literal_reference_implementation` passing is therefore not evidence of fidelity.
  - _Requirements: 2.11, 2.14, 2.18, 2.19_

  - [ ] 2.1 Answer the seven questions with exact file, symbol and line
    - Q-A: does `PoissonInput` target `v` or `g`? What are `N`, `rate`, `weight`? → feeds H4, F2.
    - Q-B: is the stimulated population's refractory period suppressed, or does
      `refractory='t_rfc'` apply to driven neurons too? → feeds H4, H5, F2.
    - Q-C: does Brian2's `(unless refractory)` clause on `dg/dt` accumulate events without decay,
      as R1 assumes? → feeds H3.
    - Q-D: how many neurons are driven in the published experiments, for how long per trial, how
      many trials? → feeds H9, F7.
    - Q-E: does the reference simulate all 139,255 neurons or named subsets? → feeds H9, F7.
    - Q-F: does the reference contain adaptation, depression, normalisation or any global gain
      term? → feeds H8, F3–F5.
    - Q-G: is drive applied continuously for the whole trial, or as a pulse? → feeds H9, F2.
    - Deliverable: an answers table appended to
      `.kiro/specs/whole-brain-network-stability/reference-findings.md`.
    - _Requirements: 2.11, 2.14, 2.18, 2.19_

  - [ ] 2.2 Record unresolved questions and mark dependent conclusions conditional
    - For any question that cannot be resolved from the public code, record it as **UNRESOLVED**
      with what was searched. **Do not substitute a guess.**
    - Mark every dependent verdict (H3 from Q-C, H4 from Q-A/Q-B, H8 from Q-F, H9 from Q-D/Q-E/Q-G)
      and any F2 provenance claim as **CONDITIONAL** on the unresolved item, and propagate that
      conditionality into the task 12 gate.
    - Record the specific consequence: if Q-A/Q-B/Q-G show the reference does what we do, then any
      F2 change is **category C, OUR ENGINEERING**, not a category-B restoration.
    - _Requirements: 2.18, 2.19, 3.3_

---

## Phase 2 — Diagnostic harness and regression coverage (written before any fix)

- [ ] 3. Build the stability diagnostics module and the CLI driver
  - New files: `simulation/diagnostics/stability.py`, `simulation/environment/run_stability_diagnostic.py`.
  - No engine change in this task; the diagnostics read the engine, they do not modify it.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9, 2.15, 2.16, 2.22_

  - [ ] 3.1 Implement observables D1–D8
    - D1 brain spikes per injected spike, **per seed with spread** (≥ 5 seeds); D2 total activity
      vs injected spike count over the ORN-count sweep; D3 `gamma` over 1.0–150.0 Hz; D4 250 ms
      binned time course and time-to-plateau (first bin within 5% of the mean of the next three);
      D5 active fraction per condition; D6 hop0–hop3 mean rate and population by BFS hop distance
      over the unsigned adjacency, **each printed with its fraction of the 454.5 Hz refractory
      ceiling**; D7 seed-exclusion report (out-degree of every candidate seed, which were excluded
      and why); D8 post-drive decay at 100 ms after drive removal as a fraction of driven level.
    - `engine.spike_counts` stays cumulative-since-`reset()`; windowed measurement is an addition,
      never a change of documented semantics.
    - Every observable carries its provenance category (A real FlyWire data / B published Shiu /
      C our engineering).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9, 2.22, 3.13_

  - [ ] 3.2 Implement estimators B1, B2, B3 and the B4 bound
    - B1 `sigma_rest` (**PRIMARY, the estimator S4 is tested against**): full network at rest
      (`v = v_0`, `g = 0`, no drive), force exactly one spike by writing its outgoing events into
      the ring buffer, run with no drive until 36 consecutive spike-free steps (extinct) or a cap
      (5,000 spikes or 500 ms). `sigma_rest = 1 - 1/mean(S)` over non-truncated avalanches,
      reported with the truncation fraction `P(ignition)`; if anything truncates, `sigma_rest > 1`
      is established directly and the point estimate is a **lower bound**. Record the stated
      limitation: measured at rest where `v` sits 7 mV below threshold, so it is a lower bound on
      the driven-state ratio.
    - B2 `sigma_drive`: paired identical-seed runs, one extra forced spike at step `t0`, excess
      counted over **generation 1 only** (`[t0+18, t0+36)`), ≥ 20 injection times, ≥ 3 seeds.
      Longer windows measure chaotic divergence, not branching, and are forbidden.
    - B3 `sigma_transient` (corroborating): 18-step generation bins over the **ignition transient
      only**, fit `ln n(k)` vs `k`, restricted to generations below 25% of plateau, fit window
      reported; net out the external contribution by starting at the first generation whose count
      exceeds 10x the expected injected count.
    - B4 spectral bound: ARPACK largest |lambda| of the threshold-normalised excitatory gain
      matrix. **L0 early falsifier only** — record the rejection reasons (LIF map non-linear near
      threshold, per-edge gain depends on the momentary `v` distribution and refractoriness,
      looseness on a signed strongly non-normal 3.73M-edge operator unquantified).
    - Guard in code: refuse to compute any branching ratio from a plateau window (Property 3).
    - _Requirements: 2.15, 2.16, 2.22_

  - [ ] 3.3 Implement the criterion `S` evaluator
    - Fixed configuration: 258-ORN driven set, `r_lo = 1.0 Hz`, `r_hi = 150.0 Hz`, duration
      1,500 ms, measurement window 1,000–1,500 ms, ≥ 3 seeds.
    - S1 `gamma = ln(R(r_hi)/R(r_lo)) / ln(r_hi/r_lo) >= 0.5`; S2 active fraction `>= 0.02`
      (≥ 2,785 neurons) **and** `R(r_hi) >= 0.5 Hz` **and** DNa01 pair `>= 1 spike/s`;
      S3 no hop-level mean rate above `0.25 × 454.5 = 113.6 Hz`; S4 `sigma_rest <= 1.0` **and**
      `sigma_drive <= 1.0`.
    - Report but do not gate: ignition threshold `k*`, the subcritical/critical/supercritical
      classification, `sigma_transient`, time-to-plateau, D8.
    - Label all four constants **OUR ENGINEERING (category C)** — a criterion we chose, not
      published values.
    - _Requirements: 2.22, 2.23, 3.3_

  - [ ] 3.4 Implement the CLI driver
    - One condition per OS process; incremental CSV writes so a partial run is still usable;
      **prints an ETA from `T_wall ≈ 0.19 ms × steps + 0.032 ms × spikes` before doing any work**.
    - `--rung {L0,L1,L1b,L2,L3,L4,L5,L6,L7}`, `--seeds`, `--out`, `--max-procs` defaulting to 6.
    - Refuses to launch more than 6 concurrent full-scale engines (~300 MB each against ~3.8 GB
      available).
    - Enforces `plasticity_enabled = False` and asserts `wdata` unmodified at exit; no navigation
      controller, no target/food coordinates into the brain.
    - _Requirements: 2.22, 3.10, 3.11, 3.12_

  - [ ] 3.5 Unit tests for the harness itself
    - Generation binning uses `delay_steps = 18`; a synthetic Galton-Watson process with known
      `sigma` is recovered by B3 within tolerance.
    - Avalanche bookkeeping: a forced spike on an isolated neuron gives `S = 1`.
    - B1 soundness on synthetic trees with known `sigma`, including near `sigma = 1` where
      variance is largest, validating the truncation policy independently of the connectome.
    - _Requirements: 2.15, 2.16_

- [ ] 4. Create `pytest.ini` with the `wholebrain` marker
  - No pytest configuration exists in the repository today, so markers must be registered here.
  - ```ini
    [pytest]
    markers =
        wholebrain: full 139,255-neuron simulation; minutes, opt-in
        slow: over 10 s
    addopts = -m "not wholebrain"
    ```
  - Confirm `python -m pytest tests` still collects and passes the 86 existing tests in ~26 s and
    that the default path excludes `wholebrain`.
  - Record the two invocations: `python -m pytest tests -m wholebrain` (~12 min, opt-in) and
    `python -m pytest tests -m "not wholebrain"` (the ~34 s default).
  - _Requirements: 2.31, 3.5_

- [ ] 5. Write the whole-brain bug condition exploration test
  - **Property 1: Bug Condition** - Sensory drive must modulate whole-brain activity
  - **CRITICAL**: This test MUST FAIL on unfixed code — the failure confirms the bug exists.
  - **DO NOT attempt to fix the test or the code when it fails.**
  - **NOTE**: This test encodes criterion `S`, i.e. the expected behaviour. It becomes the fix
    validator in task 20.2 when it passes.
  - **GOAL**: surface counterexamples that demonstrate the whole-brain network is in a
    stimulus-independent attractor.
  - **Scoped PBT approach**: the bug is deterministic given a seed, so scope the property to the
    concrete failing configuration rather than sampling the whole domain — full v783 connectome,
    258-ORN driven set, `duration_ms = 1500 >= 500`, driven set non-empty with at least one
    `outDegree(n) > 0` neuron, i.e. exactly `isBugCondition(X) == true`. Parameterise over
    `rate_hz in {1.0, 150.0}` and seeds `{0, 1, 2}`.
  - New file `tests/test_whole_brain_stability.py`, marked `@pytest.mark.wholebrain` and
    `@pytest.mark.xfail(strict=True, reason="whole-brain-network-stability: network is in a
    stimulus-independent attractor; see .kiro/specs/whole-brain-network-stability")`.
  - Assertions are exactly `expectedBehavior(result)` from the design: `gamma >= 0.5` (S1),
    `active_fraction >= 0.02` and `brain_rate_hz >= 0.5` and `dna01_rate_hz >= 1.0` (S2),
    `max_hop_rate_hz <= 113.6` (S3), `sigma_rest <= 1.0` and `sigma_drive <= 1.0` (S4). Includes
    the single-connected-seed ignition bound via a 20-neuron B1 subset.
  - Budget ≤ 5 min: B1 with 20 seed neurons plus the two-point 1.0/150.0 Hz sweep at 1,500 ms.
    Memory ceiling 1 GB.
  - **EXPECTED OUTCOME: FAILS on the unfixed engine.** Recorded counterexamples, from Evidence B/C:
    `gamma = ln(4.76/4.09)/ln(150) = 0.030`, failing S1 by a factor of 16; hop1 = 230.09 Hz =
    `0.506` of the 454.5 Hz ceiling, failing S3; a single connected ORN (index 1574) at 20 Hz
    converting ~40 injected spikes into 457,635–888,253 brain spikes, failing S4.
  - **Do not run the `wholebrain` marker from a blocking tool call.** The confirming measurement is
    the L4 run in task 16; the `xfail(strict=True)` marker carries the recorded failure until then,
    and an accidental pass is itself an error.
  - Mark complete when the test is written, registered, collected under `-m wholebrain`, and the
    pre-fix failure is documented with the counterexamples above.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.22, 2.23, 2.31_

- [ ] 6. Write the whole-brain specificity regression test
  - Whole-brain counterpart to the `tests/test_circuits.py` specificity assertion: bitter GRN drive
    produces **zero** proboscis motor neuron spikes at full scale, where the runaway manifests —
    the subgraph version passes at a scale where the runaway does not appear, so it is not evidence.
  - `@pytest.mark.wholebrain`, `@pytest.mark.xfail(strict=True, ...)` referencing this spec.
    Budget ~1.5 min.
  - Un-xfailed in task 20.2 when a fix passes.
  - _Requirements: 2.32_

- [ ] 7. Write preservation property tests and capture golden vectors (BEFORE any engine change)
  - **Property 2: Preservation** - Sub-threshold-scale behaviour is bit-identical
  - **IMPORTANT**: follow observation-first methodology, and capture these vectors on the
    **UNFIXED** engine **before any engine change whatsoever** — including before the task 10
    instrumentation lands.
  - Observe `spike_counts` on the unfixed engine and store golden vectors keyed by
    `(subgraph spec, drive spec, seed, duration, dtype)` under
    `tests/data/preservation_golden/`, with the task 1.3 baseline hash recorded in the file.
  - Generator domain, all strictly inside `NOT isBugCondition(X)` (Hypothesis or equivalent):
    subgraph size **50–4,000** with the index set drawn deterministically from the seed; driven set
    **1–50** indices from the subgraph; rate **1–300 Hz**; duration **5–400 ms** (strictly below
    the 500 ms bug-condition floor); `dtype` in **{float32, float64}**; engine seed **0–1000**.
  - Property-based testing is the right tool: the domain is large and the assertion is exact
    equality, so every generated case is a real test and the boundaries where an
    activity-dependent fix would leak are the ones most likely to be sampled.
  - Also record as fixed-point preservation cases: `test_matches_literal_reference_implementation`
    (120-neuron LPLC2/DNp01), `test_determinism` (150-neuron),
    `test_float32_and_float64_give_identical_spikes`, the ~4,000-neuron escape-subnetwork tests,
    and the bilateral convergence figures hop0 77.85% → hop1 2.11% / hop2 6.41% / hop3 8.93%
    (asserted to stay **within tolerance of the recorded values, not improved**, and labelled
    REAL CONNECTOME).
  - Assert `sign(wdata) == _w_sign` with no entry crossing zero, and `plasticity_enabled` false
    with `wdata` unmodified, across the whole generated domain.
  - Bounded default-suite variant: `max_examples=25`, subgraphs capped at 1,000 neurons, so the
    default `python -m pytest tests` budget grows by ≤ 8 s to ~34 s total.
  - **EXPECTED OUTCOME: PASSES on the UNFIXED engine** — this is the baseline to preserve.
  - Mark complete when the vectors are captured, committed, and the property test passes against
    the unfixed engine.
  - _Requirements: 3.1, 3.2, 3.5, 3.6, 3.7, 3.10, 3.11, 3.12, 3.13_

---

## Phase 3 — Escalation ladder L0 → L2 (under 6 minutes total; all must be read before any higher rung)

- [ ] 8. Rung **L0** — scalar, structural and spectral probes (< 10 s)
  - Falsifies H1 (values), H5, H7; gives an early signal on H6.
  - _Requirements: 2.9, 2.11, 2.15, 2.17, 3.8, 3.9_

  - [ ] 8.1 H5 unit test — Poisson target with non-zero `rfc_len` (WRITE THIS FIRST)
    - **The cheapest probe in the entire design (milliseconds). Written before anything else in L0.**
    - Subgraph with a Poisson target whose `rfc_len` is non-zero; compare engine against R1.
      Engine stage 4 is `spk = np.flatnonzero(v > self._vth32)` with **no** `rfc_left == 0`
      condition; R1's condition is `v[i] > p.v_th and rfc[i] == 0`.
    - Today the two agree, because a refractory neuron's `v` is pinned at `v_rst` and the only
      path that raises `v` is the stage-3 Poisson kick, which applies only to `_poi_idx` whose
      `rfc_len` is forced to 0. **The divergence is latent and candidate fix F2 makes it live.**
    - Record the verdict for H5, and record the binding constraint on F2: any F2 variant that
      restores refractoriness on driven neurons **must add the `rfc_left == 0` guard in the same
      change**, and that guard must be shown not to alter any subgraph result.
    - _Requirements: 2.12, 2.14_

  - [ ] 8.2 Exact scalar coefficient check (H1, values only)
    - Assert `_ev = exp(-a*dt) = 0.99501248`, `_eg = exp(-b*dt) = 0.98019867`,
      `_kg = a(_eg - _ev)/(a - b) = 0.00493794` with `a = 1/t_mbr = 0.05`, `b = 1/tau = 0.2`,
      `dt = 0.1`, in **both float64 and the float32 hot-loop variants**.
    - Record that this settles the *values* only; the accumulated-drift-at-scale part of H1 is
      rung L7 (task 18) and its verdict stays `__TBD__` until then.
    - _Requirements: 2.11_

  - [ ] 8.3 E/I sign control (H7 — CONTROL, not a suspect)
    - Confirm at full scale that no neuron carries sign 0 or an inverted sign under the Shiu
      convention (ACH/DA/OCT/SER excitatory; GABA/GLUT inhibitory).
    - Report 62.33%/37.67% per-connection and 61.45%/38.55% synapse-weighted E/I, with
      `nt_resolved` counts ACH 94,182, GLUT 22,726, GABA 18,389, SER 1,435, DA 791, OCT 106, zero
      unknown, as a **control result**.
    - **Escalate H7 to suspect status only if this control fails.**
    - _Requirements: 2.17, 3.9_

  - [ ] 8.4 Structural statistics and seed-exclusion report (D7)
    - Report 13.57 mean synapses per connection, the 3.73 mV `g` step at `w_syn = 0.275` against
      the 7.0 mV threshold gap (`v_0 = -52`, `v_th = -45`), and mean out-degree
      3,732,460/139,255 ≈ 26.8.
    - Check out-degree of every candidate ignition seed against the loaded connectome; exclude
      anatomically isolated neurons and report which and why. R[0] index 164 has out-degree 0 and
      is excluded — **a real connectome fact, left unconnected, never read as absence of ignition.**
    - _Requirements: 2.9, 2.15, 3.8_

  - [ ] 8.5 B4 linearised spectral bound (H6 early falsifier only)
    - A few ARPACK iterations for the largest |lambda| of the threshold-normalised excitatory gain
      matrix, seconds of cost.
    - A value far above 1 is early corroboration of H6 that justifies paying for L2. **This value
      is never reported as `sigma`** — B1 is the estimator S4 is tested against.
    - _Requirements: 2.15_

- [ ] 9. Rung **L1** — quiescent control, full network, no drive, 200 ms (~2 s + 1.8 s load)
  - Assert **exactly zero** spikes with no Poisson drive on the full 139,255-neuron network.
  - Falsifies any hypothesis predicting drive-free activity. Expected to pass on unfixed code; a
    failure here reframes the whole investigation and must halt the ladder.
  - Small enough for the default suite; add it there.
  - _Requirements: 2.16, 3.1_

- [ ] 10. Rung **L1b** — instrumented 1 s ignited run (H2, H3) (~14 s)
  - **This is the first task that touches the engine, so task 7's golden vectors must already be
    captured.**
  - _Requirements: 2.12, 2.13, 3.6_

  - [ ] 10.1 Add the `check_invariants` instrumentation flag
    - `LIFEngine(..., check_invariants=False)` by default. **The flag is checked once outside the
      arithmetic so the default hot path adds no operations** (Property 2).
    - Exhaustive per-step check `_n_refractory == count_nonzero(rfc_left)` over the whole network.
      Measured cost: `np.count_nonzero` on the 139,255-entry int32 `rfc_left` is **22.3 us**, so
      20,000 steps cost **0.45 s**, ~1.1% overhead on a ~41 s run. **No sampling** — an exhaustive
      check is strictly better here and that is the only reason sampling is not used.
    - On violation record the step index, `_n_refractory`, `count_nonzero(rfc_left)` and the number
      of neurons wrongly integrated.
    - H3 accounting rides the same flag: two extra `n`-length float32 arrays holding `g` at
      refractory entry and the counterfactual decayed value, differenced on exit, ~30 us/step.
    - _Requirements: 2.12, 2.13_

  - [ ] 10.2 Instrumentation-off bit-identity check
    - Re-run task 7's golden-vector property test with `check_invariants=False` and assert
      bit-identity against the pre-instrumentation vectors. A failure here blocks the rung.
    - _Requirements: 3.6, 3.12_

  - [ ] 10.3 H2 verdict — exhaustive refractory invariant over 1 s ignited runs, 3 seeds
    - Report the number of timesteps on which the refractory hold was skipped.
    - The static reading in the design predicts a pass: after the decrement `_n_refractory` is
      recomputed as `count_nonzero(rfc_left[ridx])` over exactly the previously-nonzero set, and
      newly-refractory neurons cannot already be in `ridx` because a refractory neuron's `v` is
      held at `v_rst = -52 mV`. The one path that makes it transiently **too high** — `set_poisson`
      mid-run zeroing `rfc_left` without decrementing — is safe and self-heals.
    - **The static argument is not the answer**; 2.12 demands the measurement, and this kind of
      reasoning already lost once to this bug.
    - Detection guarantee to record: exhaustive over every step of every instrumented run, so a
      pass falsifies H2 **for that trajectory class with no sampling gap** — not a proof for all
      inputs, and the limit is recorded rather than overclaimed.
    - _Requirements: 2.12_

  - [ ] 10.4 H3 verdict — refractory `g` inflation magnitude
    - Stage 1 runs `g += pending` for all neurons, the `g_hold` snapshot is taken **after** that
      add and restored **after** `g *= _eg`, so a refractory neuron accumulates incoming drive
      across its whole 2.2 ms refractory period with no decay.
    - Report mean and maximum extra `g` per refractory neuron per refractory period at full scale,
      and state whether it is material against the 7.0 mV threshold gap.
    - **R1 does exactly the same thing**, so `test_matches_literal_reference_implementation`
      passing is not evidence of fidelity here. **Whether this is faithful is decided by
      R-TASK-1 Q-C (task 2), not by R1.** If Q-C is UNRESOLVED, the H3 verdict is recorded as
      CONDITIONAL.
    - _Requirements: 2.13, 2.18_

- [ ] 11. Rung **L2** — avalanche battery, B1 `sigma_rest` (~4 min single-process, ~1 min on 6 cores)
  - 200 single-spike perturbations from rest, stratified by out-degree, all with
    `outDegree > 0` per task 8.4's exclusion list.
  - Primary probe for **H6**; yields `sigma_rest` and the `k* = 1` ignition check.
  - Prefer the task 3.4 driver with `--max-procs 6` (~1.8 GB). L2 is the highest rung that may be
    run from a blocking call, and even then the parallel driver is preferred over a 4 min block.
  - Expected counterexample on unfixed code: a non-zero truncation fraction, i.e.
    `sigma_rest > 1`, reported as a lower bound.
  - _Requirements: 2.15, 2.16_

---

## Phase 4 — Consolidation GATE

- [ ] 12. **GATE** — fill every hypothesis verdict and decide the root-cause fork (2.10)
  - **This is a gate. Nothing in Phase 7 (fix selection or implementation) may start until it is
    passed.**
  - Fill every `__TBD__` verdict field in **H1 through H9** in `design.md`, each with the
    measurement that supported or falsified it and the task it came from:
    H1 values ← 8.2 (drift-at-scale deferred to 18, recorded as such); H2 ← 10.3; H3 ← 10.4
    conditional on Q-C; H4 ← task 2 (Q-A, Q-B); H5 ← 8.1; H6 ← 11 primary, 8.5 early signal;
    H7 ← 8.3 control; H8 ← task 2 (Q-F); H9 ← task 2 (Q-D, Q-E, Q-G).
  - State the fork decision for requirement 2.10: hypothesis **(i) engine deviation**
    (supported by H1–H5, H8) versus **(ii) unstable published parameterization in an untested
    regime** (supported by H6, H9), with the discriminating measurements named.
  - **STOP CONDITION: if the surviving hypothesis set is empty, or the surviving verdicts are
    mutually contradictory, halt and re-hypothesise. Do not proceed to fix selection.** Raise it
    with the user with the evidence table.
  - **F7 CONDITION: if H6 and H9 both survive while F1 and F2 find no deviation, then F7 — no fix,
    re-scope, retract — becomes the live outcome.** Record that plainly and do not force a fix:
    the honest result is that the published parameterization does not support sustained whole-brain
    Poisson drive, all claims are restricted to the regime R-TASK-1 shows the reference exercises,
    Phase 1–3C are retracted, and any stabilisation offered is labelled category C engineering that
    departs from the published model.
  - Carry every CONDITIONAL marker from task 2.2 into the gate decision; a conditional verdict may
    not be silently promoted to settled.
  - _Requirements: 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19_

---

## Phase 5 — Recording tasks and the deferred module audit (not blocked by the fix)

- [ ] 13. Record the retractions and the real-connectome findings
  - **These are documentation tasks, not experiments** — the design classifies them as recording
    work against Evidence A–G, which was already measured against the loaded connectome. They have
    **no dependency on the fix and no dependency on any rung above L2**, and may be done at any
    point after task 1.
  - _Requirements: 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.30, 3.7, 3.14, 3.15_

  - [ ] 13.1 Retract Phase 1 through Phase 3C
    - Mark every Phase 1–3C neural conclusion as measured **inside the stimulus-independent
      attractor and therefore uninterpretable**, requiring re-derivation after the fix before it is
      cited as evidence again.
    - Phase 3C "no DNa asymmetry above the matched-rate null" is **not evidence about chemotaxis**;
      state the reason — the null and bio arms had identical U distributions because both sampled
      the same stimulus-independent state.
    - Record that the Phase 3C activation median U of 0.3261 (null) / 0.3298 (bio) equals
      `(2/62 − 1/61)/(2/62 + 1/61) = 0.326087` exactly — one DNa neuron firing 2 spikes/s against
      the other firing 1 — that across 1,820 measurements the observable took only **16 distinct
      values**, and that **28%** of windows had one side at exactly zero. Readout dynamic range is
      a **separate defect** from network stability and is **not** claimed as fixed here.
    - Record that Phase 3C locomotion was mean forward 0.054, max 0.158, against
      `MIN_LOCOMOTION_DRIVE = 0.12`, so only **1.5%** of windows crossed the walking threshold and
      **no navigation was tested**.
    - _Requirements: 2.24, 2.25, 2.26, 2.27_

  - [ ] 13.2 Record the phase3b relationship
    - This spec is **strictly upstream** of `.kiro/specs/phase3b-forward-readout-diagnostic`; a
      readout audit is not meaningful until the network encodes its input; the phase3b preservation
      guarantee no longer holds and must be re-established against the task 1.3 baseline.
    - Leave the phase3b scope **intact and unexecuted**; do not absorb, supersede or duplicate its
      clauses in this spec.
    - _Requirements: 2.28, 3.15_

  - [ ] 13.3 Record Evidence D as a REAL CONNECTOME finding — must NOT be fixed
    - hop0 77.85% input contrast (40 Hz left / 5 Hz right) collapsing to hop1 2.11%, hop2 6.41%,
      hop3 8.93%, **within 0.2 pp of the matched 20.3/20.3 Hz condition**. Attribute the residual
      2/6/9% asymmetries to **structural population differences** (3,181 vs 3,418 neurons at hop2;
      9,029 vs 9,073 at hop3), not to stimulus.
    - Record the mechanism: 94.1% of left-ORN and 96.3% of right-ORN synapses terminate on
      first-order targets receiving input from **both** antennae, across **63 shared targets**
      including DM1_lPN, VA2_adPN, DM4_adPN, DM2_lPN (splitting 49.7/50.3) and the lLN2 and
      il3LN6 local neurons.
    - State that this is expected adult *Drosophila* anatomy — ORNs project bilaterally and the
      antennal lobe sums the two antennae — and frame it as a **constraint on which readouts can
      ever work**, not a defect to repair. Preserve it **uncompensated and uncorrected**: no
      rebalancing, offset or lateral-contrast enhancement anywhere in the path.
    - _Requirements: 2.29, 2.30, 3.7_

  - [ ] 13.4 Record provenance and the standing scope limit
    - Every mechanism in `F'` absent from Shiu et al. 2024, and every constant in criterion `S`, is
      labelled **OUR ENGINEERING** with its category and is never presented as the published model;
      any changed published constant is recorded with the deviation and its justification.
    - Preserve the existing provenance headers and inline citations in
      `brain/neuron_models/lif.py` and `simulation/engine/lif_engine.py`, including Shiu et al.
      2024 Nature 634:210-219 (doi:10.1038/s41586-024-07763-9) and
      `philshiu/Drosophila_brain_model`.
    - Restate the standing scope limit: FAFB v783 is a **brain** connectome, the VNC and leg/wing
      motor neurons are absent, and everything downstream of the descending-neuron stage is an
      explicitly-labelled engineering body model, not a biological claim.
    - _Requirements: 2.22, 3.3, 3.4, 3.14_

- [ ] 14. HEAD-vs-working-tree comparison for `descending.py` and `fly_body.py`
  - **Deferred deliberately: neither module is in the neural simulation path**, so neither can
    affect a stability number. They matter only for the downstream Phase 3B/3C claims that task
    13.1 retracts.
  - Method: `git worktree add ../fruit-fly-lab-head 26672e0`; set
    `FLYWIRE_V783_DIR=<repo>/data/flywire_v783` in that worktree; copy or link `data/derived/`.
  - Swap **one** module at a time into the working tree, run the same diagnostic, diff outputs.
    One-at-a-time is necessary because the modules are interdependent
    (`fly_body <- descending <- engine`) and a whole-tree swap would not isolate the cause.
  - Record bit-identity per module. **Expect `descending.py` / `fly_body.py` to differ** — both
    were substantially rewritten (+483/-121 and +802/-147) — and record that as the reason the
    Phase 3B/3C numbers are not reproducible, consistent with 2.24 rather than a new problem.
  - Completes requirement 2.21 alongside the already-verified `lif_engine.py` result.
  - _Requirements: 2.21, 2.24_

---

## Phase 6 — Ladder rungs above L2 (user-executed; L0–L2 must be read first)

Every task in this phase is handed to the user as a PowerShell command. **None may be launched
from a blocking tool call.** Each command prints an ETA from
`T_wall ≈ 0.19 ms × steps + 0.032 ms × spikes` before doing work, and writes incremental CSV.

- [ ] 15. Rung **L3** — single-ORN ignition, ≥ 5 seeds × 2 s (~2.5 min)
  - Reproduces Evidence A **with a spread**, on a connected seed from task 8.4's list (index 1574,
    `odor_vinegar` left) at 20 Hz.
  - **≥ 5 seeds mandatory, spread reported.** Recorded measurement for seeds 0–2: 888,253 /
    457,635 / 871,195 brain spikes — a **1.94x** spread — against active counts 9,094 / 8,934 /
    9,056 (1.8% spread) and a stable 4.06–4.13 Hz plateau. Ignition **latency** is the variance
    source: at 20 Hz only ~40 Poisson events land in 2 s, so when the first few land dominates the
    total. **The Evidence A figure of 579,293 spikes is one draw of a high-variance quantity and is
    never quoted alone.**
  - Reports D1 with spread and D4 time-to-plateau; ETA ≈ 5 × 32 s.
  - ```powershell
    python -m simulation.environment.run_stability_diagnostic --rung L3 --seeds 0,1,2,3,4 --out stability_L3.csv
    ```
  - _Requirements: 2.1, 2.4, 2.9_

- [ ] 16. Rung **L4** — drive sweep, `gamma`, and the Evidence E silent endpoint (~4 min + ~2 min)
  - 4 rates × 3 seeds × 1,500 ms on the 258-ORN set; yields `gamma` (D3), D1–D6, `sigma_transient`
    (B3), and measures S1/S2/S3. Confirms the task 5 pre-fix failure by measurement.
  - Also re-run the two-point drive sweep at **`w_syn = 0.0344`** (2 extra points, ~2 min):
    Evidence E swept `w_syn` at a single drive, so the silent endpoint's `gamma` was never
    measured, and requirement 2.23 requires demonstrating that it fails `S` too.
  - Report each hop rate alongside the 454.5 Hz ceiling and its fraction thereof; flag any relay
    above 0.25 of ceiling as saturated and therefore unable to encode stimulus magnitude.
  - Also run **B2 `sigma_drive`** here (paired runs, one generation, ≥ 20 injection times,
    ≥ 3 seeds, ~40 s per seed) since it needs the driven steady state.
  - Record explicitly that no `w_syn` value tested yields an input-encoding regime — the network
    goes saturated → silent — so **F6 stays falsified as a sufficient fix**.
  - ```powershell
    python -m simulation.environment.run_stability_diagnostic --rung L4 --seeds 0,1,2 --out stability_L4.csv
    ```
  - _Requirements: 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.15, 2.22, 2.23_

- [ ] 17. Rung **L5** — ignition threshold `k*` and criticality classification (~6.5 min)
  - Bisection over `k` in {1, 2, 4, ..., 258} driven connected ORNs, 3 seeds; the minimum drive or
    minimum number of driven connected neurons at which self-sustaining activity appears.
  - Classify the network subcritical / critical / supercritical from `k*` **together with**
    `sigma_rest` (task 11) and `sigma_drive` (task 16). Reported, not gated.
  - ```powershell
    python -m simulation.environment.run_stability_diagnostic --rung L5 --seeds 0,1,2 --out stability_L5.csv
    ```
  - _Requirements: 2.16, 2.22_

- [ ] 18. Rung **L7** — R-TASK-2 full-scale conformance and the float32/float64 pair
  - Settles the drift-at-scale half of **H1** and requirement 2.11's "at full scale" clause.
  - _Requirements: 2.11_

  - [ ] 18.1 Write R2, an independent CSR float64 reference
    - R1 cannot be used at scale for two independent reasons: it builds
      `np.asarray(conn.w.todense())`, which at full scale is `139,255² × 8 B = 155 GB` (77 GB in
      float32) against ~3.8 GB available — dense-bound to roughly 20,000 neurons (3.2 GB), which
      still does not fit; and its inner loop is per-neuron Python, so a 2 s full-scale run is
      2.8e9 iterations.
    - R2 keeps R1's deliberate independence from the engine (no shared code, no shared constants
      beyond `LIFParams`) but uses **CSR** instead of dense `W`, vectorising only what cannot
      change semantics. New file `tests/test_reference_conformance.py`.
    - Validate R2 against R1 for **bit-identical spike trains** on the existing 120-neuron
      LPLC2/DNp01 case before trusting it at any scale.
    - _Requirements: 2.11_

  - [ ] 18.2 Full-scale conformance window and the paired float32/float64 run
    - Run R2 at full scale over a **short window only** — target 50 ms, calibrate the achievable
      window empirically before committing to a number — and report the deviation of the engine's
      float32 trajectory from R2. **State plainly that the comparison covers a ~50 ms window and
      not a full-length run.**
    - Paired float32/float64 full-scale 500 ms run (~50 s; float64 doubles the four edge-sized
      arrays at ~30 MB each and roughly doubles step cost).
    - **Bit-identity will break at some step through chaotic divergence; that is expected and is
      not itself evidence.** The falsification metric is the **statistics** — plateau rate, active
      fraction, `sigma` — plus the step of first divergence as a diagnostic. H1 is confirmed only
      if float32 shows a systematic positive bias in `v` and the float64 run has a materially lower
      plateau rate or fails to ignite.
    - `test_float32_and_float64_give_identical_spikes` already proves identity on a
      few-thousand-neuron subgraph, so H1 is **only** about scale.
    - Write the H1 drift verdict back into the task 12 gate record.
    - _Requirements: 2.11_

---

## Phase 7 — Fix selection and implementation (only after the task 12 gate passes)

- [ ] 19. Select the fix path from F1–F7 against the gate verdicts
  - **Blocked on task 12. Conformance restoration is attempted before any unpublished mechanism.**
  - **F1 — correct an engine deviation.** *Gate:* H1, H2, H3 or H5 survives. *Provenance:*
    restores **category B**; a bug fix, not a new mechanism.
  - **F2 — match the reference drive protocol.** *Gate:* R-TASK-1 answers Q-A/Q-B/Q-G differently
    from what we do. Variants: `set_poisson` stops zeroing `rfc_len` **and H5's
    `v > v_th AND rfc_left == 0` guard is added in the same change**; deliver drive to `g` instead
    of `v`; deliver drive as a pulse. *Provenance:* **category B restoration if and only if
    R-TASK-1 shows the reference differs** — otherwise the change is **category C, our deviation
    from the published parameterization**, and must be labelled so.
    Context for F2: `w_syn * f_poi = 68.75 mV` added directly to `v` against a 7.0 mV gap is a
    ~10x forcing margin, so every Poisson event is a guaranteed spike; and `set_poisson` zeroing
    `rfc_len` leaves a driven ORN with no refractory period at all, so at 150 Hz commanded drive
    (`p = 0.015` per step) it can fire on consecutive 0.1 ms steps.
  - **F3 — divisive gain control / per-neuron input normalisation.** *Gate:* H6 survives **and**
    F1/F2 are insufficient. *Provenance:* **OUR ENGINEERING**, absent from Shiu et al. 2024.
  - **F4 — spike-frequency adaptation.** *Gate:* as F3. *Provenance:* **OUR ENGINEERING**.
  - **F5 — short-term synaptic depression.** *Gate:* as F3. *Provenance:* **OUR ENGINEERING**.
  - **F6 — uniform gain reduction (`w_syn` scaling): ALREADY FALSIFIED AS A SUFFICIENT FIX AND MAY
    NEVER BE APPLIED ALONE.** Evidence E measured 0.2750 / 0.1375 / 0.0688 / 0.0344 giving
    4.192 / 1.649 / 0.639 / 0.040 Hz and found the network passes from saturated to silent with
    **no input-encoding regime in between**; hop3 left/right contrast moved only 0.17/0.20/0.61 pp
    and DNa01 went silent from `w_syn = 0.0688` down. Requirement 2.23 makes an encoding regime
    mandatory. F6 is retained **only** as a component of a composite, and any composite containing
    it must still pass S1. **This conclusion is not revisited.**
  - **F7 — do not fix; re-scope. A legitimate terminal outcome.** *Gate:* H6 and H9 both survive,
    F1 and F2 find no deviation, and no reference-conforming configuration passes `S`. Then record
    the honest result per task 12's F7 condition. **This outcome stays on the table throughout —
    "no fix, correctly characterised" is preferable to an unattributable fix.**
  - Record the mechanistic argument for why F3–F5 can succeed where F6 cannot, **as an argument to
    be tested and not assumed**: F6 scales `sigma` uniformly, and near criticality a branching
    process has divergent susceptibility, so the `w_syn` window in which the network is both alive
    and non-saturating is vanishingly narrow — exactly the saturated-to-silent jump Evidence E
    observed. F3–F5 are **activity-dependent**, making `sigma_effective(R)` a decreasing function
    of rate, which turns the operating point into a stable fixed point with real dynamic range.
    Requirement 2.23 demands this be demonstrated by measuring `gamma`, not argued.
  - Preservation constraint on the **form** of any fix: an unconditional change to the integration
    kernel or to `w_syn` breaks Property 2, whereas an activity-dependent mechanism inactive at the
    low rates subgraph tests produce can satisfy it — **but only if verified, not assumed**.
  - _Requirements: 2.8, 2.10, 2.18, 2.22, 2.23, 3.3, 3.6_

- [ ] 20. Fix for the stimulus-independent whole-brain attractor

  - [ ] 20.1 Implement the selected fix
    - Apply the path chosen in task 19, in `simulation/engine/lif_engine.py`; touch
      `brain/neuron_models/lif.py` only if a published constant changes, and then record the
      deviation with its justification.
    - If F2 is chosen, add H5's `rfc_left == 0` threshold guard in the **same** change.
    - Label any mechanism absent from Shiu et al. 2024 **OUR ENGINEERING** at the point of
      definition, with its provenance category.
    - No topology edits, no added or removed connections, no synaptic sign changes; `_w_sign`
      preserved and no synapse crossing zero.
    - _Bug_Condition: `isBugCondition(X)` — full v783 connectome, `duration_ms >= 500`, non-empty
      driven set with at least one `outDegree(n) > 0` neuron_
    - _Expected_Behavior: `expectedBehavior(result)` = criterion `S` — `gamma >= 0.5`,
      `active_fraction >= 0.02`, `brain_rate_hz >= 0.5`, `dna01_rate_hz >= 1.0`,
      `max_hop_rate_hz <= 113.6`, `sigma_rest <= 1.0`, `sigma_drive <= 1.0`_
    - _Preservation: Preservation Requirements from design — bit-identical for all
      `NOT isBugCondition(X)`: every subgraph run, every run under 500 ms, every non-full-v783
      connectome; all 86 tests; real FlyWire topology and signs; bilateral convergence uncorrected;
      cumulative `spike_counts` semantics_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.10, 2.22, 2.23, 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.10, 3.11, 3.13_

  - [ ] 20.2 Rung **L6** — evaluate the candidate against the full criterion `S`
    - **Property 1: Expected Behavior** - Sensory drive must modulate whole-brain activity
    - **IMPORTANT**: re-run the **SAME** test from task 5 — do **not** write a new one. It already
      encodes criterion `S`; when it passes, the expected behaviour is satisfied.
    - L6 = L2 + L4 + L5 for this candidate, ~15 min per candidate, user-executed:
      ```powershell
      python -m simulation.environment.run_stability_diagnostic --rung L6 --seeds 0,1,2 --out stability_L6_<candidate>.csv
      ```
    - **A candidate that lowers activity without reaching `gamma >= 0.5` is REJECTED**, per
      requirement 2.23 and the F6 falsification. Repeat 20.1/20.2 for the next candidate in the
      F1 → F2 → F3/F4/F5 order.
    - Report every criterion value with its provenance category, and report the ignition threshold,
      criticality classification, `sigma_transient`, time-to-plateau and D8 alongside.
    - **EXPECTED OUTCOME: test PASSES (confirms the bug is fixed).** Then remove the
      `xfail(strict=True)` markers from the task 5 stability test and the task 6 specificity test
      and confirm both pass under `-m wholebrain`.
    - Amplification must now be bounded and consistent with the measured branching ratio rather
      than the current ~14,900 brain spikes per injected spike.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.16, 2.22, 2.23, 2.31, 2.32_

  - [ ] 20.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Sub-threshold-scale behaviour is bit-identical
    - **IMPORTANT**: re-run the **SAME** golden vectors and property test from task 7 — do **not**
      write new ones.
    - Assert bit-identity of `spike_counts` across the full generated domain (subgraph 50–4,000,
      driven 1–50, rate 1–300 Hz, duration 5–400 ms, both dtypes, seeds 0–1000).
    - Confirm unchanged: `test_matches_literal_reference_implementation`, `test_determinism`,
      `test_float32_and_float64_give_identical_spikes`, the ~4,000-neuron escape-subnetwork tests,
      and the bilateral convergence figures (hop0 77.85% → 2.11% / 6.41% / 8.93%) **reproduced,
      not improved**.
    - Confirm `check_invariants=False` runs stay bit-identical, `plasticity_enabled` stays false
      and `wdata` is unmodified.
    - **EXPECTED OUTCOME: tests PASS (confirms no regressions).** Any candidate that changes
      behaviour outside the bug condition is **rejected on preservation grounds regardless of how
      well it stabilises the full network** — return to task 19.
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 3.7, 3.10, 3.11, 3.12, 3.13_

  - [ ] 20.4 Re-derive the retracted conclusions or leave them retracted
    - If a fix landed, re-derive any Phase 1–3C conclusion before it is cited as evidence again;
      if F7 is the outcome, leave the retractions standing and record the re-scoped claim set.
    - Readout dynamic range (task 13.1) stays a separate defect and is still not claimed as fixed.
    - _Requirements: 2.24, 2.26, 2.28_

---

## Phase 8 — Checkpoint

- [ ] 21. Checkpoint - Ensure all tests pass
  - `python -m pytest tests` (the pinned invocation) — all 86 existing tests plus the additions,
    default budget ~34 s.
  - `python -m pytest tests -m wholebrain` — ~12 min, user-executed, memory ceiling 1 GB.
  - Confirm the baseline hash, seeds and parameters are recorded for every reported number, that
    every `__TBD__` verdict is filled or explicitly CONDITIONAL, and that every category-C
    mechanism and criterion constant is labelled OUR ENGINEERING.
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 2.20, 2.22, 3.3, 3.4, 3.5, 3.14_
