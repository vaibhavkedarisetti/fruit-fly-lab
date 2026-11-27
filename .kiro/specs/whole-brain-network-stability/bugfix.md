# Bugfix Requirements Document

## Introduction

The whole-brain LIF simulation over the real FlyWire FAFB v783 connectome (139,255 neurons, 3,732,460 connections, 50,666,648 synapses) enters a self-sustaining, stimulus-independent activity state. Once ignited, whole-brain activity barely depends on the sensory drive, so no sensory information is encoded and no descending readout can carry a signal.

Measured directly against the loaded connectome, not estimated:

- Driving **one** real ORN (`odor_vinegar`, left index 1574) at 20 Hz for 2 s injects 39 spikes and produces 579,293 brain spikes across 9,002 neurons — roughly **14,900 brain spikes per injected spike**. Reproduced on three of four single ORNs tested (L[0] 579,293; L[64] 1,072,586; R[64] 577,384). R[0] (index 164) produced 39 spikes and no propagation: it is an anatomically isolated ORN, which is why an earlier single-ORN probe appeared to contradict this. The discrepancy is resolved; the ignition result stands.
- Going from 2 to 258 driven ORNs is **123x more injected spikes for +6.5% total activity**. Holding 258 ORNs and sweeping per-neuron drive over 1.0 / 5.0 / 20.3 / 150.0 Hz gives whole-brain rates of 4.09 / 4.15 / 4.25 / 4.76 Hz — a **150x input range for +16% output**. Activity reaches plateau inside the first 500 ms bin and is flat thereafter.
- The first relay clamps: hop1 sits at 230 Hz at 20.3 Hz drive and 254 Hz at 150 Hz drive, roughly half the absolute refractory ceiling of 1000/2.2 = 454 Hz. Whole-brain activation is 6.8–6.9% of neurons in every condition.
- Reducing `w_syn` scales total activity down monotonically (4.192 → 1.649 → 0.639 → 0.040 Hz) but stimulus dependence never returns. The network goes from **saturated to silent without passing through an input-encoding regime** at any tested value.
- The runaway is not a regression from the working tree: the committed HEAD `lif_engine.py` and the modified version produce bit-identical results (579,293 spikes / 9,002 active for single-ORN ignition; 1,167,389 / 9,491 / 4.192 Hz for 258 ORNs at 20.3 Hz).
- The existing suite cannot see it: 86 tests pass in 26 s, and every LIF engine test runs on a subgraph (200, 50, 300, a 120-neuron LPLC2/DNp01 reference-conformance graph, a 150-neuron determinism graph, a ~4,000-neuron escape subnetwork). The full 139,255-neuron network is never simulated in any test.

### Scientific questions

**Q0 — Does the whole-brain simulation encode its sensory input at all, or does it occupy a stimulus-independent attractor in which every neural measurement to date is uninterpretable?**

Q0 decomposes into the sub-questions every clause below is tagged against:

- **Q1 — Can the reported numbers be reproduced from a clean tree?** About 2,600 uncommitted lines span `brain/motor/descending.py`, `config.py`, `fly/body/fly_body.py` and `simulation/engine/lif_engine.py`. Until a baseline exists, no stability number is evidence about anything.
- **Q2 — Is the network subcritical, critical or supercritical, quantitatively?** The discriminating observables are input-output gain over a drive range, the per-spike branching ratio, and the ignition threshold. None has ever been measured at full scale.
- **Q3 — The root-cause fork: does our engine deviate from the published model in a way that amplifies activity, or is the published parameterization genuinely unstable under sustained whole-brain Poisson drive, a regime Shiu et al. avoided?** The fix differs completely between the two, and the project's honesty constraints make the distinction mandatory.
- **Q4 — Which specific engine mechanisms deviate, if any?** Integration coefficients at full scale, refractory-count bookkeeping, refractory-period `g` handling, and how Poisson drive is applied to `v`.
- **Q5 — Which measurements are real connectome properties that must be recorded rather than repaired?** Bilateral ORN convergence, E/I balance, degree and synapse-per-connection statistics, isolated ORNs.
- **Q6 — What is the provenance of any stabilisation applied?** A mechanism not in Shiu et al. 2024 is our engineering and can never be presented as the published model.
- **Q7 — Which prior conclusions are invalidated, and what replaces them?** Phase 1 through Phase 3C were all measured inside the attractor.
- **Q8 — What regression coverage detects this class of defect?** Evidence G shows the current suite structurally cannot.

This spec is strictly **upstream** of `.kiro/specs/phase3b-forward-readout-diagnostic`. That spec audits the forward/steering readout; its tasks were never executed and its preservation guarantee no longer holds because the shared modules were modified. No readout audit is meaningful until the network encodes its input, so this spec must not duplicate its scope.

### Bug condition and property

**C(X) — the bug condition.** X is a whole-brain drive configuration: a set of driven sensory indices, a per-neuron Poisson rate, a duration and a seed.

```pascal
FUNCTION isBugCondition(X)
  INPUT:  X of type WholeBrainDrive
  OUTPUT: boolean

  // Sustained Poisson drive applied to the FULL network,
  // i.e. the regime in which ignition has been demonstrated.
  RETURN X.network = FULL_CONNECTOME_V783
     AND X.duration_ms >= 500
     AND X.driven_indices IS NON-EMPTY
     AND EXISTS n IN X.driven_indices WHERE isConnected(n)
END FUNCTION
```

`isConnected(n)` excludes the anatomically isolated case (R[0], index 164), which is a real connectome fact and not an exception to the bug.

**P(result) — the property required of the fixed engine F'.** Stated as a gain over a drive range, and deliberately not satisfiable by silencing:

```pascal
// Property: Fix Checking - sensory drive must modulate network activity
FOR ALL X WHERE isBugCondition(X) DO
  lo   <- F'(X with drive = r_lo)
  hi   <- F'(X with drive = r_hi)      // r_hi / r_lo spans a stated range
  ASSERT gain(lo, hi) >= GAIN_MIN      // input-output gain threshold
     AND lo.active_fraction  >= FLOOR  // NOT satisfiable by silencing
     AND hi.hop1_rate_hz     <= CEIL   // NOT satisfiable by saturation
     AND branching_ratio(F', X) <= 1.0
END FOR
```

`GAIN_MIN`, `FLOOR`, `CEIL` and the drive range are to be fixed in design; the current state, where a 150x input change yields +16% output, must fail the criterion, and both the saturated (`w_syn=0.275`) and silent (`w_syn=0.0344`) endpoints of Evidence E must also fail it.

**Preservation goal.**

```pascal
// Property: Preservation Checking
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
END FOR
```

Every subgraph regime the 86 existing tests exercise (50 to ~4,000 neurons, short windows, the 120-neuron literal-reference comparison) falls under `NOT isBugCondition(X)` and must be bit-identical after the fix.

## Bug Analysis

### Current Behavior (Defect)

Clause tags name the sub-question served and, where relevant, whether the observation discriminates an engine defect from a property of the published model.

**The network occupies a stimulus-independent attractor (Q0 — does not yet discriminate; symptom only).**

1.1 WHEN a single connected ORN is driven at 20 Hz for 2 s on the full connectome THEN the system converts 39 injected spikes into 579,293 brain spikes across 9,002 neurons, a gain of roughly 14,900 brain spikes per injected spike.

1.2 WHEN the number of driven ORNs is increased from 2 to 258 at fixed 20 Hz THEN the system responds to 123x more injected spikes with only +6.5% total activity (1,091,897 → 1,163,027 spikes, 3.920 → 4.176 Hz), so the observable does not carry stimulus magnitude.

1.3 WHEN per-neuron drive on 258 ORNs is swept over 1.0 / 5.0 / 20.3 / 150.0 Hz THEN the system produces whole-brain rates of 4.09 / 4.15 / 4.25 / 4.76 Hz, a 150x input range for +16% output.

1.4 WHEN activity is binned in time THEN the system reaches its plateau within the first 500 ms bin and is flat thereafter, so no transient carries stimulus onset information either.

1.5 WHEN whole-brain activation is measured in any of these conditions THEN the system activates 6.8–6.9% of neurons regardless of drive, so the size of the active population is also stimulus-independent.

1.6 WHEN per-hop rates are measured by BFS hop distance from the driven ORNs THEN the system clamps at the first relay: at 20.3 Hz drive hop0 20.15 Hz (n=258), hop1 230.09 Hz (n=127), hop2 41.11 Hz (n=6,621), hop3 12.97 Hz (n=18,161); a 7.4x increase to 150 Hz drive moves hop1 by +10%, hop2 by +15% and hop3 by −1%.

1.7 WHEN hop1 is compared with the absolute refractory ceiling of 1000/2.2 ms = 454 Hz THEN the system is running that relay at roughly half its maximum possible rate, so it has no headroom in which to represent a stronger stimulus.

1.8 WHEN `w_syn` is reduced by `dataclasses.replace` over 0.2750 / 0.1375 / 0.0688 / 0.0344 THEN the system scales total activity monotonically down (4.192 / 1.649 / 0.639 / 0.040 Hz) while stimulus dependence never returns, passing from saturated to silent with no input-encoding regime in between; hop3 left/right contrast changes by only 0.17 / 0.20 / 0.61 pp between matched and extreme drive, and DNa01 goes silent from `w_syn=0.0688` downward.

1.9 WHEN a single-ORN probe is used to test for ignition THEN the system's answer depends on which ORN was chosen, because R[0] (index 164) is anatomically isolated and yields no propagation, and no connectivity precheck distinguishes an isolated seed from a genuine absence of ignition.

**The root-cause fork is undecided, so no fix can be justified (Q3 — this is the discriminating gap).**

1.10 WHEN the runaway is discussed THEN the system offers no experiment that separates hypothesis (i) "our engine deviates from the published model in a way that amplifies activity" from hypothesis (ii) "the published parameterization is genuinely unstable under sustained whole-brain Poisson drive and Shiu et al. avoided this regime", so any stabilisation applied now would be unattributable.

1.11 WHEN conformance to the reference implementation is asserted THEN the system verifies it only on a 120-neuron LPLC2/DNp01 subgraph in `test_matches_literal_reference_implementation`, so the exact linear integration coefficients (`a = 1/t_mbr = 0.05`, `b = 1/tau = 0.2`, `dt = 0.1`, `_ev = exp(-0.005) = 0.995012`, `_eg = exp(-0.02) = 0.980199`, `_kg = a*(_eg-_ev)/(a-b) = 0.0049378`) are never checked at full scale or in float32 across 139,255 neurons.

1.12 WHEN `LIFEngine.step` maintains `_n_refractory` THEN the system recomputes it as `count_nonzero(rfc_left[ridx])` over only the previously-refractory subset and then increments it on spikes, so if that count can reach zero while neurons are still refractory, the `if self._n_refractory:` guard skips the refractory hold entirely for that timestep and refractory neurons are integrated.

1.13 WHEN a neuron is refractory THEN the system applies `g += pending` before taking the `g_hold` snapshot and restores `g_hold` after `g *= _eg`, so refractory neurons accumulate incoming synaptic drive without decaying it, and the magnitude of the resulting inflation of drive is unquantified.

1.14 WHEN external drive is delivered THEN the system adds `w_syn * f_poi = 68.75 mV` directly to `v` against a 7.0 mV threshold gap (`v_0 = -52`, `v_th = -45`), i.e. every Poisson event forces a spike with roughly 10x margin, and `set_poisson` additionally sets `rfc_len = 0` for driven neurons so they never go refractory; whether the reference applies PoissonInput to `v` in exactly this way is asserted from the reference `default_params` but never verified against reference behaviour at scale.

1.15 WHEN network gain is discussed THEN the system reports structural quantities — mean 13.57 synapses per connection giving a 3.73 mV `g` step at `w_syn = 0.275` against a 7.0 mV gap, mean out-degree 3,732,460/139,255 ≈ 26.8 — but never measures the actual per-spike branching ratio, which is the quantity that determines whether activity is self-sustaining.

1.16 WHEN ignition is discussed THEN the system never measures an ignition threshold, so subcritical, critical and supercritical regimes cannot be told apart and there is no way to know how far from criticality any candidate fix lands.

1.17 WHEN excitation/inhibition balance is cited THEN the system reports 62.33% excitatory / 37.67% inhibitory connections (61.45%/38.55% synapse-weighted) with `nt_resolved` counts ACH 94,182, GLUT 22,726, GABA 18,389, SER 1,435, DA 791, OCT 106 and zero unknown, following the Shiu convention, but does not confirm at scale that no neuron ends up with sign 0 or an inverted sign, so E/I is treated as verified without a full-scale control.

1.18 WHEN the model is run THEN the system provides no spike-frequency adaptation, no synaptic depression, no global gain control and no per-neuron input normalisation, and does not record whether that absence is faithful to Shiu et al. or a gap in our transcription.

1.19 WHEN the published model is invoked as justification THEN the system does not establish whether the reference implementation ever drives large neuron sets for sustained windows, so it cannot say whether sustained whole-brain drive is a defect in our engine or an untested regime of the published model.

**No reproducible baseline exists (Q1 — prerequisite for every other clause).**

1.20 WHEN current stability numbers are cited THEN the system cannot reproduce them from a clean tree, because roughly 2,600 uncommitted lines are spread across `brain/motor/descending.py`, `config.py`, `fly/body/fly_body.py` and `simulation/engine/lif_engine.py`.

1.21 WHEN HEAD-versus-working-tree equivalence is claimed THEN the system has verified it for `lif_engine.py` only (bit-identical: 579,293 / 9,002 single-ORN; 1,167,389 / 9,491 / 4.192 Hz for 258 ORNs at 20.3 Hz), leaving the other three modified modules unaudited, so "the runaway is inherent to the committed model" is established for the engine but not for the readout, config and body layers that produce the reported downstream numbers.

**No stability criterion exists, and the obvious ones are gameable (Q2, Q6).**

1.22 WHEN a fix is proposed THEN the system has no quantitative, falsifiable stability criterion to test it against, so "fixed" is not currently a decidable statement.

1.23 WHEN total activity is used as the health metric THEN the system would score `w_syn = 0.0344` (0.040 Hz whole-brain, hop1 2.8 Hz, DNa01 silent) as an improvement, although Evidence E shows that state encodes no input either, so an activity-reduction criterion is satisfiable by silencing the network.

**Prior neural conclusions were measured inside the attractor (Q7).**

1.24 WHEN results from Phase 1 through Phase 3C are cited THEN the system presents them as neural conclusions, although every one was measured while the network was in the stimulus-independent state and is therefore uninterpretable.

1.25 WHEN the Phase 3C headline "no DNa asymmetry above the matched-rate null" is cited THEN the system presents it as evidence about chemotaxis, although the null and bio arms had identical U distributions precisely because both sampled the same stimulus-independent state, so the comparison could not have come out any other way.

1.26 WHEN the Phase 3C activation U is reported THEN the system reports a median of 0.3261 (null) and 0.3298 (bio) which equals `(2/62 − 1/61)/(2/62 + 1/61) = 0.326087` exactly — one DNa neuron firing 2 spikes per second against the other firing 1 — and across 1,820 measurements the activation took only 16 distinct values with 28% of windows having one side at exactly zero, so the observable had no dynamic range regardless of network state.

1.27 WHEN Phase 3C locomotion is reported THEN the system reports mean forward 0.054 with max 0.158 against `MIN_LOCOMOTION_DRIVE = 0.12`, so only 1.5% of windows crossed the walking threshold and no navigation was actually tested.

1.28 WHEN `.kiro/specs/phase3b-forward-readout-diagnostic` is consulted THEN the system presents a preservation guarantee that no longer holds, because the shared modules it was written against have since been modified, and nothing records that a readout audit is meaningless while the network is stimulus-independent.

**Bilateral convergence is real anatomy but is not recorded as such (Q5 — real connectome effect, must NOT be fixed).**

1.29 WHEN left ORNs are driven at 40 Hz and right at 5 Hz (78% input contrast) versus matched 20.3/20.3 Hz THEN the system produces per-hop left/right contrast of hop0 77.85%, hop1 2.11%, hop2 6.41%, hop3 8.93% (extreme) against hop0 0.46%, hop1 2.33%, hop2 6.28%, hop3 9.06% (matched), i.e. downstream contrast is unchanged within 0.2 pp under an 8:1 input ratio, and the output does not state that the residual 2/6/9% asymmetries are structural (3,181 vs 3,418 neurons at hop2; 9,029 vs 9,073 at hop3) rather than stimulus-driven.

1.30 WHEN that lateral collapse is observed THEN the system does not record the verified mechanism — 94.1% of left-ORN and 96.3% of right-ORN synapses terminate on first-order targets receiving input from BOTH antennae, across 63 shared targets including DM1_lPN, VA2_adPN, DM4_adPN, DM2_lPN (splitting 49.7/50.3) and the lLN2 and il3LN6 local neurons — as expected adult Drosophila anatomy, so it is exposed to being mistaken for a defect and "corrected".

**No whole-brain regression coverage (Q8).**

1.31 WHEN the test suite runs THEN the system passes all 86 tests in 26 s while simulating only subgraphs — `c.subgraph(np.arange(200))`, `c.subgraph(np.arange(50))`, `c.subgraph(np.arange(300))`, a 120-neuron LPLC2/DNp01 graph, a 150-neuron determinism graph and a ~4,000-neuron escape subnetwork — and never simulates the full 139,255-neuron network.

1.32 WHEN `tests/test_circuits.py` asserts specificity (bitter GRN drive produces exactly zero proboscis motor neuron spikes) THEN the system verifies it at a scale where the runaway does not manifest, so a passing suite is not evidence that the whole-brain model is specific.

### Expected Behavior (Correct)

Each clause corresponds to the identically numbered defect clause. Every clause states the sub-question it answers and, where applicable, what it discriminates.

**A quantified, stimulus-dependent network (Q0 — the outcome the rest of the spec serves).**

2.1 WHEN a single connected ORN is driven on the full connectome THEN the system SHALL report the spikes-out-per-spike-in amplification factor as a primary, categorised observable, and after the fix SHALL show an amplification that is bounded and consistent with the measured branching ratio rather than the current ~14,900.

2.2 WHEN the number of driven ORNs is swept THEN the system SHALL report total activity as a function of injected spike count and SHALL show a monotone dependence that exceeds a stated minimum gain over the swept range, so that stimulus magnitude is recoverable from network activity.

2.3 WHEN per-neuron drive rate is swept over at least the 1.0–150.0 Hz range THEN the system SHALL report the input-output gain across that range against a stated threshold, and SHALL treat the present +16% response to a 150x input change as a failing value.

2.4 WHEN activity is binned in time THEN the system SHALL report the time course, SHALL distinguish a stimulus-locked response from an ignition plateau, and SHALL report the time-to-plateau so that the "flat after the first 500 ms bin" signature is detectable automatically.

2.5 WHEN whole-brain activation is measured THEN the system SHALL report the active fraction per condition and SHALL show that it varies with drive, so the current 6.8–6.9% invariance is visible as a failure when it recurs.

2.6 WHEN per-hop rates are measured THEN the system SHALL report hop0–hop3 mean rates and populations by BFS hop distance over the unsigned adjacency for every drive condition, and SHALL show that hop1 and hop2 respond to drive by more than the current +10% and +15% under a 7.4x increase.

2.7 WHEN any hop rate is reported THEN the system SHALL report it alongside the 1000/`t_rfc` = 454 Hz absolute refractory ceiling and flag any relay running above a stated fraction of that ceiling as saturated and therefore unable to encode stimulus magnitude.

2.8 WHEN gain parameters such as `w_syn` are swept THEN the system SHALL report, at each value, both total activity and the input-output gain, and SHALL explicitly test the Evidence E finding that no tested value yields an input-encoding regime; if the fix is a parameter change, the system SHALL demonstrate an encoding regime rather than merely a lower activity level.

2.9 WHEN a seed neuron is selected for an ignition probe THEN the system SHALL check its out-degree against the loaded connectome first, SHALL exclude anatomically isolated neurons such as R[0] (index 164) from ignition statistics, and SHALL report which seeds were excluded and why, so an isolated seed can never be read as absence of ignition.

**The root-cause fork decided by experiment (Q3 — the discriminating requirement).**

2.10 WHEN the cause of the runaway is stated THEN the system SHALL have run experiments that discriminate hypothesis (i) engine deviation from hypothesis (ii) unstable published parameterization under sustained whole-brain drive, SHALL enumerate the candidate causes in 2.11–2.19 as hypotheses without pre-committing to any, and SHALL record for each hypothesis the measurement that supports or falsifies it together with the verdict.

2.11 WHEN conformance to the reference implementation is claimed THEN the system SHALL verify the exact linear integration coefficients at full scale — `_ev = exp(-a*dt)`, `_eg = exp(-b*dt)`, `_kg = a*(_eg-_ev)/(a-b)` with `a = 0.05`, `b = 0.2`, `dt = 0.1`, expected 0.995012, 0.980199, 0.0049378 — including the float32 hot-loop constants and accumulated drift over a full-length run across all 139,255 neurons, and SHALL report the deviation from a float64 reference trajectory. This clause discriminates engine defect from published-model property.

2.12 WHEN refractory bookkeeping is evaluated THEN the system SHALL determine whether `_n_refractory` can reach zero while any `rfc_left > 0`, SHALL do so by an invariant check comparing `_n_refractory` against `count_nonzero(rfc_left)` over the whole network at every step of a full-scale run, and SHALL report the number of timesteps on which the refractory hold was skipped. This clause discriminates engine defect from published-model property.

2.13 WHEN refractory `g` handling is evaluated THEN the system SHALL quantify the drive inflation caused by `g += pending` preceding the `g_hold` snapshot with `g_hold` restored after `g *= _eg`, SHALL express it as the mean and maximum extra `g` accumulated per refractory neuron per refractory period at full scale, and SHALL state whether that inflation is material to ignition against the 7.0 mV threshold gap. This clause discriminates engine defect from published-model property.

2.14 WHEN external drive is evaluated THEN the system SHALL confirm against the reference implementation whether PoissonInput is applied to `v` as we apply it, SHALL report `w_syn * f_poi = 68.75 mV` against the 7.0 mV gap as an approximately 10x forcing margin, and SHALL report the consequence of `set_poisson` zeroing `rfc_len` for driven neurons, including the effective maximum rate of a driven ORN at 150 Hz commanded drive. This clause discriminates engine usage from published-model parameterization.

2.15 WHEN network gain is characterised THEN the system SHALL measure the empirical per-spike branching ratio at full scale — postsynaptic spikes causally attributable per presynaptic spike — SHALL report whether it exceeds 1.0, and SHALL report it alongside the structural quantities (13.57 mean synapses per connection, 3.73 mV `g` step at `w_syn = 0.275`, mean out-degree 26.8). This clause is the primary discriminator between a subcritical and a supercritical network.

2.16 WHEN ignition is characterised THEN the system SHALL measure an ignition threshold — the minimum drive, or minimum number of driven connected neurons, at which self-sustaining activity appears — and SHALL classify the network as subcritical, critical or supercritical from the branching ratio and that threshold together.

2.17 WHEN E/I assignment is treated as a control THEN the system SHALL confirm at full scale that no neuron carries sign 0 or an inverted sign under the Shiu convention (ACH/DA/OCT/SER excitatory; GABA/GLUT inhibitory), SHALL report the per-connection and synapse-weighted E/I fractions (62.33%/37.67% and 61.45%/38.55%) and the `nt_resolved` counts as a control result, and SHALL NOT treat E/I as a prime suspect unless that control fails.

2.18 WHEN stabilising mechanisms are considered THEN the system SHALL record which of spike-frequency adaptation, synaptic depression, global gain control and per-neuron input normalisation are absent from Shiu et al. 2024 and which are absent only from our transcription, and SHALL treat any mechanism it adds under the provenance rules in 2.22 and 3.3.

2.19 WHEN the published model is invoked THEN the system SHALL establish, from the reference code `philshiu/Drosophila_brain_model`, the neuron-set sizes and drive durations the reference actually exercises, and SHALL state whether sustained whole-brain Poisson drive is an untested regime of the published model rather than a defect in our engine. This clause is the decisive evidence for hypothesis (ii).

**A reproducible baseline before any number counts as evidence (Q1).**

2.20 WHEN stability numbers are treated as evidence THEN the system SHALL first establish a clean reproducible baseline for the ~2,600 uncommitted lines across `brain/motor/descending.py`, `config.py`, `fly/body/fly_body.py` and `simulation/engine/lif_engine.py` — by committing, reverting or explicitly pinning each module — and SHALL record the exact revision, seeds and parameters that every reported number was produced under.

2.21 WHEN HEAD-versus-working-tree equivalence is claimed THEN the system SHALL extend the side-by-side comparison already done for `lif_engine.py` to the other three modified modules, and SHALL report for each whether results are bit-identical, so that "inherent to the committed model rather than introduced by working-tree edits" is established for the whole measurement path and not just the engine.

**A falsifiable stability criterion that silencing cannot satisfy (Q2, Q6).**

2.22 WHEN the fix is evaluated THEN the system SHALL apply a single quantitative, falsifiable stability criterion of the form given in the Introduction: a defined input-output gain measured over a stated drive range SHALL exceed a stated threshold; the measured per-spike branching ratio SHALL NOT exceed 1.0; the ignition threshold SHALL be reported; and each criterion value SHALL be reported with its provenance category (real connectome effect / engineering abstraction / readout artifact).

2.23 WHEN the stability criterion is defined THEN the system SHALL make it unsatisfiable by silencing the network, by requiring a minimum active fraction and a minimum absolute response alongside the gain, and SHALL demonstrate that both Evidence E endpoints fail it: the saturated state at `w_syn = 0.275` (4.192 Hz, hop1 230 Hz) and the silent state at `w_syn = 0.0344` (0.040 Hz, hop1 2.8 Hz, DNa01 silent).

**Prior conclusions retracted explicitly (Q7).**

2.24 WHEN any Phase 1 through Phase 3C neural conclusion is referenced THEN the system SHALL mark it as measured inside the stimulus-independent attractor and therefore uninterpretable, and SHALL require re-derivation after the fix before it is cited again as evidence.

2.25 WHEN the Phase 3C "no DNa asymmetry above the matched-rate null" result is referenced THEN the system SHALL state that it is not evidence about chemotaxis, and SHALL state the reason: the null and bio arms had identical U distributions because both sampled the same stimulus-independent state.

2.26 WHEN the Phase 3C activation is referenced THEN the system SHALL record that the median U of 0.3261 (null) and 0.3298 (bio) equals `(2/62 − 1/61)/(2/62 + 1/61) = 0.326087` exactly, that across 1,820 measurements the observable took only 16 distinct values, and that 28% of windows had one side at exactly zero, so the observable had no dynamic range independently of network state; the system SHALL treat readout dynamic range as a separate defect from network stability and SHALL NOT claim to have fixed it here.

2.27 WHEN the Phase 3C locomotion result is referenced THEN the system SHALL record that mean forward was 0.054 with max 0.158 against `MIN_LOCOMOTION_DRIVE = 0.12`, that only 1.5% of windows crossed the walking threshold, and that no navigation was tested.

2.28 WHEN `.kiro/specs/phase3b-forward-readout-diagnostic` is referenced THEN the system SHALL record that this spec is strictly upstream of it, that the readout audit is not meaningful until the network encodes its input, and that the phase3b preservation guarantee must be re-established against the baseline from 2.20; the system SHALL NOT duplicate the phase3b scope in this spec.

**Bilateral convergence recorded as a real connectome constraint (Q5 — real connectome effect).**

2.29 WHEN lateral contrast is measured THEN the system SHALL record the Evidence D result as a REAL CONNECTOME finding — hop0 77.85% input contrast collapsing to hop1 2.11%, hop2 6.41%, hop3 8.93%, within 0.2 pp of the matched 20.3/20.3 Hz condition — SHALL attribute the residual 2/6/9% asymmetries to structural population differences (3,181 vs 3,418 at hop2; 9,029 vs 9,073 at hop3), and SHALL NOT treat it as part of the bug.

2.30 WHEN the mechanism is reported THEN the system SHALL record that 94.1% of left-ORN and 96.3% of right-ORN synapses terminate on first-order targets receiving input from both antennae across 63 shared targets (DM1_lPN, VA2_adPN, DM4_adPN, DM2_lPN at 49.7/50.3, and the lLN2 and il3LN6 local neurons), SHALL state that this is expected adult Drosophila anatomy in which ORNs project bilaterally and the antennal lobe sums the two antennae, and SHALL frame it as a constraint on which readouts can ever work rather than a defect to repair.

**Whole-brain regression coverage (Q8).**

2.31 WHEN regression coverage is added THEN the system SHALL add at least one test that simulates the full 139,255-neuron network and asserts the stability criterion from 2.22 — including an ignition bound from a single connected seed and an input-output gain over a drive range — and SHALL keep runtime and memory bounded enough to run in the suite, marking it slow if required rather than omitting it.

2.32 WHEN specificity is asserted THEN the system SHALL add a whole-brain counterpart to the `tests/test_circuits.py` specificity assertion, so that "bitter GRN drive produces zero proboscis motor neuron spikes" is verified at the scale where the runaway manifests and not only on a subgraph.

### Unchanged Behavior (Regression Prevention)

Scientific and engineering constraints the fix must preserve.

3.1 WHEN the network is loaded and simulated THEN the system SHALL CONTINUE TO use the real FlyWire FAFB v783 connectivity exactly as loaded — 139,255 neurons, 3,732,460 connections, 50,666,648 synapses — with no topology edits, no added or removed connections, and no synaptic sign changes.

3.2 WHEN synaptic weights are modified by any mechanism THEN the system SHALL CONTINUE TO preserve every anatomical sign through `_w_sign`, SHALL CONTINUE TO forbid a synapse crossing through zero, and SHALL CONTINUE TO refuse to create or delete edges.

3.3 WHEN a stabilisation mechanism not present in Shiu et al. 2024 is applied THEN the system SHALL CONTINUE TO label it explicitly as OUR ENGINEERING with its provenance category, SHALL CONTINUE TO record any changed published constant together with the deviation and its justification, and SHALL CONTINUE TO never present either as the published model.

3.4 WHEN published constants are cited THEN the system SHALL CONTINUE TO carry the existing provenance headers and inline citations in `brain/neuron_models/lif.py` and `simulation/engine/lif_engine.py`, including the Shiu et al. 2024 reference (Nature 634:210-219, doi:10.1038/s41586-024-07763-9) and `philshiu/Drosophila_brain_model`.

3.5 WHEN the test suite runs THEN the system SHALL CONTINUE TO pass all 86 existing tests, including `test_matches_literal_reference_implementation` on the 120-neuron LPLC2/DNp01 subgraph, the determinism test on the 150-neuron subgraph, and the ~4,000-neuron escape subnetwork tests.

3.6 WHEN any configuration outside the bug condition is simulated — subgraphs of 50 to roughly 4,000 neurons, short windows, the existing reference-conformance comparison — THEN the system SHALL CONTINUE TO produce bit-identical results to the pre-fix engine.

3.7 WHEN bilateral ORN convergence is observed THEN the system SHALL CONTINUE TO preserve it uncompensated and uncorrected, with no rebalancing, offset or lateral-contrast enhancement applied anywhere in the path.

3.8 WHEN an anatomically isolated neuron such as R[0] (index 164) is encountered THEN the system SHALL CONTINUE TO report it as a real connectome fact and SHALL CONTINUE TO leave it unconnected.

3.9 WHEN E/I assignment is applied THEN the system SHALL CONTINUE TO follow the Shiu convention (ACH/DA/OCT/SER excitatory; GABA/GLUT inhibitory) with the existing `nt_resolved` mapping and zero unknown neurotransmitters.

3.10 WHEN the stability diagnostic runs THEN the system SHALL CONTINUE TO use no hand-written navigation controller, SHALL CONTINUE TO withhold target direction, food coordinates and distance-to-target from the neural simulation, and SHALL CONTINUE TO run with no reward and no runtime plasticity, leaving `plasticity_enabled` false and `wdata` unmodified for the whole diagnostic.

3.11 WHEN sensory drive is delivered THEN the system SHALL CONTINUE TO pass only neuron indices and firing rates into the brain through `set_poisson` / `clear_poisson`, resolved from the loaded connectome, with no synthetic neurons created.

3.12 WHEN the diagnostic is run twice with the same seed, revision and configuration THEN the system SHALL CONTINUE TO produce the same result, preserving the explicitly seeded RNG and deterministic neuron selection.

3.13 WHEN `engine.spike_counts` is read by existing callers THEN the system SHALL CONTINUE TO expose it as cumulative since `reset()`, so any windowed measurement added for the diagnostic is an addition rather than a change of documented engine semantics.

3.14 WHEN provenance is reported THEN the system SHALL CONTINUE TO state the standing scope limit that FAFB v783 is a brain connectome, that the VNC and leg/wing motor neurons are absent, and that everything downstream of the descending-neuron stage is an explicitly-labelled engineering body model and not a biological claim.

3.15 WHEN this spec proceeds THEN the system SHALL CONTINUE TO leave the scope of `.kiro/specs/phase3b-forward-readout-diagnostic` intact and unexecuted, treating that spec as downstream work rather than absorbing or superseding its clauses.
