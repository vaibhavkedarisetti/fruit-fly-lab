# Bugfix Requirements Document

## Introduction

The Phase 3B "TRUE SEPARATED FORWARD" diagnostic is meant to answer one scientific question:

**Q0 — In the Phase 3 spatial-odor experiment, was the observed forward locomotion a genuine bilateral locomotion command, or was it an arithmetic re-labelling of unilateral DNa01/DNa02 steering activity?**

The forward channel in `brain/motor/descending.py` is currently `forward = max(left_DNa, right_DNa) * FORWARD_GAIN` with `FORWARD_GAIN = 4.0`. The proposed replacement is `forward = mean(left_DNa, right_DNa) * FORWARD_GAIN`. Because `max(L, R) == (L + R) / 2 + |L - R| / 2`, the existing rule adds a term proportional to the bilateral *imbalance* into a channel labelled "forward". If that imbalance term dominates, then reported forward locomotion is a re-labelling of steering asymmetry, which is an artifact of our chosen readout arithmetic — not a connectome effect and not a biological forward-walking command.

Q0 decomposes into the sub-questions the requirements below are traced to:

- **Q1 — Does the diagnostic execute at all against the real modules?** A diagnostic that crashes answers nothing.
- **Q2 — Is the headline metric capable of returning a negative result?** A metric that is positive by construction cannot falsify Q0.
- **Q3 — Are the DN rates entering the readout measured over the window they are divided by?** If not, every reported rate and channel magnitude is wrong in the same direction, and saturation is a measurement artifact rather than a neural observation.
- **Q4 — Is the readout window long enough to estimate a rate from the actual DN population sizes?** DNa01 and DNa02 have one reconstructed neuron per side in v783; MDN has two per side.
- **Q5 — Is the spatial odor gradient above the Poisson noise floor?** If not, the closed-loop portion is uninterpretable regardless of which forward rule is used.
- **Q6 — Which comparisons are controlled and which are confounded by the closed loop?** Driving the body with a different forward rule changes the trajectory, which changes the sampled odor, which changes the sensory input.
- **Q7 — Is every reported number attributed to real connectome effect, engineering abstraction, or scaling/readout artifact?**

The current draft of the diagnostic fails on all seven counts. It cannot run, and if the adapter layer were repaired without further change it would produce numbers that cannot discriminate a connectome effect from an interface effect. The scope of this fix is: make the diagnostic run against the real APIs, make its headline measurement falsifiable, stop it inheriting the cumulative-spike-count measurement bug that is shared by every existing sandbox script, and make it report the sensory contrast and clipping state required to interpret its own output.

The cumulative-spike-count defect (1.10–1.13 below) is a pre-existing bug in the shipped sandbox scripts, not only in the draft. Its consequence is that all previously reported firing rates, motor-channel magnitudes and saturation observations from those scripts — including "mean forward = 0.989 identical across all three steering gains" — are suspect and must be re-derived before they are used as evidence for or against Q0.

## Bug Analysis

### Current Behavior (Defect)

What currently happens. Clause tags name the sub-question served and the category the defect belongs to.

**Adapter layer does not match any real API (Q1 — engineering abstraction).**

1.1 WHEN the draft constructs the simulation with `LIFEngine(seed=seed)` THEN the system raises a missing-argument error, because `LIFEngine.__init__` requires `connectome` as the first positional argument (`LIFEngine(connectome, params=DEFAULT, seed=0, dtype=np.float32)`).

1.2 WHEN the draft applies sensory drive via `engine.set_orn_drive(...)` THEN the system raises an attribute error, because no such method exists; the real interface is `engine.set_poisson(indices, rates_hz)` and `engine.clear_poisson()`, and it requires ORN indices resolved from the loaded connectome.

1.3 WHEN the draft supplies ORN drive without resolving per-side indices from the real connectome THEN the system cannot deliver a bilateral sensory difference at all, because left and right ORN populations are never distinguished; the real path is `BY_KEY["odor_vinegar"]` then `connectome.by_cell_types(modality.cell_types)` split on the `side` column.

1.4 WHEN the draft advances the brain with `engine.step(dt_ms)` THEN the system raises a signature error, because `step()` takes no arguments and always advances exactly one fixed `params.dt = 0.1 ms` timestep; a wall-clock interval must be advanced with `engine.run(duration_ms)`.

1.5 WHEN the draft constructs `DescendingReadout(engine)` THEN the system fails while reading `connectome.by_cell_type`, `connectome.neurons` and `connectome.idx`, because `DescendingReadout` is constructed from the connectome, not from the engine.

1.6 WHEN the draft calls `readout.channels()` with no arguments THEN the system raises a missing-argument error, because `channels(spike_counts, window_ms)` requires both.

1.7 WHEN the draft calls `body.reset(x=..., y=..., heading_deg=...)` and later reads `body.x`, `body.y` or `body.heading_deg` THEN the system raises argument and attribute errors, because `FlyBody.reset()` takes no arguments and position/heading live on `body.state` as `x_mm`, `y_mm`, `z_mm`, `heading_deg`, `speed_mm_s`.

1.8 WHEN the draft advances the body with `body.step(channels, dt_ms)` THEN the system raises an attribute error, because the real interface is `body.update(dt_ms=, channels=, t_ms=, escape_laterality=, proboscis_drive=)`.

1.9 WHEN `t_ms` is omitted from or faked in the body update THEN the system produces different locomotion behaviour from the same neural trace, because `FlyBody` uses `t_ms` to drive its `LOCOMOTION_PERSISTENCE_MS = 250.0` / `MIN_LOCOMOTION_DRIVE = 0.12` hysteresis, so the walking/resting decision depends on the timestamps the caller passes.

**Cumulative spike counts inflate every rate (Q3 — artifact of our readout).**

1.10 WHEN a sandbox script passes `engine.spike_counts` together with a per-chunk `window_ms` THEN the system computes a firing rate that grows with total elapsed time since `reset()`, because `spike_counts` is documented and implemented as cumulative ("Cumulative spike counts since episode reset"; `spike_counts[spk] += 1`; cleared only in `reset()`).

1.11 WHEN elapsed time grows THEN the system drives `act = hz / (hz + CHANNEL_HALF_MAX_HZ)` with `CHANNEL_HALF_MAX_HZ = 60.0` toward 1.0 for every tracked DN type, so all motor channels saturate as a function of wall-clock time rather than of neural activity.

1.12 WHEN the forward channel saturates at the 1.0 clip THEN the system reports a mean forward command near 1.0 that is invariant to steering gain, which is a plausible mechanical explanation for the previously reported "mean forward = 0.989 identical across all three steering gains" and is indistinguishable, in the current output, from a genuine neural result.

1.13 WHEN prior quantitative results from `run_sandbox.py`, `run_sandbox_phase2.py`, `run_sandbox_phase3_closed_loop_odor.py`, `run_sandbox_separated_four_channel_clean.py`, `run_sandbox_separated_locomotion_steering.py`, `run_sandbox_steering_gain_sweep.py`, `run_sandbox_bilateral_symmetry_diagnostic.py`, `run_sandbox_descending_readout_diagnostic.py` or `run_sandbox_forward_backward_only.py` are cited as evidence THEN the system offers no indication that those rate and channel magnitudes were computed from cumulative counts and are therefore inflated.

**Headline metric is a tautology (Q2 — artifact of our readout arithmetic).**

1.14 WHEN the draft reports `corr(forward_delta, unilaterality)` as the KEY result THEN the system reports a quantity that is positive by construction whenever `(L + R)` is roughly stable, because the pre-clip delta between the two rules is exactly `G * |L - R| / 2 = 2 * |L - R|` at `G = 4` while `U = |L - R| / (L + R)`, so the two variables are two monotone functions of the same `|L - R|`.

1.15 WHEN that correlation comes out positive THEN the system presents it as evidence about Q0, although a positive value carries no information and no possible dataset could have produced a negative one.

1.16 WHEN either forward rule reaches the `np.clip(..., 0.0, 1.0)` ceiling THEN the system silently collapses the between-rule delta to 0, so the headline metric is blind exactly in the saturated regime the prior runs reported (mean forward = 0.989), and no output distinguishes "the two rules agree" from "both rules are clipped".

1.17 WHEN the diagnostic runs THEN the system reports neither the distribution of the pre-clip unilaterality `U(t)` nor the deterministic contamination fraction `U / (1 + U)`, which are the quantities that actually answer Q0 and can return an arbitrarily small value.

**Readout window too short for the real DN population sizes (Q4 — real connectome fact plus readout artifact).**

1.18 WHEN the draft uses a `dt_ms = 1.0` readout window THEN the system converts a single-neuron spike count into a rate of either 0 Hz or 1000 Hz, because the loaded connectome contains DNa01 = 1 left + 1 right, DNa02 = 1 left + 1 right, MDN = 2 left + 2 right.

1.19 WHEN the rate is binary THEN the system produces an activation that is binary (0.0 or 1000/1060 = 0.943), so `left`, `right`, `forward`, the between-rule delta and every statistic derived from them are quantization noise rather than measurements.

1.20 WHEN a per-side DNa rate of 0.00 Hz is reported (for example DNa02_right = 0.00 Hz) THEN the system presents it in the same terms as a population measurement, although it means one individual reconstructed neuron did not cross threshold in the window, so the output invites reading single-cell threshold behaviour as population-level lateralization.

**Spatial odor gradient below the Poisson noise floor (Q5 — engineering abstraction of the environment).**

1.21 WHEN the closed-loop portion runs with the draft's field configuration (`peak_rate_hz = 75`, `decay_mm = 400`, food at (800, 350), start at (200, 350)) THEN the system presents a local rate of about 16.7 Hz and a gradient of about 0.042 Hz/mm at the start position, so the 0.7 mm antenna span yields a bilateral difference of about 0.03 Hz, or roughly 0.18%.

1.22 WHEN that bilateral difference is sampled over a 100 ms window with 129 ORNs per side THEN the system compares a ~0.18% signal against a Poisson standard deviation of about 7%, i.e. an instantaneous signal roughly 40x below the noise floor, and reports the resulting left/right activity as if it were a usable sensory contrast.

1.23 WHEN the achieved sensory contrast and its noise floor are not reported THEN the system allows a null steering result to be attributed to the brain or to the forward rule, when it is fully explained by the environment configuration.

**Cross-run comparison presented as a controlled measurement (Q6 — confound).**

1.24 WHEN the draft states that "both formulas are evaluated on the SAME neural trace" THEN the system makes a claim that holds only within a single run, because across its two runs the loop is closed: the forward rule changes the trajectory, the trajectory changes the sampled odor, and the sampled odor changes the sensory input to the brain.

1.25 WHEN the cross-run behavioural metrics (`end_dist`, `min_dist`, `heading_range`) are reported THEN the system presents them as an isolated effect of the readout interface, although they aggregate the interface change together with a different sensory history.

**Category attribution missing (Q7 — provenance).**

1.26 WHEN the diagnostic reports a number THEN the system does not state whether it is a real connectome effect, an engineering abstraction, or an artifact of our chosen scaling/readout, so an interface artifact and a connectome observation appear in the same undifferentiated output.

1.27 WHEN the forward channel is reported THEN the system does not restate that a forward command derived from DNa01/DNa02 is an engineering interface, leaving the output open to being read as a claim that DNa01/DNa02 are forward-walking neurons.

### Expected Behavior (Correct)

What should happen instead. Each clause corresponds to the identically numbered defect clause.

**Adapter layer matches the real APIs (Q1 — engineering abstraction).**

2.1 WHEN the diagnostic constructs the simulation THEN the system SHALL load the real connectome first and pass it positionally as `LIFEngine(connectome, seed=<explicit seed>)`, and SHALL record the seed in the output header.

2.2 WHEN the diagnostic applies sensory drive THEN the system SHALL use `engine.set_poisson(indices, rates_hz)` and SHALL call `engine.clear_poisson()` on teardown.

2.3 WHEN the diagnostic resolves the ORN population THEN the system SHALL take `BY_KEY["odor_vinegar"]`, call `connectome.by_cell_types(modality.cell_types)`, split on the `side` column, and SHALL deterministically balance the two sides to 129 real neurons each so the 137-vs-129 population-count difference cannot be mistaken for a bilateral signal; the system SHALL report the per-side counts actually used and SHALL NOT create synthetic neurons.

2.4 WHEN the diagnostic advances the brain by a wall-clock interval THEN the system SHALL call `engine.run(duration_ms)` and SHALL NOT pass any argument to `engine.step()`.

2.5 WHEN the diagnostic constructs the motor readout THEN the system SHALL call `DescendingReadout(connectome)`.

2.6 WHEN the diagnostic reads motor channels THEN the system SHALL call `readout.channels(spike_counts, window_ms)` with both arguments, passing the same window that the spike counts were measured over.

2.7 WHEN the diagnostic initialises or reads body state THEN the system SHALL call `FlyBody.reset()` with no arguments and SHALL read and write position and heading through `body.state.x_mm`, `body.state.y_mm`, `body.state.z_mm`, `body.state.heading_deg` and `body.state.speed_mm_s`.

2.8 WHEN the diagnostic advances the body THEN the system SHALL call `body.update(dt_ms=..., channels=..., t_ms=..., escape_laterality=..., proboscis_drive=...)`.

2.9 WHEN the diagnostic advances the body THEN the system SHALL pass the true accumulated simulation time as `t_ms` so that the `LOCOMOTION_PERSISTENCE_MS = 250.0` and `MIN_LOCOMOTION_DRIVE = 0.12` hysteresis behaves identically to the Phase 3 experiment being audited, and SHALL use the same `t_ms` sequence for both forward rules within a run.

**Per-window spike counts (Q3 — removes a readout artifact from every rate).**

2.10 WHEN the diagnostic converts DN spikes into a rate THEN the system SHALL use a per-window spike count — either a snapshot-and-difference of `engine.spike_counts` across the window, or an engine-level windowed accessor — so that the numerator and the `window_ms` denominator describe the same interval.

2.11 WHEN per-window counts are used THEN the system SHALL report the resulting DN rates and channel activations as a function of neural activity only, and SHALL show that they no longer drift monotonically toward the 1.0 ceiling with elapsed time; a rate/activation trace across the run SHALL be emitted so this can be checked directly.

2.12 WHEN a saturated forward command is observed THEN the system SHALL make it possible to attribute the saturation, by reporting the underlying per-window DN rates alongside the activation, so that "saturated because the cells fire above 60 Hz" is distinguishable from "saturated because counts accumulated".

2.13 WHEN the diagnostic reports its results THEN the system SHALL state explicitly that previously reported rate and motor-channel magnitudes from the existing sandbox scripts were computed from cumulative counts, are therefore suspect, and SHALL re-derive under the per-window measurement any prior quantity it relies on as evidence — including the "mean forward = 0.989, invariant across steering gains" observation.

**Falsifiable headline metric (Q2 — discriminates interface artifact from connectome effect).**

2.14 WHEN the diagnostic reports its primary result THEN the system SHALL report the distribution of the pre-clip unilaterality `U(t) = |L - R| / (L + R)` (at minimum: mean, median, spread, and the full time series), and SHALL NOT present `corr(forward_delta, unilaterality)` as the key result.

2.15 WHEN the correlation between the between-rule delta and unilaterality is reported at all THEN the system SHALL label it as algebraically determined rather than empirical, stating that `max(L, R) = (L + R)/2 + |L - R|/2` makes it positive by construction, so that it cannot be read as evidence.

2.16 WHEN either forward rule is evaluated THEN the system SHALL report, separately for the max rule and the mean rule, the fraction of windows in which the pre-clip value exceeded 1.0 and was clipped, and SHALL flag any window in which both rules were clipped as one in which the between-rule comparison carries no information.

2.17 WHEN the diagnostic answers Q0 THEN the system SHALL report the deterministic contamination fraction `U / (1 + U)` — the share of the max-rule forward command attributable to bilateral imbalance rather than to bilateral drive — as its headline quantity, computed pre-clip, and SHALL state the interpretation in both directions: a small value means the observed forward locomotion was a genuine bilateral command, a large value means it was a re-labelling of unilateral steering activity.

**Readout window sized for the real DN populations (Q4 — separates connectome fact from quantization).**

2.18 WHEN the diagnostic sets its readout window THEN the system SHALL use a window long enough to estimate single-cell rates — of order 50 to 100 ms, and SHALL default to the existing `SIMULATION_CHUNK_MS = 100.0` convention — while still closing the sensorimotor loop at that same cadence, and SHALL report the window used and the resulting rate quantum.

2.19 WHEN a per-side activation is reported THEN the system SHALL report the spike count and the window alongside it, so the achievable resolution of each activation is visible and quantization is not mistaken for graded activity.

2.20 WHEN bilateral DNa asymmetry is reported THEN the system SHALL state that it is a comparison of two individual reconstructed neurons per cell type (DNa01: 1 left + 1 right; DNa02: 1 left + 1 right; MDN: 2 left + 2 right), SHALL record the per-type per-side neuron counts in the output, and SHALL NOT describe it as population-level lateralization.

**Sensory contrast measured, not assumed (Q5 — quantifies the engineering environment).**

2.21 WHEN the closed-loop portion runs THEN the system SHALL compute and report the achieved sensory contrast at the sampled positions: the local per-ORN rate, the local gradient, the bilateral rate difference implied by the antenna separation actually used, and that difference as a fraction of the local rate.

2.22 WHEN the sensory contrast is reported THEN the system SHALL also report the Poisson noise floor for the window and per-side population actually used, and the ratio of contrast to noise floor, and SHALL flag the closed-loop result as uninterpretable when the contrast is below that floor.

2.23 WHEN the contrast is below the noise floor and a remedy is applied THEN the system SHALL remedy it only by making the odor field steeper or rescaling it, or by temporal sampling, and SHALL NOT enlarge the antenna separation beyond biological values and SHALL NOT introduce a hand-written navigation rule; any change to the field SHALL be reported as an engineering abstraction of the environment.

**Controlled and descriptive comparisons separated (Q6 — removes the closed-loop confound).**

2.24 WHEN the diagnostic compares the two forward rules THEN the system SHALL treat the within-run paired comparison — both rules evaluated from the same per-window DN spike counts, with only one of them driving the body — as the controlled measurement, and SHALL state which rule was in control of the trajectory.

2.25 WHEN cross-run behavioural metrics (`end_dist`, `min_dist`, `heading_range`) are reported THEN the system SHALL label them descriptive only and SHALL state that they are confounded, because changing the driving rule changes the trajectory, hence the sampled odor, hence the sensory input; the system SHALL NOT present them as an isolated interface effect.

**Category attribution on every number (Q7 — provenance).**

2.26 WHEN the diagnostic reports any number THEN the system SHALL attribute it explicitly to exactly one of: real connectome effect, engineering abstraction, or artifact of our chosen scaling/readout.

2.27 WHEN the forward channel is reported THEN the system SHALL restate that a forward command derived from DNa01/DNa02 is an engineering interface at the brain-to-body boundary and is NOT a claim that DNa01/DNa02 are forward-walking neurons, and SHALL NOT convert any engineering abstraction into a biological claim.

### Unchanged Behavior (Regression Prevention)

Scientific constraints and existing behaviour that the fix must preserve.

3.1 WHEN the diagnostic loads and simulates the brain THEN the system SHALL CONTINUE TO use the real FlyWire FAFB v783 connectivity as loaded, with no topology edits, no added or removed connections, and no changes to synaptic sign.

3.2 WHEN the diagnostic produces motor output THEN the system SHALL CONTINUE TO derive it only from simulated descending-neuron activity, with no hand-written navigation controller anywhere in the loop.

3.3 WHEN sensory input is delivered THEN the system SHALL CONTINUE TO pass only ORN indices and firing rates into the brain, and SHALL CONTINUE TO withhold food coordinates, target direction and distance-to-target from the neural simulation; the environment may know the food position for scoring only.

3.4 WHEN the diagnostic runs THEN the system SHALL CONTINUE TO run with no reward and no runtime plasticity, leaving `plasticity_enabled` false and `wdata` unmodified for the whole diagnostic.

3.5 WHEN a DNa lateral asymmetry is observed THEN the system SHALL CONTINUE TO log it as a connectome-level observation and SHALL CONTINUE TO leave it uncompensated; no correction, rebalancing or offset SHALL be applied to make the two sides equal.

3.6 WHEN the body is driven THEN the system SHALL CONTINUE TO give `FlyBody` only motor channels — forward, backward, left, right, the escape channels, escape laterality and proboscis drive — and SHALL CONTINUE TO withhold all environment state from it.

3.7 WHEN the diagnostic changes how forward is computed THEN the system SHALL CONTINUE TO leave the shared `DescendingReadout.channels` contract and the existing `FORWARD_GAIN = 4.0` / `BACKWARD_GAIN = 1.0` / `TURN_GAIN = 1.0` / `CHANNEL_HALF_MAX_HZ = 60.0` constants intact for all other callers, so that existing scripts and tests are not silently altered by this diagnostic.

3.8 WHEN the antenna geometry is configured THEN the system SHALL CONTINUE TO keep the bilateral antenna separation within biologically plausible values, and SHALL CONTINUE TO report the value used rather than tuning it for signal strength.

3.9 WHEN the diagnostic reports provenance THEN the system SHALL CONTINUE TO state the standing scope limit that FAFB v783 is a brain connectome, that the VNC and leg/wing motor neurons are absent, and that nothing downstream of the descending-neuron stage is a biological claim.

3.10 WHEN the diagnostic is run twice with the same seed and configuration THEN the system SHALL CONTINUE TO produce the same result, preserving the deterministic ORN selection and the explicitly seeded RNG.

3.11 WHEN the per-window spike-count fix is applied to the diagnostic THEN the system SHALL CONTINUE TO leave `engine.spike_counts` cumulative for existing callers, so the fix adds a correct windowed measurement path without changing documented engine semantics under other scripts.

3.12 WHEN escape, freeze, backward-walking and proboscis behaviour occur THEN the system SHALL CONTINUE TO be driven by their existing published DN associations and thresholds, unaffected by the change to the forward rule.
