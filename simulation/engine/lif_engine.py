"""
Vectorised whole-brain LIF engine over the real FlyWire v783 connectome.

PROVENANCE
----------
A. REAL DATA
   Connectivity and synapse counts come from the FlyWire FAFB v783
   connectome.

B. PUBLISHED MODEL
   Neuron/synapse equations and constants come from the LIF model defined
   in brain/neuron_models/lif.py and the Shiu et al. 2024 reference model.

C. OUR ENGINEERING
   This file is an engineering reimplementation of the reference network
   that can be advanced one timestep at a time.

Reward-modulated plasticity is an engineering learning mechanism operating
on the existing FlyWire topology. It does NOT create new anatomical
connections.

IMPORTANT
---------
The FlyWire topology is fixed.

Plasticity can change the magnitude of an existing synapse, but it can never:

    - create a new connection
    - delete a connection from the topology
    - change excitation into inhibition
    - change inhibition into excitation

The learned values live in self.wdata.

Delayed reward learning
-----------------------
Short neuron traces record recent activity.

A separate per-edge eligibility value accumulates coactivity during an
episode. This allows a reward at the END of a long episode to reinforce
or weaken synapses that participated earlier in that episode.

This is an ENGINEERING learning mechanism, not a claim that this is the
exact biological plasticity mechanism used by Drosophila.

Numerical method
----------------
The reference model uses Brian2 with method='linear', i.e. exact
integration of the linear subthreshold system.

For:

    a = 1 / tau_mbr
    b = 1 / tau_syn

the exact update is:

    g(t+dt) = g(t) * exp(-b*dt)

    v(t+dt) =
        v0
        + (v(t) - v0) * exp(-a*dt)
        + a*g(t)*(exp(-b*dt) - exp(-a*dt))/(a-b)

Refractoriness follows Brian2's `(unless refractory)` semantics:
during refractory periods v and g are not integrated, while incoming
synaptic events still accumulate.

Spike propagation is event-driven: only CSR rows belonging to neurons
that actually spiked are processed.
"""

from __future__ import annotations

import numpy as np

from brain.neuron_models.lif import LIFParams, DEFAULT


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ragged_indices(
    indptr: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """
    Return flat CSR data positions for all entries belonging to `rows`.

    This avoids constructing a Python list of CSR row ranges.
    """
    starts = indptr[rows]
    lens = indptr[rows + 1] - starts

    total = int(lens.sum())

    if total == 0:
        return np.empty(0, dtype=np.int64)

    cum = np.cumsum(lens)

    offsets = np.repeat(
        starts - (cum - lens),
        lens,
    )

    return np.arange(
        total,
        dtype=np.int64,
    ) + offsets


# ---------------------------------------------------------------------------
# LIF engine
# ---------------------------------------------------------------------------

class LIFEngine:
    """
    Steppable whole-brain LIF simulation on the real FlyWire connectome.
    """

    def __init__(
        self,
        connectome,
        params: LIFParams = DEFAULT,
        seed: int = 0,
        dtype=np.float32,
    ):
        self.c = connectome
        self.p = params
        self.dtype = dtype
        self.n = connectome.n

        # Independent RNG for reproducible Poisson stimulation.
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Real FlyWire connectivity
        # ------------------------------------------------------------------

        w = connectome.w.tocsr()
        w.sort_indices()

        self.indptr = w.indptr.astype(np.int64)
        self.indices = w.indices.astype(np.int32)

        # Signed synaptic weight:
        #
        #   FlyWire synapse count × model synaptic weight
        #
        self.wdata = (
            w.data.astype(np.float64) * params.w_syn
        ).astype(dtype)

        # ------------------------------------------------------------------
        # Reward-modulated plasticity
        # ------------------------------------------------------------------
        #
        # pre_trace/post_trace:
        #     short-term neuron activity traces.
        #
        # eligibility:
        #     per-existing-edge episode eligibility.
        #
        # The topology itself is never changed.
        # ------------------------------------------------------------------

        self.pre_trace = np.zeros(
            self.n,
            dtype=self.dtype,
        )

        self.post_trace = np.zeros(
            self.n,
            dtype=self.dtype,
        )

        self.plasticity_enabled = False

        self.plasticity_learning_rate = dtype(0.0001)

        self.plasticity_trace_tau_ms = dtype(500.0)

        # Per-synapse eligibility.
        #
        # This is the same length as wdata, so it corresponds exactly
        # to existing CSR edges.
        self.eligibility = np.zeros_like(
            self.wdata,
            dtype=self.dtype,
        )

        # Keep the sign of every anatomical connection fixed.
        self._w_sign = np.sign(
            self.wdata
        ).astype(self.dtype)

        # Initial absolute magnitude is used to prevent runaway growth.
        self._w_abs_initial = np.abs(
            self.wdata
        ).astype(self.dtype)

        # ------------------------------------------------------------------
        # Exact linear integration coefficients
        # ------------------------------------------------------------------

        a = 1.0 / params.t_mbr
        b = 1.0 / params.tau
        dt = params.dt

        self._ev = np.float64(
            np.exp(-a * dt)
        )

        self._eg = np.float64(
            np.exp(-b * dt)
        )

        self._kg = np.float64(
            a * (self._eg - self._ev) / (a - b)
        )

        # Float32 hot-loop constants.
        self._ev32 = dtype(self._ev)
        self._eg32 = dtype(self._eg)
        self._kg32 = dtype(self._kg)

        self._v0rest32 = dtype(
            params.v_0 * (1.0 - self._ev)
        )

        self._vth32 = dtype(
            params.v_th
        )

        self._poi_w32 = dtype(
            params.w_syn * params.f_poi
        )

        self._delay = params.delay_steps

        self._rfc_steps = params.refractory_steps

        # ------------------------------------------------------------------
        # Initialise state
        # ------------------------------------------------------------------

        self.reset()

    # ----------------------------------------------------------------------
    # State
    # ----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset neural state for a new episode.

        IMPORTANT:
            Learned synaptic weights are NOT reset.

        This means:

            Episode 1
                ↓
            learning
                ↓
            Episode 2
                ↓
            learned weights remain

        while voltage, spikes, traces, eligibility and sensory input are
        reset.
        """

        p = self.p
        n = self.n

        # Membrane voltage.
        self.v = np.full(
            n,
            p.v_0,
            dtype=self.dtype,
        )

        # Synaptic conductance.
        self.g = np.zeros(
            n,
            dtype=self.dtype,
        )

        # Remaining refractory steps.
        self.rfc_left = np.zeros(
            n,
            dtype=np.int32,
        )

        # Per-neuron refractory length.
        self.rfc_len = np.full(
            n,
            self._rfc_steps,
            dtype=np.int32,
        )

        # Circular delayed-synaptic-event buffer.
        self._ring = np.zeros(
            (
                self._delay + 1,
                n,
            ),
            dtype=self.dtype,
        )

        self._slot = 0

        # Scratch array used by the integration loop.
        self._tmp = np.zeros(
            n,
            dtype=self.dtype,
        )

        self._n_refractory = 0

        # Simulation clock.
        self.step_count = 0
        self.t_ms = 0.0

        # Cumulative spike counts since episode reset.
        self.spike_counts = np.zeros(
            n,
            dtype=np.int32,
        )

        # Reset short-term learning traces.
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)

        # Reset episode eligibility.
        #
        # IMPORTANT:
        # This does NOT modify self.wdata.
        self.eligibility.fill(0.0)

        # Clear external Poisson stimulation.
        self._poi_idx = np.empty(
            0,
            dtype=np.int64,
        )

        self._poi_p = np.empty(
            0,
            dtype=np.float64,
        )

        # Clear silencing.
        self._silenced = np.zeros(
            n,
            dtype=bool,
        )

    # ----------------------------------------------------------------------
    # External stimulation
    # ----------------------------------------------------------------------

    def set_poisson(
        self,
        indices,
        rates_hz,
    ) -> None:
        """
        Drive selected neurons with independent Poisson events.

        This reproduces the reference PoissonInput behaviour:

            target_var = v
            N = 1
            rate = r
            weight = w_syn * f_poi

        Poisson-targeted neurons have no refractory period.
        """

        idx = np.atleast_1d(
            np.asarray(
                indices,
                dtype=np.int64,
            )
        )

        rates = np.atleast_1d(
            np.asarray(
                rates_hz,
                dtype=np.float64,
            )
        )

        if rates.size == 1 and idx.size > 1:
            rates = np.repeat(
                rates,
                idx.size,
            )

        if idx.size != rates.size:
            raise ValueError(
                "indices and rates_hz must have the same length"
            )

        # Restore refractory settings for previous Poisson targets.
        if self._poi_idx.size:
            self.rfc_len[
                self._poi_idx
            ] = self._rfc_steps

        self._poi_idx = idx

        # Hz × ms -> probability per timestep.
        self._poi_p = np.clip(
            rates * self.p.dt * 1e-3,
            0.0,
            1.0,
        )

        # Poisson neurons do not use refractory periods.
        self.rfc_len[idx] = 0
        self.rfc_left[idx] = 0

    def clear_poisson(self) -> None:
        """Remove the current external Poisson drive."""

        if self._poi_idx.size:
            self.rfc_len[
                self._poi_idx
            ] = self._rfc_steps

        self._poi_idx = np.empty(
            0,
            dtype=np.int64,
        )

        self._poi_p = np.empty(
            0,
            dtype=np.float64,
        )

    # ----------------------------------------------------------------------
    # Silencing
    # ----------------------------------------------------------------------

    def silence(self, indices) -> None:
        """
        Silence selected neurons by preventing their outgoing spikes.

        This corresponds to the model's engineering implementation of
        Shiu et al.'s silence() operation.
        """

        idx = np.asarray(
            indices,
            dtype=np.int64,
        )

        self._silenced[idx] = True

    def unsilence_all(self) -> None:
        """Restore all neuron outputs."""

        self._silenced[:] = False

    # ----------------------------------------------------------------------
    # One timestep
    # ----------------------------------------------------------------------

    def step(self) -> np.ndarray:
        """
        Advance the network by one timestep.

        Returns
        -------
        np.ndarray
            Indices of neurons that spiked during this timestep.
        """

        p = self.p

        v = self.v
        g = self.g

        # ------------------------------------------------------------------
        # 1. Deliver delayed synaptic input.
        # ------------------------------------------------------------------

        pending = self._ring[self._slot]

        # Incoming events accumulate even during refractoriness.
        g += pending

        pending[:] = 0.0

        # ------------------------------------------------------------------
        # 2. Exact linear integration.
        # ------------------------------------------------------------------

        # Brian2's `(unless refractory)` semantics:
        #
        #   refractory neurons:
        #       v and g are not integrated
        #
        # We integrate everything and restore the refractory subset.
        if self._n_refractory:
            ridx = np.flatnonzero(
                self.rfc_left
            )

            v_hold = v[ridx].copy()
            g_hold = g[ridx].copy()

        else:
            ridx = None

        # Uses old g.
        np.multiply(
            g,
            self._kg32,
            out=self._tmp,
        )

        # v(t+dt)
        v *= self._ev32

        v += self._v0rest32

        v += self._tmp

        # g(t+dt)
        g *= self._eg32

        # Restore refractory neurons.
        if ridx is not None:

            v[ridx] = v_hold
            g[ridx] = g_hold

            self.rfc_left[ridx] -= 1

            self._n_refractory = int(
                np.count_nonzero(
                    self.rfc_left[ridx]
                )
            )

        # ------------------------------------------------------------------
        # 3. External Poisson drive.
        # ------------------------------------------------------------------

        if self._poi_idx.size:

            fired = (
                self.rng.random(
                    self._poi_idx.size
                )
                < self._poi_p
            )

            if fired.any():

                v[
                    self._poi_idx[fired]
                ] += self._poi_w32

        # ------------------------------------------------------------------
        # 4. Threshold crossing.
        # ------------------------------------------------------------------

        spk = np.flatnonzero(
            v > self._vth32
        )

        # ------------------------------------------------------------------
        # 5. Plasticity traces.
        # ------------------------------------------------------------------

        if self.plasticity_enabled:
            self.update_plasticity_traces(spk)

        # ------------------------------------------------------------------
        # 6. Accumulate episode eligibility.
        # ------------------------------------------------------------------
        #
        # This operates ONLY on edges whose presynaptic neuron spiked.
        #
        # We deliberately do not decay the entire 50-million-edge array
        # every timestep. That would make the simulator prohibitively
        # expensive.
        #
        # Eligibility therefore represents accumulated coactivity during
        # the current episode.
        # ------------------------------------------------------------------

        if (
            self.plasticity_enabled
            and spk.size
        ):

            self._accumulate_eligibility(
                spk
            )

        # ------------------------------------------------------------------
        # 7. Reset spiking neurons.
        # ------------------------------------------------------------------

        if spk.size:

            v[spk] = p.v_rst

            # Reference reset: synaptic conductance is cleared.
            g[spk] = 0.0

            rl = self.rfc_len[spk]

            self.rfc_left[spk] = rl

            self._n_refractory += int(
                np.count_nonzero(rl)
            )

            self.spike_counts[spk] += 1

            # ----------------------------------------------------------------
            # 8. Schedule outgoing synaptic events.
            # ----------------------------------------------------------------

            emit = spk[
                ~self._silenced[spk]
            ]

            if emit.size:

                flat = _ragged_indices(
                    self.indptr,
                    emit,
                )

                if flat.size:

                    target = (
                        self._slot
                        + self._delay
                    ) % (
                        self._delay + 1
                    )

                    self._ring[target] += np.bincount(
                        self.indices[flat],
                        weights=self.wdata[flat],
                        minlength=self.n,
                    )

        # ------------------------------------------------------------------
        # 9. Advance clock.
        # ------------------------------------------------------------------

        self._slot = (
            self._slot + 1
        ) % (
            self._delay + 1
        )

        self.step_count += 1

        self.t_ms = (
            self.step_count
            * p.dt
        )

        return spk

    # ----------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------

    def run(
        self,
        duration_ms: float,
    ):
        """
        Run for a duration.

        Returns
        -------
        list[np.ndarray]
            Per-timestep spike-index arrays.
        """

        n_steps = int(
            round(
                duration_ms
                / self.p.dt
            )
        )

        return [
            self.step()
            for _ in range(n_steps)
        ]

    # ----------------------------------------------------------------------
    # Plasticity
    # ----------------------------------------------------------------------

    def enable_plasticity(
        self,
        learning_rate: float = 0.0001,
        trace_tau_ms: float = 500.0,
    ) -> None:
        """
        Enable reward-modulated synaptic plasticity.

        Parameters
        ----------
        learning_rate:
            Magnitude of weight modification produced by eligibility
            and reward.

        trace_tau_ms:
            Time constant of the neuron-level activity traces.

        Notes
        -----
        The episode eligibility itself accumulates throughout the episode.
        This is intentional: it allows delayed reward at episode end.
        """

        if learning_rate < 0:
            raise ValueError(
                "learning_rate must be >= 0"
            )

        if trace_tau_ms <= 0:
            raise ValueError(
                "trace_tau_ms must be > 0"
            )

        self.plasticity_enabled = True

        self.plasticity_learning_rate = self.dtype(
            learning_rate
        )

        self.plasticity_trace_tau_ms = self.dtype(
            trace_tau_ms
        )

    def disable_plasticity(self) -> None:
        """Disable synaptic plasticity."""

        self.plasticity_enabled = False

    def update_plasticity_traces(
        self,
        spiking_neurons: np.ndarray,
    ) -> None:
        """
        Update short-term pre/post neuron traces.

        Both traces decay exponentially and receive an increment of 1
        whenever a neuron spikes.
        """

        if not self.plasticity_enabled:
            return

        dt_ms = float(
            self.p.dt
        )

        tau_ms = float(
            self.plasticity_trace_tau_ms
        )

        decay = np.exp(
            -dt_ms / tau_ms
        )

        self.pre_trace *= decay
        self.post_trace *= decay

        if spiking_neurons.size:

            self.pre_trace[
                spiking_neurons
            ] += 1.0

            self.post_trace[
                spiking_neurons
            ] += 1.0

    def _accumulate_eligibility(
        self,
        spiking_neurons: np.ndarray,
    ) -> None:
        """
        Add current pre/post coactivity to existing real FlyWire edges.

        CSR rows correspond to presynaptic neurons.

        For every presynaptic neuron that spiked:

            eligibility(edge) +=
                pre_trace(pre)
                × post_trace(post)

        Only existing CSR edges are touched.
        """

        if not self.plasticity_enabled:
            return

        if spiking_neurons.size == 0:
            return

        for pre in spiking_neurons:

            start = int(
                self.indptr[pre]
            )

            end = int(
                self.indptr[pre + 1]
            )

            if start == end:
                continue

            posts = self.indices[
                start:end
            ]

            pre_value = float(
                self.pre_trace[pre]
            )

            if pre_value == 0.0:
                continue

            post_values = self.post_trace[
                posts
            ]

            # Only active post traces contribute.
            mask = (
                post_values != 0.0
            )

            if not np.any(mask):
                continue

            self.eligibility[
                start:end
            ][mask] += (
                pre_value
                * post_values[mask]
            )

    def apply_reward(
        self,
        reward: float,
    ) -> None:
        """
        Apply delayed reward/punishment to episode eligibility.

        Positive reward:
            strengthens eligible synapses.

        Negative reward:
            weakens eligible synapses.

        The anatomical topology never changes.

        Only the magnitude of existing synaptic weights is modified.

        Excitatory/inhibitory sign is preserved.

        Learned magnitude is limited to at most 3× its original magnitude.
        """

        if not self.plasticity_enabled:
            return

        reward = float(reward)

        if reward == 0.0:
            return

        lr = float(
            self.plasticity_learning_rate
        )

        if lr == 0.0:
            return

        eligible = self.eligibility

        mask = (
            eligible != 0.0
        )

        if not np.any(mask):
            return

        # --------------------------------------------------------------
        # Reward-modulated change.
        # --------------------------------------------------------------

        delta = (
            lr
            * reward
            * eligible[mask]
        )

        # --------------------------------------------------------------
        # Preserve anatomical sign.
        # --------------------------------------------------------------

        signs = self._w_sign[mask]

        magnitude = (
            np.abs(
                self.wdata[mask]
            )
            + delta
        )

        # --------------------------------------------------------------
        # Do not allow a synapse to cross through zero.
        # --------------------------------------------------------------

        magnitude = np.maximum(
            magnitude,
            0.0,
        )

        # --------------------------------------------------------------
        # Prevent runaway growth.
        # --------------------------------------------------------------

        maximum = (
            self._w_abs_initial[mask]
            * 3.0
        )

        magnitude = np.minimum(
            magnitude,
            maximum,
        )

        # --------------------------------------------------------------
        # Write learned values back to real CSR edges.
        # --------------------------------------------------------------

        self.wdata[mask] = (
            signs
            * magnitude
        )

        # Eligibility belongs to the completed episode.
        self.eligibility.fill(0.0)

    # ----------------------------------------------------------------------
    # Measurements
    # ----------------------------------------------------------------------

    def firing_rates_hz(
        self,
        window_ms: float = None,
    ) -> np.ndarray:
        """
        Mean firing rate per neuron.

        If window_ms is None:
            use all time since reset.

        Otherwise:
            interpret spike_counts as the measurement window requested
            by the caller.
        """

        elapsed = (
            self.t_ms
            if window_ms is None
            else window_ms
        )

        if elapsed <= 0:
            return np.zeros(
                self.n,
                dtype=np.float64,
            )

        return (
            self.spike_counts
            / (elapsed * 1e-3)
        )

    # ----------------------------------------------------------------------
    # Provenance
    # ----------------------------------------------------------------------

    @property
    def provenance(self) -> dict:
        """Return provenance information for the simulation."""

        return {
            "connectome": self.c.dataset,

            "n_neurons": self.n,

            "n_connections": int(
                self.c.w.nnz
            ),

            "n_synapses": int(
                np.abs(
                    self.c.w.data
                ).sum()
            ),

            "model": (
                "Shiu et al. 2024 LIF "
                "(exact linear integration)"
            ),

            "plasticity": {
                "enabled": bool(
                    self.plasticity_enabled
                ),

                "learning_rate": float(
                    self.plasticity_learning_rate
                ),

                "trace_tau_ms": float(
                    self.plasticity_trace_tau_ms
                ),

                "eligibility": (
                    "episode-level accumulated "
                    "pre/post coactivity"
                ),

                "topology_changed": False,

                "synaptic_sign_preserved": True,
            },

            "params": self.p.as_dict(),
        }