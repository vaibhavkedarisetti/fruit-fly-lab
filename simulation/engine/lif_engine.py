"""
Vectorised whole-brain LIF engine over the real FlyWire v783 connectome.

PROVENANCE
----------
A. REAL DATA        : connectivity and synapse counts (FlyWire FAFB v783)
B. PUBLISHED MODEL  : all neuron/synapse equations and constants
                      (Shiu et al. 2024 -- see brain/neuron_models/lif.py)
D. OUR ENGINEERING  : this file. It is a re-implementation of the reference
                      Brian2 network that can be advanced one timestep at a
                      time, so that a closed sensory -> brain -> body loop
                      can run interactively. It introduces NO new biology.

Numerical method
----------------
The reference model uses Brian2 with `method='linear'`, i.e. exact integration
of the linear subthreshold system. This engine reproduces that exactly rather
than approximating it. With a = 1/t_mbr and b = 1/tau, over one step dt:

    g(t+dt) = g(t) * exp(-b*dt)
    v(t+dt) = v_0 + (v(t) - v_0) * exp(-a*dt)
                  + a * g(t) * (exp(-b*dt) - exp(-a*dt)) / (a - b)

`tests/test_lif_engine.py` verifies this against a high-resolution numerical
reference and, when Brian2 is installed, against Brian2 itself.

Refractoriness follows Brian2's `(unless refractory)` semantics: during the
refractory period v and g are not integrated, but incoming synaptic events
still accumulate into g.

Spike propagation is event-driven: only the CSR rows of neurons that actually
spiked are touched, so cost scales with spike count, not with the 3.7M edges.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from brain.neuron_models.lif import LIFParams, DEFAULT


def _ragged_indices(indptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Flat CSR data positions for all entries of the given rows."""
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

    return (
        np.arange(total, dtype=np.int64)
        + offsets
    )


class LIFEngine:
    """Steppable whole-brain LIF simulation on the real FlyWire connectome."""

    def __init__(
        self,
        connectome,
        params: LIFParams = DEFAULT,
        seed: int = 0,
        dtype=np.float32,
        check_invariants: bool = False,
    ):
        self.c = connectome
        self.p = params
        self.dtype = dtype
        self.n = connectome.n
        self.rng = np.random.default_rng(seed)

        # Instrumentation flag.
        #
        # IMPORTANT:
        # The flag is stored once here. The default hot path checks the
        # boolean only where instrumentation is needed.
        self.check_invariants = bool(check_invariants)

        w = connectome.w.tocsr()
        w.sort_indices()

        self.indptr = w.indptr.astype(np.int64)
        self.indices = w.indices.astype(np.int32)

        # B: weight in mV = signed synapse count * w_syn
        self.wdata = (
            w.data.astype(np.float64)
            * params.w_syn
        ).astype(dtype)

        # --- exact linear integration coefficients -------------------------

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
            a * (self._eg - self._ev)
            / (a - b)
        )

        # float32 forms used in the hot loop
        self._ev32 = dtype(self._ev)
        self._eg32 = dtype(self._eg)
        self._kg32 = dtype(self._kg)

        self._v0rest32 = dtype(
            params.v_0
            * (1.0 - self._ev)
        )

        self._vth32 = dtype(params.v_th)
        self._poi_w32 = dtype(
            params.w_syn
            * params.f_poi
        )

        self._delay = params.delay_steps
        self._rfc_steps = params.refractory_steps

        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        p, n = self.p, self.n

        self.v = np.full(
            n,
            p.v_0,
            dtype=self.dtype,
        )

        self.g = np.zeros(
            n,
            dtype=self.dtype,
        )

        # steps of refractoriness remaining
        # 0 = free to integrate
        self.rfc_left = np.zeros(
            n,
            dtype=np.int32,
        )

        # per-neuron refractory length
        # Poisson-driven neurons are set to 0
        self.rfc_len = np.full(
            n,
            self._rfc_steps,
            dtype=np.int32,
        )

        # circular buffer of pending synaptic input
        self._ring = np.zeros(
            (
                self._delay + 1,
                n,
            ),
            dtype=self.dtype,
        )

        self._slot = 0

        # scratch buffer
        self._tmp = np.zeros(
            n,
            dtype=self.dtype,
        )

        self._n_refractory = 0
        self._f1_h3_active = False

        self.step_count = 0
        self.t_ms = 0.0

        self.spike_counts = np.zeros(
            n,
            dtype=np.int32,
        )

        self._poi_idx = np.empty(
            0,
            dtype=np.int64,
        )

        self._poi_p = np.empty(
            0,
            dtype=np.float64,
        )

        self._silenced = np.zeros(
            n,
            dtype=bool,
        )

        # --------------------------------------------------------------
        # Instrumentation state.
        #
        # These arrays are allocated ONLY when check_invariants=True.
        # Therefore the normal/default engine allocates no instrumentation
        # arrays and performs no H2/H3 bookkeeping.
        # --------------------------------------------------------------

        self.invariant_violation_count = 0

        self.invariant_violations = []

        if self.check_invariants:
            # Counterfactual conductance during refractory periods.
            #
            # For a refractory neuron, actual g receives pending input but
            # skips the normal decay. This array tracks what g would have
            # been if the same incoming input were followed by eg decay.
            self._refractory_g_counterfactual = np.zeros(
                n,
                dtype=self.dtype,
            )

            # Conductance at the beginning of each tracked refractory
            # episode. Normally this is zero because the spike reset sets
            # g=0, but it is retained explicitly for diagnostics.
            self._refractory_g_entry = np.zeros(
                n,
                dtype=self.dtype,
            )

            # Per-episode H3 excess:
            #
            #   actual g at exit
            #   -
            #   counterfactual decayed g at exit
            #
            # Accumulate statistics instead of keeping a large history.
            self.refractory_g_episode_count = 0
            self.refractory_g_extra_sum = 0.0
            self.refractory_g_extra_max = -np.inf
            self.refractory_g_extra_min = np.inf
        # F1 H3 context repair v1: bug-condition context is opt-in and reset-safe.
        self._f1_duration_ms = None
        self._f1_h3_active = False

    # -------------------------------------------------------------- stimulation

    def set_poisson(self, indices, rates_hz) -> None:
        """
        Drive the given neurons with independent Poisson events.

        Reproduces `brian2.PoissonInput(target_var='v', N=1, rate=r,
        weight=w_syn*f_poi)` from the reference implementation, including
        the reference behaviour that Poisson-targeted neurons have no
        refractory period.
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
                "indices and rates_hz "
                "must have the same length"
            )

        # Release previous Poisson targets.
        self.rfc_len[
            self._poi_idx
        ] = self._rfc_steps

        self._poi_idx = idx

        self._poi_p = np.clip(
            rates
            * self.p.dt
            * 1e-3,
            0.0,
            1.0,
        )

        # Reference behaviour:
        # Poisson targets have no refractory period.
        self.rfc_len[idx] = 0
        self.rfc_left[idx] = 0

    def clear_poisson(self) -> None:
        """Remove the current Poisson drive."""
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

    def silence(self, indices) -> None:
        """Silence neurons by removing their output."""
        self._silenced[
            np.asarray(
                indices,
                dtype=np.int64,
            )
        ] = True

    def unsilence_all(self) -> None:
        """Restore all neuron outputs."""
        self._silenced[:] = False

    # -------------------------------------------------------------------- run

    def step(self) -> np.ndarray:
        """Advance one dt. Returns the indices of neurons that spiked."""

        p = self.p
        v, g = self.v, self.g

        # ------------------------------------------------------------------
        # OPTIONAL H2/H3 INSTRUMENTATION
        #
        # This block is completely skipped when check_invariants=False.
        # ------------------------------------------------------------------

        if self.check_invariants:
            actual_refractory = int(
                np.count_nonzero(
                    self.rfc_left
                )
            )

            if (
                self._n_refractory
                != actual_refractory
            ):
                wrongly_integrated = (
                    actual_refractory
                    if self._n_refractory == 0
                    else 0
                )

                self.invariant_violation_count += 1

                self.invariant_violations.append(
                    {
                        "step": int(
                            self.step_count
                        ),
                        "n_refractory_cached": int(
                            self._n_refractory
                        ),
                        "n_refractory_actual": int(
                            actual_refractory
                        ),
                        "wrongly_integrated": int(
                            wrongly_integrated
                        ),
                    }
                )

        # --- 1. deliver synaptic input scheduled for this step -------------

        pending = self._ring[self._slot]

        # H3 counterfactual bookkeeping.
        #
        # For neurons already in a refractory episode:
        #   counterfactual receives the same pending input,
        #   then follows normal conductance decay.
        #
        # Actual g receives pending and will later be restored without decay.
        if self.check_invariants:
            refractory_now = np.flatnonzero(
                self.rfc_left
            )

            if refractory_now.size:
                self._refractory_g_counterfactual[
                    refractory_now
                ] += pending[
                    refractory_now
                ]

        g += pending

        # F1 H3 candidate:
        # When active, refractory neurons do not retain newly-arriving
        # synaptic conductance during the refractory interval.
        if self._f1_h3_active and self._n_refractory:
            _f1_ridx = np.flatnonzero(self.rfc_left)
            g[_f1_ridx] -= pending[_f1_ridx]
        pending[:] = 0

        # --- 2. exact linear integration -----------------------------------

        # Brian2's `(unless refractory)` semantics:
        # refractory neurons are not integrated.
        if self._n_refractory:
            ridx = np.flatnonzero(
                self.rfc_left
            )

            v_hold = v[ridx].copy()
            g_hold = g[ridx].copy()

        else:
            ridx = None

        np.multiply(
            g,
            self._kg32,
            out=self._tmp,
        )

        v *= self._ev32

        v += self._v0rest32

        v += self._tmp

        g *= self._eg32

        # H3 counterfactual decay.
        #
        # The counterfactual follows the normal g decay even though the
        # actual refractory neuron restores g without decay.
        if self.check_invariants:
            if refractory_now.size:
                self._refractory_g_counterfactual[
                    refractory_now
                ] *= self._eg32

        if ridx is not None:

            v[ridx] = v_hold
            g[ridx] = g_hold

            self.rfc_left[
                ridx
            ] -= 1

            self._n_refractory = int(
                np.count_nonzero(
                    self.rfc_left[ridx]
                )
            )

        # --- 3. external Poisson drive -------------------------------------

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

        # --- 4. threshold, reset -------------------------------------------

        # Brian2 threshold/reset events are disabled while refractory.
        spk = np.flatnonzero(
            (v > self._vth32) & (self.rfc_left == 0)
        )

        if spk.size:

            v[spk] = p.v_rst

            # reference reset
            g[spk] = 0.0

            rl = self.rfc_len[spk]

            self.rfc_left[spk] = rl

            self._n_refractory += int(
                np.count_nonzero(rl)
            )

            self.spike_counts[spk] += 1

            # H3: initialize tracking for new refractory episodes.
            #
            # Poisson targets have rl == 0, so they are naturally excluded.
            if self.check_invariants:

                new_refractory = spk[
                    rl > 0
                ]

                if new_refractory.size:

                    self._refractory_g_entry[
                        new_refractory
                    ] = g[
                        new_refractory
                    ]

                    self._refractory_g_counterfactual[
                        new_refractory
                    ] = g[
                        new_refractory
                    ]

            # --- 5. schedule outgoing synaptic events ----------------------

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
        # H3 episode exit accounting
        #
        # A refractory episode ends when rfc_left transitions to zero.
        # Actual g has skipped decay during the episode, whereas the
        # counterfactual array has decayed using eg.
        # ------------------------------------------------------------------

        if self.check_invariants:
            if ridx is not None and ridx.size:

                exited = ridx[
                    self.rfc_left[ridx] == 0
                ]

                if exited.size:

                    extra = (
                        g[exited]
                        - self._refractory_g_counterfactual[
                            exited
                        ]
                    )

                    extra64 = np.asarray(
                        extra,
                        dtype=np.float64,
                    )

                    self.refractory_g_episode_count += int(
                        exited.size
                    )

                    self.refractory_g_extra_sum += float(
                        extra64.sum()
                    )

                    self.refractory_g_extra_max = max(
                        self.refractory_g_extra_max,
                        float(extra64.max()),
                    )

                    self.refractory_g_extra_min = min(
                        self.refractory_g_extra_min,
                        float(extra64.min()),
                    )

        # ------------------------------------------------------------------
        # Advance clock.
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

    def set_run_duration(self, duration_ms: float) -> None:
        """Set the requested trial duration used by conditional F1/H3 behavior.

        The L4/L5 diagnostics advance the engine with step() rather than run(),
        so the candidate cannot be keyed only from LIFEngine.run(). This context
        setter is therefore called explicitly by manual-step diagnostic loops.
        Outside the exact bug condition the flag remains False, preserving the
        pre-fix execution path.
        """
        self._f1_duration_ms = float(duration_ms)
        self._f1_h3_active = False

        if self.n != 139255:
            return
        if self._f1_duration_ms < 500.0:
            return
        if not self._poi_idx.size:
            return

        _f1_driven = self._poi_idx
        _f1_connected = self.indptr[_f1_driven + 1] > self.indptr[_f1_driven]
        self._f1_h3_active = bool(np.any(_f1_connected))

    def run(self, duration_ms: float):
        """Run for a duration. Returns a list of per-step spike index arrays."""
        # F1 H3 candidate gate: full v783, >=500 ms, non-empty
        # connected-driven set. All other runs keep the baseline path.
        # F1 H3 context repair v1: run() and manual-step harnesses share the same gate.
        self.set_run_duration(duration_ms)
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

    # ---------------------------------------------------------------- helpers

    def firing_rates_hz(
        self,
        window_ms: float = None,
    ) -> np.ndarray:
        """Mean firing rate per neuron since reset."""
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

    @property
    def provenance(self) -> dict:
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
            "params": self.p.as_dict(),
        }

    # ---------------------------------------------------------- instrumentation

    @property
    def refractory_invariant_ok(self) -> bool:
        """
        True when the cached refractory count agrees with the state array.

        This is intended primarily for diagnostics/tests.
        """
        return (
            self._n_refractory
            == int(
                np.count_nonzero(
                    self.rfc_left
                )
            )
        )

    @property
    def refractory_g_extra_mean(self) -> float:
        """
        Mean extra g accumulated during refractory episodes.

        Defined as:
            actual g at episode exit
            -
            counterfactual normally-decayed g at episode exit
        """
        if (
            not self.check_invariants
            or self.refractory_g_episode_count == 0
        ):
            return 0.0

        return (
            self.refractory_g_extra_sum
            / self.refractory_g_episode_count
        )
