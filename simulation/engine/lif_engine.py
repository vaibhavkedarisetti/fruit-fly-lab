"""
Vectorised whole-brain LIF engine over the real FlyWire v783 connectome.

PROVENANCE
----------
A. REAL DATA        : connectivity and synapse counts (FlyWire FAFB v783)
B. PUBLISHED MODEL  : all neuron/synapse equations and constants
                      (Shiu et al. 2024 -- see brain/neuron_models/lif.py)
D. OUR ENGINEERING  : this file. It is a re-implementation of the reference
                      Brian2 network that can be advanced one timestep at a
                      time, so that a closed sensory -> brain -> body loop can
                      run interactively. It introduces NO new biology.

Numerical method
----------------
The reference model uses Brian2 with `method='linear'`, i.e. *exact* integration
of the linear subthreshold system. This engine reproduces that exactly rather
than approximating it. With a = 1/t_mbr and b = 1/tau, over one step dt:

    g(t+dt) = g(t) * exp(-b*dt)
    v(t+dt) = v_0 + (v(t) - v_0) * exp(-a*dt)
                  + a * g(t) * (exp(-b*dt) - exp(-a*dt)) / (a - b)

`tests/test_lif_engine.py` verifies this against a high-resolution numerical
reference and, when Brian2 is installed, against Brian2 itself.

Refractoriness follows Brian2's `(unless refractory)` semantics: during the
refractory period v and g are *not integrated*, but incoming synaptic events
still accumulate into g.

Spike propagation is event-driven: only the CSR rows of neurons that actually
spiked are touched, so cost scales with spike count, not with the 3.7M edges.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from brain.neuron_models.lif import LIFParams, DEFAULT


def _ragged_indices(indptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Flat CSR data positions for all entries of the given rows (vectorised)."""
    starts = indptr[rows]
    lens = indptr[rows + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.cumsum(lens)
    offsets = np.repeat(starts - (cum - lens), lens)
    return np.arange(total, dtype=np.int64) + offsets


class LIFEngine:
    """Steppable whole-brain LIF simulation on the real FlyWire connectome."""

    def __init__(self, connectome, params: LIFParams = DEFAULT, seed: int = 0,
                 dtype=np.float32):
        self.c = connectome
        self.p = params
        self.dtype = dtype
        self.n = connectome.n
        self.rng = np.random.default_rng(seed)

        w = connectome.w.tocsr()
        w.sort_indices()
        self.indptr = w.indptr.astype(np.int64)
        self.indices = w.indices.astype(np.int32)
        # B: weight in mV = signed synapse count * w_syn
        self.wdata = (w.data.astype(np.float64) * params.w_syn).astype(dtype)

        # --- exact linear integration coefficients -------------------------
        a = 1.0 / params.t_mbr
        b = 1.0 / params.tau
        dt = params.dt
        self._ev = np.float64(np.exp(-a * dt))
        self._eg = np.float64(np.exp(-b * dt))
        self._kg = np.float64(a * (self._eg - self._ev) / (a - b))
        # float32 forms used in the hot loop, plus the constant term v_0*(1-ev)
        self._ev32 = dtype(self._ev)
        self._eg32 = dtype(self._eg)
        self._kg32 = dtype(self._kg)
        self._v0rest32 = dtype(params.v_0 * (1.0 - self._ev))
        self._vth32 = dtype(params.v_th)
        self._poi_w32 = dtype(params.w_syn * params.f_poi)

        self._delay = params.delay_steps          # 18 steps @ dt=0.1 ms
        self._rfc_steps = params.refractory_steps  # 22 steps @ dt=0.1 ms

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        p, n = self.p, self.n
        self.v = np.full(n, p.v_0, dtype=self.dtype)
        self.g = np.zeros(n, dtype=self.dtype)
        # steps of refractoriness remaining (0 = free to integrate)
        self.rfc_left = np.zeros(n, dtype=np.int32)
        # per-neuron refractory length; Poisson-driven neurons are set to 0
        self.rfc_len = np.full(n, self._rfc_steps, dtype=np.int32)

        # circular buffer of pending synaptic input, one slot per delay step
        self._ring = np.zeros((self._delay + 1, n), dtype=self.dtype)
        self._slot = 0
        self._tmp = np.zeros(n, dtype=self.dtype)   # scratch, avoids per-step alloc
        self._n_refractory = 0

        self.step_count = 0
        self.t_ms = 0.0
        self.spike_counts = np.zeros(n, dtype=np.int32)

        self._poi_idx = np.empty(0, dtype=np.int64)
        self._poi_p = np.empty(0, dtype=np.float64)
        self._silenced = np.zeros(n, dtype=bool)

    # -------------------------------------------------------------- stimulation
    def set_poisson(self, indices, rates_hz) -> None:
        """
        Drive the given neurons with independent Poisson events.

        Reproduces `brian2.PoissonInput(target_var='v', N=1, rate=r,
        weight=w_syn*f_poi)` from the reference implementation, including the
        reference behaviour that Poisson-targeted neurons have no refractory
        period.
        """
        idx = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        rates = np.atleast_1d(np.asarray(rates_hz, dtype=np.float64))
        if rates.size == 1 and idx.size > 1:
            rates = np.repeat(rates, idx.size)
        if idx.size != rates.size:
            raise ValueError("indices and rates_hz must have the same length")

        self.rfc_len[self._poi_idx] = self._rfc_steps   # release previous targets
        self._poi_idx = idx
        self._poi_p = np.clip(rates * self.p.dt * 1e-3, 0.0, 1.0)  # Hz * ms -> prob
        self.rfc_len[idx] = 0
        self.rfc_left[idx] = 0

    def clear_poisson(self) -> None:
        self.rfc_len[self._poi_idx] = self._rfc_steps
        self._poi_idx = np.empty(0, dtype=np.int64)
        self._poi_p = np.empty(0, dtype=np.float64)

    def silence(self, indices) -> None:
        """Silence neurons by removing their output (Shiu et al. `silence()`)."""
        self._silenced[np.asarray(indices, dtype=np.int64)] = True

    def unsilence_all(self) -> None:
        self._silenced[:] = False

    # -------------------------------------------------------------------- run
    def step(self) -> np.ndarray:
        """Advance one dt. Returns the indices of neurons that spiked."""
        p = self.p
        v, g = self.v, self.g

        # --- 1. deliver synaptic input scheduled for this step --------------
        pending = self._ring[self._slot]
        g += pending                   # arrives regardless of refractoriness
        pending[:] = 0

        # --- 2. exact linear integration --------------------------------
        # Brian2's `(unless refractory)` semantics: refractory neurons are not
        # integrated. They are few, so integrate everything and restore them.
        if self._n_refractory:
            ridx = np.flatnonzero(self.rfc_left)
            v_hold = v[ridx].copy()
            g_hold = g[ridx].copy()
        else:
            ridx = None

        np.multiply(g, self._kg32, out=self._tmp)   # kg * g(t), uses old g
        v *= self._ev32
        v += self._v0rest32                          # v_0 * (1 - exp(-dt/t_mbr))
        v += self._tmp
        g *= self._eg32

        if ridx is not None:
            v[ridx] = v_hold
            g[ridx] = g_hold
            self.rfc_left[ridx] -= 1
            self._n_refractory = int(np.count_nonzero(self.rfc_left[ridx]))

        # --- 3. external Poisson drive (adds directly to v, as in reference) -
        if self._poi_idx.size:
            fired = self.rng.random(self._poi_idx.size) < self._poi_p
            if fired.any():
                v[self._poi_idx[fired]] += self._poi_w32

        # --- 4. threshold, reset ------------------------------------------
        # Refractory neurons are held at v_rst (< v_th) and cannot cross, and
        # Poisson-driven neurons are never refractory, so no extra mask needed.
        spk = np.flatnonzero(v > self._vth32)
        if spk.size:
            v[spk] = p.v_rst
            g[spk] = 0.0                       # reference reset: g = 0
            rl = self.rfc_len[spk]
            self.rfc_left[spk] = rl
            self._n_refractory += int(np.count_nonzero(rl))
            self.spike_counts[spk] += 1

            # --- 5. schedule outgoing synaptic events (delayed) ------------
            emit = spk[~self._silenced[spk]]
            if emit.size:
                flat = _ragged_indices(self.indptr, emit)
                if flat.size:
                    target = (self._slot + self._delay) % (self._delay + 1)
                    self._ring[target] += np.bincount(
                        self.indices[flat], weights=self.wdata[flat],
                        minlength=self.n,
                    )

        self._slot = (self._slot + 1) % (self._delay + 1)
        self.step_count += 1
        self.t_ms = self.step_count * p.dt
        return spk

    def run(self, duration_ms: float):
        """Run for a duration. Returns a list of per-step spike index arrays."""
        n_steps = int(round(duration_ms / self.p.dt))
        return [self.step() for _ in range(n_steps)]

    # ---------------------------------------------------------------- helpers
    def firing_rates_hz(self, window_ms: float = None) -> np.ndarray:
        """Mean firing rate per neuron since reset (or over `window_ms`)."""
        elapsed = self.t_ms if window_ms is None else window_ms
        if elapsed <= 0:
            return np.zeros(self.n, dtype=np.float64)
        return self.spike_counts / (elapsed * 1e-3)

    @property
    def provenance(self) -> dict:
        return {
            "connectome": self.c.dataset,
            "n_neurons": self.n,
            "n_connections": int(self.c.w.nnz),
            "n_synapses": int(np.abs(self.c.w.data).sum()),
            "model": "Shiu et al. 2024 LIF (exact linear integration)",
            "params": self.p.as_dict(),
        }
