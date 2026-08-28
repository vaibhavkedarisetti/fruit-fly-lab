"""
A running simulation session: environment -> sensory encoders -> LIF engine over
the real FlyWire connectome -> descending-neuron readout -> body.

This is the object the interactive laboratory drives. It owns the closed loop
and the telemetry, and it does not contain any behavioural rule: it simply
advances the network and reports what the real neurons did.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from brain.motor.descending import DescendingReadout
from brain.neuron_models.lif import LIFParams, DEFAULT
from simulation.engine.lif_engine import LIFEngine


class SpikeRecorder:
    """Sliding-window spike statistics, kept cheap enough to run every frame."""

    def __init__(self, connectome, window_frames: int = 50):
        self.c = connectome
        self.n = connectome.n
        self.window_frames = window_frames
        self._ring: deque = deque()
        self.window_sum = np.zeros(self.n, dtype=np.int32)

        # region codes for brain-region activity
        regions = connectome.neurons["primary_neuropil"].astype(str)
        self.region_names = sorted(regions.unique())
        code = {r: i for i, r in enumerate(self.region_names)}
        self.region_code = regions.map(code).to_numpy().astype(np.int64)
        self.region_sizes = np.bincount(self.region_code,
                                        minlength=len(self.region_names))

    def push(self, spike_idx: np.ndarray) -> None:
        if spike_idx.size:
            self.window_sum += np.bincount(spike_idx, minlength=self.n).astype(np.int32)
        self._ring.append(spike_idx)
        while len(self._ring) > self.window_frames:
            old = self._ring.popleft()
            if old.size:
                self.window_sum -= np.bincount(old, minlength=self.n).astype(np.int32)

    def region_activity(self, window_ms: float) -> dict:
        """Mean firing rate (Hz) per brain region over the window."""
        tot = np.bincount(self.region_code, weights=self.window_sum,
                          minlength=len(self.region_names))
        rate = tot / np.maximum(self.region_sizes, 1) / (window_ms * 1e-3)
        return {n: float(r) for n, r in zip(self.region_names, rate) if r > 0.01}

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.window_sum))


class Session:
    """One interactive experiment on the real FlyWire connectome."""

    #: how often sensory rates are recomputed from the environment
    RATE_UPDATE_MS = 1.0

    def __init__(self, connectome, params: LIFParams = DEFAULT, seed: int = 0,
                 window_ms: float = 50.0):
        self.c = connectome
        self.p = params
        self.engine = LIFEngine(connectome, params, seed=seed)
        self.readout = DescendingReadout(connectome)
        self.window_ms = window_ms
        self.recorder = SpikeRecorder(connectome,
                                      window_frames=int(window_ms / self.RATE_UPDATE_MS))
        self.encoders = []          # list of (encoder, stimulus)
        self.paused = False
        self.history = []           # telemetry frames, for replay
        self._raster = deque(maxlen=20000)

        from fly.body.fly_body import FlyBody
        self.body = FlyBody()

        # Neurons whose spikes are always reported individually in the raster.
        n = connectome.neurons
        watch = n[n["super_class"].astype(str).isin(
            ["descending", "visual_projection", "sensory"])]
        self.watch_idx = watch["idx"].to_numpy(dtype=np.int64)

    # ---------------------------------------------------------------- stimuli
    def clear_stimuli(self) -> None:
        self.encoders = []
        self.engine.clear_poisson()

    def add_stimulus(self, encoder, stimulus) -> None:
        self.encoders.append((encoder, stimulus))

    def add_modality(self, key: str, intensity: float = 1.0,
                     duration_ms: float = float("inf"),
                     delay_ms: float = 0.0):
        """
        Deliver one of the registered stimuli (see brain/sensory/modalities.py).

        Raises for a modality the connectome cannot support, rather than faking
        a response.
        """
        from brain.sensory.encoders import PopulationEncoder
        from brain.sensory.modalities import BY_KEY
        from simulation.stimuli.pulse import PulseStimulus

        m = BY_KEY.get(key)
        if m is None:
            raise KeyError("unknown modality %r" % key)
        if not m.supported:
            raise ValueError("%s is not currently modeled: %s"
                             % (m.label, m.unsupported_reason))
        enc = PopulationEncoder(self.c, m)
        stim = PulseStimulus(modality_key=key, intensity=intensity,
                             t_start_ms=self.engine.t_ms + delay_ms,
                             duration_ms=duration_ms)
        self.add_stimulus(enc, stim)
        return stim

    def add_looming(self, azimuth_deg: float = 45.0, elevation_deg: float = 0.0,
                    half_size_mm: float = 5.0, speed_mm_s: float = 250.0,
                    start_distance_mm: float = 50.0, delay_ms: float = 0.0):
        """Throw an object at the fly."""
        from brain.sensory.encoders import LoomingEncoder
        from brain.sensory.retinotopy import load_retinotopy
        from simulation.stimuli.looming import LoomingStimulus

        enc = LoomingEncoder(self.c, load_retinotopy(self.c))
        stim = LoomingStimulus(
            azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
            half_size_mm=half_size_mm, speed_mm_s=speed_mm_s,
            start_distance_mm=start_distance_mm,
            t_start_ms=self.engine.t_ms + delay_ms)
        self.add_stimulus(enc, stim)
        return stim

    def _refresh_rates(self) -> None:
        if not self.encoders:
            return
        idx_all, rate_all = [], []
        for enc, stim in self.encoders:
            idx_all.append(enc.indices)
            rate_all.append(enc.rates_hz(self.engine.t_ms, stim))
        idx = np.concatenate(idx_all)
        rates = np.concatenate(rate_all)
        # A neuron driven by several stimuli takes the strongest drive.
        order = np.argsort(-rates)
        idx, rates = idx[order], rates[order]
        uniq, first = np.unique(idx, return_index=True)
        self.engine.set_poisson(uniq, rates[first])

    # -------------------------------------------------------------------- run
    def advance(self, duration_ms: float) -> list:
        """Advance the simulation, returning one telemetry frame per ms."""
        if self.paused:
            return []
        frames = []
        n_blocks = max(1, int(round(duration_ms / self.RATE_UPDATE_MS)))
        steps = int(round(self.RATE_UPDATE_MS / self.p.dt))

        for _ in range(n_blocks):
            self._refresh_rates()
            block = []
            t0 = time.perf_counter()
            for _ in range(steps):
                block.append(self.engine.step())
            spk = (np.concatenate(block) if any(b.size for b in block)
                   else np.empty(0, dtype=np.int64))
            self.recorder.push(spk)
            frames.append(self._telemetry(spk, time.perf_counter() - t0))
        self.history.extend(frames)
        return frames

    def _telemetry(self, spk: np.ndarray, wall_s: float) -> dict:
        eng = self.engine
        win = self.window_ms
        ws = self.recorder.window_sum

        # raster entries for watched neurons only (keeps payload small)
        if spk.size:
            hit = spk[np.isin(spk, self.watch_idx, assume_unique=False)]
            for i in hit[:200]:
                self._raster.append((round(eng.t_ms, 2), int(i)))

        stim_state = [{"kind": type(s).__name__, **s.state(eng.t_ms)}
                      for _, s in self.encoders]

        channels = self.readout.channels(ws, win)
        laterality = self.readout.escape_laterality(ws)
        prob = self.readout.proboscis_drive(ws, win)

        # The body is driven ONLY by these neural readouts.
        self.body.update(self.RATE_UPDATE_MS, channels, eng.t_ms,
                         escape_laterality=laterality, proboscis_drive=prob)

        return {
            "t_ms": round(eng.t_ms, 3),
            "n_spikes": int(spk.size),
            "active_neurons": self.recorder.active_count,
            "total_spikes": int(eng.spike_counts.sum()),
            "mean_rate_hz": float(ws.sum() / self.c.n / (win * 1e-3)),
            "regions": self.recorder.region_activity(win),
            "dn_rates": self.readout.rates(ws, win),
            "channels": channels,
            "proboscis_drive": prob,
            "escape_laterality": laterality,
            "stimuli": stim_state,
            "body": self.body.as_dict(),
            "wall_ms": round(wall_s * 1000.0, 2),
        }

    # ------------------------------------------------------------------ state
    def reset(self, seed: int = 0) -> None:
        self.engine.reset()
        self.engine.rng = np.random.default_rng(seed)
        self.recorder = SpikeRecorder(
            self.c, window_frames=int(self.window_ms / self.RATE_UPDATE_MS))
        self.history.clear()
        self._raster.clear()
        self.clear_stimuli()

    def raster(self, last_ms: float = 200.0) -> list:
        t_min = self.engine.t_ms - last_ms
        return [[t, i] for t, i in self._raster if t >= t_min]

    @property
    def provenance(self) -> dict:
        return {
            "engine": self.engine.provenance,
            "motor": self.readout.provenance,
            "encoders": [e.provenance for e, _ in self.encoders],
        }
