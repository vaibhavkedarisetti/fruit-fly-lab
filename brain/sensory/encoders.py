"""
Sensory encoders: environment -> firing rates of REAL FlyWire sensory neurons.

This is the one place in the pipeline where something other than the connectome
decides a firing rate, so it is documented in detail.

WHY AN ENCODER IS NEEDED AT ALL
-------------------------------
The FlyWire connectome is a wiring diagram. It contains no phototransduction, no
odorant-receptor binding, no mechanotransduction: those are molecular processes
in the sensory periphery, not synapses. Some model must convert a physical
stimulus into sensory-neuron spike rates. Shiu et al. (2024) handled this by
driving chosen sensory neurons with Poisson input at a fixed rate, i.e. by
modelling optogenetic activation. We do the same, but modulate the rate using
each neuron's published stimulus tuning and its receptive field derived from
real FlyWire column assignments.

Everything downstream of this file -- every synapse, every spike, the entire
sensorimotor transformation -- comes from the real connectome and the published
LIF model. No behaviour is decided here.

PROVENANCE OF THE LOOMING ENCODER
---------------------------------
A. REAL DATA
   - Which neurons exist (LC4: 104 cells, LPLC2: 210 cells in FlyWire v783).
   - Each cell's receptive-field centre and radius, derived from the real
     column assignments of its presynaptic partners (see retinotopy.py).
B. PUBLISHED PHYSIOLOGY
   - LC4 encodes angular VELOCITY (dtheta/dt) and LPLC2 encodes angular SIZE
     (theta) of a looming object; the Giant Fibre integrates both channels.
     von Reyn et al. 2017, Nat Neurosci 20:1176-1186, doi:10.1038/nn.4600
   - LPLC2 is selective for outward (expanding) motion.
     Klapoetke et al. 2017, Nature 551:237-241, doi:10.1038/nature24626
   - Both classes are strongly looming-responsive and converge on DNp01.
     Ache et al. 2019, Curr Biol 29:1073-1081, doi:10.1016/j.cub.2019.01.079
C. OUR APPROXIMATION (clearly a model, not a measurement)
   - The *functional form* of the tuning is a saturating (Naka-Rushton) curve.
     The half-maximum constants below were chosen so the population reproduces
     the published qualitative behaviour (LC4 leading LPLC2, both peaking near
     collision). They are NOT measured single-cell parameters.
   - Receptive-field overlap is a Gaussian in angular distance.
   - Spikes are generated as a Poisson process, as in the reference model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.stimuli.looming import LoomingStimulus, angular_distance_deg


class SensoryEncoder:
    """Base: converts environment state into firing rates for real neurons."""

    #: FlyWire simulation indices this encoder drives
    indices: np.ndarray

    def rates_hz(self, t_ms: float, env) -> np.ndarray:
        raise NotImplementedError

    @property
    def provenance(self) -> dict:
        raise NotImplementedError


@dataclass
class LoomingTuning:
    """C: saturating tuning constants. Model parameters, not measurements."""
    lc4_max_hz: float = 150.0        # matches Shiu et al. default r_poi
    lplc2_max_hz: float = 150.0
    lc4_half_vel_deg_s: float = 300.0   # half-max angular velocity for LC4
    lplc2_half_size_deg: float = 25.0   # half-max angular half-size for LPLC2
    # LPLC2 is selective for OUTWARD motion (Klapoetke et al. 2017): its size
    # channel is gated by expansion, so a receding object barely drives it.
    lplc2_expansion_gate_deg_s: float = 10.0
    rf_gain: float = 1.0


class LoomingEncoder(SensoryEncoder):
    """
    Drives the real LC4 and LPLC2 populations from a looming stimulus.

    Each cell's rate is (published tuning) x (receptive-field overlap), where the
    receptive field came from real FlyWire column assignments.
    """

    CELL_TYPES = ("LC4", "LPLC2")

    def __init__(self, connectome, retinotopy, tuning: LoomingTuning = None):
        self.c = connectome
        self.tuning = tuning or LoomingTuning()

        rf = {t: retinotopy.receptive_fields(t) for t in self.CELL_TYPES}
        frames = []
        for t in self.CELL_TYPES:
            d = rf[t].dropna(subset=["azimuth_deg"]).copy()
            d["cell_type"] = t
            frames.append(d)
        import pandas as pd
        self.rf = pd.concat(frames, ignore_index=True)

        self.indices = self.rf["idx"].to_numpy(dtype=np.int64)
        self._az = self.rf["azimuth_deg"].to_numpy(dtype=np.float64)
        self._el = self.rf["elevation_deg"].to_numpy(dtype=np.float64)
        # Guard against degenerate radii for cells with few input columns.
        self._sigma = np.clip(
            self.rf["rf_radius_deg"].to_numpy(dtype=np.float64), 5.0, 60.0)
        self._is_lc4 = (self.rf["cell_type"] == "LC4").to_numpy()
        self._is_lplc2 = (self.rf["cell_type"] == "LPLC2").to_numpy()

    # ------------------------------------------------------------------ rates
    def rates_hz(self, t_ms: float, stim: LoomingStimulus) -> np.ndarray:
        st = stim.state(t_ms)
        if not st["active"] or st["half_angle_deg"] <= 0.0:
            return np.zeros(len(self.indices))

        theta = st["half_angle_deg"]
        rate_raw = st["expansion_rate_deg_s"]
        dtheta = max(0.0, rate_raw)                     # expansion only (B)

        # --- receptive-field overlap ---------------------------------------
        d = angular_distance_deg(st["azimuth_deg"], st["elevation_deg"],
                                 self._az, self._el)
        # The stimulus is a disc of angular radius theta; a cell is driven once
        # the disc edge reaches its receptive field.
        edge = np.maximum(0.0, d - theta)
        gate = np.exp(-(edge ** 2) / (2.0 * self._sigma ** 2)) * self.tuning.rf_gain

        # --- published tuning ----------------------------------------------
        tn = self.tuning
        rates = np.zeros(len(self.indices))
        rates[self._is_lc4] = (tn.lc4_max_hz * dtheta
                               / (dtheta + tn.lc4_half_vel_deg_s))
        # B: LPLC2 = angular size, gated by outward (expanding) motion.
        exp_gate = dtheta / (dtheta + tn.lplc2_expansion_gate_deg_s)
        rates[self._is_lplc2] = (tn.lplc2_max_hz * theta
                                 / (theta + tn.lplc2_half_size_deg)) * exp_gate
        return np.clip(rates * gate, 0.0, None)

    # ------------------------------------------------------------- provenance
    @property
    def provenance(self) -> dict:
        return {
            "drives": {t: int((self.rf["cell_type"] == t).sum())
                       for t in self.CELL_TYPES},
            "receptive_fields": (
                "derived from real FlyWire v783 column_assignment.csv of each "
                "cell's presynaptic partners"),
            "tuning_source": (
                "von Reyn et al. 2017 Nat Neurosci 20:1176 (LC4 = angular "
                "velocity, LPLC2 = angular size); Klapoetke et al. 2017 Nature "
                "551:237 (LPLC2 outward-motion selectivity)"),
            "tuning_parameters": vars(self.tuning),
            "caveat": (
                "Tuning curve SHAPE is our model (saturating Naka-Rushton); "
                "half-maximum constants are fitted, not measured. Everything "
                "downstream of these rates is the real connectome."),
        }


class PopulationEncoder(SensoryEncoder):
    """
    Drives a whole population of real sensory neurons at a rate proportional to
    stimulus intensity.

    This is the scheme used by the reference implementation (Shiu et al. 2024),
    which activated chosen neurons with Poisson input to model optogenetic
    stimulation. It is used for modalities where the connectome gives us the
    right neurons but we have no spatial receptive-field model: taste, odour,
    wind, sound, head touch, temperature, humidity.

    PROVENANCE
      A. REAL DATA : which neurons (see brain/sensory/modalities.py, each entry
                     cited)
      B. PUBLISHED : the reference model's Poisson activation scheme, and the
                     experimental work identifying what each population detects
      C. APPROX    : intensity maps linearly to firing rate, 0 -> max_rate_hz.
                     Real receptor transduction is not linear and adapts; this
                     model has no adaptation.
    """

    def __init__(self, connectome, modality):
        from brain.sensory.modalities import resolve_neurons
        self.c = connectome
        self.modality = modality
        self.indices = resolve_neurons(modality, connectome)
        if self.indices.size == 0 and modality.supported:
            raise ValueError(
                "modality %r resolves to no neurons in %s"
                % (modality.key, connectome.dataset))

    def rates_hz(self, t_ms: float, stim) -> np.ndarray:
        level = stim.level(t_ms)
        if level <= 0.0:
            return np.zeros(len(self.indices))
        return np.full(len(self.indices), self.modality.max_rate_hz * level)

    @property
    def provenance(self) -> dict:
        m = self.modality
        return {
            "modality": m.key,
            "label": m.label,
            "n_neurons_driven": int(len(self.indices)),
            "cell_types": list(m.cell_types),
            "label_group": m.label_group,
            "citation": m.citation,
            "doi": m.doi,
            "activation_scheme": (
                "Poisson drive as in Shiu et al. 2024 (w_syn * f_poi added to v)"),
            "caveat": (
                "Intensity-to-rate mapping is linear and has no receptor "
                "adaptation. Everything downstream is the real connectome."),
        }
