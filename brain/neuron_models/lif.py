"""
Leaky integrate-and-fire neuron model parameters.

PROVENANCE: category B -- published computational assumptions.

Every constant below is taken verbatim from the reference implementation of

    Shiu, P.K., Sterne, G.R., Spiller, N., et al. (2024)
    "A leaky integrate-and-fire computational model based on the connectome of
     the entire adult Drosophila brain reveals insights into sensorimotor
     processing." Nature 634, 210-219. doi:10.1038/s41586-024-07763-9
    Code: https://github.com/philshiu/Drosophila_brain_model  (model.py)

The original sources for each constant are cited inline, as in the reference code.

MODEL EQUATIONS (identical to the reference `default_params['eqs']`):

    dv/dt = (v_0 - v + g) / t_mbr     : volt (unless refractory)
    dg/dt = -g / tau                  : volt (unless refractory)

    spike when   v > v_th
    on spike     v <- v_rst ;  g <- 0
    on presynaptic spike (after t_dly):  g <- g + w

    w = sign(presynaptic neurotransmitter) * synapse_count * w_syn

Note `g` here has units of volts: it is a current-like drive term, not a
conductance. This is the formulation used in the published model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LIFParams:
    """Shiu et al. (2024) whole-brain LIF parameters. Units: ms and mV."""

    # --- membrane -----------------------------------------------------------
    # Kakaria & de Bivort 2017, doi:10.3389/fnbeh.2017.00008
    v_0: float = -52.0      # mV   resting potential
    v_rst: float = -52.0    # mV   reset potential after a spike
    v_th: float = -45.0     # mV   spike threshold
    t_mbr: float = 20.0     # ms   membrane time constant (C*R = 2 nF * 10 MOhm)

    # --- synapse ------------------------------------------------------------
    # Juergensen et al. 2021, doi:10.1088/2634-4386/ac3ba6
    tau: float = 5.0        # ms   synaptic decay time constant

    # Lazar et al. 2021, doi:10.7554/eLife.62362
    t_rfc: float = 2.2      # ms   refractory period

    # Paul et al. 2015, doi:10.3389/fncel.2015.00029
    t_dly: float = 1.8      # ms   axonal / synaptic transmission delay

    # Free parameter, fitted in Shiu et al. 2024
    w_syn: float = 0.275    # mV   voltage step contributed per single synapse

    # --- external (Poisson) drive ------------------------------------------
    # Models optogenetic / direct activation in the reference implementation.
    r_poi: float = 150.0    # Hz   default Poisson activation rate
    f_poi: float = 250.0    # -    scaling factor for the Poisson synapse

    # --- integration --------------------------------------------------------
    dt: float = 0.1         # ms   Brian2 default timestep, used by Shiu et al.

    @property
    def poisson_weight(self) -> float:
        """mV added to v per Poisson event (w_syn * f_poi = 68.75 mV).

        Large enough that a single event reliably drives a spike, which is the
        intended behaviour in the reference implementation.
        """
        return self.w_syn * self.f_poi

    @property
    def delay_steps(self) -> int:
        """Synaptic delay expressed in integration steps (1.8 ms / 0.1 ms = 18)."""
        n = round(self.t_dly / self.dt)
        if abs(n * self.dt - self.t_dly) > 1e-9:
            raise ValueError(
                "t_dly (%g ms) is not an integer multiple of dt (%g ms)"
                % (self.t_dly, self.dt)
            )
        return int(n)

    @property
    def refractory_steps(self) -> int:
        """Refractory period in integration steps (2.2 ms / 0.1 ms = 22)."""
        return int(round(self.t_rfc / self.dt))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["poisson_weight_mV"] = self.poisson_weight
        d["delay_steps"] = self.delay_steps
        d["refractory_steps"] = self.refractory_steps
        return d


DEFAULT = LIFParams()

CITATION = (
    "Shiu et al. 2024, Nature 634:210-219, doi:10.1038/s41586-024-07763-9; "
    "reference code https://github.com/philshiu/Drosophila_brain_model"
)
