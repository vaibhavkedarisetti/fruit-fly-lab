"""
Time course of a non-spatial stimulus (taste, odour, wind, touch, temperature).

PROVENANCE: category C/D. This is a stimulus envelope, not biology. It says
"the stimulus is present, at this intensity, between these times". How that
intensity becomes a firing rate is in brain/sensory/encoders.py, and what the
brain does about it is decided entirely by the connectome simulation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PulseStimulus:
    """A stimulus that turns on, holds at `intensity`, and turns off."""

    modality_key: str = "unknown"
    intensity: float = 1.0            # 0-1, scales the sensory firing rate
    t_start_ms: float = 0.0
    duration_ms: float = float("inf")
    rise_ms: float = 5.0              # smooth onset, avoids a step discontinuity
    fall_ms: float = 20.0

    def level(self, t_ms: float) -> float:
        """Stimulus intensity in [0, 1] at time `t_ms`."""
        if t_ms < self.t_start_ms:
            return 0.0
        dt = t_ms - self.t_start_ms
        if dt < self.rise_ms and self.rise_ms > 0:
            ramp = dt / self.rise_ms
        else:
            ramp = 1.0
        if dt > self.duration_ms:
            if self.fall_ms <= 0:
                return 0.0
            decay = 1.0 - (dt - self.duration_ms) / self.fall_ms
            ramp = min(ramp, max(0.0, decay))
        return float(max(0.0, min(1.0, ramp)) * self.intensity)

    def state(self, t_ms: float) -> dict:
        lvl = self.level(t_ms)
        return {
            "t_ms": t_ms,
            "modality": self.modality_key,
            "intensity": self.intensity,
            "level": lvl,
            "active": lvl > 0.0,
            "t_start_ms": self.t_start_ms,
            "duration_ms": (None if self.duration_ms == float("inf")
                            else self.duration_ms),
        }
