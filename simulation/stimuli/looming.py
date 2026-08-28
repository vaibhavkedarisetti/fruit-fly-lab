"""
Looming visual stimulus: the geometry of an object approaching the fly.

PROVENANCE
----------
A. REAL PHYSICS -- the angular expansion of an approaching object is exact
   geometry, not a model:

       theta(t) = arctan( l / d(t) )          half-angular-size
       d(t)     = d0 - v * t                  constant-velocity approach

   The standard experimental parameterisation is the ratio l/|v| (half-size over
   approach speed), which fully determines the expansion time course:

       theta(t) = arctan( (l/|v|) / (t_c - t) ),   t_c = time of collision

   l/|v| values of 10-80 ms are the standard range used in Drosophila escape
   experiments (Card & Dickinson 2008, J Exp Biol 211:341;
   von Reyn et al. 2014, Nat Neurosci 17:962;
   Muijres et al. 2014, Science 344:172).

D. OUR ENGINEERING -- the rest of this file is bookkeeping: where the object is
   in head-centred spherical coordinates and how big it is at each timestep.

This module contains no neural model and no behavioural rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoomingStimulus:
    """
    An object of half-width `half_size_mm` approaching the fly at constant speed
    along a straight line from direction (azimuth_deg, elevation_deg).

    Angles are head-centred: azimuth 0 deg = straight ahead, positive = to the
    fly's right; elevation 0 deg = horizon, positive = up.
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    half_size_mm: float = 5.0        # l  -- half-width of the approaching object
    speed_mm_s: float = 250.0        # |v| -- approach speed
    start_distance_mm: float = 100.0  # d0
    # Angular size at which the object is treated as having arrived.
    max_half_angle_deg: float = 80.0

    t_start_ms: float = 0.0
    _t_ms: float = field(default=0.0, init=False)

    # ------------------------------------------------------------- properties
    @property
    def l_over_v_ms(self) -> float:
        """The standard looming parameter l/|v|, in ms. Infinite if stationary."""
        if self.speed_mm_s == 0.0:
            return float("inf")
        return 1000.0 * self.half_size_mm / abs(self.speed_mm_s)

    @property
    def collision_time_ms(self) -> float:
        """Time of contact. Infinite for a stationary or receding object."""
        if self.speed_mm_s <= 0.0:
            return float("inf")
        return self.t_start_ms + 1000.0 * self.start_distance_mm / self.speed_mm_s

    # ------------------------------------------------------------------ state
    def distance_mm(self, t_ms: float) -> float:
        dt_s = max(0.0, (t_ms - self.t_start_ms)) / 1000.0
        return self.start_distance_mm - self.speed_mm_s * dt_s

    def half_angle_deg(self, t_ms: float) -> float:
        """Angular half-size theta(t) in degrees. Exact geometry."""
        if t_ms < self.t_start_ms:
            return 0.0
        d = self.distance_mm(t_ms)
        cap = np.radians(self.max_half_angle_deg)
        if d <= 0.0:
            return self.max_half_angle_deg
        return float(np.degrees(min(np.arctan(self.half_size_mm / d), cap)))

    def expansion_rate_deg_s(self, t_ms: float) -> float:
        """
        dtheta/dt in deg/s, analytically. For theta = arctan(l/d) with
        d(t) = d0 - v*t:   dtheta/dt = l*v / (d^2 + l^2).

        Computed in closed form rather than by finite difference, which would
        produce a spurious spike at stimulus onset.
        """
        if t_ms < self.t_start_ms:
            return 0.0
        if self.half_angle_deg(t_ms) >= self.max_half_angle_deg:
            return 0.0                      # angular size is capped
        d = self.distance_mm(t_ms)
        if d <= 0.0:
            return 0.0
        l, v = self.half_size_mm, self.speed_mm_s
        return float(np.degrees(l * v / (d * d + l * l)))

    def state(self, t_ms: float) -> dict:
        return {
            "t_ms": t_ms,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "distance_mm": max(0.0, self.distance_mm(t_ms)),
            "half_angle_deg": self.half_angle_deg(t_ms),
            "expansion_rate_deg_s": self.expansion_rate_deg_s(t_ms),
            "l_over_v_ms": self.l_over_v_ms,
            "collision_time_ms": self.collision_time_ms,
            "active": t_ms >= self.t_start_ms,
        }


def angular_distance_deg(az1, el1, az2, el2):
    """Great-circle angular distance between two viewing directions, in degrees."""
    a1, e1 = np.radians(az1), np.radians(el1)
    a2, e2 = np.radians(az2), np.radians(el2)
    cos_d = (np.sin(e1) * np.sin(e2)
             + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2))
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
