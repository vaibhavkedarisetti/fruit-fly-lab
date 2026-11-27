"""
2D sandbox environment for the digital Drosophila.

The world does NOT tell the brain where the exit is.
It only provides sensory information through the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Exit:
    """A target area the fly can eventually discover."""

    x: float
    y: float
    radius: float = 25.0


@dataclass
class World:
    """Simple 2D sandbox."""

    width: float = 1000.0
    height: float = 700.0

    # The exit exists physically in the world.
    # The brain is NOT given these coordinates.
    exit: Exit | None = None

    def __post_init__(self) -> None:
        if self.exit is None:
            self.exit = Exit(
                x=self.width * 0.8,
                y=self.height * 0.2,
                radius=25.0,
            )

    def inside(self, x: float, y: float) -> bool:
        """Return True if a position is inside the sandbox."""

        return (
            0.0 <= x <= self.width
            and 0.0 <= y <= self.height
        )

    def at_exit(self, x: float, y: float) -> bool:
        """
        Check whether the fly has reached the exit.

        IMPORTANT:
        This is an environment-side check.
        The neural system is NOT given the exit coordinates.
        """

        assert self.exit is not None

        distance = math.hypot(
            x - self.exit.x,
            y - self.exit.y,
        )

        return distance <= self.exit.radius

    def distance_to_exit(self, x: float, y: float) -> float:
        """
        Environment telemetry only.

        This value may be used for measuring learning,
        but it must NEVER be supplied directly to the brain.
        """

        assert self.exit is not None

        return math.hypot(
            x - self.exit.x,
            y - self.exit.y,
        )