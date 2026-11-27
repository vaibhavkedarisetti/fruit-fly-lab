"""
Real-time visualisation of the digital fly sandbox.
"""

from __future__ import annotations

import math
import pygame

from simulation.environment.world import World


# Window settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60


class WorldViewer:
    """Pygame viewer for the sandbox."""

    def __init__(self, world: World) -> None:
        pygame.init()

        self.world = world

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        pygame.display.set_caption(
            "Digital Drosophila — Sandbox"
        )

        self.clock = pygame.time.Clock()

        self.running = True

    def draw_world(self) -> None:
        """Draw the sandbox."""

        self.screen.fill((25, 25, 30))

        # Border
        pygame.draw.rect(
            self.screen,
            (180, 180, 180),
            pygame.Rect(
                0,
                0,
                self.world.width,
                self.world.height,
            ),
            3,
        )

        # Draw exit
        if self.world.exit is not None:
            pygame.draw.circle(
                self.screen,
                (80, 220, 120),
                (
                    int(self.world.exit.x),
                    int(self.world.exit.y),
                ),
                int(self.world.exit.radius),
                3,
            )

            font = pygame.font.Font(None, 28)

            text = font.render(
                "EXIT",
                True,
                (80, 220, 120),
            )

            self.screen.blit(
                text,
                (
                    int(self.world.exit.x - 25),
                    int(self.world.exit.y - 45),
                ),
            )

    def draw_fly(
        self,
        x: float,
        y: float,
        heading: float,
    ) -> None:
        """Draw the fly as a small directional triangle."""

        size = 12.0

        # Nose
        nose = (
            x + math.cos(heading) * size,
            y + math.sin(heading) * size,
        )

        # Rear points
        left = (
            x + math.cos(heading + 2.5) * size,
            y + math.sin(heading + 2.5) * size,
        )

        right = (
            x + math.cos(heading - 2.5) * size,
            y + math.sin(heading - 2.5) * size,
        )

        pygame.draw.polygon(
            self.screen,
            (230, 230, 230),
            [nose, left, right],
        )

        pygame.draw.circle(
            self.screen,
            (40, 40, 40),
            (int(x), int(y)),
            4,
        )

    def draw_text(self, text: str, y: int) -> None:
        """Draw status text."""

        font = pygame.font.Font(None, 26)

        surface = font.render(
            text,
            True,
            (220, 220, 220),
        )

        self.screen.blit(surface, (15, y))

    def render(
        self,
        fly_x: float,
        fly_y: float,
        heading: float,
        time_ms: float,
    ) -> None:
        """Render one frame."""

        self.draw_world()

        self.draw_fly(
            fly_x,
            fly_y,
            heading,
        )

        self.draw_text(
            f"Simulation time: {time_ms:.0f} ms",
            15,
        )

        self.draw_text(
            "Brain-controlled digital Drosophila",
            42,
        )

        pygame.display.flip()

    def process_events(self) -> None:
        """Handle window events."""

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.running = False

    def close(self) -> None:
        """Close the viewer."""

        pygame.quit()