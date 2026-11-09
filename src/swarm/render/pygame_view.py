"""pygame renderer. Optional, and never imported during training.

Kept entirely separate from the simulator: it takes a :class:`SwarmState` and
draws it, and holds no episode logic of its own. pygame is imported at module
scope here, so this module must only be imported by the viewer and recording
entry points -- an import-hygiene test asserts that ``swarm.sim`` and
``swarm.render`` stay free of it.

Rendering works without a display. Passing ``headless=True`` selects SDL's dummy
video driver and draws to an off-screen surface, which is what lets episodes be
recorded over SSH or in CI where there is no window to open.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from swarm.sim.state import SwarmState

# Chosen so the four cell categories stay distinguishable without relying on
# hue alone: brightness increases from empty to filled target.
COLOR_BACKGROUND = (18, 18, 20)
COLOR_GRID = (42, 42, 48)
COLOR_TARGET = (58, 72, 104)
COLOR_MODULE = (228, 228, 234)
COLOR_MODULE_ON_TARGET = (92, 198, 122)
COLOR_TEXT = (210, 210, 218)
COLOR_TEXT_DIM = (130, 130, 140)

HUD_HEIGHT = 56
MAX_WINDOW = 900


@dataclass
class PygameRenderer:
    """Draws swarm states as frames, to a window or to an array.

    ``cell_size=None`` fits the grid to :data:`MAX_WINDOW` pixels, so a 100x100
    grid and an 8x8 grid both produce a usable picture.
    """

    grid_shape: tuple[int, int]
    cell_size: int | None = None
    headless: bool = False
    show_grid_lines: bool = True
    _surface: object = field(init=False, default=None)
    _display: object = field(init=False, default=None)
    _font: object = field(init=False, default=None)
    _small_font: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.font.init()

        height, width = self.grid_shape
        if self.cell_size is None:
            self.cell_size = max(2, MAX_WINDOW // max(height, width))

        self._font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 18)
        self._small_font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self._surface = pygame.Surface(self.surface_size)

    # -- geometry --------------------------------------------------------

    @property
    def surface_size(self) -> tuple[int, int]:
        """(width, height) in pixels, including the HUD strip."""
        height, width = self.grid_shape
        return (width * self.cell_size, height * self.cell_size + HUD_HEIGHT)

    # -- drawing ---------------------------------------------------------

    def draw(self, state: SwarmState, hud: list[str] | None = None):
        """Draw one state onto the internal surface and return it."""
        pygame = self._pygame
        surface = self._surface
        surface.fill(COLOR_BACKGROUND)

        size = self.cell_size
        occupancy, target = state.occupancy, state.target

        # Target cells first, so modules standing on them draw over the top.
        for row, col in zip(*np.nonzero(target & ~occupancy), strict=True):
            self._cell(row, col, COLOR_TARGET, inset=max(1, size // 6))

        for row, col in zip(*np.nonzero(occupancy), strict=True):
            color = COLOR_MODULE_ON_TARGET if target[row, col] else COLOR_MODULE
            self._cell(row, col, color, inset=max(0, size // 10))

        if self.show_grid_lines and size >= 6:
            height, width = self.grid_shape
            for row in range(height + 1):
                y = row * size
                pygame.draw.line(surface, COLOR_GRID, (0, y), (width * size, y))
            for col in range(width + 1):
                x = col * size
                pygame.draw.line(surface, COLOR_GRID, (x, 0), (x, height * size))

        self._draw_hud(state, hud or [])
        return surface

    def _cell(self, row: int, col: int, color, inset: int = 0) -> None:
        size = self.cell_size
        rect = (
            col * size + inset,
            row * size + inset,
            max(1, size - 2 * inset),
            max(1, size - 2 * inset),
        )
        self._pygame.draw.rect(self._surface, color, rect)

    def _draw_hud(self, state: SwarmState, extra: list[str]) -> None:
        height = self.grid_shape[0] * self.cell_size
        progress = (
            f"{state.n_filled}/{state.n_targets} filled ({state.fill_fraction:.0%})"
            if state.n_targets
            else "no target"
        )
        primary = f"step {state.step}    {progress}"
        self._surface.blit(self._font.render(primary, True, COLOR_TEXT), (8, height + 8))
        if extra:
            line = self._small_font.render("    ".join(extra), True, COLOR_TEXT_DIM)
            self._surface.blit(line, (8, height + 32))

    # -- outputs ---------------------------------------------------------

    def frame(self, state: SwarmState, hud: list[str] | None = None) -> np.ndarray:
        """Draw a state and return it as an (H, W, 3) uint8 array."""
        self.draw(state, hud)
        return self._pygame.surfarray.array3d(self._surface).transpose(1, 0, 2).copy()

    def show(self, state: SwarmState, hud: list[str] | None = None) -> bool:
        """Draw to a window. Returns False once the user asks to close it."""
        pygame = self._pygame
        if self._display is None:
            self._display = pygame.display.set_mode(self.surface_size)
            pygame.display.set_caption("swarm-robot-sim")

        self.draw(state, hud)
        self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
        return True

    def tick(self, fps: float) -> None:
        self._pygame.time.wait(int(1000 / max(fps, 0.1)))

    def close(self) -> None:
        self._pygame.quit()
        self._display = None
