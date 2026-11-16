"""
The pygame viewer.
It draws hex cells, robots, and a target overlay from a SwarmSim's state.
It's only imported when you actually render, so training never depends on it.

Modes (same names as PettingZoo's render_mode):
  "human"     -> opens a window, frame rate capped at fps
  "rgb_array" -> draws off-screen and returns an (H, W, 3) uint8 frame, e.g. for imageio GIFs

pygame is imported inside Renderer, so the geometry helpers work without it.
Cell positions come from hexgrid.to_xy. The board array is slanted, so never draw it as an image.
"""

import numpy as np

from swarm import hexgrid

MODES = ("human", "rgb_array")

BACKGROUND = (250, 250, 250)
GRID_LINE = (210, 210, 210)
TARGET = (255, 214, 140)
ROBOT_ON_TARGET = (40, 150, 70)
ROBOT_OFF_TARGET = (60, 100, 200)
MOVED_RING = (220, 50, 50)
TEXT = (30, 30, 30)

STATUS_PX = 24  # height of the status line above the board
ROBOT_RADIUS = 0.35  # robot disc radius, as a fraction of cell_px
RING_RADIUS = 0.45  # moved-highlight ring radius, as a fraction of cell_px


def hex_corners(cell_px):
    """Pixel offsets (6, 2) of a pointy-top hexagon's corners around its centre.

    Circumradius is cell_px / sqrt(3), so neighbouring hexagons touch.
    Corners are at 30 + 60k degrees.
    """
    angles = np.radians(30 + 60 * np.arange(6))
    return (cell_px / np.sqrt(3)) * np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def cell_to_pixel(cells, cell_px, center_px):
    """Lattice cells (..., 2) -> screen pixels (..., 2).

    hexgrid.to_xy * cell_px, y flipped (screen y points down), plus center_px.
    """
    xy = hexgrid.to_xy(cells) * cell_px
    return np.stack([xy[..., 0], -xy[..., 1]], axis=-1) + np.asarray(center_px, dtype=np.float64)


class Renderer:
    """Draws one frame per call from a SwarmSim.

    Layers, back to front: empty cell outlines, target cells, robots, a status line.
    Robots on a target cell and robots off target use different colours. Robots that moved
    this step get a highlight ring.
    """

    def __init__(self, view_radius, cell_px=24, mode="human", fps=10):
        """Import pygame and size the surface to fit every cell within view_radius of the origin.

        view_radius must be <= the sim's M. Pass the largest shape radius plus a small margin,
        not M (most of the board is empty margin). The cells to draw come from
        hexgrid.window_offsets(view_radius).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        import pygame

        self.pygame = pygame
        self.view_radius = int(view_radius)
        self.cell_px = cell_px
        self.mode = mode
        self.fps = fps

        # The view disc spans x in [-R, R] and y in [-R*sqrt(3)/2, R*sqrt(3)/2], plus one hexagon.
        half_w = self.view_radius * cell_px + cell_px / 2
        half_h = self.view_radius * cell_px * np.sqrt(3) / 2 + cell_px / np.sqrt(3)
        self.size = (int(np.ceil(2 * half_w)) + 2, int(np.ceil(2 * half_h)) + STATUS_PX + 2)
        self.center = (self.size[0] / 2, STATUS_PX + (self.size[1] - STATUS_PX) / 2)

        self.cells = hexgrid.window_offsets(self.view_radius)
        self.cell_centers = cell_to_pixel(self.cells, cell_px, self.center)
        self.corners = hex_corners(cell_px)

        pygame.font.init()
        self.font = pygame.font.Font(None, STATUS_PX)
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("swarm")
            self.surface = pygame.display.set_mode(self.size)
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface(self.size)

    def draw(self, sim, step=None, moved=None):
        """Draw the current state of sim (positions, target, completion()).

        moved: the (N,) bool array from sim.apply_actions, used for highlighting.
        step: shown in the status line next to filled / N and completion.
        Robots outside view_radius are clipped; the status line counts them.
        Returns the RGB frame in "rgb_array" mode, otherwise None.
        """
        if self.view_radius > sim.m:
            raise ValueError(f"view_radius {self.view_radius} is larger than the board (M={sim.m})")
        pg = self.pygame
        self.surface.fill(BACKGROUND)

        idx = hexgrid.to_index(self.cells, sim.m)
        is_target = sim.target[idx[:, 0], idx[:, 1]]
        for center, target in zip(self.cell_centers, is_target, strict=True):
            polygon = (center + self.corners).tolist()
            if target:
                pg.draw.polygon(self.surface, TARGET, polygon)
            pg.draw.polygon(self.surface, GRID_LINE, polygon, width=1)

        pos_idx = hexgrid.to_index(sim.positions, sim.m)
        on_target = sim.target[pos_idx[:, 0], pos_idx[:, 1]]
        moved = np.zeros(sim.n, dtype=bool) if moved is None else np.asarray(moved, dtype=bool)
        robot_px = cell_to_pixel(sim.positions, self.cell_px, self.center)
        for center, hit, ring in zip(robot_px, on_target, moved, strict=True):
            center = center.tolist()
            color = ROBOT_ON_TARGET if hit else ROBOT_OFF_TARGET
            pg.draw.circle(self.surface, color, center, ROBOT_RADIUS * self.cell_px)
            if ring:
                pg.draw.circle(self.surface, MOVED_RING, center, RING_RADIUS * self.cell_px, 2)

        status = f"filled {sim.filled()}/{sim.n}  completion {sim.completion():.2f}"
        if step is not None:
            status = f"step {step}  " + status
        off_view = int((hexgrid.hex_distance(sim.positions) > self.view_radius).sum())
        if off_view:
            status += f"  ({off_view} off view)"
        self.surface.blit(self.font.render(status, True, TEXT), (6, 4))

        if self.mode == "human":
            pg.event.pump()  # keeps the window responsive
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        # surfarray is (W, H, 3); frames are (H, W, 3).
        return np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2)).astype(np.uint8)

    def close(self):
        """Close the window and shut down pygame."""
        self.pygame.quit()
