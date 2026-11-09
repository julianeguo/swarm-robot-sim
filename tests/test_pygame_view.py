"""Renderer tests.

These run headless: the renderer draws to an off-screen surface under SDL's
dummy driver, which is what makes episode recording work without a display.
"""

import numpy as np
import pytest

from swarm.render.pygame_view import (
    COLOR_BACKGROUND,
    COLOR_MODULE,
    COLOR_MODULE_ON_TARGET,
    COLOR_TARGET,
    HUD_HEIGHT,
    MAX_WINDOW,
    PygameRenderer,
)
from swarm.render.recording import save_frames, save_gif

from .helpers import state_from_cells

pytest.importorskip("pygame")

GRID = (8, 8)
CELL = 20


@pytest.fixture
def renderer():
    view = PygameRenderer(GRID, cell_size=CELL, headless=True)
    yield view
    view.close()


def cell_centre(frame: np.ndarray, row: int, col: int, cell: int = CELL) -> tuple:
    """The pixel at the middle of a grid cell."""
    return tuple(int(v) for v in frame[row * cell + cell // 2, col * cell + cell // 2])


# -- geometry ----------------------------------------------------------


def test_surface_size_includes_the_hud(renderer):
    assert renderer.surface_size == (GRID[1] * CELL, GRID[0] * CELL + HUD_HEIGHT)


def test_frame_shape_matches_the_surface(renderer):
    state = state_from_cells([(1, 1)], GRID, target=[(4, 4)])
    frame = renderer.frame(state)
    width, height = renderer.surface_size
    assert frame.shape == (height, width, 3)
    assert frame.dtype == np.uint8


def test_cell_size_is_fitted_to_large_grids():
    """A 100x100 grid has to produce a usable picture, not a 2000px one."""
    view = PygameRenderer((100, 100), headless=True)
    try:
        assert view.cell_size == MAX_WINDOW // 100
        assert max(view.surface_size) <= MAX_WINDOW + HUD_HEIGHT
    finally:
        view.close()


def test_non_square_grids_are_handled():
    view = PygameRenderer((4, 16), cell_size=10, headless=True)
    try:
        assert view.surface_size == (160, 40 + HUD_HEIGHT)
    finally:
        view.close()


# -- what gets drawn ---------------------------------------------------


def test_each_cell_category_gets_its_own_colour(renderer):
    state = state_from_cells([(1, 1), (2, 1)], GRID, target=[(2, 1), (5, 5)])
    frame = renderer.frame(state)

    assert cell_centre(frame, 1, 1) == COLOR_MODULE  # module, off target
    assert cell_centre(frame, 2, 1) == COLOR_MODULE_ON_TARGET  # module, on target
    assert cell_centre(frame, 5, 5) == COLOR_TARGET  # target, unfilled
    assert cell_centre(frame, 7, 7) == COLOR_BACKGROUND  # empty


def test_modules_draw_over_target_cells(renderer):
    """A filled target must read as filled, not as a target with something on it."""
    state = state_from_cells([(3, 3)], GRID, target=[(3, 3)])
    frame = renderer.frame(state)
    assert cell_centre(frame, 3, 3) == COLOR_MODULE_ON_TARGET


def test_row_zero_is_drawn_at_the_top(renderer):
    """Row-major state has to render the same way the ASCII renderer does."""
    state = state_from_cells([(0, 0)], GRID, target=[(7, 7)])
    frame = renderer.frame(state)
    assert cell_centre(frame, 0, 0) == COLOR_MODULE
    assert cell_centre(frame, 7, 7) == COLOR_TARGET


def test_columns_are_the_second_axis(renderer):
    state = state_from_cells([(0, 7)], GRID, target=[(7, 0)])
    frame = renderer.frame(state)
    assert cell_centre(frame, 0, 7) == COLOR_MODULE
    assert cell_centre(frame, 7, 0) == COLOR_TARGET


def test_hud_strip_is_drawn_below_the_grid(renderer):
    state = state_from_cells([(1, 1)], GRID, target=[(4, 4)])
    frame = renderer.frame(state, hud=["greedy", "episode 1/1"])
    hud_band = frame[GRID[0] * CELL :]
    assert hud_band.shape[0] == HUD_HEIGHT
    # Text is lighter than the background, so the band cannot be uniform.
    assert hud_band.max() > COLOR_BACKGROUND[0] + 40


def test_grid_lines_are_dropped_when_cells_are_tiny():
    view = PygameRenderer((200, 200), cell_size=3, headless=True)
    try:
        state = state_from_cells([(1, 1)], (200, 200), target=[(4, 4)])
        view.frame(state)  # must not raise; lines are skipped below 6px cells
    finally:
        view.close()


def test_renderer_does_not_mutate_the_state(renderer):
    state = state_from_cells([(1, 1), (2, 1)], GRID, target=[(2, 1), (5, 5)])
    occupancy, positions = state.occupancy.copy(), state.positions.copy()
    renderer.frame(state)
    assert np.array_equal(state.occupancy, occupancy)
    assert np.array_equal(state.positions, positions)


def test_frames_differ_as_the_swarm_moves(renderer):
    first = renderer.frame(state_from_cells([(1, 1)], GRID, target=[(4, 4)]))
    second = renderer.frame(state_from_cells([(1, 2)], GRID, target=[(4, 4)]))
    assert not np.array_equal(first, second)


def test_empty_target_renders(renderer):
    """The random-walk demo has no target shape at all."""
    state = state_from_cells([(1, 1), (1, 2)], GRID)
    frame = renderer.frame(state)
    assert cell_centre(frame, 1, 1) == COLOR_MODULE


# -- recording ---------------------------------------------------------


def test_save_gif_writes_a_file(tmp_path, renderer):
    pytest.importorskip("imageio")
    state = state_from_cells([(1, 1)], GRID, target=[(4, 4)])
    frames = [renderer.frame(state) for _ in range(3)]
    path = save_gif(frames, tmp_path / "nested" / "run.gif", fps=8)
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_frames_writes_numbered_pngs(tmp_path, renderer):
    pytest.importorskip("imageio")
    state = state_from_cells([(1, 1)], GRID, target=[(4, 4)])
    directory = save_frames([renderer.frame(state)] * 2, tmp_path / "frames")
    written = sorted(p.name for p in directory.glob("*.png"))
    assert written == ["frame_0000.png", "frame_0001.png"]


def test_saving_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        save_gif([], tmp_path / "empty.gif")
    with pytest.raises(ValueError, match="no frames"):
        save_frames([], tmp_path / "empty")
