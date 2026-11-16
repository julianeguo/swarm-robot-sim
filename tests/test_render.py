# Checks for render.py. The geometry tests need no pygame; Renderer tests skip without it.

import os
import subprocess
import sys

import numpy as np
import pytest

from swarm import hexgrid, render
from swarm.sim import SwarmSim

# --- Geometry ------------------------------------------------------------------------------


def test_hex_corners_on_circumcircle():
    corners = render.hex_corners(30)
    assert corners.shape == (6, 2)
    assert np.allclose(np.hypot(corners[:, 0], corners[:, 1]), 30 / np.sqrt(3))


def test_neighbouring_hexagons_share_an_edge():
    """Each hexagon shares exactly 2 corners with every neighbour: no gaps, no overlap."""
    cell_px = 20
    corners = render.hex_corners(cell_px)
    origin = render.cell_to_pixel([0, 0], cell_px, (0, 0)) + corners
    for nb in hexgrid.DIRS:
        other = render.cell_to_pixel(nb, cell_px, (0, 0)) + corners
        shared = np.isclose(origin[:, None, :], other[None, :, :]).all(-1).sum()
        assert shared == 2, nb


def test_cell_to_pixel_axes_and_flip():
    c = (100, 50)
    assert np.allclose(render.cell_to_pixel([0, 0], 10, c), c)
    assert np.allclose(render.cell_to_pixel([1, 0], 10, c), (110, 50))  # a points right
    # b is +60 degrees: up on screen, so screen y gets smaller.
    assert np.allclose(render.cell_to_pixel([0, 1], 10, c), (105, 50 - 5 * np.sqrt(3)))
    assert render.cell_to_pixel(np.zeros((4, 3, 2)), 10, c).shape == (4, 3, 2)


def test_import_does_not_load_pygame():
    """Training must not depend on pygame: importing swarm.render alone must not load it."""
    code = "import sys, swarm.render; assert 'pygame' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}  # find swarm like pytest does
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- Renderer ------------------------------------------------------------------------------


@pytest.fixture
def pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # no real window, works headless
    return pytest.importorskip("pygame")


@pytest.fixture
def scene():
    """3 robots: (0,0) and (1,0) on target, (-1,0) off it. Target (0,1) is empty."""
    sim = SwarmSim(m=4)
    sim.reset([[0, 0], [1, 0], [-1, 0]], [[0, 0], [1, 0], [0, 1]])
    return sim


def _pixel(renderer, frame, cell):
    x, y = np.rint(render.cell_to_pixel(cell, renderer.cell_px, renderer.center)).astype(int)
    return tuple(frame[y, x])


def test_rgb_array_frame_shape(pygame, scene):
    r = render.Renderer(view_radius=3, mode="rgb_array")
    frame = r.draw(scene, step=0)
    w, h = r.size
    assert frame.shape == (h, w, 3) and frame.dtype == np.uint8
    r.close()


def test_frame_colours(pygame, scene):
    r = render.Renderer(view_radius=3, cell_px=30, mode="rgb_array")
    frame = r.draw(scene)
    assert _pixel(r, frame, [0, 0]) == render.ROBOT_ON_TARGET
    assert _pixel(r, frame, [-1, 0]) == render.ROBOT_OFF_TARGET
    assert _pixel(r, frame, [0, 1]) == render.TARGET  # empty target cell
    assert _pixel(r, frame, [0, -2]) == render.BACKGROUND  # empty, not target
    r.close()


def test_moved_robots_get_a_ring(pygame, scene):
    r = render.Renderer(view_radius=3, cell_px=30, mode="rgb_array")
    plain = r.draw(scene)
    ringed = r.draw(scene, moved=[False, False, True])
    assert not np.array_equal(plain, ringed)

    # The ring sits between the robot disc and the cell edge, on the moved robot only.
    def ring_on_row(cell):
        x, y = np.rint(render.cell_to_pixel(cell, r.cell_px, r.center)).astype(int)
        row = ringed[y, x : x + int(r.cell_px / 2)]
        return (row == render.MOVED_RING).all(-1).any()

    assert ring_on_row([-1, 0])
    assert not ring_on_row([0, 0])
    r.close()


def test_human_mode_returns_none(pygame, scene):
    r = render.Renderer(view_radius=3, mode="human", fps=1000)
    assert r.draw(scene, step=1) is None
    r.close()


def test_bad_arguments(pygame, scene):
    with pytest.raises(ValueError, match="mode"):
        render.Renderer(view_radius=3, mode="video")
    r = render.Renderer(view_radius=5, mode="rgb_array")
    with pytest.raises(ValueError, match="view_radius"):
        r.draw(scene)  # board is only M=4
    r.close()


def test_status_counts_robots_outside_view(pygame):
    """Robots past view_radius are clipped, so the status line must say how many."""
    sim = SwarmSim(m=6)
    sim.reset([[0, 0], [5, 0]], [[0, 0], [1, 0]])
    r = render.Renderer(view_radius=2, mode="rgb_array")
    texts = []
    real_font = r.font

    class RecordingFont:  # pygame Font methods are read-only, so wrap the font instead
        def render(self, text, *args):
            texts.append(text)
            return real_font.render(text, *args)

    r.font = RecordingFont()
    r.draw(sim)
    assert "(1 off view)" in texts[-1]
    r.close()
