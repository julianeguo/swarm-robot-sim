"""Writing recorded episodes to disk.

Split from the renderer so that producing frames and encoding them are
independent: the renderer needs pygame, this needs imageio, and neither is
required to run the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_gif(frames: list[np.ndarray], path: str | Path, fps: float = 8.0) -> Path:
    """Write frames to an animated GIF, creating parent directories as needed."""
    if not frames:
        raise ValueError("no frames to save")

    import imageio.v3 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.stack(frames), duration=1000 / max(fps, 0.1), loop=0)
    return path


def save_frames(frames: list[np.ndarray], directory: str | Path, stem: str = "frame") -> Path:
    """Write frames as numbered PNGs, for stepping through an episode by hand."""
    if not frames:
        raise ValueError("no frames to save")

    import imageio.v3 as imageio

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        imageio.imwrite(directory / f"{stem}_{index:04d}.png", frame)
    return directory
