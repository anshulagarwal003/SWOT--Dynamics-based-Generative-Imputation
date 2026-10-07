"""Load the land/ocean mask used to blank out land in map plots."""

import numpy as np


def load_land_mask(path: str) -> np.ndarray:
    """Load a land mask saved as an .npz file with a single array named 'mask'."""
    with np.load(path) as data:
        return data["mask"][:, :]
