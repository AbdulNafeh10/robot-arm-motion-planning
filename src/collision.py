"""Collision sampling shared by planning and tests."""

import numpy as np


def edge_is_free(start, end, is_free, resolution=2.0):
    """Sample a straight joint-space edge, including both endpoints.

    A smaller resolution detects narrower obstacles, at a higher cost.
    Sampling is an approximation, not continuous collision certification.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != end.shape:
        raise ValueError("Both configurations must have the same dimensions")
    steps = max(1, int(np.ceil(np.linalg.norm(end - start) / resolution)))
    for fraction in np.linspace(0, 1, steps + 1):
        if not is_free(start + fraction * (end - start)):
            return False
    return True
