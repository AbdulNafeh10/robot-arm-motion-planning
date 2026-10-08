"""Smooth cubic joint trajectories from the HW4 polynomial exercise."""

import numpy as np


def cubic_trajectory(start, end, duration=2.0, steps=51):
    """Cubic interpolation with zero velocity at the segment endpoints."""
    if duration <= 0 or steps < 2:
        raise ValueError("Duration must be positive and steps >= 2")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != end.shape:
        raise ValueError("Both poses must have the same dimensions")
    u = np.linspace(0.0, 1.0, steps)
    blend = 3 * u**2 - 2 * u**3
    return start + blend[:, None] * (end - start)


def trajectory_for_path(path, duration=2.0, steps_per_segment=51):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("Expected at least two path configurations")
    segments = [cubic_trajectory(a, b, duration, steps_per_segment)
                for a, b in zip(path[:-1], path[1:])]
    # Avoid a repeated endpoint between adjacent segments.
    return np.vstack([segments[0], *(segment[1:] for segment in segments[1:])])
