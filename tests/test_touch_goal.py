import numpy as np
import pytest
from src.touch_goal import sphere_surface

def test_surface_point():
    np.testing.assert_allclose(sphere_surface([0, 0, 0], 0.05, [1, 0, 0]), [0.05, 0, 0])

def test_surface_diagonal():
    p = sphere_surface([1, 2, 3], 0.1, [2, 3, 3])
    assert np.linalg.norm(p - [1, 2, 3]) == pytest.approx(0.1)

def test_invalid_direction():
    with pytest.raises(ValueError):
        sphere_surface([0, 0, 0], 0.1, [0, 0, 0])
