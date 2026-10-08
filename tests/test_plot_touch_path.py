import numpy as np
import pytest
from src.plot_touch_path import sample_path


def test_sample_path_endpoints_and_time():
    t, joints = sample_path([[0] * 6, [24, 0, 0, 0, 0, 0], [24, 6, 0, 0, 0, 0]])
    assert np.allclose(joints[0], [0] * 6)
    assert np.allclose(joints[-1], [24, 6, 0, 0, 0, 0])
    assert np.all(np.diff(t) > 0)
    assert t[-1] == pytest.approx(4)


def test_invalid_path_rejected():
    with pytest.raises(ValueError):
        sample_path([[0, 0], [1, 1]])
