import json

import numpy as np
import pytest

from src.check_scene import START, GOAL
from src.play_scene import animate_segment, load_path


def write_path(tmp_path, waypoints, **details):
    filename = tmp_path / 'path.json'
    filename.write_text(json.dumps({
        'scene': 'Q3Scene.ttt', 'units': 'degrees',
        'waypoints_deg': waypoints, **details,
    }), encoding='utf-8')
    return filename


def test_load_valid_path(tmp_path):
    filename = write_path(tmp_path, [START, GOAL])
    result = load_path(filename)
    assert result.shape == (2, 6)
    np.testing.assert_allclose(result[0], START)


@pytest.mark.parametrize('waypoints', [
    [[0] * 6, GOAL],
    [START, [0] * 6],
    [START, [180.1] * 6, GOAL],
    [START],
    [START, [0, 0], GOAL],
])
def test_reject_invalid_waypoints(tmp_path, waypoints):
    with pytest.raises((ValueError, TypeError)):
        load_path(write_path(tmp_path, waypoints))


def test_reject_wrong_scene(tmp_path):
    with pytest.raises(ValueError):
        load_path(write_path(tmp_path, [START, GOAL], scene='Q2scene.ttt'))


def test_segment_is_smooth_and_reaches_endpoint():
    seen = []
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    animate_segment([0, 0], [30, -12], lambda x: seen.append(x.copy()),
                    seconds=2.0, fps=10, sleep=sleep, clock=lambda: now[0])
    assert len(seen) == 20
    np.testing.assert_allclose(seen[-1], [30, -12])
    assert np.linalg.norm(seen[0]) < np.linalg.norm(seen[1])
    assert now[0] == pytest.approx(2.0)
