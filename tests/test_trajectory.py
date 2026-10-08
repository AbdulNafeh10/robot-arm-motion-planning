import numpy as np
import pytest

from src.trajectory import cubic_trajectory, trajectory_for_path


def test_cubic_endpoints_and_zero_end_velocity():
    start = np.array([0.0, 90.0])
    goal = np.array([40.0, 10.0])
    poses = cubic_trajectory(start, goal, steps=1001)
    np.testing.assert_allclose(poses[0], start)
    np.testing.assert_allclose(poses[-1], goal)
    assert np.max(abs(poses[1] - poses[0])) < 0.001
    assert np.max(abs(poses[-1] - poses[-2])) < 0.001


def test_combined_trajectory_has_no_duplicate_waypoint():
    poses = trajectory_for_path([[0, 0], [10, 10], [20, 0]], steps_per_segment=11)
    assert poses.shape == (21, 2)
    np.testing.assert_allclose(poses[0], [0, 0])
    np.testing.assert_allclose(poses[10], [10, 10])
    np.testing.assert_allclose(poses[-1], [20, 0])


def test_invalid_duration():
    with pytest.raises(ValueError):
        cubic_trajectory([0], [1], duration=0)


def test_animation_reaches_endpoint():
    from src.trajectory import animate_segment
    seen = []
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    animate_segment([0, 0], [30, -12], lambda pose: seen.append(pose.copy()),
                    seconds=2, fps=10, sleep=sleep, clock=lambda: now[0])
    np.testing.assert_allclose(seen[-1], [30, -12])
    assert len(seen) == 20
