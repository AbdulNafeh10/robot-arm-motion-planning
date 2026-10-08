import numpy as np
import pytest

from src.collision import edge_is_free
from src.rrt import plan_rrt


def free_around_wall(p):
    x, y = p
    return not (-20 <= x <= 20 and -70 <= y <= 70)


def test_rrt_finds_actual_collision_free_route():
    start, goal = [-80, 0], [80, 0]
    path, nodes, parents = plan_rrt(start, goal, free_around_wall,
                                    limits=(-100, 100), step_size=12,
                                    edge_resolution=1, seed=8)
    assert path is not None
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert len(path) < len(nodes)
    assert len(parents) == len(nodes)
    for a, b in zip(path[:-1], path[1:]):
        assert edge_is_free(a, b, free_around_wall, resolution=0.5)


def test_rejects_edge_crossing_obstacle():
    assert not edge_is_free([-50, 0], [50, 0], free_around_wall, resolution=1)


def test_rejects_blocked_goal():
    path, _, _ = plan_rrt([-80, 0], [0, 0], free_around_wall)
    assert path is None


def test_rejects_blocked_start():
    path, _, _ = plan_rrt([0, 0], [80, 0], free_around_wall)
    assert path is None


def test_no_path_with_zero_iterations():
    path, _, _ = plan_rrt([-80, 0], [80, 0], free_around_wall, max_iterations=0)
    assert path is None


def test_same_start_and_goal():
    path, _, _ = plan_rrt([1, 2, 3], [1, 2, 3], lambda _: True)
    np.testing.assert_allclose(path, [[1, 2, 3]])


def test_invalid_resolution_rejected():
    with pytest.raises(ValueError):
        edge_is_free([0], [1], lambda _: True, resolution=0)


def test_six_joint_dimensions_supported():
    start = np.zeros(6)
    goal = np.ones(6) * 12
    path, _, _ = plan_rrt(start, goal, lambda _: True, step_size=15,
                          edge_resolution=2, seed=11)
    assert path is not None
    np.testing.assert_allclose(path[-1], goal)
