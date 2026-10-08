import numpy as np
from src.rrt import plan_rrt_connect
from src.collision import edge_is_free


def clear(p):
    return not (-20 <= p[0] <= 20 and -65 <= p[1] <= 65)


def test_detour():
    route = plan_rrt_connect([-80, 0], [80, 0], clear, limits=(-100, 100),
                             step_size=12, edge_resolution=1, seed=2,
                             max_iterations=1000)
    assert route is not None
    np.testing.assert_allclose(route[0], [-80, 0])
    np.testing.assert_allclose(route[-1], [80, 0])
    assert all(edge_is_free(a, b, clear, resolution=0.5)
               for a, b in zip(route[:-1], route[1:]))


def test_blocked_endpoint():
    assert plan_rrt_connect([-80, 0], [0, 0], clear) is None


def test_zero_iterations():
    assert plan_rrt_connect([-80, 0], [80, 0], clear,
                            max_iterations=0) is None
