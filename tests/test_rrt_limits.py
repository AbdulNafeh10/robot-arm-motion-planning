from src.rrt import plan_rrt


def test_rrt_progress_and_time_limit():
    calls = []
    path, nodes, parents = plan_rrt(
        [0, 0], [80, 80], lambda q: not (20 < q[0] < 60),
        max_iterations=100, max_seconds=1,
        progress=lambda iteration, size, elapsed: calls.append(iteration))
    assert calls and calls[0] == 0
    assert path is None
    assert len(nodes) >= 1
