"""Joint-space RRT recovered from HW4, with checked edges and parent links."""

import time

import numpy as np

from .collision import edge_is_free


def plan_rrt(start, goal, is_free, *, limits=(-180.0, 180.0),
             step_size=15.0, edge_resolution=2.0, goal_bias=0.25,
             max_iterations=2000, seed=7, max_seconds=None, progress=None):
    """Return (path, nodes, parents), where path is None if no route was found.

    Configurations use degrees. Limits may be scalars or one pair per joint.
    Collision checking is supplied by the caller, so this planner is testable
    without CoppeliaSim. Joint angles are treated as bounded, not wrapping.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.ndim != 1 or start.shape != goal.shape or not start.size:
        raise ValueError("Start and goal must be equally sized joint vectors")
    low, high = limits
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("Joint angles must be finite")
    if np.any(np.asarray(low) >= np.asarray(high)) or step_size <= 0 or edge_resolution <= 0:
        raise ValueError("Invalid joint limits or step sizes")
    if not 0 <= goal_bias <= 1 or max_iterations < 0:
        raise ValueError("Invalid RRT settings")
    if np.any(start < low) or np.any(start > high) or np.any(goal < low) or np.any(goal > high):
        raise ValueError("Start or goal is outside joint limits")

    nodes = [start.copy()]
    parents = [-1]
    if not is_free(start) or not is_free(goal):
        return None, np.array(nodes), parents
    if np.allclose(start, goal):
        return np.array(nodes), np.array(nodes), parents

    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for iteration in range(max_iterations):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        if progress is not None and iteration % 20 == 0:
            progress(iteration, len(nodes), time.monotonic() - started)
        sample = goal if rng.random() < goal_bias else rng.uniform(low, high, start.size)
        nearest_index = int(np.argmin([np.linalg.norm(node - sample) for node in nodes]))
        nearest = nodes[nearest_index]
        delta = sample - nearest
        distance = np.linalg.norm(delta)
        if distance < 1e-10:
            continue
        new_node = nearest + delta * min(1.0, step_size / distance)
        if not edge_is_free(nearest, new_node, is_free, edge_resolution):
            continue

        nodes.append(new_node)
        parents.append(nearest_index)
        current_index = len(nodes) - 1
        if np.linalg.norm(new_node - goal) <= step_size and edge_is_free(
                new_node, goal, is_free, edge_resolution):
            if not np.array_equal(new_node, goal):
                nodes.append(goal.copy())
                parents.append(current_index)
                current_index = len(nodes) - 1

            path = []
            while current_index != -1:
                path.append(nodes[current_index])
                current_index = parents[current_index]
            return np.array(path[::-1]), np.array(nodes), parents

    return None, np.array(nodes), parents


def plan_rrt_connect(start, goal, is_free, *, limits=(-180.0, 180.0),
                     step_size=18.0, edge_resolution=2.0,
                     max_iterations=500, seed=7, max_seconds=45,
                     progress=None):
    """Bidirectional RRT-Connect. Return a start-to-goal path or None.

    Both trees contain only sampled collision-free edges. A failed search
    doesn't establish that no path exists.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.ndim != 1 or start.shape != goal.shape or not start.size:
        raise ValueError('Expected matching joint vectors')
    low, high = np.asarray(limits[0]), np.asarray(limits[1])
    if (step_size <= 0 or edge_resolution <= 0 or max_iterations < 0 or
            max_seconds is not None and max_seconds <= 0 or
            np.any(low >= high) or
            np.any(start < low) or np.any(start > high) or
            np.any(goal < low) or np.any(goal > high)):
        raise ValueError('Invalid planning settings')
    if not is_free(start) or not is_free(goal):
        return None
    if np.allclose(start, goal):
        return np.array([start])

    rng = np.random.default_rng(seed)
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    begin = time.monotonic()

    def timed_out():
        return max_seconds is not None and time.monotonic() - begin >= max_seconds

    def extend(tree, target):
        nodes, parents = tree
        nearest_idx = int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
        nearest = nodes[nearest_idx]
        delta = target-nearest
        distance = np.linalg.norm(delta)
        if distance < 1e-9:
            return nearest_idx, True
        proposed = nearest + delta * min(1, step_size/distance)
        if not edge_is_free(nearest, proposed, is_free, edge_resolution):
            return None, False
        nodes.append(proposed)
        parents.append(nearest_idx)
        return len(nodes)-1, distance <= step_size

    def branch(tree, index):
        nodes, parents = tree
        path = []
        while index != -1:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    for iteration in range(max_iterations):
        if timed_out():
            break
        side = iteration % 2
        other = 1-side
        sample = (trees[other][0][-1] if rng.random() < 0.2
                  else rng.uniform(low, high, start.size))
        added, _ = extend(trees[side], sample)
        if added is not None:
            target = trees[side][0][added]
            while not timed_out():
                connected, reached = extend(trees[other], target)
                if connected is None:
                    break
                if reached:
                    if side == 0:
                        path = branch(trees[0], added) + branch(trees[1], connected)[::-1]
                    else:
                        path = branch(trees[0], connected) + branch(trees[1], added)[::-1]
                    # Remove duplicate meeting node if exactly equal.
                    clean = [path[0]]
                    for point in path[1:]:
                        if not np.allclose(point, clean[-1]):
                            clean.append(point)
                    return np.array(clean)
        if progress is not None and iteration % 25 == 0:
            progress(iteration, len(trees[0][0]), len(trees[1][0]),
                     time.monotonic()-begin)
    return None
