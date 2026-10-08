"""Offline 2D illustration of the same RRT used for robot joint planning."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .collision import edge_is_free
from .rrt import plan_rrt


def main():
    start = np.array([-80.0, -70.0])
    goal = np.array([80.0, 70.0])
    obstacles = [(-30, -70, 20, 105), (25, 10, 30, 75)]

    def is_free(pose):
        return all(not (x <= pose[0] <= x + w and y <= pose[1] <= y + h)
                   for x, y, w, h in obstacles)

    path, nodes, parents = plan_rrt(start, goal, is_free, limits=(-100, 100),
                                    step_size=10, edge_resolution=1, seed=9)
    if path is None:
        raise RuntimeError("No path found in the demo")
    for a, b in zip(path[:-1], path[1:]):
        assert edge_is_free(a, b, is_free, resolution=0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    for (x, y, w, h) in obstacles:
        ax.add_patch(Rectangle((x, y), w, h, facecolor='gray', alpha=0.6))
    for index in range(1, len(nodes)):
        a, b = nodes[parents[index]], nodes[index]
        ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=0.5, alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label='Reconstructed path')
    ax.scatter(*start, marker='o', s=60, label='Start')
    ax.scatter(*goal, marker='x', s=80, label='Goal')
    ax.set(xlim=(-100, 100), ylim=(-100, 100), xlabel='Axis 1', ylabel='Axis 2',
           title='2D RRT demonstration (not a UR5 simulation)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / 'media' / 'rrt_2d.png'
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f'Planned {len(path)} waypoints; explored {len(nodes)} nodes')
    print(f'Validated every planned edge; saved {output}')


if __name__ == '__main__':
    main()
