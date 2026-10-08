"""Plot joint angles of the saved, collision-checked goal approach."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .trajectory import cubic_trajectory

ROOT = Path(__file__).resolve().parents[1]
PATH_FILE = ROOT / 'output' / 'touch_goal_path.json'
IMAGE_FILE = ROOT / 'media' / 'ur5_joint_trajectory.png'


def sample_path(waypoints, degrees_per_second=12, fps=20):
    """Sample each segment with the same speed limit and easing as playback."""
    path = np.asarray(waypoints, dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('Expected at least two finite six-joint waypoints')
    times = [0.0]
    positions = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        duration = max(2.0, float(np.max(np.abs(end - start))) / degrees_per_second)
        frames = max(2, int(np.ceil(duration * fps)))
        segment = cubic_trajectory(start, end, steps=frames + 1)
        times.extend(times[-1] + np.arange(1, frames + 1) * duration / frames)
        positions.extend(segment[1:])
    return np.asarray(times), np.asarray(positions)


def main():
    if not PATH_FILE.exists():
        raise SystemExit('No saved route. Run: py -m src.touch_goal plan')
    saved = json.loads(PATH_FILE.read_text(encoding='utf-8'))
    t, angles = sample_path(saved['waypoints_deg'])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for joint in range(6):
        ax.plot(t, angles[:, joint], label=f'Joint {joint + 1}', linewidth=1.7)
    ax.set(title='UR5 planned joint trajectories', xlabel='Time (s)', ylabel='Joint angle (degrees)')
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_FILE, dpi=150)
    plt.close(fig)
    print(f'Saved: {IMAGE_FILE}')
    print(f'Waypoints: {len(saved["waypoints_deg"])} | Forward motion: {t[-1]:.1f} seconds')
    print('Plot shows the planned forward motion, not measured physical joint tracking.')


if __name__ == '__main__':
    main()
