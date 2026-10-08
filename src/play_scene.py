"""Preview a checked UR5 RRT route in the original Q3 CoppeliaSim scene.

The simulation must remain stopped. Joint motion here is kinematic only.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .check_scene import START, GOAL
from .collision import edge_is_free
from .trajectory import cubic_trajectory

PATH_FILE = Path(__file__).resolve().parents[1] / 'output' / 'ur5_path.json'


def load_path(filename):
    """Load and check the format and endpoints of a planned joint-space path."""
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    if data.get('scene') != 'Q3Scene.ttt' or data.get('units') != 'degrees':
        raise ValueError('Expected a Q3Scene path with joint angles in degrees')

    path = np.asarray(data.get('waypoints_deg'), dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
        raise ValueError('Expected two or more 6-joint waypoints')
    if not np.isfinite(path).all() or np.any(np.abs(path) > 180.000001):
        raise ValueError('Path has invalid joint angles')
    if not np.allclose(path[0], START, atol=0.001) or not np.allclose(
            path[-1], GOAL, atol=0.001):
        raise ValueError('Path endpoints do not match the original HW4 poses')
    return path


def animate_segment(start, end, set_joints, *, seconds=1.8, fps=20, sleep=time.sleep,
                    clock=time.monotonic):
    """Show one segment with eased-in/eased-out joint positions."""
    start, end = np.asarray(start), np.asarray(end)
    frames = max(2, math.ceil(seconds * fps))
    frames_seconds = seconds / frames
    samples = cubic_trajectory(start, end, steps=frames + 1)
    begin = clock()
    for index, pose in enumerate(samples[1:], start=1):
        delay = begin + index * frames_seconds - clock()
        if delay > 0:
            sleep(delay)
        set_joints(pose)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment-seconds', type=float, default=1.8,
                        help='Minimum seconds for each planned segment (default: 1.8)')
    args = parser.parse_args()
    if not 0.5 <= args.segment_seconds <= 20:
        parser.error('--segment-seconds must be between 0.5 and 20')
    if not PATH_FILE.exists():
        raise SystemExit('No saved path. First run: py -m src.plan_scene')
    path = load_path(PATH_FILE)

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sim = RemoteAPIClient().require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop the simulation before running this preview')

    arm = sim.getObject('/UR5')
    table = sim.getObject('/customizableTable')  # Also confirms the Q3 table exists.
    joints = [sim.getObject('/UR5/' + '/'.join(['joint'] * i))
              for i in range(1, 7)]
    original = np.rad2deg([sim.getJointPosition(joint) for joint in joints])

    robot_shapes = sim.createCollection(0)
    table_shapes = sim.createCollection(0)
    try:
        sim.addItemToCollection(robot_shapes, sim.handle_tree, arm, 0)
        sim.addItemToCollection(table_shapes, sim.handle_tree, table, 0)

        def set_joints(angles_deg):
            for joint, angle in zip(joints, np.deg2rad(angles_deg)):
                sim.setJointPosition(joint, float(angle))

        def is_free(angles_deg):
            set_joints(angles_deg)
            table_hit = sim.checkCollision(robot_shapes, table_shapes)[0]
            scene_hit = sim.checkCollision(robot_shapes, sim.handle_all)[0]
            if table_hit not in (0, 1) or scene_hit not in (0, 1):
                raise RuntimeError('Could not determine CoppeliaSim collision state')
            return table_hit == 0 and scene_hit == 0

        print('Rechecking the saved route against the current Q3 scene...')
        # Validate the approach from the current pose, plus every planned edge.
        edges = [(original, path[0]), *zip(path[:-1], path[1:])]
        for index, (start, end) in enumerate(edges):
            if not edge_is_free(start, end, is_free, resolution=0.5):
                raise RuntimeError(f'Collision in segment {index}; playback cancelled')
        print('Preflight passed: all edges sampled at 0.5 degrees or finer.')
        set_joints(original)
        print('Playing forward to the goal, then returning to your original pose.')
        print('This is kinematic preview with the simulation stopped, not a physics run.')

        def move(start, end):
            # Slow down long joint moves so the arm never races across the scene.
            seconds = max(args.segment_seconds, 1.5 * max(abs(end - start)) / 30.0)
            animate_segment(start, end, set_joints, seconds=seconds)

        if not np.allclose(original, path[0]):
            move(original, path[0])
        for index, (start, end) in enumerate(zip(path[:-1], path[1:]), start=1):
            move(start, end)
            print(f'  Completed forward segment {index}/{len(path) - 1}')
        print('Reached planned goal.')
        time.sleep(1)
        for start, end in zip(path[:0:-1], path[-2::-1]):
            move(start, end)
        if not np.allclose(original, path[0]):
            move(path[0], original)
        print('Playback complete. Original pose restored.')
    finally:
        # Even on Ctrl+C or an error, do not leave a modified robot pose.
        for joint, angle in zip(joints, np.deg2rad(original)):
            sim.setJointPosition(joint, float(angle))
        sim.destroyCollection(robot_shapes)
        sim.destroyCollection(table_shapes)


if __name__ == '__main__':
    main()
