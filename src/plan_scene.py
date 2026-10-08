"""Check and plan a UR5 joint-space path in the original Q3 scene.

Run with the CoppeliaSim simulation stopped. This does not play the path.
"""

import json
from pathlib import Path

import numpy as np

from .check_scene import START, GOAL
from .collision import edge_is_free
from .rrt import plan_rrt


OUTPUT = Path(__file__).resolve().parents[1] / 'output' / 'ur5_path.json'


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sim = RemoteAPIClient().require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop the simulation before planning')

    arm = sim.getObject('/UR5')
    table = sim.getObject('/customizableTable')
    joints = [sim.getObject('/UR5/' + '/'.join(['joint'] * i))
              for i in range(1, 7)]
    original = [sim.getJointPosition(joint) for joint in joints]

    robot_shapes = sim.createCollection(0)
    table_shapes = sim.createCollection(0)
    try:
        sim.addItemToCollection(robot_shapes, sim.handle_tree, arm, 0)
        sim.addItemToCollection(table_shapes, sim.handle_tree, table, 0)
        checks = 0

        def is_free(angles_deg):
            nonlocal checks
            for joint, angle in zip(joints, np.deg2rad(angles_deg)):
                sim.setJointPosition(joint, float(angle))
            table_hit = sim.checkCollision(robot_shapes, table_shapes)[0]
            scene_hit = sim.checkCollision(robot_shapes, sim.handle_all)[0]
            if table_hit not in (0, 1) or scene_hit not in (0, 1):
                raise RuntimeError('CoppeliaSim could not determine collision state')
            checks += 1
            return table_hit == 0 and scene_hit == 0

        # A previous planning attempt must not be mistaken for today's result.
        OUTPUT.unlink(missing_ok=True)
        print('Checking start and goal against table and collidable scene objects...')
        if not is_free(START) or not is_free(GOAL):
            print('Start or goal is in collision; cannot plan this route.')
            return

        print('Checking the straight joint-space path...')
        direct_free = edge_is_free(START, GOAL, is_free, resolution=2)
        print('Direct path:', 'clear' if direct_free else 'blocked')

        print('Planning RRT (this may take a minute)...')
        path, nodes, _ = plan_rrt(
            START, GOAL, is_free, limits=(-180, 180),
            step_size=25, edge_resolution=3, goal_bias=0.55,
            max_iterations=250, seed=9,
        )
        if path is None:
            print(f'No route found within 250 iterations ({checks} collision checks).')
            return

        for first, second in zip(path[:-1], path[1:]):
            if not edge_is_free(first, second, is_free, resolution=1):
                raise RuntimeError('The planned path failed finer collision sampling')

        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(json.dumps({
            'scene': 'Q3Scene.ttt',
            'units': 'degrees',
            'collision_scope': 'robot vs table and other collidable scene objects',
            'validation_resolution_deg': 1,
            'waypoints_deg': np.round(path, 6).tolist(),
        }, indent=2) + '\n', encoding='utf-8')
        print(f'RRT found {len(path)} waypoints using {len(nodes)} tree nodes.')
        print(f'Validated each edge at 1 degree or finer ({checks} collision checks total).')
        print('Saved candidate path:', OUTPUT.relative_to(OUTPUT.parents[1]))
        print('Planning only. No path playback was performed.')
    finally:
        for joint, angle in zip(joints, original):
            sim.setJointPosition(joint, angle)
        sim.destroyCollection(robot_shapes)
        sim.destroyCollection(table_shapes)
        print('Original joint positions restored.')


if __name__ == '__main__':
    main()
