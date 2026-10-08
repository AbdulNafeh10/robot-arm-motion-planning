"""Non-destructive CoppeliaSim Q3 scene check (simulation must be stopped)."""

import numpy as np

# Joint angles were taken from the original HW4 Question1 script.
START = [0.219, 0.025, 1.001, 0.0, 90.0, 180.0]
GOAL = [-0.501, 0.432, 0.598, -40.8, 48.59, 109.1]


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sim = RemoteAPIClient().require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop the simulation before running this check')

    arm = sim.getObject('/UR5')
    table = sim.getObject('/customizableTable')
    joints = [sim.getObject('/UR5/' + '/'.join(['joint'] * i))
              for i in range(1, 7)]

    robot_shapes = sim.createCollection(0)
    table_shapes = sim.createCollection(0)
    sim.addItemToCollection(robot_shapes, sim.handle_tree, arm, 0)
    sim.addItemToCollection(table_shapes, sim.handle_tree, table, 0)

    saved_angles = [sim.getJointPosition(joint) for joint in joints]
    try:
        for label, pose in [('Original start', START), ('Original goal', GOAL)]:
            for joint, radians in zip(joints, np.deg2rad(pose)):
                sim.setJointPosition(joint, float(radians))
            actual = np.rad2deg([sim.getJointPosition(j) for j in joints])
            result = sim.checkCollision(robot_shapes, table_shapes)
            print(label)
            print('  Measured joints (deg):', np.round(actual, 2).tolist())
            print('  Robot/table collision result:', result[0])
    finally:
        for joint, angle in zip(joints, saved_angles):
            sim.setJointPosition(joint, angle)
        sim.destroyCollection(robot_shapes)
        sim.destroyCollection(table_shapes)
    print('Restored original joint positions.')
    print('Note: A zero result alone does not establish whole-scene collision safety.')


if __name__ == '__main__':
    main()
