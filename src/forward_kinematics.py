"""Standard DH forward kinematics, based on HW2 and HW3."""

import numpy as np

from .transformations import dh_transform

# Original UR5 DH parameters from HW3, in meters and radians.
UR5_DH = np.array([
    [0, 0.089159, 0, np.pi / 2],
    [0, 0, -0.425, 0],
    [0, 0, -0.39225, 0],
    [0, 0.10915, 0, np.pi / 2],
    [0, 0.09465, 0, -np.pi / 2],
    [0, 0.0823, 0, 0],
])


def forward_kinematics(dh_parameters):
    result = np.eye(4)
    for row in dh_parameters:
        result = result @ dh_transform(*row)
    return result


def ur5_forward_kinematics(joint_angles_deg):
    angles = np.asarray(joint_angles_deg, dtype=float)
    if angles.shape != (6,):
        raise ValueError("Expected six joint angles in degrees")
    dh = UR5_DH.copy()
    dh[:, 0] = np.deg2rad(angles)
    return forward_kinematics(dh)
