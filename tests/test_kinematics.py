import numpy as np

from src.forward_kinematics import forward_kinematics, ur5_forward_kinematics
from src.transformations import rot_x, rot_y, rot_z, euler_zyz, rpy


def test_rotation_matrices_are_orthogonal():
    for rotation in (rot_x, rot_y, rot_z):
        matrix = rotation(np.pi / 3)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-12)


def test_rotation_combinations_match_original_conventions():
    np.testing.assert_allclose(euler_zyz(0.1, 0.2, 0.3),
                               rot_z(0.1) @ rot_y(0.2) @ rot_z(0.3))
    np.testing.assert_allclose(rpy(0.1, 0.2, 0.3),
                               rot_z(0.1) @ rot_y(0.2) @ rot_x(0.3))


def test_dh_zero_translation():
    transform = forward_kinematics([[0, 0, 0, 0]])
    np.testing.assert_allclose(transform, np.eye(4))


def test_ur5_home_pose_from_original_dh():
    actual = ur5_forward_kinematics([0] * 6)
    np.testing.assert_allclose(actual[:3, 3], [-0.81725, -0.19145, -0.005491], atol=1e-8)
    np.testing.assert_allclose(actual[3], [0, 0, 0, 1])
