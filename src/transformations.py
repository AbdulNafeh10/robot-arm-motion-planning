"""Rotation and DH transformations, recovered from HW2."""

import numpy as np


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyz(phi, theta, psi):
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


def rpy(yaw, pitch, roll):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def dh_transform(theta, d, a, alpha):
    """Standard DH convention: RotZ(theta), TransZ(d), TransX(a), RotX(alpha)."""
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s * ca, s * sa, a * c],
        [s, c * ca, -c * sa, a * s],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])
