"""PX4 (NED world / FRD body) <-> CYCLOPS (z-up world / FLU body) conversions.

Both conventions differ by a 180 deg rotation about x:  F = diag(1, -1, -1).
    p_cyclops = F p_ned           R_cyclops = F R_ned F
"""
import numpy as np

F = np.diag([1.0, -1.0, -1.0])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def ned_to_cyclops_pos(p):
    return F @ np.asarray(p, float)


def cyclops_to_ned_vec(v):
    return F @ np.asarray(v, float)


def ned_quat_to_cyclops_R(q_wxyz):
    return F @ quat_to_rot(q_wxyz) @ F


def cyclops_yaw_to_ned(yaw):
    # heading measured from x towards y(left) becomes heading from x(north) towards y(east)
    return -yaw
