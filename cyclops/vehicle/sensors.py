"""Onboard sensor models and a light state estimator.

Module 2/3: flight controller sensors, GPS autopilot.
Module 7: "collect more data with other sensors".

IMU (gyro + accelerometer) at 200 Hz, GPS at 10 Hz, barometer, and an AHRS
attitude output. The estimator is a per-axis linear Kalman filter that
propagates with world-frame acceleration from the IMU and corrects with
GPS (x, y, z) and barometer (z) - a simplified version of what PX4's EKF2 does.
"""
import numpy as np

from ..config import G, SensorParams
from ..geometry import quat_to_rot, rot_to_quat, euler_to_rot, rot_to_euler


class SensorSuite:
    def __init__(self, sp: SensorParams, rng=None):
        self.p = sp
        self.rng = rng or np.random.default_rng()
        self.gyro_bias = self.rng.normal(0, 0.002, 3)
        self.accel_bias = self.rng.normal(0, 0.02, 3)
        self._gps_timer = 0.0
        self.gps_healthy = True
        self.imu_healthy = True

    def imu(self, quad):
        gyro = quad.w + self.gyro_bias + self.rng.normal(0, self.p.gyro_noise, 3)
        acc = quad.specific_force_body() + self.accel_bias + self.rng.normal(0, self.p.accel_noise, 3)
        return gyro, acc

    def ahrs(self, quad):
        e = rot_to_euler(quad.R) + self.rng.normal(0, self.p.att_noise, 3)
        return euler_to_rot(*e)

    def gps(self, quad, dt):
        """Returns a position fix when a new GPS sample is due, else None."""
        self._gps_timer += dt
        if self._gps_timer < 1.0 / self.p.gps_rate or not self.gps_healthy:
            return None
        self._gps_timer = 0.0
        n = np.array([self.p.gps_noise_xy, self.p.gps_noise_xy, self.p.gps_noise_z])
        return quad.p + self.rng.normal(0, 1, 3) * n

    def baro(self, quad):
        return quad.p[2] + self.rng.normal(0, self.p.baro_noise)


class StateEstimator:
    """Per-axis constant-acceleration-input KF: state [pos, vel] for x, y, z."""

    def __init__(self, sp: SensorParams):
        self.sp = sp
        self.reset()

    def reset(self, pos=(0, 0, 0)):
        self.x = np.zeros((3, 2))
        self.x[:, 0] = pos
        self.P = np.tile(np.diag([0.5, 0.1]), (3, 1, 1))
        self.R_WB = np.eye(3)
        self.omega = np.zeros(3)

    @property
    def pos(self):
        return self.x[:, 0].copy()

    @property
    def vel(self):
        return self.x[:, 1].copy()

    def predict(self, acc_body, gyro, R_WB, dt):
        self.R_WB = R_WB
        self.omega = gyro
        a_w = R_WB @ acc_body - np.array([0, 0, G])
        F = np.array([[1, dt], [0, 1]])
        B = np.array([0.5 * dt * dt, dt])
        q = (self.sp.accel_noise * 4) ** 2
        Q = q * np.outer(B, B) + np.diag([1e-6, 1e-4])
        for i in range(3):
            self.x[i] = F @ self.x[i] + B * a_w[i]
            self.P[i] = F @ self.P[i] @ F.T + Q

    def _update_axis(self, i, z, r):
        H = np.array([1.0, 0.0])
        S = H @ self.P[i] @ H + r
        K = self.P[i] @ H / S
        self.x[i] += K * (z - self.x[i, 0])
        self.P[i] = (np.eye(2) - np.outer(K, H)) @ self.P[i]

    def update_gps(self, z):
        r = [self.sp.gps_noise_xy ** 2, self.sp.gps_noise_xy ** 2, self.sp.gps_noise_z ** 2]
        for i in range(3):
            self._update_axis(i, z[i], r[i])

    def update_baro(self, z):
        self._update_axis(2, z, self.sp.baro_noise ** 2)
