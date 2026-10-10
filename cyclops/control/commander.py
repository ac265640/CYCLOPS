"""Commander: arming, preflight checks, flight modes and failsafes.

Syllabus links
--------------
Module 2  RC transmitters, telemetry
Module 3  preflight checks, flight modes, manual mode, GPS autopilot,
          intelligent flight modes (here: AUTO_MISSION with obstacle avoidance)
Module 6  RTL / landing failsafes for safe operation

Modes (named after the PX4 equivalents)
    MANUAL      sticks -> roll/pitch angle, yaw rate, raw throttle   (stabilize)
    ALTCTL      like MANUAL but throttle stick = climb rate           (altitude hold)
    POSCTL      sticks -> horizontal velocity; needs GPS              (GPS mode)
    TAKEOFF     climb to mission altitude
    MISSION     fly to goal; obstacle avoidance planner in the loop    (intelligent mode)
    RTL         return to launch at mission altitude
    LAND        descend and disarm on touchdown
    STANDBY     on the ground, disarmed, waiting for preflight checks
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import rot_to_euler


@dataclass
class RCInput:
    """Normalised RC sticks (Mode 2 transmitter layout)."""
    roll: float = 0.0       # right stick x   [-1, 1]
    pitch: float = 0.0      # right stick y   [-1, 1]  (+ = forward)
    yaw: float = 0.0        # left  stick x   [-1, 1]
    throttle: float = 0.5   # left  stick y   [ 0, 1]


MODES = ['STANDBY', 'MANUAL', 'ALTCTL', 'POSCTL', 'TAKEOFF', 'MISSION', 'RTL', 'LAND']


class Commander:
    def __init__(self, cfg, quad, sensors, est):
        self.cfg = cfg
        self.quad = quad
        self.sensors = sensors
        self.est = est
        self.reset()

    def reset(self, home=(0, 0, 0)):
        self.armed = False
        self.mode = 'STANDBY'
        self.home = np.array(home, dtype=float)
        self.events = []
        self.hold_pos = None
        self.failsafe = None

    def log(self, t, msg):
        self.events.append((t, msg))

    # ------------------------------------------------------------ preflight
    def preflight_checks(self, perception_ready=True):
        """Return list of (check, passed, detail). Mirrors a QGC preflight list."""
        b = self.quad.battery
        roll, pitch, _ = rot_to_euler(self.est.R_WB)
        pos_std = np.sqrt(self.est.P[:, 0, 0]).max()
        af = self.cfg.airframe
        checks = [
            ('Battery voltage', b.cell_voltage > 3.7, f'{b.voltage:.2f} V ({b.cell_voltage:.2f} V/cell)'),
            ('Battery charge', b.soc > 0.3, f'{100 * b.soc:.0f} %'),
            ('IMU / gyro', self.sensors.imu_healthy and np.linalg.norm(self.est.omega) < 0.1,
             f'|w| = {np.linalg.norm(self.est.omega):.3f} rad/s'),
            ('Level attitude', abs(roll) < np.radians(5) and abs(pitch) < np.radians(5),
             f'roll {np.degrees(roll):.1f} deg, pitch {np.degrees(pitch):.1f} deg'),
            ('GPS fix', self.sensors.gps_healthy, '3D fix'),
            ('Position estimate', pos_std < 1.0, f'sigma = {pos_std:.2f} m'),
            ('Thrust / weight', af.thrust_to_weight > 1.8, f'{af.thrust_to_weight:.2f}'),
            ('Camera + perception', perception_ready, 'detector & depth net loaded'),
        ]
        return checks

    def arm(self, t, perception_ready=True):
        checks = self.preflight_checks(perception_ready)
        failed = [c for c in checks if not c[1]]
        if failed:
            self.log(t, 'ARM REJECTED: ' + ', '.join(c[0] for c in failed))
            return False, checks
        self.armed = True
        self.home = self.est.pos
        self.log(t, 'ARMED')
        return True, checks

    def disarm(self, t):
        self.armed = False
        self.log(t, 'DISARMED')

    def set_mode(self, t, mode):
        if mode == 'POSCTL' and not self.sensors.gps_healthy:
            self.log(t, 'POSCTL rejected: no GPS')
            return
        if mode != self.mode:
            self.log(t, f'MODE {self.mode} -> {mode}')
        self.mode = mode
        self.hold_pos = self.est.pos

    # ------------------------------------------------------------- setpoint
    def setpoint(self, t, rc: RCInput, mission_cmd=None):
        """mission_cmd: velocity setpoint dict produced by the planner (MISSION)."""
        cfg = self.cfg
        b = self.quad.battery
        if self.armed and self.failsafe is None and b.low and self.mode not in ('RTL', 'LAND'):
            self.failsafe = 'LOW BATTERY'
            self.log(t, 'FAILSAFE: low battery -> RTL')
            self.set_mode(t, 'RTL')

        alt = cfg.sim.cruise_alt
        max_tilt = np.radians(cfg.ctrl.max_tilt_deg) * 0.6
        if self.mode == 'MANUAL':
            return {'type': 'attitude', 'roll': rc.roll * max_tilt, 'pitch': rc.pitch * max_tilt,
                    'yaw_rate': -rc.yaw * np.radians(90),
                    'thrust': rc.throttle * cfg.airframe.max_thrust * 0.8}
        if self.mode == 'ALTCTL':
            return {'type': 'altitude', 'roll': rc.roll * max_tilt, 'pitch': rc.pitch * max_tilt,
                    'yaw_rate': -rc.yaw * np.radians(90), 'vz': (rc.throttle - 0.5) * 2 * cfg.ctrl.max_vel_z}
        if self.mode == 'POSCTL':
            _, _, yaw = rot_to_euler(self.est.R_WB)
            fwd, left = rc.pitch * 5.0, -rc.roll * 5.0
            v = np.array([fwd * np.cos(yaw) - left * np.sin(yaw), fwd * np.sin(yaw) + left * np.cos(yaw),
                          (rc.throttle - 0.5) * 2 * cfg.ctrl.max_vel_z])
            return {'type': 'velocity', 'v': v, 'yaw_rate': -rc.yaw * np.radians(90)}
        if self.mode == 'TAKEOFF':
            target = np.array([self.home[0], self.home[1], alt])
            if self.est.pos[2] > alt - 0.15:
                self.set_mode(t, 'MISSION')
            return {'type': 'position', 'p': target, 'v_ff': np.array([0, 0, 0.8])}
        if self.mode == 'MISSION':
            if mission_cmd is None:
                return {'type': 'position', 'p': self.hold_pos}
            return mission_cmd
        if self.mode == 'RTL':
            target = np.array([self.home[0], self.home[1], alt])
            if np.linalg.norm(self.est.pos[:2] - self.home[:2]) < 0.5:
                self.set_mode(t, 'LAND')
            return {'type': 'position', 'p': target}
        # LAND (and STANDBY, which only occurs disarmed)
        p = self.hold_pos if self.hold_pos is not None else self.est.pos
        if self.armed and self.quad.on_ground and self.est.pos[2] < 0.1:
            self.disarm(t)
        return {'type': 'velocity', 'v': np.array([0.8 * (p[0] - self.est.pos[0]),
                                                    0.8 * (p[1] - self.est.pos[1]), -0.7])}
