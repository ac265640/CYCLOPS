"""Cascaded multicopter flight controller (PX4 architecture).

    position P  ->  velocity PID  ->  thrust vector / attitude  ->  attitude P
        ->  body-rate PID  ->  mixer  ->  ESC motor commands

Syllabus links
--------------
Module 2  flight controllers
Module 3  roll / pitch / throttle / yaw, flight modes, manual mode, GPS autopilot
Module 5  stability and control: the attitude + rate loops provide the
          artificial static/dynamic stability the bare airframe lacks;
          pitch loop = longitudinal control, yaw loop = directional control,
          roll loop = lateral/roll control.

A setpoint is a dict with a 'type' field:
    {'type': 'attitude', 'roll', 'pitch', 'yaw_rate', 'thrust'}   MANUAL / STABILIZE
    {'type': 'altitude', 'roll', 'pitch', 'yaw_rate', 'vz'}       ALTCTL
    {'type': 'velocity', 'v': (3,), 'yaw' or 'yaw_rate'}          POSCTL / OFFBOARD
    {'type': 'position', 'p': (3,), 'v_ff': (3,), 'yaw'}           AUTO
"""
import numpy as np

from ..config import G, Config
from ..geometry import euler_to_rot, rot_to_euler, wrap_angle, rot_to_quat, quat_mul, quat_to_rot


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_between(a, b):
    """Shortest rotation taking unit vector a onto unit vector b."""
    c = np.cross(a, b)
    w = 1.0 + a @ b
    if w < 1e-6:                      # anti-parallel: rotate 180 deg about any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        return np.concatenate([[0.0], axis / np.linalg.norm(axis)])
    q = np.concatenate([[w], c])
    return q / np.linalg.norm(q)


def attitude_error(q, q_d, yaw_weight):
    """PX4-style attitude error with tilt prioritised over yaw.

    1. q_red: desired attitude reached with the *shortest tilt only* (no yaw change)
    2. the remaining yaw rotation is scaled by yaw_weight
    3. error vector = 2 * vector part of q^-1 * q_cmd   (body frame, small-angle = rad)
    Prioritising tilt matters because thrust direction (i.e. position control)
    depends only on tilt, and yaw authority of a quadrotor is weak.
    """
    R, R_d = quat_to_rot(q), quat_to_rot(q_d)
    z, z_d = R[:, 2], R_d[:, 2]
    q_red = quat_mul(quat_between(z, z_d), q)
    q_mix = quat_mul(quat_conj(q_red), q_d)
    q_mix = q_mix * np.sign(q_mix[0] if q_mix[0] != 0 else 1.0)
    q_mix[0], q_mix[3] = np.clip(q_mix[0], -1, 1), np.clip(q_mix[3], -1, 1)
    q_cmd = quat_mul(q_red, np.array([np.cos(yaw_weight * np.arccos(q_mix[0])), 0, 0,
                                       np.sin(yaw_weight * np.arcsin(q_mix[3]))]))
    qe = quat_mul(quat_conj(q), q_cmd)
    return 2.0 * np.sign(qe[0] if qe[0] != 0 else 1.0) * qe[1:]


class PID:
    def __init__(self, kp, ki, kd, i_limit):
        self.kp, self.ki, self.kd = map(np.asarray, (kp, ki, kd))
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.integ = np.zeros(3)
        self.prev = None

    def __call__(self, err, dt, d_meas=None):
        self.integ = np.clip(self.integ + err * dt, -self.i_limit, self.i_limit)
        if d_meas is not None:
            deriv = d_meas
        elif self.prev is None:
            deriv = np.zeros(3)
        else:
            deriv = (err - self.prev) / dt
        self.prev = err.copy()
        return self.kp * err + self.ki * self.integ + self.kd * deriv


class FlightController:
    def __init__(self, cfg: Config, quad):
        self.cfg = cfg
        c = cfg.ctrl
        self.quad = quad
        self.mass = cfg.airframe.mass
        self.J = cfg.airframe.J
        self.vel_pid = PID(c.vel_p, c.vel_i, c.vel_d, i_limit=2.0)
        self.rate_pid = PID(c.rate_p, c.rate_i, c.rate_d, i_limit=1.0)
        self.max_tilt = np.radians(c.max_tilt_deg)
        self.max_rate = np.radians(c.max_rate_deg)
        self.max_yaw_rate = np.radians(c.max_yaw_rate_deg)
        self.hover_thrust = self.mass * G
        self.reset()

    def reset(self):
        self.vel_pid.reset()
        self.rate_pid.reset()
        self.yaw_sp = None
        self.last = {}

    # ---------------------------------------------------------------- loops
    def _velocity_to_accel(self, v_sp, vel, dt):
        c = self.cfg.ctrl
        v_sp = np.array(v_sp, dtype=float)
        sxy = np.linalg.norm(v_sp[:2])
        if sxy > c.max_vel_xy:
            v_sp[:2] *= c.max_vel_xy / sxy
        v_sp[2] = np.clip(v_sp[2], -c.max_vel_z, c.max_vel_z)
        a = self.vel_pid(v_sp - vel, dt)
        self.last['v_sp'] = v_sp
        return a

    def _accel_to_attitude(self, a_sp, yaw_sp):
        """Thrust-vector construction: desired body z axis is along m(a + g)."""
        F = self.mass * (np.asarray(a_sp) + np.array([0, 0, G]))
        F[2] = max(F[2], 0.2 * self.hover_thrust)
        # tilt limit
        tilt = np.arctan2(np.linalg.norm(F[:2]), F[2])
        if tilt > self.max_tilt:
            F[:2] *= np.tan(self.max_tilt) * F[2] / np.linalg.norm(F[:2])
        zb = F / np.linalg.norm(F)
        xc = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        R_d = np.column_stack([xb, yb, zb])
        return R_d, F

    def _attitude_to_rate(self, R, R_d, yaw_rate_ff=0.0):
        att_p = np.array(self.cfg.ctrl.att_p)
        yaw_w = att_p[2] / att_p[0]
        att_p = att_p.copy()
        att_p[2] = att_p[0]          # yaw handled by the weighting (as in PX4)
        e = attitude_error(rot_to_quat(R), rot_to_quat(R_d), yaw_w)
        rate_sp = att_p * e
        rate_sp[2] += yaw_rate_ff
        rate_sp[:2] = np.clip(rate_sp[:2], -self.max_rate, self.max_rate)
        rate_sp[2] = np.clip(rate_sp[2], -self.max_yaw_rate, self.max_yaw_rate)
        return rate_sp

    def _rate_to_torque(self, rate_sp, omega, dt):
        alpha = self.rate_pid(rate_sp - omega, dt)
        return self.J @ alpha + np.cross(omega, self.J @ omega)

    def _mix(self, thrust, torque):
        """Control allocation with saturation handling (yaw is sacrificed first)."""
        af = self.cfg.airframe
        mixer = self.quad.mixer
        t_max = af.kf * self.quad.w_max_now() ** 2
        t_min = af.kf * af.motor_w_min ** 2
        thrust = np.clip(thrust, 4 * t_min, 4 * t_max * 0.95)
        T = mixer.allocate(thrust, torque)
        if T.min() < t_min or T.max() > t_max:
            torque = torque.copy()
            torque[2] = 0.0
            T = mixer.allocate(thrust, torque)
            if T.min() < t_min or T.max() > t_max:
                base = mixer.allocate(thrust, np.zeros(3))
                delta = T - base
                with np.errstate(divide='ignore', invalid='ignore'):
                    s_hi = np.where(delta > 0, (t_max - base) / delta, np.inf)
                    s_lo = np.where(delta < 0, (t_min - base) / delta, np.inf)
                s = np.clip(min(s_hi.min(), s_lo.min(), 1.0), 0.0, 1.0)
                T = base + s * delta
        T = np.clip(T, t_min, t_max)
        return self.quad.thrust_to_w(T)

    # ----------------------------------------------------------------- main
    def update(self, sp, est, dt):
        """sp: setpoint dict, est: StateEstimator. Returns motor speed commands."""
        R = est.R_WB
        omega = est.omega
        _, _, yaw = rot_to_euler(R)
        if self.yaw_sp is None:
            self.yaw_sp = yaw
        typ = sp['type']
        yaw_rate_ff = 0.0

        if typ in ('attitude', 'altitude'):
            self.yaw_sp = wrap_angle(self.yaw_sp + sp.get('yaw_rate', 0.0) * dt)
            R_d = euler_to_rot(sp['roll'], sp['pitch'], self.yaw_sp)
            if typ == 'attitude':
                thrust = sp['thrust']
            else:
                vz_err = sp['vz'] - est.vel[2]
                a_z = self.vel_pid.kp[2] * vz_err
                thrust = self.mass * (a_z + G) / max(np.cos(sp['roll']) * np.cos(sp['pitch']), 0.5)
            yaw_rate_ff = sp.get('yaw_rate', 0.0)
        else:
            if typ == 'position':
                p_err = np.asarray(sp['p']) - est.pos
                v_sp = np.array(self.cfg.ctrl.pos_p) * p_err + np.asarray(sp.get('v_ff', np.zeros(3)))
            else:
                v_sp = np.asarray(sp['v'], dtype=float)
            a_sp = self._velocity_to_accel(v_sp, est.vel, dt)
            if 'yaw' in sp:
                err = wrap_angle(sp['yaw'] - self.yaw_sp)
                step = np.clip(err, -self.max_yaw_rate * dt, self.max_yaw_rate * dt)
                self.yaw_sp = wrap_angle(self.yaw_sp + step)
            else:
                self.yaw_sp = wrap_angle(self.yaw_sp + sp.get('yaw_rate', 0.0) * dt)
                yaw_rate_ff = sp.get('yaw_rate', 0.0)
            R_d, F = self._accel_to_attitude(a_sp, self.yaw_sp)
            thrust = F @ R[:, 2]

        rate_sp = self._attitude_to_rate(R, R_d, yaw_rate_ff)
        torque = self._rate_to_torque(rate_sp, omega, dt)
        w_cmd = self._mix(thrust, torque)
        self.last.update(R_d=R_d, rate_sp=rate_sp, thrust=thrust, torque=torque)
        return w_cmd
