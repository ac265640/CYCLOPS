"""6-DOF quadrotor rigid-body model with motor/ESC lag and a LiPo battery.

Syllabus links
--------------
Module 1  airframe, motors, ESC, propellers, batteries
Module 4  quadcopter physical characteristics, vehicle dynamics
          (this file is the Python equivalent of the Simulink plant block)
Module 7  altering speed and increasing flight times (battery model)

Newton-Euler equations (world frame W, body frame B, z up):
    p_dot     = v
    m v_dot   = R_WB [0, 0, sum T_i]^T  - m g e_z  + R_WB F_drag,B
    q_dot     = 1/2 q (x) [0, w]
    J w_dot   = tau - w x J w - c_r w
    w_i_dot   = (w_i,cmd - w_i) / tau_motor            (ESC + BLDC motor)
    T_i = k_f w_i^2 ,   Q_i = k_m T_i
"""
import numpy as np

from ..config import G, AirframeParams, BatteryParams
from ..geometry import quat_mul, quat_to_rot, rot_to_euler


class QuadXMixer:
    """Quad-X geometry and control allocation (PX4 motor numbering).

    motor 1 front-right CCW, 2 rear-left CCW, 3 front-left CW, 4 rear-right CW
    """

    def __init__(self, af: AirframeParams):
        d = af.arm_length / np.sqrt(2)
        self.pos = np.array([[d, -d], [-d, d], [d, d], [-d, -d]])
        self.spin = np.array([+1, +1, -1, -1])           # +1 = CCW seen from above
        k = af.km_ratio
        # [T, tau_x, tau_y, tau_z]^T = A @ [T1..T4]
        self.A = np.vstack([
            np.ones(4),
            self.pos[:, 1],
            -self.pos[:, 0],
            -self.spin * k,
        ])
        self.A_inv = np.linalg.inv(self.A)

    def allocate(self, thrust, torque):
        return self.A_inv @ np.concatenate([[thrust], torque])

    def wrench(self, T):
        out = self.A @ T
        return out[0], out[1:]


class Battery:
    def __init__(self, bp: BatteryParams):
        self.p = bp
        self.reset()

    def reset(self, soc=1.0):
        self.soc = soc
        self.voltage = self.ocv()
        self.current = 0.0
        self.energy_used_wh = 0.0

    def ocv(self):
        return self.p.cells * np.interp(self.soc, self.p.soc_pts, self.p.ocv_pts)

    def step(self, shaft_power, dt):
        p_elec = shaft_power / self.p.eta_esc_motor + self.p.avionics_w
        ocv = self.ocv()
        disc = max(ocv ** 2 - 4 * p_elec * self.p.r_internal, 0.0)
        v = 0.5 * (ocv + np.sqrt(disc))           # loaded terminal voltage (sag)
        i = p_elec / max(v, 1e-3)
        self.voltage, self.current = v, i
        self.soc = max(self.soc - i * dt / 3600.0 / (self.p.capacity_mah / 1000.0), 0.0)
        self.energy_used_wh += p_elec * dt / 3600.0

    @property
    def cell_voltage(self):
        return self.voltage / self.p.cells

    @property
    def low(self):
        return self.cell_voltage < self.p.min_cell_v + 0.2


class Quadrotor:
    def __init__(self, af: AirframeParams, bp: BatteryParams):
        self.af = af
        self.mixer = QuadXMixer(af)
        self.battery = Battery(bp)
        self.J = af.J
        self.J_inv = np.linalg.inv(self.J)
        self.reset()

    def reset(self, pos=(0, 0, 0), yaw=0.0):
        self.p = np.array(pos, dtype=float)
        self.v = np.zeros(3)
        self.q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.w = np.zeros(3)
        self.rotor = np.zeros(4)
        self.on_ground = self.p[2] <= 0.01
        self.battery.reset()
        self.acc_world = np.zeros(3)

    # ------------------------------------------------------------------ helpers
    @property
    def R(self):
        return quat_to_rot(self.q)

    @property
    def euler(self):
        return rot_to_euler(self.R)

    def w_max_now(self):
        # BLDC no-load speed scales with supply voltage (Kv * V)
        return self.af.motor_w_max * self.battery.voltage / (3.85 * self.battery.p.cells)

    def thrust_to_w(self, T):
        return np.sqrt(np.maximum(T, 0.0) / self.af.kf)

    # --------------------------------------------------------------------- step
    def step(self, w_cmd, dt, external_force=np.zeros(3)):
        af = self.af
        w_cmd = np.clip(w_cmd, 0.0, self.w_max_now())
        self.rotor += (w_cmd - self.rotor) * (dt / af.motor_tau)
        T_i = af.kf * self.rotor ** 2
        thrust, torque = self.mixer.wrench(T_i)

        R = self.R
        v_body = R.T @ self.v
        f_drag = R @ (-np.array(af.lin_drag) * v_body)
        f = R @ np.array([0, 0, thrust]) + f_drag + external_force
        acc = f / af.mass - np.array([0, 0, G])

        if self.on_ground:
            if acc[2] <= 0:
                acc = np.zeros(3)
                self.v[:] = 0
                self.w[:] = 0
            else:
                self.on_ground = False

        self.acc_world = acc
        self.v += acc * dt
        self.p += self.v * dt

        if not self.on_ground:
            w_dot = self.J_inv @ (torque - np.cross(self.w, self.J @ self.w) - af.rot_drag * self.w)
            self.w += w_dot * dt
            self.q = quat_mul(self.q, np.concatenate([[1.0], 0.5 * self.w * dt]))
            self.q /= np.linalg.norm(self.q)

        if self.p[2] < 0.0:
            self.p[2] = 0.0
            if self.v[2] < 0:
                self.v[:] = 0
            self.on_ground = thrust < af.mass * G

        shaft_power = np.sum(af.km_ratio * T_i * self.rotor)
        self.battery.step(shaft_power, dt)
        return T_i

    def specific_force_body(self):
        """What an accelerometer measures (body frame)."""
        return self.R.T @ (self.acc_world + np.array([0, 0, G]))
