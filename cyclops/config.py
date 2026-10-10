"""Central parameter set for the whole stack.

Every physical number lives here so the report can cite a single table.
The airframe values follow the PX4/Gazebo "Iris" quadrotor model
(syllabus Module 1: airframe, motors, ESC, propellers, battery; Module 4:
quadcopter physical characteristics).
"""
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class AirframeParams:
    """Quad-X airframe + propulsion (Module 1 / Module 4)."""
    mass: float = 1.5                     # kg, incl. 4S battery
    arm_length: float = 0.25              # m, centre to motor
    inertia: tuple = (0.029, 0.029, 0.055)  # kg m^2 (Ixx, Iyy, Izz)
    # propeller: T = kf * w^2, Q = km * w^2   (Gazebo iris motorConstant / momentConstant)
    kf: float = 8.54858e-6                # N / (rad/s)^2
    km_ratio: float = 0.016               # Q = km_ratio * T  ->  m
    prop_diameter: float = 0.254          # m (10 inch)
    # brushless motor + ESC: first-order lag on rotor speed
    motor_tau: float = 0.03               # s
    motor_w_max: float = 1100.0           # rad/s at nominal battery voltage
    motor_w_min: float = 100.0            # rad/s idle when armed
    # aerodynamics
    lin_drag: tuple = (0.25, 0.25, 0.35)  # N/(m/s), body-frame linear drag
    rot_drag: float = 0.002               # N m /(rad/s)

    @property
    def J(self):
        return np.diag(self.inertia)

    @property
    def max_thrust(self):
        return 4 * self.kf * self.motor_w_max ** 2

    @property
    def thrust_to_weight(self):
        return self.max_thrust / (self.mass * G)


@dataclass
class BatteryParams:
    """LiPo pack (Module 1: batteries; Module 7: increasing flight time)."""
    cells: int = 4                 # 4S
    capacity_mah: float = 5000.0
    r_internal: float = 0.020      # ohm, whole pack
    eta_esc_motor: float = 0.75    # electrical -> shaft efficiency
    avionics_w: float = 8.0        # FC + companion computer + camera
    min_cell_v: float = 3.3        # failsafe threshold (per cell)

    # open-circuit voltage per cell vs state of charge (typical LiPo curve)
    soc_pts = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95, 1.0])
    ocv_pts = np.array([3.27, 3.50, 3.61, 3.69, 3.76, 3.85, 3.97, 4.12, 4.20])

    @property
    def nominal_v(self):
        return 3.7 * self.cells


@dataclass
class SensorParams:
    """Module 7: 'collect more data with other sensors'."""
    imu_rate: float = 200.0
    gyro_noise: float = 0.005      # rad/s
    accel_noise: float = 0.05      # m/s^2
    gps_rate: float = 10.0
    gps_noise_xy: float = 0.15     # m (RTK-ish; plain GPS would be ~1.5 m)
    gps_noise_z: float = 0.25
    baro_noise: float = 0.10
    att_noise: float = 0.003       # rad, AHRS output noise


@dataclass
class CameraParams:
    """Forward-looking monocular FPV camera (Module 7: add a camera and FPV)."""
    width: int = 160
    height: int = 120
    hfov_deg: float = 90.0
    rate: float = 15.0             # Hz
    tilt_deg: float = 0.0          # positive = pitched up relative to body
    pixel_noise: float = 4.0       # std of additive noise (0..255)
    max_range: float = 40.0

    @property
    def fx(self):
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def fy(self):
        return self.fx

    @property
    def cx(self):
        return (self.width - 1) / 2

    @property
    def cy(self):
        return (self.height - 1) / 2

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass
class ControllerParams:
    """Cascaded PX4-style controller gains (Module 2 flight controller, Module 5 stability)."""
    pos_p: tuple = (1.0, 1.0, 1.2)
    vel_p: tuple = (1.8, 1.8, 4.0)
    vel_i: tuple = (0.4, 0.4, 1.0)
    vel_d: tuple = (0.05, 0.05, 0.0)
    att_p: tuple = (6.0, 6.0, 2.5)       # roll, pitch, yaw  [(rad/s) / rad]
    # rate loop gains are angular-acceleration gains; torque = J * (PID output)
    rate_p: tuple = (16.0, 16.0, 6.0)    # [1/s]
    rate_i: tuple = (4.0, 4.0, 1.0)      # [1/s^2]
    rate_d: tuple = (0.12, 0.12, 0.0)    # [-]
    max_tilt_deg: float = 35.0
    max_vel_xy: float = 8.0
    max_vel_z: float = 2.0
    max_yaw_rate_deg: float = 90.0
    max_rate_deg: float = 220.0


@dataclass
class PlannerParams:
    """Reactive avoidance planner (report section 8.4)."""
    cruise_speed: float = 3.0          # m/s
    max_speed: float = 4.0             # hard constraint
    max_accel: float = 6.0             # m/s^2, setpoint slew limit
    max_yaw_rate_deg: float = 90.0     # constraint on heading changes
    fov_cone_deg: float = 25.0         # motion must stay within this angle of the optical axis (hfov/2 = 45)
    goal_tolerance: float = 1.0
    k_att: float = 1.0
    k_rep: float = 15.0
    influence_dist: float = 5.0        # d0: obstacles further than this are ignored
    min_clearance: float = 0.4         # required gap between hull and obstacle surface
    drone_radius: float = 0.35
    brake_accel: float = 2.5           # m/s^2 assumed braking capability
    response_lag: float = 0.35         # s, velocity-loop lag compensated in the braking rule
    vortex_gain: float = 1.2           # tangential field to escape APF local minima
    predict_horizon: float = 1.5       # s, moving-obstacle extrapolation
    track_memory: float = 4.0          # s, keep tracks that left the FOV


@dataclass
class SimParams:
    dt: float = 0.005                 # physics step (200 Hz)
    ctrl_rate: float = 200.0
    planner_rate: float = 15.0
    max_time: float = 60.0
    cruise_alt: float = 2.0


@dataclass
class Config:
    airframe: AirframeParams = field(default_factory=AirframeParams)
    battery: BatteryParams = field(default_factory=BatteryParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    camera: CameraParams = field(default_factory=CameraParams)
    ctrl: ControllerParams = field(default_factory=ControllerParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    sim: SimParams = field(default_factory=SimParams)
