"""Closed-loop simulation: plant + sensors + estimator + commander + controller
+ camera + perception + tracker + planner, with metric logging.

Node structure mirrors the ROS 2 architecture in the report (section 8.5):

    [camera] --rgb--> [perception node] --detections--> [tracker]
                                                           |
    [estimator] --odometry------------------------------> [planning node]
                                                           |
                                    velocity/yaw setpoint (OFFBOARD)
                                                           v
    [commander] -> [flight controller (PX4)] -> [mixer/ESC] -> [quadrotor]

Rates:  physics + controller 200 Hz, camera / perception / planning 15 Hz.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..control.commander import Commander, RCInput
from ..control.flight_controller import FlightController
from ..geometry import rot_to_euler, camera_rotation
from ..perception.perception import OraclePerception
from ..perception.tracker import ObstacleTracker
from ..planning.planners import APFPlanner, APFRolloutPlanner, StraightLinePlanner
from ..vehicle.quadrotor import Quadrotor
from ..vehicle.sensors import SensorSuite, StateEstimator
from ..vision.camera import Camera, boxes_from_instances


@dataclass
class RunResult:
    outcome: str = 'running'          # success | collision | timeout
    time: float = 0.0
    log: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Simulation:
    """method: 'vision' (deep monocular pipeline), 'oracle' (ground-truth obstacles,
    upper bound) or 'baseline' (straight line, no avoidance).
    planner: 'apf_rollout' (default) or 'apf' (classic potential field)."""

    def __init__(self, world, method='vision', cfg: Config = None, perception=None, seed=0,
                 render_always=False, planner='apf_rollout'):
        self.cfg = cfg or Config()
        self.world = world
        self.method = method
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.quad = Quadrotor(c.airframe, c.battery)
        self.sensors = SensorSuite(c.sensors, self.rng)
        self.est = StateEstimator(c.sensors)
        self.fc = FlightController(c, self.quad)
        self.cmd = Commander(c, self.quad, self.sensors, self.est)
        self.camera = Camera(c.camera, self.rng)
        self.tracker = ObstacleTracker(c.planner, c.camera)
        self.render_always = render_always or method == 'vision'
        if method == 'vision':
            assert perception is not None, 'vision method needs a Perception instance'
            self.perception = perception
        elif method == 'oracle':
            self.perception = OraclePerception(c.camera, world)
        else:
            self.perception = None
        if method == 'baseline':
            self.planner = StraightLinePlanner(c.planner, c.sim.cruise_alt)
        elif planner == 'apf':
            self.planner = APFPlanner(c.planner, c.sim.cruise_alt)
        else:
            self.planner = APFRolloutPlanner(c.planner, c.sim.cruise_alt, c.camera.hfov_deg)
        self.rc = RCInput()
        self.reset()

    # --------------------------------------------------------------------- setup
    def reset(self):
        w = self.world
        start = w.start.copy()
        start[2] = 0.0
        yaw0 = np.arctan2(w.goal[1] - w.start[1], w.goal[0] - w.start[0])
        self.quad.reset(start, yaw0)
        self.est.reset(start)
        self.est.R_WB = self.quad.R
        self.fc.reset()
        self.cmd.reset(start)
        self.tracker.reset()
        self.planner.reset()
        self.t = 0.0
        self._cam_timer = 1e9
        self._baro_timer = 0.0
        self.mission_cmd = None
        self.plan_info = {}
        self.dets = []
        self.depth_pred = None
        self.frame = None
        self.result = RunResult()
        self.log = {k: [] for k in ('t', 'pos', 'vel', 'euler', 'rotor', 'volt', 'soc', 'current',
                                    'clearance', 'v_sp', 'mode', 'latency', 'n_tracks', 'est_pos')}
        self.det_stats = dict(tp=0, fp=0, fn=0, range_err=[], range_err_ground=[], range_err_size=[])
        self.min_clearance = np.inf
        self.closest_obstacle = -1
        self.last_checks = []

    def arm_and_takeoff(self):
        ok, self.last_checks = self.cmd.arm(self.t, perception_ready=self.method != 'vision' or self.perception is not None)
        if ok:
            self.cmd.set_mode(self.t, 'TAKEOFF')
        return ok

    # ---------------------------------------------------------------------- step
    def step(self):
        c = self.cfg
        dt = c.sim.dt
        self.world.step(dt)

        # ----- sensors + state estimation (200 Hz)
        gyro, acc = self.sensors.imu(self.quad)
        self.est.predict(acc, gyro, self.sensors.ahrs(self.quad), dt)
        gps = self.sensors.gps(self.quad, dt)
        if gps is not None:
            self.est.update_gps(gps)
        self._baro_timer += dt
        if self._baro_timer >= 0.05:
            self._baro_timer = 0.0
            self.est.update_baro(self.sensors.baro(self.quad))

        # ----- camera + perception + tracking + planning (15 Hz)
        self._cam_timer += dt
        if self._cam_timer >= 1.0 / c.camera.rate:
            self._cam_timer = 0.0
            self._vision_and_planning()

        # ----- commander + flight controller (200 Hz)
        if self.cmd.armed:
            sp = self.cmd.setpoint(self.t, self.rc, self.mission_cmd if self.cmd.mode == 'MISSION' else None)
            w_cmd = self.fc.update(sp, self.est, dt)
        else:
            w_cmd = np.zeros(4)
        self.quad.step(w_cmd, dt)
        self.t += dt

        # ----- safety monitor / metrics (ground truth)
        d, oid = self.world.min_distance(self.quad.p, c.planner.drone_radius)
        if self.cmd.mode in ('MISSION',) and d < self.min_clearance:
            self.min_clearance, self.closest_obstacle = d, oid
        if d < 0 and not self.quad.on_ground:
            self._finish('collision', f'COLLISION with obstacle #{oid}')
        if self.cmd.mode == 'MISSION' and np.linalg.norm(self.quad.p[:2] - self.world.goal[:2]) < c.planner.goal_tolerance:
            self._finish('success', 'GOAL REACHED')
        if self.t > c.sim.max_time and self.result.outcome == 'running':
            self._finish('timeout', 'TIMEOUT')
        self._log(d)
        return self.result.outcome

    def _vision_and_planning(self):
        c = self.cfg
        if self.cmd.mode not in ('MISSION', 'TAKEOFF', 'POSCTL', 'ALTCTL', 'MANUAL') and not self.render_always:
            return
        R_WB = self.est.R_WB
        cam_pos = self.est.pos
        if self.render_always:
            # the renderer stands in for the real camera, so it is not part of the latency budget
            self.frame = self.camera.render(self.quad.p, self.quad.R, self.world)
        t0 = time.perf_counter()
        if self.perception is not None:
            rgb = self.frame[0] if self.frame is not None else None
            dets, self.depth_pred = self.perception.process(rgb, cam_pos, R_WB)
            self.dets = dets
            if self.method == 'vision':
                self._score_detections(dets, self.frame)
            self.tracker.update(self.t, dets, cam_pos, R_WB)
        obstacles = self.tracker.obstacles()
        _, _, yaw = rot_to_euler(R_WB)
        if self.cmd.mode == 'MISSION':
            self.mission_cmd, self.plan_info = self.planner.plan(
                self.est.pos, self.est.vel, yaw, self.world.goal, obstacles, 1.0 / c.camera.rate)
        self.latency_ms = (time.perf_counter() - t0) * 1e3

    def _score_detections(self, dets, frame):
        _, depth, inst = frame
        gts = boxes_from_instances(inst, depth, min_pixels=40, max_depth=20.0)
        # score raw detector boxes: columns of a split wide obstacle are merged back into one box
        scored, groups = [], {}
        for d in dets:
            if 'column_of' in d:
                groups.setdefault(d['column_of'], []).append(d)
            else:
                scored.append(d)
        for cols in groups.values():
            b = np.array([c['box'] for c in cols])
            scored.append(dict(cols[len(cols) // 2], box=(b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()),
                               r_ground=np.nan, merged=True))
        used = set()
        for d in scored:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                v = iou(d['box'], g['box'])
                if v > best and j not in used:
                    best, bj = v, j
            if best >= 0.4:
                used.add(bj)
                self.det_stats['tp'] += 1
                if d.get('merged'):
                    continue
                gt_r = gts[bj]['depth'] / max(d['ray_c'][2], 0.2)
                self.det_stats['range_err'].append((gt_r, d['r_depth'] - gt_r))
                if np.isfinite(d['r_ground']):
                    self.det_stats['range_err_ground'].append((gt_r, d['r_ground'] - gt_r))
                self.det_stats['range_err_size'].append((gt_r, d['r_size'] - gt_r))
            else:
                # detections of far/small objects that the strict GT filter dropped are not FPs
                loose = boxes_from_instances(inst, depth, min_pixels=8, max_depth=40.0)
                if not any(iou(d['box'], g['box']) >= 0.3 for g in loose):
                    self.det_stats['fp'] += 1
        self.det_stats['fn'] += len(gts) - len(used)

    def _log(self, clearance):
        L = self.log
        L['t'].append(self.t)
        L['pos'].append(self.quad.p.copy())
        L['est_pos'].append(self.est.pos)
        L['vel'].append(self.quad.v.copy())
        L['euler'].append(self.quad.euler)
        L['rotor'].append(self.quad.rotor.copy())
        L['volt'].append(self.quad.battery.voltage)
        L['current'].append(self.quad.battery.current)
        L['soc'].append(self.quad.battery.soc)
        L['clearance'].append(clearance)
        L['v_sp'].append(self.mission_cmd['v'].copy() if self.mission_cmd is not None else np.zeros(3))
        L['mode'].append(self.cmd.mode)
        L['latency'].append(getattr(self, 'latency_ms', 0.0))
        L['n_tracks'].append(len(self.tracker.tracks))

    def _finish(self, outcome, msg):
        if self.result.outcome == 'running':
            self.result.outcome = outcome
            self.result.time = self.t
            self.cmd.log(self.t, msg)
            if outcome == 'success':
                self.cmd.set_mode(self.t, 'LAND')

    # ----------------------------------------------------------------------- run
    def run(self, max_time=None):
        if max_time is not None:
            self.cfg.sim.max_time = max_time
        self.arm_and_takeoff()
        while self.result.outcome == 'running':
            self.step()
        return self.summarize()

    def summarize(self):
        L = {k: np.array(v) if k != 'mode' else v for k, v in self.log.items()}
        mission = np.array([m == 'MISSION' for m in L['mode']])
        res = self.result
        pos = L['pos'][mission] if mission.any() else L['pos']
        s, g = self.world.start[:2], self.world.goal[:2]
        line = (g - s) / np.linalg.norm(g - s)
        rel = pos[:, :2] - s
        cross = rel[:, 0] * line[1] - rel[:, 1] * line[0]
        alt_err = pos[:, 2] - self.cfg.sim.cruise_alt
        path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
        t_mission = float(np.sum(mission) * self.cfg.sim.dt)
        lat = L['latency'][L['latency'] > 0]
        ds = self.det_stats
        tp, fp, fn = ds['tp'], ds['fp'], ds['fn']
        m = dict(
            outcome=res.outcome,
            mission_time=t_mission,
            path_length=path_len,
            min_clearance=float(self.min_clearance),
            rmse_cross_track=float(np.sqrt(np.mean(cross ** 2))),
            rmse_altitude=float(np.sqrt(np.mean(alt_err ** 2))),
            energy_wh=float(self.quad.battery.energy_used_wh),
            soc_end=float(self.quad.battery.soc),
            mean_latency_ms=float(lat.mean()) if len(lat) else 0.0,
            p95_latency_ms=float(np.percentile(lat, 95)) if len(lat) else 0.0,
            det_tp=tp, det_fp=fp, det_fn=fn,
            precision=tp / (tp + fp) if tp + fp else float('nan'),
            recall=tp / (tp + fn) if tp + fn else float('nan'),
        )
        for key in ('range_err', 'range_err_ground', 'range_err_size'):
            arr = np.array(ds[key]).reshape(-1, 2)
            if len(arr):
                m[key.replace('range_err', 'range_mae') or 'range_mae'] = float(np.mean(np.abs(arr[:, 1])))
                m[key.replace('range_err', 'range_rel')] = float(np.mean(np.abs(arr[:, 1]) / arr[:, 0]))
        res.metrics = m
        res.log = L
        res.events = list(self.cmd.events)
        return res
