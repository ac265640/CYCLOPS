"""Unit / integration tests:  python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cyclops.config import Config, G  # noqa: E402
from cyclops.geometry import euler_to_rot, rot_to_euler, rot_to_quat, quat_to_rot, camera_rotation  # noqa: E402
from cyclops.perception.tracker import ObstacleTracker, h_meas, inv_meas  # noqa: E402
from cyclops.planning.planners import APFRolloutPlanner  # noqa: E402
from cyclops.sim.simulator import Simulation  # noqa: E402
from cyclops.vehicle.quadrotor import Quadrotor  # noqa: E402
from cyclops.vision.camera import Camera, boxes_from_instances  # noqa: E402
from cyclops.world.world import World, Obstacle, make_world  # noqa: E402


def test_rotation_roundtrip():
    e = np.array([0.2, -0.3, 1.1])
    R = euler_to_rot(*e)
    assert np.allclose(rot_to_euler(R), e)
    assert np.allclose(quat_to_rot(rot_to_quat(R)), R)


def test_mixer_hover_and_torques():
    cfg = Config()
    q = Quadrotor(cfg.airframe, cfg.battery)
    T = q.mixer.allocate(cfg.airframe.mass * G, np.zeros(3))
    assert np.allclose(T, cfg.airframe.mass * G / 4)
    T = q.mixer.allocate(cfg.airframe.mass * G, np.array([0.1, 0, 0]))   # +roll torque
    thrust, torque = q.mixer.wrench(T)
    assert np.isclose(thrust, cfg.airframe.mass * G) and np.allclose(torque, [0.1, 0, 0])


def test_measurement_model_inverse():
    R_WC = camera_rotation(euler_to_rot(0.05, -0.1, 0.7))
    cam = np.array([1.0, 2.0, 2.0])
    p = np.array([6.0, 4.0, 1.5])
    az, el, r = h_meas(p, cam, R_WC)
    assert np.allclose(inv_meas(az, el, r, cam, R_WC), p)


def test_renderer_labels_obstacle_ahead():
    cfg = Config()
    ob = Obstacle(0, 'cylinder', np.array([8.0, 0, 0]), np.array([0.5, 6.0]), np.array([0.4, 0.3, 0.2]))
    w = World([ob], [0, 0, 2], [20, 0, 2])
    rgb, depth, inst = Camera(cfg.camera).render(np.array([0, 0, 2.0]), np.eye(3), w)
    labels = boxes_from_instances(inst, depth)
    assert len(labels) == 1
    assert abs(labels[0]['depth'] - 7.5) < 0.1          # surface at 8 - 0.5 m
    x0, _, x1, _ = labels[0]['box']
    assert abs((x0 + x1) / 2 - cfg.camera.cx) < 2       # centred


def test_hover_and_takeoff_are_stable():
    sim = Simulation(World([], [0, 0, 2], [100, 0, 2]), 'baseline')
    sim.arm_and_takeoff()
    for _ in range(int(4 / sim.cfg.sim.dt)):
        sim.step()
    assert abs(sim.quad.p[2] - 2.0) < 0.2
    assert np.degrees(np.abs(sim.quad.euler[:2])).max() < 15


def test_tracker_converges_on_static_obstacle():
    cfg = Config()
    tr = ObstacleTracker(cfg.planner, cfg.camera)
    rng = np.random.default_rng(0)
    target = np.array([10.0, 1.0, 2.0])
    for k in range(30):
        cam = np.array([0.1 * k, 0.0, 2.0])
        R = np.eye(3)
        az, el, r = h_meas(target, cam, camera_rotation(R))
        det = dict(az=az + rng.normal(0, 0.01), el=el, range=r - 0.3 + rng.normal(0, 0.3), width=0.6,
                   range_conf=0.9, ray_c=np.array([np.sin(az), 0, np.cos(az)]), box=(60, 20, 70, 100))
        tr.update(0.1 * k, [det], cam, R)
    obs = tr.obstacles()
    assert len(obs) == 1 and np.linalg.norm(obs[0]['pos'][:2] - target[:2]) < 0.6


def test_rollout_planner_turns_away_from_obstacle():
    cfg = Config()
    pl = APFRolloutPlanner(cfg.planner, 2.0)
    obs = [dict(id=0, pos=np.array([4.0, 0.0, 2.0]), vel=np.zeros(3), radius=0.5, sigma=0.1)]
    sp, info = pl.plan(np.array([0, 0, 2.0]), np.array([1.0, 0, 0]), 0.0, np.array([20, 0, 2.0]), obs, 1 / 15)
    assert abs(sp['v'][1]) > 0.05 or np.linalg.norm(sp['v'][:2]) < 1.0   # deviates or slows


def test_oracle_mission_success():
    sim = Simulation(make_world('pillars', seed=3), 'oracle', seed=3)
    res = sim.run(max_time=60)
    assert res.outcome == 'success' and res.metrics['min_clearance'] > 0
