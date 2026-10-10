"""Synthetic, auto-labelled training data from the simulator camera.

Each sample is a random UAV pose inside a random world (domain randomised
lighting, fog, exposure, colours). Labels come for free from the renderer's
instance and depth buffers, so no manual annotation is needed.
"""
import numpy as np

from ..config import CameraParams
from ..geometry import euler_to_rot
from ..vision.camera import Camera, boxes_from_instances
from ..world.world import random_world


def sample_close_pose(rng, world):
    """Hard example: camera 0.5-4 m from an obstacle surface, roughly facing it."""
    ob = world.obstacles[rng.integers(len(world.obstacles))]
    if ob.kind == 'box':
        return None
    ang = rng.uniform(-np.pi, np.pi)
    dist = ob.radius_xy + 0.4 + rng.uniform(0.1, 3.6)
    z = rng.uniform(0.8, 3.5) if ob.kind == 'cylinder' else ob.pos[2] + rng.uniform(-0.5, 0.5)
    p = np.array([ob.pos[0] + dist * np.cos(ang), ob.pos[1] + dist * np.sin(ang), max(z, 0.6)])
    if world.min_distance(p, 0.4)[0] < 0.1:
        return None
    yaw = ang + np.pi + rng.uniform(-0.6, 0.6)
    R = euler_to_rot(rng.normal(0, 0.1), rng.normal(0, 0.1), yaw)
    return p, R


def sample_pose(rng, world):
    L = world.goal[0]
    x = rng.uniform(-2, L)
    y = rng.uniform(-4, 4) if rng.random() < 0.7 else rng.uniform(-9, 9)
    z = rng.uniform(0.8, 4.0)
    p = np.array([x, y, z])
    # stay outside obstacles
    if world.min_distance(p, 0.4)[0] < 0.2:
        return None
    yaw = rng.normal(0, 0.5) if rng.random() < 0.8 else rng.uniform(-np.pi, np.pi)
    R = euler_to_rot(rng.normal(0, 0.12), rng.normal(0, 0.12), yaw)
    return p, R


def generate(n, seed=0, cp: CameraParams = None, world_every=25, progress=True, close_frac=0.0):
    cp = cp or CameraParams()
    rng = np.random.default_rng(seed)
    cam = Camera(cp, rng)
    rgbs = np.zeros((n, cp.height, cp.width, 3), np.uint8)
    depths = np.zeros((n, cp.height, cp.width), np.float16)
    labels = []
    poses = np.zeros((n, 12), np.float32)           # camera position (3) + R_WB (9)
    world = None
    it = range(n)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=f'render seed={seed}')
        except ImportError:
            pass
    for i in it:
        if world is None or i % world_every == 0:
            world = random_world(rng)
            cam.randomize_appearance()
            # let movers move a bit so they are spread out
            world.step(rng.uniform(0, 5))
        pose = None
        while pose is None:
            pose = sample_close_pose(rng, world) if rng.random() < close_frac else sample_pose(rng, world)
        rgb, depth, inst = cam.render(pose[0], pose[1], world)
        rgbs[i] = rgb
        poses[i] = np.concatenate([pose[0], pose[1].ravel()])
        depths[i] = np.minimum(depth, 60.0)
        labels.append([(*l['box'], l['depth']) for l in boxes_from_instances(inst, depth)])
    # flat label array: (image_idx, x0, y0, x1, y1, depth)
    flat = np.array([(i, *b) for i, ls in enumerate(labels) for b in ls], np.float32).reshape(-1, 6)
    return rgbs, depths, flat, poses
