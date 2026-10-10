"""Obstacle world and mission scenarios.

Obstacle primitives
    Cylinder : trees, poles, pillars (vertical, standing on the ground)
    Sphere   : floating / moving obstacles (balloons, other drones, birds)
    Box      : buildings, walls, containers (axis-aligned)

Scenarios are tied to the application areas in syllabus Module 6:
    forest_sar        search-and-rescue corridor through a forest
    urban_inspection  infrastructure inspection between buildings and poles
    dynamic           static clutter plus obstacles crossing the flight path
    pillars           a dense slalom of pillars (stress test for local minima)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    oid: int
    kind: str                       # 'cylinder' | 'sphere' | 'box'
    pos: np.ndarray                 # cylinder: base centre; sphere: centre; box: centre
    size: np.ndarray                # cylinder: (r, h); sphere: (r,); box: half extents (3,)
    color: np.ndarray               # base RGB 0..1
    texture: str = 'plain'          # 'bark' | 'windows' | 'stripes' | 'plain'
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds: tuple = None            # (lo, hi) arrays for bouncing motion
    moving: bool = False

    def step(self, dt):
        if not self.moving:
            return
        self.pos = self.pos + self.vel * dt
        if self.bounds is not None:
            lo, hi = self.bounds
            for i in range(3):
                if self.pos[i] < lo[i] or self.pos[i] > hi[i]:
                    self.vel[i] *= -1
                    self.pos[i] = np.clip(self.pos[i], lo[i], hi[i])

    @property
    def radius_xy(self):
        """Footprint radius used for clearance / evaluation."""
        if self.kind == 'box':
            return float(np.linalg.norm(self.size[:2]))
        return float(self.size[0])

    def surface_distance(self, p):
        """Euclidean distance from point p to the obstacle surface (negative = inside)."""
        if self.kind == 'cylinder':
            r, h = self.size
            dxy = np.linalg.norm(p[:2] - self.pos[:2]) - r
            dz = max(p[2] - h, 0.0)
            if dxy < 0 and dz == 0:
                return max(dxy, p[2] - h)
            return float(np.hypot(max(dxy, 0.0), dz)) if dxy > 0 or dz > 0 else dxy
        if self.kind == 'sphere':
            return float(np.linalg.norm(p - self.pos) - self.size[0])
        q = np.abs(p - self.pos) - self.size
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0))


class World:
    def __init__(self, obstacles, start, goal, name='custom', bounds=None):
        self.obstacles = obstacles
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.name = name
        self.bounds = bounds
        self.t = 0.0

    def step(self, dt):
        self.t += dt
        for o in self.obstacles:
            o.step(dt)

    def min_distance(self, p, radius=0.0):
        """(distance from a sphere of given radius at p to the nearest surface, obstacle id)."""
        best, bid = np.inf, -1
        for o in self.obstacles:
            d = o.surface_distance(p) - radius
            if d < best:
                best, bid = d, o.oid
        return best, bid

    def by_id(self, oid):
        for o in self.obstacles:
            if o.oid == oid:
                return o
        return None


# --------------------------------------------------------------------------- scenarios
TREE_COLS = [(0.36, 0.25, 0.16), (0.30, 0.22, 0.14), (0.42, 0.32, 0.22), (0.25, 0.20, 0.12)]
POLE_COLS = [(0.55, 0.55, 0.58), (0.65, 0.62, 0.55), (0.45, 0.47, 0.50), (0.85, 0.85, 0.80)]
BLD_COLS = [(0.70, 0.62, 0.52), (0.60, 0.60, 0.65), (0.78, 0.74, 0.66), (0.52, 0.45, 0.42)]
MOVE_COLS = [(0.85, 0.15, 0.12), (0.95, 0.55, 0.05), (0.90, 0.80, 0.10), (0.20, 0.35, 0.85)]


class _Builder:
    def __init__(self, rng, start, goal, keep_out=3.0):
        self.rng = rng
        self.obs = []
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)
        self.keep_out = keep_out

    def _free(self, xy, r):
        for p in (self.start, self.goal):
            if np.linalg.norm(xy - p[:2]) < self.keep_out + r:
                return False
        for o in self.obs:
            if o.moving:
                continue
            if np.linalg.norm(xy - o.pos[:2]) < r + o.radius_xy + 1.6:
                return False
        return True

    def cylinder(self, xy, r, h, kind='tree'):
        if not self._free(np.asarray(xy), r):
            return False
        cols = TREE_COLS if kind == 'tree' else POLE_COLS
        col = np.array(cols[self.rng.integers(len(cols))]) * self.rng.uniform(0.85, 1.15)
        self.obs.append(Obstacle(len(self.obs), 'cylinder', np.array([xy[0], xy[1], 0.0]),
                                 np.array([r, h]), np.clip(col, 0, 1),
                                 'bark' if kind == 'tree' else 'stripes'))
        return True

    def box(self, xy, half, h):
        if not self._free(np.asarray(xy), float(np.linalg.norm(half))):
            return False
        col = np.array(BLD_COLS[self.rng.integers(len(BLD_COLS))]) * self.rng.uniform(0.9, 1.1)
        self.obs.append(Obstacle(len(self.obs), 'box', np.array([xy[0], xy[1], h / 2]),
                                 np.array([half[0], half[1], h / 2]), np.clip(col, 0, 1), 'windows'))
        return True

    def mover(self, pos, vel, r, lo, hi):
        col = np.array(MOVE_COLS[self.rng.integers(len(MOVE_COLS))])
        self.obs.append(Obstacle(len(self.obs), 'sphere', np.array(pos, float), np.array([r]), col,
                                 'plain', vel=np.array(vel, float), bounds=(np.array(lo), np.array(hi)),
                                 moving=True))

    def on_path(self, n, r_range, lateral=1.2, kind='tree', h=(5, 9)):
        """Place n obstacles close to the straight start->goal line (forces avoidance)."""
        L = np.linalg.norm(self.goal[:2] - self.start[:2])
        placed, tries = 0, 0
        while placed < n and tries < 400:
            tries += 1
            s = self.rng.uniform(0.15, 0.85) * L
            y = self.rng.uniform(-lateral, lateral)
            r = self.rng.uniform(*r_range)
            if self.cylinder((self.start[0] + s, self.start[1] + y), r, self.rng.uniform(*h), kind):
                placed += 1

    def scatter(self, n, r_range, area, kind='tree', h=(5, 9)):
        for _ in range(n * 20):
            if n <= 0:
                break
            xy = (self.rng.uniform(*area[0]), self.rng.uniform(*area[1]))
            if self.cylinder(xy, self.rng.uniform(*r_range), self.rng.uniform(*h), kind):
                n -= 1


SCENARIOS = ['forest_sar', 'urban_inspection', 'dynamic', 'pillars']


def make_world(name='forest_sar', seed=0, length=40.0, alt=2.0, difficulty=1.0):
    rng = np.random.default_rng(seed)
    start = np.array([0.0, 0.0, alt])
    goal = np.array([length, 0.0, alt])
    b = _Builder(rng, start, goal)
    area = ((-4, length + 6), (-12, 12))

    if name == 'forest_sar':
        b.on_path(int(4 * difficulty), (0.25, 0.55), kind='tree')
        b.scatter(int(45 * difficulty), (0.2, 0.6), area, kind='tree')
    elif name == 'urban_inspection':
        for _ in range(int(3 * difficulty)):
            s = rng.uniform(0.25, 0.8) * length
            b.box((s, rng.uniform(-2.5, 2.5)), (rng.uniform(1.0, 2.0), rng.uniform(1.0, 2.0)), rng.uniform(6, 12))
        for _ in range(int(10 * difficulty)):
            b.box((rng.uniform(*area[0]), rng.choice([-1, 1]) * rng.uniform(5, 11)),
                  (rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0)), rng.uniform(6, 14))
        b.on_path(int(2 * difficulty), (0.15, 0.3), kind='pole', h=(6, 8))
        b.scatter(int(12 * difficulty), (0.12, 0.3), area, kind='pole', h=(5, 8))
    elif name == 'dynamic':
        b.on_path(int(2 * difficulty), (0.3, 0.5), kind='tree')
        b.scatter(int(20 * difficulty), (0.2, 0.5), area, kind='tree')
        for k in range(int(3 * difficulty)):
            x = (0.3 + 0.2 * k) * length + rng.uniform(-2, 2)
            side = rng.choice([-1, 1])
            speed = rng.uniform(0.6, 1.4) * difficulty
            b.mover((x, side * rng.uniform(4, 7), alt + rng.uniform(-0.2, 0.3)), (0, -side * speed, 0),
                    rng.uniform(0.4, 0.6), (x - 1, -8, 0.8), (x + 1, 8, 3.5))
    elif name == 'pillars':
        for i, s in enumerate(np.arange(7, length - 4, 4.5)):
            b.cylinder((s, rng.uniform(-0.8, 0.8)), rng.uniform(0.35, 0.6), 8, 'pole')
            b.cylinder((s + 2.2, rng.choice([-1, 1]) * rng.uniform(2.4, 3.5)), rng.uniform(0.3, 0.5), 8, 'pole')
        b.scatter(int(10 * difficulty), (0.3, 0.5), area, kind='pole', h=(6, 9))
    else:
        raise ValueError(name)

    return World(b.obs, start, goal, name, bounds=area)


def random_world(rng, alt=2.0):
    """Highly randomised world used to generate the perception training set."""
    name = SCENARIOS[rng.integers(len(SCENARIOS))]
    w = make_world(name, seed=int(rng.integers(1 << 30)), length=rng.uniform(30, 50), alt=alt,
                   difficulty=rng.uniform(0.6, 1.6))
    # extra moving obstacles everywhere for diversity
    b = _Builder(rng, w.start, w.goal)
    b.obs = w.obstacles
    for _ in range(rng.integers(0, 5)):
        x = rng.uniform(3, w.goal[0])
        b.mover((x, rng.uniform(-8, 8), rng.uniform(0.6, 4.0)), rng.normal(0, 1.0, 3) * [1, 1, 0.2],
                rng.uniform(0.25, 0.9), (x - 3, -9, 0.5), (x + 3, 9, 4.5))
    return w
