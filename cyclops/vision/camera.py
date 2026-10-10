"""Monocular FPV camera: a vectorised ray-casting renderer.

Produces, for every frame:
    rgb     (H, W, 3) uint8      what the onboard camera sees (with sensor noise)
    depth   (H, W)    float32    ground-truth z-depth in metres (training target only)
    inst    (H, W)    int32      obstacle id per pixel, -1 = background (auto-labelling)

Only `rgb` is ever given to the perception pipeline at run time; depth and
instance maps exist so we can *train* and *evaluate* the networks.
Pinhole model:  u = fx * Xc/Zc + cx ,  v = fy * Yc/Zc + cy
"""
import numpy as np

from ..config import CameraParams
from ..geometry import camera_rotation

SKY_TOP = np.array([0.38, 0.58, 0.88])
SKY_HOR = np.array([0.80, 0.87, 0.95])
GRASS_A = np.array([0.33, 0.50, 0.22])
GRASS_B = np.array([0.38, 0.55, 0.25])


class Camera:
    def __init__(self, cp: CameraParams, rng=None):
        self.p = cp
        self.rng = rng or np.random.default_rng()
        u, v = np.meshgrid(np.arange(cp.width), np.arange(cp.height))
        d = np.stack([(u - cp.cx) / cp.fx, (v - cp.cy) / cp.fy, np.ones_like(u, float)], -1)
        self.dirs_c = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
        self.z_of_dir = self.dirs_c[:, 2]               # cos of angle to optical axis
        self.appearance = dict(sun=np.array([0.4, 0.3, 0.85]), exposure=1.0, fog=70.0, tint=np.ones(3))

    def randomize_appearance(self):
        r = self.rng
        sun = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0.3, 1.0)])
        self.appearance = dict(sun=sun / np.linalg.norm(sun), exposure=r.uniform(0.7, 1.25),
                               fog=r.uniform(35, 120), tint=r.uniform(0.9, 1.1, 3))

    # ------------------------------------------------------------------ render
    def render(self, pos, R_WB, world, noise=True):
        cp = self.p
        R_WC = camera_rotation(R_WB, cp.tilt_deg)
        D = self.dirs_c @ R_WC.T                         # ray directions in world
        o = np.asarray(pos, float)
        n_pix = D.shape[0]
        t_best = np.full(n_pix, np.inf)
        oid = np.full(n_pix, -1, np.int32)
        normal = np.zeros((n_pix, 3))
        tex = np.ones(n_pix)

        fwd = R_WC[:, 2]
        for ob in world.obstacles:
            rel = ob.pos - o
            if ob.kind == 'cylinder':
                rel = rel + np.array([0, 0, ob.size[1] / 2])
            dist = np.linalg.norm(rel)
            rad = ob.radius_xy + (ob.size[1] if ob.kind == 'cylinder' else 0)
            if dist - rad > cp.max_range * 1.5 or (rel @ fwd < -rad):
                continue
            if ob.kind == 'cylinder':
                t, nrm, tx = self._hit_cylinder(o, D, ob)
            elif ob.kind == 'sphere':
                t, nrm, tx = self._hit_sphere(o, D, ob)
            else:
                t, nrm, tx = self._hit_box(o, D, ob)
            closer = t < t_best
            if closer.any():
                t_best[closer] = t[closer]
                oid[closer] = ob.oid
                normal[closer] = nrm[closer]
                tex[closer] = tx[closer]

        # ground plane z = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(D[:, 2] < -1e-6, -o[2] / D[:, 2], np.inf)
        ground = tg < t_best
        t_best = np.where(ground, tg, t_best)
        oid[ground] = -1

        rgb = self._shade(o, D, t_best, oid, normal, tex, ground, world)
        depth = np.minimum(t_best * self.z_of_dir, cp.max_range * 2)
        rgb = rgb.reshape(cp.height, cp.width, 3)
        if noise:
            rgb = rgb + self.rng.normal(0, cp.pixel_noise / 255.0, rgb.shape)
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return rgb8, depth.reshape(cp.height, cp.width).astype(np.float32), oid.reshape(cp.height, cp.width)

    # -------------------------------------------------------------- primitives
    @staticmethod
    def _hit_cylinder(o, D, ob):
        r, h = ob.size
        c = ob.pos
        ox, oy = o[0] - c[0], o[1] - c[1]
        a = D[:, 0] ** 2 + D[:, 1] ** 2
        b = 2 * (D[:, 0] * ox + D[:, 1] * oy)
        cc = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * cc
        t = np.full(D.shape[0], np.inf)
        ok = (disc > 0) & (a > 1e-9)
        sq = np.sqrt(np.where(ok, disc, 0))
        t1 = np.where(ok, (-b - sq) / (2 * np.where(a > 1e-9, a, 1)), np.inf)
        z = o[2] + t1 * D[:, 2]
        side = ok & (t1 > 0.05) & (z >= 0) & (z <= h)
        t[side] = t1[side]
        hit = o + np.where(np.isfinite(t), t, 0)[:, None] * D
        nrm = np.zeros_like(D)
        nrm[:, 0] = (hit[:, 0] - c[0]) / r
        nrm[:, 1] = (hit[:, 1] - c[1]) / r
        # top cap
        with np.errstate(divide='ignore', invalid='ignore'):
            tc = (h - o[2]) / D[:, 2]
        cap_xy = o[:2] + tc[:, None] * D[:, :2] - c[:2]
        cap = (tc > 0.05) & (np.sum(cap_xy ** 2, 1) <= r * r) & (tc < t)
        t[cap] = tc[cap]
        nrm[cap] = [0, 0, 1]
        ang = np.arctan2(hit[:, 1] - c[1], hit[:, 0] - c[0])
        if ob.texture == 'bark':
            tx = 0.85 + 0.15 * np.sin(ang * 9 + 3 * np.sin(hit[:, 2] * 2.0))
        else:
            tx = np.where((np.floor(hit[:, 2] / 0.5) % 2) == 0, 1.0, 0.75)
        return t, nrm, tx

    @staticmethod
    def _hit_sphere(o, D, ob):
        r = ob.size[0]
        oc = o - ob.pos
        b = D @ oc
        cc = oc @ oc - r * r
        disc = b * b - cc
        ok = disc > 0
        t = np.where(ok, -b - np.sqrt(np.where(ok, disc, 0)), np.inf)
        t = np.where(t > 0.05, t, np.inf)
        hit = o + np.where(np.isfinite(t), t, 0)[:, None] * D
        nrm = (hit - ob.pos) / r
        tx = 0.9 + 0.1 * np.sign(np.sin(6 * np.arctan2(nrm[:, 1], nrm[:, 0])))
        return t, nrm, tx

    @staticmethod
    def _hit_box(o, D, ob):
        lo, hi = ob.pos - ob.size, ob.pos + ob.size
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / D
            t0 = (lo - o) * inv
            t1 = (hi - o) * inv
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        tmin = np.nan_to_num(tmin, nan=-np.inf)
        tmax = np.nan_to_num(tmax, nan=np.inf)
        tn = tmin.max(1)
        tf = tmax.min(1)
        ok = (tn <= tf) & (tn > 0.05)
        t = np.where(ok, tn, np.inf)
        axis = tmin.argmax(1)
        nrm = np.zeros_like(D)
        nrm[np.arange(D.shape[0]), axis] = -np.sign(D[np.arange(D.shape[0]), axis])
        hit = o + np.where(ok, t, 0)[:, None] * D
        # window grid on vertical faces
        u = np.where(axis == 0, hit[:, 1], hit[:, 0])
        win = (np.mod(u, 1.6) > 0.5) & (np.mod(hit[:, 2], 2.2) > 0.9) & (axis != 2)
        tx = np.where(win, 0.55, 1.0)
        return t, nrm, tx

    # ------------------------------------------------------------------- shade
    def _shade(self, o, D, t, oid, normal, tex, ground, world):
        ap = self.appearance
        n_pix = D.shape[0]
        col = np.zeros((n_pix, 3))
        # sky
        sky = ~np.isfinite(t)
        el = np.clip(D[:, 2], 0, 1)[:, None]
        col[sky] = (SKY_HOR * (1 - el) + SKY_TOP * el)[sky]
        # ground: 2 m checker + fine noise pattern
        if ground.any():
            hit = o + t[ground, None] * D[ground]
            chk = ((np.floor(hit[:, 0] / 2) + np.floor(hit[:, 1] / 2)) % 2)[:, None]
            g = GRASS_A * (1 - chk) + GRASS_B * chk
            g = g * (0.9 + 0.1 * np.sin(hit[:, 0] * 3.1)[:, None] * np.cos(hit[:, 1] * 2.7)[:, None])
            col[ground] = g * (0.55 + 0.45 * ap['sun'][2])
        # obstacles
        obj = oid >= 0
        if obj.any():
            base = np.zeros((n_pix, 3))
            for ob in world.obstacles:
                m = oid == ob.oid
                if m.any():
                    base[m] = ob.color
            lam = np.clip(normal[obj] @ ap['sun'], 0, 1)
            col[obj] = base[obj] * (0.5 + 0.7 * lam)[:, None] * tex[obj, None]
        # atmospheric fog
        fin = np.isfinite(t)
        f = np.exp(-np.where(fin, t, 0) / ap['fog'])[:, None]
        col[fin] = (col * f + SKY_HOR * (1 - f))[fin]
        return col * ap['exposure'] * ap['tint']


def boxes_from_instances(inst, depth, min_pixels=12, max_depth=30.0):
    """Auto-label: tight 2-D boxes for every visible obstacle instance.

    Returns list of dicts {oid, box=(x0,y0,x1,y1), depth, pixels}.
    """
    labels = []
    ids = np.unique(inst)
    for i in ids[ids >= 0]:
        ys, xs = np.nonzero(inst == i)
        if len(xs) < min_pixels:
            continue
        d = float(np.percentile(depth[ys, xs], 10))
        if d > max_depth:
            continue
        labels.append(dict(oid=int(i), box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                           depth=d, pixels=len(xs)))
    return labels
