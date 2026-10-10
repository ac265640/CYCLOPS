"""Run-time perception node: image -> obstacle detections with range & bearing.

Per detection we compute
    bearing (az, el)  from the box centre and the pinhole intrinsics
                      (exact, no learning needed -> always available)
    range estimates   three monocular cues, echoing the paper's localisation modes
        r_depth   deep monocular depth network (learned)          -> Mode 1 range
        r_ground  flat-ground contact geometry (box bottom row,
                  known altitude & attitude)                       -> classical cue
        r_size    known-size heuristic  r = f * W / w_px           -> classical cue
    range_conf        reliability of the learned range: the fraction of the box
                      centre whose predicted depth agrees with the estimate. When it is low the tracker falls
                      back to bearing-only updates (the paper's Mode 2).
"""
import os
import time

import numpy as np
import torch

from ..config import CameraParams
from ..geometry import camera_rotation
from .networks import ObstacleDetector, MonoDepthNet, to_tensor, decode_detections

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


class Perception:
    NOMINAL_WIDTH = 0.8   # m, prior obstacle width for the size heuristic
    WIDTH_CALIB = 0.85    # box-width -> object-width correction (estimated on the validation set)
    COLUMN_WIDTH = 1.0    # m, wide detections are split into columns of about this width
    SPLIT_ABOVE = 2.0     # m, only obstacles wider than this (walls, buildings) are split

    def __init__(self, cp: CameraParams, device='cpu', det_thresh=0.35, models_dir=MODELS):
        self.cp = cp
        self.device = device
        self.det = ObstacleDetector().to(device).eval()
        self.depth = MonoDepthNet().to(device).eval()
        self.det.load_state_dict(torch.load(os.path.join(models_dir, 'detector.pt'), map_location=device))
        self.depth.load_state_dict(torch.load(os.path.join(models_dir, 'depth.pt'), map_location=device))
        self.det_thresh = det_thresh
        self.K_inv = np.linalg.inv(cp.K)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.last_latency_ms = 0.0

    @torch.no_grad()
    def infer(self, rgb):
        x = to_tensor(rgb[None]).to(self.device)
        hm, wh, off = self.det(x)
        dets = decode_detections(hm, wh, off, thresh=self.det_thresh)[0]
        depth = torch.exp(self.depth(x))[0, 0].cpu().numpy()
        return dets, depth

    def process(self, rgb, cam_pos, R_WB):
        t0 = time.perf_counter()
        dets, depth_map = self.infer(rgb)
        out = []
        H, W = depth_map.shape
        R_WC = camera_rotation(R_WB, self.cp.tilt_deg)
        for x0, y0, x1, y1, score in dets:
            x0, x1 = np.clip([x0, x1], 0, W - 1)
            y0, y1 = np.clip([y0, y1], 0, H - 1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            d = self._localise((x0, y0, x1, y1), score, depth_map, cam_pos, R_WC)
            if d['width'] <= self.SPLIT_ABOVE:
                out.append(d)
                continue
            # wide obstacle (wall / building): split into ~1 m image columns ("stixels"), each
            # with its own bearing and learned depth, so the planner sees the true outline
            # (corners, oblique faces) instead of one big circle.
            n = int(np.ceil(d['width'] / self.COLUMN_WIDTH))
            edges = np.linspace(x0, x1, n + 1)
            for k in range(n):
                c = self._localise((edges[k], y0, edges[k + 1], y1), score, depth_map, cam_pos, R_WC)
                c['column_of'] = len(out)
                out.append(c)
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3
        return out, depth_map

    # ------------------------------------------------------------------
    def _ray(self, u, v):
        d = self.K_inv @ np.array([u, v, 1.0])
        return d / np.linalg.norm(d)

    def _localise(self, box, score, depth_map, cam_pos, R_WC):
        cp = self.cp
        x0, y0, x1, y1 = box
        uc, vc = (x0 + x1) / 2, (y0 + y1) / 2
        ray_c = self._ray(uc, vc)
        az = np.arctan2(ray_c[0], ray_c[2])                    # + right
        el = np.arctan2(-ray_c[1], np.hypot(ray_c[0], ray_c[2]))  # + up

        # --- learned depth inside the central part of the box
        bw, bh = x1 - x0, y1 - y0
        ix0, ix1 = int(x0 + 0.25 * bw), int(np.ceil(x1 - 0.25 * bw))
        iy0, iy1 = int(y0 + 0.2 * bh), int(np.ceil(y1 - 0.2 * bh))
        patch = depth_map[iy0:max(iy1, iy0 + 1), ix0:max(ix1, ix0 + 1)]
        z_d = float(np.percentile(patch, 30))
        # confidence = fraction of the box centre whose depth agrees with the estimate
        consistent = float(np.mean(np.abs(patch - z_d) < 0.15 * z_d + 0.2))
        r_depth = z_d / max(ray_c[2], 0.2)

        # --- flat-ground contact geometry (valid for objects standing on the ground)
        r_ground = np.nan
        if y1 < cp.height - 2:
            ray_w = R_WC @ self._ray(uc, y1)
            if ray_w[2] < -1e-3:
                t = -cam_pos[2] / ray_w[2]
                r_ground = float(t * np.linalg.norm(ray_w[:2]) / max(np.hypot(ray_c[0], ray_c[2]), 1e-3))

        # --- known-size heuristic
        r_size = cp.fx * self.NOMINAL_WIDTH / max(bw, 1.0)

        # detector boxes are ~10 % wider than the object (validation set) -> calibrated
        width_m = self.WIDTH_CALIB * bw / cp.fx * r_depth
        conf = consistent
        return dict(box=(x0, y0, x1, y1), score=score, az=float(az), el=float(el),
                    r_depth=float(r_depth), r_ground=r_ground, r_size=float(r_size),
                    range=float(r_depth), width=float(width_m), range_conf=conf,
                    ray_c=ray_c)


class OraclePerception:
    """Ground-truth perception (ablation / upper bound): perfect range & bearing
    for every obstacle inside the camera frustum and range."""

    def __init__(self, cp: CameraParams, world):
        self.cp = cp
        self.world = world
        self.last_latency_ms = 0.0

    def _visible(self, ob, c, cam_pos, R_WC):
        """Centre or nearest surface point inside the image."""
        pts = [c]
        if ob.kind == 'box':
            pts.append(np.clip(cam_pos, ob.pos - ob.size, ob.pos + ob.size))
        else:
            d = cam_pos[:2] - c[:2]
            pts.append(np.array([*(c[:2] + d / max(np.linalg.norm(d), 1e-6) * ob.radius_xy), c[2]]))
        for q in pts:
            pc = R_WC.T @ (q - cam_pos)
            if pc[2] <= 0.2:
                continue
            u = self.cp.fx * pc[0] / pc[2] + self.cp.cx
            v = self.cp.fy * pc[1] / pc[2] + self.cp.cy
            if 0 <= u < self.cp.width and 0 <= v < self.cp.height:
                return True
        return False

    def process(self, rgb, cam_pos, R_WB):
        cp = self.cp
        R_WC = camera_rotation(R_WB, cp.tilt_deg)
        out = []
        for ob in self.world.obstacles:
            c = ob.pos.copy()
            if ob.kind == 'cylinder':
                c[2] = min(cam_pos[2], ob.size[1])
            elif ob.kind == 'box':
                c[2] = np.clip(cam_pos[2], ob.pos[2] - ob.size[2], ob.pos[2] + ob.size[2])
            pc = R_WC.T @ (c - cam_pos)
            rng_c = np.linalg.norm(pc)
            r_surf = rng_c - ob.radius_xy
            if r_surf > 25 or not self._visible(ob, c, cam_pos, R_WC):
                continue
            u = cp.fx * pc[0] / max(pc[2], 0.1) + cp.cx
            v = cp.fy * pc[1] / max(pc[2], 0.1) + cp.cy
            ray_c = pc / rng_c
            az = np.arctan2(ray_c[0], ray_c[2])
            el = np.arctan2(-ray_c[1], np.hypot(ray_c[0], ray_c[2]))
            w_px = cp.fx * 2 * ob.radius_xy / pc[2]
            out.append(dict(box=(u - w_px / 2, v - w_px / 2, u + w_px / 2, v + w_px / 2), score=1.0,
                            az=az, el=el, r_depth=r_surf, r_ground=np.nan, r_size=np.nan,
                            range=r_surf, width=2 * ob.radius_xy, range_conf=1.0, ray_c=ray_c,
                            gt_id=ob.oid))
        return out, None
