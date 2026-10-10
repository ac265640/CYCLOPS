"""Multi-obstacle tracker: world-frame EKF per obstacle with range/bearing updates.

State per track:  x = [px, py, pz, vx, vy, vz]   (constant-velocity model)

Measurement (camera frame, from the perception node):
    z = [azimuth, elevation, range-to-centre]

Localisation modes (after Zhang, Zhang & Cao, ICUAS 2020)
    MODE 1  range + bearing   : used when the learned range is reliable
    MODE 2  bearing only      : range row dropped; the position becomes
                                observable only through UAV motion
                                (parallax), so the tracker reports an
                                observability index = accumulated line-of-sight
                                rotation and the position covariance.
The planner inflates every obstacle by its 1-sigma position uncertainty,
i.e. the avoidance is planned around what the camera actually knows.

Moving obstacles: the velocity part of the state lets the planner extrapolate
each obstacle `predict_horizon` seconds ahead.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import camera_rotation, wrap_angle


def h_meas(p, cam_pos, R_WC):
    pc = R_WC.T @ (p - cam_pos)
    return np.array([np.arctan2(pc[0], pc[2]), np.arctan2(-pc[1], np.hypot(pc[0], pc[2])), np.linalg.norm(pc)])


def inv_meas(az, el, r, cam_pos, R_WC):
    ce = np.cos(el)
    pc = r * np.array([ce * np.sin(az), -np.sin(el), ce * np.cos(az)])
    return cam_pos + R_WC @ pc


class Track:
    _next_id = 0

    def __init__(self, det, cam_pos, R_WC, t, mode):
        self.id = Track._next_id
        Track._next_id += 1
        r_c = det['range'] + det['width'] / 2
        p = inv_meas(det['az'], det['el'], r_c, cam_pos, R_WC)
        self.x = np.concatenate([p, np.zeros(3)])
        sr = 0.15 * r_c + 0.3 if mode == 1 else 0.5 * r_c
        los = R_WC @ det['ray_c']
        # initial covariance: tight across the line of sight, loose along it
        Pp = np.outer(los, los) * sr ** 2 + (np.eye(3) - np.outer(los, los)) * (0.03 * r_c + 0.1) ** 2
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = Pp
        self.P[3:, 3:] = np.eye(3) * 1.0 ** 2
        self.widths = [] if det.get('truncated') else [det['width']]
        self.radius = max(det['width'] / 2, 0.15)
        self.hits = 1
        self.misses = 0
        self.t_last = t
        self.t_first = t
        self.confirmed = False
        self.mode = mode
        self.los0 = los
        self.parallax = 0.0
        self.last_det = det

    @property
    def pos(self):
        return self.x[:3]

    @property
    def vel(self):
        return self.x[3:]

    @property
    def sigma(self):
        """1-sigma horizontal position uncertainty (largest axis)."""
        return float(np.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))

    def predict(self, dt, q_acc, t):
        # velocity is mean-reverting (most obstacles are static): slow decay while the
        # obstacle is observed, fast decay once it has left the field of view, so an
        # unverified velocity is never extrapolated for long.
        tau = 4.0 if t - self.t_last < 0.3 else 0.6
        a = np.exp(-dt / tau)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        F[3:, 3:] = np.eye(3) * a
        F[5, 5] = a * 0.5           # obstacles rarely move vertically
        G = np.vstack([np.eye(3) * 0.5 * dt * dt, np.eye(3) * dt])
        Q = G @ G.T * q_acc ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q


class ObstacleTracker:
    MAX_RANGE = 20.0      # detections further away are irrelevant for a 5 m influence radius

    def __init__(self, planner_params, cam_params, q_acc=0.6, sigma_bearing=0.03, gate=16.0):
        self.pp = planner_params
        self.cp = cam_params
        self.q_acc = q_acc
        self.sb = sigma_bearing
        self.gate = gate
        self.tracks = []
        self.t = None

    def reset(self):
        self.tracks = []
        self.t = None
        Track._next_id = 0

    def _R(self, det, mode):
        r = det['range']
        sr = 0.12 * r + 0.15 + 0.3 * (1 - det['range_conf']) * r
        sb = self.sb * (3.0 if det.get('truncated') else 1.0)   # box cut by the image edge
        if mode == 2:
            return np.diag([sb ** 2, sb ** 2])
        return np.diag([sb ** 2, sb ** 2, sr ** 2])

    @staticmethod
    def _mode(det):
        return 1 if det['range_conf'] > 0.4 and det['range'] < 18 else 2

    def _jac(self, x, cam_pos, R_WC, m):
        H = np.zeros((m, 6))
        z0 = h_meas(x[:3], cam_pos, R_WC)[:m]
        eps = 1e-4
        for i in range(3):
            dx = np.zeros(3)
            dx[i] = eps
            H[:, i] = (h_meas(x[:3] + dx, cam_pos, R_WC)[:m] - z0) / eps
        return z0, H

    def _innov(self, tr, det, cam_pos, R_WC, mode):
        m = 3 if mode == 1 else 2
        z_pred, H = self._jac(tr.x, cam_pos, R_WC, m)
        z = np.array([det['az'], det['el'], det['range'] + tr.radius])[:m]
        y = z - z_pred
        y[:2] = wrap_angle(y[:2])
        S = H @ tr.P @ H.T + self._R(det, mode)
        return y, H, S

    def update(self, t, dets, cam_pos, R_WB):
        R_WC = camera_rotation(R_WB, self.cp.tilt_deg)
        W, H = self.cp.width, self.cp.height
        for d in dets:
            x0, y0, x1, y1 = d['box']
            d['truncated'] = x0 < 2 or x1 > W - 3
        dets = [d for d in dets if d['range'] < self.MAX_RANGE]
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        for tr in self.tracks:
            tr.predict(dt, self.q_acc, t)

        # ---- data association (Hungarian on Mahalanobis distance)
        nT, nD = len(self.tracks), len(dets)
        cost = np.full((nT, nD), 1e6)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(dets):
                y, _, S = self._innov(tr, d, cam_pos, R_WC, 1)
                md = float(y @ np.linalg.solve(S, y))
                if md < self.gate:
                    cost[i, j] = md
                else:
                    # Euclidean fallback guards against an over-confident covariance
                    p_det = inv_meas(d['az'], d['el'], d['range'] + tr.radius, cam_pos, R_WC)
                    e = np.linalg.norm(p_det[:2] - tr.pos[:2])
                    if e < 0.5 + 0.1 * d['range']:
                        cost[i, j] = self.gate * 0.99
        matched_t, matched_d = set(), set()
        if nT and nD:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < self.gate:
                    self._correct(self.tracks[i], dets[j], cam_pos, R_WC, t)
                    matched_t.add(i)
                    matched_d.add(j)

        # ---- unmatched tracks: miss bookkeeping
        keep = []
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                if self._should_see(tr, cam_pos, R_WC):
                    tr.misses += 1
                age_out = t - tr.t_last > self.pp.track_memory
                if (not tr.confirmed and tr.misses > 2) or tr.misses > 8 or age_out:
                    continue
            keep.append(tr)
        self.tracks = keep

        # ---- new tracks
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(d, cam_pos, R_WC, t, self._mode(d)))
        self._merge_duplicates()
        return self.tracks

    def _merge_duplicates(self):
        """Two tracks closer than their radii describe the same obstacle: keep the better one."""
        self.tracks.sort(key=lambda tr: -tr.hits)
        keep = []
        for tr in self.tracks:
            dup = False
            for k in keep:
                if np.linalg.norm(tr.pos[:2] - k.pos[:2]) < max(tr.radius, k.radius) + 0.3:
                    dup = True
                    k.confirmed = k.confirmed or tr.confirmed
                    break
            if not dup:
                keep.append(tr)
        self.tracks = keep

    def _correct(self, tr, det, cam_pos, R_WC, t):
        mode = self._mode(det)
        y, H, S = self._innov(tr, det, cam_pos, R_WC, mode)
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ y
        tr.P = (np.eye(6) - K @ H) @ tr.P
        tr.P = 0.5 * (tr.P + tr.P.T)
        if not det.get('truncated'):
            tr.widths = (tr.widths + [det['width']])[-10:]
            tr.radius = max(float(np.median(tr.widths)) / 2, 0.15)
        tr.hits += 1
        tr.misses = 0
        tr.t_last = t
        tr.mode = mode
        tr.last_det = det
        los = R_WC @ det['ray_c']
        tr.parallax = float(np.degrees(np.arccos(np.clip(los @ tr.los0, -1, 1))))
        if tr.hits >= 3:
            tr.confirmed = True

    def _should_see(self, tr, cam_pos, R_WC):
        pc = R_WC.T @ (tr.pos - cam_pos)
        if pc[2] < 0.5 or np.linalg.norm(pc) > 20:
            return False
        u = self.cp.fx * pc[0] / pc[2] + self.cp.cx
        v = self.cp.fy * pc[1] / pc[2] + self.cp.cy
        m = 8
        return m <= u < self.cp.width - m and m <= v < self.cp.height - m

    def obstacles(self):
        """Confirmed tracks in the format the planner consumes."""
        return [dict(id=tr.id, pos=tr.pos.copy(), vel=tr.vel.copy(), radius=tr.radius, sigma=tr.sigma,
                     mode=tr.mode) for tr in self.tracks if tr.confirmed]
