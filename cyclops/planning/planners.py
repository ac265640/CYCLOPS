"""Local planners that turn obstacle estimates into velocity + yaw setpoints.

APFPlanner       - artificial potential field with
                     * attractive field to the goal (saturated at cruise speed)
                     * repulsive field from every tracked obstacle, evaluated at
                       its current AND extrapolated future position (moving obstacles)
                     * obstacle radius inflated by the drone radius, the required
                       clearance and the obstacle's 1-sigma position uncertainty
                     * vortex (tangential) field so the UAV slides round obstacles
                       instead of stalling in APF local minima
                   constraints: |v| <= v_max, |dv/dt| <= a_max, |yaw rate| <= r_max,
                   and a stopping-distance rule that caps the approach speed to every
                   obstacle so the minimum clearance can always be held.
                   Yaw follows the velocity so the forward camera looks where it goes.
StraightLinePlanner - the obstacle-unaware baseline from the report (8.5).
"""
import numpy as np

from ..geometry import wrap_angle


def _limit_norm(v, m):
    n = np.linalg.norm(v)
    return v * (m / n) if n > m else v


class StraightLinePlanner:
    name = 'baseline'

    def __init__(self, pp, alt):
        self.pp = pp
        self.alt = alt

    def reset(self):
        pass

    def plan(self, pos, vel, yaw, goal, obstacles, dt):
        d = goal[:2] - pos[:2]
        dist = np.linalg.norm(d)
        speed = min(self.pp.cruise_speed, 0.8 * dist)
        v = np.zeros(3)
        if dist > 1e-3:
            v[:2] = d / dist * speed
        v[2] = 1.2 * (self.alt - pos[2])
        return dict(type='velocity', v=v, yaw=np.arctan2(d[1], d[0])), {}


class APFPlanner:
    name = 'apf'

    def __init__(self, pp, alt):
        self.pp = pp
        self.alt = alt
        self.reset()

    def reset(self):
        self.v_prev = None
        self.yaw_cmd = None
        self.side = {}            # obstacle id -> chosen vortex direction (+1 / -1)

    def plan(self, pos, vel, yaw, goal, obstacles, dt):
        pp = self.pp
        p = pos[:2]
        to_goal = goal[:2] - p
        dist_goal = np.linalg.norm(to_goal)
        g_dir = to_goal / max(dist_goal, 1e-6)

        # ---------------- attractive field (velocity-shaped, saturated)
        speed = min(pp.cruise_speed, pp.k_att * dist_goal)
        v_att = g_dir * speed

        # ---------------- repulsive + vortex fields
        v_rep = np.zeros(2)
        min_clear = np.inf
        debug_obs = []
        heading = vel[:2] / np.linalg.norm(vel[:2]) if np.linalg.norm(vel[:2]) > 0.3 else g_dir
        for ob in obstacles:
            R_eff = ob['radius'] + pp.drone_radius + min(0.5 * ob.get('sigma', 0.0), 0.5)
            worst = None
            for tau in (0.0, 0.5 * pp.predict_horizon, pp.predict_horizon):
                c = ob['pos'][:2] + ob['vel'][:2] * tau
                diff = p - c
                d_c = np.linalg.norm(diff)
                d = d_c - R_eff
                # discount future positions slightly
                d_eff = d + 0.3 * tau
                if worst is None or d_eff < worst[0]:
                    worst = (d_eff, diff / max(d_c, 1e-6), d)
            d_eff, n_away, d_now = worst
            min_clear = min(min_clear, d_now)
            if d_eff >= pp.influence_dist:
                continue
            # ignore obstacles that are behind us and not closing in
            if -n_away @ heading < -0.3 and d_now > pp.min_clearance:
                continue
            d_s = max(d_eff, 0.05)
            mag = pp.k_rep * (1.0 / d_s - 1.0 / pp.influence_dist) / d_s ** 2
            mag = min(mag, 2 * pp.max_speed)
            # vortex direction: pass the obstacle on the side that is closer to the goal line
            if ob['id'] not in self.side:
                cross = g_dir[0] * (-n_away[1]) - g_dir[1] * (-n_away[0])
                self.side[ob['id']] = 1.0 if cross <= 0 else -1.0
            s = self.side[ob['id']]
            tangent = s * np.array([-n_away[1], n_away[0]])
            if tangent @ g_dir < 0:
                tangent = -tangent
            v_rep += mag * n_away + pp.vortex_gain * mag * tangent
            debug_obs.append((ob['id'], d_now, mag))

        v_xy = v_att + v_rep
        v_xy = _limit_norm(v_xy, pp.max_speed)

        # ---------------- acceleration (slew-rate) constraint
        if self.v_prev is not None:
            v_xy = self.v_prev[:2] + _limit_norm(v_xy - self.v_prev[:2], pp.max_accel * dt)

        # ---------------- stopping-distance constraint (applied after the slew limit so safety wins)
        # the approach speed towards every obstacle is capped so the UAV can still brake
        # before violating the clearance:  v_into <= sqrt(2 a_brake (d - d_min))
        # d is shortened by the distance covered during the controller's response lag.
        for ob in obstacles:
            R_eff = ob['radius'] + pp.drone_radius + min(0.5 * ob.get('sigma', 0.0), 0.5)
            diff = ob['pos'][:2] + ob['vel'][:2] * 0.3 - p
            dist = np.linalg.norm(diff)
            n_to = diff / max(dist, 1e-6)
            d = dist - R_eff - max(vel[:2] @ n_to, 0.0) * pp.response_lag
            allowed = np.sqrt(2 * pp.brake_accel * max(d - pp.min_clearance, 0.0)) + ob['vel'][:2] @ n_to
            if d < pp.min_clearance:
                allowed = -0.8 * (pp.min_clearance - d)
            into = v_xy @ n_to
            if into > allowed:
                v_xy = v_xy - (into - allowed) * n_to

        v_xy = _limit_norm(v_xy, pp.max_speed)
        v_cmd = np.array([v_xy[0], v_xy[1], 1.2 * (self.alt - pos[2])])
        self.v_prev = v_cmd.copy()

        # ---------------- yaw: look along the direction of travel, rate limited
        if np.linalg.norm(v_cmd[:2]) > 0.5:
            yaw_des = np.arctan2(v_cmd[1], v_cmd[0])
        else:
            yaw_des = np.arctan2(g_dir[1], g_dir[0])
        if self.yaw_cmd is None:
            self.yaw_cmd = yaw
        step = np.clip(wrap_angle(yaw_des - self.yaw_cmd), -np.radians(pp.max_yaw_rate_deg) * dt,
                       np.radians(pp.max_yaw_rate_deg) * dt)
        self.yaw_cmd = wrap_angle(self.yaw_cmd + step)

        info = dict(v_att=v_att, v_rep=v_rep, min_clear_est=min_clear, active=debug_obs)
        return dict(type='velocity', v=v_cmd, yaw=self.yaw_cmd), info


class APFRolloutPlanner:
    """Sampling-based local planner that scores candidate velocities with the APF
    potential along short predicted rollouts (Dynamic-Window / VFH style).

    For each candidate velocity v_c (headings fanned around the current camera
    axis x speeds) the UAV and every tracked obstacle are rolled forward for
    `horizon` seconds (obstacles with their estimated velocity, i.e. moving
    obstacles are extrapolated). Then
        reject   if the predicted clearance ever drops below min_clearance
        cost  =  w_goal  * remaining distance to goal
               + w_rep   * sum of APF repulsive potential along the rollout
               + w_fov   * heading change outside the camera field of view
               + w_speed * (cruise - |v_c|)
               + w_smooth* |v_c - v_prev|
    The lowest-cost admissible candidate becomes the velocity setpoint; the
    zero-velocity candidate (hover) is always admissible as a fallback, and
    while hovering the UAV yaws to scan for a way round. Constraints on speed,
    acceleration and yaw rate are enforced as in the APF planner.
    Restricting candidates to a cone (fov_cone_deg) well inside the camera's
    field of view means the UAV never
    flies into space it has not looked at - the monocular sensing limitation
    is designed into the planner (the central idea of the reference paper).
    """
    name = 'apf_rollout'

    def __init__(self, pp, alt, hfov_deg=90.0):
        self.pp = pp
        self.alt = alt
        self.half_fov = np.radians(hfov_deg) / 2
        self.reset()

    def reset(self):
        self.v_prev = np.zeros(2)
        self.yaw_cmd = None
        self.blocked_time = 0.0
        self.scan_dir = 1.0

    def _rollout_clearance(self, p, v_now, cands, obstacles, T, n=10, lag=0.35):
        """Return (min clearance, repulsive potential) per candidate."""
        pp = self.pp
        ts = np.linspace(T / n, T, n)
        # first-order velocity response of the closed loop
        k = 1 - np.exp(-ts / lag)
        disp_now = v_now[None, :] * (lag * k)[:, None]                       # (n,2)
        disp_c = (ts - lag * k)[None, :, None] * cands[:, None, :]          # (m,n,2)
        traj = p + disp_now[None] + disp_c                                   # (m,n,2)
        dmin = np.full(len(cands), np.inf)
        slack = np.full(len(cands), np.inf)     # >0 <=> candidate respects every clearance margin
        pot = np.zeros(len(cands))
        for ob in obstacles:
            R_eff = ob['radius'] + pp.drone_radius + min(0.5 * ob.get('sigma', 0.0), 0.5)
            c = ob['pos'][:2] + ob['vel'][:2] * ts[:, None]                  # (n,2)
            d = np.linalg.norm(traj - c[None], axis=-1) - R_eff               # (m,n)
            dmin = np.minimum(dmin, d.min(1))
            # if a (noisy) estimate already puts us inside the margin, require only that the
            # clearance does not shrink further, instead of declaring every motion inadmissible
            d_now = np.linalg.norm(p - ob['pos'][:2]) - R_eff
            margin = min(pp.min_clearance, d_now - 0.05)
            slack = np.minimum(slack, d.min(1) - margin)
            ds = np.maximum(d, 0.05)
            rep = np.where(ds < pp.influence_dist, 0.5 * (1 / ds - 1 / pp.influence_dist) ** 2, 0.0)
            pot += rep.mean(1)
        return dmin, slack, pot, traj

    def plan(self, pos, vel, yaw, goal, obstacles, dt):
        pp = self.pp
        p = pos[:2]
        to_goal = goal[:2] - p
        dist_goal = np.linalg.norm(to_goal)
        goal_ang = np.arctan2(to_goal[1], to_goal[0])
        cruise = min(pp.cruise_speed, max(pp.k_att * dist_goal, 0.3))

        # candidate set: headings inside the camera FOV around current yaw (+ goal bearing if visible)
        cone = np.radians(pp.fov_cone_deg)
        rel = np.linspace(-cone, cone, 13)
        heads = yaw + rel
        g_rel = wrap_angle(goal_ang - yaw)
        if abs(g_rel) < cone:
            heads = np.append(heads, goal_ang)
        speeds = cruise * np.array([0.35, 0.65, 1.0])
        H, S = np.meshgrid(heads, speeds)
        cands = np.stack([S.ravel() * np.cos(H.ravel()), S.ravel() * np.sin(H.ravel())], 1)
        cands = np.vstack([cands, np.zeros((1, 2))])                         # hover fallback

        T = pp.predict_horizon + 0.5
        v_now = vel[:2]
        if obstacles:
            dmin, slack, pot, traj = self._rollout_clearance(p, v_now, cands, obstacles, T)
        else:
            dmin = np.full(len(cands), np.inf)
            slack = dmin
            pot = np.zeros(len(cands))
            traj = p + cands[:, None, :] * np.linspace(T / 10, T, 10)[None, :, None]
        end = traj[:, -1]
        d_goal = np.linalg.norm(goal[:2] - end, axis=1)
        head_c = np.arctan2(cands[:, 1], cands[:, 0])
        off_axis = np.abs(wrap_angle(head_c - yaw))
        sp_c = np.linalg.norm(cands, axis=1)
        cost = (1.0 * d_goal
                + pp.k_rep * 0.5 * pot
                + 1.5 * np.maximum(off_axis - np.radians(20), 0) * (sp_c > 0)
                + 1.0 * (cruise - sp_c)
                + 0.3 * np.linalg.norm(cands - self.v_prev, axis=1))
        # admissibility: predicted clearance never below the minimum (stopping distance built in)
        ok = slack > 0
        ok[-1] = True
        cost[~ok] = np.inf
        best = int(np.argmin(cost))
        v_xy = cands[best]
        blocked = best == len(cands) - 1 and dist_goal > pp.goal_tolerance

        # evasive manoeuvre: hovering is not safe if a tracked MOVING obstacle is closing in.
        # Then - and only then - consider every direction, scored against the known obstacles.
        movers = [o for o in obstacles if np.linalg.norm(o['vel'][:2]) > 0.4]
        if blocked and movers and dmin[-1] <= pp.min_clearance:
            ang = np.linspace(-np.pi, np.pi, 24, endpoint=False)
            esc = np.concatenate([s * np.stack([np.cos(ang), np.sin(ang)], 1) for s in (1.0, 2.0)])
            e_dmin, _, _, _ = self._rollout_clearance(p, v_now, esc, obstacles, T)
            if e_dmin.max() > dmin[-1]:
                v_xy = esc[int(np.argmax(e_dmin))]

        # if every candidate leaves too little room: brake and hover (the space behind is
        # unobserved, so only back away when contact is imminent)
        if blocked and obstacles:
            near = min(obstacles, key=lambda o: np.linalg.norm(o['pos'][:2] - p) - o['radius'])
            diff = p - near['pos'][:2]
            dn = np.linalg.norm(diff) - near['radius'] - pp.drone_radius
            if dn < 0.2:
                v_xy = 0.5 * diff / max(np.linalg.norm(diff), 1e-6)

        # slew-rate (acceleration) and speed constraints
        v_xy = self.v_prev + _limit_norm(v_xy - self.v_prev, pp.max_accel * dt)
        v_xy = _limit_norm(v_xy, pp.max_speed)
        self.v_prev = v_xy.copy()

        # yaw: look along the chosen motion; when blocked, scan towards the goal side
        if self.yaw_cmd is None:
            self.yaw_cmd = yaw
        if blocked:
            self.blocked_time += dt
            if self.blocked_time > 3.0:
                self.scan_dir *= -1
                self.blocked_time = 0.0
            yaw_des = self.yaw_cmd + self.scan_dir * np.radians(60)
        else:
            self.blocked_time = 0.0
            self.scan_dir = 1.0 if g_rel >= 0 else -1.0
            if best < len(cands) - 1:
                h_rel = wrap_angle(head_c[best] - yaw)
                # yaw leads the motion: if the chosen heading is at the edge of the cone keep turning
                yaw_des = head_c[best] + (np.sign(h_rel) * cone if abs(h_rel) > 0.75 * cone else 0.0)
                # bias the camera towards the goal when the goal is outside the cone of candidates
                if abs(g_rel) > cone and abs(wrap_angle(goal_ang - head_c[best])) > cone:
                    yaw_des = head_c[best] + np.sign(wrap_angle(goal_ang - head_c[best])) * cone
            else:
                yaw_des = goal_ang
        step = np.clip(wrap_angle(yaw_des - self.yaw_cmd), -np.radians(pp.max_yaw_rate_deg) * dt,
                       np.radians(pp.max_yaw_rate_deg) * dt)
        self.yaw_cmd = wrap_angle(self.yaw_cmd + step)

        v_cmd = np.array([v_xy[0], v_xy[1], 1.2 * (self.alt - pos[2])])
        info = dict(cands=cands, ok=ok, best=best, blocked=blocked, traj=traj[best],
                    min_clear_est=float(dmin[best]) if np.isfinite(dmin[best]) else np.inf)
        return dict(type='velocity', v=v_cmd, yaw=self.yaw_cmd), info
