"""CYCLOPS live demo: real-time visualisation of the full closed loop.

Panels
    FPV        onboard camera + detections + FPV-goggle OSD (horizon, flight-path
               marker, mode, battery, altitude, speed)        [Module 2 FPV / telemetry]
    DEPTH      MonoDepthNet output
    MAP        top-down tactical map: true obstacles, tracked estimates (with
               uncertainty), camera FOV, planner rollouts, and a ghost
               baseline drone flying the same mission without avoidance
    TELEMETRY  battery, motors (ESC), attitude, controller, perception metrics
    LOG        commander events (preflight, arming, mode changes, failsafes)

Keys
    space  pause / resume            1-4  pick scenario        s  new random seed
    r      restart                   v    vision <-> oracle    b  toggle baseline ghost
    m      MISSION <-> POSCTL (manual RC: arrows = pitch/roll, q/e = yaw, z/x = down/up)
    h      return to launch (RTL)    l    land                 +/-  sim speed
"""
import copy
import time

import matplotlib
import numpy as np

from ..config import Config
from ..control.commander import RCInput
from ..geometry import rot_to_euler, camera_rotation
from ..sim.simulator import Simulation
from ..world.world import make_world, SCENARIOS

BG = '#0e1117'
PANEL = '#161b22'
FG = '#e6edf3'
DIM = '#8b949e'
ACC = '#3fb950'
WARN = '#d29922'
BAD = '#f85149'
CYAN = '#39c5cf'

SCENARIO_TITLES = {
    'forest_sar': 'Search & rescue - forest corridor',
    'urban_inspection': 'Infrastructure inspection - urban',
    'dynamic': 'Dynamic obstacles crossing the path',
    'pillars': 'Dense pillar slalom',
}


def range_color(r):
    if r < 3:
        return BAD
    if r < 7:
        return WARN
    return ACC


class LiveDemo:
    def __init__(self, scenario='forest_sar', seed=11, method='vision', perception=None,
                 ghost=True, speed=1.0, record=None, duration=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.cfg = Config()
        self.scenario = scenario
        self.seed = seed
        self.method = method
        self.perception = perception
        self.ghost_on = ghost
        self.speed = speed
        self.paused = False
        self.record = record
        self.duration = duration
        self.rc = RCInput()
        self.keys_down = set()
        self._build_figure()
        self.load()

    # =================================================================== sim set-up
    def load(self):
        self.world = make_world(self.scenario, seed=self.seed)
        cfg = Config()
        cfg.sim.max_time = 90.0
        self.cfg = cfg
        method = self.method if (self.method != 'vision' or self.perception is not None) else 'oracle'
        self.sim = Simulation(self.world, method, cfg=cfg, perception=self.perception, seed=self.seed,
                              render_always=True)
        self.sim.rc = self.rc
        self.ghost = Simulation(copy.deepcopy(self.world), 'baseline', cfg=Config(), seed=self.seed)
        ok = self.sim.arm_and_takeoff()
        self.ghost.arm_and_takeoff()
        self.preflight = self.sim.last_checks
        self.trail = []
        self.ghost_trail = []
        self.clear_hist = []
        self.t_wall0 = time.time()
        self.sim_t0 = 0.0
        self._draw_static_map()
        self._fps = 0.0
        self._last_frame = time.time()
        if not ok:
            self.sim.cmd.log(0, 'preflight failed - not armed')

    # =================================================================== figure
    def _build_figure(self):
        plt = self.plt
        plt.rcParams.update({'font.family': 'monospace', 'font.size': 9, 'text.color': FG,
                             'axes.labelcolor': DIM, 'xtick.color': DIM, 'ytick.color': DIM,
                             'axes.edgecolor': '#30363d'})
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title('CYCLOPS - monocular obstacle avoidance')
        except Exception:
            pass
        gs = self.fig.add_gridspec(3, 4, left=0.03, right=0.985, top=0.895, bottom=0.04,
                                   wspace=0.12, hspace=0.22, height_ratios=[1, 1, 0.78])
        self.ax_cam = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_map = self.fig.add_subplot(gs[0:2, 2:4])
        self.ax_depth = self.fig.add_subplot(gs[2, 0])
        self.ax_tel = self.fig.add_subplot(gs[2, 1])
        self.ax_met = self.fig.add_subplot(gs[2, 2])
        self.ax_log = self.fig.add_subplot(gs[2, 3])
        for ax in (self.ax_cam, self.ax_map, self.ax_depth, self.ax_tel, self.ax_met, self.ax_log):
            ax.set_facecolor(PANEL)
        self.title = self.fig.text(0.03, 0.962, '', fontsize=15, weight='bold', color=FG)
        self.subtitle = self.fig.text(0.03, 0.935, '', fontsize=9.5, color=DIM)
        self.fig.text(0.985, 0.965, 'space pause | 1-4 scenario | s seed | r restart | v vision/oracle | '
                      'b ghost | m manual | h RTL | +/- speed', fontsize=8, color=DIM, ha='right')

        # camera
        cp = self.cfg.camera
        self.cam_img = self.ax_cam.imshow(np.zeros((cp.height, cp.width, 3), np.uint8),
                                          extent=(0, cp.width, cp.height, 0), interpolation='bilinear')
        self.ax_cam.set_xlim(0, cp.width)
        self.ax_cam.set_ylim(cp.height, 0)
        self.ax_cam.set_xticks([])
        self.ax_cam.set_yticks([])
        self.ax_cam.set_title('FPV camera  |  CNN detections + monocular range', loc='left', color=FG, fontsize=10)
        self.horizon, = self.ax_cam.plot([], [], color='white', lw=1.0, alpha=0.8)
        self.fpm, = self.ax_cam.plot([], [], marker='o', ms=9, mfc='none', mec=ACC, mew=1.6)
        cx, cy = cp.cx, cp.cy
        self.ax_cam.plot([cx - 9, cx - 3], [cy, cy], color='white', lw=1.4)
        self.ax_cam.plot([cx + 3, cx + 9], [cy, cy], color='white', lw=1.4)
        self.ax_cam.plot([cx], [cy], marker='+', color='white', ms=5)
        self.osd_tl = self.ax_cam.text(3, 6, '', fontsize=10, color='white', va='top', weight='bold')
        self.osd_tr = self.ax_cam.text(cp.width - 3, 6, '', fontsize=10, color='white', va='top', ha='right', weight='bold')
        self.osd_bl = self.ax_cam.text(3, cp.height - 3, '', fontsize=10, color='white', va='bottom', weight='bold')
        self.osd_br = self.ax_cam.text(cp.width - 3, cp.height - 3, '', fontsize=10, color='white', va='bottom', ha='right', weight='bold')
        self.osd_center = self.ax_cam.text(cx, cy + 22, '', fontsize=16, color=BAD, ha='center', weight='bold')
        self.det_artists = []

        # depth
        self.depth_img = self.ax_depth.imshow(np.zeros((cp.height, cp.width)), cmap='magma_r',
                                              vmin=np.log(0.5), vmax=np.log(40))
        self.ax_depth.set_xticks([])
        self.ax_depth.set_yticks([])
        self.ax_depth.set_title('MonoDepthNet (log depth)', loc='left', color=FG, fontsize=9)

        # map
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title('Tactical map  |  truth (grey) vs camera estimate (cyan)', loc='left', color=FG, fontsize=10)
        self.ax_map.grid(color='#30363d', lw=0.5)
        self.trail_line, = self.ax_map.plot([], [], color=ACC, lw=2.0, label='CYCLOPS')
        self.ghost_line, = self.ax_map.plot([], [], color=BAD, lw=1.4, ls='--', label='baseline (no avoidance)')
        self.ghost_dot, = self.ax_map.plot([], [], marker='o', color=BAD, ms=6)
        self.ghost_x, = self.ax_map.plot([], [], marker='X', color=BAD, ms=14, ls='none')
        self.drone_dot, = self.ax_map.plot([], [], marker=(3, 0, 0), color=FG, ms=12)
        self.fov_poly = matplotlib.patches.Polygon(np.zeros((3, 2)), closed=True, fc=FG, alpha=0.08, ec=FG, lw=0.6)
        self.ax_map.add_patch(self.fov_poly)
        self.roll_lines = matplotlib.collections.LineCollection([], lw=0.7)
        self.ax_map.add_collection(self.roll_lines)
        self.best_line, = self.ax_map.plot([], [], color='#f0f6fc', lw=2.2, alpha=0.9)
        self.track_coll = None
        self.mover_patches = []
        self.ax_map.legend(loc='lower right', facecolor=PANEL, edgecolor='#30363d', labelcolor=FG, fontsize=8)

        for ax in (self.ax_tel, self.ax_met, self.ax_log):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.ax_tel.set_title('Telemetry', loc='left', color=FG, fontsize=9)
        self.ax_met.set_title('Mission metrics', loc='left', color=FG, fontsize=9)
        self.ax_log.set_title('Commander log', loc='left', color=FG, fontsize=9)
        self.tel_text = self.ax_tel.text(0.03, 0.95, '', va='top', fontsize=8.5, color=FG)
        self.motor_bars = self.ax_tel.barh([0.30, 0.22, 0.14, 0.06], [0, 0, 0, 0], height=0.06, left=0.62, color=CYAN)
        for k, y in enumerate([0.30, 0.22, 0.14, 0.06]):
            self.ax_tel.text(0.55, y, f'M{k + 1}', va='center', fontsize=7.5, color=DIM)
        self.ax_tel.text(0.55, 0.40, 'ESC / rotor speed', fontsize=7.5, color=DIM)
        self.met_text = self.ax_met.text(0.03, 0.95, '', va='top', fontsize=8.5, color=FG)
        self.log_text = self.ax_log.text(0.03, 0.95, '', va='top', fontsize=8, color=FG)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)

    def _draw_static_map(self):
        ax = self.ax_map
        for p in getattr(self, '_static_patches', []):
            p.remove()
        self._static_patches = []
        self.mover_patches = []
        for ob in self.world.obstacles:
            if ob.kind == 'box':
                p = matplotlib.patches.Rectangle(ob.pos[:2] - ob.size[:2], 2 * ob.size[0], 2 * ob.size[1],
                                                 fc='#484f58', ec='#6e7681', lw=0.8)
            else:
                p = matplotlib.patches.Circle(ob.pos[:2], ob.radius_xy, fc='#6e7681' if not ob.moving else '#a5555a',
                                              ec='#8b949e', lw=0.6)
            ax.add_patch(p)
            self._static_patches.append(p)
            if ob.moving:
                self.mover_patches.append((ob, p))
        s, g = self.world.start, self.world.goal
        self._static_patches += ax.plot([s[0], g[0]], [s[1], g[1]], color=DIM, lw=0.8, ls=':')
        self._static_patches += ax.plot([g[0]], [g[1]], marker='*', color='#f2cc60', ms=18)
        self._static_patches += ax.plot([s[0]], [s[1]], marker='s', color=DIM, ms=7)
        ax.set_xlim(-3, g[0] + 4)
        ax.set_ylim(-11, 11)
        t = SCENARIO_TITLES.get(self.scenario, self.scenario)
        self.title.set_text(f'CYCLOPS  -  {t}')
        mode = 'deep monocular vision' if self.sim.method == 'vision' else 'ground-truth oracle'
        self.subtitle.set_text(f'perception: {mode}  |  planner: APF-rollout (FOV-aware)  |  seed {self.seed}  |  '
                               f'Camera-only Yaw-aware Collision avoidance with Learned Obstacle Perception for Small UAVs')

    # =================================================================== input
    def _on_key(self, ev):
        k = ev.key
        self.keys_down.add(k)
        if k == ' ':
            self.paused = not self.paused
        elif k in ('1', '2', '3', '4'):
            self.scenario = SCENARIOS[int(k) - 1]
            self.load()
        elif k == 's':
            self.seed += 1
            self.load()
        elif k == 'r':
            self.load()
        elif k == 'v':
            self.method = 'oracle' if self.method == 'vision' else 'vision'
            self.load()
        elif k == 'b':
            self.ghost_on = not self.ghost_on
        elif k == 'm':
            cmd = self.sim.cmd
            cmd.set_mode(self.sim.t, 'POSCTL' if cmd.mode == 'MISSION' else 'MISSION')
        elif k == 'h':
            self.sim.cmd.set_mode(self.sim.t, 'RTL')
        elif k == 'l':
            self.sim.cmd.set_mode(self.sim.t, 'LAND')
        elif k in ('+', '='):
            self.speed = min(self.speed * 1.5, 8)
        elif k == '-':
            self.speed = max(self.speed / 1.5, 0.25)
        self._update_rc()

    def _on_key_release(self, ev):
        self.keys_down.discard(ev.key)
        self._update_rc()

    def _update_rc(self):
        kd = self.keys_down
        self.rc.pitch = (1.0 if 'up' in kd else 0.0) - (1.0 if 'down' in kd else 0.0)
        self.rc.roll = (1.0 if 'right' in kd else 0.0) - (1.0 if 'left' in kd else 0.0)
        self.rc.yaw = (1.0 if 'e' in kd else 0.0) - (1.0 if 'q' in kd else 0.0)
        self.rc.throttle = 0.5 + (0.4 if 'x' in kd else 0.0) - (0.4 if 'z' in kd else 0.0)

    # =================================================================== loop
    def advance(self, sim_dt):
        n = max(1, int(round(sim_dt / self.cfg.sim.dt)))
        for _ in range(n):
            if self.sim.result.outcome == 'running' or self.sim.cmd.armed:
                self.sim.step()
            if self.ghost_on and self.ghost.result.outcome == 'running':
                self.ghost.step()
        self.trail.append(self.sim.quad.p[:2].copy())
        self.ghost_trail.append(self.ghost.quad.p[:2].copy())
        d, _ = self.world.min_distance(self.sim.quad.p, self.cfg.planner.drone_radius)
        self.clear_hist.append(d)

    def frame(self, _=None):
        if not self.paused:
            self.advance(self.speed / self.cfg.camera.rate)
        self.draw()
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_frame, 1e-3)
        self._last_frame = now
        return []

    # =================================================================== drawing
    def draw(self):
        sim = self.sim
        cfg = self.cfg
        cp = cfg.camera
        quad = sim.quad
        roll, pitch, yaw = quad.euler
        if sim.frame is not None:
            self.cam_img.set_data(sim.frame[0])
        if sim.depth_pred is not None:
            self.depth_img.set_data(np.log(np.clip(sim.depth_pred, 0.3, 40)))
        elif sim.frame is not None:
            self.depth_img.set_data(np.log(np.clip(sim.frame[1], 0.3, 40)))

        # ---- detections
        for a in self.det_artists:
            a.remove()
        self.det_artists = []
        for d in sim.dets:
            x0, y0, x1, y1 = d['box']
            col = range_color(d['range'])
            r = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.8)
            self.ax_cam.add_patch(r)
            lbl = self.ax_cam.text(x0, max(y0 - 1, 4), f"{d['range']:.1f}m {np.degrees(d['az']):+.0f}°",
                                   fontsize=7.5, color=col, weight='bold', va='bottom',
                                   bbox=dict(fc='black', alpha=0.45, pad=0.5, lw=0))
            self.det_artists += [r, lbl]

        # ---- OSD: artificial horizon + flight path marker (true camera projection)
        R_WC = camera_rotation(quad.R, cp.tilt_deg)
        pts = []
        for a in np.linspace(-np.pi / 3, np.pi / 3, 9):
            dw = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            pc = R_WC.T @ dw
            if pc[2] > 0.05:
                pts.append((cp.fx * pc[0] / pc[2] + cp.cx, cp.fy * pc[1] / pc[2] + cp.cy))
        if pts:
            pts = np.array(pts)
            self.horizon.set_data(pts[:, 0], pts[:, 1])
        v = quad.v
        if np.linalg.norm(v) > 0.5:
            pc = R_WC.T @ v
            if pc[2] > 0.1:
                self.fpm.set_data([cp.fx * pc[0] / pc[2] + cp.cx], [cp.fy * pc[1] / pc[2] + cp.cy])
        else:
            self.fpm.set_data([], [])
        b = quad.battery
        mode = sim.cmd.mode if sim.cmd.armed else 'DISARMED'
        self.osd_tl.set_text(f'{mode}\n{"ARMED" if sim.cmd.armed else "SAFE"}')
        bcol = BAD if b.low else ('white' if b.soc > 0.3 else WARN)
        self.osd_tr.set_text(f'{b.voltage:4.1f}V {100 * b.soc:3.0f}%\n{b.current:4.1f}A')
        self.osd_tr.set_color(bcol)
        self.osd_bl.set_text(f'ALT {quad.p[2]:4.1f}m\nSPD {np.linalg.norm(quad.v[:2]):4.1f}m/s')
        self.osd_br.set_text(f'HDG {np.degrees(yaw) % 360:3.0f}°\nT+{sim.t:5.1f}s')
        out = sim.result.outcome
        self.osd_center.set_text({'collision': 'COLLISION', 'success': 'GOAL REACHED', 'timeout': 'TIMEOUT'}.get(out, ''))
        self.osd_center.set_color(BAD if out in ('collision', 'timeout') else ACC)

        # ---- map
        tr = np.array(self.trail)
        if len(tr):
            self.trail_line.set_data(tr[:, 0], tr[:, 1])
        if self.ghost_on:
            gt = np.array(self.ghost_trail)
            self.ghost_line.set_data(gt[:, 0], gt[:, 1])
            self.ghost_dot.set_data([self.ghost.quad.p[0]], [self.ghost.quad.p[1]])
            if self.ghost.result.outcome == 'collision':
                self.ghost_x.set_data([self.ghost.quad.p[0]], [self.ghost.quad.p[1]])
            else:
                self.ghost_x.set_data([], [])
        else:
            for ln in (self.ghost_line, self.ghost_dot, self.ghost_x):
                ln.set_data([], [])
        self.drone_dot.set_data([quad.p[0]], [quad.p[1]])
        self.drone_dot.set_marker((3, 0, np.degrees(yaw) - 90))
        hf = np.radians(cp.hfov_deg) / 2
        L = 12.0
        self.fov_poly.set_xy(np.array([quad.p[:2],
                                       quad.p[:2] + L * np.array([np.cos(yaw + hf), np.sin(yaw + hf)]),
                                       quad.p[:2] + L * np.array([np.cos(yaw - hf), np.sin(yaw - hf)])]))
        for ob, patch in self.mover_patches:
            patch.center = ob.pos[:2]
        # tracked obstacles
        if self.track_coll is not None:
            self.track_coll.remove()
        circles = []
        cols = []
        for o in sim.tracker.obstacles():
            circles.append(matplotlib.patches.Circle(o['pos'][:2], o['radius']))
            circles.append(matplotlib.patches.Circle(o['pos'][:2], o['radius'] + min(0.5 * o['sigma'], 0.5)
                                                     + cfg.planner.drone_radius + cfg.planner.min_clearance))
            cols += [CYAN, 'none']
        self.track_coll = matplotlib.collections.PatchCollection(
            circles, facecolors=[(0.22, 0.77, 0.81, 0.35) if c == CYAN else (0, 0, 0, 0) for c in cols],
            edgecolors=[CYAN if c == CYAN else (0.22, 0.77, 0.81, 0.45) for c in cols],
            linestyles=['-' if c == CYAN else '--' for c in cols], linewidths=0.9)
        self.ax_map.add_collection(self.track_coll)
        # planner rollouts
        info = sim.plan_info or {}
        if 'cands' in info and sim.cmd.mode == 'MISSION':
            p = quad.p[:2]
            segs = [np.array([p, p + c * 1.2]) for c in info['cands'][:-1]]
            colors = [(0.25, 0.73, 0.31, 0.55) if ok else (0.97, 0.32, 0.29, 0.45) for ok in info['ok'][:-1]]
            self.roll_lines.set_segments(segs)
            self.roll_lines.set_color(colors)
            tj = info['traj']
            self.best_line.set_data(tj[:, 0], tj[:, 1])
        else:
            self.roll_lines.set_segments([])
            self.best_line.set_data([], [])

        # ---- telemetry
        tel = (f'battery  {b.voltage:5.2f} V  {b.cell_voltage:4.2f} V/cell\n'
               f'         {100 * b.soc:5.1f} %   {b.current:5.1f} A\n'
               f'energy   {b.energy_used_wh:5.2f} Wh\n'
               f'att  R {np.degrees(roll):+5.1f}  P {np.degrees(pitch):+5.1f}\n'
               f'     Y {np.degrees(yaw) % 360:5.1f} deg\n'
               f'pos  {quad.p[0]:5.1f} {quad.p[1]:+5.1f} {quad.p[2]:4.1f} m')
        self.tel_text.set_text(tel)
        wmax = cfg.airframe.motor_w_max
        for bar, w in zip(self.motor_bars, quad.rotor):
            bar.set_width(0.36 * w / wmax)
            bar.set_color(BAD if w > 0.95 * quad.w_max_now() else CYAN)

        ds = sim.det_stats
        tp, fp, fn = ds['tp'], ds['fp'], ds['fn']
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        clr = min(self.clear_hist) if self.clear_hist else np.inf
        cur = self.clear_hist[-1] if self.clear_hist else np.inf
        obs = sim.tracker.obstacles()
        n_m1 = sum(1 for o in obs if o['mode'] == 1)
        g = self.ghost.result.outcome
        met = (f'clearance  now {cur:5.2f} m\n'
               f'           min {sim.min_clearance if np.isfinite(sim.min_clearance) else clr:5.2f} m\n'
               f'tracks     {len(obs):3d}  (Mode1 {n_m1} / Mode2 {len(obs) - n_m1})\n'
               f'detector   P {prec:4.2f}  R {rec:4.2f}\n'
               f'latency    {getattr(sim, "latency_ms", 0):5.1f} ms (perc+plan)\n'
               f'goal dist  {np.linalg.norm(self.world.goal[:2] - quad.p[:2]):5.1f} m\n'
               f'baseline   {g.upper() if g != "running" else "flying"}\n'
               f'sim x{self.speed:.2f}   gui {self._fps:4.1f} fps')
        self.met_text.set_text(met)
        ev = sim.cmd.events[-7:]
        lines = []
        if sim.t < 4.0:
            lines = ['PREFLIGHT CHECKS'] + [f"{'[OK]' if c[1] else '[!!]'} {c[0]}" for c in self.preflight[:7]]
        else:
            lines = [f'{t:6.1f}s  {m}' for t, m in ev]
            if self.ghost.result.events and self.ghost.result.outcome == 'collision':
                lines.append(f'{self.ghost.result.time:6.1f}s  baseline: COLLISION')
        self.log_text.set_text('\n'.join(lines[-9:]))

    # =================================================================== run
    def run(self):
        if self.record:
            self._record()
            return
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.frame, interval=1, blit=False, cache_frame_data=False)
        self.plt.show()

    def _record(self):
        from matplotlib.animation import FFMpegWriter
        fps = int(self.cfg.camera.rate)
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        total = self.duration or 40.0
        with writer.saving(self.fig, self.record, dpi=100):
            n_frames = int(total * fps / self.speed)
            post = 0
            for i in range(n_frames):
                self.frame()
                writer.grab_frame()
                if self.sim.result.outcome != 'running':
                    post += 1
                    if post > fps * 2:
                        break
        print('saved', self.record)
