"""CYCLOPS live demo: real-time dashboard of the full closed loop.

Layout
    left sidebar   clickable controls: scenario, perception mode, play/pause,
                   restart, new map, ghost baseline, manual flight, return home,
                   land, help and a simulation-speed slider, plus colour legends
    status banner  what the drone is doing right now, in plain English, with a
                   mission progress bar
    FPV camera     onboard image + CNN detections coloured by distance, with an
                   FPV-goggle OSD (horizon, flight-path marker, battery, altitude)
    tactical map   top-down view: real obstacles, camera estimates with safety
                   margins, camera field of view, candidate paths, ghost baseline
    bottom row     depth-network output, telemetry (battery, ESC/motors, attitude),
                   mission metrics, commander log (preflight, modes, failsafes)

Keyboard shortcuts (the buttons do the same):
    space pause/resume   1-4 scenario   s new map   r restart   v vision/perfect
    b ghost   m manual flight (arrows = move, q/e = turn, z/x = down/up)
    h return home   l land   +/- speed   ? help
"""
import copy
import time
import warnings

import matplotlib
import numpy as np

from ..config import Config
from ..control.commander import RCInput
from ..geometry import camera_rotation
from ..sim.simulator import Simulation
from ..world.world import make_world, SCENARIOS

BG = '#0d1117'
PANEL = '#161b22'
EDGE = '#30363d'
FG = '#e6edf3'
DIM = '#8b949e'
ACC = '#3fb950'
WARN = '#d29922'
BAD = '#f85149'
CYAN = '#39c5cf'
BLUE = '#58a6ff'
PURPLE = '#bc8cff'
BTN = '#21262d'
BTN_HOVER = '#30363d'

SCENARIO_LABELS = {
    'forest_sar': 'Forest search & rescue',
    'urban_inspection': 'Urban inspection',
    'dynamic': 'Moving obstacles',
    'pillars': 'Pillar slalom',
}
SCENARIO_BLURB = {
    'forest_sar': 'Find a way through a dense forest, as in a search-and-rescue mission.',
    'urban_inspection': 'Fly between buildings and poles, as in infrastructure inspection.',
    'dynamic': 'Obstacles move across the flight path; the drone predicts where they go.',
    'pillars': 'A tight slalom of pillars: a stress test for the planner.',
}

HELP_TEXT = """HOW TO READ THIS SCREEN

FPV CAMERA (left)   What the drone's single camera sees.
  Boxes = obstacles found by the neural network, with the
  estimated distance and direction. Box colour = distance:
  red < 3 m, amber 3-7 m, green > 7 m.
  White line = horizon, green circle = where the drone is heading.

TACTICAL MAP (right)   Bird's-eye view.
  Grey shapes = real obstacles (the drone does NOT know these).
  Cyan circles = where the camera THINKS obstacles are;
  dashed ring = safety margin kept around them.
  Grey wedge = camera field of view.
  Green/red rays = candidate directions (safe / unsafe).
  Green line = CYCLOPS path. Red dashed = baseline drone with no
  avoidance, flying the same route for comparison.

BOTTOM ROW   Depth network output (bright = close), battery and
  motors, mission metrics, and the flight-controller log.

Click 'Help' again to close."""


def range_color(r):
    if r < 3:
        return BAD
    if r < 7:
        return WARN
    return ACC


class LiveDemo:
    def __init__(self, scenario='forest_sar', seed=108, method='vision', perception=None,
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
        self.help_on = False
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
        self.goal_dist0 = float(np.linalg.norm(self.world.goal[:2] - self.world.start[:2]))
        self._draw_static_map()
        self._fps = 0.0
        self._last_frame = time.time()
        if not ok:
            self.sim.cmd.log(0, 'preflight failed - not armed')
        self._sync_buttons()

    # =================================================================== figure
    def _panel(self, rect, title=None):
        ax = self.fig.add_axes(rect)
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(EDGE)
        if title:
            ax.set_title(title, loc='left', color=FG, fontsize=10, weight='bold', pad=5)
        return ax

    def _build_figure(self):
        plt = self.plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': FG,
                             'axes.labelcolor': DIM, 'xtick.color': DIM, 'ytick.color': DIM,
                             'axes.edgecolor': EDGE, 'toolbar': 'None'})
        self.fig = plt.figure(figsize=(14, 8.2), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title('CYCLOPS - monocular obstacle avoidance demo')
        except Exception:
            pass
        self._build_sidebar()

        # ---------------- header: title, status banner, progress bar
        self.title = self.fig.text(0.195, 0.962, '', fontsize=16, weight='bold', color=FG)
        self.blurb = self.fig.text(0.195, 0.936, '', fontsize=9.5, color=DIM)
        self.banner = self.fig.text(0.195, 0.895, '', fontsize=12.5, weight='bold', color=BLUE,
                                    bbox=dict(boxstyle='round,pad=0.45', fc=PANEL, ec=EDGE))
        self.banner2 = self.fig.text(0.99, 0.895, '', fontsize=9.5, color=DIM, ha='right')
        self.ax_prog = self.fig.add_axes([0.195, 0.862, 0.795, 0.011])
        self.ax_prog.set_xlim(0, 1)
        self.ax_prog.set_ylim(0, 1)
        self.ax_prog.axis('off')
        self.ax_prog.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, fc=BTN, ec='none'))
        self.prog_bar = matplotlib.patches.Rectangle((0, 0), 0, 1, fc=BLUE, ec='none')
        self.ax_prog.add_patch(self.prog_bar)

        # ---------------- FPV camera
        cp = self.cfg.camera
        self.ax_cam = self._panel([0.195, 0.335, 0.39, 0.49], 'What the drone sees  (single camera + neural network)')
        self.cam_img = self.ax_cam.imshow(np.zeros((cp.height, cp.width, 3), np.uint8),
                                          extent=(0, cp.width, cp.height, 0), interpolation='bilinear')
        self.ax_cam.set_xlim(0, cp.width)
        self.ax_cam.set_ylim(cp.height, 0)
        self.ax_cam.set_xticks([])
        self.ax_cam.set_yticks([])
        self.horizon, = self.ax_cam.plot([], [], color='white', lw=1.0, alpha=0.8)
        self.fpm, = self.ax_cam.plot([], [], marker='o', ms=10, mfc='none', mec=ACC, mew=1.8)
        cx, cy = cp.cx, cp.cy
        self.ax_cam.plot([cx - 9, cx - 3], [cy, cy], color='white', lw=1.4)
        self.ax_cam.plot([cx + 3, cx + 9], [cy, cy], color='white', lw=1.4)
        osd = dict(fontsize=10, color='white', weight='bold', family='DejaVu Sans Mono')
        self.osd_tl = self.ax_cam.text(3, 5, '', va='top', **osd)
        self.osd_tr = self.ax_cam.text(cp.width - 3, 5, '', va='top', ha='right', **osd)
        self.osd_bl = self.ax_cam.text(3, cp.height - 3, '', va='bottom', **osd)
        self.osd_br = self.ax_cam.text(cp.width - 3, cp.height - 3, '', va='bottom', ha='right', **osd)
        self.osd_center = self.ax_cam.text(cx, cy + 24, '', fontsize=20, color=BAD, ha='center', weight='bold',
                                           bbox=dict(fc='black', alpha=0.0, lw=0))
        self.det_artists = []

        # ---------------- tactical map
        self.ax_map = self._panel([0.605, 0.335, 0.385, 0.49], "Bird's-eye map  (grey = real, cyan = camera estimate)")
        self.ax_map.set_aspect('equal', adjustable='datalim')
        self.ax_map.grid(color=EDGE, lw=0.5)
        self.ax_map.tick_params(labelsize=7.5)
        self.trail_line, = self.ax_map.plot([], [], color=ACC, lw=2.2, label='CYCLOPS path')
        self.ghost_line, = self.ax_map.plot([], [], color=BAD, lw=1.4, ls='--', label='baseline drone (no avoidance)')
        self.ghost_dot, = self.ax_map.plot([], [], marker='o', color=BAD, ms=6)
        self.ghost_x, = self.ax_map.plot([], [], marker='X', color=BAD, ms=15, ls='none')
        self.drone_dot, = self.ax_map.plot([], [], marker=(3, 0, 0), color=FG, ms=13, ls='none')
        self.fov_poly = matplotlib.patches.Polygon(np.zeros((3, 2)), closed=True, fc=FG, alpha=0.07, ec=FG, lw=0.6)
        self.ax_map.add_patch(self.fov_poly)
        self.roll_lines = matplotlib.collections.LineCollection([], lw=0.8)
        self.ax_map.add_collection(self.roll_lines)
        self.best_line, = self.ax_map.plot([], [], color='#f0f6fc', lw=2.2, alpha=0.9, label='chosen path')
        self.track_coll = None
        self.mover_patches = []
        h = [matplotlib.lines.Line2D([], [], color=ACC, lw=2.2, label='CYCLOPS path'),
             matplotlib.lines.Line2D([], [], color=BAD, lw=1.4, ls='--', label='baseline (no avoidance)'),
             matplotlib.lines.Line2D([], [], marker='o', ls='none', mfc='#6e7681', mec='#8b949e', ms=8, label='real obstacle'),
             matplotlib.lines.Line2D([], [], marker='o', ls='none', mfc=(0.22, 0.77, 0.81, 0.35), mec=CYAN, ms=8,
                                     label='camera estimate'),
             matplotlib.lines.Line2D([], [], marker='*', ls='none', color='#f2cc60', ms=11, label='goal')]
        self.ax_map.legend(handles=h, loc='lower left', facecolor=PANEL, edgecolor=EDGE, labelcolor=FG,
                           fontsize=7.5, ncol=1, framealpha=0.85)

        # ---------------- bottom row
        self.ax_depth = self._panel([0.195, 0.04, 0.17, 0.235], 'Depth network  (bright = close)')
        self.depth_img = self.ax_depth.imshow(np.zeros((cp.height, cp.width)), cmap='magma_r',
                                              vmin=np.log(0.5), vmax=np.log(40))
        self.ax_depth.set_xticks([])
        self.ax_depth.set_yticks([])
        self.ax_tel = self._panel([0.38, 0.04, 0.2, 0.235], 'Battery & motors')
        self.ax_met = self._panel([0.595, 0.04, 0.19, 0.235], 'Mission metrics')
        self.ax_log = self._panel([0.80, 0.04, 0.19, 0.235], 'Flight controller log')
        mono = dict(fontsize=8.5, color=FG, family='DejaVu Sans Mono', va='top')
        for ax in (self.ax_tel, self.ax_met, self.ax_log):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.tel_text = self.ax_tel.text(0.04, 0.95, '', **mono)
        ys = [0.30, 0.22, 0.14, 0.06]
        self.motor_bars = self.ax_tel.barh(ys, [0, 0, 0, 0], height=0.06, left=0.2, color=CYAN)
        for k, y in enumerate(ys):
            self.ax_tel.text(0.04, y, f'motor {k + 1}', va='center', fontsize=7.5, color=DIM)
        self.met_text = self.ax_met.text(0.04, 0.95, '', **mono)
        self.log_text = self.ax_log.text(0.04, 0.95, '', **dict(mono, fontsize=8))

        # ---------------- help overlay
        self.help_box = self.fig.text(0.59, 0.58, HELP_TEXT, fontsize=10, color=FG, ha='center', va='center',
                                      multialignment='left',
                                      family='DejaVu Sans Mono', visible=False, zorder=100,
                                      bbox=dict(boxstyle='round,pad=1.0', fc='#010409', ec=BLUE, alpha=0.96))

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)

    def _build_sidebar(self):
        from matplotlib.widgets import Button, RadioButtons, Slider
        f = self.fig
        f.patches.append(matplotlib.patches.FancyBboxPatch((0.006, 0.012), 0.168, 0.976, transform=f.transFigure,
                                                           boxstyle='round,pad=0,rounding_size=0.006',
                                                           fc=PANEL, ec=EDGE, zorder=-1))
        f.text(0.02, 0.955, 'CYCLOPS', fontsize=20, weight='bold', color=FG)
        f.text(0.02, 0.928, 'Monocular obstacle avoidance', fontsize=9, color=DIM)
        f.text(0.02, 0.912, 'for small autonomous drones', fontsize=9, color=DIM)

        def label(y, txt):
            f.text(0.02, y, txt.upper(), fontsize=8, weight='bold', color=DIM)

        label(0.872, 'Mission scenario')
        ax = f.add_axes([0.015, 0.735, 0.15, 0.13], facecolor=PANEL)
        ax.set_frame_on(False)
        self.rb_scen = RadioButtons(ax, [SCENARIO_LABELS[s] for s in SCENARIOS],
                                    active=SCENARIOS.index(self.scenario), activecolor=BLUE)
        self._style_radio(self.rb_scen)
        self.rb_scen.on_clicked(self._on_scenario)

        label(0.715, 'Obstacle perception')
        ax = f.add_axes([0.015, 0.645, 0.15, 0.065], facecolor=PANEL)
        ax.set_frame_on(False)
        self.rb_perc = RadioButtons(ax, ['Camera + neural nets', 'Perfect (cheat) sensor'],
                                    active=0 if self.method == 'vision' else 1, activecolor=BLUE)
        self._style_radio(self.rb_perc)
        self.rb_perc.on_clicked(self._on_perception)

        label(0.615, 'Controls')
        self.buttons = {}
        specs = [('play', 'Pause', 0.015, 0.565), ('restart', 'Restart', 0.095, 0.565),
                 ('newmap', 'New map', 0.015, 0.515), ('ghost', 'Ghost: on', 0.095, 0.515),
                 ('manual', 'Fly manually', 0.015, 0.465), ('home', 'Return home', 0.095, 0.465),
                 ('land', 'Land', 0.015, 0.415), ('help', 'Help', 0.095, 0.415)]
        for key, txt, x, y in specs:
            ax = f.add_axes([x, y, 0.075, 0.04])
            b = Button(ax, txt, color=BTN, hovercolor=BTN_HOVER)
            b.label.set_color(FG)
            b.label.set_fontsize(9)
            for s in ax.spines.values():
                s.set_color(EDGE)
            b.on_clicked(lambda _e, k=key: self._on_button(k))
            self.buttons[key] = b

        label(0.37, 'Simulation speed')
        ax = f.add_axes([0.03, 0.335, 0.11, 0.025], facecolor=BTN)
        self.sl_speed = Slider(ax, '', 0.25, 4.0, valinit=self.speed, color=BLUE, valfmt='%.2fx')
        self.sl_speed.valtext.set_color(FG)
        self.sl_speed.on_changed(lambda v: setattr(self, 'speed', float(v)))

        # legends
        label(0.29, 'Camera box colours')
        for i, (c, t) in enumerate([(BAD, 'closer than 3 m'), (WARN, '3 - 7 m away'), (ACC, 'more than 7 m')]):
            y = 0.265 - i * 0.025
            f.patches.append(matplotlib.patches.Rectangle((0.02, y - 0.004), 0.014, 0.014, transform=f.transFigure,
                                                          fc='none', ec=c, lw=2))
            f.text(0.042, y, t, fontsize=8.5, color=FG, va='center')
        label(0.175, 'Keyboard (click the window first)')
        f.text(0.02, 0.165, 'space pause   1-4 scenario\n'
                            's new map    r restart\n'
                            'm manual: arrows move,\n'
                            '  q/e turn, z/x down/up\n'
                            'h home   l land   ? help',
               fontsize=7.8, color=DIM, va='top', family='DejaVu Sans Mono', linespacing=1.5)

    @staticmethod
    def _style_radio(rb):
        for lbl in rb.labels:
            lbl.set_color(FG)
            lbl.set_fontsize(9)

    # =================================================================== input
    def _on_scenario(self, label):
        for s, l in SCENARIO_LABELS.items():
            if l == label and s != self.scenario:
                self.scenario = s
                self.load()

    def _on_perception(self, label):
        m = 'vision' if label.startswith('Camera') else 'oracle'
        if m != self.method:
            self.method = m
            self.load()

    def _on_button(self, key):
        cmd = self.sim.cmd
        if key == 'play':
            self.paused = not self.paused
        elif key == 'restart':
            self.load()
        elif key == 'newmap':
            self.seed += 1
            self.load()
        elif key == 'ghost':
            self.ghost_on = not self.ghost_on
        elif key == 'manual':
            if cmd.armed:
                cmd.set_mode(self.sim.t, 'POSCTL' if cmd.mode != 'POSCTL' else 'MISSION')
        elif key == 'home':
            if cmd.armed:
                cmd.set_mode(self.sim.t, 'RTL')
        elif key == 'land':
            if cmd.armed:
                cmd.set_mode(self.sim.t, 'LAND')
        elif key == 'help':
            self.help_on = not self.help_on
            self.help_box.set_visible(self.help_on)
        self._sync_buttons()

    def _sync_buttons(self):
        if not hasattr(self, 'buttons'):
            return
        self.buttons['play'].label.set_text('Play' if self.paused else 'Pause')
        self.buttons['ghost'].label.set_text('Ghost: on' if self.ghost_on else 'Ghost: off')
        manual = hasattr(self, 'sim') and self.sim.cmd.mode == 'POSCTL'
        self.buttons['manual'].label.set_text('Resume auto' if manual else 'Fly manually')

    def _on_key(self, ev):
        k = ev.key
        self.keys_down.add(k)
        if k == ' ':
            self._on_button('play')
        elif k in ('1', '2', '3', '4'):
            self.rb_scen.set_active(int(k) - 1)
        elif k == 's':
            self._on_button('newmap')
        elif k == 'r':
            self._on_button('restart')
        elif k == 'v':
            self.rb_perc.set_active(1 if self.method == 'vision' else 0)
        elif k == 'b':
            self._on_button('ghost')
        elif k == 'm':
            self._on_button('manual')
        elif k == 'h':
            self._on_button('home')
        elif k == 'l':
            self._on_button('land')
        elif k == '?':
            self._on_button('help')
        elif k in ('+', '='):
            self.sl_speed.set_val(min(self.speed * 1.5, 4.0))
        elif k == '-':
            self.sl_speed.set_val(max(self.speed / 1.5, 0.25))
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
        self._static_patches += ax.plot([g[0]], [g[1]], marker='*', color='#f2cc60', ms=20)
        self._static_patches.append(ax.text(g[0], g[1] + 1.3, 'GOAL', color='#f2cc60', ha='center',
                                            fontsize=8, weight='bold'))
        self._static_patches += ax.plot([s[0]], [s[1]], marker='s', color=DIM, ms=7)
        self._static_patches.append(ax.text(s[0], s[1] - 1.6, 'START', color=DIM, ha='center', fontsize=8))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ax.set_xlim(-3, g[0] + 4)
            ax.set_ylim(-11, 11)
        self.title.set_text(SCENARIO_LABELS.get(self.scenario, self.scenario))
        mode = 'camera + neural networks' if self.sim.method == 'vision' else 'perfect (cheat) obstacle sensor'
        self.blurb.set_text(f'{SCENARIO_BLURB.get(self.scenario, "")}   Perception: {mode}.   Map #{self.seed}')

    def _status(self):
        sim = self.sim
        out = sim.result.outcome
        mode = sim.cmd.mode
        dist = float(np.linalg.norm(self.world.goal[:2] - sim.quad.p[:2]))
        clr = sim.min_clearance if np.isfinite(sim.min_clearance) else np.inf
        if out == 'success':
            return f'GOAL REACHED in {sim.result.time:.1f} s   closest approach {clr:.2f} m', ACC
        if out == 'collision':
            return f'COLLISION after {sim.result.time:.1f} s   try Restart or New map', BAD
        if out == 'timeout':
            return 'Mission timed out: the drone could not find a way through', WARN
        if self.paused:
            return 'Paused: click Play to continue', DIM
        if not sim.cmd.armed:
            return 'On the ground (disarmed)', DIM
        if mode == 'TAKEOFF':
            return 'Preflight checks passed. Taking off...', WARN
        if mode == 'POSCTL':
            return 'MANUAL: arrow keys move, Q/E turn, Z/X down/up (click the window first)', PURPLE
        if mode == 'RTL':
            return 'Returning home...', WARN
        if mode == 'LAND':
            return 'Landing...', WARN
        if sim.plan_info.get('blocked'):
            return f'Path blocked: hovering and looking around for a gap   ({dist:.0f} m to goal)', WARN
        n = len(sim.tracker.obstacles())
        return f'Flying autonomously, {n} obstacles tracked   {dist:.0f} m to goal', BLUE

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

        # ---- banner + progress
        txt, col = self._status()
        self.banner.set_text(txt)
        self.banner.set_color(col)
        self.banner.get_bbox_patch().set_edgecolor(col)
        g = self.ghost.result
        if not self.ghost_on:
            self.banner2.set_text('')
        elif g.outcome == 'collision':
            self.banner2.set_text(f'Baseline drone (no avoidance) crashed after {g.time:.1f} s')
            self.banner2.set_color(BAD)
        else:
            self.banner2.set_text('Baseline drone (no avoidance) is flying the same route')
            self.banner2.set_color(DIM)
        dist = float(np.linalg.norm(self.world.goal[:2] - quad.p[:2]))
        self.prog_bar.set_width(np.clip(1 - dist / self.goal_dist0, 0, 1))
        self.prog_bar.set_color(ACC if sim.result.outcome == 'success' else (BAD if sim.result.outcome == 'collision' else BLUE))

        # ---- detections
        for a in self.det_artists:
            a.remove()
        self.det_artists = []
        for d in sim.dets:
            x0, y0, x1, y1 = d['box']
            col = range_color(d['range'])
            r = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.8)
            self.ax_cam.add_patch(r)
            self.det_artists.append(r)
            if 'column_of' in d and d is not sim.dets[-1]:
                continue
            lbl = self.ax_cam.text(x0, max(y0 - 1, 4), f"{d['range']:.1f} m", fontsize=8, color=col,
                                   weight='bold', va='bottom', bbox=dict(fc='black', alpha=0.5, pad=0.6, lw=0))
            self.det_artists.append(lbl)

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
        self.osd_tl.set_text(mode)
        self.osd_tr.set_text(f'{b.voltage:4.1f}V {100 * b.soc:3.0f}%')
        self.osd_tr.set_color(BAD if b.low else ('white' if b.soc > 0.3 else WARN))
        self.osd_bl.set_text(f'ALT {quad.p[2]:3.1f} m   SPD {np.linalg.norm(quad.v[:2]):3.1f} m/s')
        self.osd_br.set_text(f'T+{sim.t:4.1f}s')
        out = sim.result.outcome
        self.osd_center.set_text({'collision': 'COLLISION', 'success': 'GOAL REACHED', 'timeout': 'TIMEOUT'}.get(out, ''))
        self.osd_center.set_color(BAD if out in ('collision', 'timeout') else ACC)
        self.osd_center.get_bbox_patch().set_alpha(0.55 if out != 'running' else 0.0)

        # ---- map
        tr = np.array(self.trail)
        self.trail_line.set_data(tr[:, 0] if len(tr) else [], tr[:, 1] if len(tr) else [])
        if self.ghost_on and self.ghost_trail:
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
        if self.track_coll is not None:
            self.track_coll.remove()
        circles, est = [], []
        for o in sim.tracker.obstacles():
            circles.append(matplotlib.patches.Circle(o['pos'][:2], o['radius']))
            circles.append(matplotlib.patches.Circle(o['pos'][:2], o['radius'] + min(0.5 * o['sigma'], 0.5)
                                                     + cfg.planner.drone_radius + cfg.planner.min_clearance))
            est += [True, False]
        self.track_coll = matplotlib.collections.PatchCollection(
            circles, facecolors=[(0.22, 0.77, 0.81, 0.35) if e else (0, 0, 0, 0) for e in est],
            edgecolors=[CYAN if e else (0.22, 0.77, 0.81, 0.45) for e in est],
            linestyles=['-' if e else '--' for e in est], linewidths=0.9)
        self.ax_map.add_collection(self.track_coll)
        info = sim.plan_info or {}
        if 'cands' in info and sim.cmd.mode == 'MISSION' and sim.result.outcome == 'running':
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
        self.tel_text.set_text(
            f'Battery  {b.voltage:5.2f} V   {100 * b.soc:3.0f} %\n'
            f'Current  {b.current:5.1f} A\n'
            f'Used     {b.energy_used_wh:5.2f} Wh\n'
            f'Roll {np.degrees(roll):+5.1f}  Pitch {np.degrees(pitch):+5.1f}\n'
            f'Heading {np.degrees(yaw) % 360:5.1f} deg\n'
            f'Height  {quad.p[2]:4.1f} m')
        wmax = cfg.airframe.motor_w_max
        for bar, w in zip(self.motor_bars, quad.rotor):
            bar.set_width(0.75 * w / wmax)
            bar.set_color(BAD if w > 0.95 * quad.w_max_now() else CYAN)

        ds = sim.det_stats
        tp, fp, fn = ds['tp'], ds['fp'], ds['fn']
        prec = f'{tp / (tp + fp):.2f}' if tp + fp else ' -- '
        rec = f'{tp / (tp + fn):.2f}' if tp + fn else ' -- '
        cur = self.clear_hist[-1] if self.clear_hist else np.inf
        mn = sim.min_clearance if np.isfinite(sim.min_clearance) else cur
        obs = sim.tracker.obstacles()
        self.met_text.set_text(
            f'Gap to nearest  {cur:5.2f} m\n'
            f'Closest so far  {mn:5.2f} m\n'
            f'Goal distance   {dist:5.1f} m\n'
            f'Obstacles seen  {len(obs):5d}\n'
            f'Detector P / R  {prec} / {rec}\n'
            f'Think time      {getattr(sim, "latency_ms", 0):5.1f} ms\n'
            f'Sim speed {self.speed:.2f}x  {self._fps:4.1f} fps')
        if sim.t < 4.0:
            lines = ['Preflight checklist:'] + [f"{'OK ' if c[1] else 'FAIL'} {c[0]}" for c in self.preflight[:7]]
        else:
            lines = [f'{t:5.1f}s {m}' for t, m in sim.cmd.events[-8:]]
        self.log_text.set_text('\n'.join(lines[-8:]))
        self._sync_buttons()

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
        with writer.saving(self.fig, self.record, dpi=110):
            n_frames = int(total * fps / self.speed)
            post = 0
            for _ in range(n_frames):
                self.frame()
                writer.grab_frame()
                if self.sim.result.outcome != 'running':
                    post += 1
                    if post > fps * 2:
                        break
        print('saved', self.record)
