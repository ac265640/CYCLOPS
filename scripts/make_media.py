"""Generate the screenshots used in the report and on the project website.

    python scripts/make_media.py
Writes docs/figures/ui_*.png, docs/figures/fpv_*.png and docs/figures/architecture.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from cyclops.config import Config  # noqa: E402
from cyclops.demo.live import LiveDemo  # noqa: E402
from cyclops.perception.perception import Perception  # noqa: E402

FIG = os.path.join(ROOT, 'docs', 'figures')

# (scenario, seed, simulated seconds before the snapshot)
SHOTS = [('forest_sar', 108, 7.0), ('urban_inspection', 107, 6.5), ('dynamic', 109, 6.0), ('pillars', 105, 6.0)]


def dashboard_shots(perc):
    for sc, seed, t in SHOTS:
        d = LiveDemo(sc, seed, 'vision', perc)
        while d.sim.t < t:
            d.frame()
        d.fig.savefig(os.path.join(FIG, f'ui_{sc}.png'), dpi=110, facecolor=d.fig.get_facecolor())
        # FPV camera crop: the onboard image with detections, at higher resolution
        ext = d.ax_cam.get_window_extent().transformed(d.fig.dpi_scale_trans.inverted())
        d.fig.savefig(os.path.join(FIG, f'fpv_{sc}.png'), dpi=160, bbox_inches=ext.expanded(1.02, 1.08),
                      facecolor=d.fig.get_facecolor())
        if sc == 'forest_sar':
            d._on_button('help')
            d.draw()
            d.fig.savefig(os.path.join(FIG, 'ui_help.png'), dpi=110, facecolor=d.fig.get_facecolor())
            d._on_button('help')
            while d.sim.result.outcome == 'running' and d.sim.t < 60:
                d.frame()
            d.draw()
            d.fig.savefig(os.path.join(FIG, 'ui_goal_reached.png'), dpi=110, facecolor=d.fig.get_facecolor())
        plt.close(d.fig)
        print('saved', sc)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                                       fc=fc, ec='#1f2328', lw=1.3))
        ax.text(x + w / 2, y + h - 0.18, title, ha='center', va='top', fontsize=11, weight='bold')
        ax.text(x + w / 2, y + h - 0.52, '\n'.join(lines), ha='center', va='top', fontsize=8.6, linespacing=1.35)

    def arrow(x0, y0, x1, y1, label=None, dy=0.12, ha='center'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#1f2328', mutation_scale=14))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha=ha, fontsize=8.2, color='#57606a', style='italic')

    P, V, C, S = '#ddf4ff', '#dafbe1', '#fff8c5', '#ffebe9'
    top, bot, h = 3.0, 0.3, 1.8
    box(0.2, top, 2.2, h, 'FPV camera', ['monocular, 90 deg FOV', '160x120 @ 15 Hz', '(ray-cast renderer)'], S)
    box(3.0, top, 3.0, h, 'Perception', ['CNN detector (CenterNet-style)', 'MonoDepthNet (U-Net)',
                                         'bearing + 3 range cues', 'wide obstacles -> columns'], P)
    box(6.6, top, 2.7, h, 'EKF tracker', ['world-frame CV model', 'Mode 1: range + bearing',
                                          'Mode 2: bearing only', 'Hungarian association'], P)
    box(9.9, top, 2.9, h, 'APF-rollout planner', ['candidates inside camera FOV', 'APF cost along rollouts',
                                                  'stopping distance, v/a/yaw limits', 'moving-obstacle prediction'], V)
    box(9.9, bot, 2.9, h, 'Commander', ['preflight checks, arming', 'MANUAL / ALTCTL / POSCTL',
                                        'TAKEOFF / MISSION / RTL / LAND', 'low-battery failsafe'], C)
    box(6.6, bot, 2.7, h, 'Flight controller', ['position P -> velocity PID', 'tilt-prioritised attitude',
                                                'body-rate PID', 'Quad-X mixer'], C)
    box(3.0, bot, 3.0, h, '6-DOF quadrotor', ['Newton-Euler dynamics', 'motor/ESC lag, propellers',
                                               'LiPo battery with sag', 'drag, ground contact'], S)
    box(0.2, bot, 2.2, h, 'Sensors + EKF', ['IMU 200 Hz, GPS 10 Hz', 'barometer, AHRS', 'Kalman estimator'], S)
    ym, yb = top + h / 2, bot + h / 2
    arrow(2.4, ym, 3.0, ym, 'RGB')
    arrow(6.0, ym, 6.6, ym, 'az, el, r')
    arrow(9.3, ym, 9.9, ym, 'tracks')
    arrow(11.0, top, 11.0, bot + h)
    ax.text(11.08, top - 0.22, 'velocity + yaw', fontsize=8.2, color='#57606a', style='italic')
    arrow(9.9, yb, 9.3, yb, 'setpoint')
    arrow(6.6, yb, 6.0, yb, 'rotor speed')
    arrow(3.0, yb, 2.4, yb, 'motion')
    arrow(0.7, bot + h, 0.7, top, 'pose', dy=0, ha='left')
    # odometry: sensors -> planner, routed through the gap between the rows
    gy = (bot + h + top) / 2
    ax.plot([1.9, 1.9, 12.3], [bot + h, gy, gy], color='#8c959f', lw=1.3, ls='--')
    ax.annotate('', xy=(12.3, top), xytext=(12.3, gy),
                arrowprops=dict(arrowstyle='-|>', lw=1.3, color='#8c959f', ls='--', mutation_scale=12))
    ax.text(6.4, gy + 0.08, 'odometry: position, velocity, attitude', fontsize=8.2, color='#57606a',
            style='italic', ha='center')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'architecture.png'), dpi=160)
    plt.close(fig)
    print('saved architecture')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    architecture()
    dashboard_shots(Perception(Config().camera))
