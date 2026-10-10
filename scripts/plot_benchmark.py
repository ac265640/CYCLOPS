"""Figures for the benchmark (reads results/benchmark*.json/csv)."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, 'docs', 'figures')
COL = {'baseline': '#d1495b', 'apf_vision': '#e8833a', 'cyclops': '#2a6fdb', 'oracle': '#3fa45b'}
LBL = {'baseline': 'Baseline (straight line)', 'apf_vision': 'Classic APF + vision',
       'cyclops': 'CYCLOPS (APF-rollout + vision)', 'oracle': 'Oracle perception (upper bound)'}
SC = ['forest_sar', 'urban_inspection', 'dynamic', 'pillars']


def plot_all():
    os.makedirs(FIG, exist_ok=True)
    summ = json.load(open(os.path.join(ROOT, 'results', 'benchmark_summary.json')))
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'benchmark.csv'))))
    methods = [m for m in COL if m in summ]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    x = np.arange(len(SC) + 1)
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [100 * summ[m][s]['success_rate'] for s in SC + ['ALL']]
        ax[0].bar(x + i * wdt - 0.4 + wdt / 2, vals, wdt, color=COL[m], label=LBL[m])
    ax[0].set_xticks(x, [s.replace('_', '\n') for s in SC] + ['ALL'])
    ax[0].set_ylabel('success rate [%]')
    ax[0].set_ylim(0, 105)
    ax[0].set_title('Avoidance success rate')
    ax[0].legend(fontsize=7.5, loc='lower left')
    for i, m in enumerate(methods):
        cl = [float(r['min_clearance']) for r in rows if r['method'] == m]
        pos = i + 1
        ax[1].boxplot([cl], positions=[pos], widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor=COL[m], alpha=0.6), medianprops=dict(color='k'))
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xticks(range(1, len(methods) + 1), [m for m in methods])
    ax[1].set_ylabel('min clearance to obstacle surface [m]')
    ax[1].set_title('Minimum clearance (< 0 = collision)')
    for i, m in enumerate(methods):
        t = [float(r['mission_time']) for r in rows if r['method'] == m and r['outcome'] == 'success']
        if t:
            ax[2].boxplot([t], positions=[i + 1], widths=0.6, patch_artist=True,
                          boxprops=dict(facecolor=COL[m], alpha=0.6), medianprops=dict(color='k'))
    ax[2].set_xticks(range(1, len(methods) + 1), [m for m in methods])
    ax[2].set_ylabel('mission time of successful runs [s]')
    ax[2].set_title('Mission completion time')
    for a in ax:
        a.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'benchmark.png'), dpi=130)

    # trajectories
    tp = os.path.join(ROOT, 'results', 'benchmark_trajectories.json')
    if os.path.exists(tp):
        trajs = json.load(open(tp))
        from cyclops.world.world import make_world
        seed0 = min(int(r['seed']) for r in rows)
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        for ax, sc in zip(axes.ravel(), SC):
            w = make_world(sc, seed=seed0)
            for ob in w.obstacles:
                if ob.kind == 'box':
                    ax.add_patch(plt.Rectangle(ob.pos[:2] - ob.size[:2], *(2 * ob.size[:2]), fc='#9aa0a6'))
                else:
                    ax.add_patch(plt.Circle(ob.pos[:2], ob.radius_xy, fc='#9aa0a6' if not ob.moving else '#e8a0a8'))
            for m in methods:
                t = trajs.get(f'{m}|{sc}')
                if t:
                    t = np.array(t)
                    ax.plot(t[:, 0], t[:, 1], color=COL[m], lw=2, label=LBL[m])
                    ax.plot(t[-1, 0], t[-1, 1], 'o', color=COL[m])
            ax.plot(*w.goal[:2], marker='*', ms=16, color='#f2cc60')
            ax.set_aspect('equal')
            ax.set_xlim(-3, w.goal[0] + 4)
            ax.set_ylim(-10, 10)
            ax.set_title(sc)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7.5, loc='lower left')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'benchmark_trajectories.png'), dpi=120)
    print('figures written to', FIG)


if __name__ == '__main__':
    plot_all()
