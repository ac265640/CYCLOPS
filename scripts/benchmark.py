"""Monte-Carlo benchmark (report section 8.6).

Methods
    baseline      straight-line waypoint follower, no obstacle awareness   (report 8.5)
    apf_vision    classic artificial potential field + deep monocular perception
    cyclops       APF-rollout planner (FOV-aware) + deep monocular perception  <- proposed
    oracle        APF-rollout planner + ground-truth obstacles (upper bound)

    python scripts/benchmark.py --seeds 10 --workers 6
Outputs results/benchmark.csv, results/benchmark_summary.json, docs/figures/benchmark*.png
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

METHODS = {
    'baseline': dict(method='baseline', planner='apf_rollout'),
    'apf_vision': dict(method='vision', planner='apf'),
    'cyclops': dict(method='vision', planner='apf_rollout'),
    'oracle': dict(method='oracle', planner='apf_rollout'),
}
SCENARIOS = ['forest_sar', 'urban_inspection', 'dynamic', 'pillars']
_PERC = None


def run_one(job):
    global _PERC
    import torch
    torch.set_num_threads(1)
    from cyclops.config import Config
    from cyclops.sim.simulator import Simulation
    from cyclops.world.world import make_world
    name, scenario, seed, max_time, keep_traj = job
    spec = METHODS[name]
    perc = None
    if spec['method'] == 'vision':
        if _PERC is None:
            from cyclops.perception.perception import Perception
            _PERC = Perception(Config().camera)
            torch.set_num_threads(1)
        perc = _PERC
    world = make_world(scenario, seed=seed)
    sim = Simulation(world, spec['method'], perception=perc, seed=seed, planner=spec['planner'])
    t0 = time.time()
    res = sim.run(max_time=max_time)
    m = dict(res.metrics)
    m.update(method=name, scenario=scenario, seed=seed, wall_s=time.time() - t0)
    traj = res.log['pos'][::20, :2].tolist() if keep_traj else None
    return m, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed0', type=int, default=100)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max-time', type=float, default=60.0)
    ap.add_argument('--methods', default=','.join(METHODS))
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)

    jobs = [(m, sc, args.seed0 + k, args.max_time, k == 0)
            for m in args.methods.split(',') for sc in SCENARIOS for k in range(args.seeds)]
    rows, trajs = [], {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            m, traj = f.result()
            rows.append(m)
            if traj is not None:
                trajs[f"{m['method']}|{m['scenario']}"] = traj
            print(f"[{i + 1}/{len(jobs)}] {m['method']:10s} {m['scenario']:17s} seed {m['seed']}  "
                  f"{m['outcome']:9s} clr {m['min_clearance']:5.2f}  T {m['mission_time']:5.1f}s  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    keys = ['method', 'scenario', 'seed', 'outcome', 'mission_time', 'path_length', 'min_clearance',
            'rmse_cross_track', 'rmse_altitude', 'energy_wh', 'mean_latency_ms', 'p95_latency_ms',
            'precision', 'recall', 'det_tp', 'det_fp', 'det_fn', 'range_mae', 'range_rel', 'wall_s']
    with open(os.path.join(ROOT, 'results', 'benchmark.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['method'], r['scenario'], r['seed'])):
            w.writerow(r)
    with open(os.path.join(ROOT, 'results', 'benchmark_trajectories.json'), 'w') as f:
        json.dump(trajs, f)

    summary = {}
    for m in args.methods.split(','):
        summary[m] = {}
        for sc in SCENARIOS + ['ALL']:
            rs = [r for r in rows if r['method'] == m and (sc == 'ALL' or r['scenario'] == sc)]
            if not rs:
                continue
            succ = [r for r in rs if r['outcome'] == 'success']
            tp = sum(r['det_tp'] for r in rs)
            fp = sum(r['det_fp'] for r in rs)
            fn = sum(r['det_fn'] for r in rs)
            summary[m][sc] = dict(
                runs=len(rs),
                success_rate=len(succ) / len(rs),
                collision_rate=sum(r['outcome'] == 'collision' for r in rs) / len(rs),
                timeout_rate=sum(r['outcome'] == 'timeout' for r in rs) / len(rs),
                min_clearance_mean=float(np.mean([r['min_clearance'] for r in succ])) if succ else None,
                min_clearance_worst=float(np.min([r['min_clearance'] for r in rs])),
                mission_time_mean=float(np.mean([r['mission_time'] for r in succ])) if succ else None,
                path_length_mean=float(np.mean([r['path_length'] for r in succ])) if succ else None,
                rmse_cross_track_mean=float(np.mean([r['rmse_cross_track'] for r in succ])) if succ else None,
                rmse_altitude_mean=float(np.mean([r['rmse_altitude'] for r in rs])),
                energy_wh_mean=float(np.mean([r['energy_wh'] for r in succ])) if succ else None,
                latency_ms_mean=float(np.mean([r['mean_latency_ms'] for r in rs])),
                latency_ms_p95=float(np.mean([r['p95_latency_ms'] for r in rs])),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
            )
    with open(os.path.join(ROOT, 'results', 'benchmark_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({m: summary[m].get('ALL') for m in summary}, indent=2))
    from plot_benchmark import plot_all
    plot_all()


if __name__ == '__main__':
    main()
