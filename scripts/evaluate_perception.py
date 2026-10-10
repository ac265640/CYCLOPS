"""Offline evaluation of the perception networks on the held-out validation set.

Reports (report section 8.6):
  * detection precision / recall / F1 / AP@0.5 on recorded simulation frames
  * monocular depth metrics: AbsRel, RMSE, delta < 1.25
  * per-obstacle range error of the three monocular range cues, binned by distance
  * inference latency on CPU (companion-computer proxy)
Writes results/perception_metrics.json and figures to docs/figures/.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from cyclops.config import CameraParams  # noqa: E402
from cyclops.perception.networks import (ObstacleDetector, MonoDepthNet, to_tensor,  # noqa: E402
                                           decode_detections)
from cyclops.sim.simulator import iou  # noqa: E402

FIG = os.path.join(ROOT, 'docs', 'figures')
RES = os.path.join(ROOT, 'results')


def average_precision(scored, n_gt):
    """scored: list of (score, is_tp). VOC-style all-point interpolated AP."""
    if not scored:
        return 0.0, np.array([0]), np.array([0])
    scored = sorted(scored, key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([1 - s[1] for s in scored])
    rec = tp / max(n_gt, 1)
    prec = tp / (tp + fp)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), rec, prec


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    cp = CameraParams()
    d = np.load(os.path.join(ROOT, 'data', 'val.npz'))
    rgb, depth, labels = d['rgb'], d['depth'].astype(np.float32), d['labels']
    det = ObstacleDetector().eval()
    dep = MonoDepthNet().eval()
    det.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'detector.pt'), map_location='cpu'))
    dep.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'depth.pt'), map_location='cpu'))

    n = len(rgb)
    preds, depth_pred = [], []
    with torch.no_grad():
        for k in range(0, n, 100):
            x = to_tensor(rgb[k:k + 100])
            hm, wh, off = det(x)
            preds += decode_detections(hm, wh, off, k=30, thresh=0.05)
            depth_pred.append(torch.exp(dep(x))[:, 0].numpy())
    depth_pred = np.concatenate(depth_pred)

    # ---------------------------------------------------------------- detection
    gts = [[] for _ in range(n)]
    for i, x0, y0, x1, y1, dd in labels:
        gts[int(i)].append((x0, y0, x1, y1, dd))
    results = {}
    for name, max_d in (('all (<=30 m)', 30.0), ('near (<=15 m)', 15.0)):
        scored, n_gt = [], 0
        tp_t = fp_t = fn_t = 0
        for i in range(n):
            g = [b for b in gts[i] if b[4] <= max_d]
            ignore = [b for b in gts[i] if b[4] > max_d]
            n_gt += len(g)
            used = set()
            for p in preds[i]:
                best, bj = 0.0, -1
                for j, b in enumerate(g):
                    v = iou(p[:4], b[:4])
                    if v > best and j not in used:
                        best, bj = v, j
                if best >= 0.5:
                    used.add(bj)
                    scored.append((p[4], 1))
                    if p[4] >= 0.35:
                        tp_t += 1
                elif any(iou(p[:4], b[:4]) >= 0.3 for b in ignore):
                    continue
                else:
                    scored.append((p[4], 0))
                    if p[4] >= 0.35:
                        fp_t += 1
            fn_t += len(g) - len([1 for j in used])
            # FN at the operating threshold
        ap, rec, prec = average_precision(scored, n_gt)
        P = tp_t / max(tp_t + fp_t, 1)
        R = tp_t / max(n_gt, 1)
        results[name] = dict(AP50=ap, precision=P, recall=R, F1=2 * P * R / max(P + R, 1e-9), n_gt=n_gt)
        if name.startswith('all'):
            pr_curve = (rec, prec)

    # ---------------------------------------------------------------- depth map
    gt = np.clip(depth, 0.3, 40.0)
    valid = depth < 40.0
    pr = depth_pred
    absrel = float(np.mean(np.abs(pr[valid] - gt[valid]) / gt[valid]))
    rmse = float(np.sqrt(np.mean((pr[valid] - gt[valid]) ** 2)))
    ratio = np.maximum(pr[valid] / gt[valid], gt[valid] / pr[valid])
    d125 = float(np.mean(ratio < 1.25))
    results['depth'] = dict(AbsRel=absrel, RMSE_m=rmse, delta_1_25=d125)

    # ------------------------------------------------- per-obstacle range cues
    from cyclops.perception.perception import Perception
    perc = Perception(cp, device='cpu', det_thresh=0.35)
    errs = {'depth net (Mode 1)': [], 'ground contact': [], 'known size': []}
    poses = d['poses']
    for i in range(0, n, 2):
        out, _ = perc.process(rgb[i], poses[i, :3].astype(float), poses[i, 3:].reshape(3, 3).astype(float))
        for o in out:
            best, bj = 0.0, None
            for b in gts[i]:
                v = iou(o['box'], b[:4])
                if v > best:
                    best, bj = v, b
            if best < 0.5:
                continue
            gt_r = bj[4] / max(o['ray_c'][2], 0.2)
            errs['depth net (Mode 1)'].append((gt_r, o['r_depth']))
            errs['known size'].append((gt_r, o['r_size']))
            if np.isfinite(o['r_ground']):
                errs['ground contact'].append((gt_r, o['r_ground']))
    bins = [0, 5, 10, 15, 20, 30]
    range_tab = {}
    for k, v in errs.items():
        a = np.array(v).reshape(-1, 2)
        row = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            row[f'{lo}-{hi} m'] = float(np.mean(np.abs(a[m, 1] - a[m, 0]) / a[m, 0])) if m.any() else None
        row['overall_rel'] = float(np.mean(np.abs(a[:, 1] - a[:, 0]) / a[:, 0])) if len(a) else None
        row['overall_mae_m'] = float(np.mean(np.abs(a[:, 1] - a[:, 0]))) if len(a) else None
        row['n'] = int(len(a))
        range_tab[k] = row
    results['range_relative_error'] = range_tab

    # ---------------------------------------------------------------- latency
    x = to_tensor(rgb[:1])
    with torch.no_grad():
        for _ in range(5):
            det(x), dep(x)
        t0 = time.perf_counter()
        for _ in range(50):
            det(x)
        t_det = (time.perf_counter() - t0) / 50 * 1e3
        t0 = time.perf_counter()
        for _ in range(50):
            dep(x)
        t_dep = (time.perf_counter() - t0) / 50 * 1e3
    results['latency_ms_cpu'] = dict(detector=t_det, depth=t_dep, total=t_det + t_dep)
    results['params'] = dict(detector=sum(p.numel() for p in det.parameters()),
                             depth=sum(p.numel() for p in dep.parameters()))

    with open(os.path.join(RES, 'perception_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(pr_curve[0], pr_curve[1], color='#2a6fdb', lw=2)
    ax[0].set_xlabel('recall')
    ax[0].set_ylabel('precision')
    ax[0].set_title(f"Detector PR curve (AP50 = {results['all (<=30 m)']['AP50']:.2f})")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.02)
    ax[0].grid(alpha=0.3)
    cols = ['#2a6fdb', '#e8833a', '#3fa45b']
    centers = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
    for c, (k, row) in zip(cols, range_tab.items()):
        ys = [row[f'{lo}-{hi} m'] for lo, hi in zip(bins[:-1], bins[1:])]
        ax[1].plot(centers, [np.nan if y is None else 100 * y for y in ys], 'o-', color=c, label=k, lw=2)
    ax[1].set_xlabel('true range to obstacle [m]')
    ax[1].set_ylabel('mean relative range error [%]')
    ax[1].set_title('Monocular range cues')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'perception_eval.png'), dpi=130)

    # qualitative examples
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    rng = np.random.default_rng(0)
    for col, i in enumerate(rng.choice(n, 4, replace=False)):
        axes[0, col].imshow(rgb[i])
        for p in preds[i]:
            if p[4] >= 0.35:
                axes[0, col].add_patch(plt.Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False, ec='lime', lw=1.5))
        axes[1, col].imshow(np.log(np.clip(depth[i], 0.3, 40)), cmap='magma_r', vmin=np.log(0.5), vmax=np.log(40))
        axes[2, col].imshow(np.log(depth_pred[i]), cmap='magma_r', vmin=np.log(0.5), vmax=np.log(40))
        for r in range(3):
            axes[r, col].axis('off')
    axes[0, 0].set_title('camera + detections', loc='left')
    axes[1, 0].set_title('ground-truth depth', loc='left')
    axes[2, 0].set_title('predicted depth (MonoDepthNet)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'perception_examples.png'), dpi=110)


if __name__ == '__main__':
    main()
