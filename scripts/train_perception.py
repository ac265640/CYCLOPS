"""Generate the synthetic dataset and train the detector + monocular depth network.

    python scripts/train_perception.py                # full run (~10 min on an M-series Mac)
    python scripts/train_perception.py --quick        # smoke test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cyclops.perception.dataset import generate                       # noqa: E402
from cyclops.perception.networks import (ObstacleDetector, MonoDepthNet, to_tensor,  # noqa: E402
                                           gaussian_radius, focal_loss)

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data')
MODELS = os.path.join(ROOT, 'models')


def get_split(name, n, seed, close_frac=0.0):
    path = os.path.join(DATA, f'{name}.npz')
    if os.path.exists(path):
        d = np.load(path)
        if len(d['rgb']) == n:
            return d['rgb'], d['depth'], d['labels']
    rgb, depth, labels, poses = generate(n, seed=seed, close_frac=close_frac)
    np.savez_compressed(path, rgb=rgb, depth=depth, labels=labels, poses=poses)
    return rgb, depth, labels


def build_targets(labels, n, H, W, stride=4):
    h, w = H // stride, W // stride
    hm = np.zeros((n, 1, h, w), np.float32)
    wh = np.zeros((n, 2, h, w), np.float32)
    off = np.zeros((n, 2, h, w), np.float32)
    mask = np.zeros((n, 1, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for i, x0, y0, x1, y1, _ in labels:
        i = int(i)
        bw, bh = (x1 - x0) / stride, (y1 - y0) / stride
        cx, cy = (x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride
        ix, iy = min(int(cx), w - 1), min(int(cy), h - 1)
        r = max(0, int(gaussian_radius(bw, bh)))
        sigma = (2 * r + 1) / 6
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2))
        hm[i, 0] = np.maximum(hm[i, 0], g)
        hm[i, 0, iy, ix] = 1.0
        wh[i, :, iy, ix] = (bw, bh)
        off[i, :, iy, ix] = (cx - ix, cy - iy)
        mask[i, 0, iy, ix] = 1.0
    return hm, wh, off, mask


def augment(x):
    """Photometric augmentation on the [-1,1] tensor batch + horizontal flip flag."""
    b = x.shape[0]
    gain = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    bias = torch.empty(b, 3, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    return ((x + 1) * gain - 1 + bias).clamp(-1, 1)


def train_detector(rgb, labels, device, epochs, bs=64, init=None, max_lr=3e-3):
    n, H, W, _ = rgb.shape
    hm, wh, off, mask = map(torch.from_numpy, build_targets(labels, n, H, W))
    net = ObstacleDetector().to(device)
    if init:
        net.load_state_dict(torch.load(init, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-4)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps)
    X = to_tensor(rgb)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        net.train()
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            x = augment(X[idx].to(device))
            t_hm, t_wh, t_off, t_m = (a[idx].to(device) for a in (hm, wh, off, mask))
            if torch.rand(1).item() < 0.5:                  # horizontal flip
                x, t_hm, t_wh, t_m = x.flip(-1), t_hm.flip(-1), t_wh.flip(-1), t_m.flip(-1)
                # mirrored centre: x -> W - x, cell ix -> w-1-ix, so offset_x -> 1 - offset_x
                t_off = t_off.flip(-1)
                t_off = torch.stack([(1 - t_off[:, 0]) * t_m[:, 0], t_off[:, 1]], 1)
                x, t_hm, t_wh, t_off, t_m = (a.contiguous() for a in (x, t_hm, t_wh, t_off, t_m))
            p_hm, p_wh, p_off = net(x)
            n_obj = t_m.sum().clamp(min=1)
            l_hm = focal_loss(p_hm, t_hm)
            l_wh = (torch.abs(p_wh - t_wh) * t_m).sum() / n_obj
            l_off = (torch.abs(p_off - t_off) * t_m).sum() / n_obj
            loss = l_hm + 0.1 * l_wh + l_off
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f'  detector epoch {ep + 1:2d}/{epochs}  loss {tot / n:.4f}', flush=True)
    return net


def train_depth(rgb, depth, device, epochs, bs=64, init=None, max_lr=3e-3):
    n = rgb.shape[0]
    net = MonoDepthNet().to(device)
    if init:
        net.load_state_dict(torch.load(init, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-4)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps)
    X = to_tensor(rgb)
    D = torch.from_numpy(np.log(np.clip(depth.astype(np.float32), MonoDepthNet.MIN_D, MonoDepthNet.MAX_D)))[:, None]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        net.train()
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            x = augment(X[idx].to(device))
            d = D[idx].to(device)
            if torch.rand(1).item() < 0.5:
                x, d = x.flip(-1).contiguous(), d.flip(-1).contiguous()
            pred = net(x)
            diff = pred - d
            # scale-invariant term + absolute L1 in log space + edge-aware gradient matching
            l1 = diff.abs().mean()
            si = (diff ** 2).mean() - 0.5 * diff.mean() ** 2
            gx = (diff[..., :, 1:] - diff[..., :, :-1]).abs().mean()
            gy = (diff[..., 1:, :] - diff[..., :-1, :]).abs().mean()
            loss = l1 + si + 0.5 * (gx + gy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f'  depth    epoch {ep + 1:2d}/{epochs}  loss {tot / n:.4f}', flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-train', type=int, default=8000)
    ap.add_argument('--n-val', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-close', type=int, default=4000, help='extra close-range hard examples')
    ap.add_argument('--finetune', action='store_true', help='start from the existing models/*.pt')
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_val, args.epochs, args.n_close = 400, 100, 2, 100
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    t0 = time.time()
    rgb, depth, labels = get_split('train', args.n_train, seed=1)
    get_split('val', args.n_val, seed=2)
    if args.n_close:
        # hard-example mining: close-range views that the generic sampler rarely produces
        rgb_c, depth_c, labels_c = get_split('train_close', args.n_close, seed=3, close_frac=1.0)
        labels_c = labels_c.copy()
        labels_c[:, 0] += len(rgb)
        rgb = np.concatenate([rgb, rgb_c])
        depth = np.concatenate([depth, depth_c])
        labels = np.concatenate([labels, labels_c])
    print(f'dataset ready: {len(rgb)} train frames, {len(labels)} boxes ({time.time() - t0:.0f} s)')

    t0 = time.time()
    det_path, dep_path = os.path.join(MODELS, 'detector.pt'), os.path.join(MODELS, 'depth.pt')
    init_det = det_path if args.finetune else None
    init_dep = dep_path if args.finetune else None
    lr = 1e-3 if args.finetune else 3e-3
    det = train_detector(rgb, labels, device, args.epochs, init=init_det, max_lr=lr)
    torch.save(det.state_dict(), det_path)
    dep = train_depth(rgb, depth, device, args.epochs, init=init_dep, max_lr=lr)
    torch.save(dep.state_dict(), dep_path)
    info = dict(n_train=len(rgb), n_close=args.n_close, finetune=args.finetune, n_boxes=int(len(labels)), epochs=args.epochs, device=device,
                train_seconds=time.time() - t0,
                det_params=sum(p.numel() for p in det.parameters()),
                depth_params=sum(p.numel() for p in dep.parameters()))
    with open(os.path.join(MODELS, 'train_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
