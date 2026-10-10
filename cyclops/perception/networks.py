"""Deep networks for monocular perception (PyTorch).

ObstacleDetector  - CenterNet-style anchor-free single-class detector.
                    Predicts a centre heat-map, box size and sub-pixel offset at
                    stride 4.  ~0.2 M parameters -> runs in a few ms on a CPU,
                    the right budget for a companion computer (Raspberry Pi /
                    Jetson class) on a small quadrotor.
MonoDepthNet      - small U-Net that regresses metric log-depth from a single
                    RGB frame (MiDaS-style encoder/decoder, trained on our
                    simulator's ground-truth depth so its output is metric).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(cin, cout, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DWBlock(nn.Module):
    """Depthwise-separable residual block (MobileNet style)."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return F.relu(x + y)


class Encoder(nn.Module):
    def __init__(self, ch=(16, 32, 64, 96)):
        super().__init__()
        self.s1 = nn.Sequential(cbr(3, ch[0], 2), DWBlock(ch[0]))            # /2
        self.s2 = nn.Sequential(cbr(ch[0], ch[1], 2), DWBlock(ch[1]))         # /4
        self.s3 = nn.Sequential(cbr(ch[1], ch[2], 2), DWBlock(ch[2]), DWBlock(ch[2]))  # /8
        self.s4 = nn.Sequential(cbr(ch[2], ch[3], 2), DWBlock(ch[3]), DWBlock(ch[3]))  # /16

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f1, f2, f3, f4


def up_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class ObstacleDetector(nn.Module):
    STRIDE = 4

    def __init__(self, ch=(16, 32, 64, 96), head=48):
        super().__init__()
        self.enc = Encoder(ch)
        self.lat4 = nn.Conv2d(ch[3], head, 1)
        self.lat3 = nn.Conv2d(ch[2], head, 1)
        self.lat2 = nn.Conv2d(ch[1], head, 1)
        self.fuse = nn.Sequential(cbr(head, head), cbr(head, head))
        self.hm = nn.Sequential(cbr(head, head), nn.Conv2d(head, 1, 1))
        self.wh = nn.Sequential(cbr(head, head), nn.Conv2d(head, 2, 1))
        self.off = nn.Sequential(cbr(head, head), nn.Conv2d(head, 2, 1))
        self.hm[-1].bias.data.fill_(-2.19)       # prior p = 0.1

    def forward(self, x):
        _, f2, f3, f4 = self.enc(x)
        p = self.lat4(f4)
        p = up_to(p, f3) + self.lat3(f3)
        p = up_to(p, f2) + self.lat2(f2)
        p = self.fuse(p)
        return self.hm(p), self.wh(p), self.off(p)


class MonoDepthNet(nn.Module):
    MIN_D, MAX_D = 0.3, 40.0

    def __init__(self, ch=(16, 32, 64, 96)):
        super().__init__()
        self.enc = Encoder(ch)
        self.d3 = cbr(ch[3] + ch[2], 64)
        self.d2 = cbr(64 + ch[1], 48)
        self.d1 = cbr(48 + ch[0], 32)
        self.out = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x):
        f1, f2, f3, f4 = self.enc(x)
        y = self.d3(torch.cat([up_to(f4, f3), f3], 1))
        y = self.d2(torch.cat([up_to(y, f2), f2], 1))
        y = self.d1(torch.cat([up_to(y, f1), f1], 1))
        y = up_to(self.out(y), x)
        # sigmoid-bounded log depth
        lo, hi = torch.log(torch.tensor(self.MIN_D)), torch.log(torch.tensor(self.MAX_D))
        return lo + (hi - lo) * torch.sigmoid(y)          # log depth


# ----------------------------------------------------------------------- utils
def to_tensor(rgb_batch):
    """uint8 (N,H,W,3) numpy -> float tensor (N,3,H,W) in [-1, 1]."""
    x = (torch.as_tensor(rgb_batch).float().permute(0, 3, 1, 2) / 127.5 - 1.0).contiguous()
    return x


def gaussian_radius(w, h, min_overlap=0.7):
    """CenterNet heat-map radius for a box of size (w, h) on the output grid."""
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + (b1 ** 2 - 4 * a1 * c1) ** 0.5) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + (b2 ** 2 - 4 * a2 * c2) ** 0.5) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + (b3 ** 2 - 4 * a3 * c3) ** 0.5) / 2
    return max(0.0, min(r1, r2, r3))


def focal_loss(pred_logits, gt):
    """CenterNet penalty-reduced pixel-wise focal loss."""
    p = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


@torch.no_grad()
def decode_detections(hm_logits, wh, off, k=20, thresh=0.3):
    """Heat-map peak extraction (3x3 max-pool NMS). Returns per-image lists of (x0,y0,x1,y1,score)."""
    s = ObstacleDetector.STRIDE
    hm = torch.sigmoid(hm_logits)
    keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
    hm = hm * keep
    n, _, H, W = hm.shape
    scores, idx = hm.view(n, -1).topk(k)
    out = []
    for b in range(n):
        dets = []
        for sc, i in zip(scores[b].tolist(), idx[b].tolist()):
            if sc < thresh:
                break
            y, x = divmod(i, W)
            cx = (x + off[b, 0, y, x].item()) * s
            cy = (y + off[b, 1, y, x].item()) * s
            w = wh[b, 0, y, x].item() * s
            h = wh[b, 1, y, x].item() * s
            dets.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, sc))
        out.append(dets)
    return out
