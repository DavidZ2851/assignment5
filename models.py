import torch
import torch.nn as nn
import torch.nn.functional as F


def shared_mlp(dims):
    """Per-point MLP implemented with 1x1 Conv1d + BN + ReLU. Input/output: (B, C, N)."""
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Conv1d(dims[i], dims[i + 1], 1), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# ------ Q1: PointNet classification (no transformation blocks) ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.feat = shared_mlp([3, 64, 64, 64, 128, 1024])
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = self.feat(points.transpose(1, 2))      # (B, 1024, N)
        g = torch.max(x, dim=2)[0]                 # symmetric max pool -> (B, 1024)
        return self.head(g)                        # logits


# ------ Q2: PointNet segmentation (no transformation blocks) ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()
        self.local = shared_mlp([3, 64, 64])                 # per-point local features
        self.glob = shared_mlp([64, 64, 128, 1024])          # lifted to global features
        self.head = nn.Sequential(
            shared_mlp([64 + 1024, 512, 256, 128]),
            nn.Conv1d(128, num_seg_classes, 1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        N = points.shape[1]
        local = self.local(points.transpose(1, 2))               # (B, 64, N)
        g = torch.max(self.glob(local), dim=2, keepdim=True)[0]  # (B, 1024, 1)
        x = torch.cat([local, g.expand(-1, -1, N)], dim=1)       # (B, 1088, N)
        return self.head(x).transpose(1, 2)                      # (B, N, num_seg_classes)


# ------ Q4 (bonus): PointNet++ (single-scale grouping) ------
def _gather(x, idx):
    """x: (B, N, C), idx: (B, S, K) long -> (B, S, K, C)."""
    B, S, K = idx.shape
    flat = idx.reshape(B, S * K, 1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, flat).reshape(B, S, K, x.shape[-1])


def _fps_and_group(xyz, feats, npoint, radius, nsample):
    """xyz: (B, N, 3); feats: (B, N, C) or None.
    Returns new_xyz (B, S, 3) and grouped (B, S, K, 3 + C) with xyz relative to the centroid."""
    from pytorch3d.ops import sample_farthest_points, ball_query
    npoint = min(npoint, xyz.shape[1])  # can't sample more centroids than there are points
    new_xyz, _ = sample_farthest_points(xyz, K=npoint, random_start_point=True)
    _, idx, _ = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=False)  # (B, S, K), -1 = empty
    # pad empty slots with the first neighbor (the ball always contains at least the centroid itself)
    first = idx[:, :, :1].expand_as(idx)
    idx = torch.where(idx < 0, first, idx)
    grouped_xyz = _gather(xyz, idx) - new_xyz.unsqueeze(2)
    if feats is not None:
        grouped = torch.cat([grouped_xyz, _gather(feats, idx)], dim=-1)
    else:
        grouped = grouped_xyz
    return new_xyz, grouped


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        # shared per-point MLP applied to every (centroid, neighbour) pair; flattened to (B, C, S*K)
        # (cuDNN BatchNorm2d on permuted (B, C, S, K) tensors is extremely slow, BN1d is equivalent)
        self.mlp = shared_mlp([in_ch + 3] + mlp)

    def forward(self, xyz, feats):
        if self.npoint is None:   # group everything into a single global region
            grouped = xyz.unsqueeze(1) if feats is None else torch.cat([xyz, feats], -1).unsqueeze(1)
            new_xyz = xyz.new_zeros(xyz.shape[0], 1, 3)
        else:
            new_xyz, grouped = _fps_and_group(xyz, feats, self.npoint, self.radius, self.nsample)
        B, S, K, C = grouped.shape
        x = self.mlp(grouped.reshape(B, S * K, C).transpose(1, 2))     # (B, C', S*K)
        x = x.reshape(B, -1, S, K).max(dim=-1)[0]                      # max over neighbours -> (B, C', S)
        return new_xyz, x.transpose(1, 2).contiguous()                 # (B, S, C')


class FeaturePropagation(nn.Module):
    """Upsample features from (xyz2, f2) to xyz1 by inverse-distance weighted 3-NN interpolation."""
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = shared_mlp([in_ch] + mlp)

    def forward(self, xyz1, xyz2, f1, f2):
        from pytorch3d.ops import knn_points, knn_gather
        if xyz2.shape[1] == 1:
            interp = f2.expand(-1, xyz1.shape[1], -1)
        else:
            d, idx, _ = knn_points(xyz1, xyz2, K=3)
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(-1, keepdim=True)
            interp = (knn_gather(f2, idx) * w.unsqueeze(-1)).sum(2)
        x = interp if f1 is None else torch.cat([f1, interp], -1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class pn2_cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256, [256, 512, 1024])
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        xyz, f = self.sa1(points, None)
        xyz, f = self.sa2(xyz, f)
        _, f = self.sa3(xyz, f)
        return self.head(f.squeeze(1))


class pn2_seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.1, 32, 0, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(None, None, None, 256, [256, 512, 1024])
        self.fp4 = FeaturePropagation(1024 + 256, [256, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128 + 3, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Conv1d(128, num_seg_classes, 1))

    def forward(self, points):
        l0 = points
        l1, f1 = self.sa1(l0, None)
        l2, f2 = self.sa2(l1, f1)
        l3, f3 = self.sa3(l2, f2)
        l4, f4 = self.sa4(l3, f3)
        f3 = self.fp4(l3, l4, f3, f4)
        f2 = self.fp3(l2, l3, f2, f3)
        f1 = self.fp2(l1, l2, f1, f2)
        f0 = self.fp1(l0, l1, l0, f1)
        return self.head(f0.transpose(1, 2)).transpose(1, 2)


def build_model(task, arch="pointnet", num_cls_class=3, num_seg_class=6):
    if task == "cls":
        return (cls_model if arch == "pointnet" else pn2_cls_model)(num_cls_class)
    return (seg_model if arch == "pointnet" else pn2_seg_model)(num_seg_class)
