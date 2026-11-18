"""Generates every number and visualization used in index.html (Q1-Q4).

Usage:  python run_report.py [--archs pointnet pointnet2]
Writes images/*.gif and results.json.
"""
import argparse
import json
import os

import imageio
import numpy as np
import torch
import pytorch3d
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

from models import build_model
from utils import get_points_renderer, rotate_points, batched_predict, create_dir

CLS_NAMES = ['chair', 'vase', 'lamp']
SEG_COLORS = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                           [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG = 'images'
N_VIEWS = 30


def render_frames(verts, colors, image_size=256):
    """verts (N,3), colors (N,3) -> uint8 frames (N_VIEWS, H, W, 3)."""
    azim = [180 - 12 * i for i in range(N_VIEWS)]
    R, T = look_at_view_transform(dist=3, elev=0, azim=azim, device=DEV)
    cams = FoVPerspectiveCameras(R=R, T=T, fov=60, device=DEV)
    pc = pytorch3d.structures.Pointclouds(
        points=verts.float().unsqueeze(0).repeat(N_VIEWS, 1, 1),
        features=colors.float().unsqueeze(0).repeat(N_VIEWS, 1, 1)).to(DEV)
    rend = get_points_renderer(image_size=image_size, device=DEV)(pc, cameras=cams)
    return (rend[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_gif(frames, name):
    imageio.mimsave(os.path.join(IMG, name), list(frames), duration=1000 / 15, loop=0)
    return name


def cls_gif(verts, name):
    return save_gif(render_frames(verts, torch.tensor([0.25, 0.45, 0.85]).expand(verts.shape[0], 3)), name)


def seg_gif(verts, gt, pred, name):
    """Side-by-side gif: ground truth (left) | prediction (right)."""
    a = render_frames(verts, SEG_COLORS[gt])
    b = render_frames(verts, SEG_COLORS[pred])
    sep = np.full((N_VIEWS, a.shape[1], 4, 3), 200, np.uint8)
    return save_gif(np.concatenate([a, sep, b], axis=2), name)


def load(task, arch):
    m = build_model(task, arch).to(DEV)
    d = './checkpoints/{}{}'.format(task, '' if arch == 'pointnet' else '_' + arch)
    m.load_state_dict(torch.load(os.path.join(d, 'best_model.pt'), map_location=DEV))
    return m.eval()


def subsample(data, n, seed=0):
    ind = np.random.default_rng(seed).choice(data.shape[1], n, replace=False)
    return ind


def main(args):
    create_dir(IMG)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    cls_x = torch.from_numpy(np.load('./data/cls/data_test.npy'))
    cls_y = torch.from_numpy(np.load('./data/cls/label_test.npy')).long()
    seg_x = torch.from_numpy(np.load('./data/seg/data_test.npy'))
    seg_y = torch.from_numpy(np.load('./data/seg/label_test.npy')).long()
    R = {}

    for arch in args.archs:
        tag = '' if arch == 'pointnet' else arch + '_'
        res = R[arch] = {}
        cls_m, seg_m = load('cls', arch), load('seg', arch)

        # ---------------- Q1 ----------------
        cp = batched_predict(cls_m, cls_x, 32, DEV)
        res['cls_acc'] = cp.eq(cls_y).float().mean().item()
        res['cls_per_class_acc'] = {CLS_NAMES[c]: cp[cls_y == c].eq(c).float().mean().item() for c in range(3)}
        conf = torch.zeros(3, 3, dtype=torch.long)
        for g, p in zip(cls_y, cp):
            conf[g, p] += 1
        res['cls_confusion'] = conf.tolist()
        print(arch, 'cls acc', res['cls_acc'], res['cls_per_class_acc'])

        if arch == 'pointnet':
            samples = []
            for c in range(3):  # two random correct samples per class
                idx = torch.where((cls_y == c) & (cp == c))[0].numpy()
                for i in rng.choice(idx, 2, replace=False):
                    samples.append(dict(i=int(i), gt=CLS_NAMES[c], pred=CLS_NAMES[int(cp[i])],
                                        gif=cls_gif(cls_x[i], f'q1_ok_{i}.gif')))
            res['cls_samples'] = samples
            fails = []
            for c in range(3):  # every class: a few failures
                idx = torch.where((cls_y == c) & (cp != c))[0].numpy()
                for i in idx[:2]:
                    fails.append(dict(i=int(i), gt=CLS_NAMES[c], pred=CLS_NAMES[int(cp[i])],
                                      gif=cls_gif(cls_x[i], f'q1_fail_{i}.gif')))
            res['cls_failures'] = fails
        else:  # how does this model do on the objects PointNet got wrong?
            res['on_pointnet_failures'] = [dict(i=d['i'], gt=d['gt'], pred=CLS_NAMES[int(cp[d['i']])])
                                           for d in R['pointnet']['cls_failures']]
            pn_wrong = set(torch.where(R['pointnet']['_cp'] != cls_y)[0].tolist())
            me_wrong = set(torch.where(cp != cls_y)[0].tolist())
            res['cls_wrong_overlap'] = dict(pointnet=len(pn_wrong), this=len(me_wrong),
                                            both=len(pn_wrong & me_wrong))
        res['_cp'] = cp

        # ---------------- Q2 ----------------
        sp = batched_predict(seg_m, seg_x, 16, DEV)
        per_obj = sp.eq(seg_y).float().mean(1)
        res['seg_acc'] = sp.eq(seg_y).float().mean().item()
        res['seg_obj_acc_stats'] = dict(min=per_obj.min().item(), median=per_obj.median().item(),
                                        max=per_obj.max().item())
        print(arch, 'seg acc', res['seg_acc'])
        order = torch.argsort(per_obj)
        if arch == 'pointnet':
            picks = [int(i) for i in rng.choice(len(per_obj), 4, replace=False)] + \
                    [int(order[0]), int(order[1])]
            q2_ids = picks
        else:
            picks = q2_ids  # same objects as PointNet, for side-by-side comparison
        res['seg_samples'] = [dict(i=i, acc=per_obj[i].item(),
                                   gif=seg_gif(seg_x[i], seg_y[i], sp[i], f'q2_{tag}{i}.gif')) for i in picks]

        # ---------------- Q3 (PointNet only) ----------------
        if arch != 'pointnet':
            continue

        # Exp 1: rotation about the z axis (camera viewing axis -> in-plane tilt)
        rot = []
        vis_cls, vis_seg = res['cls_samples'][0]['i'], picks[0]
        for deg in [0, 15, 30, 45, 60, 90, 135, 180]:
            xc = rotate_points(cls_x, deg, 'z')
            xs = rotate_points(seg_x, deg, 'z')
            cpr = batched_predict(cls_m, xc, 32, DEV)
            spr = batched_predict(seg_m, xs, 16, DEV)
            e = dict(deg=deg, cls_acc=cpr.eq(cls_y).float().mean().item(),
                     seg_acc=spr.eq(seg_y).float().mean().item())
            if deg in (30, 60, 90):
                e['cls_vis'] = dict(i=vis_cls, pred=CLS_NAMES[int(cpr[vis_cls])], gt=CLS_NAMES[int(cls_y[vis_cls])],
                                    gif=cls_gif(xc[vis_cls], f'q3_rot{deg}_cls_{vis_cls}.gif'))
                e['seg_vis'] = dict(i=vis_seg, acc=spr[vis_seg].eq(seg_y[vis_seg]).float().mean().item(),
                                    gif=seg_gif(xs[vis_seg], seg_y[vis_seg], spr[vis_seg],
                                                f'q3_rot{deg}_seg_{vis_seg}.gif'))
            rot.append(e)
            print('rot', deg, e['cls_acc'], e['seg_acc'])
        res['q3_rotation'] = rot

        # Exp 2: number of input points
        npts = []
        for n in [10000, 5000, 1000, 500, 100, 50, 10]:
            ind = subsample(cls_x, n, seed=1)
            cpr = batched_predict(cls_m, cls_x[:, ind], 32, DEV)
            spr = batched_predict(seg_m, seg_x[:, ind], 16, DEV)
            sy = seg_y[:, ind]
            e = dict(n=n, cls_acc=cpr.eq(cls_y).float().mean().item(), seg_acc=spr.eq(sy).float().mean().item())
            if n in (1000, 100, 50):
                e['cls_vis'] = dict(i=vis_cls, pred=CLS_NAMES[int(cpr[vis_cls])], gt=CLS_NAMES[int(cls_y[vis_cls])],
                                    gif=cls_gif(cls_x[vis_cls, ind], f'q3_n{n}_cls_{vis_cls}.gif'))
                e['seg_vis'] = dict(i=vis_seg, acc=spr[vis_seg].eq(sy[vis_seg]).float().mean().item(),
                                    gif=seg_gif(seg_x[vis_seg, ind], sy[vis_seg], spr[vis_seg],
                                                f'q3_n{n}_seg_{vis_seg}.gif'))
            npts.append(e)
            print('npts', n, e['cls_acc'], e['seg_acc'])
        res['q3_num_points'] = npts

    with open('results.json', 'w') as f:
        json.dump({a: {k: v for k, v in r.items() if not k.startswith('_')} for a, r in R.items()}, f, indent=1)
    print('wrote results.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archs', nargs='+', default=['pointnet'])
    main(p.parse_args())
