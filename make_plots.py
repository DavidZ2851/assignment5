"""Q3 accuracy plots from results.json -> images/q3_rotation.png, images/q3_num_points.png"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'      # categorical slots 1 & 2 (validated pair)
INK, MUTED, GRID = '#1d1d1f', '#5f6368', '#e6e6e6'
r = json.load(open('results.json'))['pointnet']


def plot(xs, cls, seg, xlabel, path, logx=False, xticks=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(xs, cls, color=BLUE, lw=2, marker='o', ms=6, mec='white', mew=1.5, label='Classification')
    ax.plot(xs, seg, color=ORANGE, lw=2, marker='s', ms=6, mec='white', mew=1.5, label='Segmentation')
    # direct labels at the right-most point of each series
    for ys, name, col in [(cls, 'cls', BLUE), (seg, 'seg', ORANGE)]:
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    if logx:
        ax.set_xscale('log')
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel('Test accuracy', color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, facecolor='white')
    plt.close(fig)


rot = r['q3_rotation']
plot([e['deg'] for e in rot], [e['cls_acc'] for e in rot], [e['seg_acc'] for e in rot],
     'Rotation about z (degrees)', 'images/q3_rotation.png', xticks=[e['deg'] for e in rot])
npt = sorted(r['q3_num_points'], key=lambda e: e['n'])
plot([e['n'] for e in npt], [e['cls_acc'] for e in npt], [e['seg_acc'] for e in npt],
     'Points per object (log scale)', 'images/q3_num_points.png', logx=True, xticks=[e['n'] for e in npt])
print('ok')
