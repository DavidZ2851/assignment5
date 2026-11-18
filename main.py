"""Single entry point for running all code of Assignment 5.

Usage:
    python main.py q1            # train PointNet classification
    python main.py q2            # train PointNet segmentation
    python main.py q4            # train PointNet++ classification + segmentation
    python main.py train         # q1 + q2 + q4
    python main.py report        # every number and gif in index.html (results.json, images/*.gif)
    python main.py plots         # Q3 plots (needs results.json from `report`)
    python main.py all           # train everything, then report + plots

    # Evaluate a single setting; extra flags go straight to eval_cls.py / eval_seg.py:
    python main.py eval_cls --num_points 100
    python main.py eval_cls --rot_deg 45 --rot_axis z --viz --i 524 --arch pointnet2
    python main.py eval_seg --num_points 1000 --i 499
"""
import subprocess
import sys

PY = sys.executable

TRAIN = {
    'q1': [['train.py', '--task', 'cls', '--num_epochs', '60']],
    'q2': [['train.py', '--task', 'seg', '--num_epochs', '60', '--cosine']],
    'q4': [['train.py', '--task', 'cls', '--arch', 'pointnet2', '--num_epochs', '40', '--cosine'],
           ['train.py', '--task', 'seg', '--arch', 'pointnet2', '--num_epochs', '40', '--cosine']],
}
REPORT = [['run_report.py', '--archs', 'pointnet', 'pointnet2']]
PLOTS = [['make_plots.py']]

STAGES = {
    **TRAIN,
    'train': TRAIN['q1'] + TRAIN['q2'] + TRAIN['q4'],
    'report': REPORT,
    'plots': PLOTS,
}
STAGES['all'] = STAGES['train'] + REPORT + PLOTS


def run(cmds):
    for cmd in cmds:
        print('>>', 'python', ' '.join(cmd), flush=True)
        subprocess.run([PY] + cmd, check=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    stage, extra = sys.argv[1], sys.argv[2:]
    if stage in ('eval_cls', 'eval_seg'):
        run([[stage + '.py'] + extra])
    elif stage in STAGES:
        run(STAGES[stage])
    else:
        sys.exit(f'unknown stage {stage!r}; choose from: '
                 + ', '.join(list(STAGES) + ['eval_cls', 'eval_seg']))


if __name__ == '__main__':
    main()
