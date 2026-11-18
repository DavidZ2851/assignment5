# 16-825 Assignment 5: Point Cloud Classification and Segmentation

PointNet (without transformation blocks) for 3-class classification (chair / vase / lamp) and 6-part chair segmentation, two robustness studies (Q3), and PointNet++ (single-scale grouping) for the locality bonus (Q4).

## Setup

Use the environment from earlier assignments (PyTorch + PyTorch3D, plus `requirements.txt`).

Download the data into the repository root:

```bash
pip install huggingface_hub
hf download learning3dvision/assignment5 a5_data.zip --repo-type dataset --local-dir data_download
unzip data_download/a5_data.zip -d ./
```

This creates `data/cls` and `data/seg`.

## Running everything: `main.py`

`main.py` is the single entry point for all the code in this assignment.

| Command | What it does |
|---|---|
| `python main.py q1` | Train PointNet classification (60 epochs, constant lr) |
| `python main.py q2` | Train PointNet segmentation (60 epochs, cosine lr) |
| `python main.py q4` | Train PointNet++ classification and segmentation (40 epochs, cosine lr) |
| `python main.py train` | `q1` + `q2` + `q4` |
| `python main.py report` | Compute every number and GIF in the report (`results.json`, `images/*.gif`) |
| `python main.py plots` | Q3 accuracy plots (`images/q3_rotation.png`, `images/q3_num_points.png`) |
| `python main.py all` | Train everything, then `report` and `plots` |
| `python main.py eval_cls [flags]` | Evaluate classification once |
| `python main.py eval_seg [flags]` | Evaluate segmentation once |

Flags after `eval_cls` / `eval_seg` go straight to `eval_cls.py` / `eval_seg.py`:

| Flag | Meaning |
|---|---|
| `--arch pointnet2` | Use PointNet++ instead of PointNet |
| `--num_points N` | Evaluate with N points per object (Q3 exp. 2) |
| `--rot_deg D --rot_axis z` | Rotate every test object by D degrees (Q3 exp. 1) |
| `--i IDX` | Object to visualize |
| `--viz` | Save a GIF of object `IDX` (classification) |

Examples:

```bash
python main.py eval_cls
python main.py eval_cls --rot_deg 45 --rot_axis z
python main.py eval_seg --num_points 100 --i 499
python main.py eval_cls --arch pointnet2
```

Trained checkpoints are in `checkpoints/{cls,seg,cls_pointnet2,seg_pointnet2}/best_model.pt`, so the evaluation, `report` and `plots` commands work without retraining.

## Files

| File | Contents |
|---|---|
| `main.py` | Entry point that runs all stages |
| `models.py` | PointNet classification / segmentation and PointNet++ (SSG) models |
| `train.py` | Training loop (random 2500-point subsets per batch, optional cosine lr) |
| `eval_cls.py`, `eval_seg.py` | Test accuracy, rotation and point-count robustness, visualization |
| `run_report.py` | Generates all report numbers and GIFs |
| `make_plots.py` | Q3 plots from `results.json` |
| `utils.py`, `data_loader.py` | Rendering, rotation, batched prediction, data loading |
| `results.json` | All numbers used in the report |
| `logs_txt/` | Training logs |

## Results

| Model | Classification acc | Segmentation acc |
|---|---|---|
| PointNet | 97.17% | 90.17% |
| PointNet++ (SSG) | 97.69% | 91.42% |

All test numbers use the full 10,000 points per object.
