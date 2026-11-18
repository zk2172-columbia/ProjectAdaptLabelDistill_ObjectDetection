# STAC baseline

Reproduction of Sohn et al., *A Simple Semi-Supervised Learning Framework for Object Detection*
(arXiv 2005.04757), on this repo's splits, student and scoring — so its number is directly
comparable to the main pipeline's.

A YOLO detector is trained on the labeled set only, used to pseudo-label the unlabeled pool at a
high confidence threshold τ = 0.9, and a fresh student is trained on both with the unsupervised
half scaled by a single global `lambda_u`. Three stages, each in its own subprocess.

| Stage | Script | Produces |
|---|---|---|
| 1 | `pseudolabels.py --stage teacher` | `runs/<run>/raw_predictions.jsonl` |
| 2 | `pseudolabels.py --stage filter` | `runs/<run>/pseudolabels/`, `pseudolabel_stats.json` |
| 3 | `train_student.py` | `runs/<run>/train/`, `coco_metrics.json` |

This directory is **self-contained**: its code is copied from `pseudo_labeling/` and
`student_training/` and stripped, rather than imported, so it reads as the method it reproduces.
The one exception is scoring, which imports `evaluation/yolo/coco_eval.py` — a reproduction
measured by different code than the arms it is compared against is not a comparison.

## Usage

```bash
python run_stac.py --run-name stac-<dataset> --dry-run   # manifest + preflight, creates nothing
python run_stac.py --run-name stac-<dataset>             # all three stages
python run_stac.py --run-name stac-<dataset> --stages student   # re-run one stage

# re-score a finished run
python train_student.py --eval-only --run-name stac-<dataset>
```

Read `runs/<run>/pseudolabel_stats.json` after stage 2, before spending GPU hours on stage 3: at
τ = 0.9 a teacher with modest recall can leave most of the pool empty.
