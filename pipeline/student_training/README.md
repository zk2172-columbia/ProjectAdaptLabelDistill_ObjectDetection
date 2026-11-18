# student_training

Stage 3 — train the compact YOLOv11 student on the labeled set plus the confidence-weighted
pseudo-labels.

100 epochs at 640×640, batch 16. Ground-truth boxes contribute at full weight; each pseudo-label
box contributes in proportion to the teacher's confidence in it, and a box below the zero-gating
threshold η is kept for label assignment but given **zero** loss weight — so its region is ignored
rather than trained as background.

**In:** `yolo_data/<run-name>/label/` from `pseudo_labeling/`, plus the GT splits
**Out:** `training_runs/<run-name>/<arm>/` (full ultralytics run dir + `coco_metrics.json`) and
`weights/<run-name>/<arm>/best_standard.pt`

Each arm is scored on the held-out split automatically when it finishes.

## The arms

Selected in `config.yaml`'s `experiments:` block:

| Arm | Trains on | Weighting |
|---|---|---|
| `gt_only` | GT only | none — the limited-label baseline |
| `gt_pl_stock` | GT + PL | none — isolates "does PL data help?" |
| `gt_pl_unweighted_absolute` | GT + PL | one scalar `lambda1` per PL image |
| `gt_pl_perbox` | GT + PL | per-box teacher confidence, with zero-gating — **the paper's method** |

## Usage

```bash
# GT -> YOLO labels, then every enabled arm
python run_student_training.py --config config.yaml --run-name my-run

# one arm only
python train.py --config config.yaml --experiments gt_pl_perbox

# the data prep step on its own
python prepare_data.py --config config.yaml

# the all-human-annotated upper reference dataset
python prepare_data.py --human-annotated --output-dir ../yolo_data/oracle
```
