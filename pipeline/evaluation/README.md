# evaluation

COCOeval scoring. Every arm in this repo — YOLO student, SAM 3 teacher, or STAC baseline — is
scored the same way: **mAP50, mAP50-95, APs, APm, APl** on the fixed held-out `eval` split, at
confidence 0 and at most 300 detections per image.

Those three settings (`conf=0.0`, NMS `iou=0.7`, `max_det=300`) are pinned in `yolo/coco_eval.py`
and are deliberately **not** configurable: they decide which detections COCOeval ever sees, and
narrowing the set quietly produces a number that compares against nothing.

| Directory | Scores | Engine |
|---|---|---|
| `yolo/` | every student and baseline arm | `faster_coco_eval.COCOeval_faster`, driven by `ultralytics.YOLO.val()` |
| `sam3/` | the SAM 3 teacher directly | pycocotools' `COCOeval`, via SAM 3's own hydra harness |

Both read the same ground truth — `../data/ground_truths/<dataset>/eval/annotations.coco.json` —
so a teacher number and a student number land in the same table. The engines differ because each
is the one already verified against that model's inference path; the metric definitions are
identical. Ultralytics' own mAP is never reported: it credits precision across recall the model
never reaches.

## Usage

```bash
cd yolo

# one checkpoint
python eval_coco.py --weights ../../weights/my-run/gt_pl_perbox/best_standard.pt

# several, labelled, in one table
python eval_coco.py --target gt_only=../../weights/my-run/gt_only/best_standard.pt \
                    --target perbox=../../weights/my-run/gt_pl_perbox/best_standard.pt

# re-score detections that already exist — no GPU, no inference, no checkpoint
python eval_coco.py --rescore ../../eval_results/my-run/gt_only/predictions.json
```

Scoring the SAM 3 teacher, through SAM 3's own trainer:

```bash
cd sam3
python run_eval.py --config configs/pcatcher_eval.yaml          # adapted teacher
python run_eval.py --config configs/pcatcher_eval_pretrained.yaml   # zero-shot
```

`rescore_dumps.py` re-scores stored SAM 3 prediction dumps on CPU.

Defaults for `--gt` / `--data` / `--split` / imgsz / batch / device come from
`yolo/evaluation_config.yaml`.

## Using it from code

This is how the training entry points score their own runs:

```python
import sys; sys.path.insert(0, "<pipeline>/evaluation/yolo")
from coco_eval import EvalSpec, score_checkpoint, write_metrics

spec = EvalSpec(gt_json=..., data_yaml=..., split="test", imgsz=640, batch=16, device="0")
payload = score_checkpoint("best_standard.pt", spec, out_dir=run_dir)
write_metrics(payload, run_dir / "coco_metrics.json")
```
