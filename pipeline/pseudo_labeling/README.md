# pseudo_labeling

Stage 2 — run the adapted teacher once over the unlabeled pool and turn its predictions into
confidence-weighted YOLO labels.

Two steps: SAM 3 inference over every image in the `label` split, then filtering — within-class
NMS at IoU 0.5, a confidence threshold τ = 0.25, and a cross-class score-gap merge — writing YOLO
labels plus each surviving box's teacher confidence.

**In:** `merged_checkpoint.pt` from `teacher_adaptation/`, and the `label` split
**Out:** `yolo_data/<run-name>/label/` — `labels/`, `images/`, `dataset.yaml`, and
`confidence_weights.json` (the per-box weights the student's loss reads)

## Usage

```bash
python run_pseudo_labeling.py --config config.yaml --run-name my-run
```

Or one step at a time:

```bash
python run_inference.py     --config config.yaml   # -> teacher_predictions/<run>/
python filter_and_convert.py --config config.yaml  # -> yolo_data/<run>/label/
```

`mask_low_confidence.py` is an opt-in prototype that paints out low-confidence regions instead of
gating them in the loss. It writes a separate `label_masked/` dataset and never modifies the
originals; it is disabled in `config.yaml`'s `steps:` list.
