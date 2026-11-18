# Adapt, Label, Distill: Confidence-Guided Foundation Model Supervision for Efficient Object Detection

A vision foundation model (SAM 3) is adapted on a small labeled target set, used **offline** to
pseudo-label a much larger unlabeled set, and distilled into a compact YOLOv11 student that is the
only model deployed. Pseudo-labels are weighted by teacher confidence, and uncertain detections are
*zero-gated* — kept during label assignment but given zero loss weight, so they are never learned
as background.

<img src="pipeline/docs/pipeline_overview.png" width="800">

The teacher is used only to build the annotation set and is absent at deployment. On an NVIDIA
Jetson AGX Orin the student runs at **188 FPS** against SAM 3's **2.9 FPS**.

This code is for research use.

## Installation

```bash
git clone --recurse-submodules git@github.com:zk2172-columbia/ProjectPcatcher-NOAA.git
cd ProjectPcatcher-NOAA/Object_detection/Student_Teacher_Pipeline

conda env create -f experiment_files/conda_env/ald_env.yml
conda activate ald

export PRJROOT=$PWD/pipeline
```

One environment carries `sam3`, `ultralytics`, `torch`, `pycocotools` and `faster-coco-eval`
together. `faster-coco-eval` is an *optional* ultralytics extra and every reported metric depends
on it — reinstalling ultralytics without it silently removes the evaluator.

## Data preparation

Each dataset is partitioned into `train` (the labeled set `D_l`, ~5%), `label` (the unlabeled pool
`D_u`), `val` and `eval` (the held-out test split). Splits are written once per dataset and never
regenerated for a run.

```bash
cd $PRJROOT/data

# edit split_config.yaml: source annotations, out_dir, shots, train_frac, seed
python split_dataset.py --config split_config.yaml
```

The split's `ground_truth_export:` block then writes `data/ground_truths/<dataset>/` — the fixed
COCO + YOLO ground truth every number in the paper is scored against. Add the dataset's class list
as `data/classes/<dataset>.yaml`.

## Training

Every config ships as a template with a `<dataset>` placeholder. Replace it before the first run.

### Step 0: set variables

```bash
export DATASET=garbage-classification     # a directory under data/ground_truths/
export RUN=my-run                         # names every output directory for this run
```

### Step 1: adapt the teacher on the labeled set

SAM 3, 20 epochs at 1008×1008, batch 2, AdamW at 8e-5, vision and language backbones frozen.

```bash
cd $PRJROOT/teacher_adaptation
python run_teacher_adaptation.py --config config.yaml --run-name ft-sam3-$DATASET
```

Produces `experiment_files/experiments/<run-name>/merged_checkpoint.pt`. Point
`pseudo_labeling/config.yaml`'s `sam3.checkpoint_path` at it.

### Step 2: generate pseudo-labels for the unlabeled set

The adapted teacher runs once over `D_u`; predictions are NMS'd at IoU 0.5, thresholded at
τ = 0.25, and each surviving box keeps its confidence as a per-box weight.

```bash
cd $PRJROOT/pseudo_labeling
python run_pseudo_labeling.py --config config.yaml --run-name $RUN
```

Produces `yolo_data/$RUN/label/` with YOLO labels and `confidence_weights.json`.

### Step 3: train the student

YOLOv11, 100 epochs at 640×640, batch 16. Ground-truth images contribute at full weight;
pseudo-labeled boxes contribute in proportion to teacher confidence, and boxes below the
zero-gating threshold η = 0.4 contribute nothing.

```bash
cd $PRJROOT/student_training
python run_student_training.py --config config.yaml --run-name $RUN
```

Or run steps 2 and 3 together under one run name:

```bash
cd $PRJROOT && python run_pipeline.py --run-name $RUN
```

## Evaluation

**mAP50, mAP50-95, APs, APm, APl** from COCOeval on the fixed held-out `eval` split, at
confidence 0 and at most 300 detections per image. Those three settings are pinned in
`evaluation/yolo/coco_eval.py` and are deliberately not configurable — they decide which
detections COCOeval ever sees.

Step 3 scores itself when it finishes, writing `coco_metrics.json` into the run directory. To
score a checkpoint on demand:

```bash
cd $PRJROOT/evaluation/yolo
python eval_coco.py --weights ../../weights/$RUN/gt_pl_perbox/best_standard.pt
```

The SAM 3 teacher is scored through its own harness — see [`evaluation/README.md`](evaluation/README.md).

## Baselines

STAC-style self-training (Sohn et al.), sharing the same splits, student and scoring:

```bash
cd $PRJROOT/baselines/stac
python run_stac.py --run-name stac-$DATASET
```

## Results

mAP50:95 on held-out test splits, 5% labeled target data, YOLOv11m student:

| Setting | Garbage | Fruitjes | Floating Waste | VisDrone | Xray | Pig Detection | Avg |
|---|---|---|---|---|---|---|---|
| Labeled only | 30.8 | 60.4 | 36.4 | 12.5 | 73.0 | 60.8 | 45.7 |
| STAC-style | 36.3 | 60.1 | 38.9 | 14.1 | 73.3 | 61.7 | 47.4 |
| **Ours** | **51.6** | **68.1** | **40.0** | **18.5** | 69.8 | 61.7 | **51.6** |
| Full supervision (100%) | 54.1 | 69.9 | 50.3 | 20.3 | 77.1 | 72.6 | 57.3 |

## Citation

```bibtex
@inproceedings{adaptlabeldistill2027,
  title     = {Adapt, Label, Distill: Confidence-Guided Foundation Model Supervision
               for Efficient Object Detection},
  author    = {Anonymous},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgement

- [SAM 3](https://github.com/facebookresearch/sam3) — the foundation teacher, included as a submodule.
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11 student training and inference.
- [STAC](https://github.com/google-research/ssl_detection) — the semi-supervised baseline this is compared against.
- [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) — the COCOeval implementation behind every reported metric.
