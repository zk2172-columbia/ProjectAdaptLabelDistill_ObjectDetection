# ALD — Adapt, Label, Distill

Private integration candidate for the original ALD paper pipeline. Adapt a SAM3 teacher,
label the unlabeled pool offline, then train a confidence-weighted YOLO11 student.
Only the student is required at deployment.

This tree is prepared for private testing, not approved for public distribution. The
source snapshot and fixes are described in [the integration notes](docs/PRIVATE_INTEGRATION.md).
No dataset or model checkpoint is included.

## Install (Linux, Python 3.12)

Create a dedicated environment; do not install into a running research environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.2.0 setuptools==84.1.0
# CPU verification:
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
# CUDA installation must be validated separately against the selected PyTorch wheel and host driver.
python -m pip install -r environments/requirements-student.txt
```

For Adapt and Label:

```bash
python scripts/setup_sam3.py
python -m pip install -r environments/requirements-teacher.txt
python -m pip install -e ./SAM3
```

SAM3 is reconstructed from public commit `f6e51f59500a87c576c2df2323ce56b9fd7a12de`
and `patches/sam3-image-training.patch`. The patch includes the original image-training
and optional logging changes, plus CPU-safe device handling and modern resource loading. It does not replace
SAM3 with the latest upstream. The setup command refuses to modify an existing checkout.
Video inference is outside this image-only ALD workflow.

For an exact CPU dependency installation, use `python -m pip install -r environments/verified-cpu.lock.txt` before the editable SAM3 install.
`environments/verified-cpu.lock.txt` records the tested Linux CPU environment. The older
Conda export under `experiment_files/conda_env/` is historical provenance, not the supported
installation path. A fresh CUDA training run has not been certified by the CPU checks.

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/check_entrypoints.py
python scripts/smoke_student.py --output /tmp/ald-student-smoke-01
python scripts/smoke_pipeline.py --output /tmp/ald-pipeline-smoke-01
```

Use fresh output directories. The student smoke trains for one epoch on generated images
without pretrained weights, exports a portable checkpoint, and scores a held-out synthetic
split. The pipeline smoke runs the actual stage CLIs using synthetic cached teacher detections.
It is a plumbing test, not an accuracy benchmark.

With an existing trusted local SAM3 checkpoint, the optional heavier check is:

```bash
python scripts/smoke_teacher.py --checkpoint /path/to/sam3.pt --output /tmp/ald-teacher-smoke-01
```

No test downloads a model checkpoint or enables online tracking. Obtain base weights through
the official SAM3 access process and record their SHA256 before use. Python model checkpoints
must be trusted; never load arbitrary third-party pickle files.

## Configure a dataset

Prepare four disjoint COCO partitions: `train/`, `label/`, `val/`, `eval/`. Each contains its
images and `_annotations.coco.json`. `train` is the labeled budget; `label` is the unlabeled
pool. The class YAML contains `classes: [{id: 0, name: ..., text_prompt: ...}, ...]`.
Do not regenerate paper splits while tuning the method.

```bash
python scripts/configure_dataset.py \
  --split-root /data/my_dataset \
  --classes-file pipeline/data/classes/my_dataset.yaml \
  --output configs/local/my_dataset \
  --work-root /work/ald/my_dataset \
  --teacher-checkpoint /work/ald/my_dataset/teacher-01/teacher/merged_checkpoint.pt
```

The command validates input paths and filename-level split overlap, then writes local
configuration files. It does not train or alter source data. Patient/video/scene independence
must be established in the dataset split itself; a filename check cannot prove that.

Adapt once, then reuse the adapted checkpoint for Label and Distill:

```bash
python pipeline/teacher_adaptation/run_teacher_adaptation.py \
  --config configs/local/my_dataset/teacher_adaptation.yaml --run-name teacher-01
python pipeline/run_pipeline.py \
  --config configs/local/my_dataset/pipeline.yaml --run-name ald-01
```

The default student arm is `gt_pl_perbox`, with `lambda1=0.5`, zero-gate `0.5`, and the
original combined loader. The default pseudo-label cutoff is `0.25`, within-class NMS is
`0.3`, and optional image masking is disabled. These are explicit configuration values;
the cached main-table run evidence is recorded separately in `results/`.
Balanced-loader, pure-mosaic and masking variants remain opt-in research paths.

A gated box remains in target assignment. Loss is suppressed for anchors assigned to that
box; other background anchors still receive classification loss. This is not a guarantee
that all pixels inside an uncertain object are excluded from background supervision.

W&B and optional online trackers are disabled by default. To enable W&B, install it separately
and explicitly configure `wandb.enabled`, project and entity. OS power-management commands
are not part of the release pipeline.

## Evaluation and recorded results

The fixed protocol uses confidence `0.0`, detector NMS IoU `0.7`, per-image `max_det=300`,
COCO AP at maxDets `300`, and AR at maxDets `100`. Do not mix these values with default
COCO AP@100 or Ultralytics' own AP integration. See [evaluation protocol](pipeline/EVALUATION.md).
Required evaluation failure returns a nonzero exit and records failure status.
Re-evaluation must use a fresh prediction directory to avoid reading stale detections.

The six original YOLO11m ALD cached predictions were re-scored on 5090 with unchanged GT hashes.
All six matched the saved metrics exactly. This validates the evaluator against recorded
predictions, not a new teacher/student training reproduction.

| Dataset | Recorded mAP50:95 (%) |
|---|---:|
| Garbage | 51.638083 |
| Fruitjes | 68.128132 |
| Floating Waste | 39.963739 |
| VisDrone | 18.524274 |
| Xray | 69.798432 |
| Pig Detection | 61.705138 |

Exact values, input hashes and test scope are in `results/paper_cached_regression.json`.
The original baseline and full-supervision runs remain separate from this six-cell check.

## Attribution and distribution

Original ALD source: `zk2172-columbia/ProjectPcatcher-NOAA`, subtree
`Object_detection/Student_Teacher_Pipeline`, parent commit `6489372b...` plus the audited
working-tree delta. Recent source commits are attributed to `anrunwang`.

SAM3 retains its SAM License; Ultralytics retains AGPL-3.0. See
[third-party notices](THIRD_PARTY_NOTICES.md). Private integration does not grant a new
public redistribution license for code, weights or datasets. Public release, publication
metadata and final licensing remain outside this private candidate's scope.
