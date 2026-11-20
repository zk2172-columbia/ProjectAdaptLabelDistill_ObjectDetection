# Installation and usage

Run commands from the repository root. For the method overview, see the [README](../README.md).
Datasets and model weights are obtained separately.

## Install

For new setups, use the [environment installer or Docker image](ENVIRONMENT.md).
It selects a compatible CPU/CUDA dependency set, prepares the pinned SAM3 source,
and records installed package versions. The historical CPU lock remains a record
of earlier validation.

## One-command training

Use the [README quickstart](../README.md#getting-started) to run all stages with
`scripts/run_ald.py`. It uses the same stage implementations described below.
The convenience default is pretrained YOLO11n; the legacy dataset configuration
generator below retains its original YOLO11m default.

The command writes generated configurations under the run's `configs/` directory
and the SAM3 training config under `artifacts/run/teacher/configs/`. No generated
config needs to be written into the source checkout, which allows a read-only
container filesystem. Stage failure or timeout stops the process group and
records the failure in `status.json`; use a fresh output directory for a new run.
Per-stage configurations remain available for deliberate manual recovery.

`--stage-timeout SECONDS` places a wall-clock cap on each stage. Student AMP is
off by default to avoid its additional automatic model download; opt in with
`--student-amp` when appropriate. Teacher mixed precision follows its SAM3 config.
Use `--student-model yolo11n.yaml` only when you intentionally want random
initialization, such as a small smoke test. This is not the pretrained default.

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

For offline checkpoint merging, set `sam3.base_checkpoint_path` in the generated
teacher configuration to a trusted local full SAM3 checkpoint. Relative paths resolve
from the configuration file. If omitted, merging obtains the base through Hugging Face.

Adapt once, then reuse the adapted checkpoint for Label and Distill:

```bash
python pipeline/teacher_adaptation/run_teacher_adaptation.py \
  --config "$PWD/configs/local/my_dataset/teacher_adaptation.yaml" --run-name teacher-01
python pipeline/run_pipeline.py \
  --config "$PWD/configs/local/my_dataset/pipeline.yaml" --run-name ald-01
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
COCO AP@100 or Ultralytics' own AP integration. See [evaluation protocol](../pipeline/EVALUATION.md).
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
[third-party notices](../THIRD_PARTY_NOTICES.md). Private integration does not grant a new
public redistribution license for code, weights or datasets. Public release, publication
metadata and final licensing remain outside this private candidate's scope.
