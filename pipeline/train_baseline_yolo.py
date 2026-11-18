"""train_baseline_yolo.py — stock ultralytics baseline training, no pipeline machinery.

A plain YOLO11m run on a single dataset yaml: no pseudo-labels, no confidence weighting,
ultralytics' own DetectionModel/v8DetectionLoss. Use it for reference points the student training
arms are measured against.

Outputs follow the pipeline layout (see student_training/train.py's _export_standard_checkpoint):

    training_runs/{run_name}/{arm}/          untouched ultralytics run dir
    weights/{run_name}/{arm}/best_standard.pt   promoted checkpoint

The arm defaults to `gt_only` so evaluation/run_evaluation.py --run-name {run_name} finds
the checkpoint where its weights_dir: "../weights/{run_name}" expects it.

Usage
-----
    cd pipeline
    python train_baseline_yolo.py \
        --data yolo_data/garbage-classification-gt/human_annotated_dataset.yaml \
        --run-name baseline-yolo-all-human-annotated
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

try:
    import wandb
except ImportError:
    wandb = None

PIPELINE_DIR = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Stock YOLO baseline training")
    p.add_argument("--data", required=True, help="Dataset yaml")
    p.add_argument("--run-name", required=True, help="training_runs/ and weights/ subdirectory")
    p.add_argument("--arm", default="gt_only", help="Run subdirectory under the run name")
    p.add_argument("--model", default=str(PIPELINE_DIR / "student_training" / "yolo11m.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--wandb-project", default="pcatcher")
    p.add_argument("--wandb-entity", default="gpunewbie")
    p.add_argument("--wandb-name", default=None, help="Defaults to --run-name")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = (PIPELINE_DIR / data).resolve()
    if not data.exists():
        raise SystemExit(f"Dataset yaml not found: {data}")

    project = PIPELINE_DIR / "training_runs" / args.run_name
    weights_dir = PIPELINE_DIR / "weights" / args.run_name / args.arm

    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        optimizer=args.optimizer,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=str(project),
        name=args.arm,
        exist_ok=True,
        plots=True,
    )

    # Init before the trainer runs so ultralytics' wandb callback (which no-ops when a run is
    # already open) logs into this run rather than deriving its own project name from the
    # local output path. It still owns wandb.finish() at on_train_end.
    if not args.no_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.run_name,
            group=args.run_name,
            job_type=args.arm,
            config={**train_kwargs, "model": args.model},
        )

    model = YOLO(args.model)
    model.train(**train_kwargs)

    # Stock DetectionModel checkpoints already load without student_training/ on sys.path, so the
    # promoted copy is a straight copy of best.pt under the standard name.
    best = project / args.arm / "weights" / "best.pt"
    if best.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        promoted = weights_dir / "best_standard.pt"
        shutil.copy2(best, promoted)
        print(f"Copied promoted checkpoint: {promoted}")
    else:
        print(f"Warning: no best.pt at {best}; nothing promoted.")


if __name__ == "__main__":
    main()
