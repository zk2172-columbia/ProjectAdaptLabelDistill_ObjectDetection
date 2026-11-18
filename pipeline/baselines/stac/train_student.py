#!/usr/bin/env python3
"""train_student.py — STAC stage 3: train the student on labeled + pseudo-labeled data.

    "Apply strong data augmentations to unlabeled images ... compute unsupervised loss and
     supervised loss to train a detector."   — STAC §3.2, steps 3-4

One configuration, no arm registry. The student is a COCO-pretrained YOLOv11m (a fresh one — not
the teacher's weights), trained with:

  - balanced batches, 50% labeled / 50% pseudo-labeled, via STACTrainer's dual loader;
  - single-source mosaics, which the dual loader gives for free;
  - the loss of loss.py: every pseudo-labeled sample's entire detection loss multiplied by
    lambda_u, under a normalizer taken from the unweighted target scores;
  - ultralytics' default augmentation, standing in for STAC's RandAugment-based strong
    augmentation (see README.md, "Departures from the paper").

Everything student_training/train.py carried for its multi-arm comparison is gone: the ARMS registry,
--experiments, --mosaic-avg, --pure-mosaic, --balanced-loader, --label-lambda, the run-label
suffixing they needed, and the "an arm the config does not mention defaults to ENABLED" trap.

Usage
-----
    python train_student.py --config stac_config.yaml --run-name stac-seed42
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# The shared COCOeval implementation — pipeline/evaluation/yolo/coco_eval.py. This reproduction is
# otherwise self-contained by design (see README), and scoring is the deliberate exception: every
# arm in this repo, this one included, must be measured by the SAME code on the SAME ground truth
# or the comparison this directory exists to make is not a comparison. See pipeline/EVALUATION.md.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "evaluation" / "yolo"))
from coco_eval import (  # noqa: E402
    CONF_FLOOR,
    EvalSpec,
    score_checkpoint,
    wandb_summary,
    write_metrics,
)

from trainer import STACTrainer
from utils import class_names, load_config

try:
    import wandb
except ImportError:
    wandb = None

# Ultralytics hyperparameter keys forwarded from the config's student: block, when present.
# Anything absent keeps ultralytics' default — which for augmentation is the point (see README).
_PASSTHROUGH = (
    "epochs", "batch", "imgsz", "optimizer", "device", "workers", "patience", "seed",
    "close_mosaic", "lr0", "lrf", "mosaic", "mixup", "cutmix", "copy_paste",
)


def write_dataset_yaml(path: Path, gt_train: str, pl_train: str, gt_val: str, names: list[str]) -> Path:
    """The yaml ultralytics' check_det_dataset validates and reads nc/names/channels from.

    Its `train:` list names both sources, which is what makes the file an honest description of
    what the run trains on — but STACTrainer.get_dataloader does NOT partition this list to find
    the two halves. It receives them as explicit paths, so a mis-parsed list cannot silently
    produce an unbalanced loader. `val:` is the GT val split, and it IS used: validation runs
    through the stock single loader.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            {"train": [gt_train, pl_train], "val": gt_val, "nc": len(names), "names": names},
            default_flow_style=False,
            sort_keys=False,
        )
    )
    return path


def export_standard_checkpoint(best_pt: Path, dest_dir: Path) -> Path | None:
    """Re-save best.pt with the stock DetectionModel class, so it loads without this package.

    STACDetectionModel is a DetectionModel subclass, so swapping __class__ makes the checkpoint
    loadable with plain ultralytics. Without it the pickle records the class by reference as
    `loss.STACDetectionModel`, which only resolves when this directory is on sys.path — and
    resolves to the WRONG module when it is not, since a PyPI package named `loss` exists. The
    saved dict is otherwise unchanged, so best_standard.pt is a full replacement for best.pt,
    not a stripped inference-only file.
    """
    from ultralytics.nn.tasks import DetectionModel

    best_pt = Path(best_pt)
    if not best_pt.exists():
        return None
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt["model"].__class__ = DetectionModel
    out = best_pt.parent / "best_standard.pt"
    torch.save(ckpt, out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, dest_dir / "best_standard.pt")
    print(f"Saved standard checkpoint: {out}")
    return out


def run_held_out_eval(
    weights: Path,
    run_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    train_overrides: Dict[str, Any],
    wandb_run_id: str | None,
    conf: float | None = None,
    max_det: int | None = None,
) -> Dict[str, Any]:
    """Score the finished student on the held-out split — the number this whole pipeline reports.

    Deliberately NOT the val split ultralytics scores each epoch: that one SELECTS the
    checkpoint, so quoting it would be reporting the split that picked the model. This is the
    same fixed eval split (1,045 images) every other arm in this repo is scored on.

    Reports the five numbers pipeline/EVALUATION.md standardizes on — mAP50, mAP50-95, APs, APm,
    APl — through the shared evaluator, on the same fixed ground truth every other arm in this
    repo is scored against. That shared code path is what makes this reproduction's number
    comparable to the arms it is meant to be compared with.

    Ultralytics' own DetMetrics mAP is deliberately NOT recorded. Its AP integration appends a
    sentinel at (recall=1.0, precision=0.0) and credits the recall range the model never reaches,
    where COCOeval scores that range zero; the gap grows as recall falls, which is exactly the
    regime a semi-supervised arm lands in when pseudo-labels are sparse. Mixing the two in one
    table is not a comparison. See coco_eval.py.

    Scores best_standard.pt, not raw best.pt, so what is measured is byte-identical to what
    downstream code loads.

    `conf` / `max_det` are one-off overrides of the repo-standard detection floor and per-image
    cap (--eval-only only). They write to a separate coco_metrics_*.json so the standard-conf
    result this arm is compared against is never overwritten.
    """
    spec = EvalSpec.from_config(
        {
            "gt_json": cfg["data"]["eval_gt_json"],
            "dataset_yaml": cfg["data"]["eval_dataset_yaml"],
            "split": cfg["data"].get("eval_split", "test"),
        },
        base_dir=Path(cfg["_config_dir"]) if cfg.get("_config_dir") else None,
        # Score at exactly the settings the student trained at.
        imgsz=train_overrides.get("imgsz"),
        batch=train_overrides.get("batch"),
        device=train_overrides.get("device"),
    )

    tag = ""
    overrides: Dict[str, Any] = {}
    if conf is not None:
        overrides["conf"] = conf
        tag += f"_conf{conf}"
    if max_det is not None:
        overrides["max_det"] = max_det
        tag += f"_maxdet{max_det}"
    if overrides:
        spec = dataclasses.replace(spec, **overrides)

    print(f"\n{'-' * 64}\n  Held-out COCOeval on '{spec.split}' of {spec.data_yaml}"
          f"\n  ground truth: {spec.gt_json}\n{'-' * 64}\n", flush=True)

    # Eval artifacts sit with the training run that produced them; the metrics JSON goes to the
    # run root, next to pseudolabel_stats.json, so one directory holds both gates.
    payload = score_checkpoint(weights, spec, out_dir=Path(run_dir), name=f"{spec.split}_eval{tag}")
    out_json = write_metrics(payload, Path(out_dir) / f"coco_metrics{tag}.json")
    print(f"Wrote held-out metrics: {out_json}")

    m = payload["metrics"]
    print("  " + "  ".join(f"{k} {m[k]:.4f}" for k in ("mAP50", "mAP50-95", "APs", "APm", "APl")))

    # Ultralytics' own callback already called wandb.finish() at on_train_end, so the run is
    # reopened by id rather than initialized fresh — a fresh init would create a second run.
    if wandb is not None and wandb_run_id:
        try:
            run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"].get("entity", "gpunewbie"), id=wandb_run_id,
                             resume="allow", reinit=True)
            run.summary.update(wandb_summary(payload))
            run.finish()
        except Exception as exc:  # noqa: BLE001 — a logging failure must not lose the metrics
            print(f"Warning: could not log held-out metrics to W&B: {exc}")

    return payload


def run_eval_only(args: argparse.Namespace) -> None:
    """Re-score a finished run's promoted checkpoint, without training.

    Training already scores itself at the end (see run_held_out_eval); this path exists to
    re-score on demand — after a crash, or when the ground truth is rebuilt.
    """
    cfg = load_config(args.config)
    run_dir_root = Path(cfg["output"]["root"]) / args.run_name
    if not run_dir_root.is_absolute():
        run_dir_root = HERE / run_dir_root

    weights = run_dir_root / "weights" / "best_standard.pt"
    if not weights.is_file():
        raise SystemExit(f"checkpoint not found: {weights}\nTrain the student first.")

    student = cfg["student"]
    payload = run_held_out_eval(
        weights,
        run_dir=run_dir_root / "train",
        out_dir=run_dir_root,
        cfg=cfg,
        train_overrides={k: student[k] for k in ("imgsz", "batch", "device")},
        wandb_run_id=None,
        conf=args.conf,
        max_det=args.max_det,
    )
    payload["run_name"] = args.run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="STAC stage 3: student training")
    parser.add_argument("--config", default="stac_config.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--skip-eval", action="store_true",
                        help="train only; do not score the held-out split afterwards")
    parser.add_argument(
        "--tau", type=float, default=None,
        help=(
            "The tau the pseudo-labels in this run were built with. Training does not read tau — "
            "it reads the labels on disk — but the run's W&B config does, and a stale value there "
            "would mislabel the whole experiment. Pass the same --tau given to "
            "pseudolabels.py --stage filter; run_stac.py forwards one value to both."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume from runs/<run-name>/train/weights/last.pt after an interrupted run. "
            "Ultralytics restores epoch, optimizer, EMA, and scheduler state from the checkpoint "
            "itself; gt_img_path/pl_img_path/lambda_u/balance_ratio are NOT in that checkpoint "
            "(STACTrainer strips them before ultralytics ever sees them), so they still come from "
            "--config exactly as in a fresh run. Logs to a new W&B run — the crashed run's curve "
            "stops where it crashed and this one picks up from the checkpoint epoch."
        ),
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="skip training; re-score this run's existing weights/best_standard.pt",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help=(
            f"--eval-only: one-off override of the eval confidence floor (repo standard: "
            f"{CONF_FLOOR}). Scores to a separate coco_metrics_*.json so the standard-conf "
            "result this arm is compared against is never overwritten."
        ),
    )
    parser.add_argument(
        "--max-det", type=int, default=None,
        help=(
            "--eval-only: one-off override of the per-image detection cap (repo standard: 300, "
            "ultralytics' own default). Only meaningful together with --conf: at the standard "
            "floor max_det is not the binding constraint, but a much lower --conf can saturate "
            "300/image, at which point raising this is the only way to see the rest of the "
            "detection set."
        ),
    )
    args = parser.parse_args()

    if args.eval_only:
        run_eval_only(args)
        return

    cfg = load_config(args.config)
    if args.tau is not None:
        cfg["filter"]["tau"] = float(args.tau)
    student = cfg["student"]
    run_dir_root = Path(cfg["output"]["root"]) / args.run_name
    names = class_names(cfg)

    gt_train = cfg["data"]["gt_train_images"]
    gt_val = cfg["data"]["gt_val_images"]
    pl_images = str(run_dir_root / "pseudolabels" / "images")

    for label, path in (("gt_train_images", gt_train), ("gt_val_images", gt_val),
                        ("pseudolabels/images", pl_images)):
        if not Path(path).is_dir():
            raise SystemExit(
                f"{label} not found: {path}\n"
                + ("Run pseudolabels.py --stage filter first." if "pseudo" in label else "")
            )

    data_yaml = write_dataset_yaml(
        run_dir_root / "stac_dataset.yaml", gt_train, pl_images, gt_val, names
    )

    # One seed for everything: ultralytics applies it as init_seeds(seed + 1 + RANK) with
    # deterministic mode on, and STACTrainer forwards it to BOTH dataloader generators.
    seed = int(cfg["seed"])
    overrides: Dict[str, Any] = {
        "model": student["model"],
        "data": str(data_yaml),
        "project": str(run_dir_root),
        "name": "train",
        "exist_ok": True,
        "seed": seed,
        "deterministic": True,
        # STACTrainer's own keys, stripped before ultralytics sees them.
        "gt_img_path": gt_train,
        "pl_img_path": pl_images,
        "lambda_u": float(student["lambda_u"]),
        "balance_ratio": float(student.get("balance_ratio", 0.5)),
    }
    for key in _PASSTHROUGH:
        if key in student:
            overrides[key] = student[key]
    overrides["seed"] = seed  # the config's top-level seed wins over a stray student: seed

    if args.resume:
        last_pt = run_dir_root / "train" / "weights" / "last.pt"
        if not last_pt.is_file():
            raise SystemExit(f"--resume: no checkpoint at {last_pt}")
        overrides["resume"] = str(last_pt)

    n_gt = len(list(Path(gt_train).iterdir()))
    n_pl = len(list(Path(pl_images).iterdir()))
    print(f"\n{'=' * 64}")
    print(f"  STAC student: {args.run_name}" + ("  (resuming)" if args.resume else ""))
    print(f"  labeled  : {n_gt:,} images  ({gt_train})")
    print(f"  pseudo   : {n_pl:,} images  ({pl_images})")
    print(f"  lambda_u : {overrides['lambda_u']}   balance_ratio: {overrides['balance_ratio']}")
    print(f"  epochs   : {overrides.get('epochs')}   patience: {overrides.get('patience')}"
          f"   batch: {overrides.get('batch')}   seed: {seed}")
    if args.resume:
        print(f"  resuming from: {overrides['resume']}")
    print(f"{'=' * 64}\n", flush=True)

    # Init W&B before the trainer: ultralytics' own callback checks `if not wb.run` and no-ops
    # when a run is already open, so it logs into this one instead of deriving a messy project
    # name from the local output path. It still owns wandb.finish() at on_train_end.
    wandb_run_id = None
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        # Setting enabled: false is NOT enough on its own. Ultralytics' own callback initializes
        # a wandb run whenever `wb.run` is unset, deriving a project name from the local output
        # path — so skipping our init just hands the logging to it. WANDB_MODE=disabled is what
        # actually stops any run from being created.
        os.environ["WANDB_MODE"] = "disabled"
    elif wandb is not None:
        run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity", "gpunewbie"),
            group=args.run_name,
            name=f"{args.run_name}-student",
            job_type="student",
            config={**overrides, "tau": cfg["filter"]["tau"], "nms_iou": cfg["filter"]["nms_iou"]},
            reinit=True,
        )
        wandb_run_id = getattr(run, "id", None)

    trainer = STACTrainer(overrides=overrides)
    trainer.train()
    best = Path(trainer.best)
    run_dir = Path(trainer.save_dir)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_standard = export_standard_checkpoint(best, run_dir_root / "weights")

    if args.skip_eval:
        print("--skip-eval: not scoring the held-out split.")
        return
    if best_standard is None:
        print(f"Warning: no best.pt at {best}; skipping held-out evaluation.")
        return
    run_held_out_eval(best_standard, run_dir, run_dir_root, cfg, overrides, wandb_run_id)


if __name__ == "__main__":
    main()
