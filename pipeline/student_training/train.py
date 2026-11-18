"""train.py — student training Stage A: run GT-only, GT+PL-unweighted, GT+PL-weighted experiments.

Usage
-----
    cd pipeline/student_training
    python train.py --config config.yaml

    # Override specific fields:
    python train.py --config config.yaml --epochs 50 --batch 8
    python train.py --config config.yaml --experiments gt_pl_perbox  # single exp
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

# The shared COCOeval implementation — pipeline/evaluation/yolo/coco_eval.py. Reached the same way
# run_teacher_adaptation.py reaches pipeline_common.py. Every arm in this repo, including the paper
# reproductions, scores through this one module; see pipeline/EVALUATION.md.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation" / "yolo"))
from coco_eval import EvalSpec, score_checkpoint, wandb_summary, write_metrics  # noqa: E402

from utils import (
    create_combined_dataset_yaml,
    dataset_train_paths,
    load_config,
    maybe_shutdown,
)
from trainer import (
    BalancedDualDetectionTrainer,
    Phase2DetectionTrainer,
    Phase2Spec,
    PureMosaicDetectionTrainer,
)

try:
    import wandb
except ImportError:
    wandb = None

# Augmentation keys forwarded from the config training: block to ultralytics. Decision (a):
# the box-count-changing augs are zeroed for ALL experiments so the per-image vs per-box
# comparison shares identical augmentation. See README.md, "Augmentation".
_AUG_KEYS = (
    "mosaic", "mixup", "cutmix", "copy_paste",
    "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
    "hsv_h", "hsv_s", "hsv_v",
)


@dataclass(frozen=True)
class Arm:
    """One experiment arm: which data it trains on, plus how its weights are built and applied.

    `spec` (see trainer.py) carries the dataset flags and loss config together. `lambda1` and
    `zerogate` are DEFAULTS — the config's experiments: block overrides them per run.
    """

    data: str                        # "gt" | "combined"
    spec: Phase2Spec
    use_weights: bool = True         # pass weights_json_path / lambda1 at all
    lambda1: float = 0.5
    zerogate: float = 0.0


# ── The experiment arms ───────────────────────────────────────────────────────────────────
# Adding an arm is one entry here plus a block in config.yaml. Note that _should_run
# defaults `enabled` to TRUE for any arm the config does not mention, so an omitted block means
# the arm runs silently — every arm below has an explicit block.
ARMS: dict[str, Arm] = {
    # Stock ultralytics DetectionModel / v8DetectionLoss, not ours weighting nothing, so the
    # baselines cannot drift with our loss code.
    "gt_only": Arm(
        data="gt",
        spec=Phase2Spec(
            per_image=False, loc_weight_src="none", cls_weight_src="none", stock_loss=True,
        ),
        use_weights=False,
    ),
    "gt_pl_stock": Arm(
        data="combined",
        spec=Phase2Spec(
            per_image=False, loc_weight_src="none", cls_weight_src="none", stock_loss=True,
        ),
        use_weights=False,
    ),

    # Every PL image carries the same weight lambda1, so lambda1 is a clean global knob.
    # mosaic_w_image="average" because a mosaic can mix GT tiles at 1.0 with PL tiles at lambda1.
    "gt_pl_unweighted_absolute": Arm(
        data="combined", spec=Phase2Spec(mosaic_w_image="average"),
    ),

    # Per-box w_sam3 instead of a per-image mean, plus the zero gate: a PL box below `zerogate`
    # contributes no loss while its pixels stay in the image.
    "gt_pl_perbox": Arm(
        data="combined",
        spec=Phase2Spec(
            per_box=True, per_image=False,
            loc_weight_src="box", cls_weight_src="box",
        ),
    ),
}


def _build_base_overrides(cfg: dict, extra: dict | None = None) -> dict:
    """Build the ultralytics overrides dict from config.yaml sections."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    overrides: dict = {
        "model": f"{model_cfg['architecture']}.pt",
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch"],
        "imgsz": train_cfg["imgsz"],
        "optimizer": train_cfg["optimizer"],
        "device": train_cfg["device"],
        # Shared by every arm so they differ only in the thing under test, not in weight init,
        # augmentation sampling or data order. Ultralytics applies it as
        # init_seeds(seed + 1 + RANK, deterministic=...) at the start of each run.
        "seed": train_cfg.get("seed", 0),
        "workers": train_cfg["workers"],
        "patience": train_cfg["patience"],
        "freeze": model_cfg.get("freeze_backbone", 0),
        "project": str(Path(__file__).parent / out_cfg["project"]),
        "exist_ok": True,
    }
    # Periodic numbered checkpoints (epoch10.pt, epoch20.pt, ...) alongside the last.pt/best.pt
    # ultralytics already writes every epoch. -1 (ultralytics' default) is off. Forwarded only
    # when the config names it, so a config written before this key existed is unchanged.
    if "save_period" in train_cfg:
        overrides["save_period"] = train_cfg["save_period"]
    # Forward augmentation overrides present in the training: block (decision (a) — shared
    # across all experiments). Keys absent from the config keep ultralytics' defaults.
    for k in _AUG_KEYS:
        if k in train_cfg:
            overrides[k] = train_cfg[k]
    if extra:
        overrides.update(extra)
    return overrides


def _export_standard_checkpoint(best_pt: Path, weights_dir: Path | None, name: str) -> Path | None:
    """Re-save best.pt with the standard DetectionModel class. Returns the path written.

    Phase2DetectionModel is a DetectionModel subclass, so swapping __class__ makes the
    checkpoint loadable without student_training/loss.py on sys.path. (Without this the pickle records
    the class by reference as `loss.Phase2DetectionModel`, which only resolves when student_training/ is
    on sys.path — and resolves to the WRONG module when it is not, since a PyPI package named
    `loss` exists.) The saved dict is otherwise unchanged, so best_standard.pt is a full
    replacement for best.pt, not a stripped inference-only file.

    If weights_dir is set, the resulting best_standard.pt is also copied there
    under <weights_dir>/<name>/ — the promoted-checkpoint tree kept separate
    from the full training_runs/ artifact tree.
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
    print(f"Saved standard checkpoint: {out}")

    if weights_dir is not None:
        promoted_dir = weights_dir / name
        promoted_dir.mkdir(parents=True, exist_ok=True)
        promoted = promoted_dir / "best_standard.pt"
        shutil.copy2(out, promoted)
        print(f"Copied promoted checkpoint: {promoted}")
    return out


def run_test_eval(
    name: str,
    weights: Path,
    save_dir: Path,
    eval_cfg: dict,
    train_overrides: dict,
    weights_dir: Path | None = None,
    wandb_cfg: dict | None = None,
    wandb_run_id: str | None = None,
) -> dict | None:
    """Score one finished arm's best checkpoint on the held-out split with COCOeval.

    Reports the five numbers pipeline/EVALUATION.md standardizes on — mAP50, mAP50-95, APs, APm,
    APl — via the shared evaluator in pipeline/evaluation/yolo/coco_eval.py. Ultralytics' own
    DetMetrics numbers are deliberately not recorded: the two use different AP integration
    conventions and disagree by an amount that grows as recall falls, so a table mixing them is
    not a comparison. See coco_eval.py's docstring for the measured size of that gap.

    Runs on `evaluation.gt_json` + `evaluation.dataset_yaml`, a FIXED pair shared by every arm of
    every sweep — the point is one number per arm on one common yardstick, so it deliberately does
    NOT follow the arm's own training data. That is also why it is separate from the val split
    ultralytics scores each epoch on: that one selects the checkpoint (and for a GT+PL arm is
    drawn from the same combined data the arm trained on), so quoting it as the arm's result would
    be scoring the model on the split that picked it.

    The checkpoint evaluated is best_standard.pt — the promoted export, not raw best.pt — so what
    gets scored is byte-identical to what downstream code loads, and loading it needs no student_training/
    modules on sys.path.

    Metrics are written to <save_dir>/coco_metrics.json (copied next to the promoted weights) and
    pushed to the arm's W&B run summary under 'eval/'. The W&B run is already CLOSED by this point
    — ultralytics' callback calls wandb.finish() at on_train_end — so it is reopened by id with
    resume="allow" rather than initialized fresh, which would make a second run.
    """
    # imgsz/batch/device default to what the arm TRAINED with, so a config that changes them does
    # not silently score at a different resolution than it trained at. conf/iou/max_det are NOT
    # configurable — they decide which detections COCOeval sees. See EvalSpec.from_config.
    spec = EvalSpec.from_config(
        eval_cfg,
        imgsz=train_overrides.get("imgsz"),
        batch=train_overrides.get("batch"),
        device=train_overrides.get("device"),
    )

    print(f"\n{'-' * 64}")
    print(f"  Evaluating {name} on '{spec.split}' of {spec.data_yaml}")
    print(f"  ground truth: {spec.gt_json}")
    print(f"  weights: {weights}")
    print(f"{'-' * 64}\n", flush=True)

    payload = score_checkpoint(
        weights, spec,
        # Inside the arm's own run directory, so the eval artifacts sit with the run that produced
        # them instead of accumulating in ultralytics' default runs/detect/val*.
        out_dir=Path(save_dir), name=f"{spec.split}_eval",
    )
    payload["run"] = name

    out_json = write_metrics(payload, Path(save_dir) / "coco_metrics.json")
    print(f"Wrote held-out metrics: {out_json}")
    if weights_dir is not None:
        promoted_dir = Path(weights_dir) / name
        promoted_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, promoted_dir / "coco_metrics.json")

    m = payload["metrics"]
    print("  " + "  ".join(f"{k} {m[k]:.4f}" for k in ("mAP50", "mAP50-95", "APs", "APm", "APl")))

    if wandb_cfg and wandb_cfg.get("enabled", True) and wandb is not None and wandb_run_id:
        try:
            run = wandb.init(
                project=wandb_cfg.get("project", "pcatcher"),
                entity=wandb_cfg.get("entity", "gpunewbie"),
                id=wandb_run_id,
                resume="allow",
                reinit=True,
            )
            # summary, not log(): this is one final measurement per run, not a time series, and
            # summary keys are what W&B's run table and sweep comparisons sort on.
            run.summary.update(wandb_summary(payload))
            run.finish()
        except Exception as exc:  # noqa: BLE001 — a logging failure must not fail the sweep cell
            print(f"Warning: could not log held-out metrics to W&B: {exc}")

    return payload


def run_experiment(
    name: str,
    overrides: dict,
    weights_dir: Path | None,
    wandb_cfg: dict | None = None,
    wandb_group: str | None = None,
    trainer_cls: type[Phase2DetectionTrainer] = Phase2DetectionTrainer,
    eval_cfg: dict | None = None,
) -> None:
    """Run one experiment. `name` is the RUN LABEL (arm, plus a loader suffix when applicable).

    trainer_cls selects how the train data is assembled: Phase2DetectionTrainer draws each batch
    from one shuffled combined dataset (mosaics may mix GT and PL tiles),
    PureMosaicDetectionTrainer keeps that shuffle but splits the dataset by source so mosaics never
    mix, and BalancedDualDetectionTrainer additionally fixes the GT:PL ratio of every batch.
    Everything else about the arm is identical in all three.
    """
    print(f"\n{'=' * 64}")
    print(f"  student training Experiment: {name}  [{overrides['spec']}]")
    print(f"  Loader: {trainer_cls.__name__}")
    print(f"{'=' * 64}\n")

    # Init explicitly, before the trainer runs, with our own project/group/name.
    # Ultralytics' own wandb callback (on_pretrain_routine_start) checks `if not
    # wb.run` and no-ops when a run is already open, so it logs into this one
    # instead of starting its own (which would otherwise derive a messy project
    # name from the local output.project path). It still owns wandb.finish() at
    # on_train_end, so we don't call that ourselves.
    wandb_run_id = None
    if wandb_cfg and wandb_cfg.get("enabled", True) and wandb is not None:
        wandb_name = f"{wandb_group}-{name.replace('_', '-')}" if wandb_group else name
        run = wandb.init(
            project=wandb_cfg.get("project", "pcatcher"),
                entity=wandb_cfg.get("entity", "gpunewbie"),
            group=wandb_group,
            name=wandb_name,
            job_type=name,
            config=overrides,
            reinit=True,
        )
        # Kept so the post-training eval can reopen THIS run to log its metrics: ultralytics
        # finishes the run at on_train_end, and a fresh init afterwards would be a second run.
        wandb_run_id = getattr(run, "id", None)

    trainer = trainer_cls(overrides=overrides)
    trainer.train()
    best_standard = _export_standard_checkpoint(trainer.best, weights_dir, name)
    save_dir = Path(trainer.save_dir)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # After the trainer is freed, so the eval does not contend with training's GPU memory. A
    # failure here is reported but does not fail the arm — the checkpoint is already on disk and
    # a sweep of 30 cells should not die on a scoring problem; it can be re-scored from the
    # promoted checkpoint at any time.
    if eval_cfg is not None and eval_cfg.get("enabled", True):
        if best_standard is None:
            print(f"Warning: no best_standard.pt for {name}; skipping held-out COCOeval.")
        else:
            try:
                run_test_eval(
                    name, best_standard, save_dir, eval_cfg, overrides,
                    weights_dir=weights_dir, wandb_cfg=wandb_cfg, wandb_run_id=wandb_run_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"!! {name}: post-training evaluation failed: {exc}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def _forwarded_flags(args) -> list[str]:
    """The flags a per-arm child needs to reproduce this invocation.

    Deliberately omits three: --experiments (the caller supplies exactly one arm), --shutdown
    (the parent owns it, or the machine would go down after the first arm) and --isolate-arms
    (the child must run its single arm in-process, or it would recurse forever).
    """
    out = ["--config", args.config]
    if args.epochs is not None:
        out += ["--epochs", str(args.epochs)]
    if args.batch is not None:
        out += ["--batch", str(args.batch)]
    if args.save_period is not None:
        out += ["--save-period", str(args.save_period)]
    for flag, enabled in (
        ("--skip-test-eval", args.skip_test_eval),
        ("--label-lambda", args.label_lambda),
        ("--resume", args.resume),
        ("--overwrite", args.overwrite),
        ("--balanced-loader", args.balanced_loader),
        ("--pure-mosaic", args.pure_mosaic),
    ):
        if enabled:
            out.append(flag)
    return out


def run_arms_isolated(args, names: list[str]) -> None:
    """Run each arm in its own subprocess, sequentially.

    `del trainer; gc.collect(); torch.cuda.empty_cache()` in run_experiment returns cached blocks
    to the driver but does NOT tear down the CUDA context — across arms in one process the context,
    its allocator state and any driver-side state are all inherited. A process boundary is the only
    thing that actually destroys them, so this is what to reach for when the question is whether an
    arm is affected by whatever ran before it.

    Arms still run ONE AT A TIME on the one GPU: this changes process lifetime, not concurrency.

    A failing arm aborts the rest, which is what running them in one process already did (an
    exception out of trainer.train() takes down the whole invocation). Only the eval step is
    exempt, and run_experiment already swallows that on its own.
    """
    script = Path(__file__).resolve()
    for i, name in enumerate(names, 1):
        cmd = [sys.executable, str(script), *_forwarded_flags(args), "--experiments", name]
        print(f"\n{'=' * 64}\n  [isolate-arms {i}/{len(names)}] {name}\n  {' '.join(cmd)}\n{'=' * 64}",
              flush=True)
        result = subprocess.run(cmd, cwd=script.parent)
        if result.returncode != 0:
            raise SystemExit(
                f"arm '{name}' exited {result.returncode} ({i} of {len(names)}); "
                f"not starting the remaining {len(names) - i}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="student training Stage A training experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(ARMS),
        default=None,
        help="Run only specified experiments (default: all enabled in config)",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help=(
            "Do not score the finished arm on the config's evaluation: split. Training "
            "and checkpoint promotion are unaffected; the arm simply produces no coco_metrics.json."
        ),
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        help=(
            "Write a numbered checkpoint every N epochs (epoch10.pt, epoch20.pt, ...), on top of "
            "the last.pt/best.pt ultralytics already writes every epoch. Overrides the config's "
            "training.save_period. -1 disables. Rollback points only — recovering an interrupted "
            "run is --resume's job, not this one's."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted arm from its own weights/last.pt instead of starting over. "
            "Safe for this pipeline specifically: ultralytics' check_resume replaces trainer.args "
            "from the checkpoint, but every student training knob (spec, lambda1, zerogate, "
            "weights_json_path, pl_img_path, balance_ratio) is popped out of `overrides` BEFORE "
            "super().__init__ runs — see trainer._PHASE2_KEYS and trainer._BALANCED_KEYS "
            "— so the loss and loader config cannot be clobbered by the restore. An arm with no "
            "last.pt starts normally, so --resume is safe to pass to a whole chain."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permit training into a run directory that already holds a results.csv. Off by "
            "default because train.py sets exist_ok: True and ultralytics APPENDS to results.csv "
            "— a re-run into a finished directory silently interleaves two runs' epoch histories "
            "rather than failing. Use --resume to continue an interrupted run; use this only to "
            "deliberately train over one."
        ),
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after all experiments complete (overrides config shutdown_on_completion)",
    )
    parser.add_argument(
        "--isolate-arms",
        action="store_true",
        help=(
            "Run each selected arm in its own subprocess instead of looping over them in this "
            "one. Arms still run sequentially on the single GPU; what changes is that every arm "
            "starts from a fresh CUDA context rather than inheriting the previous arm's. "
            "empty_cache() between arms does not do this — it frees cached blocks but keeps the "
            "context. Off by default: it changes nothing about what is trained, only process "
            "lifetime, and the in-process path stays the tested one."
        ),
    )
    parser.add_argument(
        "--label-lambda",
        action="store_true",
        help=(
            "Append '-lam<lambda1>' to every weighted arm's run label, so the SAME arm can be run "
            "at several lambda1 values under one output.project without colliding — each gets its "
            "own run directory, W&B run and promoted weights. Off by default, so existing run "
            "directories keep their bare arm names. Arms that carry no weights (gt_only, "
            "gt_pl_stock) have no lambda1 and are never suffixed. See sweep_lambda.py."
        ),
    )
    parser.add_argument(
        "--balanced-loader",
        action="store_true",
        help=(
            "Draw every batch from TWO single-source loaders at a fixed GT:PL ratio (STAC-style) "
            "instead of one shuffled combined dataset, for every GT+PL arm. Also makes each "
            "mosaic single-source. Runs land under '<arm>-balanced'. An individual arm's "
            "`balanced_loader:` in experiments: overrides this. See data.py."
        ),
    )
    parser.add_argument(
        "--pure-mosaic",
        action="store_true",
        help=(
            "Split the combined dataset by source so a mosaic never mixes GT and PL tiles, while "
            "leaving batch composition at the default binomial shuffle, for every GT+PL arm. "
            "Isolates the mosaic-purity half of what --balanced-loader changes; the two are "
            "mutually exclusive. Runs land under '<arm>-puremosaic'. An individual arm's "
            "`pure_mosaic:` in experiments: overrides this. See trainer.py."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    # ── Paths ────────────────────────────────────────────────────────────────
    pl_cfg = cfg.get("pl_data", {})
    gt_cfg = cfg.get("gt", {})
    out_cfg = cfg.get("output", {})

    def _resolve(p: str | None) -> str | None:
        if p is None:
            return None
        resolved = Path(p) if Path(p).is_absolute() else (config_dir / p).resolve()
        return str(resolved)

    # Held-out split every finished arm is scored on, via COCOeval. Absent block = no eval,
    # so configs written before this existed keep behaving exactly as they did.
    eval_cfg = cfg.get("evaluation")
    if eval_cfg is not None and args.skip_test_eval:
        eval_cfg = None
    if eval_cfg is not None:
        eval_cfg = {
            **eval_cfg,
            "dataset_yaml": _resolve(eval_cfg.get("dataset_yaml")),
            "gt_json": _resolve(eval_cfg.get("gt_json")),
        }
        # Checked BEFORE any training starts: a typo'd path is worth one line now, not hours of
        # GPU followed by a failed eval on every arm. Covers the COCO ground truth as well as the
        # dataset yaml — scoring needs both, and the GT is the newer of the two requirements.
        if eval_cfg.get("enabled", True):
            EvalSpec.from_config(eval_cfg).preflight()

    gt_yaml      = _resolve(gt_cfg.get("dataset_yaml"))
    pl_yaml      = _resolve(pl_cfg.get("dataset_yaml"))
    weights_json = _resolve(pl_cfg.get("weights_json_path"))
    weights_dir  = _resolve(out_cfg.get("weights_dir"))
    weights_dir  = Path(weights_dir) if weights_dir else None

    if gt_yaml and not Path(gt_yaml).exists():
        print(f"Warning: gt.dataset_yaml not found at {gt_yaml}.")
        print("  Run `python prepare_data.py --config config.yaml` first.")
        print("  Skipping experiments that require GT data.")
        gt_yaml = None

    # Resolve combined dataset yaml (GT + PL). Both keys live at the config's top
    # level (not under output:), so read them from cfg, not out_cfg.
    combined_yaml = cfg.get("combined_dataset_yaml")
    if combined_yaml is None and gt_yaml and pl_yaml:
        combined_out = _resolve(cfg.get("combined_dataset_yaml_output", "combined_dataset.yaml"))
        combined_yaml = str(create_combined_dataset_yaml(gt_yaml, pl_yaml, Path(combined_out)))
        print(f"Created combined dataset yaml: {combined_yaml}")
    elif combined_yaml is None and pl_yaml:
        # No GT yaml — use PL yaml for combined experiments
        combined_yaml = pl_yaml

    # Which entries of the combined yaml's `train:` list are the pseudo-label half. Only the
    # balanced loader needs this; it partitions the combined dataset back into its two sources.
    pl_img_path = dataset_train_paths(pl_yaml) if pl_yaml else []

    # ── Build base overrides ─────────────────────────────────────────────────
    cli_overrides: dict = {}
    if args.epochs is not None:
        cli_overrides["epochs"] = args.epochs
    if args.batch is not None:
        cli_overrides["batch"] = args.batch

    base = _build_base_overrides(cfg, cli_overrides)

    # ── Weights & Biases ─────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    # output.project is ".../training_runs/<run_name>" once {run_name} has been
    # resolved (by run_student_training.py, or left literal if train.py is run standalone
    # — see this module's docstring) — its last path segment IS the run_name,
    # so this needs no separate run_name plumbing into train.py.
    wandb_group = Path(base["project"]).name

    # ── Resolve which experiments to run ────────────────────────────────────
    experiments_cfg = cfg.get("experiments", {})
    requested = set(args.experiments) if args.experiments else None

    def _should_run(name: str) -> bool:
        exp = experiments_cfg.get(name, {})
        if not exp.get("enabled", True):
            return False
        if requested is not None and name not in requested:
            return False
        return True

    # One subprocess per arm, then done — the child re-enters main() with a single --experiments
    # and takes the in-process path below. Placed after the config/yaml setup above so a broken
    # config still fails once, here, rather than once per spawned arm.
    if args.isolate_arms:
        selected = [name for name in ARMS if _should_run(name)]
        if not selected:
            print("No experiments selected; nothing to run.")
        else:
            run_arms_isolated(args, selected)
        maybe_shutdown(cfg, force=args.shutdown)
        return

    yaml_for = {"gt": gt_yaml, "combined": combined_yaml}

    for name, arm in ARMS.items():
        if not _should_run(name):
            continue
        data_yaml = yaml_for[arm.data]
        if not data_yaml:
            print(f"Skipping {name}: no {arm.data} dataset yaml available.")
            continue

        exp_cfg = experiments_cfg.get(name, {})

        # Varying this BETWEEN arms makes the data pipeline differ across them, so a resulting
        # mAP gap no longer isolates the loss; set it only when mosaic weighting is itself under
        # test.
        mosaic_w_image = exp_cfg.get("mosaic_w_image")
        if mosaic_w_image is not None:
            if not arm.spec.per_image:
                raise SystemExit(
                    f"experiments.{name}.mosaic_w_image is set, but {name} weights per box, "
                    "which already carries through mosaic at full fidelity. Remove the key."
                )
            spec = replace(arm.spec, mosaic_w_image=mosaic_w_image)
        else:
            spec = arm.spec

        # Whether this arm's batches come from TWO single-source loaders at a fixed GT:PL ratio
        # (STAC-style, and mosaic never mixes sources) or from one shuffled combined dataset.
        # Precedence mirrors mosaic_w_image: the arm's own `balanced_loader:` beats the global
        # --balanced-loader flag, which beats the default (off = the original loader).
        balanced = exp_cfg.get("balanced_loader", args.balanced_loader)
        if balanced and arm.data != "combined":
            raise SystemExit(
                f"experiments.{name}.balanced_loader is set, but {name} trains on GT alone — "
                "there is no pseudo-label half to balance against. Remove the key."
            )
        if balanced and not pl_img_path:
            raise SystemExit(
                f"{name}: balanced_loader needs pl_data.dataset_yaml so the combined dataset can "
                "be split back into its GT and PL halves; none is configured."
            )
        if balanced and not gt_yaml:
            raise SystemExit(
                f"{name}: balanced_loader needs a GT half, but no gt.dataset_yaml is available "
                "(the combined dataset is pseudo-labels only). Run prepare_data.py first."
            )

        # Whether this arm keeps the default binomial batch composition but splits the dataset by
        # source so mosaics never mix GT and PL tiles — the mosaic-purity half of what the
        # balanced loader changes, on its own. Same precedence as the two knobs above.
        pure_mosaic = exp_cfg.get("pure_mosaic", args.pure_mosaic)
        if pure_mosaic and balanced:
            raise SystemExit(
                f"experiments.{name}: pure_mosaic and balanced_loader are mutually exclusive — "
                "the balanced loader already makes every mosaic single-source, as a side effect "
                "of splitting the loader. Pick one; the pair of runs is the A/B."
            )
        if pure_mosaic and arm.data != "combined":
            raise SystemExit(
                f"experiments.{name}.pure_mosaic is set, but {name} trains on GT alone — a mosaic "
                "there is already single-source. Remove the key."
            )
        if pure_mosaic and not pl_img_path:
            raise SystemExit(
                f"{name}: pure_mosaic needs pl_data.dataset_yaml so the combined dataset can be "
                "split back into its GT and PL halves; none is configured."
            )
        if pure_mosaic and not gt_yaml:
            raise SystemExit(
                f"{name}: pure_mosaic needs a GT half, but no gt.dataset_yaml is available (the "
                "combined dataset is pseudo-labels only). Run prepare_data.py first."
            )

        # Resolved here rather than at the overrides dict below because --label-lambda needs it to
        # build the run label. None for the arms that carry no weighting mechanism at all.
        lambda1 = exp_cfg.get("lambda1", arm.lambda1) if arm.use_weights else None
        zerogate = exp_cfg.get("zerogate", arm.zerogate) if arm.use_weights else 0.0

        # A run that changes the loader — or, under --label-lambda, one that changes lambda1 — gets
        # its own output directory, W&B run and promoted-weights dir, so the same arm can be run
        # every way under one --run-name. Without this they would collide, and ultralytics APPENDS
        # to results.csv rather than overwriting it.
        #
        # ':g' so a label reads lam0.1 / lam2 rather than lam0.100000 / lam2.0, and so the same
        # lambda1 written as 2 or 2.0 in yaml lands in the SAME directory instead of two.
        run_label = name if lambda1 is None or not args.label_lambda else f"{name}-lam{lambda1:g}"
        if zerogate:
            run_label = f"{run_label}-zerogate{zerogate:g}"
        if balanced:
            run_label = f"{run_label}-balanced"
        elif pure_mosaic:
            run_label = f"{run_label}-puremosaic"

        overrides = {**base, "data": data_yaml, "name": run_label, "spec": spec}
        if args.save_period is not None:
            overrides["save_period"] = args.save_period

        # Where ultralytics will put this arm. `base["project"]` is already absolute (see
        # _build_base_overrides), and ultralytics joins project/name with exist_ok=True.
        run_dir = Path(base["project"]) / run_label
        last_pt = run_dir / "weights" / "last.pt"
        if args.resume and last_pt.exists():
            # A STRING path, not True: True sends ultralytics' check_resume to get_latest_run(),
            # which globs the whole runs tree for the most recent last.pt and would happily resume
            # a DIFFERENT arm into this one's directory.
            overrides["resume"] = str(last_pt)
            print(f"\n[resume] {run_label}: continuing from {last_pt}")
        elif (run_dir / "results.csv").exists() and not args.overwrite:
            # exist_ok=True plus ultralytics APPENDING to results.csv means this would not fail on
            # its own -- it would interleave two runs' epoch histories in one file, and the
            # best.pt selection would be made across both. Unrecoverable from the artifacts, so
            # refuse here instead.
            raise SystemExit(
                f"{run_label}: {run_dir / 'results.csv'} already exists.\n"
                f"  train.py sets exist_ok: True and ultralytics APPENDS to results.csv, so "
                f"continuing would silently interleave this run with the previous one.\n"
                f"  --resume    continue the interrupted run from weights/last.pt"
                f"{' (present)' if last_pt.exists() else ' (NOT present -- nothing to resume)'}\n"
                f"  --overwrite train over it deliberately\n"
                f"  or move {run_dir} aside."
            )
        if arm.use_weights:
            overrides["weights_json_path"] = weights_json
            overrides["lambda1"] = lambda1
            if zerogate:
                overrides["zerogate"] = zerogate
        if balanced:
            overrides["pl_img_path"] = pl_img_path
            overrides["balance_ratio"] = exp_cfg.get("balance_ratio", 0.5)
        elif pure_mosaic:
            overrides["pl_img_path"] = pl_img_path

        if balanced:
            trainer_cls = BalancedDualDetectionTrainer
        elif pure_mosaic:
            trainer_cls = PureMosaicDetectionTrainer
        else:
            trainer_cls = Phase2DetectionTrainer

        run_experiment(
            run_label, overrides, weights_dir,
            wandb_cfg=wandb_cfg, wandb_group=wandb_group,
            trainer_cls=trainer_cls, eval_cfg=eval_cfg,
        )

    maybe_shutdown(cfg, force=args.shutdown)


if __name__ == "__main__":
    main()
