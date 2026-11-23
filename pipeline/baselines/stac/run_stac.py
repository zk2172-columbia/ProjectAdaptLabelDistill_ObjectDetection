#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""run_stac.py — the STAC driver: teacher -> pseudo-labels -> student.

Three stages against one config and one --run-name:

    teacher   pseudolabels.py --stage teacher  -> runs/<run>/raw_predictions.jsonl
    filter    pseudolabels.py --stage filter   -> runs/<run>/pseudolabels/, pseudolabel_stats.json
    student   train_student.py                 -> runs/<run>/train/, coco_metrics.json

Each stage is a separate subprocess so a crash in one cannot leave the others holding a
half-initialized CUDA context, and so any stage can be re-run alone.

Usage
-----
    python run_stac.py --run-name stac-seed42
    python run_stac.py --run-name stac-seed42 --dry-run
    python run_stac.py --run-name stac-seed42 --stages filter student
    python run_stac.py --run-name stac-seed42 --stages teacher -- --limit 20

Read runs/<run>/pseudolabel_stats.json after the filter stage, BEFORE spending GPU hours on the
student: at tau = 0.9 a teacher with modest recall may leave most of the unlabeled pool empty,
and that is worth knowing from a JSON rather than from a finished training curve.

Launch detached so it survives the terminal that started it:

    setsid nohup ./run_stac.py --run-name stac-seed42 > stac.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils import load_config

HERE = Path(__file__).resolve().parent

# Each stage's argv after the interpreter. Separate subprocesses on purpose: a crash in one must
# not leave the others holding a half-initialized CUDA context, and any stage can be re-run alone.
STAGES = {
    "teacher": ["pseudolabels.py", "--stage", "teacher"],
    "filter": ["pseudolabels.py", "--stage", "filter"],
    "student": ["train_student.py"],
}


def preflight(cfg: dict) -> list[str]:
    """Everything that must exist before any stage runs. Returns a list of problems."""
    problems = []
    checks = [
        ("teacher checkpoint", cfg["teacher"]["checkpoint"], "file"),
        ("unlabeled COCO json", cfg["data"]["unlabeled_coco_json"], "file"),
        ("image pool", cfg["data"]["image_dir"], "dir"),
        ("GT train images", cfg["data"]["gt_train_images"], "dir"),
        ("GT val images", cfg["data"]["gt_val_images"], "dir"),
        ("held-out eval yaml", cfg["data"]["eval_dataset_yaml"], "file"),
        ("classes.yaml", cfg["classes_file"], "file"),
    ]
    for label, path, kind in checks:
        p = Path(path)
        ok = p.is_file() if kind == "file" else p.is_dir()
        if not ok:
            problems.append(f"{label} not found: {path}")

    # Guards against the ONE mistake that would invalidate the whole experiment without failing:
    # a teacher that had seen the unlabeled split would make the pseudo-labels circular.
    if not problems:
        import yaml
        args_yaml = Path(cfg["teacher"]["checkpoint"]).parent.parent / "args.yaml"
        if args_yaml.is_file():
            teacher_data = yaml.safe_load(args_yaml.read_text()).get("data", "")
            print(f"  teacher trained on: {teacher_data}")
            if "label" in Path(str(teacher_data)).name:
                problems.append(
                    f"teacher's training data yaml looks like it includes the unlabeled split: "
                    f"{teacher_data}. The teacher must be trained on the labeled split ALONE."
                )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="STAC: teacher -> pseudo-labels -> student")
    parser.add_argument("--config", default="stac_config.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                        help="stages to run, in the given order (default: all three)")
    parser.add_argument(
        "--tau", type=float, default=None,
        help=(
            "Override filter.tau, forwarded to BOTH the filter and student stages. A tau variant "
            "therefore reuses this run's config file unchanged, so nothing else can differ "
            "between it and the baseline. The teacher stage ignores it — inference runs at "
            "conf 0.001 precisely so tau can be retuned without re-running it (see "
            "--reuse-predictions)."
        ),
    )
    parser.add_argument(
        "--reuse-predictions", metavar="RUN_NAME", default=None,
        help=(
            "Symlink another run's raw_predictions.jsonl into this run and skip the teacher "
            "stage. Sound because teacher inference does not depend on tau: same checkpoint, "
            "same images, same eval-mode forward pass. Saves ~25 min and 470 MB per variant."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the manifest and the commands; create nothing")
    parser.add_argument("extra", nargs="*",
                        help="extra args forwarded to every stage (put them after a bare --)")
    args = parser.parse_args()

    config_path = (HERE / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path)
    run_dir = Path(cfg["output"]["root"]) / args.run_name
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir

    print("=" * 78)
    print(f"  STAC — run '{args.run_name}'")
    print("=" * 78)
    print(f"  teacher     : {cfg['teacher']['checkpoint']}")
    print(f"  unlabeled   : {cfg['data']['unlabeled_coco_json']}")
    print(f"  labeled     : {cfg['data']['gt_train_images']}")
    tau = cfg["filter"]["tau"] if args.tau is None else args.tau
    tau_note = "" if args.tau is None else "  (--tau override)"
    print(f"  filter      : NMS IoU {cfg['filter']['nms_iou']}, then tau {tau}{tau_note}")
    print(f"  student     : {cfg['student']['model']}, lambda_u {cfg['student']['lambda_u']}, "
          f"{cfg['student']['epochs']} epochs, patience {cfg['student']['patience']}, "
          f"batch {cfg['student']['batch']} split {1 - cfg['student']['balance_ratio']:.0%}/"
          f"{cfg['student']['balance_ratio']:.0%}")
    print(f"  seed        : {cfg['seed']}")
    print(f"  held-out    : split '{cfg['data']['eval_split']}' of {cfg['data']['eval_dataset_yaml']}")
    print(f"  W&B         : project '{cfg['wandb']['project']}', group '{args.run_name}'")
    print(f"  output      : {run_dir}")
    print(f"  stages      : {' -> '.join(args.stages)}")
    print("=" * 78)

    problems = preflight(cfg)
    if problems:
        print("\nPreflight FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("  preflight OK\n")

    # The teacher stage takes no --tau: it runs at conf 0.001 regardless, which is what makes a
    # tau variant a filter-stage-only rerun.
    tau_args = [] if args.tau is None else ["--tau", str(args.tau)]
    commands = [
        [sys.executable, str(HERE / STAGES[s][0]), *STAGES[s][1:], "--config", str(config_path),
         "--run-name", args.run_name, *(tau_args if s != "teacher" else []), *args.extra]
        for s in args.stages
    ]

    if args.reuse_predictions and not args.dry_run:
        source = run_dir.parent / args.reuse_predictions / "raw_predictions.jsonl"
        if not source.is_file():
            raise SystemExit(f"--reuse-predictions: no raw_predictions.jsonl at {source}")
        run_dir.mkdir(parents=True, exist_ok=True)
        link = run_dir / "raw_predictions.jsonl"
        if not link.exists():
            link.symlink_to(source)
        print(f"  reusing predictions: {link} -> {source}\n")

    if args.dry_run:
        print("--dry-run — nothing created. Would run, in order:\n")
        for cmd in commands:
            print("  " + " ".join(cmd))
        return

    for stage, cmd in zip(args.stages, commands):
        print(f"\n{'=' * 78}\n  stage: {stage}   started {datetime.now():%F %T}\n{'=' * 78}\n",
              flush=True)
        result = subprocess.run(cmd, cwd=HERE)
        if result.returncode != 0:
            raise SystemExit(f"stage '{stage}' failed with exit code {result.returncode}")
        print(f"\n  stage '{stage}' finished {datetime.now():%F %T}", flush=True)

    print(f"\n{'=' * 78}")
    print(f"  STAC complete — {datetime.now():%F %T}")
    print(f"  pseudo-label stats : {run_dir / 'pseudolabel_stats.json'}")
    print(f"  training run       : {run_dir / 'train'}")
    print(f"  held-out metrics   : {run_dir / 'coco_metrics.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
