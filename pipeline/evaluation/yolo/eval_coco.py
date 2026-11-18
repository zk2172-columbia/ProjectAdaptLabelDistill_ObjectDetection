#!/usr/bin/env python3
"""eval_coco.py — score checkpoints with COCOeval. The manual entry point, and the replacement
for reaching for `model.val()` by hand.

Reports mAP50, mAP50-95, APs, APm, APl — the five numbers pipeline/EVALUATION.md standardizes on —
for one checkpoint, several, or a whole sweep. All scoring lives in coco_eval.py; this file is
argument handling and reporting only.

Usage
-----
    PY=/home/zklab/miniconda3/envs/ald/bin/python

    # One checkpoint, against the configured defaults
    $PY eval_coco.py --weights ../weights/all-label-yolo/gt_only/best_standard.pt

    # Several, labelled, into one table
    $PY eval_coco.py --target gt_only=../weights/naive-sweep-seed42-gtonly/gt_only/best_standard.pt \
                     --target perbox=../weights/mask-sweep-seed42-mosaicfix-t0.5/gt_pl_perbox-lam0.5/best_standard.pt

    # Several checkpoints in one pass, from a batch file (schema below)
    $PY eval_coco.py --config my_runs.yaml

    # Re-score detections that already exist — no GPU, no inference, no checkpoint needed
    $PY eval_coco.py --rescore ../eval_results/all-label-yolo/gt_only/predictions.json

Defaults for --gt / --data / --split / imgsz / batch / device come from evaluation_config.yaml, so
the bare --weights form works with no other arguments. A targets file may override any of them.

Outputs, under --out-dir (default from the config):
    <label>/                    the raw ultralytics val() run dir, including predictions.json
    <label>/coco_metrics.json   that checkpoint's five metrics + full provenance
    metrics.json                every scored target in one file, keyed by label
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import (  # noqa: E402
    CONF_FLOOR,
    EvalSpec,
    format_table,
    print_report,
    score_checkpoint,
    score_predictions,
    write_metrics,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "evaluation_config.yaml"


def _resolve(base: Path, p) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def auto_label(weights: Path) -> str:
    """Name a checkpoint by its path, since every arm's file is called best_standard.pt.

    weights/<run>/<arm>/best_standard.pt  ->  "<run>/<arm>"
    runs/<run>/weights/best_standard.pt   ->  "<run>"   (the ultralytics layout, where the
                                                         containing directory is the
                                                         uninformative literal "weights")
    """
    parent, grandparent = weights.parent, weights.parent.parent
    if parent.name == "weights":
        return grandparent.name
    return f"{grandparent.name}/{parent.name}" if grandparent.name else parent.name


def _free_gpu() -> None:
    """Release the model between targets — a long sweep otherwise accumulates until it OOMs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_defaults(config_path: Path) -> tuple[dict, Path]:
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}, config_path.parent


def build_targets(args, cfg: dict, base: Path) -> tuple[list[tuple[str, Path]], dict, Path]:
    """Resolve the target list and the settings they are scored under.

    Precedence: CLI flag > targets-file key > evaluation_config.yaml. A targets file is just a
    config with a `targets:` list, so a retired sweep grid is a data file rather than a script.
    """
    settings = {k: cfg.get(k) for k in ("gt", "data", "split", "imgsz", "batch", "device", "out_dir")}
    targets: list[tuple[str, Path]] = []

    if args.config:
        tfile = Path(args.config).resolve()
        with open(tfile) as f:
            tcfg = yaml.safe_load(f) or {}
        tbase = tfile.parent
        for k in settings:
            if tcfg.get(k) is not None:
                settings[k] = tcfg[k]
        for entry in tcfg.get("targets", []):
            targets.append((entry["label"], _resolve(tbase, entry["weights"])))
        base = tbase

    for k, flag in (("gt", args.gt), ("data", args.data), ("split", args.split),
                    ("out_dir", args.out_dir)):
        if flag is not None:
            settings[k] = flag
    for k, flag in (("imgsz", args.imgsz), ("batch", args.batch), ("device", args.device)):
        if flag is not None:
            settings[k] = flag

    for w in args.weights or []:
        targets.append((auto_label(Path(w).resolve()), Path(w).resolve()))
    for t in args.target or []:
        if "=" not in t:
            raise SystemExit(f"--target must be LABEL=PATH, got {t!r}")
        label, path = t.split("=", 1)
        targets.append((label, Path(path).resolve()))

    return targets, settings, base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score checkpoints with COCOeval (mAP50, mAP50-95, APs, APm, APl)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[-1],
    )
    parser.add_argument("--weights", nargs="+", help="checkpoint(s) to score, labelled by path")
    parser.add_argument("--target", action="append", help="LABEL=PATH; repeatable")
    parser.add_argument("--config", help="batch file: gt/data/out_dir + a targets: list of "
                                          "{label, weights} pairs")
    parser.add_argument("--rescore", nargs="+",
                        help="existing predictions.json file(s) to re-score. No inference, no GPU.")
    parser.add_argument("--defaults", default=str(DEFAULT_CONFIG),
                        help="config supplying gt/data/split/imgsz/batch/device defaults")
    parser.add_argument("--gt", help="ground truth COCO json (overrides the config)")
    parser.add_argument("--data", help="dataset yaml for val() (overrides the config)")
    parser.add_argument("--split", help="which key of the dataset yaml to score")
    parser.add_argument("--out-dir", help="where run dirs and metrics land")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--conf", type=float, default=None,
        help=f"override the eval confidence floor (repo standard: {CONF_FLOOR}); overwrites the "
             "standard coco_metrics.json / metrics.json in place",
    )
    parser.add_argument(
        "--max-det", type=int, default=None,
        help="override the per-image detection cap (repo standard: 300)",
    )
    args = parser.parse_args()

    cfg, base = load_defaults(Path(args.defaults).resolve())

    # ── Re-score path: cheap, CPU-only, needs no dataset yaml ────────────────
    if args.rescore:
        gt = _resolve(base, args.gt or cfg.get("gt"))
        results = {}
        for pj in args.rescore:
            pj = Path(pj).resolve()
            label = f"{pj.parent.parent.name}/{pj.parent.name}"
            payload = score_predictions(gt, pj, quiet=True)
            results[label] = payload
            print_report(label, payload)
        if len(results) > 1:
            print("\n" + format_table(list(results.items())))
        return

    targets, settings, base = build_targets(args, cfg, base)
    if not targets:
        raise SystemExit("nothing to score — pass --weights, --target, --config or --rescore.")

    out_dir = _resolve(base, settings["out_dir"] or "../../eval_results/coco_eval")
    spec_base = EvalSpec(
        gt_json=_resolve(base, settings["gt"]),
        data_yaml=_resolve(base, settings["data"]),
        split=settings.get("split") or "test",
        imgsz=settings.get("imgsz") or 640,
        batch=settings.get("batch") or 16,
        device=str(settings.get("device") or "0"),
    )
    spec_base.preflight()
    overrides = {}
    if args.conf is not None:
        overrides["conf"] = args.conf
    if args.max_det is not None:
        overrides["max_det"] = args.max_det
    if overrides:
        spec_base = dataclasses.replace(spec_base, **overrides)

    results: dict[str, dict] = {}
    for label, weights in targets:
        if not Path(weights).is_file():
            print(f"SKIP {label}: no checkpoint at {weights}")
            continue
        print(f"\n{'=' * 64}\n  {label}\n{'=' * 64}", flush=True)
        # One directory per label, so a re-run overwrites its own predictions.json rather than
        # silently scoring against another target's.
        payload = score_checkpoint(weights, spec_base, out_dir, name=label.replace("/", "-"))
        write_metrics(payload, out_dir / label.replace("/", "-") / "coco_metrics.json")
        results[label] = payload
        print_report(label, payload)
        _free_gpu()

    if not results:
        raise SystemExit(f"no checkpoints found for any target under {out_dir} — nothing evaluated.")

    combined = out_dir / "metrics.json"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote combined metrics: {combined}")
    print("\n" + format_table(list(results.items())))


if __name__ == "__main__":
    main()
