"""utils.py — Shared utilities for student training weighted YOLO training."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import yaml
from confidence import load_confidence


def load_weights_json(path: str | Path) -> Dict[str, List[float]]:
    """Load confidence_weights.json and return {stem: [w_sam3_per_box]}.

    The JSON produced by pseudo-labeling is keyed by absolute image path.  We index by
    stem so the lookup is robust to path changes between machines / runs.
    """
    return {stem: [b["w_sam3"] for b in boxes]
            for stem, boxes in load_confidence(path).items()}


def _reject_placeholders(config_path: Path, text: str) -> None:
    """A shipped config is a template; refuse to run one whose <dataset> is still unfilled."""
    if "<dataset>" in text:
        raise SystemExit(
            f"{config_path.name} still contains the <dataset> placeholder.\n"
            f"  {config_path}\n"
            "  Replace every <dataset> with your dataset's name before running. "
            "See pipeline/NEW_DATASET.md."
        )


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return the parsed dict.

    Supports a `classes_file:` key as an alternative to inlining `classes:` directly — the referenced yaml's `classes:` list
    is merged in. Skipped if `classes` is already populated, so this is safe to call
    more than once (e.g. on a wrapper-resolved temp config that already inlined it).
    """
    path = Path(path).resolve()
    text = path.read_text()
    _reject_placeholders(path, text)
    cfg = yaml.safe_load(text)

    if cfg.get("classes_file") and not cfg.get("classes"):
        classes_file = Path(cfg["classes_file"])
        if not classes_file.is_absolute():
            classes_file = (path.parent / classes_file).resolve()
        with open(classes_file) as f:
            classes_doc = yaml.safe_load(f)
        cfg["classes"] = classes_doc["classes"]

    return cfg


def resolve_dataset_paths(cfg: dict, key: str, yaml_path: str | Path | None = None) -> list[str]:
    """Resolve a dataset yaml's `train:`/`val:` entry to a list of absolute directories.

    Entries may be a single string or a list, absolute or relative to the yaml's `path:`.

    Relative entries are normalized WITHOUT following symlinks, so the `/images/` segment in
    paths like `train/images` survives — ultralytics' img2label_paths derives the sibling
    `labels/` directory from it, and `Path.resolve()` would follow the per-file symlinks out of
    the split and break that.
    """
    val = cfg.get(key)
    if val is None:
        return []
    base = Path(cfg.get("path") or "")
    if not base.is_absolute():
        base = (Path(yaml_path).absolute().parent if yaml_path else Path.cwd()) / base
    if isinstance(val, str):
        val = [val]
    resolved = []
    for p_str in val:
        p = Path(p_str)
        if not p.is_absolute():
            p = Path(os.path.normpath(Path(base) / p_str))
        resolved.append(str(p))
    return resolved


def dataset_train_paths(dataset_yaml_path: str | Path) -> list[str]:
    """The absolute image directories a dataset yaml's `train:` key points at.

    Used to tell student training's balanced loader which entries of the combined dataset's `train:` list
    are the pseudo-label half (see balanced_loader.split_source_paths).
    """
    with open(dataset_yaml_path) as f:
        cfg: dict = yaml.safe_load(f)
    return resolve_dataset_paths(cfg, "train", dataset_yaml_path)


def _relative_to(path: str | Path, base: Path) -> str:
    """Express *path* relative to *base*, falling back to absolute if it cannot be.

    Used for dataset yaml entries so the file does not hardcode one machine's layout. The
    round-trip is checked rather than assumed: ultralytics resolves each entry as
    `(base / entry).resolve()`, and .resolve() follows symlinks, so a relative form that does
    not land back on the same directory is rejected in favour of the absolute one.
    """
    target = Path(path).resolve()
    try:
        rel = os.path.relpath(target, base)
    except ValueError:  # different drives (Windows); no relative form exists
        return str(target)
    return rel if (base / rel).resolve() == target else str(target)


def create_combined_dataset_yaml(
    gt_yaml_path: str | Path,
    pl_yaml_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a combined dataset.yaml listing both GT and PL image directories.

    Ultralytics supports a list under the 'train' key. The yaml carries no 'path' key, so
    check_det_dataset uses the yaml's own directory as the dataset root and every entry below
    is written relative to it — which keeps the file portable across machines. An entry that
    cannot be expressed relative to that directory is left absolute.

    Returns the resolved output path.
    """
    with open(gt_yaml_path) as f:
        gt_cfg: dict = yaml.safe_load(f)
    with open(pl_yaml_path) as f:
        pl_cfg: dict = yaml.safe_load(f)

    gt_train = resolve_dataset_paths(gt_cfg, "train", gt_yaml_path)
    pl_train = resolve_dataset_paths(pl_cfg, "train", pl_yaml_path)
    gt_val = resolve_dataset_paths(gt_cfg, "val", gt_yaml_path) or resolve_dataset_paths(pl_cfg, "val", pl_yaml_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent.resolve()

    train = [_relative_to(p, base) for p in gt_train + pl_train]
    val = [_relative_to(p, base) for p in gt_val]

    combined = {
        "train": train,
        "val": val[0] if len(val) == 1 else val or None,
        "nc": pl_cfg["nc"],
        "names": pl_cfg["names"],
    }

    with open(output_path, "w") as f:
        yaml.dump(combined, f, default_flow_style=False, sort_keys=False)

    return output_path


def maybe_shutdown(cfg, force=False) -> None:
    """Power management is deliberately outside the release pipeline."""
    if force or cfg.get("shutdown_on_completion", False):
        raise ValueError("Power management is not supported by the release pipeline")
