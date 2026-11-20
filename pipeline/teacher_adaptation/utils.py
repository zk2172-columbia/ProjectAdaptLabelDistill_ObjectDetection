"""
utils.py — Shared utilities for teacher adaptation SAM3 teacher finetuning.

Functions
---------
load_config              Load and resolve config.yaml.
set_dotted               Set a nested dict value from a dotted key path.
apply_overrides          Apply a dict of dotted-path overrides onto a config tree.
merge_checkpoint_state   Merge a finetuned state dict into an HF base state dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Fixed structural fact of the SAM3 package layout (not user-configurable): train.py is
# invoked as `python sam3/train/train.py` from sam3.repo_root, but Hydra's
# initialize_config_module("sam3.train") resolves config_name relative to the
# sam3/train/ package dir itself, one level below repo_root. sam3.template_config and
# sam3.generated_config_relpath in config.yaml are relative to this.
TRAIN_MODULE_RELPATH = Path("sam3/train")

# ─── Config loading ───────────────────────────────────────────────────────────


def _reject_placeholders(config_path: Path, text: str) -> None:
    """A shipped config is a template; refuse to run one whose <dataset> is still unfilled."""
    if "<dataset>" in text:
        raise SystemExit(
            f"{config_path.name} still contains the <dataset> placeholder.\n"
            f"  {config_path}\n"
            "  Replace every <dataset> with your dataset's name before running. "
            "See pipeline/NEW_DATASET.md."
        )


# Dotted `overrides:` keys whose value is a filesystem path. Hydra reads the generated config
# from inside the SAM3 submodule, so a relative value there would resolve against the wrong
# directory — these are made absolute against the config file's own directory at load time.
PATH_OVERRIDE_KEYS = ("paths.bpe_path", "paths.roboflow_vl_100_root")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate config.yaml.

    Returns the config dict. Relative `sam3.repo_root`, `output.*` and PATH_OVERRIDE_KEYS paths
    are resolved relative to the directory containing the config file, so this works correctly
    regardless of where the step script is invoked from.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    text = config_path.read_text()
    _reject_placeholders(config_path, text)
    cfg = yaml.safe_load(text)

    config_dir = config_path.parent

    sam3_cfg = cfg.get("sam3", {})
    repo_root = Path(sam3_cfg["repo_root"])
    if not repo_root.is_absolute():
        repo_root = (config_dir / repo_root).resolve()
    sam3_cfg["repo_root"] = str(repo_root)

    # template_config lives outside the SAM3 submodule (see config.yaml's comment),
    # so it's resolved relative to this config file's directory, same as output.* below —
    # NOT relative to repo_root/sam3/train/ like generated_config_relpath still is.
    template_config = Path(sam3_cfg["template_config"])
    if not template_config.is_absolute():
        template_config = (config_dir / template_config).resolve()
    sam3_cfg["template_config"] = str(template_config)

    out = cfg.get("output", {})
    for key in ("experiment_dir", "merged_checkpoint_path"):
        if out.get(key) and not Path(out[key]).is_absolute():
            out[key] = str((config_dir / out[key]).resolve())

    overrides = cfg.get("overrides") or {}
    for key in PATH_OVERRIDE_KEYS:
        value = overrides.get(key)
        if value and not Path(value).is_absolute():
            overrides[key] = str((config_dir / value).resolve())

    if not cfg.get("dataset"):
        raise ValueError("Config must set 'dataset' to a folder under experiment_files/finetuning-data/.")
    if not sam3_cfg.get("template_config"):
        raise ValueError("Config must set sam3.template_config.")
    if sam3_cfg.get("generated_config_path"):
        generated = Path(sam3_cfg["generated_config_path"])
        if not generated.is_absolute():
            generated = config_dir / generated
        generated = generated.resolve()
        if not out.get("experiment_dir"):
            raise ValueError("External config requires output.experiment_dir")
        config_root = Path(out["experiment_dir"]).resolve() / "configs"
        if not generated.is_relative_to(config_root) or generated == config_root:
            raise ValueError("External generated config must remain under output.experiment_dir/configs")
        sam3_cfg["generated_config_path"] = str(generated)
    elif not sam3_cfg.get("generated_config_relpath"):
        raise ValueError("Config must set sam3.generated_config_path or sam3.generated_config_relpath.")

    return cfg


# ─── Dotted-path overrides ────────────────────────────────────────────────────


def set_dotted(tree: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set tree["a"]["b"]["c"] = value given dotted_key "a.b.c", creating dicts as needed.

    Raises if an intermediate path segment exists but is not a dict, since that would
    silently discard existing sibling keys instead of overriding a single leaf.
    """
    parts = dotted_key.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(
                f"Cannot apply override '{dotted_key}': '{part}' is not a mapping "
                f"(found {type(child).__name__})."
            )
        node = child
    node[parts[-1]] = value


def apply_overrides(tree: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Apply each dotted-path override in *overrides* onto *tree*, in place.

    Returns *tree* for convenience. No-op if *overrides* is empty/None.
    """
    for dotted_key, value in (overrides or {}).items():
        set_dotted(tree, dotted_key, value)
    return tree


# ─── Checkpoint merging ───────────────────────────────────────────────────────


def merge_checkpoint_state(
    hf_state: Dict[str, Any], ft_state: Dict[str, Any], *,
    allowed_skipped: list[str] | None = None,
) -> Tuple[Dict[str, Any], int, List[str]]:
    """Overlay a finetuned state dict onto an HF base state dict.

    Mirrors experiment_files/jupyter_notebooks/download_checkpoints.ipynb: finetuned keys are
    prefixed with "detector." if not already present, then overlaid onto the HF state
    for keys that exist in both with matching tensor shape. Keys with a shape mismatch
    or no HF counterpart are left out of the merge (segmentation-head weights the
    detection-only finetune never touched, for example).

    Parameters
    ----------
    hf_state:
        State dict from the HuggingFace base checkpoint.
    ft_state:
        State dict from the finetuned checkpoint (as saved by sam3's Trainer, i.e.
        NOT yet prefixed with "detector.").

    Returns
    -------
    (merged_state, replaced_count, skipped_keys)
    """
    ft_state_prefixed = {
        (k if k.startswith("detector.") else f"detector.{k}"): v for k, v in ft_state.items()
    }

    merged = dict(hf_state)
    replaced = 0
    skipped: List[str] = []

    for k, v in ft_state_prefixed.items():
        if k in merged and merged[k].shape == v.shape:
            merged[k] = v
            replaced += 1
        else:
            skipped.append(k)

    if replaced == 0:
        raise ValueError("Checkpoint merge replaced zero parameters")
    unexpected = set(skipped) - set(allowed_skipped or [])
    if unexpected:
        raise ValueError(f"Unexpected unmatched checkpoint keys: {sorted(unexpected)}")
    return merged, replaced, skipped
