#!/usr/bin/env python3
"""
generate_ft_config.py — teacher adaptation, Step 1: Generate a per-run SAM3 Hydra config
==============================================================================

Copies the shared SAM3 finetuning template (sam3.template_config) and edits it for
this run: pins `all_roboflow_supercategories` to the single dataset being trained
(at task_index 0), points `paths.experiment_log_dir` at this run's experiment dir,
and applies any free-form `overrides:` from config.yaml. Writes the result
under the SAM3 submodule's own configs/ tree, since train.py's
`initialize_config_module("sam3.train")` can only discover configs physically
located there (see config.yaml's comment on generated_config_relpath).

Usage
-----
    python generate_ft_config.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from utils import TRAIN_MODULE_RELPATH, apply_overrides, load_config

SENTINEL = "__SET_BY_generate_ft_config__"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a per-run SAM3 finetuning config.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sam3_cfg = cfg["sam3"]
    train_module_root = Path(sam3_cfg["repo_root"]) / TRAIN_MODULE_RELPATH

    # template_config is already absolute (load_config resolves it relative to the config
    # file's own directory) — unlike generated_config_relpath below, it lives outside the
    # SAM3 submodule now, so it is NOT joined onto train_module_root.
    template_path = Path(sam3_cfg["template_config"])
    if not template_path.exists():
        raise FileNotFoundError(f"SAM3 template config not found: {template_path}")
    with open(template_path) as f:
        tmpl = yaml.safe_load(f)

    tmpl["all_roboflow_supercategories"] = [cfg["dataset"]]
    tmpl.setdefault("submitit", {}).setdefault("job_array", {})["task_index"] = 0
    tmpl.setdefault("paths", {})["experiment_log_dir"] = cfg["output"]["experiment_dir"]

    apply_overrides(tmpl, cfg.get("overrides"))

    unset = [k for k, v in tmpl.get("paths", {}).items() if str(v).startswith(SENTINEL)]
    if unset:
        raise SystemExit(
            f"paths.{', paths.'.join(unset)} still hold the template sentinel.\n"
            f"  Set them in {args.config}'s overrides: block (see PATH_OVERRIDE_KEYS in utils.py)."
        )

    generated_path = train_module_root / sam3_cfg["generated_config_relpath"]
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    with open(generated_path, "w") as f:
        # Hydra's "@package _global_" directive lives in a YAML comment, which
        # yaml.safe_load/yaml.dump silently drops on a round-trip. Without it, Hydra
        # nests this config under a group key instead of merging at the top level, and
        # train.py's cfg.launcher/cfg.paths/etc. lookups fail with "not in struct".
        f.write("# @package _global_\n")
        yaml.dump(tmpl, f, default_flow_style=False, sort_keys=False)

    print(f"[generate_ft_config] dataset:            {cfg['dataset']}")
    print(f"[generate_ft_config] experiment_log_dir:  {cfg['output']['experiment_dir']}")
    print(f"[generate_ft_config] wrote generated config: {generated_path}")


if __name__ == "__main__":
    main()
