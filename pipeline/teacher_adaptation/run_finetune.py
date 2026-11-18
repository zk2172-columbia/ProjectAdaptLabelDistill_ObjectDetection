#!/usr/bin/env python3
"""
run_finetune.py — teacher adaptation, Step 2: Run SAM3 Finetuning
=======================================================

Shells out to the SAM3 submodule's own train.py against the config generate_ft_config.py
just wrote, using the exact invocation convention documented in
SAM3/scripts/eval/*/README.md:

    cd <sam3.repo_root> && python sam3/train/train.py -c <generated_config> \\
        --use-cluster <sam3.use_cluster> --num-gpus <sam3.num_gpus>

train.py itself owns all Hydra composition, checkpointing, and (if use_cluster=1)
submitit/SLURM job submission — this script is a thin, config-driven wrapper so the
pipeline's run_name conventions reach it without duplicating any training logic.

Usage
-----
    python run_finetune.py --config config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 finetuning via the SAM3 submodule's train.py.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sam3_cfg = cfg["sam3"]
    repo_root = Path(sam3_cfg["repo_root"])

    # sys.executable, not a bare "python": train.py needs the aw3813 conda env (torch,
    # ultralytics, sam3), and a bare name resolves through PATH, which is only correct as
    # long as that env happens to be first. This pins the trainer to the same interpreter
    # that launched this step, matching how pipeline_common.run_phase already spawns steps.
    cmd = [
        sys.executable,
        "sam3/train/train.py",
        "-c",
        sam3_cfg["generated_config_relpath"],
        "--use-cluster",
        str(sam3_cfg["use_cluster"]),
        "--num-gpus",
        str(sam3_cfg["num_gpus"]),
    ]
    print(f"[run_finetune] cwd: {repo_root}")
    print(f"[run_finetune] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
