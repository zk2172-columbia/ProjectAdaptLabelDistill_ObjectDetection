#!/usr/bin/env python3
"""run_pipeline.py — Run the pipeline (pseudo-labeling -> student training) for one run_name.

Runs each stage listed in pipeline_config.yaml in order, all pinned to a single --run-name so
every stage reads and writes the same run's data. Each stage's own config owns the `steps:` it
executes. Teacher adaptation is not part of this: it has its own --run-name and is run
standalone from teacher_adaptation/.

This calls pipeline_common.run_phase() directly (in-process) rather than shelling out to the
per-stage runners, so there is exactly one place that decides shutdown: each stage's own
shutdown is forced off (via the resolved temp config), and the real shutdown happens once here
after the final stage. The standalone stage runners remain usable on their own.

Usage
-----
    python run_pipeline.py --run-name my-run

    # Run only some stages (e.g. resume at the student after pseudo-labeling already ran):
    python run_pipeline.py --run-name my-run --stages student_training

    # Shut the machine down when the whole pipeline finishes:
    python run_pipeline.py --run-name my-run --shutdown
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pipeline_common import maybe_shutdown, run_phase

PIPELINE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline end to end.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to pipeline_config.yaml")
    parser.add_argument("--run-name", required=True, help="Substituted for {run_name} across all stages")
    parser.add_argument("--stages", nargs="+", default=None, help="Run only these named stages")
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after the whole pipeline completes (overrides config)",
    )
    args = parser.parse_args()
    if args.shutdown:
        parser.error("Power management is not supported by the release pipeline")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PIPELINE_DIR / config_path).resolve()
    with open(config_path) as f:
        pcfg = yaml.safe_load(f)

    only = set(args.stages) if args.stages else None
    if only and not only <= {stage["name"] for stage in pcfg["stages"]}:
        parser.error("Unknown stage requested")
    for stage in pcfg["stages"]:
        if only is not None and stage["name"] not in only:
            print(f"\n[skip stage] {stage['name']} (not in requested --stages)")
            continue
        stage_config = (config_path.parent / stage["config"]).resolve()
        print(f"\n{'=' * 70}\n  PIPELINE STAGE: {stage['name']}  (run_name={args.run_name})\n{'=' * 70}")
        run_phase(stage_config, args.run_name)

    maybe_shutdown(args.shutdown or pcfg.get("shutdown_on_completion", False))


if __name__ == "__main__":
    main()
