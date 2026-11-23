#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""run_student_training.py — Run all of student training end to end.

Executes the enabled steps in config.yaml's `steps:` list in order
(by default: prepare_data.py -> train.py), against a config whose {run_name}
placeholders have been resolved from --run-name. Step scripts never see the placeholder:
pipeline_common.write_resolved_config hands each one a fully-resolved temp config.

Usage
-----
    python run_student_training.py --config config.yaml --run-name 5-percent-ft-sam3-iter3

    # Run only a subset of the configured steps (e.g. re-train without re-preparing GT data):
    python run_student_training.py --config config.yaml --run-name my-run --steps train

    # Shut the machine down when the whole phase finishes:
    python run_student_training.py --config config.yaml --run-name my-run --shutdown

Per-step flags (e.g. train.py's --experiments) are configured in the step's `args:`
list in config.yaml, not passed here. Adding or reordering steps (e.g. a future
Stage B train_ema.py) is a config edit; this script needs no changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pipeline_common lives one level up, at the pipeline root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common import config_shutdown_flag, maybe_shutdown, run_phase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all of student training end to end.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-name", required=True, help="Substituted for {run_name} in output paths")
    parser.add_argument("--steps", nargs="+", default=None, help="Run only these named steps")
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after the phase completes (overrides config shutdown_on_completion)",
    )
    args = parser.parse_args()
    if args.shutdown:
        parser.error("Power management is not supported by the release pipeline")

    config_path = Path(args.config).resolve()
    run_phase(config_path, args.run_name, only_steps=args.steps)
    maybe_shutdown(args.shutdown or config_shutdown_flag(config_path))


if __name__ == "__main__":
    main()
