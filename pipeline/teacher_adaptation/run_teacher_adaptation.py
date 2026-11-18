#!/usr/bin/env python3
"""run_teacher_adaptation.py — Run all of teacher adaptation end to end.

Executes the enabled steps in config.yaml's `steps:` list in order (by default:
generate_ft_config.py -> run_finetune.py -> merge_checkpoint.py), against a config whose
{run_name} placeholders have been resolved from --run-name.

Note: teacher adaptation's --run-name is its OWN experiment namespace — it is NOT meant to be
the same value used for a pipeline run. Finetuning SAM3 is coarse-grained (GPU-hours, run
rarely); the later stages get re-run many times against the
same resulting checkpoint. See config.yaml's header comment.

Usage
-----
    python run_teacher_adaptation.py --config config.yaml --run-name 5-percent-ft-sam3-iter3

    # Run only a subset of the configured steps:
    python run_teacher_adaptation.py --config config.yaml --run-name my-ft-run --steps run_finetune

    # Shut the machine down when the whole phase finishes:
    python run_teacher_adaptation.py --config config.yaml --run-name my-ft-run --shutdown

Per-step flags are configured in the step's `args:` list in config.yaml, not
passed here — so they stay associated with the correct step. Adding or reordering steps
is a config edit; this script needs no changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pipeline_common lives one level up, at the pipeline root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common import config_shutdown_flag, maybe_shutdown, run_phase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all of teacher adaptation end to end.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-name", required=True, help="This phase's own experiment name")
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
