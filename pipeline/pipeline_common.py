"""pipeline_common.py — shared orchestration for the phase and pipeline runners.

Used by pseudo_labeling/run_pseudo_labeling.py, student_training/run_student_training.py, and run_pipeline.py. Each phase
config's `steps:` list is what a runner actually executes.

Two problems this module centralizes:

1. **{run_name} templating.** Config files (config.yaml/config.yaml)
   carry a literal `{run_name}` placeholder in their output paths so a single
   --run-name value resolves consistently across an entire pipeline run. The
   individual step scripts (run_inference.py, filter_and_convert.py,
   prepare_gt_data.py, train.py) are unaware templating exists — they only ever
   see a fully-resolved temp config written by `write_resolved_config`.

2. **Shutdown ordering.** Every step script independently calls its own
   maybe_shutdown() at its end, with no CLI way to force it off. If left alone, a
   config with shutdown_on_completion:true would power the machine down after the
   first step, before later steps (or later phases) ever ran. `write_resolved_config`
   forces shutdown_on_completion:false in the temp config every step sees, and the
   *runner* (phase wrapper or run_pipeline.py) performs the real shutdown exactly
   once at the very end via `maybe_shutdown`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import yaml

RUN_NAME_PLACEHOLDER = "{run_name}"


def _substitute(obj, run_name: str):
    """Recursively replace the {run_name} placeholder in every string in a config tree."""
    if isinstance(obj, dict):
        return {k: _substitute(v, run_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, run_name) for v in obj]
    if isinstance(obj, str):
        return obj.replace(RUN_NAME_PLACEHOLDER, run_name)
    return obj


def config_shutdown_flag(config_path: str | Path) -> bool:
    """Return the raw shutdown_on_completion value from a config, without resolving anything."""
    with open(config_path) as f:
        return bool(yaml.safe_load(f).get("shutdown_on_completion", False))


def write_resolved_config(config_path: str | Path, run_name: str) -> Path:
    """Write a temp copy of the phase config with {run_name} substituted and shutdown forced off.

    The temp file is created *next to the original config* (not in a scratch dir) so
    that relative paths inside it — classes_file, image dirs, output dirs — resolve
    against the same directory the original would have. Returns the temp file's Path;
    the caller is responsible for deleting it (run_phase does this in a finally).

    Note: `classes_file` is left as a reference (not inlined); every step script loads
    the config through a classes_file-aware load_config, so the reference resolves the
    same way it would from the original config.
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg = _substitute(cfg, run_name)
    cfg["shutdown_on_completion"] = False  # runner owns shutdown, never the sub-steps

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{config_path.stem}.",
        suffix=".resolved.yaml",
        dir=config_path.parent,
        delete=False,
    )
    with handle as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return Path(handle.name)


def run_phase(
    config_path: str | Path,
    run_name: str,
    only_steps: Iterable[str] | None = None,
) -> None:
    """Run every enabled step of a phase in order against a {run_name}-resolved temp config.

    Steps come from the config's `steps:` list (name/script/args/enabled). Each is run
    as a subprocess `python <script> --config <resolved> <args...>` with cwd set to the
    config's directory, so `from utils import ...` inside each script resolves correctly.

    Does NOT handle shutdown — that is the caller's responsibility, so run_pipeline.py can
    suppress per-phase shutdown and do it once after the final phase. Raises SystemExit
    if any step exits non-zero (the phase aborts rather than pressing on).
    """
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    steps = raw.get("steps")
    if not steps:
        raise SystemExit(f"No 'steps:' list found in {config_path}")

    only = set(only_steps) if only_steps else None
    resolved = write_resolved_config(config_path, run_name)
    try:
        for step in steps:
            name = step["name"]
            if only is not None and name not in only:
                print(f"[skip] {name} (not in requested --steps)")
                continue
            if not step.get("enabled", True):
                print(f"[skip] {name} (disabled in config)")
                continue
            cmd = [sys.executable, step["script"], "--config", str(resolved)]
            cmd += [str(a) for a in step.get("args", [])]
            print(f"\n[run ] {name}: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=config_dir)
            if result.returncode != 0:
                raise SystemExit(
                    f"Step '{name}' failed (exit {result.returncode}); aborting phase."
                )
    finally:
        resolved.unlink(missing_ok=True)


def maybe_shutdown(enabled: bool) -> None:
    """Shut down the machine after a 10s cancellable countdown, if enabled.

    `enabled` is computed by the caller (typically `--shutdown` flag OR the config's
    shutdown_on_completion value). Requires passwordless `sudo shutdown`.
    """
    if not enabled:
        return

    print("\n── Shutdown scheduled ───────────────────────────────────────────")
    print("  Machine will shut down in 10 seconds.")
    print("  Press Ctrl+C to cancel.")
    print("─────────────────────────────────────────────────────────────────")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Shutdown cancelled.")
        return

    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
