#!/usr/bin/env python3
"""
merge_checkpoint.py — teacher adaptation, Step 3: Merge Finetuned Weights into HF Base Checkpoint
=======================================================================================

Ports experiment_files/jupyter_notebooks/download_checkpoints.ipynb into a script: SAM3's own
Trainer only saves the weights it actually trained (the detector head), so the
finetuned checkpoint on its own is missing the rest of the model (e.g. the
segmentation head). This overlays the finetuned weights onto a full HuggingFace base
checkpoint, keeping the base for everything the finetune didn't touch or that has a
mismatched shape (see utils.merge_checkpoint_state).

Outputs
-------
    <output.merged_checkpoint_path>
        {"model": merged_state_dict} — the format config.yaml's
        sam3.checkpoint_path expects.

The raw trainer checkpoint is always preserved. Cleanup is an explicit operation outside
this pipeline so failed validation cannot destroy the only resumable training state.

Usage
-----
    python merge_checkpoint.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import tempfile
import sys
from pathlib import Path

import torch
from utils import load_config, merge_checkpoint_state

# ── Locate SAM3 submodule ─────────────────────────────────────────────────────
# pipeline/teacher_adaptation/merge_checkpoint.py -> Student_Teacher_Pipeline/SAM3/
_THIS_DIR = Path(__file__).resolve().parent
_SAM3_SUBMODULE = _THIS_DIR.parent.parent / "SAM3"
if str(_SAM3_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SAM3_SUBMODULE))

from sam3.model_builder import download_ckpt_from_hf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a finetuned SAM3 checkpoint into the HF base checkpoint.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ft_checkpoint_path = Path(cfg["output"]["experiment_dir"]) / "checkpoints" / "checkpoint.pt"
    merged_path = Path(cfg["output"]["merged_checkpoint_path"])

    if not ft_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Finetuned checkpoint not found: {ft_checkpoint_path}\n"
            "run_finetune.py must complete (and save a checkpoint) before this step."
        )

    local_base = cfg["sam3"].get("base_checkpoint_path")
    if local_base:
        hf_path = Path(local_base).expanduser()
        if not hf_path.is_absolute():
            hf_path = Path(args.config).resolve().parent / hf_path
        hf_path = hf_path.resolve()
        if not hf_path.is_file():
            raise FileNotFoundError(f"Local base checkpoint not found: {hf_path}")
        print(f"[merge_checkpoint] using local base checkpoint: {hf_path}")
    else:
        print("[merge_checkpoint] downloading HF base checkpoint...")
        hf_path = download_ckpt_from_hf()
    hf_ckpt = torch.load(hf_path, map_location="cpu")
    hf_state = hf_ckpt["model"] if "model" in hf_ckpt else hf_ckpt

    print(f"[merge_checkpoint] loading finetuned checkpoint: {ft_checkpoint_path}")
    ft_ckpt = torch.load(ft_checkpoint_path, map_location="cpu")
    ft_state = ft_ckpt["model"] if "model" in ft_ckpt else ft_ckpt

    merged, replaced, skipped = merge_checkpoint_state(
        hf_state, ft_state, allowed_skipped=cfg.get("output", {}).get("allowed_skipped_keys", []))
    print(f"[merge_checkpoint] replaced keys: {replaced}")
    print(f"[merge_checkpoint] skipped keys:  {len(skipped)}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {merged_path}")
    fd, temporary = tempfile.mkstemp(dir=merged_path.parent, suffix=".pt")
    os.close(fd)
    try:
        torch.save({"model": merged}, temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=True)["model"]
        if set(restored) != set(merged) or any(
            not torch.equal(restored[k], merged[k]) for k in merged
        ):
            raise RuntimeError("Merged checkpoint failed readback verification")
        os.replace(temporary, merged_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    merged_path.with_suffix(".merge.json").write_text(json.dumps({
        "base_sha256": sha256(hf_path), "finetuned_sha256": sha256(ft_checkpoint_path),
        "merged_sha256": sha256(merged_path), "replaced": replaced, "skipped": skipped,
        "readback_verified": True,
    }, indent=2))
    print(f"[merge_checkpoint] wrote merged checkpoint: {merged_path}")
    print(
        "[merge_checkpoint] next: paste this path into config.yaml's "
        "sam3.checkpoint_path"
    )

    print(f"[merge_checkpoint] preserved raw trainer checkpoint: {ft_checkpoint_path}")


if __name__ == "__main__":
    main()
