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

Once the merge succeeds, the raw trainer checkpoint (<experiment_dir>/checkpoints/
checkpoint.pt) is deleted — it's ~3x the size of the merged checkpoint (model weights
plus optimizer/scaler/step state) and is only useful for resuming interrupted training,
which the later stages never do. Set output.keep_raw_checkpoint: true in the config to keep it.

Usage
-----
    python merge_checkpoint.py --config config.yaml
"""

from __future__ import annotations

import argparse
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

    print("[merge_checkpoint] downloading HF base checkpoint...")
    hf_path = download_ckpt_from_hf()
    hf_ckpt = torch.load(hf_path, map_location="cpu")
    hf_state = hf_ckpt["model"] if "model" in hf_ckpt else hf_ckpt

    print(f"[merge_checkpoint] loading finetuned checkpoint: {ft_checkpoint_path}")
    ft_ckpt = torch.load(ft_checkpoint_path, map_location="cpu")
    ft_state = ft_ckpt["model"] if "model" in ft_ckpt else ft_ckpt

    merged, replaced, skipped = merge_checkpoint_state(hf_state, ft_state)
    print(f"[merge_checkpoint] replaced keys: {replaced}")
    print(f"[merge_checkpoint] skipped keys:  {len(skipped)}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged}, merged_path)
    print(f"[merge_checkpoint] wrote merged checkpoint: {merged_path}")
    print(
        "[merge_checkpoint] next: paste this path into config.yaml's "
        "sam3.checkpoint_path"
    )

    if cfg.get("output", {}).get("keep_raw_checkpoint", False):
        print(f"[merge_checkpoint] keeping raw trainer checkpoint (output.keep_raw_checkpoint: true): {ft_checkpoint_path}")
    else:
        ft_checkpoint_path.unlink()
        print(f"[merge_checkpoint] deleted raw trainer checkpoint (output.keep_raw_checkpoint: false): {ft_checkpoint_path}")


if __name__ == "__main__":
    main()
