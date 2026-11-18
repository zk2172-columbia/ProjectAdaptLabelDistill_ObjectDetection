#!/usr/bin/env python3
"""
run_inference.py — pseudo-labeling, Step A: SAM3 Oracle Inference
==========================================================

Runs the fine-tuned SAM3 model on every unlabeled image and saves ALL raw
detections to raw_predictions.json. No filtering is applied here — that is
entirely the responsibility of filter_and_convert.py (Step B).

Usage
-----
    python run_inference.py --config config.yaml
    python run_inference.py --config config.yaml --resume     # skip already-processed images

Outputs
-------
    outputs/raw_predictions.json
        A list of detection dicts, one entry per detected box:
        {
            "image_path": "/abs/path/to/image.jpg",
            "class_id":   0,
            "class_name": "person",
            "bbox_xyxy":  [x1, y1, x2, y2],   # pixel coordinates
            "score":      0.87,                # SAM3 grounding confidence
            "image_w":    1920,
            "image_h":    1080,
        }

Design notes
------------
- confidence_threshold=0.0 is passed to Sam3Processor so that all SAM3
  detections are kept regardless of score. The 85th-percentile quality
  filter is applied in Step B after the full score distribution is known.
- set_image() is called once per image; set_text_prompt() is called once
  per class (it re-uses the cached backbone features from set_image()).
- Visual prompting (add_geometric_prompt) is a geometric-prompt fallback
  using seed boxes from teacher adaptation. Enable with inference.use_visual_prompting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from PIL import Image
from tqdm import tqdm

# ── Locate SAM3 submodule ─────────────────────────────────────────────────────
# The SAM3 submodule lives two levels above this script:
# pipeline/pseudo_labeling/run_inference.py → Student_Teacher_Pipeline/SAM3/
_THIS_DIR = Path(__file__).resolve().parent
_SAM3_SUBMODULE = _THIS_DIR.parent.parent / "SAM3"
if str(_SAM3_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SAM3_SUBMODULE))

from utils import (  # noqa: E402  (local import after sys.path update)
    get_sam3_bpe_path,
    load_config,
    load_coco_image_list,
    maybe_shutdown,
    scan_image_dir,
)


# ─── Atomic JSON write ────────────────────────────────────────────────────────


def _write_json_atomic(
    data: Any,
    dest: Path,
    indent: int | None = None,
) -> None:
    """Write *data* as JSON to *dest* atomically.

    Writes to a sibling `.tmp` file first, then uses ``os.replace()`` to swap
    it into place. On POSIX systems ``os.replace`` is an atomic rename, so
    *dest* is never left in a partially-written state if the process is killed
    during the write.
    """
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, dest)


# ─── Model loading ────────────────────────────────────────────────────────────


def build_model_and_processor(cfg: Dict[str, Any]):
    """Load SAM3 and wrap in Sam3Processor.

    Parameters
    ----------
    cfg:
        Top-level config dict (from load_config).

    Returns
    -------
    (model, processor)
    """
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_cfg = cfg["sam3"]
    device = sam3_cfg.get("device", "cuda")
    checkpoint_path = sam3_cfg.get("checkpoint_path", None)
    resolution = sam3_cfg.get("resolution", 1008)
    confidence_threshold = sam3_cfg.get("confidence_threshold", 0.0)

    # Enable TF32 for Ampere GPUs (matches eval.py pattern)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    bpe_path = get_sam3_bpe_path()

    print(f"Loading SAM3 model (checkpoint: {checkpoint_path or 'HuggingFace default'}) ...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,  # None → HuggingFace weights
        enable_segmentation=True,
        device=device,
    )
    model.eval()

    processor = Sam3Processor(
        model,
        resolution=resolution,
        device=device,
        confidence_threshold=confidence_threshold,  # 0.0 → keep everything
    )
    print("Model ready.")
    return model, processor


# ─── Seed box loading (visual prompting) ──────────────────────────────────────


def load_seed_boxes(cfg: Dict[str, Any]) -> Dict[int, List[List[float]]]:
    """Load teacher adaptation seed bounding boxes from a COCO JSON, grouped by class_id.

    Returns
    -------
    {class_id: [[cx_norm, cy_norm, w_norm, h_norm], ...]}
    """
    inf_cfg = cfg.get("inference", {})
    seed_json = inf_cfg.get("seed_coco_json", None)
    seed_img_dir = inf_cfg.get("seed_image_dir", None)

    if not seed_json:
        return {}

    with open(seed_json) as f:
        data = json.load(f)

    # Build image-size lookup
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # Build category-id → config class_id mapping
    # Assumes category names match config class names (case-insensitive)
    class_name_to_id = {
        cls["name"].lower(): cls["id"] for cls in cfg["classes"]
    }
    coco_cat_to_class_id = {}
    for cat in data.get("categories", []):
        name = cat["name"].lower()
        if name in class_name_to_id:
            coco_cat_to_class_id[cat["id"]] = class_name_to_id[name]

    seed_boxes: Dict[int, List[List[float]]] = {}
    for ann in data.get("annotations", []):
        coco_cat_id = ann["category_id"]
        if coco_cat_id not in coco_cat_to_class_id:
            continue
        class_id = coco_cat_to_class_id[coco_cat_id]
        img_w, img_h = img_sizes[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]  # COCO xyxy top-left + wh
        cx_norm = (x + bw / 2) / img_w
        cy_norm = (y + bh / 2) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h
        seed_boxes.setdefault(class_id, []).append(
            [cx_norm, cy_norm, w_norm, h_norm]
        )

    print(
        f"Loaded seed boxes: "
        + ", ".join(f"class {k}: {len(v)} boxes" for k, v in seed_boxes.items())
    )
    return seed_boxes


# ─── Per-image inference ──────────────────────────────────────────────────────


@torch.inference_mode()
def infer_image(
    img_path: Path,
    processor,
    classes: List[Dict[str, Any]],
    dtype: str,
    device: str,
    use_visual_prompting: bool,
    seed_boxes: Dict[int, List[List[float]]],
) -> List[Dict[str, Any]]:
    """Run SAM3 on one image for all classes.

    Returns a list of raw detection dicts (no NMS, no filtering).
    """
    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    detections: List[Dict[str, Any]] = []

    autocast_ctx = torch.autocast(device_type=device.split(":")[0], dtype=getattr(torch, dtype))

    with autocast_ctx:
        # Set image once; backbone features are cached in inference_state
        inference_state = processor.set_image(image)

        for cls in classes:
            class_id = cls["id"]
            class_name = cls["name"]
            text_prompt = cls["text_prompt"]

            # ── Text prompting ───────────────────────────────────────────────
            state = processor.set_text_prompt(prompt=text_prompt, state=inference_state)
            boxes = state["boxes"].cpu().float()   # [N, 4] xyxy pixels
            scores = state["scores"].cpu().float() # [N]

            for box, score in zip(boxes.tolist(), scores.tolist()):
                detections.append(
                    {
                        "image_path": str(img_path),
                        "class_id":   class_id,
                        "class_name": class_name,
                        "bbox_xyxy":  box,
                        "score":      float(score),
                        "image_w":    img_w,
                        "image_h":    img_h,
                        "prompt_type": "text",
                    }
                )

            # ── Visual prompting (geometric fallback) ────────────────────────
            if use_visual_prompting and class_id in seed_boxes:
                # Re-set image to get a clean state for geometric prompts
                geo_state = processor.set_image(image)
                # Use "visual" as the text so the model relies on geometric cue
                geo_state = processor.set_text_prompt(prompt="visual", state=geo_state)

                for seed_box in seed_boxes[class_id]:
                    geo_state = processor.add_geometric_prompt(
                        box=seed_box,   # [cx_norm, cy_norm, w_norm, h_norm]
                        label=True,
                        state=geo_state,
                    )
                    geo_boxes = geo_state["boxes"].cpu().float()
                    geo_scores = geo_state["scores"].cpu().float()

                    for box, score in zip(geo_boxes.tolist(), geo_scores.tolist()):
                        detections.append(
                            {
                                "image_path": str(img_path),
                                "class_id":   class_id,
                                "class_name": class_name,
                                "bbox_xyxy":  box,
                                "score":      float(score),
                                "image_w":    img_w,
                                "image_h":    img_h,
                                "prompt_type": "visual",
                            }
                        )

    return detections


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pseudo-labeling Step A: SAM3 inference → raw_predictions.json"
    )
    parser.add_argument(
        "--config", required=True, help="Path to config.yaml"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in raw_predictions.json",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after completion (overrides config shutdown_on_completion)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    sam3_cfg = cfg["sam3"]
    data_cfg = cfg["data"]
    inf_cfg = cfg.get("inference", {})
    out_cfg = cfg["output"]

    device = sam3_cfg.get("device", "cuda")
    dtype = sam3_cfg.get("dtype", "bfloat16")
    use_visual_prompting = inf_cfg.get("use_visual_prompting", False)

    # ── Build image list ─────────────────────────────────────────────────────
    image_dir = data_cfg["image_dir"]
    coco_json = data_cfg.get("coco_json", None)

    if coco_json:
        records = load_coco_image_list(coco_json, image_dir)
        image_paths = [r["file_path"] for r in records]
        print(f"Image list from COCO JSON: {len(image_paths)} images")
    else:
        image_paths = scan_image_dir(image_dir)
        print(f"Image list from directory scan: {len(image_paths)} images")

    # ── Resume: load existing predictions ───────────────────────────────────
    raw_pred_path = Path(out_cfg["raw_predictions_path"])
    raw_pred_path.parent.mkdir(parents=True, exist_ok=True)

    existing_predictions: List[Dict[str, Any]] = []
    already_processed: Set[str] = set()

    if args.resume and raw_pred_path.exists():
        try:
            with open(raw_pred_path) as f:
                existing_predictions = json.load(f)
            already_processed = {p["image_path"] for p in existing_predictions}
            print(
                f"Resume mode: {len(already_processed)} images already processed, "
                f"{len(image_paths) - len(already_processed)} remaining."
            )
        except json.JSONDecodeError as exc:
            print(
                f"Warning: raw_predictions.json is corrupted ({exc}).\n"
                f"  The file was likely truncated by an interrupted write.\n"
                f"  Starting fresh — all images will be re-processed.\n"
                f"  The corrupt file has been renamed to {raw_pred_path}.corrupt"
            )
            raw_pred_path.rename(raw_pred_path.with_suffix(".corrupt"))
            existing_predictions = []
            already_processed = set()

    images_to_process = [
        p for p in image_paths if str(p) not in already_processed
    ]

    if not images_to_process:
        print("All images already processed. Nothing to do.")
        return

    # ── Build model ──────────────────────────────────────────────────────────
    model, processor = build_model_and_processor(cfg)

    # ── Load seed boxes (for visual prompting) ───────────────────────────────
    seed_boxes: Dict[int, List[List[float]]] = {}
    if use_visual_prompting:
        seed_boxes = load_seed_boxes(cfg)
        if not seed_boxes:
            print(
                "Warning: use_visual_prompting=true but no seed boxes loaded. "
                "Check seed_coco_json in config."
            )

    # ── Run inference ────────────────────────────────────────────────────────
    all_predictions = list(existing_predictions)
    n_new_boxes = 0
    n_images_processed = 0

    for img_path in tqdm(images_to_process, desc="Inference", unit="img"):
        try:
            dets = infer_image(
                img_path=img_path,
                processor=processor,
                classes=cfg["classes"],
                dtype=dtype,
                device=device,
                use_visual_prompting=use_visual_prompting,
                seed_boxes=seed_boxes,
            )
        except Exception as exc:
            print(f"\nWarning: failed on {img_path}: {exc}")
            continue

        all_predictions.extend(dets)
        n_new_boxes += len(dets)
        n_images_processed += 1

        # Checkpoint every 100 *images* (not predictions) so progress is not
        # lost if the run is interrupted. Atomic write via .tmp + os.replace().
        if n_images_processed % 100 == 0:
            _write_json_atomic(all_predictions, raw_pred_path, indent=None)
            torch.cuda.empty_cache()

    # ── Final write ──────────────────────────────────────────────────────────
    _write_json_atomic(all_predictions, raw_pred_path, indent=2)

    print(
        f"\nDone. {len(images_to_process)} images processed, "
        f"{n_new_boxes} new detections.\n"
        f"Total predictions: {len(all_predictions)}\n"
        f"Saved → {raw_pred_path}"
    )

    maybe_shutdown(cfg, force=args.shutdown)


if __name__ == "__main__":
    main()
