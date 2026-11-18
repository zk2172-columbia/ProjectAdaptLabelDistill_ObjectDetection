"""
utils.py — Shared utilities for pseudo-labeling pseudo-label generation.

Functions
---------
scan_image_dir          Glob a directory for images.
load_coco_image_list    Parse a COCO JSON for image metadata.
xyxy_to_yolo            Convert pixel xyxy bbox → YOLO normalized cxcywh.
iou                     Compute IoU between two xyxy boxes.
nms                     Pure-NumPy non-maximum suppression.
merge_multiclass_boxes  Cross-class spatial merge; returns winning class per location.
get_sam3_bpe_path       Locate the SAM3 BPE vocabulary file.
load_config             Load and validate config.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ─── Image discovery ──────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def scan_image_dir(image_dir: str | Path) -> List[Path]:
    """Return a sorted list of image paths found under *image_dir* (recursive).

    Searches for files with common image extensions (case-insensitive).
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {root}")

    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def load_coco_image_list(
    coco_json: str | Path,
    image_dir: str | Path,
) -> List[Dict[str, Any]]:
    """Parse a COCO JSON file and return a list of image dicts.

    Each dict has at minimum:
        image_id   int
        file_path  Path   (resolved: image_dir / file_name)
        width      int
        height     int

    Parameters
    ----------
    coco_json:
        Path to a COCO-format annotation file.
    image_dir:
        Root directory under which file_name paths are resolved.
    """
    with open(coco_json) as f:
        data = json.load(f)

    root = Path(image_dir)
    records = []
    for img in data["images"]:
        records.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "file_path": root / img["file_name"],
                "width": img.get("width", None),
                "height": img.get("height", None),
            }
        )
    return records


# ─── Box coordinate conversion ────────────────────────────────────────────────


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert a pixel-space xyxy bounding box to YOLO normalized cxcywh.

    Parameters
    ----------
    x1, y1, x2, y2:
        Pixel coordinates of the top-left and bottom-right corners.
    img_w, img_h:
        Image width and height in pixels.

    Returns
    -------
    (cx_norm, cy_norm, w_norm, h_norm) each in [0, 1].
    """
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clamp to [0, 1] to handle minor out-of-bounds from model rounding
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    return cx, cy, w, h


# ─── IoU ──────────────────────────────────────────────────────────────────────


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union for two xyxy boxes.

    Parameters
    ----------
    box_a, box_b:
        [x1, y1, x2, y2] in pixel or normalized coordinates.

    Returns
    -------
    IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


# ─── NMS ──────────────────────────────────────────────────────────────────────


def nms(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.3,
) -> List[int]:
    """Pure-NumPy non-maximum suppression.

    Parameters
    ----------
    boxes:
        List of [x1, y1, x2, y2] xyxy boxes.
    scores:
        Confidence score for each box (same length as boxes).
    iou_threshold:
        IoU threshold above which a lower-scoring box is suppressed.

    Returns
    -------
    Indices of kept boxes, in descending score order.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    order = scores_arr.argsort()[::-1]  # highest score first
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        ious = np.array(
            [iou(boxes_arr[i].tolist(), boxes_arr[j].tolist()) for j in rest]
        )
        suppressed = ious >= iou_threshold
        order = rest[~suppressed]

    return keep


# ─── Cross-class spatial merge ────────────────────────────────────────────────


def merge_multiclass_boxes(
    predictions: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
    score_gap: float = 0.3,
) -> List[Dict[str, Any]]:
    """Merge spatially overlapping detections from different classes.

    For each pair of boxes that overlap above *iou_thresh*, two cases arise:

    1. **Score gap > score_gap** (one class clearly dominates):
       The lower-scoring box is suppressed. SAM3 is unambiguously preferring one
       class at this location — almost certainly the same physical object detected
       under two class labels. Only the winner is kept; its score becomes `w_sam3`.

    2. **Score gap ≤ score_gap** (scores are close):
       **Both boxes are kept.** Close scores signal that SAM3 is genuinely
       uncertain which class applies — most likely two co-located or overlapping
       objects of different classes (e.g., two types of debris touching). Suppressing
       either would discard a real detection.

    Within-class duplicates are already collapsed by NMS before this step, so any
    remaining cross-class overlap is between distinct class predictions only.

    Parameters
    ----------
    predictions:
        List of dicts, each with keys:
            image_path  str
            class_id    int
            class_name  str
            bbox_xyxy   [x1, y1, x2, y2]
            score       float
            image_w     int
            image_h     int
    iou_thresh:
        IoU threshold for considering two boxes as the same spatial location.
        Pairs below this overlap are left untouched regardless of score gap.
    score_gap:
        Minimum score difference required to suppress the lower-scoring box.
        If ``winner_score - loser_score <= score_gap``, both boxes are kept.

    Returns
    -------
    List of dicts with an additional key ``w_sam3`` (= each box's own score).
    May be equal in size to or smaller than the input, but never discards a box
    unless a clear winner (gap > score_gap) exists at that spatial location.
    """
    if not predictions:
        return []

    # Sort by score descending so the highest-confidence box is always the
    # "seed" when evaluating a cluster.
    sorted_preds = sorted(predictions, key=lambda p: p["score"], reverse=True)
    suppressed = [False] * len(sorted_preds)

    for i, pred in enumerate(sorted_preds):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(sorted_preds)):
            if suppressed[j]:
                continue
            overlap = iou(pred["bbox_xyxy"], sorted_preds[j]["bbox_xyxy"])
            if overlap >= iou_thresh:
                gap = pred["score"] - sorted_preds[j]["score"]
                if gap > score_gap:
                    # Clear winner → suppress the loser
                    suppressed[j] = True
                # else: scores are close → keep both (different real objects)

    return [
        {**p, "w_sam3": p["score"]}
        for p, sup in zip(sorted_preds, suppressed)
        if not sup
    ]


# ─── Shutdown helper ─────────────────────────────────────────────────────────


def maybe_shutdown(cfg: Dict[str, Any], force: bool = False) -> None:
    """Shut down the machine if configured or forced.

    Parameters
    ----------
    cfg:
        Top-level config dict. Reads ``shutdown_on_completion`` (bool, default False).
    force:
        If True, shut down regardless of the config value (e.g. from --shutdown CLI flag).
    """
    if not (force or cfg.get("shutdown_on_completion", False)):
        return

    import subprocess
    import time

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


# ─── SAM3 utilities ───────────────────────────────────────────────────────────


def get_sam3_bpe_path() -> str:
    """Return the path to the SAM3 BPE vocabulary file bundled with the package."""
    import sam3 as _sam3

    bpe = Path(_sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.exists():
        raise FileNotFoundError(
            f"SAM3 BPE file not found at expected path: {bpe}\n"
            "Make sure the SAM3 submodule is initialized and installed."
        )
    return str(bpe)


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


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate config.yaml.

    Returns the config dict. Relative `output.*` paths are resolved
    relative to the directory containing config.yaml.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    text = config_path.read_text()
    _reject_placeholders(config_path, text)
    cfg = yaml.safe_load(text)

    # Resolve all relative paths against the config file's directory so that
    # the script works correctly regardless of where it is invoked from.
    config_dir = config_path.parent

    # classes_file: shared class list, shared with the student-training config.
    # Skipped if classes is already populated (e.g. a wrapper-resolved temp config
    # that already inlined the list), so this is safe to call more than once.
    if cfg.get("classes_file") and not cfg.get("classes"):
        classes_file = Path(cfg["classes_file"])
        if not classes_file.is_absolute():
            classes_file = (config_dir / classes_file).resolve()
        with open(classes_file) as f:
            classes_doc = yaml.safe_load(f)
        cfg["classes"] = classes_doc["classes"]

    # data.image_dir and data.coco_json
    data = cfg.get("data", {})
    for key in ("image_dir", "coco_json"):
        if data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str((config_dir / data[key]).resolve())

    # output paths
    out = cfg.get("output", {})
    for key in ("raw_predictions_path", "yolo_labels_dir", "weights_json_path",
                "dataset_yaml_path", "masked_data_dir"):
        if key in out and out[key] and not Path(out[key]).is_absolute():
            out[key] = str((config_dir / out[key]).resolve())

    # Validate required fields
    if not cfg.get("classes"):
        raise ValueError(
            "Config must define at least one entry under 'classes', either inline "
            "or via 'classes_file' pointing at a yaml with a 'classes:' list."
        )
    if not cfg.get("data", {}).get("image_dir"):
        raise ValueError("Config must set data.image_dir.")

    return cfg
