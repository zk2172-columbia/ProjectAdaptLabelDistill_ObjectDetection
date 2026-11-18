#!/usr/bin/env python3
"""
filter_and_convert.py — pseudo-labeling, Step B: Dual Filter → YOLO Labels
===================================================================

Loads raw_predictions.json (from run_inference.py) and applies the two-level
filtering described in pcatcher_pipeline_plan.md, then converts surviving
detections to YOLO format.

Step B pipeline
---------------

    raw_predictions.json  (all SAM3 outputs, unfiltered)
              ↓
    hard score cutoff  (optional, e.g. score > 0.25)
        → drops detections below a fixed absolute confidence threshold
              ↓
    85th-percentile quality filter
        → removes globally low-confidence detections
              ↓
    within-class NMS  (per image, per class)
        → collapses same-class spatial duplicates
              ↓
    cross-class spatial merge
        → assigns winning class + w^SAM3_i per spatial location
              ↓
    outputs/pseudolabeled_data/images/<stem>.jpg   (per-file symlink to source image)
    outputs/pseudolabeled_data/labels/<stem>.txt   (YOLO normalized cxcywh)
    outputs/pseudolabeled_data/confidence_weights.json  (w^SAM3_i per box)
    outputs/pseudolabeled_data/dataset.yaml             (Ultralytics YOLO training config)

Usage
-----
    python filter_and_convert.py --config config.yaml
    python filter_and_convert.py --config config.yaml --percentile 90
    python filter_and_convert.py --config config.yaml --hard-cutoff 0.25

    # Constant-memory read, required for multi-GB raw_predictions.json:
    python filter_and_convert.py --config config.yaml --load-mode stream

Load modes
----------
raw_predictions.json is one flat array of every detection, so it can reach tens
of GB on large datasets. --load-mode / filtering.load_mode selects how it is
read: "memory" (default) json.load()s it whole, "stream" uses two constant-
memory ijson passes, "auto" picks by file size. See the LOAD_MODES notes below
for the tradeoff — both modes produce identical output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple

import ijson
import numpy as np
import yaml
from tqdm import tqdm

from utils import (
    iou,
    load_config,
    maybe_shutdown,
    merge_multiclass_boxes,
    nms,
    xyxy_to_yolo,
)


# ─── Filtering ────────────────────────────────────────────────────────────────


def apply_hard_cutoff_filter(
    predictions: List[Dict[str, Any]],
    cutoff: float,
) -> List[Dict[str, Any]]:
    """Keep only predictions whose score is strictly above a fixed cutoff.

    Parameters
    ----------
    predictions:
        Raw predictions from run_inference.py.
    cutoff:
        Absolute score threshold, e.g. 0.25 -> keep score > 0.25.

    Returns
    -------
    Filtered predictions (a subset of the input).
    """
    return [p for p in predictions if p["score"] > cutoff]


def apply_percentile_filter(
    predictions: List[Dict[str, Any]],
    percentile: float,
) -> tuple[List[Dict[str, Any]], float]:
    """Keep only predictions whose score is above the Nth percentile.

    Parameters
    ----------
    predictions:
        Raw predictions from run_inference.py.
    percentile:
        E.g. 85 → keep the top 15% by score.

    Returns
    -------
    (filtered_predictions, threshold)
    """
    if not predictions:
        return [], 0.0

    scores = [p["score"] for p in predictions]
    threshold = float(np.percentile(scores, percentile))
    filtered = [p for p in predictions if p["score"] >= threshold]
    return filtered, threshold


def apply_within_class_nms(
    predictions: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """Apply NMS within each (image, class) group.

    Parameters
    ----------
    predictions:
        Predictions already filtered by the percentile step.
    iou_threshold:
        IoU threshold for NMS (boxes with IoU >= threshold are suppressed).

    Returns
    -------
    Predictions after NMS (a subset of the input).
    """
    # Group by (image_path, class_id)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        groups[(p["image_path"], p["class_id"])].append(p)

    kept = []
    for group_preds in groups.values():
        boxes = [p["bbox_xyxy"] for p in group_preds]
        scores = [p["score"] for p in group_preds]
        keep_idx = nms(boxes, scores, iou_threshold)
        kept.extend(group_preds[i] for i in keep_idx)

    return kept


# ─── Load modes ───────────────────────────────────────────────────────────────
#
# Two interchangeable ways to get from raw_predictions.json to merged_by_image;
# both produce identical output (see tests/test_filter_and_convert_modes.py).
#
#   "memory" (default) — json.load() the whole file, then filter in memory.
#       Simplest, marginally faster on small files, and makes no assumption
#       about the order of records in the file.
#       Peak RSS is several times the file size, so this OOMs on large files.
#
#   "stream" — two ijson passes, never holding more than one image's records.
#       Peak RSS stays in the low hundreds of MB regardless of file size, but
#       it assumes all records for a given image_path are contiguous in the
#       file. run_inference.py guarantees this (it writes one image's dets at a
#       time via all_predictions.extend(dets)); stream_filter_and_group raises
#       if it ever encounters a file that violates it, rather than silently
#       producing wrong output.
#
#   "auto" — "stream" if the file is larger than AUTO_STREAM_THRESHOLD_BYTES,
#       else "memory".
#
# Rule of thumb: a 25GB raw_predictions.json (~54M detections) needs well over
# 50GB of RAM in "memory" mode but under 200MB in "stream" mode.

LOAD_MODES = ("memory", "stream", "auto")
AUTO_STREAM_THRESHOLD_BYTES = 2 * 1024**3  # 2 GB


class FilterResult(NamedTuple):
    """Counts at each filtering stage plus the final per-image merged boxes."""

    n_raw: int
    n_after_cutoff: int
    n_after_percentile: int
    n_after_nms: int
    n_final: int
    threshold: float | None
    merged_by_image: Dict[str, List[Dict[str, Any]]]


def resolve_load_mode(load_mode: str, raw_pred_path: Path) -> str:
    """Map a possibly-"auto" load mode to a concrete "memory" or "stream"."""
    if load_mode not in LOAD_MODES:
        raise SystemExit(
            f"Invalid load_mode {load_mode!r}; expected one of {', '.join(LOAD_MODES)}."
        )
    if load_mode != "auto":
        return load_mode

    size = raw_pred_path.stat().st_size
    chosen = "stream" if size > AUTO_STREAM_THRESHOLD_BYTES else "memory"
    print(
        f"load_mode=auto → {chosen} "
        f"({size / 1024**3:.2f} GB file, threshold "
        f"{AUTO_STREAM_THRESHOLD_BYTES / 1024**3:.0f} GB)"
    )
    return chosen


# ─── Streaming load (for raw_predictions.json too large to json.load() whole) ──


def _iter_raw_predictions(raw_pred_path: Path) -> Iterator[Dict[str, Any]]:
    # use_float=True: ijson defaults to decimal.Decimal for JSON numbers, which
    # breaks downstream float/numpy arithmetic (iou, xyxy_to_yolo, np.percentile).
    with open(raw_pred_path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def stream_collect_scores(
    raw_pred_path: Path,
    hard_cutoff: float | None,
) -> Tuple[int, int, np.ndarray]:
    """Pass 1: stream the file once to count records and collect post-cutoff scores.

    Returns (n_raw, n_after_cutoff, scores_after_cutoff) — scores_after_cutoff is
    only what apply_percentile_filter needs (a flat float array), not the full
    records, so this pass is cheap regardless of file size.
    """
    n_raw = 0
    scores: List[float] = []
    for p in tqdm(_iter_raw_predictions(raw_pred_path), desc="Pass 1/2: scanning scores", unit="det"):
        n_raw += 1
        if hard_cutoff is None or p["score"] > hard_cutoff:
            scores.append(p["score"])
    return n_raw, len(scores), np.asarray(scores, dtype=np.float64)


def _process_image_buffer(
    buffer: List[Dict[str, Any]],
    nms_iou_threshold: float,
    cross_class_iou: float,
    cross_class_score_gap: float,
    min_boxes: int,
) -> Tuple[int, List[Dict[str, Any]] | None]:
    """Run within-class NMS + cross-class merge for one image's already-filtered
    detections. Returns (n_after_nms_for_image, merged_list_or_None)."""
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in buffer:
        by_class[p["class_id"]].append(p)

    after_nms: List[Dict[str, Any]] = []
    for class_preds in by_class.values():
        boxes = [p["bbox_xyxy"] for p in class_preds]
        scores = [p["score"] for p in class_preds]
        keep_idx = nms(boxes, scores, nms_iou_threshold)
        after_nms.extend(class_preds[i] for i in keep_idx)

    merged = merge_multiclass_boxes(
        after_nms, iou_thresh=cross_class_iou, score_gap=cross_class_score_gap
    )
    if len(merged) >= min_boxes:
        return len(after_nms), merged
    return len(after_nms), None


def stream_filter_and_group(
    raw_pred_path: Path,
    hard_cutoff: float | None,
    percentile_threshold: float | None,
    nms_iou_threshold: float,
    cross_class_iou: float,
    cross_class_score_gap: float,
    min_boxes: int,
) -> Tuple[int, int, int, Dict[str, List[Dict[str, Any]]]]:
    """Pass 2: stream the file again, filter, group by (contiguous) image_path,
    and run NMS + cross-class merge per image as soon as its group ends.

    Returns (n_after_percentile, n_after_nms, n_final, merged_by_image).
    """
    n_after_percentile = 0
    n_after_nms = 0
    n_final = 0
    merged_by_image: Dict[str, List[Dict[str, Any]]] = {}
    seen_images: set[str] = set()

    current_image: str | None = None
    buffer: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal n_after_nms, n_final
        if current_image is None or not buffer:
            return
        n_nms, merged = _process_image_buffer(
            buffer, nms_iou_threshold, cross_class_iou, cross_class_score_gap, min_boxes
        )
        n_after_nms += n_nms
        if merged is not None:
            merged_by_image[current_image] = merged
            n_final += len(merged)

    for p in tqdm(_iter_raw_predictions(raw_pred_path), desc="Pass 2/2: filter+group", unit="det"):
        if hard_cutoff is not None and p["score"] <= hard_cutoff:
            continue
        if percentile_threshold is not None and p["score"] < percentile_threshold:
            continue
        n_after_percentile += 1

        img = p["image_path"]
        if img != current_image:
            flush()
            if img in seen_images:
                raise RuntimeError(
                    f"raw_predictions.json is not grouped by image: {img!r} "
                    "reappeared after another image's records. The streaming "
                    "filter assumes run_inference.py's per-image write order. "
                    "Re-run with --load-mode memory (needs RAM several times the "
                    "file size), which makes no ordering assumption."
                )
            seen_images.add(img)
            current_image = img
            buffer = [p]
        else:
            buffer.append(p)

    flush()
    return n_after_percentile, n_after_nms, n_final, merged_by_image


# ─── Filter drivers (one per load mode) ───────────────────────────────────────


def filter_in_memory(
    raw_pred_path: Path,
    hard_cutoff: float | None,
    percentile: float | None,
    nms_iou_threshold: float,
    cross_class_iou: float,
    cross_class_score_gap: float,
    min_boxes: int,
) -> FilterResult:
    """Whole-file path: json.load() everything, then filter in memory.

    Peak RSS is several times the file size — see the load-mode notes above.
    """
    print(f"Loading raw predictions from {raw_pred_path} ...")
    with open(raw_pred_path) as f:
        raw_predictions = json.load(f)

    n_raw = len(raw_predictions)
    print(f"Loaded {n_raw:,} raw detections.")
    if n_raw == 0:
        return FilterResult(0, 0, 0, 0, 0, None, {})

    # ── Hard score cutoff (optional, runs before the percentile filter) ──
    pre_cutoff = raw_predictions
    n_after_cutoff = n_raw
    if hard_cutoff is not None:
        print(f"\nApplying hard score cutoff (score > {hard_cutoff}) ...")
        pre_cutoff = apply_hard_cutoff_filter(raw_predictions, hard_cutoff)
        n_after_cutoff = len(pre_cutoff)
        print(f"  {n_raw:,} → {n_after_cutoff:,}")

    # ── Percentile quality filter (optional) ─────────────────────────────
    if percentile is not None:
        print(f"\nApplying {percentile}th-percentile quality filter ...")
        filtered, threshold = apply_percentile_filter(pre_cutoff, percentile)
        n_after_percentile = len(filtered)
        print(f"  {n_after_cutoff:,} → {n_after_percentile:,} (threshold={threshold:.4f})")
    else:
        print("\nSkipping percentile quality filter (score_percentile is null).")
        filtered = pre_cutoff
        n_after_percentile = n_after_cutoff
        threshold = None

    # ── Within-class NMS ─────────────────────────────────────────────────
    print(f"\nApplying within-class NMS (IoU threshold={nms_iou_threshold}) ...")
    after_nms = apply_within_class_nms(filtered, nms_iou_threshold)
    n_after_nms = len(after_nms)
    print(f"  {n_after_percentile:,} → {n_after_nms:,}")

    # ── Cross-class spatial merge (per image) ────────────────────────────
    print(
        f"\nCross-class spatial merge "
        f"(IoU threshold={cross_class_iou}, score gap={cross_class_score_gap}) ..."
    )
    print(
        f"  Overlapping cross-class boxes are suppressed only when score gap "
        f"> {cross_class_score_gap}; otherwise both are kept."
    )

    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in after_nms:
        by_image[p["image_path"]].append(p)

    merged_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for img_path, img_preds in by_image.items():
        merged = merge_multiclass_boxes(
            img_preds,
            iou_thresh=cross_class_iou,
            score_gap=cross_class_score_gap,
        )
        if len(merged) >= min_boxes:
            merged_by_image[img_path] = merged

    n_final = sum(len(v) for v in merged_by_image.values())
    print(f"  {n_after_nms:,} → {n_final:,} final boxes across {len(merged_by_image):,} images")

    return FilterResult(
        n_raw, n_after_cutoff, n_after_percentile, n_after_nms, n_final, threshold, merged_by_image
    )


def filter_streaming(
    raw_pred_path: Path,
    hard_cutoff: float | None,
    percentile: float | None,
    nms_iou_threshold: float,
    cross_class_iou: float,
    cross_class_score_gap: float,
    min_boxes: int,
) -> FilterResult:
    """Two-pass streaming path: never holds more than one image's records.

    Pass 1 collects only the post-cutoff scores (a flat float array) so the
    percentile threshold can be computed globally; pass 2 re-reads the file to
    filter, group by image, and merge.
    """
    print(f"Streaming raw predictions from {raw_pred_path} ...")
    if hard_cutoff is not None:
        print(
            f"Pass 1/2: applying hard score cutoff (score > {hard_cutoff}) "
            "and collecting scores ..."
        )
    else:
        print("Pass 1/2: collecting scores ...")

    n_raw, n_after_cutoff, cutoff_scores = stream_collect_scores(raw_pred_path, hard_cutoff)
    print(
        f"  Raw: {n_raw:,}"
        + (f"  →  after cutoff: {n_after_cutoff:,}" if hard_cutoff is not None else "")
    )

    if n_raw == 0:
        return FilterResult(0, 0, 0, 0, 0, None, {})
    if n_after_cutoff == 0:
        return FilterResult(n_raw, 0, 0, 0, 0, None, {})

    if percentile is not None:
        threshold = float(np.percentile(cutoff_scores, percentile))
        print(f"\n{percentile}th-percentile threshold = {threshold:.4f}")
    else:
        print("\nSkipping percentile quality filter (score_percentile is null).")
        threshold = None
    del cutoff_scores  # done with it before pass 2

    print(
        f"\nPass 2/2: filtering + within-class NMS (IoU={nms_iou_threshold}) "
        f"+ cross-class merge (IoU={cross_class_iou}, score gap={cross_class_score_gap}) ..."
    )
    print(
        f"  Overlapping cross-class boxes are suppressed only when score gap "
        f"> {cross_class_score_gap}; otherwise both are kept."
    )
    n_after_percentile, n_after_nms, n_final, merged_by_image = stream_filter_and_group(
        raw_pred_path,
        hard_cutoff,
        threshold,
        nms_iou_threshold,
        cross_class_iou,
        cross_class_score_gap,
        min_boxes,
    )
    print(f"  {n_after_cutoff:,} → {n_after_percentile:,} after percentile filter")
    print(f"  {n_after_percentile:,} → {n_after_nms:,} after within-class NMS")
    print(f"  {n_after_nms:,} → {n_final:,} final boxes across {len(merged_by_image):,} images")

    return FilterResult(
        n_raw, n_after_cutoff, n_after_percentile, n_after_nms, n_final, threshold, merged_by_image
    )


# ─── YOLO output ──────────────────────────────────────────────────────────────


def write_yolo_labels(
    merged_by_image: Dict[str, List[Dict[str, Any]]],
    yolo_labels_dir: Path,
) -> int:
    """Write per-image YOLO label .txt files.

    Parameters
    ----------
    merged_by_image:
        {image_path: [merged_detection_dict, ...]}, after cross-class merge.
    yolo_labels_dir:
        Directory to write <image_stem>.txt files into.

    Returns
    -------
    Total number of boxes written.
    """
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0

    for img_path_str, dets in merged_by_image.items():
        stem = Path(img_path_str).stem
        label_file = yolo_labels_dir / f"{stem}.txt"

        lines = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            img_w, img_h = det["image_w"], det["image_h"]
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(lines)

    return total_boxes


def write_confidence_weights(
    merged_by_image: Dict[str, List[Dict[str, Any]]],
    weights_json_path: Path,
) -> None:
    """Write confidence_weights.json with w^SAM3_i per box.

    Format
    ------
    {
        "/path/to/image.jpg": [
            {
                "class_id": 0,
                "class_name": "person",
                "cx": 0.5, "cy": 0.3, "w": 0.2, "h": 0.15,
                "w_sam3": 0.87
            },
            ...
        ],
        ...
    }
    """
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    output: Dict[str, List[Dict[str, Any]]] = {}

    for img_path_str, dets in merged_by_image.items():
        entries = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, det["image_w"], det["image_h"])
            entries.append(
                {
                    "class_id":   det["class_id"],
                    "class_name": det["class_name"],
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w":  round(w, 6),
                    "h":  round(h, 6),
                    "w_sam3": round(det["w_sam3"], 6),
                }
            )
        output[img_path_str] = entries

    with open(weights_json_path, "w") as f:
        json.dump(output, f, indent=2)


_IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"}


def _populate_labeled_images_dir(
    images_dir: Path,
    source_dir: Path,
    labels_dir: Path,
) -> int:
    """Create images_dir as a real directory and symlink only labeled images from source_dir.

    Only images that have a corresponding .txt in labels_dir receive a symlink,
    so images/ contains exactly the images Ultralytics will train on.

    If images_dir currently exists as a directory symlink (from the old layout),
    it is unlinked first so a real directory can be created in its place.

    Returns the number of symlinks created or confirmed.
    """
    if images_dir.is_symlink():
        images_dir.unlink()  # remove old directory symlink before creating real dir

    images_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src_file in source_dir.iterdir():
        if src_file.suffix.lower().lstrip(".") not in _IMG_FORMATS:
            continue
        if not (labels_dir / (src_file.stem + ".txt")).exists():
            continue  # no label for this image — skip
        link = images_dir / src_file.name
        if link.is_symlink():
            if link.resolve() == src_file.resolve():
                n += 1
                continue  # already correct
            link.unlink()  # stale — recreate
        elif link.exists():
            raise RuntimeError(
                f"Cannot create symlink at {link}: a real file already exists there. "
                "Remove it manually if intentional."
            )
        link.symlink_to(src_file.resolve())
        n += 1
    return n


def write_dataset_yaml(
    cfg: Dict[str, Any],
    dataset_yaml_path: Path,
) -> None:
    """Write a YOLO-compatible dataset.yaml for Ultralytics training.

    References the images/ directory directly.  images/ is a real directory
    (containing per-file symlinks to the source images), so Ultralytics'
    check_det_dataset calling .resolve() returns its own path unchanged —
    preserving the '/images/' segment that img2label_paths needs to locate
    the labels/ sibling directory.
    """
    dataset_root = dataset_yaml_path.parent.resolve()
    classes = cfg["classes"]

    doc = {
        "path": str(dataset_root),
        "train": "images",
        "val":   None,
        "nc":    len(classes),
        "names": [cls["name"] for cls in classes],
    }

    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ─── Summary printing ─────────────────────────────────────────────────────────


def print_summary(
    n_raw: int,
    n_after_percentile: int,
    n_after_nms: int,
    n_final: int,
    threshold: float | None,
    percentile: float | None,
    merged_by_image: Dict[str, List[Dict[str, Any]]],
    classes: List[Dict[str, Any]],
    hard_cutoff: float | None = None,
    n_after_cutoff: int | None = None,
) -> None:
    id_to_name = {cls["id"]: cls["name"] for cls in classes}
    class_counts: Dict[int, int] = defaultdict(int)
    for dets in merged_by_image.values():
        for det in dets:
            class_counts[det["class_id"]] += 1

    print("\n── pseudo-labeling Step B Summary ──────────────────────────────────────")
    print(f"  Raw predictions (Step A output)  : {n_raw:>8,}")
    if hard_cutoff is not None:
        print(f"  After hard cutoff (score>{hard_cutoff})    : {n_after_cutoff:>8,}")
    if percentile is not None:
        print(f"  After {percentile:g}th-percentile filter   : {n_after_percentile:>8,}  (threshold={threshold:.4f})")
    else:
        print(f"  After percentile filter (skipped): {n_after_percentile:>8,}")
    print(f"  After within-class NMS           : {n_after_nms:>8,}")
    print(f"  After cross-class merge (final)  : {n_final:>8,}")
    print(f"  Images with ≥1 box               : {len(merged_by_image):>8,}")
    print(f"\n  Per-class box counts (final):")
    for cls_id, count in sorted(class_counts.items()):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        print(f"    [{cls_id}] {name:<20s}: {count:>6,}")
    print("────────────────────────────────────────────────────────────────\n")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pseudo-labeling Step B: dual filter + YOLO export"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Override filtering.score_percentile from config (e.g. 90)",
    )
    parser.add_argument(
        "--hard-cutoff",
        type=float,
        default=None,
        help="Override filtering.hard_score_cutoff from config (e.g. 0.25)",
    )
    parser.add_argument(
        "--load-mode",
        choices=LOAD_MODES,
        default=None,
        help=(
            "How to read raw_predictions.json. 'memory' (default) json.load()s the "
            "whole file — simplest, but peak RAM is several times the file size. "
            "'stream' uses two constant-memory ijson passes, needed for multi-GB "
            "files, and assumes records are grouped by image (run_inference.py's "
            "write order). 'auto' picks stream above "
            f"{AUTO_STREAM_THRESHOLD_BYTES / 1024**3:.0f} GB. "
            "Overrides filtering.load_mode from config."
        ),
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after completion (overrides config shutdown_on_completion)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    filter_cfg = cfg.get("filtering", {})
    out_cfg = cfg["output"]

    load_mode = (
        args.load_mode if args.load_mode is not None
        else filter_cfg.get("load_mode", "memory")
    )

    percentile = (
        args.percentile if args.percentile is not None
        else filter_cfg.get("score_percentile", 85)
    )
    hard_cutoff = (
        args.hard_cutoff if args.hard_cutoff is not None
        else filter_cfg.get("hard_score_cutoff")
    )
    nms_iou_threshold = filter_cfg.get("nms_iou_threshold", 0.3)
    cross_class_iou = filter_cfg.get("cross_class_merge_iou", 0.5)
    cross_class_score_gap = filter_cfg.get("cross_class_score_gap", 0.3)
    min_boxes = filter_cfg.get("min_boxes_per_image", 0)

    raw_pred_path = Path(out_cfg["raw_predictions_path"])
    yolo_labels_dir = Path(out_cfg["yolo_labels_dir"])
    weights_json_path = Path(out_cfg["weights_json_path"])
    dataset_yaml_path = Path(out_cfg["dataset_yaml_path"])

    # ── Load + filter raw predictions ────────────────────────────────────────
    if not raw_pred_path.exists():
        raise FileNotFoundError(
            f"raw_predictions.json not found: {raw_pred_path}\n"
            "Run run_inference.py first."
        )

    mode = resolve_load_mode(load_mode, raw_pred_path)
    driver = filter_in_memory if mode == "memory" else filter_streaming
    result = driver(
        raw_pred_path,
        hard_cutoff,
        percentile,
        nms_iou_threshold,
        cross_class_iou,
        cross_class_score_gap,
        min_boxes,
    )

    n_raw = result.n_raw
    n_after_cutoff = result.n_after_cutoff
    n_after_percentile = result.n_after_percentile
    n_after_nms = result.n_after_nms
    n_final = result.n_final
    threshold = result.threshold
    merged_by_image = result.merged_by_image

    if n_raw == 0:
        print("No predictions to filter. Exiting.")
        return
    if n_after_cutoff == 0:
        print("No predictions survive the hard cutoff. Exiting.")
        return

    # ── Step 4: Write YOLO labels ────────────────────────────────────────────
    print(f"\nWriting YOLO label files to {yolo_labels_dir} ...")
    total_boxes = write_yolo_labels(merged_by_image, yolo_labels_dir)
    print(f"  {total_boxes:,} boxes written across {len(merged_by_image):,} images.")

    # ── Step 4b: Populate pseudolabeled_data/images/ with per-file symlinks ───
    # images/ is a real directory so Ultralytics .resolve() returns its own path
    # unchanged, preserving '/images/' for img2label_paths → labels/ resolution.
    # Only labeled images get a symlink, matching the label set exactly.
    images_dir = yolo_labels_dir.parent / "images"
    image_dir = Path(cfg["data"]["image_dir"]).resolve()
    print(f"Populating images/ dir: {images_dir} (symlinks → {image_dir}) ...")
    n_imgs = _populate_labeled_images_dir(images_dir, image_dir, yolo_labels_dir)
    print(f"  {n_imgs:,} per-file symlinks written.")

    # ── Step 5: Write confidence weights ─────────────────────────────────────
    print(f"Writing confidence_weights.json → {weights_json_path} ...")
    write_confidence_weights(merged_by_image, weights_json_path)

    # ── Step 6: Write dataset.yaml ───────────────────────────────────────────
    print(f"Writing dataset.yaml → {dataset_yaml_path} ...")
    write_dataset_yaml(cfg, dataset_yaml_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(
        n_raw=n_raw,
        n_after_percentile=n_after_percentile,
        n_after_nms=n_after_nms,
        n_final=n_final,
        threshold=threshold,
        percentile=percentile,
        merged_by_image=merged_by_image,
        classes=cfg["classes"],
        hard_cutoff=hard_cutoff,
        n_after_cutoff=n_after_cutoff,
    )

    print("pseudo-labeling Step B complete.")
    print(f"  Pseudolabeled data root: {yolo_labels_dir.parent}/")
    print(f"    images/            : {images_dir}/  (real dir, {n_imgs:,} per-file symlinks)")
    print(f"    labels/            : {yolo_labels_dir}/")
    print(f"    confidence_weights : {weights_json_path}")
    print(f"    dataset.yaml       : {dataset_yaml_path}")
    stale_txt = yolo_labels_dir.parent / "train_images.txt"
    if stale_txt.exists():
        print(f"\nStale file from old layout (safe to delete): {stale_txt}")

    maybe_shutdown(cfg, force=args.shutdown)


if __name__ == "__main__":
    main()
