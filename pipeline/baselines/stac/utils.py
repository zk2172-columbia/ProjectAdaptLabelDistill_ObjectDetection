"""utils.py — the only shared helpers baselines/stac needs.

Copied (not imported) from pseudo_labeling/utils.py and student_training/utils.py so this directory has no
dependency on the SAM3 pipeline's phases. The copies are trimmed to what STAC uses: the
cross-class merge, the percentile filter, the masking helpers, the SAM3 BPE path lookup and
the confidence-weight loaders are all absent, because STAC has no use for any of them.

One deliberate change from the pseudo_labeling original: `nms` sorts STABLY. See its docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ─── Config ───────────────────────────────────────────────────────────────────


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
    """Load stac_config.yaml, inline `classes_file:`, and resolve every path key to absolute.

    Paths in the config are written relative to this directory, which is the only sane way to
    write them; resolving here means no downstream function ever has to know where the config
    lived. `classes:` is merged in from `classes_file:` (pipeline/data/classes/<ds>.yaml), shared with the
    rest of the repo so class ids cannot drift.
    """
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    text = config_path.read_text()
    _reject_placeholders(config_path, text)
    cfg: Dict[str, Any] = yaml.safe_load(text)

    if cfg.get("classes_file") and not cfg.get("classes"):
        classes_file = _abs(cfg["classes_file"], config_dir)
        with open(classes_file) as f:
            cfg["classes"] = yaml.safe_load(f)["classes"]

    for section, keys in (
        ("teacher", ("checkpoint",)),
        ("data", ("unlabeled_coco_json", "image_dir", "gt_train_images", "gt_val_images",
                  "eval_dataset_yaml", "eval_gt_json")),
        ("output", ("root",)),
    ):
        block = cfg.get(section) or {}
        for key in keys:
            if block.get(key):
                block[key] = str(_abs(block[key], config_dir))

    cfg["_config_dir"] = str(config_dir)
    return cfg


def _abs(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def class_names(cfg: Dict[str, Any]) -> List[str]:
    """Class names ordered by id, as ultralytics' `names:` list wants them."""
    return [c["name"] for c in sorted(cfg["classes"], key=lambda c: c["id"])]


# ─── COCO image list ──────────────────────────────────────────────────────────


def load_coco_image_list(
    coco_json: str | Path,
    image_dir: str | Path,
) -> List[Dict[str, Any]]:
    """The images a COCO json names, resolved against *image_dir*.

    Used twice, and it must return the same list both times: the teacher iterates it to produce
    predictions, and the filter iterates it to guarantee EVERY unlabeled image ends up with a
    label file — including the ones where nothing survived the threshold.
    """
    with open(coco_json) as f:
        data = json.load(f)

    root = Path(image_dir)
    return [
        {
            "image_id": img["id"],
            "file_name": img["file_name"],
            "file_path": root / img["file_name"],
            "width": img.get("width"),
            "height": img.get("height"),
        }
        for img in data["images"]
    ]


# ─── Boxes ────────────────────────────────────────────────────────────────────


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Pixel-space xyxy -> YOLO normalized (cx, cy, w, h), each clamped to [0, 1]."""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(w, 0.0, 1.0)),
        float(np.clip(h, 0.0, 1.0)),
    )


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-Union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def nms(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float,
) -> List[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first.

    STABLE by design, unlike the pseudo_labeling original this was copied from: ties are broken by input
    order rather than by whatever `argsort()[::-1]` happens to produce. A detector emitting two
    boxes at the same score is not exotic (quantized/fp16 outputs collide), and with an unstable
    sort the pseudo-label set could differ between two runs of the same command — which would
    quietly break the seed-42 reproducibility this pipeline is supposed to have.

    A box's fate depends only on HIGHER-scoring boxes, which is why applying a score threshold
    before or after this is equivalent. STAC specifies NMS first, so pseudolabels.py --stage filter does
    NMS first.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.asarray(boxes, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_arr, kind="stable")

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array(
            [iou(boxes_arr[i].tolist(), boxes_arr[int(j)].tolist()) for j in rest]
        )
        order = rest[ious < iou_threshold]

    return keep
