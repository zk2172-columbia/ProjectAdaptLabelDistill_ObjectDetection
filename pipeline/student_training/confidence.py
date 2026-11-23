# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""Strict validation of portable pseudo-label confidence metadata."""
import json
import math
from pathlib import Path


def load_confidence(path):
    with open(path) as f:
        raw = json.load(f, object_pairs_hook=_unique_pairs)
    if not isinstance(raw, dict):
        raise ValueError("Confidence metadata must be an image-path mapping")
    result = {}
    for image, boxes in raw.items():
        stem = Path(image).stem
        if stem in result:
            raise ValueError(f"Ambiguous image stem in confidence metadata: {stem}")
        if not isinstance(boxes, list):
            raise ValueError(f"Expected a box list for {image}")
        validated = []
        for box in boxes:
            values = {k: float(box[k]) for k in ("cx", "cy", "w", "h", "w_sam3")}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite confidence/coordinate for {image}")
            if not 0 <= values["w_sam3"] <= 1:
                raise ValueError(f"Confidence outside [0,1] for {image}")
            if not all(0 <= values[k] <= 1 for k in ("cx", "cy", "w", "h")):
                raise ValueError(f"Non-normalized box for {image}")
            cls = float(box["class_id"])
            if not math.isfinite(cls) or cls < 0 or not cls.is_integer():
                raise ValueError(f"Invalid class id for {image}")
            validated.append({**values, "class_id": int(cls)})
        result[stem] = validated
    return result


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result
