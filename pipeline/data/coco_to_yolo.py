#!/usr/bin/env python3
"""
coco_to_yolo.py — Convert a single COCO-annotation-format dataset into Ultralytics YOLO format
================================================================================================

Thin wrapper around `ultralytics.data.converter.convert_coco` for the common case: one
`_annotations.coco.json` + its source image directory in, one YOLO-ready dataset directory out.

Output structure
----------------
    <dst>/
      images/        <- real directory of per-file symlinks into --image-dir
      labels/        <- one .txt per image, YOLO cxcywh format (written by convert_coco)
      dataset.yaml   <- path, train: images, nc, names

`dataset.yaml`'s `train:` key is a placeholder — for multi-split datasets, run this script once
per split (into separate --dst directories) and hand-edit a combined yaml that points `train:`
and `val:` at the respective `images/` directories, the same way
`pipeline/student_training/prepare_gt_data.py` assembles `gt_dataset.yaml`.

Category id requirement
------------------------
convert_coco (with cls91to80=False) assigns YOLO class id = category_id - 1. This is only
correct if COCO category ids are a contiguous range starting at 1. This script validates that
up front and errors out with a clear message otherwise, rather than silently writing wrong
class ids.

Class ordering vs. the dataset's class list
-----------------------------------------
The COCO json's category order is whatever the source export used (Roboflow, in particular,
orders categories ALPHABETICALLY, not in the class list's order). Left alone, that
means the class id baked into the written .txt labels (0=alphabetically-first category, ...)
silently disagrees with every checkpoint trained against classes.yaml's id order — the boxes
still localize fine, but every prediction gets scored against the wrong class. This script
remaps each label's class id from the COCO json's order to classes.yaml's id order (matching
by name) before writing the .txt files, and writes dataset.yaml's `names:` in classes.yaml's
order too, so a converted dataset's ids always line up with the rest of the pipeline. Every
category name in the COCO json must exist in classes.yaml — otherwise this fails loudly
rather than writing a dataset with an undetectable numbering mismatch.

Usage
-----
    python coco_to_yolo.py --coco-json path/to/_annotations.coco.json \\
                            --image-dir path/to/images \\
                            --dst path/to/output_dataset

    # Non-default class list:
    python coco_to_yolo.py --coco-json ... --image-dir ... --dst ... \\
                            --classes-file path/to/classes.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics.data.converter import convert_coco

IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"}

DEFAULT_CLASSES_FILE = Path(__file__).resolve().parent.parent / "classes" / "garbage-classification.yaml"


def _validate_contiguous_category_ids(coco: dict) -> list[str]:
    """Return category names sorted by id, after checking ids are a contiguous 1..N range.

    Raises ValueError with a clear message if ids are not contiguous starting at 1, since
    convert_coco's cls91to80=False path assumes `yolo_class_id = category_id - 1`.

    Roboflow COCO exports commonly include an id=0 "root" supercategory placeholder
    (e.g. name "GARBAGE-GARBAGE-CLASSIFICATION") that no annotation ever references —
    real classes start at id 1. That placeholder is dropped before the contiguity check
    so such datasets validate correctly.
    """
    categories = coco["categories"]
    ids_in_use = {ann["category_id"] for ann in coco["annotations"]}
    unused_root = [cat for cat in categories if cat["id"] == 0 and 0 not in ids_in_use]
    if unused_root:
        categories = [cat for cat in categories if cat["id"] != 0]

    ids = sorted(cat["id"] for cat in categories)
    expected = list(range(1, len(categories) + 1))
    if ids != expected:
        raise ValueError(
            f"Category ids {ids} are not a contiguous range starting at 1 ({expected}). "
            "convert_coco assumes yolo_class_id = category_id - 1, so non-contiguous or "
            "non-1-indexed category ids would silently produce wrong class ids. Remap "
            "category ids in the source COCO json before converting."
        )
    by_id = {cat["id"]: cat["name"] for cat in categories}
    return [by_id[i] for i in ids]


def _load_canonical_names(classes_file: Path) -> list[str]:
    """Return the class list's names, ordered by id (index i = class id i)."""
    with open(classes_file) as f:
        doc = yaml.safe_load(f)
    by_id = {c["id"]: c["name"] for c in doc["classes"]}
    ids = sorted(by_id)
    if ids != list(range(len(ids))):
        raise ValueError(f"{classes_file} class ids are not a contiguous 0..N-1 range: {ids}")
    return [by_id[i] for i in ids]


def _build_remap(coco_order_names: list[str], canonical_names: list[str]) -> dict[int, int]:
    """Map COCO-order YOLO class id -> classes.yaml id, matching by name (case-insensitive).

    Raises ValueError naming any COCO category absent from classes.yaml, since silently
    dropping or mis-mapping a class would corrupt every label id downstream.
    """
    lookup = {name.lower(): i for i, name in enumerate(canonical_names)}
    remap: dict[int, int] = {}
    missing: list[str] = []
    for coco_id, name in enumerate(coco_order_names):
        canonical_id = lookup.get(name.lower())
        if canonical_id is None:
            missing.append(name)
        else:
            remap[coco_id] = canonical_id
    if missing:
        raise ValueError(
            f"COCO categories not found in classes.yaml: {missing}. "
            f"classes.yaml has: {canonical_names}. "
            "Add the missing class(es) to classes.yaml (or fix the name mismatch) before "
            "converting — otherwise the written label ids would not mean what the rest of "
            "the pipeline expects them to mean."
        )
    return remap


def _remap_label_ids(labels_dir: Path, remap: dict[int, int]) -> None:
    """Rewrite each label .txt's leading class id in place per `remap`."""
    for txt in labels_dir.glob("*.txt"):
        lines = txt.read_text().splitlines()
        out_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            coco_id = int(parts[0])
            parts[0] = str(remap[coco_id])
            out_lines.append(" ".join(parts))
        txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def _populate_images_dir(images_dir: Path, source_dir: Path) -> None:
    """Create images_dir as a real directory of per-file symlinks into source_dir.

    Mirrors prepare_gt_data.py's approach: images_dir must be a real directory (not itself a
    symlink), because ultralytics' check_det_dataset resolves paths with .resolve(), which
    follows a directory symlink and would otherwise break img2label_paths.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    for src_file in source_dir.iterdir():
        if src_file.suffix.lower().lstrip(".") not in IMG_FORMATS:
            continue
        link = images_dir / src_file.name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(src_file.resolve())


def convert(coco_json: Path, image_dir: Path, dst: Path, classes_file: Path = DEFAULT_CLASSES_FILE) -> None:
    with open(coco_json) as f:
        coco = json.load(f)
    coco_order_names = _validate_contiguous_category_ids(coco)
    canonical_names = _load_canonical_names(classes_file)
    remap = _build_remap(coco_order_names, canonical_names)
    if any(coco_id != canonical_id for coco_id, canonical_id in remap.items()):
        print(f"Remapping COCO class order {coco_order_names} -> classes.yaml order {canonical_names}")

    dst.mkdir(parents=True, exist_ok=True)

    # convert_coco scans *every* json in labels_dir, so stage a copy of just this one file
    # in an isolated temp dir to avoid pulling in sibling split files (e.g. a train json
    # sitting next to an eval json in the same source directory).
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_parent:
        shutil.copy2(coco_json, Path(tmp_in) / coco_json.name)
        # convert_coco calls increment_path(save_dir), which bumps to a sibling path if
        # save_dir already exists — so pass a not-yet-created subdirectory, not tmp_parent
        # itself (TemporaryDirectory() always pre-creates its path).
        tmp_out = Path(tmp_parent) / "conv"
        convert_coco(
            labels_dir=tmp_in,
            save_dir=str(tmp_out),
            use_segments=False,
            use_keypoints=False,
            cls91to80=False,
        )
        # convert_coco writes to <tmp_out>/labels/<json_stem>/*.txt
        labels_src = next((tmp_out / "labels").iterdir())
        labels_dst = dst / "labels"
        labels_dst.mkdir(parents=True, exist_ok=True)
        for txt in labels_src.rglob("*.txt"):
            shutil.move(str(txt), labels_dst / txt.name)

    _remap_label_ids(labels_dst, remap)
    _populate_images_dir(dst / "images", image_dir)

    dataset_yaml = {
        "path": str(dst.resolve()),
        "train": "images",
        "nc": len(canonical_names),
        "names": canonical_names,
    }
    with open(dst / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Converted {len(coco['images'])} images, {len(coco_order_names)} classes -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-json", required=True, type=Path, help="Path to _annotations.coco.json")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing the source images")
    parser.add_argument("--dst", required=True, type=Path, help="Output directory for the YOLO dataset")
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=DEFAULT_CLASSES_FILE,
        help=f"classes.yaml to remap label ids against (default: {DEFAULT_CLASSES_FILE})",
    )
    args = parser.parse_args()

    convert(args.coco_json.resolve(), args.image_dir.resolve(), args.dst.resolve(), args.classes_file.resolve())


if __name__ == "__main__":
    main()
