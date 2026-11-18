#!/usr/bin/env python3
"""
prepare_data.py — student training pre-step: convert COCO annotations to YOLO format
====================================================================================

Two modes over the same COCO -> YOLO conversion:

  (default)           the GT splits named in config.yaml's `gt:` block — train, val and
                      the held-out eval — into yolo_data/<run_name>/ + gt_dataset.yaml.
                      This is the mode config.yaml's `steps:` list runs.

  --human-annotated   ALL FOUR splits including `label/`, read from their own human COCO
                      annotations instead of pseudo-labeling's SAM3 pseudo-labels, into a separate
                      yolo_data entry + human_annotated_dataset.yaml. The upper reference point
                      for the pipeline: the exact image set a `gt_pl_*` arm trains on, but with
                      the pool carrying human labels. The gap between student training's `gt_only` arm and
                      a run on this dataset is the headroom pseudo-labeling is chasing.
                      This mode deliberately reads NO config.yaml, whose `gt:` block is
                      shared by every sweep and has fixed per-split keys with no room for
                      `label/`. It must never target a pseudo-labeling / student training entry, whose `label/`
                      holds SAM3 pseudo-labels; point --output-dir at a separate entry.

Output structure (per split, both modes)
----------------------------------------
    yolo_data/<entry>/
      <split>/
        images/  <-  real directory of per-file symlinks to the source images
        labels/  <-  one .txt per image, YOLO cxcywh format
      <mode's dataset yaml>

train = GT training images. val = validation during training. eval = true held-out set, written
to the yaml's `test:` key (ultralytics' convention for a held-out evaluation split).
--human-annotated additionally converts `label/` and lists it beside train/ under `train:`.

NOTE on the real-directory images/ approach:
  Ultralytics check_det_dataset resolves paths with .resolve() which follows symlinks.
  If train: references an images/ directory that is itself a symlink, the symlink is
  followed and the resolved path loses the '/images/' segment, so img2label_paths
  cannot find labels.  Making images/ a REAL directory (containing per-file symlinks
  to source images) sidesteps this: .resolve() on a real directory returns its own
  path unchanged, preserving '/images/' for img2label_paths.

Usage
-----
    cd pipeline/student_training
    python prepare_data.py --config config.yaml
    python prepare_data.py --human-annotated
    python prepare_data.py --human-annotated --output-dir ../yolo_data/my-entry
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from utils import _relative_to, load_config


# ─── COCO → YOLO mapping ──────────────────────────────────────────────────────


def build_coco_to_yolo_map(
    classes_cfg: List[Dict],
    coco_categories: List[Dict],
) -> Dict[int, int]:
    """Build a {coco_category_id: yolo_class_id} mapping from config and COCO metadata.

    Matching is case-insensitive, so COCO names like "PLASTIC" match config
    names like "plastic".  Categories whose lowercase name is not in the config
    (e.g. the root "GARBAGE-GARBAGE-CLASSIFICATION" node) are silently skipped.
    """
    name_to_yolo_id = {cls["name"].lower(): cls["id"] for cls in classes_cfg}
    return {
        cat["id"]: name_to_yolo_id[cat["name"].lower()]
        for cat in coco_categories
        if cat["name"].lower() in name_to_yolo_id
    }


# ─── Images directory helper ───────────────────────────────────────────────────

IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"}


def _populate_images_dir(
    images_dir: Path,
    source_dir: Path,
    split_filenames: Set[str],
) -> None:
    """Create images_dir as a real directory and symlink this split's images into it.

    Only files named in *split_filenames* (the file_name entries of this split's
    COCO JSON) are linked.  source_dir can also hold images left behind by an
    earlier split assignment.  Linking those in would hand Ultralytics an image
    with no corresponding labels/*.txt,
    which it silently reads as "this image contains zero objects" — turning a
    real annotated image into a phantom background sample (and, in the eval
    split, into pure false-positive pressure).  The COCO JSON is the authority
    on split membership; the directory listing is not.

    If images_dir currently exists as a directory symlink (from the old layout),
    it is unlinked first so a real directory can be created in its place.

    Handles three cases per file:
    - Correct symlink already exists → no-op.
    - Stale symlink (wrong target) → unlink and recreate.
    - Real file at that path → raise RuntimeError (never clobber real data).

    Links already in images_dir that are not in *split_filenames* are pruned, so
    re-running over a directory populated by the previous unfiltered behaviour
    repairs it in place instead of leaving the bad links behind.
    """
    if images_dir.is_symlink():
        images_dir.unlink()  # remove old directory symlink before creating real dir

    images_dir.mkdir(parents=True, exist_ok=True)

    # Prune links this split no longer claims (repairs dirs written before the
    # split_filenames filter existed). Only symlinks are removed — a real file
    # here is unexpected and left alone for the loop below to report.
    for existing in images_dir.iterdir():
        if existing.is_symlink() and existing.name not in split_filenames:
            existing.unlink()

    for src_file in source_dir.iterdir():
        if src_file.suffix.lower().lstrip(".") not in IMG_FORMATS:
            continue
        if src_file.name not in split_filenames:
            continue  # belongs to a different split — see docstring
        link = images_dir / src_file.name
        if link.is_symlink():
            if link.resolve() == src_file.resolve():
                continue  # already correct
            link.unlink()  # stale — recreate
        elif link.exists():
            raise RuntimeError(
                f"Cannot create symlink at {link}: a real file already exists there. "
                "Remove it manually if intentional."
            )
        link.symlink_to(src_file.resolve())


# ─── Conversion ───────────────────────────────────────────────────────────────


def convert_split(
    coco_json_path: Path,
    image_dir: Path,
    labels_out_dir: Path,
    images_dir: Path,
    coco_to_yolo: Dict[int, int],
) -> Tuple[int, int, int]:
    """Convert one COCO split to YOLO .txt label files.

    Parameters
    ----------
    coco_json_path:
        Path to the COCO-format _annotations.coco.json for this split.
    image_dir:
        Directory containing the actual image files (source for symlinks).
    labels_out_dir:
        Directory where per-image YOLO .txt files are written.
    images_dir:
        Real directory into which per-file symlinks to image_dir are created.
    coco_to_yolo:
        {coco_category_id: yolo_class_id} mapping from build_coco_to_yolo_map().

    Returns
    -------
    (n_images, n_annotations_converted, n_annotations_skipped)

    `iscrowd: 1` annotations count towards skipped. They are IGNORE REGIONS, not objects:
    COCOeval discards a detection matched to one instead of scoring it either way, and a YOLO
    label file has no way to say that — a line there is a training target, and one ultralytics'
    own val() would then score the model on. `data/prepare_ground_truth.py` keeps them
    out of its YOLO labels for the same reason, while keeping them in the COCO ground truth.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    anns_by_image: Dict[int, List] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    labels_out_dir.mkdir(parents=True, exist_ok=True)

    n_converted = 0
    n_skipped = 0
    n_ignore = 0

    split_filenames: Set[str] = {Path(img["file_name"]).name for img in coco["images"]}

    for img in coco["images"]:
        img_id = img["id"]
        img_w: int = img["width"]
        img_h: int = img["height"]
        stem = Path(img["file_name"]).stem

        lines: List[str] = []
        for ann in anns_by_image.get(img_id, []):
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in coco_to_yolo:
                n_skipped += 1
                continue
            if ann.get("iscrowd", 0):  # ignore region, not an object — see docstring
                n_ignore += 1
                n_skipped += 1
                continue
            yolo_class_id = coco_to_yolo[coco_cat_id]

            # COCO bbox: [x_min, y_min, width, height] in pixels
            x_min, y_min, bw, bh = ann["bbox"]
            cx = (x_min + bw / 2.0) / img_w
            cy = (y_min + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h
            lines.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_converted += 1

        label_text = "\n".join(lines) + ("\n" if lines else "")
        (labels_out_dir / f"{stem}.txt").write_text(label_text)

    _populate_images_dir(images_dir, image_dir, split_filenames)

    # images/ and labels/ must correspond 1:1 — a shortfall means the COCO JSON
    # names an image that isn't on disk, which Ultralytics would quietly train on
    # as a smaller dataset rather than error on.
    n_linked = sum(1 for p in images_dir.iterdir() if not p.name.startswith("."))
    if n_linked != len(split_filenames):
        missing = sorted(split_filenames - {p.name for p in images_dir.iterdir()})
        raise RuntimeError(
            f"{coco_json_path.parent.name} split: linked {n_linked} images but the COCO JSON "
            f"lists {len(split_filenames)}. Missing from {image_dir}: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    if n_ignore:
        print(f"    ({n_ignore:,} of the skipped are iscrowd ignore regions, which have no "
              f"YOLO representation)")

    return len(coco["images"]), n_converted, n_skipped


# ─── Dataset yaml ─────────────────────────────────────────────────────────────


def write_gt_dataset_yaml(
    gt_data_root: Path,
    classes_cfg: List[Dict],
    output_path: Path,
) -> None:
    """Write gt_dataset.yaml referencing train/images, val/images, eval/images directly.

    images/ inside each split is a real directory (not a symlink), so
    check_det_dataset's .resolve() call returns the path unchanged and
    img2label_paths can correctly locate the labels/ sibling directory.

    `eval/` (the true held-out final-model evaluation split) is written under
    ultralytics' `test:` key — the conventional slot for a held-out set that's
    evaluated once, distinct from `val:` which is used during training.

    No `path:` key is written: the yaml sits in gt_data_root itself, and check_det_dataset
    falls back to the yaml's own directory when `path:` is absent, so the split entries below
    resolve correctly on any machine. An explicit relative `path:` would NOT work — ultralytics
    resolves that one against its global DATASETS_DIR.
    """
    doc = {
        "train": "train/images",
        "val":   "val/images",
        "test":  "eval/images",
        "nc":    len(classes_cfg),
        "names": [cls["name"] for cls in classes_cfg],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ─── Main ─────────────────────────────────────────────────────────────────────


SCRIPT_DIR = Path(__file__).resolve().parent

SPLITS = ["train", "val", "eval"]

# --human-annotated additionally converts `label/`, the pool a gt_pl_* arm pseudo-labels.
HUMAN_SPLITS = SPLITS + ["label"]
HUMAN_TRAIN_SPLITS = ["train", "label"]  # the two halves of its training set


def load_classes(classes_file: Path) -> List[Dict]:
    """Read the `classes:` list out of classes.yaml.

    Read directly rather than through utils.load_config, which expects a full phase
    config with a `classes_file:` key to merge; --human-annotated has no config of its own.
    """
    with open(classes_file) as f:
        doc = yaml.safe_load(f)
    return doc["classes"]


def write_human_annotated_yaml(
    data_root: Path,
    classes_cfg: List[Dict],
    output_path: Path,
) -> None:
    """Write the dataset yaml pointing train: at both train/images and label/images.

    Same shape as write_gt_dataset_yaml — no `path:` key, entries relative to the yaml's own
    directory — except that `train:` is a two-element list. Ultralytics resolves a list
    entry-by-entry against that same root.
    """
    base = output_path.parent.resolve()
    doc = {
        "train": [_relative_to(data_root / s / "images", base) for s in HUMAN_TRAIN_SPLITS],
        "val":   _relative_to(data_root / "val" / "images", base),
        "test":  _relative_to(data_root / "eval" / "images", base),
        "nc":    len(classes_cfg),
        "names": [cls["name"] for cls in classes_cfg],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


def run_human_annotated(args: argparse.Namespace) -> None:
    """Convert all four splits from their own human COCO annotations."""

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()

    split_root  = _resolve(args.split_root)
    data_root   = _resolve(args.output_dir)
    classes_cfg = load_classes(_resolve(args.classes_file))
    yaml_out    = data_root / args.yaml_name

    coco_jsons = {split: split_root / split / "_annotations.coco.json" for split in HUMAN_SPLITS}
    image_dirs = {split: split_root / split for split in HUMAN_SPLITS}

    missing = [str(p) for p in coco_jsons.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing COCO annotation file(s):\n  " + "\n  ".join(missing))

    print(f"Split root : {split_root}")
    print(f"Output     : {data_root}")

    total_imgs = total_conv = total_skip = 0

    for split in HUMAN_SPLITS:
        print(f"\nConverting {split} split: {coco_jsons[split]} ...")

        # Rebuild the map from this split's own categories rather than assuming the
        # train split's ids carry over.  The COCO categories are alphabetical
        # (1 BIODEGRADABLE ... 6 PLASTIC) while classes.yaml is 0 plastic ... 5 paper,
        # so the match is by lowercase name; the root GARBAGE-GARBAGE-CLASSIFICATION
        # node has no counterpart and is dropped by build_coco_to_yolo_map.
        with open(coco_jsons[split]) as f:
            categories = json.load(f)["categories"]
        coco_to_yolo = build_coco_to_yolo_map(classes_cfg, categories)
        print(f"  COCO->YOLO map: {coco_to_yolo}")

        n_imgs, n_conv, n_skip = convert_split(
            coco_json_path=coco_jsons[split],
            image_dir=image_dirs[split],
            labels_out_dir=data_root / split / "labels",
            images_dir=data_root / split / "images",
            coco_to_yolo=coco_to_yolo,
        )
        print(f"  {n_imgs:,} images | {n_conv:,} boxes converted | {n_skip:,} skipped")
        total_imgs += n_imgs
        total_conv += n_conv
        total_skip += n_skip

    print(f"\nWriting {args.yaml_name} -> {yaml_out} ...")
    write_human_annotated_yaml(data_root, classes_cfg, yaml_out)

    n_train = sum(1 for s in HUMAN_TRAIN_SPLITS for _ in (data_root / s / "images").iterdir())
    print("\nHuman-annotated data preparation complete.")
    print(f"  Data root  : {data_root}/")
    print(f"  Totals     : {total_imgs:,} images | {total_conv:,} boxes | {total_skip:,} skipped")
    print(f"  train:     : {n_train:,} images ({' + '.join(HUMAN_TRAIN_SPLITS)})")
    print(f"  dataset yaml: {yaml_out}")
    print(
        "\nNext step:\n"
        "  cd pipeline && python train_baseline_yolo.py \\\n"
        f"      --data {yaml_out} --run-name all-label-yolo \\\n"
        "      --epochs 100 --patience 100 --seed 42"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="student training pre-step: convert COCO annotations to YOLO format"
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (required unless --human-annotated)",
    )
    parser.add_argument(
        "--human-annotated",
        action="store_true",
        help="Convert all four splits (incl. label/) from their own human COCO annotations",
    )
    parser.add_argument(
        "--split-root",
        default="../../experiment_files/finetuning-data/garbage-classification",
        help="--human-annotated: dir holding train/ val/ eval/ label/, each with _annotations.coco.json",
    )
    parser.add_argument(
        "--output-dir",
        default="../yolo_data/5-85-10-gt-human-seed42",
        help="--human-annotated: yolo_data entry to write (must not be a pseudo-labeling pseudo-label entry)",
    )
    parser.add_argument("--classes-file", default="../classes/garbage-classification.yaml",
                        help="--human-annotated: classes.yaml to read the class order from")
    parser.add_argument("--yaml-name", default="human_annotated_dataset.yaml",
                        help="--human-annotated: name of the dataset yaml to write")
    args = parser.parse_args()

    if args.human_annotated:
        run_human_annotated(args)
        return
    if not args.config:
        parser.error("--config is required unless --human-annotated is given")

    # load_config merges a `classes_file:` reference into cfg["classes"]; raw yaml.safe_load
    # would leave classes unresolved and KeyError below.
    cfg = load_config(args.config)

    config_dir = Path(args.config).resolve().parent

    gt_cfg      = cfg["gt"]
    classes_cfg = cfg["classes"]

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (config_dir / p).resolve()

    gt_data_root     = _resolve(gt_cfg["output_dir"])
    dataset_yaml_out = gt_data_root / "gt_dataset.yaml"

    coco_jsons = {split: _resolve(gt_cfg[f"{split}_coco_json"]) for split in SPLITS}
    image_dirs = {split: _resolve(gt_cfg[f"{split}_image_dir"]) for split in SPLITS}

    # Build COCO→YOLO map from the train split categories (same in all splits)
    with open(coco_jsons["train"]) as f:
        coco_meta = json.load(f)
    coco_to_yolo = build_coco_to_yolo_map(classes_cfg, coco_meta["categories"])

    print(f"COCO→YOLO class map: {coco_to_yolo}")

    # ── Convert each split ────────────────────────────────────────────────────
    for split in SPLITS:
        print(f"\nConverting {split} split: {coco_jsons[split]} ...")
        n_imgs, n_conv, n_skip = convert_split(
            coco_json_path=coco_jsons[split],
            image_dir=image_dirs[split],
            labels_out_dir=gt_data_root / split / "labels",
            images_dir=gt_data_root / split / "images",
            coco_to_yolo=coco_to_yolo,
        )
        print(f"  {n_imgs:,} images | {n_conv:,} boxes converted | {n_skip:,} skipped")

    # ── Write gt_dataset.yaml ─────────────────────────────────────────────────
    print(f"\nWriting gt_dataset.yaml -> {dataset_yaml_out} ...")
    write_gt_dataset_yaml(gt_data_root, classes_cfg, dataset_yaml_out)

    print("\nGT data preparation complete.")
    print(f"  GT data root : {gt_data_root}/")
    for split in SPLITS:
        print(f"    {split}/images/ : {gt_data_root / split / 'images'}/  (real dir, per-file symlinks)")
        print(f"    {split}/labels/ : {gt_data_root / split / 'labels'}/")
    print(f"  gt_dataset.yaml: {dataset_yaml_out}")
    print("\nNext step: python train.py --config config.yaml")


if __name__ == "__main__":
    main()
