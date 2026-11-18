#!/usr/bin/env python3
"""prepare_ground_truth.py — export a finished split into pipeline/data/ground_truths/<dataset>/.

Runs AFTER a data strategy has partitioned a dataset. Takes the four split directories
that strategy wrote (each holding `_annotations.coco.json` plus symlinks to its own images) and
re-exports every split in the two formats the rest of this repo consumes:

    ground_truths/<dataset>/
      manifest.json                 provenance: source paths + sha256, per-split/per-class counts
      classes.yaml                  frozen copy of the class order used for the remap
      dataset.yaml                  train: / val: / test: -> eval/images
      <split>/
        annotations.coco.json       COCO, categories in classes.yaml order, ids 1..N
        images/                     REAL directory of per-file symlinks
        labels/*.txt                YOLO cxcywh, class ids in classes.yaml order

WHY THIS EXISTS
---------------
The held-out COCO json is the yardstick every arm in this repo is scored against (see
pipeline/EVALUATION.md §3). Before this script it was a single file sitting inside one run's
output directory with no code that produced it — unversioned, unverifiable, and impossible to
regenerate for a new split. Everything a COCOeval number depends on is now built by one
deterministic step whose inputs are recorded in manifest.json.

CATEGORY ORDER IS THE WHOLE GAME
--------------------------------
A Roboflow export orders categories ALPHABETICALLY and prepends a dummy id-0 supercategory node.
`classes.yaml` orders them by the literal YOLO class index every checkpoint in this repo was
trained against. Left alone, the two disagree and every prediction is scored against the wrong
class — boxes still localize fine, so the only symptom is a quietly collapsed mAP. This script
remaps both outputs into classes.yaml order:

    YOLO label class id  =  classes.yaml id            (0-based)
    COCO category_id     =  classes.yaml id + 1        (1-based, contiguous)

Matching is by name, case-insensitively. A source category with no classes.yaml counterpart (the
dummy supercategory) is dropped and its annotations counted as skipped. A classes.yaml entry the
source never mentions is reported — it means the class list and the dataset disagree.

IGNORE REGIONS
--------------
An annotation with `iscrowd: 1` is written to the COCO json and NOT to the YOLO label file. It
marks a region COCOeval discards detections against instead of scoring them either way, and a
YOLO label file has no way to express that — a line there is a training target, and one ultralytics'
own val() would score the model on. `manifest.json` counts these separately as
`n_ignore_annotations`, leaving `per_class` a count of real objects.

DETERMINISM
-----------
Images and annotations are sorted by id, source ids are PRESERVED (not renumbered), `area` is
recomputed as w·h in original pixels (what COCOeval's size bands key off), and Roboflow's extra
keys (license/date_captured/extra/segmentation) are dropped. Re-running over unchanged input
produces a byte-identical tree.

DATASET-AGNOSTIC
----------------
Like every other module under data/, this imports nothing from the rest of the pipeline
and knows nothing about any specific dataset. `--out-dir` and `--classes-file` are caller-supplied;
with no --classes-file the class order is derived from the source COCO categories themselves.

Usage
-----
    # Standalone, for a dataset split before this step existed:
    python prepare_ground_truth.py \
        --splits-root ../../experiment_files/finetuning-data/garbage-classification \
        --out-dir ../ground_truths --dataset garbage-classification \
        --classes-file ../classes/<ds>.yaml

    # Or automatically, as the last step of a strategy (see `ground_truth_export:` in
    # split_config.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

# split_common lives beside this file, at the data root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_common import OUT_ANN_NAME, SPLITS  # noqa: E402

IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"}

# Which split lands under ultralytics' `test:` key — the held-out set scored once, distinct from
# `val:` which selects the checkpoint during training.
HELD_OUT_SPLIT = "eval"


# ── Class order ───────────────────────────────────────────────────────────────


def load_class_order(classes_file: Path | None, categories: list[dict]) -> list[dict]:
    """Return [{'id': <0-based yolo id>, 'name': ...}, ...] in the order labels must be written.

    With a classes_file, that file is the authority — its `classes:` list already carries the id
    each name must take, and this function only validates it is a contiguous 0-based range (the
    YOLO class index is a literal array offset, so a gap would mis-index `names`).

    Without one, the order is derived from the source COCO `categories`, dropping any node that
    looks like a Roboflow dataset-level supercategory (`supercategory: none` with children). That
    keeps this script usable on a dataset the pipeline has no class list for.
    """
    if classes_file is not None:
        with open(classes_file) as f:
            doc = yaml.safe_load(f)
        classes = [{"id": int(c["id"]), "name": str(c["name"])} for c in doc["classes"]]
        ids = sorted(c["id"] for c in classes)
        if ids != list(range(len(ids))):
            raise ValueError(
                f"{classes_file}: `classes:` ids must be a contiguous 0-based range, got {ids}. "
                "The id IS the YOLO class index, so a gap silently mis-indexes every prediction."
            )
        return sorted(classes, key=lambda c: c["id"])

    # No class list supplied: take the source's own order, minus the dummy root node.
    parents = {c.get("supercategory") for c in categories if c.get("supercategory") not in (None, "none")}
    kept = [c for c in categories if c["name"] not in parents]
    return [{"id": i, "name": c["name"]} for i, c in enumerate(kept)]


def build_category_map(class_order: list[dict], categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    """Map {source category_id: 0-based yolo class id}, plus the class names the source never used.

    Case-insensitive by name — a COCO export writing "PLASTIC" matches a class list saying
    "plastic". Source categories with no counterpart (the dummy supercategory) are simply absent
    from the returned map; `convert_split` counts their annotations as skipped.
    """
    name_to_yolo = {c["name"].lower(): c["id"] for c in class_order}
    mapping = {
        int(c["id"]): name_to_yolo[c["name"].lower()]
        for c in categories
        if c["name"].lower() in name_to_yolo
    }
    seen = {c["name"].lower() for c in categories}
    unused = [c["name"] for c in class_order if c["name"].lower() not in seen]
    return mapping, unused


# ── Images directory ──────────────────────────────────────────────────────────


def populate_images_dir(images_dir: Path, source_dir: Path, split_filenames: set[str]) -> None:
    """Create images_dir as a REAL directory and symlink this split's images into it.

    The real-directory part is load-bearing, not stylistic. Ultralytics' check_det_dataset
    resolves dataset paths with .resolve(), which follows symlinks; if images/ were itself a
    directory symlink, the resolved path would lose its '/images/' segment and img2label_paths
    could no longer find the labels/ sibling. A real directory resolves to itself and keeps it.

    Only files named in *split_filenames* are linked. The source directory can hold images an
    earlier, differently-seeded split run left behind; linking those in would hand ultralytics an
    image with no labels/*.txt, which it reads as "contains zero objects" — a phantom background
    sample, and in a held-out split, pure false-positive pressure. The COCO json is the authority
    on membership, never the directory listing.

    Links the split no longer claims are pruned, so re-running repairs a directory in place.
    """
    if images_dir.is_symlink():
        images_dir.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    for existing in images_dir.iterdir():
        if existing.is_symlink() and existing.name not in split_filenames:
            existing.unlink()

    for src_file in sorted(source_dir.iterdir()):
        if src_file.suffix.lower().lstrip(".") not in IMG_FORMATS:
            continue
        if src_file.name not in split_filenames:
            continue
        link = images_dir / src_file.name
        if link.is_symlink():
            if link.resolve() == src_file.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(
                f"Cannot create symlink at {link}: a real file already exists there. "
                "Remove it manually if intentional."
            )
        link.symlink_to(src_file.resolve())

    n_linked = sum(1 for p in images_dir.iterdir() if not p.name.startswith("."))
    if n_linked != len(split_filenames):
        missing = sorted(split_filenames - {p.name for p in images_dir.iterdir()})
        raise RuntimeError(
            f"linked {n_linked} images but the COCO json lists {len(split_filenames)}. "
            f"Missing from {source_dir}: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )


# ── One split ─────────────────────────────────────────────────────────────────


def convert_split(
    source_json: Path,
    source_images: Path,
    out_split_dir: Path,
    class_order: list[dict],
) -> dict:
    """Write one split's annotations.coco.json, labels/*.txt and images/ symlinks.

    Returns a stats dict for the manifest.
    """
    with open(source_json) as f:
        coco = json.load(f)

    cat_map, unused = build_category_map(class_order, coco["categories"])
    names = [c["name"] for c in class_order]

    images = sorted(coco["images"], key=lambda im: im["id"])
    anns_by_image: dict[int, list] = defaultdict(list)
    n_skipped = 0
    for ann in sorted(coco["annotations"], key=lambda a: a["id"]):
        if ann["category_id"] not in cat_map:
            n_skipped += 1  # dummy supercategory, or a class this export isn't scored on
            continue
        anns_by_image[ann["image_id"]].append(ann)

    out_images, out_anns = [], []
    per_class = Counter()
    n_ignore = 0
    labels_dir = out_split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    split_filenames = {Path(im["file_name"]).name for im in images}

    # Prune stale label files from a previous, differently-seeded run — same reasoning as the
    # symlink pruning below, and a leftover .txt is worse: ultralytics pairs labels to images by
    # stem, so an orphan .txt silently re-attaches if that image ever returns to this split.
    keep_stems = {Path(n).stem for n in split_filenames}
    for stale in labels_dir.glob("*.txt"):
        if stale.stem not in keep_stems:
            stale.unlink()

    for im in images:
        img_w, img_h = int(im["width"]), int(im["height"])
        file_name = Path(im["file_name"]).name
        out_images.append({
            "id": int(im["id"]),
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
        })

        lines = []
        for ann in anns_by_image.get(im["id"], []):
            yolo_id = cat_map[ann["category_id"]]
            iscrowd = int(ann.get("iscrowd", 0))
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            # area is recomputed rather than trusted: it is what COCOeval's size bands key off,
            # and a segmentation-derived area from the source would band boxes differently.
            out_anns.append({
                "id": int(ann["id"]),
                "image_id": int(im["id"]),
                "category_id": yolo_id + 1,  # COCO ids are 1-based and contiguous
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": iscrowd,
            })
            # An iscrowd annotation is an IGNORE REGION, not an object: COCOeval discards a
            # detection matched to one rather than scoring it either way. It therefore belongs
            # in the COCO json and nowhere else — a YOLO label file has no ignore flag, so
            # writing one would turn it into a training target and, via ultralytics' own val(),
            # into a target the model is scored on. student_training/prepare_data.py drops iscrowd for the
            # same reason. Also kept out of per_class, so the manifest's counts stay a
            # count of objects; n_ignore_annotations carries the rest.
            if iscrowd:
                n_ignore += 1
                continue
            per_class[names[yolo_id]] += 1
            lines.append(
                f"{yolo_id} {(x + bw / 2) / img_w:.6f} {(y + bh / 2) / img_h:.6f} "
                f"{bw / img_w:.6f} {bh / img_h:.6f}"
            )

        (labels_dir / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    out_doc = {
        "images": out_images,
        "annotations": out_anns,
        "categories": [{"id": c["id"] + 1, "name": c["name"]} for c in class_order],
    }
    (out_split_dir / "annotations.coco.json").write_text(json.dumps(out_doc, indent=2) + "\n")

    populate_images_dir(out_split_dir / "images", source_images, split_filenames)

    return {
        "n_images": len(out_images),
        "n_annotations": len(out_anns),
        "n_annotations_skipped": n_skipped,
        "n_ignore_annotations": n_ignore,
        "per_class": {n: per_class.get(n, 0) for n in names},
        "classes_absent_from_split": [n for n in names if per_class.get(n, 0) == 0],
        "source_classes_unused": unused,
    }


# ── Whole dataset ─────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never a hard failure
        return None


def write_dataset_yaml(root: Path, class_order: list[dict], splits: list[str]) -> Path:
    """Write dataset.yaml pointing train:/val:/test: at the split image directories.

    `eval` goes under `test:` — ultralytics' conventional slot for a held-out set evaluated once,
    kept distinct from `val:`, which selects the checkpoint during training.

    No `path:` key is written, deliberately: ultralytics' check_det_dataset falls back to
    `Path(yaml_file).parent` as the dataset root when `path:` is absent, so the split entries
    below resolve against this yaml's own directory. An explicit `path:` would have to be
    absolute (a relative one resolves against ultralytics' global DATASETS_DIR, not against the
    yaml), which would pin the file to one machine.
    """
    doc: dict = {}
    for key, split in (("train", "train"), ("val", "val"), ("test", HELD_OUT_SPLIT)):
        if split in splits:
            doc[key] = f"{split}/images"
    doc["nc"] = len(class_order)
    doc["names"] = [c["name"] for c in class_order]

    out = root / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    return out


def export_ground_truth(
    splits_root: Path,
    out_dir: Path,
    dataset: str,
    *,
    classes_file: Path | None = None,
    splits: list[str] | None = None,
    ann_name: str = OUT_ANN_NAME,
) -> dict:
    """Export every split of one dataset into out_dir/<dataset>/. Returns the manifest dict."""
    splits_root = Path(splits_root).resolve()
    root = Path(out_dir).resolve() / dataset
    splits = list(splits if splits is not None else SPLITS)

    present = [s for s in splits if (splits_root / s / ann_name).is_file()]
    if not present:
        raise SystemExit(
            f"no split annotations found under {splits_root} "
            f"(looked for {'/'.join(splits)} each containing {ann_name})"
        )
    missing = [s for s in splits if s not in present]
    if missing:
        print(f"  note: no {ann_name} for split(s) {missing} — skipping them")

    with open(splits_root / present[0] / ann_name) as f:
        class_order = load_class_order(classes_file, json.load(f)["categories"])

    root.mkdir(parents=True, exist_ok=True)
    stats, sources = {}, {}
    for split in present:
        src_json = splits_root / split / ann_name
        print(f"  {split:<6} <- {src_json}")
        stats[split] = convert_split(src_json, splits_root / split, root / split, class_order)
        sources[split] = {"json": str(src_json), "sha256": _sha256(src_json)}
        s = stats[split]
        line = f"         {s['n_images']:,} images | {s['n_annotations']:,} boxes"
        if s["n_annotations_skipped"]:
            line += f" | {s['n_annotations_skipped']:,} skipped (unmapped category)"
        print(line)

    with open(root / "classes.yaml", "w") as f:
        yaml.dump({"classes": class_order}, f, default_flow_style=False, sort_keys=False)
    dataset_yaml = write_dataset_yaml(root, class_order, present)

    manifest = {
        "dataset": dataset,
        "splits_root": str(splits_root),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "classes": class_order,
        "sources": sources,
        "splits": stats,
    }
    # generated_utc/git_rev deliberately excluded from the byte-identity claim: they are
    # provenance, not content. Everything under splits/ and sources/ is reproducible.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  wrote {root}")
    print(f"  dataset.yaml : {dataset_yaml}")
    print(f"  held-out GT  : {root / HELD_OUT_SPLIT / 'annotations.coco.json'}")
    return manifest


def export_from_config(cfg: dict, config_dir: Path | None, splits_root: Path) -> dict | None:
    """Run the export from a strategy's `ground_truth_export:` block, or return None if disabled.

    Called by each data strategy at the end of main(). Every path is a CONFIG VALUE —
    nothing here is hardcoded — so this module stays dataset-agnostic and data keeps
    importing nothing from the rest of the pipeline.

    `config_dir` is the base relative paths resolve against. Strategies whose resolve_settings()
    already made them absolute pass None; a relative path arriving with no base is an error rather
    than something silently resolved against the current working directory, which would depend on
    where the caller happened to be standing.
    """
    block = cfg.get("ground_truth_export") or {}
    if not block.get("enabled", False):
        return None

    def _resolve(p):
        if p is None:
            return None
        p = Path(p)
        if p.is_absolute():
            return p
        if config_dir is None:
            raise SystemExit(
                f"ground_truth_export path {p} is relative and no config directory was given to "
                "resolve it against. Make it absolute, or pass config_dir."
            )
        return (config_dir / p).resolve()

    print("\n── ground truth export ───────────────────────────────────────────")
    return export_ground_truth(
        splits_root=splits_root,
        out_dir=_resolve(block["out_dir"]),
        dataset=block.get("dataset") or Path(splits_root).name,
        classes_file=_resolve(block.get("classes_file")),
        splits=block.get("splits"),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a finished data partition into ground_truths/<dataset>/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[-1],
    )
    parser.add_argument("--splits-root", required=True,
                        help="directory holding the <split>/ subdirectories a strategy wrote")
    parser.add_argument("--out-dir", required=True, help="the ground_truths/ root")
    parser.add_argument("--dataset", default=None,
                        help="subdirectory name under --out-dir (default: basename of --splits-root)")
    parser.add_argument("--classes-file", default=None,
                        help="yaml with a `classes:` list giving the authoritative id order. "
                             "Omit to derive the order from the source COCO categories.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help=f"splits to export (default: {' '.join(SPLITS)})")
    parser.add_argument("--ann-name", default=OUT_ANN_NAME,
                        help=f"annotation filename inside each split dir (default: {OUT_ANN_NAME})")
    args = parser.parse_args(argv)

    splits_root = Path(args.splits_root).resolve()
    export_ground_truth(
        splits_root=splits_root,
        out_dir=Path(args.out_dir),
        dataset=args.dataset or splits_root.name,
        classes_file=Path(args.classes_file) if args.classes_file else None,
        splits=args.splits,
        ann_name=args.ann_name,
    )


if __name__ == "__main__":
    main()
