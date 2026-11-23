#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""pseudolabels.py — STAC stages 1 and 2: run the teacher, then threshold what it produced.

Two stages behind one `--stage` argument, so run_stac.py still launches each in its own
subprocess (a crash in one must not leave the other holding a half-initialized CUDA context):

    --stage teacher   inference over the unlabeled pool  -> runs/<run>/raw_predictions.jsonl
    --stage filter    NMS, then the confidence threshold  -> runs/<run>/pseudolabels/

They are separate stages, not one pass, because every threshold lives in the filter: tau can be
re-tuned without paying for inference again.

Usage
-----
    python pseudolabels.py --stage teacher --run-name stac-seed42 [--limit N]
    python pseudolabels.py --stage filter  --run-name stac-seed42


═══ Stage 1: teacher inference ═════════════════════════════════════════════════════════════

pseudolabels.py --stage teacher — STAC stage 1: run the teacher over the unlabeled pool.

    "Generate pseudo labels of unlabeled images (i.e. bounding boxes and their class labels)
     using the trained teacher model."  — STAC §3.2, step 2

The teacher is an Ultralytics YOLO checkpoint trained on the LABELED split only. This script
does inference and nothing else: every threshold lives in pseudolabels.py --stage filter, so tau can be
changed without paying for inference again.

Output: runs/<run_name>/raw_predictions.jsonl — ONE JSON OBJECT PER LINE, records for a given
image contiguous:

    {"image_path": "/abs/path.jpg", "class_id": 0, "class_name": "plastic",
     "bbox_xyxy": [x1, y1, x2, y2], "score": 0.87, "image_w": 1920, "image_h": 1080}

JSONL rather than one JSON array: at conf=0.001 x max_det=300 x 8,896 images this file is
large, and the pseudo_labeling equivalent (a 4.97 GB array) needed an ijson streaming path plus a
memory/stream/auto mode switch to read back. One object per line streams with the stdlib, so
that entire axis disappears.

Usage
-----
    python pseudolabels.py --stage teacher --config stac_config.yaml --run-name stac-seed42
    python pseudolabels.py --stage teacher --config stac_config.yaml --run-name stac-seed42 --resume
    python pseudolabels.py --stage teacher --config stac_config.yaml --run-name stac-seed42 --limit 20


═══ Stage 2: NMS + tau ═════════════════════════════════════════════════════════════════════

pseudolabels.py --stage filter — STAC stage 2: NMS, then the confidence threshold tau.

    "We use the score of each returned bounding box after NMS ... We apply confidence-based
     thresholding to further reduce potentially wrong pseudo boxes."   — STAC §3.2
    "We use tau = 0.9 and lambda_u = 2 for all our experiments."       — STAC §4.1

Two steps, in this order, and nothing else:

    1. within-class NMS at `filter.nms_iou`
    2. drop every remaining box scoring below `filter.tau`

Everything the SAM3 pipeline's filter_and_convert.py does BEYOND that is deliberately absent
from this copy rather than switched off by config:

  - the cross-class score-gap merge (`merge_multiclass_boxes`) — not in the paper. It suppresses
    a lower-scoring overlapping box of a DIFFERENT class when the score gap exceeds 0.3, which
    is a SAM3-specific heuristic for one physical object detected under two prompts.
  - the score-percentile filter — a second, data-dependent threshold; tau is the only one here.
  - `min_boxes_per_image` — see "Empty images are kept" below.
  - the ijson memory/stream/auto branch — pseudolabels.py --stage teacher writes JSONL, so one pass over
    lines is the whole reader.
  - low-confidence masking, and the confidence_weights.json it feeds. STAC weights pseudo-labels
    by one global lambda_u, never per box, so nothing downstream reads a per-box confidence.

Empty images are kept
---------------------
EVERY image in the unlabeled split gets a label file, including the ones where no box clears
tau — an empty .txt. This is the paper's behaviour: an unlabeled image is part of the
unsupervised loss whether or not the teacher found anything on it, and one with no pseudo-box
supervises pure background. Ultralytics reports these as "backgrounds" at scan time. The SAM3
pipeline instead drops such images (8,896 -> 8,740 at its cutoff of 0.25); at tau = 0.9 that
would discard most of the unlabeled pool.

Output
------
    runs/<run_name>/pseudolabels/labels/<stem>.txt   YOLO "cls cx cy w h", possibly empty
    runs/<run_name>/pseudolabels/images/<file>       symlink into the image pool
    runs/<run_name>/pseudolabels/dataset.yaml        the PL half's dataset yaml
    runs/<run_name>/pseudolabel_stats.json           the pre-training gate — read this before
                                                     spending GPU hours on the student

Usage
-----
    python pseudolabels.py --stage filter --config stac_config.yaml --run-name stac-seed42
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml
from tqdm import tqdm

from utils import class_names, load_coco_image_list, load_config, nms, xyxy_to_yolo

# How often to flush the append stream, in images. --resume restarts from the last flushed
# image, so this bounds how much work a kill can cost.
_FLUSH_EVERY = 100


def build_model(cfg: Dict[str, Any]):
    """Load the teacher checkpoint and verify its classes ARE the pipeline's classes.

    The teacher was trained off the same class list, so `box.cls` is already the
    class id everything downstream uses. That is asserted rather than assumed: a teacher trained
    against a different class order would silently relabel the entire pseudo-label set.
    """
    from ultralytics import YOLO

    ckpt = Path(cfg["teacher"]["checkpoint"])
    if not ckpt.is_file():
        raise SystemExit(f"teacher checkpoint not found: {ckpt}")

    print(f"Loading teacher: {ckpt}")
    model = YOLO(str(ckpt))

    expected = class_names(cfg)
    got = [model.names[i] for i in sorted(model.names)]
    if [n.lower() for n in got] != [n.lower() for n in expected]:
        raise SystemExit(
            "teacher class list does not match classes.yaml — its predicted class ids would "
            "mean something different downstream.\n"
            f"  teacher     : {got}\n"
            f"  classes.yaml: {expected}"
        )
    print(f"Teacher classes verified: {expected}")
    return model


def predict_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ultralytics predict() settings, chosen so the teacher filters as little as possible.

    conf is near-zero and iou near-one on purpose: ALL thresholding is pseudolabels.py --stage filter's
    job, so tau can be retuned from the same raw_predictions.jsonl. agnostic_nms stays False so
    different-class boxes never suppress each other here either.
    """
    t = cfg["teacher"]
    return {
        "conf": t.get("conf", 0.001),
        "iou": t.get("iou", 0.95),
        "max_det": t.get("max_det", 300),
        "imgsz": t.get("imgsz", 640),
        "device": t.get("device", "0"),
        "half": t.get("half", True),
        "agnostic_nms": False,
        "verbose": False,
    }


def infer_image(img_path: Path, model, kwargs: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
    """Every box the teacher predicts for one image, as raw_predictions.jsonl records."""
    result = model.predict(source=str(img_path), **kwargs)[0]
    h, w = result.orig_shape

    records = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        records.append(
            {
                "image_path": str(img_path),
                "class_id": class_id,
                "class_name": names[class_id],
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": float(box.conf.item()),
                "image_w": int(w),
                "image_h": int(h),
            }
        )
    return records


def completed_images(jsonl: Path) -> set[str]:
    """Image paths already present in a partial jsonl, for --resume.

    Reads only the `image_path` field of each line. The last line of a killed run may be
    truncated; it is skipped, and that image is simply re-inferred.
    """
    done: set[str] = set()
    if not jsonl.exists():
        return done
    with open(jsonl) as f:
        for line in f:
            try:
                done.add(json.loads(line)["image_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_teacher(args: argparse.Namespace) -> None:
    """Stage 1: run the teacher over the unlabeled pool, writing raw_predictions.jsonl."""
    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["root"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "raw_predictions.jsonl"

    images = load_coco_image_list(cfg["data"]["unlabeled_coco_json"], cfg["data"]["image_dir"])
    print(f"Unlabeled pool: {len(images):,} images")

    done: set[str] = set()
    if args.resume:
        done = completed_images(jsonl)
        print(f"Resuming: {len(done):,} images already done")
    elif jsonl.exists():
        jsonl.unlink()  # a fresh run must not append to a previous one's predictions

    todo = [r for r in images if str(r["file_path"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: running on {len(todo)} images")

    model = build_model(cfg)
    kwargs = predict_kwargs(cfg)
    names = class_names(cfg)

    n_boxes = 0
    missing = 0
    with open(jsonl, "a") as f:
        for i, record in enumerate(tqdm(todo, desc="teacher", unit="img"), start=1):
            path = record["file_path"]
            if not path.is_file():
                missing += 1
                continue
            for rec in infer_image(path, model, kwargs, names):
                f.write(json.dumps(rec) + "\n")
                n_boxes += 1
            if i % _FLUSH_EVERY == 0:
                f.flush()

    print(f"\nTeacher inference complete.")
    print(f"  images   : {len(todo):,} processed" + (f", {missing:,} missing on disk" if missing else ""))
    print(f"  boxes    : {n_boxes:,} written (before NMS and tau)")
    print(f"  output   : {jsonl}")


# Resolution of the score histogram in pseudolabel_stats.json: 20 bins of 0.05 over [0, 1].
_HIST_BINS = 20


def iter_image_groups(jsonl: Path) -> Iterator[tuple[str, List[Dict[str, Any]]]]:
    """Yield (image_path, records) for each contiguous run of records sharing an image_path.

    pseudolabels.py --stage teacher writes one image's boxes consecutively, so grouping by contiguity
    needs no sort and no dict of the whole file. A group CAN repeat if a killed run was resumed
    after a truncated final line — the caller handles that by letting the later group win, which
    is correct because re-inference reproduces the same boxes.
    """
    if not jsonl.exists():
        raise SystemExit(f"raw predictions not found: {jsonl}\nRun pseudolabels.py --stage teacher first.")

    current_path: str | None = None
    buffer: List[Dict[str, Any]] = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line of a killed run
            if rec["image_path"] != current_path:
                if current_path is not None:
                    yield current_path, buffer
                current_path, buffer = rec["image_path"], []
            buffer.append(rec)
    if current_path is not None:
        yield current_path, buffer


def filter_image(
    records: List[Dict[str, Any]],
    nms_iou: float,
    tau: float,
) -> tuple[List[Dict[str, Any]], List[float]]:
    """NMS then tau, for one image's predictions.

    Returns (surviving records, post-NMS scores). The second value is only for the histogram —
    it is what shows where tau sits relative to the teacher's actual score distribution.

    NMS runs WITHIN each class: two boxes of different classes never suppress each other, which
    is standard per-class NMS and is what STAC's detector does. Cross-class suppression is
    exactly the heuristic this module drops.
    """
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[rec["class_id"]].append(rec)

    after_nms: List[Dict[str, Any]] = []
    for class_records in by_class.values():
        keep = nms(
            [r["bbox_xyxy"] for r in class_records],
            [r["score"] for r in class_records],
            nms_iou,
        )
        after_nms.extend(class_records[i] for i in keep)

    post_nms_scores = [r["score"] for r in after_nms]
    kept = [r for r in after_nms if r["score"] >= tau]
    return kept, post_nms_scores


def write_label_file(records: List[Dict[str, Any]], path: Path) -> None:
    """Write one YOLO label file. An empty `records` writes an empty file, on purpose."""
    lines = []
    for rec in records:
        x1, y1, x2, y2 = rec["bbox_xyxy"]
        cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, rec["image_w"], rec["image_h"])
        lines.append(f"{rec['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_images(images: List[Dict[str, Any]], images_dir: Path) -> int:
    """Symlink every unlabeled image into `images_dir`, which must be a REAL directory.

    Real directory of per-file symlinks, not a directory symlink: ultralytics' img2label_paths
    derives the sibling `labels/` directory by string-replacing `/images/` in each resolved
    image path, and a directory symlink would resolve the whole path out of this tree.
    """
    if images_dir.is_symlink():
        images_dir.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for record in images:
        src = Path(record["file_path"])
        dest = images_dir / src.name
        if dest.is_symlink() or dest.exists():
            n += 1
            continue
        dest.symlink_to(src)
        n += 1
    return n


def write_dataset_yaml(path: Path, root: Path, names: List[str]) -> None:
    """The PL half's dataset yaml. `val: null` — validation is the GT val split's job."""
    path.write_text(
        yaml.dump(
            {"path": str(root), "train": "images", "val": None,
             "nc": len(names), "names": names},
            default_flow_style=False,
            sort_keys=False,
        )
    )


def run_filter(args: argparse.Namespace) -> None:
    """Stage 2: within-class NMS, then the confidence threshold tau."""
    cfg = load_config(args.config)
    nms_iou = float(cfg["filter"]["nms_iou"])
    tau = float(cfg["filter"]["tau"] if args.tau is None else args.tau)

    run_dir = Path(cfg["output"]["root"]) / args.run_name
    pl_dir = run_dir / "pseudolabels"
    labels_dir = pl_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = load_coco_image_list(cfg["data"]["unlabeled_coco_json"], cfg["data"]["image_dir"])
    names = class_names(cfg)

    print(f"Filtering with NMS IoU {nms_iou} then tau {tau} over {len(images):,} unlabeled images")

    # Per-stem so a repeated group (resume after a truncated line) overwrites rather than
    # double-counts. 8,896 small entries — the whole point of streaming was the BOX count.
    per_stem: Dict[str, Dict[str, Any]] = {}
    for image_path, records in iter_image_groups(run_dir / "raw_predictions.jsonl"):
        kept, post_nms_scores = filter_image(records, nms_iou, tau)
        stem = Path(image_path).stem
        write_label_file(kept, labels_dir / f"{stem}.txt")

        hist = [0] * _HIST_BINS
        for s in post_nms_scores:
            hist[min(int(s * _HIST_BINS), _HIST_BINS - 1)] += 1
        per_stem[stem] = {
            "raw": len(records),
            "after_nms": len(post_nms_scores),
            "kept": len(kept),
            "hist": hist,
            "classes": [r["class_id"] for r in kept],
        }

    # Every unlabeled image the teacher produced NO record for (all boxes below its own conf, or
    # the image missing on disk) still gets an empty label file — see the module docstring.
    n_no_records = 0
    for record in images:
        stem = Path(record["file_path"]).stem
        if stem in per_stem:
            continue
        write_label_file([], labels_dir / f"{stem}.txt")
        per_stem[stem] = {"raw": 0, "after_nms": 0, "kept": 0,
                          "hist": [0] * _HIST_BINS, "classes": []}
        n_no_records += 1

    n_linked = link_images(images, pl_dir / "images")
    write_dataset_yaml(pl_dir / "dataset.yaml", pl_dir, names)

    # ── Stats: the pre-training gate ──────────────────────────────────────────
    hist_total = [0] * _HIST_BINS
    class_counts = {name: 0 for name in names}
    n_raw = n_after_nms = n_kept = 0
    n_with_boxes = 0
    for entry in per_stem.values():
        n_raw += entry["raw"]
        n_after_nms += entry["after_nms"]
        n_kept += entry["kept"]
        n_with_boxes += entry["kept"] > 0
        for b, c in enumerate(entry["hist"]):
            hist_total[b] += c
        for class_id in entry["classes"]:
            class_counts[names[class_id]] += 1

    stats = {
        "run_name": args.run_name,
        "nms_iou": nms_iou,
        "tau": tau,
        "images_total": len(per_stem),
        "images_with_boxes": n_with_boxes,
        "images_empty": len(per_stem) - n_with_boxes,
        "images_with_no_teacher_records": n_no_records,
        "boxes_raw": n_raw,
        "boxes_after_nms": n_after_nms,
        "boxes_after_tau": n_kept,
        "boxes_per_image_mean": round(n_kept / max(len(per_stem), 1), 4),
        "post_nms_score_histogram": {
            f"{b / _HIST_BINS:.2f}-{(b + 1) / _HIST_BINS:.2f}": hist_total[b]
            for b in range(_HIST_BINS)
        },
        "kept_boxes_per_class": class_counts,
    }
    (run_dir / "pseudolabel_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nPseudo-labels written.")
    print(f"  boxes    : {n_raw:,} raw -> {n_after_nms:,} after NMS -> {n_kept:,} after tau={tau}")
    print(f"  images   : {len(per_stem):,} total, {n_with_boxes:,} with >=1 box, "
          f"{len(per_stem) - n_with_boxes:,} empty (kept as background)")
    print(f"  per class: {class_counts}")
    print(f"  labels   : {labels_dir}")
    print(f"  images   : {pl_dir / 'images'} ({n_linked:,} symlinks)")
    print(f"  stats    : {run_dir / 'pseudolabel_stats.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="STAC stages 1-2: teacher inference, then NMS + tau")
    parser.add_argument("--stage", required=True, choices=("teacher", "filter"))
    parser.add_argument("--config", default="stac_config.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="--stage teacher: skip images already in the jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="--stage teacher: process at most N images")
    parser.add_argument(
        "--tau", type=float, default=None,
        help=(
            "--stage filter: override filter.tau for this run. Exists so a tau variant reuses "
            "ONE config file rather than a copy of it: every other hyperparameter is then "
            "byte-identical to the baseline run by construction, instead of by inspection. The "
            "value actually used is recorded in pseudolabel_stats.json and in the student's "
            "W&B config."
        ),
    )
    args = parser.parse_args()

    (run_teacher if args.stage == "teacher" else run_filter)(args)


if __name__ == "__main__":
    main()
