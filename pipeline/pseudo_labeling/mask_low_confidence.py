#!/usr/bin/env python3
"""mask_low_confidence.py — pseudo-labeling Step C: mask low-confidence pseudo-labels.

PROTOTYPE. Runs *after* `filter_and_convert.py` (hard cutoff -> percentile ->
within-class NMS -> cross-class merge) and takes that step's surviving box set as
its input.

The idea
--------
Filtering can only ever *drop a label*. Dropping one leaves the object it
described still sitting in the image, now unlabeled — so the student is trained
to call it background. This step instead **paints the pixels out** and drops the
label, so a box we do not trust becomes neither a positive nor a false negative.

Boxes with ``w_sam3 >= mask_score_threshold`` are kept untouched. Boxes below it
are removed from the annotations *and* their pixels are filled with a flat color
in a newly written copy of the image.

Reversibility
-------------
Nothing here mutates pseudo-labeling's existing output. The original images, the
original ``labels/``, and the original ``confidence_weights.json`` are only ever
read. Everything this step produces lands in a sibling directory
(``label_masked/`` by default), so reverting is `rm -rf` on that directory, and
re-running with a different threshold needs no re-inference and no re-filtering.

``mask_manifest.json`` records, per image, every box kept, every box masked, and
how many pixels were painted — enough to audit or reconstruct any decision made
here without re-reading the images.

Two guards against erasing what we still label
------------------------------------------------
Low-confidence boxes overlap kept boxes constantly — thousands of pairs where one box
is >90% contained in the other — so naive painting would destroy pixels that surviving
labels depend on. Two rules prevent
that, and each matters in practice:

1. ``protect_kept_boxes`` (default true). The fill is applied to

       union(masked boxes) MINUS union(kept boxes)

   so a low-confidence box *inside* a kept one never punches a hole through the
   object we are still labeling.

2. ``drop_kept_if_masked`` (default 1.0). Backstop for when rule 1 is off. If the
   paint has covered a kept box to at least this fraction, that annotation is
   dropped too — a label over nothing but flat fill would ask the student to
   detect an object in a blank rectangle, which is worse than either keeping or
   dropping the box honestly. Unlike rule 1 this one *removes annotations*, so
   the label set is no longer identical across configurations. An image left
   with fewer than ``min_kept_boxes`` afterwards drops out of the dataset
   entirely.

Rule 1 and rule 2 are complementary rather than redundant: rule 1 prevents the
overlap from happening, rule 2 cleans up after it when rule 1 is disabled. With
``protect_kept_boxes`` true, rule 2 never fires.

Usage
-----
    # As a pipeline step. Disabled by default in config.yaml's `steps:`
    # while this is a prototype — flip that entry's `enabled` to run it here.
    python run_pseudo_labeling.py --config config.yaml --run-name my-run

    # Standalone, against an already-filtered run
    python mask_low_confidence.py --config config.yaml
    python mask_low_confidence.py --config config.yaml --threshold 0.5
    python mask_low_confidence.py --config config.yaml --dry-run

    # Guard override — for threshold/guard sweeps that keep variants side by side.
    # This is the "naive" configuration: the pixel guard off.
    python mask_low_confidence.py --config config.yaml \
        --no-protect-kept-boxes \
        --out-dir ../yolo_data/<run_name>/label_masked_naive

Output layout (all new; nothing above it is modified)
----------------------------------------------------
    yolo_data/{run_name}/label_masked/
        images/                  real masked image files, plus symlinks to
                                 originals for images that needed no masking
        labels/                  YOLO .txt with the kept boxes only
        confidence_weights.json  kept boxes only, same schema as pseudo-labeling's
        dataset.yaml             points at images/ above
        mask_manifest.json       full record of what was masked and why
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common import write_resolved_config  # noqa: E402

from utils import load_config, maybe_shutdown  # noqa: E402

# Ultralytics' letterbox padding value. Using it means masked regions carry the
# same pixel value the model already sees around every letterboxed image, rather
# than introducing a color (pure black) that also occurs in real image content.
DEFAULT_FILL_RGB = (114, 114, 114)


# ─── Box helpers ──────────────────────────────────────────────────────────────


def to_pixel_xyxy(box: Dict[str, Any], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Normalized cxcywh box dict -> integer pixel xyxy, clamped to the image.

    `confidence_weights.json` stores normalized coordinates only, so image
    dimensions come from the image file itself.
    """
    x1 = (box["cx"] - box["w"] / 2) * img_w
    y1 = (box["cy"] - box["h"] / 2) * img_h
    x2 = (box["cx"] + box["w"] / 2) * img_w
    y2 = (box["cy"] + box["h"] / 2) * img_h
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(img_w, int(np.ceil(x2))),
        min(img_h, int(np.ceil(y2))),
    )


def partition_by_confidence(
    boxes: Sequence[Dict[str, Any]],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split one image's boxes into (kept, to_mask) at `threshold`.

    Kept is ``w_sam3 >= threshold`` — a box exactly at the threshold survives.
    """
    kept = [b for b in boxes if b["w_sam3"] >= threshold]
    to_mask = [b for b in boxes if b["w_sam3"] < threshold]
    return kept, to_mask


def split_kept_by_mask(
    kept: Sequence[Dict[str, Any]],
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    drop_at: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split kept boxes into (still annotated, erased by the mask).

    A kept box whose pixels have all been painted over describes nothing but
    flat fill. Leaving its annotation in place would ask the student to detect
    an object in a blank rectangle — worse than either keeping or dropping the
    box honestly. So when the paint covers at least `drop_at` of a kept box, the
    annotation goes with the pixels.

    Only reachable when `protect_kept_boxes` is false: with the carve-out on, a
    kept box's rectangle is never painted, every coverage is 0, and this returns
    `(kept, [])` unchanged. That also means the drop cannot invalidate the mask
    it was computed from — the mask does not depend on `kept` in the one mode
    where boxes actually get erased, so no second pass is needed.

    Parameters
    ----------
    drop_at:
        Painted fraction at which a kept box loses its annotation. 1.0 means
        only a completely painted box is dropped; lower it to also drop boxes
        that are merely mostly painted.
    """
    still, erased = [], []
    for box in kept:
        x1, y1, x2, y2 = to_pixel_xyxy(box, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            still.append(box)
            continue
        region = mask[y1:y2, x1:x2]
        covered = float(region.mean()) if region.size else 0.0
        (erased if covered >= drop_at else still).append(box)
    return still, erased


def build_mask(
    to_mask: Sequence[Dict[str, Any]],
    kept: Sequence[Dict[str, Any]],
    img_w: int,
    img_h: int,
    padding_px: int = 0,
    protect_kept_boxes: bool = True,
) -> np.ndarray:
    """Boolean (H, W) array — True where pixels should be painted over.

    Parameters
    ----------
    to_mask, kept:
        The two halves of `partition_by_confidence`.
    padding_px:
        Grow each masked rectangle by this many pixels on every side, to catch
        the object fringe that SAM3's box tends to leave just outside itself.
    protect_kept_boxes:
        Subtract every kept box's rectangle from the mask, so no pixel a
        surviving label depends on is ever painted. See the module docstring.

    Returns
    -------
    (H, W) bool array. All-False when `to_mask` is empty, or when every masked
    region turned out to be covered by a kept box.
    """
    mask = np.zeros((img_h, img_w), dtype=bool)
    for box in to_mask:
        x1, y1, x2, y2 = to_pixel_xyxy(box, img_w, img_h)
        if padding_px:
            x1, y1 = max(0, x1 - padding_px), max(0, y1 - padding_px)
            x2, y2 = min(img_w, x2 + padding_px), min(img_h, y2 + padding_px)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    if protect_kept_boxes:
        for box in kept:
            x1, y1, x2, y2 = to_pixel_xyxy(box, img_w, img_h)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = False

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, fill_rgb: Sequence[int]) -> np.ndarray:
    """Paint `fill_rgb` into `image` wherever `mask` is True. Returns a new array."""
    out = image.copy()
    out[mask] = np.asarray(fill_rgb, dtype=out.dtype)
    return out


# ─── Writers ──────────────────────────────────────────────────────────────────


def write_yolo_label(label_path: Path, boxes: Sequence[Dict[str, Any]]) -> None:
    """Write one image's kept boxes as a YOLO .txt (empty file if none)."""
    lines = [
        f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_or_replace(link: Path, target: Path) -> None:
    """Point `link` at `target` as a symlink, replacing whatever is there.

    Images needing no masking are symlinked rather than copied — the pixels are
    identical to the original, so a copy would only waste disk.
    """
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def write_dataset_yaml(dataset_yaml_path: Path, classes: List[Dict[str, Any]]) -> None:
    """Write an Ultralytics dataset.yaml pointing at the masked images/ dir.

    images/ is a real directory (holding real files and per-file symlinks), so
    Ultralytics' `.resolve()` in check_det_dataset returns its own path
    unchanged — preserving the '/images/' segment `img2label_paths` needs to
    find the sibling labels/. This mirrors `filter_and_convert.write_dataset_yaml`;
    see the repo note on why symlinked *directories* break that resolution.
    """
    doc = {
        "path": str(dataset_yaml_path.parent.resolve()),
        "train": "images",
        "val": None,
        "nc": len(classes),
        "names": [c["name"] for c in classes],
    }
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ─── Driver ───────────────────────────────────────────────────────────────────


def run_masking(
    weights: Dict[str, List[Dict[str, Any]]],
    out_root: Path,
    threshold: float,
    fill_rgb: Sequence[int],
    padding_px: int,
    protect_kept_boxes: bool,
    drop_kept_if_masked: float,
    min_kept_boxes: int,
    jpeg_quality: int,
    dry_run: bool,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], Counter, List[float]]:
    """Mask every image in `weights` and write the parallel dataset.

    Returns
    -------
    (manifest, kept_weights, stats, masked_fracs)
        `manifest` is the full per-image record, `kept_weights` is the
        confidence_weights.json payload for the masked dataset (kept boxes only,
        re-keyed to the new image paths), `stats` counts images and boxes at each
        outcome, and `masked_fracs` is the painted-area fraction of each image
        that actually got painted — for the summary only.

    Under `dry_run` nothing is written, but every decision is still computed, so
    the returned stats are exactly what a real run would produce.
    """
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_images: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    kept_weights: Dict[str, List[Dict[str, Any]]] = {}

    stats = Counter()
    masked_fracs: List[float] = []

    for src_str, boxes in sorted(weights.items()):
        src = Path(src_str)
        stats["images_seen"] += 1
        stats["boxes_in"] += len(boxes)

        kept, to_mask = partition_by_confidence(boxes, threshold)
        stats["boxes_masked"] += len(to_mask)

        if len(kept) < min_kept_boxes:
            # Every label on this image failed the threshold. Keeping it would
            # add a near-fully-masked image with no positives — noise, not signal.
            skipped[src_str] = f"only {len(kept)} boxes above threshold (min_kept_boxes={min_kept_boxes})"
            stats["images_skipped"] += 1
            continue

        if not src.exists():
            skipped[src_str] = "source image not found"
            stats["images_missing"] += 1
            continue

        arr = mask = None
        img_w = img_h = 0
        n_px = 0
        if to_mask:
            with Image.open(src) as im:
                arr = np.array(im.convert("RGB"))
            img_h, img_w = arr.shape[:2]
            mask = build_mask(to_mask, kept, img_w, img_h, padding_px, protect_kept_boxes)
            n_px = int(mask.sum())

        # A kept box the paint has entirely covered describes nothing but fill,
        # so its annotation goes too. Decided before anything is written, since
        # it can leave the image with too few boxes to be worth emitting at all.
        if n_px:
            kept, kept_erased = split_kept_by_mask(kept, mask, img_w, img_h, drop_kept_if_masked)
        else:
            kept_erased = []
        stats["boxes_kept_erased"] += len(kept_erased)

        if len(kept) < min_kept_boxes:
            skipped[src_str] = (
                f"{len(kept_erased)} kept boxes were erased by the mask, leaving "
                f"{len(kept)} (min_kept_boxes={min_kept_boxes})"
            )
            stats["images_skipped_all_erased"] += 1
            continue

        stats["boxes_kept"] += len(kept)

        dst = images_dir / src.name
        entry: Dict[str, Any] = {
            "source": str(src.resolve()),
            "kept": kept,
            "masked": to_mask,
            "masked_painted": to_mask,
            "kept_erased_by_mask": kept_erased,
        }

        if n_px:
            frac = n_px / (img_w * img_h)
            entry["mode"] = "masked"
            entry["masked_px"] = n_px
            entry["masked_frac"] = round(frac, 6)
            masked_fracs.append(frac)
            stats["images_masked"] += 1
            if not dry_run:
                _save_image(Image.fromarray(apply_mask(arr, mask, fill_rgb)), dst, jpeg_quality)
        else:
            entry["mode"] = "symlink"
            entry["masked_px"] = 0
            entry["masked_frac"] = 0.0
            if to_mask:
                # Every masked box was fully covered by a kept box, so the
                # carve-out left nothing to paint. Emit the original pixels.
                entry["note"] = "masked boxes fully covered by kept boxes; pixels unchanged"
                stats["images_fully_protected"] += 1
            else:
                stats["images_unchanged"] += 1
            if not dry_run:
                link_or_replace(dst, src)

        if not dry_run:
            write_yolo_label(labels_dir / f"{src.stem}.txt", kept)

        manifest_images[str(dst)] = entry
        kept_weights[str(dst)] = kept

    manifest = {
        "threshold": threshold,
        "fill_rgb": list(fill_rgb),
        "mask_padding_px": padding_px,
        "protect_kept_boxes": protect_kept_boxes,
        "drop_kept_if_masked": drop_kept_if_masked,
        "min_kept_boxes": min_kept_boxes,
        "stats": dict(stats),
        "masked_frac_summary": _frac_summary(masked_fracs),
        "images": manifest_images,
        "skipped": skipped,
    }
    return manifest, kept_weights, stats, masked_fracs


def _save_image(im: Image.Image, dst: Path, jpeg_quality: int) -> None:
    """Write a masked image, keeping the source container format.

    JPEG re-encode is lossy, so quality is set high; the masked regions are flat
    color and the untouched regions have already been through one JPEG cycle.
    (`subsampling="keep"` is not usable here — it requires the in-memory image to
    have been decoded from a JPEG, and this one is built from a numpy array.)
    """
    if dst.suffix.lower() in {".jpg", ".jpeg"}:
        im.save(dst, quality=jpeg_quality, subsampling=0)
    else:
        im.save(dst)


def _frac_summary(fracs: Sequence[float]) -> Dict[str, float]:
    if not fracs:
        return {}
    a = np.asarray(fracs)
    return {
        "mean": round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "max": round(float(a.max()), 6),
    }


def print_summary(stats: Counter, masked_fracs: Sequence[float], threshold: float) -> None:
    n_in = stats["boxes_in"]
    n_masked = stats["boxes_masked"]
    print("\n" + "─" * 70)
    print(f"Masking summary (threshold = {threshold})")
    print("─" * 70)
    print(f"  images seen              : {stats['images_seen']:,}")
    print(f"    written with masking   : {stats['images_masked']:,}")
    print(f"    written unchanged      : {stats['images_unchanged']:,}")
    print(f"    unchanged (protected)  : {stats['images_fully_protected']:,}")
    print(f"    skipped (too few kept) : {stats['images_skipped']:,}")
    print(f"    skipped (all erased)   : {stats['images_skipped_all_erased']:,}")
    if stats["images_missing"]:
        print(f"    skipped (missing file) : {stats['images_missing']:,}")
    pct = (lambda n: f" ({100 * n / n_in:.1f}%)") if n_in else (lambda n: "")
    print(f"  boxes in                 : {n_in:,}")
    print(f"    kept                   : {stats['boxes_kept']:,}{pct(stats['boxes_kept'])}")
    print(f"    kept but erased by mask: {stats['boxes_kept_erased']:,}"
          f"{pct(stats['boxes_kept_erased'])}  (annotation dropped with the pixels)")
    print(f"    masked out             : {n_masked:,}{pct(n_masked)}  "
          f"(label dropped; pixels painted except where protected)")
    if masked_fracs:
        s = _frac_summary(masked_fracs)
        print(f"  painted image area       : mean {100 * s['mean']:.2f}%  "
              f"median {100 * s['median']:.2f}%  p90 {100 * s['p90']:.2f}%  "
              f"max {100 * s['max']:.2f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────


# Manifest keys that must agree with the config to reuse an existing directory. The threshold
# names the directory and is checked separately; jpeg_quality is absent from the manifest.
_GUARD_KEYS = (
    "fill_rgb", "mask_padding_px", "protect_kept_boxes", "drop_kept_if_masked", "min_kept_boxes",
)


def _mismatches(manifest_path: Path, threshold: float, mask_cfg: dict) -> list[str]:
    """Reasons an existing variant directory does NOT match what we would build now."""
    manifest = json.loads(manifest_path.read_text())
    problems = []
    if f"{manifest['threshold']:g}" != f"{threshold:g}":
        problems.append(f"threshold {manifest['threshold']} != {threshold}")
    for key in _GUARD_KEYS:
        want, got = mask_cfg.get(key), manifest.get(key)
        if key not in manifest or want is None:
            continue
        if isinstance(want, (list, tuple)):
            want, got = list(want), list(got)
        if want != got:
            problems.append(f"{key} {got!r} != config {want!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pseudo-labeling Step C: mask low-confidence pseudo-labels into a new dataset."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--run-name", default=None, help="Substituted for {run_name} in the config's paths."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override masking.mask_score_threshold (e.g. 0.5). Boxes below this are masked.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override the default sibling of label/, named label_masked_t<threshold>.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild an output directory that already exists."
    )
    parser.add_argument(
        "--no-protect-kept-boxes",
        action="store_true",
        help=(
            "Disable the kept-box carve-out — paint the full rectangle of every dropped "
            "box, even where it overlaps a box being kept. This is the naive behavior."
        ),
    )
    parser.add_argument(
        "--drop-kept-if-masked",
        type=float,
        default=None,
        help=(
            "Override masking.drop_kept_if_masked. A kept box whose pixels are painted "
            "over by at least this fraction also loses its annotation; 1.0 = only when "
            "completely painted. Inert unless --no-protect-kept-boxes is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be masked without writing any images or labels.",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shut down the machine after completion.",
    )
    args = parser.parse_args()
    if args.shutdown:
        parser.error("Power management is not supported by the release pipeline")

    config_path = Path(args.config)
    resolved = None
    if args.run_name:
        resolved = write_resolved_config(config_path.resolve(), args.run_name)
        config_path = resolved
    try:
        _run(args, config_path)
    finally:
        if resolved is not None:
            resolved.unlink(missing_ok=True)


def _run(args: argparse.Namespace, config_path: Path) -> None:
    cfg = load_config(config_path)
    mask_cfg = cfg.get("masking", {})
    out_cfg = cfg["output"]

    if not mask_cfg.get("enabled", True):
        print("masking.enabled is false — nothing to do.")
        return

    threshold = args.threshold if args.threshold is not None else mask_cfg.get(
        "mask_score_threshold", 0.4
    )
    fill_rgb = tuple(mask_cfg.get("fill_rgb", DEFAULT_FILL_RGB))
    padding_px = mask_cfg.get("mask_padding_px", 0)
    protect = mask_cfg.get("protect_kept_boxes", True)
    if args.no_protect_kept_boxes:
        protect = False
    drop_kept_if_masked = (
        args.drop_kept_if_masked if args.drop_kept_if_masked is not None
        else mask_cfg.get("drop_kept_if_masked", 1.0)
    )
    min_kept = mask_cfg.get("min_kept_boxes", 1)
    jpeg_quality = mask_cfg.get("jpeg_quality", 95)

    weights_json_path = Path(out_cfg["weights_json_path"])
    # ':g' so 0.50 and 0.5 name the same directory.
    out_root = Path(
        args.out_dir or weights_json_path.parent.parent / f"label_masked_t{threshold:g}"
    )

    manifest_path = out_root / "mask_manifest.json"
    if manifest_path.exists() and not args.force and not args.dry_run:
        problems = _mismatches(manifest_path, threshold, mask_cfg)
        if problems:
            raise SystemExit(
                f"{out_root} exists but was built differently:\n"
                + "\n".join(f"    {x}" for x in problems)
                + "\n  Delete it, or pass --force to rebuild."
            )
        print(f"{out_root} already built at t={threshold:g} — nothing to do.")
        return

    if not weights_json_path.exists():
        raise FileNotFoundError(
            f"confidence_weights.json not found: {weights_json_path}\n"
            "Run filter_and_convert.py first — this step consumes its output."
        )

    print(f"Reading filtered pseudo-labels from {weights_json_path} ...")
    with open(weights_json_path) as f:
        weights = json.load(f)
    print(f"  {len(weights):,} images, "
          f"{sum(len(v) for v in weights.values()):,} boxes.")

    print(
        f"\nMasking boxes with w_sam3 < {threshold} "
        f"(fill={tuple(fill_rgb)}, padding={padding_px}px, "
        f"protect_kept_boxes={protect}, "
        f"drop_kept_if_masked={drop_kept_if_masked}) ..."
    )
    if args.dry_run:
        print("  [dry run — no files will be written]")

    manifest, kept_weights, stats, masked_fracs = run_masking(
        weights=weights,
        out_root=out_root,
        threshold=threshold,
        fill_rgb=fill_rgb,
        padding_px=padding_px,
        protect_kept_boxes=protect,
        drop_kept_if_masked=drop_kept_if_masked,
        min_kept_boxes=min_kept,
        jpeg_quality=jpeg_quality,
        dry_run=args.dry_run,
    )

    print_summary(stats, masked_fracs, threshold)

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    weights_out = out_root / "confidence_weights.json"
    with open(weights_out, "w") as f:
        json.dump(kept_weights, f, indent=2)

    dataset_yaml = out_root / "dataset.yaml"
    write_dataset_yaml(dataset_yaml, cfg["classes"])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nMasking complete. Originals were not modified.")
    print(f"  Masked dataset root : {out_root}/")
    print(f"    images/           : {out_root / 'images'}/")
    print(f"    labels/           : {out_root / 'labels'}/")
    print(f"    confidence_weights: {weights_out}")
    print(f"    dataset.yaml      : {dataset_yaml}")
    print(f"    mask_manifest.json: {manifest_path}")
    print(f"\n  To train on this, point config.yaml's pl_data at:")
    print(f"    dataset_yaml     : {dataset_yaml}")
    print(f"    weights_json_path: {weights_out}")
    print(f"\n  To revert: rm -rf {out_root}")

    maybe_shutdown(cfg, force=args.shutdown)


if __name__ == "__main__":
    main()
