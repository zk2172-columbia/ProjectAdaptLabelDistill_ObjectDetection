# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""data.py — how student training data reaches the model: the weighted dataset, and the balanced loader.

Two halves, in the order data flows through them:

  Phase2YOLODataset   a YOLODataset that attaches SAM3 confidence weights to each sample and
                      carries them through mosaic + affine  (§ "The weighted dataset" below)
  balanced loader     two single-source dataset instances behind two InfiniteDataLoaders, merged
                      into a fixed-ratio GT/PL batch      (§ "The balanced loader" below)

The second is opt-in; the first is always used. trainer.py wires both.


═══ The weighted dataset ═══════════════════════════════════════════════════════════════════


Three flags select what it emits:

  per_box        carry a PER-BOX weight ('box_weights') through augmentation, aligned with cls
  per_image      emit a PER-IMAGE scalar ('w_image') for each sample
  mosaic_w_image how a mosaicked sample gets its w_image — "primary" or "average"

Weight assignment
-----------------
  - GT images (stem absent from confidence_weights.json): w_image = 1.0, every box weight = 1.0.
  - PL images:
        w_image     = lambda1
        box_weights = w_sam3 · lambda1, or 0.0 if w_sam3 < zerogate. Matched per box BY
                      COORDINATES (cx,cy,w,h within match_tol) rather than by row order, because
                      ultralytics dedups label rows on load (np.unique reorders/drops them).
    The gate reads the RAW w_sam3, before lambda1, so lambda1 can never change which boxes gate.

How the weights survive augmentation
------------------------------------
  box_weights ride at full fidelity through mosaic + affine, so those augmentations stay enabled:

    attach   update_labels_info    — coordinate-match each box to its JSON w_sam3 (1.0 for GT)
    mosaic   _Phase2Mosaic         — concatenate the 4 tiles' weights, apply the same zero-area
                                     `good` mask as cls
    affine   _Phase2RandomPerspective — intercept box_candidates()'s keep-mask
    emit     Format mutates labels in place and preserves extra keys, so they survive
    collate  concatenate into a (K,) tensor aligned with cls

  w_image cannot ride per box — it is one scalar for a sample that may be composed of tiles from
  both GT and PL images. Two strategies:
    "primary"  attach AFTER the whole pipeline, so a mosaic keeps the primary tile's weight.
    "average"  attach to every tile in update_labels_info (before mosaic), and let _Phase2Mosaic
               average the tiles actually used. A mosaic of 1 PL + 3 GT tiles then gets
               (w_pl + 3·1.0)/4 rather than whichever tile happened to be primary.
    "average" is still an approximation — the true per-region answer varies within the composed
    image — but it makes the weight depend on what the sample actually contains rather than on
    tile ordering.

mixup/cutmix/copy_paste are NOT carried (they compose boxes in ways nothing here overrides).
They are off by default and Phase2DetectionTrainer asserts they stay off when per_box is on.

The two transform overrides are injected by swapping __class__ on the existing pipeline instances
(build_transforms), which preserves all their state. _Phase2Mosaic._cat_labels is a method-body
copy from ultralytics 8.4.x — re-verify on upgrade, and note that a body copy's real hazard is
OMITTING a key the downstream transforms read, not miscomputing one: dropping 'mosaic_border'
silently halved every mosaicked object's scale for months. tests/test_dataset_perbox.py's
test_cat_labels_key_parity_with_ultralytics pins the output keys against the installed Mosaic so
an upgrade that adds one is caught here rather than in a sweep. _Phase2RandomPerspective instead
wraps __call__ and intercepts box_candidates, so it does not depend on RandomPerspective's
internals.


═══ The balanced loader ════════════════════════════════════════════════════════════════════


Opt-in, per arm: `balanced_loader: true` in config.yaml's experiments: block, or
`--balanced-loader` for every GT+PL arm at once. Off by default, so an arm that does not mention
it trains exactly as before. Balanced runs write to `<arm>-balanced`, so the same arm can be run
both ways under one --run-name and compared. See trainer.py for the wiring.

The problem
-----------
student training trains on a plain CONCATENATION of GT and PL images (utils.create_combined_dataset_yaml
writes both image dirs into one `train:` list), behind one dataset and one shuffled loader. Two
consequences:

  1. GT:PL per batch is binomial, not fixed. At the pipeline's ~5/95 split most batches are almost
     entirely PL, so the realized weighting depends on whatever ratio the sampler produced and
     lambda1 partly cancels under the relative normalizer (see loss.py's [D2]).
  2. A mosaic freely mixes GT and PL tiles, because ultralytics' Mosaic.get_indexes draws the
     other 3 tiles from `self.dataset.buffer` (or `randint(0, len(self.dataset) - 1)`) — the whole
     dataset. That is why data.py has a `mosaic_w_image: primary|average` axis at all: one
     scalar cannot describe a sample composed of tiles from both sources.

What STAC does
--------------
STAC (arXiv 2005.04757, google-research/ssl_detection) enforces the ratio structurally rather than
statistically. `DataFrom2List` (detection/data.py) zips `itertools.cycle(labeled_idxs)` against one
shuffled pass of the unlabeled indices and yields one labeled + one unlabeled datapoint per step —
a hard 1:1, with epoch length `len(unlabeled)`.

What this module does
---------------------
Two Phase2YOLODataset instances (GT-only, PL-only), each behind its own InfiniteDataLoader, pulled
in lockstep and merged into one batch:

    batch = merge_batches(next(gt_loader.iterator), next(pl_loader.iterator))

`InfiniteDataLoader.iterator` is persistent and infinite (its batch_sampler is wrapped in
`_RepeatSampler`, which loops `while True: yield from iter(self.sampler)`), so the smaller pool
cycles forever with no epoch boundary — exactly STAC's `itertools.cycle`.

Deliberate departures from STAC:

  - The configured `batch` is SPLIT (16 -> 8 GT + 8 PL), not doubled. The merged batch is exactly
    `batch`, so VRAM, `accumulate`, the `nbs` weight-decay scaling and the LR schedule are
    unchanged against the existing gt_pl_* arms. STAC pairs one labeled with one unlabeled
    datapoint, which would double the effective batch.
  - The GT:PL ratio is a parameter (`ratio`, default 0.5) rather than structurally 1:1.

Mosaic purity is free
---------------------
Each dataset builds its own transform pipeline — YOLODataset.build_transforms -> v8_transforms(self)
-> Mosaic(dataset=self). Both of Mosaic.get_indexes' paths (the dataset's own `buffer`, and
`randint(0, len(dataset) - 1)`) are therefore confined to the single instance that built the
transform. Two disjoint dataset instances means a mosaic can only ever compose tiles from ONE
source. The same argument covers MixUp / CutMix / dataset-sourced CopyPaste, which the config pins
to 0.0 regardless.

This does NOT collapse the `mosaic_w_image` axis: a pure-PL mosaic still composes four PL images
with four different mean(w_sam3) values, so "primary" and "average" still differ and "average" is
still the better approximation. What changes is that GT's hard 1.0 can no longer leak into a PL
sample's weight or vice versa — both variants now stay inside one source's weight distribution.

Reproducibility
---------------
`build_dataloader` hardcodes `generator.manual_seed(6148914691236517205 + RANK)` — identical for
every loader it builds and not derived from `args.seed` at all. `_build_loader` below owns the seed
instead: two loaders sharing one generator seed would draw correlated permutations, and a fixed
seed is what makes a run repeatable. The generator drives both the shuffle order (read by the
RandomSampler) and mosaic tile selection (Mosaic.get_indexes uses the stdlib `random` module in the
worker process; torch derives each worker's base seed from the loader generator, and ultralytics'
`seed_worker` reseeds `random`/`numpy` from `torch.initial_seed()` there). Keeping
`worker_init_fn=seed_worker` is what makes mosaic composition reproducible — do not drop it.

Version note: `_build_loader` mirrors ultralytics 8.4.21's `data/build.py:build_dataloader`. It is
bypassed rather than called because that helper exposes no seed parameter, and the generator is
consumed inside `InfiniteDataLoader.__init__` (which calls `super().__iter__()` eagerly), so it
cannot be re-seeded after construction. Re-verify against build.py on an ultralytics upgrade.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Mosaic, RandomPerspective
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLOConcatDataset
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

from utils import load_weights_json
from confidence import load_confidence
import os
import math

def load_weights_json_full(path: str | Path) -> dict[str, list[dict[str, float]]]:
    """Load confidence_weights.json keyed by image *stem*, preserving per-box coords + w_sam3.

    utils.load_weights_json returns only the w_sam3 list per stem, which is all the per-image
    path needs; the per-box path additionally needs each box's coordinates to match against.
    """
    return load_confidence(path)


# ── Transform overrides (carry the weights alongside cls) ──────────────────────────────────
# Each overrides a single self-contained method and is injected via __class__ swap, so it
# inherits all instance state. Both bodies are guarded on key presence, so ONE pair of classes
# serves every flag combination.

class _Phase2Mosaic(Mosaic):
    """Mosaic that carries box_weights per box and averages the tiles' w_image.

    Both behaviours are conditional on the corresponding key being present on the tile labels, so
    this is correct for per-box-only, per-image-only, and both-at-once configurations alike.
    """

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        if not mosaic_labels:
            return {}
        cls, instances, box_weights = [], [], []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
            if "box_weights" in labels:
                box_weights.append(labels["box_weights"])
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            # NOT optional. RandomPerspective reads this key to decide what it is looking at:
            # present -> the image is the 2x-size mosaic canvas, crop an imgsz window out of it;
            # absent -> treat it as an ordinary image, so LetterBox squashes the whole 2x canvas
            # down to imgsz and every object ends up at HALF its true scale. v8_transforms leaves
            # RandomPerspective.border at its (0, 0) default, so this is the only thing that
            # tells it otherwise. Omitting it is silent — the weights stay correctly aligned to
            # boxes that are simply the wrong size — and cost the per-box arms ~0.10 mAP50-95 for
            # every epoch before close_mosaic.
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if box_weights:
            final_labels["box_weights"] = np.concatenate(box_weights, 0)[good]
        tile_weights = [lb["w_image"] for lb in mosaic_labels if "w_image" in lb]
        if tile_weights:
            final_labels["w_image"] = sum(tile_weights) / len(tile_weights)
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class _Phase2RandomPerspective(RandomPerspective):
    """RandomPerspective that carries box_weights through the box_candidates drop-mask.

    Version-robust by design: rather than reimplementing apply_instances/__call__ (whose body
    differs across ultralytics point releases — e.g. 8.4.71 splits affine into
    get_params/apply_image/apply_instances, while 8.4.21 does it all inline in __call__ and never
    calls apply_instances at all, silently no-op'ing a body-copy override), this intercepts
    box_candidates() — the single low-level call BOTH implementations use, with an identical
    signature, to decide which boxes survive the affine warp. Wrapping __call__ (not replacing it)
    preserves whatever internal structure the installed version uses; only the keep-mask is
    captured and applied to box_weights.
    """

    _last_keep_mask: np.ndarray | None = None

    def box_candidates(self, *args: Any, **kwargs: Any) -> np.ndarray:
        i = super().box_candidates(*args, **kwargs)
        self._last_keep_mask = i
        return i

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        box_weights = labels.pop("box_weights", None)
        self._last_keep_mask = None
        labels = super().__call__(labels)
        if box_weights is not None:
            # box_candidates ran (captured a mask) -> apply it; otherwise (e.g. no instances to
            # warp) nothing was dropped, so box_weights passes through unchanged.
            if self._last_keep_mask is not None:
                box_weights = box_weights[self._last_keep_mask]
            labels["box_weights"] = box_weights
        return labels


def _inject_weight_carry(transform: Any, per_box: bool) -> None:
    """Recursively swap Mosaic (and, when per_box, RandomPerspective) to the carrying subclasses.

    RandomPerspective is only swapped when there are per-box weights to carry: w_image is a plain
    scalar that the affine warp cannot invalidate, so the per-image-only configurations leave the
    stock transform untouched.
    """
    items = getattr(transform, "transforms", None)
    if items is not None:
        for t in items:
            if type(t) is Mosaic:
                t.__class__ = _Phase2Mosaic
            elif per_box and type(t) is RandomPerspective:
                t.__class__ = _Phase2RandomPerspective
            _inject_weight_carry(t, per_box)
    # CopyPaste (disabled by config) nests its own pre_transform Compose.
    pre = getattr(transform, "pre_transform", None)
    if pre is not None:
        _inject_weight_carry(pre, per_box)


class Phase2YOLODataset(YOLODataset):
    """YOLODataset carrying SAM3 confidence weights at per-image and/or per-box granularity.

    Parameters
    ----------
    weights_json_path : pseudo-labeling confidence_weights.json, or None — with None every image is
        treated as GT (all weights 1.0) and the loss reduces to stock v8DetectionLoss.
    lambda1 : global PL loss scale multiplied into every PL weight, per-image and per-box alike.
    zerogate : a PL box whose raw w_sam3 is below this gets weight 0.0 — its cls/box/dfl loss
        is zeroed while its pixels stay in the image. Applied before lambda1, so lambda1 can
        never change which boxes are gated. 0.0 disables the gate.
    per_box : carry per-box 'box_weights' through augmentation.
    per_image : emit the per-image scalar 'w_image'.
    mosaic_w_image : "primary" | "average" — see the module docstring.
    match_tol : per-component coordinate match tolerance, normalized units.
    """

    def __init__(
        self,
        *args: Any,
        weights_json_path: str | Path | None = None,
        lambda1: float = 1.0,
        zerogate: float = 0.0,
        per_box: bool = False,
        per_image: bool = True,
        mosaic_w_image: str = "primary",
        match_tol: float = 1e-4,
        pseudo_label_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if mosaic_w_image not in ("primary", "average"):
            raise ValueError(
                f"mosaic_w_image must be 'primary' or 'average'; got {mosaic_w_image!r}"
            )
        if not 0.0 <= zerogate <= 1.0:
            raise ValueError(f"zerogate must be in [0, 1]; got {zerogate!r}")
        if not math.isfinite(lambda1) or lambda1 < 0:
            raise ValueError("lambda1 must be finite and non-negative")
        if not math.isfinite(match_tol) or match_tol <= 0:
            raise ValueError("match_tol must be positive and finite")
        self._pseudo_label_paths = [Path(os.path.abspath(p)) for p in (pseudo_label_paths or [])]
        if weights_json_path is not None and not self._pseudo_label_paths:
            raise ValueError("Weighted training requires explicit pseudo_label_paths")
        self._weights_json_path = weights_json_path
        self._lambda1 = lambda1
        self._zerogate = zerogate
        self._per_box = per_box
        self._per_image = per_image
        self._mosaic_w_image = mosaic_w_image
        self._match_tol = match_tol

        # The per-box path needs each box's coordinates to match against; the per-image path only
        # needs the w_sam3 values. Load whichever is sufficient, exactly once.
        self._weight_map: dict[str, list[dict[str, float]]] = {}
        self._w_sam3_map: dict[str, list[float]] = {}
        if weights_json_path is not None:
            if per_box:
                self._weight_map = load_weights_json_full(weights_json_path)
                self._w_sam3_map = {
                    stem: [b["w_sam3"] for b in boxes]
                    for stem, boxes in self._weight_map.items()
                }
            else:
                self._w_sam3_map = load_weights_json(weights_json_path)

        super().__init__(*args, **kwargs)
        # Check source identity before any augmented sample is consumed.
        self._validate_confidence_coverage()
        # self.labels is populated by super().__init__ -> get_labels()
        if self._per_image:
            self._build_image_weights()

    # ── Per-image weights ──────────────────────────────────────────────────────────────────

    def _validate_confidence_coverage(self) -> None:
        if self._weights_json_path is None:
            return
        listed = set()
        directories = []
        for root in self._pseudo_label_paths:
            if root.is_file():
                for line in root.read_text().splitlines():
                    if line.strip():
                        p = Path(line.strip())
                        listed.add(Path(os.path.abspath(root.parent / p if not p.is_absolute() else p)))
            elif root.is_dir():
                directories.append(root)
            else:
                raise FileNotFoundError(f"Pseudo-label source does not exist: {root}")
        seen = set()
        for label in self.labels:
            path = Path(os.path.abspath(label["im_file"]))
            stem = path.stem
            if stem in seen:
                raise ValueError(f"Duplicate image stem in training data: {stem}")
            seen.add(stem)
            is_pl = path in listed or any(path.is_relative_to(root) for root in directories)
            has_weights = stem in self._w_sam3_map
            if is_pl != has_weights:
                raise ValueError(f"GT/PL confidence coverage mismatch: {path}")
            if is_pl and len(self._w_sam3_map[stem]) != len(label["cls"]):
                raise ValueError(f"Pseudo-label/confidence box count mismatch: {path}")

    def _build_image_weights(self) -> None:
        """Compute self.image_weights — lambda1 for PL images, 1.0 for GT.

        _stem_index lets update_labels_info look a tile's weight up by stem, since mosaic tiles
        arrive there without their dataset index.
        """
        self.image_weights: list[float] = [
            self._lambda1 if Path(label["im_file"]).stem in self._w_sam3_map else 1.0
            for label in self.labels
        ]

        if self._mosaic_w_image == "average":
            self._stem_index: dict[str, int] = {
                Path(label["im_file"]).stem: i for i, label in enumerate(self.labels)
            }

    # ── Per-box weights ────────────────────────────────────────────────────────────────────

    def _weight_for_box(self, bbox: np.ndarray, json_boxes: list[dict[str, float]], class_id: int) -> float:
        """This box's loss weight: 1.0 for GT, 0.0 if gated, else w_sam3 · lambda1."""
        cx, cy, w, h = (float(v) for v in bbox)
        best, best_dist = None, self._match_tol
        matches = 0
        for jb in json_boxes:
            if jb["class_id"] != int(class_id):
                continue
            d = max(abs(cx - jb["cx"]), abs(cy - jb["cy"]), abs(w - jb["w"]), abs(h - jb["h"]))
            if d <= self._match_tol:
                matches += 1
                if d <= best_dist:
                    best_dist, best = d, jb["w_sam3"]
        if matches != 1:
            raise ValueError(f"Expected one confidence match for PL box, found {matches}")
        if best < self._zerogate:
            return 0.0
        return best * self._lambda1

    def gate_stats(self) -> dict[str, int]:
        """How many PL boxes the gate zeroes across the whole weights JSON. Diagnostics only."""
        total = sum(len(b) for b in self._w_sam3_map.values())
        zeroed = sum(
            1 for ws in self._w_sam3_map.values() for w in ws if w < self._zerogate
        )
        return {"pl_boxes": total, "zeroed": zeroed, "kept": total - zeroed}

    # ── Hooks ──────────────────────────────────────────────────────────────────────────────

    def update_labels_info(self, label: dict) -> dict:
        """Attach box_weights and/or this tile's w_image, so mosaic sees both per tile."""
        weights = None
        if self._per_box:
            # Read bboxes BEFORE super() pops them into an Instances object.
            bboxes = np.asarray(label["bboxes"], dtype=np.float64).reshape(-1, 4)  # xywh
            json_boxes = self._weight_map.get(Path(label["im_file"]).stem)
            weights = (
                np.array([self._weight_for_box(b, json_boxes, c)
                          for b, c in zip(bboxes, label["cls"].reshape(-1))], dtype=np.float32)
                if json_boxes is not None
                else np.ones(len(bboxes), dtype=np.float32)  # GT image
            )

        label = super().update_labels_info(label)

        if weights is not None:
            label["box_weights"] = weights
        if self._per_image and self._mosaic_w_image == "average":
            idx = self._stem_index.get(Path(label["im_file"]).stem)
            label["w_image"] = self.image_weights[idx] if idx is not None else 1.0
        return label

    def build_transforms(self, hyp=None):
        """Build the standard pipeline, then inject the weight-carrying transform overrides."""
        transforms = super().build_transforms(hyp)
        if self._per_box or (self._per_image and self._mosaic_w_image == "average"):
            _inject_weight_carry(transforms, per_box=self._per_box)
        return transforms

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Return the transformed sample with its weight keys attached as tensors."""
        result = super().__getitem__(i)

        if self._per_box and "box_weights" in result:
            n = int(result["cls"].shape[0])
            bw = result["box_weights"]
            if len(bw) != n:
                raise RuntimeError(
                    f"Phase2YOLODataset: box_weights ({len(bw)}) misaligned with cls ({n}) for "
                    f"image {i}. A box-dropping transform is not carrying weights — check that "
                    f"mixup/cutmix/copy_paste are disabled and only mosaic/affine are enabled."
                )
            result["box_weights"] = torch.as_tensor(bw, dtype=torch.float32)

        if self._per_image:
            # "average": read back whatever the pipeline attached — this image's own value for a
            # non-mosaic sample, or _Phase2Mosaic's tile average for a mosaicked one.
            # "primary": the pipeline attached nothing, so this image's own weight is used, which
            # for a mosaic is the primary tile's.
            w = (
                result.get("w_image", self.image_weights[i])
                if self._mosaic_w_image == "average"
                else self.image_weights[i]
            )
            result["w_image"] = torch.tensor(float(w), dtype=torch.float32)

        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate; stack w_image into (batch_size,) and concatenate box_weights into (K,)."""
        new_batch = YOLODataset.collate_fn(batch)
        # The parent leaves unknown keys as tuples — convert them to tensors.
        if "w_image" in new_batch:
            new_batch["w_image"] = torch.stack(list(new_batch["w_image"]))
        if "box_weights" in new_batch:
            new_batch["box_weights"] = torch.cat(
                [torch.as_tensor(w, dtype=torch.float32) for w in new_batch["box_weights"]], 0
            )
        return new_batch


# Default RNG seed for both loaders. Phase2 configs set training.seed: 42 and
# BalancedDualDetectionTrainer forwards self.args.seed, so this only applies to direct callers.
DEFAULT_SEED = 42

# Distinct offsets so the two loaders draw independent permutations from otherwise equal seeds.
_GT_SEED_OFFSET = 0
_PL_SEED_OFFSET = 1


# ── Path partitioning ──────────────────────────────────────────────────────────────────────


def split_source_paths(
    train_paths: str | Path | Sequence[str | Path],
    pl_roots: str | Path | Sequence[str | Path],
) -> tuple[list[str], list[str]]:
    """Partition a dataset's `train:` entries into (GT paths, PL paths).

    `self.data["train"]` is the combined_dataset.yaml list after ultralytics' check_det_dataset has
    resolved it to absolute directory strings. An entry counts as PL when it IS one of `pl_roots`
    or lives under one; everything else is GT.

    Both sides are compared as resolved paths (check_det_dataset already resolved the left side),
    but the ORIGINAL strings are returned so the datasets are built from exactly the paths
    ultralytics handed us. Resolving is safe here because pseudo-labeling / prepare_data.py make
    `images/` a real directory of per-file symlinks rather than a directory symlink — see
    prepare_data.py's module docstring.

    Raises:
        ValueError: if either side comes out empty. A silent mis-partition would still "work" —
            it would just train an unbalanced loader — so it must fail loudly instead.
    """
    entries = _as_list(train_paths)
    roots = [Path(p).resolve() for p in _as_list(pl_roots)]

    gt: list[str] = []
    pl: list[str] = []
    for entry in entries:
        resolved = Path(entry).resolve()
        is_pl = any(resolved == root or resolved.is_relative_to(root) for root in roots)
        (pl if is_pl else gt).append(str(entry))

    if not gt or not pl:
        raise ValueError(
            "split_source_paths needs both a GT and a PL partition, but got "
            f"{len(gt)} GT and {len(pl)} PL entries.\n"
            f"  train paths: {entries}\n"
            f"  pl roots   : {[str(r) for r in roots]}\n"
            "Check that pl_img_path points at the PL image directory named in the combined "
            "dataset yaml (typically <yolo_data>/<run_name>/label/images)."
        )
    return gt, pl


def _as_list(value: str | Path | Sequence[str | Path]) -> list[str]:
    """Normalize a scalar-or-sequence path argument to a list of strings."""
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(v) for v in value]


# ── Batch composition ──────────────────────────────────────────────────────────────────────


def split_batch(batch: int, ratio: float = 0.5) -> tuple[int, int]:
    """Split a batch size into (n_gt, n_pl) with `ratio` of the slots going to PL.

    The GT count is FLOORED, so an odd batch gives the extra slot to PL — the larger pool in
    practice.
    """
    if batch < 2:
        raise ValueError(f"a balanced batch needs batch >= 2, got {batch}.")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be strictly between 0 and 1, got {ratio}.")
    n_gt = max(1, int(batch * (1.0 - ratio)))
    n_pl = batch - n_gt
    if n_pl < 1:
        raise ValueError(
            f"ratio={ratio} with batch={batch} leaves no PL slots (n_gt={n_gt}). "
            "Lower the GT share or raise the batch size."
        )
    return n_gt, n_pl


def merge_batches(gt_batch: dict[str, Any], pl_batch: dict[str, Any]) -> dict[str, Any]:
    """Concatenate two collated batches into one, GT images first.

    Mirrors the per-image offset YOLODataset.collate_fn applies: each loader numbers its own
    `batch_idx` from 0, so the PL half must be shifted by the GT half's image count before the two
    are concatenated, or the loss would attribute PL boxes to GT images.

    Everything else is a plain concatenation: tensors on dim 0 (which covers student training's `w_image`
    (bs,) and `box_weights` (K,) — Phase2YOLODataset.collate_fn already tensorises both), and
    sequence-valued keys (`im_file`, `ori_shape`, `resized_shape`, `ratio_pad`) by sequence concat.

    Call this BEFORE preprocess_batch, never after: multi_scale draws ONE random size per call
    (detect/train.py), so preprocessing the halves separately would resize them differently and the
    concat would fail. Merging first also keeps the /255 and .to(device) passes single.
    """
    if gt_batch.keys() != pl_batch.keys():
        only_gt = sorted(set(gt_batch) - set(pl_batch))
        only_pl = sorted(set(pl_batch) - set(gt_batch))
        raise ValueError(
            f"merge_batches: batches have different keys (GT-only: {only_gt}, PL-only: {only_pl}). "
            "Both loaders must be built with the same Phase2Spec dataset flags."
        )
    if "img" not in gt_batch:
        raise ValueError("merge_batches: batches have no 'img' key; are these collated batches?")
    if gt_batch["img"].shape[1:] != pl_batch["img"].shape[1:]:
        raise ValueError(
            f"merge_batches: image shapes differ (GT {tuple(gt_batch['img'].shape)}, "
            f"PL {tuple(pl_batch['img'].shape)}). Both datasets must be built with rect=False and "
            "the same imgsz so every sample is square imgsz."
        )

    n_gt = gt_batch["img"].shape[0]
    merged: dict[str, Any] = {}
    for key, gt_value in gt_batch.items():
        pl_value = pl_batch[key]
        if key == "batch_idx":
            merged[key] = torch.cat((gt_value, pl_value + n_gt), 0)
        elif isinstance(gt_value, torch.Tensor):
            merged[key] = torch.cat((gt_value, pl_value), 0)
        elif isinstance(gt_value, list):
            merged[key] = list(gt_value) + list(pl_value)
        elif isinstance(gt_value, tuple):
            merged[key] = tuple(gt_value) + tuple(pl_value)
        else:
            raise TypeError(
                f"merge_batches: don't know how to concatenate key {key!r} of type "
                f"{type(gt_value).__name__}."
            )
    return merged


# ── The loader ─────────────────────────────────────────────────────────────────────────────


def _build_loader(
    dataset: Any,
    batch: int,
    workers: int,
    seed: int,
    seed_offset: int,
) -> InfiniteDataLoader:
    """Build one single-source InfiniteDataLoader. See the module docstring's version note.

    Differs from ultralytics' build_dataloader in exactly two ways: the generator is seeded from
    `seed + seed_offset` rather than a hardcoded constant, and drop_last is unconditional — a short
    tail batch would silently change the GT:PL ratio for that step.
    """
    nd = torch.cuda.device_count()
    nw = min((os.cpu_count() or 1) // max(nd, 1), workers)
    generator = torch.Generator()
    generator.manual_seed(seed + seed_offset)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=nw,
        sampler=None,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=nd > 0,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )


class BalancedConcatDataset(YOLOConcatDataset):
    """The `.dataset` a BalancedDualLoader exposes. Never indexed.

    The two sub-loaders index the real GT and PL datasets directly, so this exists only to satisfy
    what ultralytics' engine reads off `train_loader.dataset`:

      - `close_mosaic(hyp)` — inherited from YOLOConcatDataset, which fans out to both sub-datasets
        and therefore re-runs Phase2YOLODataset.build_transforms (re-injecting the weight-carrying
        transform overrides).
      - `labels` — DetectionTrainer.plot_training_labels reads it, and runs unconditionally when
        args.plots is on. ConcatDataset has no such attribute; ultralytics' own train_world.py
        stubs plot_training_labels out for exactly this reason.
      - `collate_fn` — overridden to student training's rather than the inherited YOLODataset one. Not
        reached today (each sub-loader collates for itself); see the method docstring for why it
        is defined regardless.
      - `__len__` — DELIBERATELY REDEFINED as images consumed per epoch, not the sum of the pool
        sizes. The engine computes
        `iterations = ceil(len(train_loader.dataset) / max(batch, nbs)) * epochs` and feeds it to
        build_optimizer, where it decides the `optimizer: "auto"` choice. With an epoch defined as
        one pass over PL (GT cycling), the pool sum would misreport that step count; images/epoch
        makes it exact. The comet callback's `len(dataset) // batch_size` becomes exact too.
    """

    def __init__(self, datasets: Sequence[Any], images_per_epoch: int) -> None:
        super().__init__(list(datasets))
        self._images_per_epoch = int(images_per_epoch)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """student training's collate, not the YOLODataset one YOLOConcatDataset would inherit.

        UNREACHABLE TODAY, and deliberately correct anyway. Nothing collates through this class:
        the two sub-loaders each collate with their own dataset's collate_fn and merge_batches
        joins the results, so this shim is only ever read for `len()`, `labels` and
        `close_mosaic`. But the inherited YOLODataset.collate_fn decides by a hardcoded set of key
        names and leaves anything else as a tuple of per-sample tensors — which is exactly what
        student training's `w_image` and `box_weights` are. Anyone who later puts a SINGLE loader over this
        class (as trainer.py does over its own concat) would silently hand the loss
        tuples where it expects tensors. Overriding it costs one line and removes that trap.
        """
        return Phase2YOLODataset.collate_fn(batch)

    def __len__(self) -> int:
        """Images consumed per epoch — see the class docstring. NOT the total pool size."""
        return self._images_per_epoch

    @property
    def labels(self) -> list[dict]:
        """All sub-datasets' labels, concatenated. Read by plot_training_labels."""
        return [label for dataset in self.datasets for label in dataset.labels]


class BalancedDualLoader:
    """Two single-source InfiniteDataLoaders pulled in lockstep, yielding one merged batch.

    Quacks like the InfiniteDataLoader `engine/trainer.py` expects: `len()`, iteration, `.dataset`,
    `.reset()` and `.num_workers`. `.sampler` is DDP-only and raises — DDP is rejected at
    construction instead.

    Attributes:
        n_gt / n_pl (int): images per batch from each source. `n_gt + n_pl == batch`.
        steps_per_epoch (int): batches per epoch — one pass over the PL loader, STAC's convention
            (`DataFrom2List.__len__` returns the unlabeled count). GT cycles.
    """

    def __init__(
        self,
        gt_loader: InfiniteDataLoader,
        pl_loader: InfiniteDataLoader,
        dataset: BalancedConcatDataset,
        n_gt: int,
        n_pl: int,
    ) -> None:
        self.gt_loader = gt_loader
        self.pl_loader = pl_loader
        self.dataset = dataset
        self.n_gt = n_gt
        self.n_pl = n_pl
        self.steps_per_epoch = len(pl_loader)

    def __len__(self) -> int:
        """Batches per epoch. Read once as `nb` and used for warmup, `base_idx` and the pbar."""
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield `steps_per_epoch` merged batches.

        Pulls from each loader's PERSISTENT infinite iterator rather than iterating the loaders, so
        whichever pool is shorter simply cycles. `.iterator` is read fresh on every pull because
        `reset()` replaces it — caching a reference would keep feeding from workers holding the
        pre-close_mosaic transforms.

        Deliberately side-effect-free until first `next()`: the engine calls `iter()` twice per
        epoch (once for the plain `enumerate`, then again for the TQDM-wrapped one) and discards
        the first without consuming it.
        """
        for _ in range(self.steps_per_epoch):
            yield merge_batches(next(self.gt_loader.iterator), next(self.pl_loader.iterator))

    def reset(self) -> None:
        """Rebuild both underlying iterators. Called after close_mosaic rebuilds the transforms."""
        self.gt_loader.reset()
        self.pl_loader.reset()

    @property
    def num_workers(self) -> int:
        """Total worker processes across both loaders — the engine logs this at train start."""
        return self.gt_loader.num_workers + self.pl_loader.num_workers

    @property
    def sampler(self) -> Any:
        """DDP-only on a normal loader. Unreachable here — DDP is rejected at construction."""
        raise NotImplementedError(
            "BalancedDualLoader has no single sampler; it drives two. DDP is not supported — "
            "train on one device (device: '0')."
        )


def build_balanced_dual_loader(
    gt_dataset: Any,
    pl_dataset: Any,
    batch: int,
    workers: int,
    ratio: float = 0.5,
    seed: int = DEFAULT_SEED,
    rank: int = -1,
) -> BalancedDualLoader:
    """Build a BalancedDualLoader over an already-constructed GT and PL dataset.

    Args:
        batch: TOTAL images per merged batch — split into n_gt + n_pl, not doubled.
        workers: total worker budget, split between the two loaders in proportion to their share
            of the batch (each gets at least 1 when workers > 0).
        ratio: fraction of each batch drawn from PL. 0.5 is STAC's 1:1.
        seed: base RNG seed; the two loaders take distinct offsets from it.
        rank: must be -1. DDP would need a DistributedSampler per loader and a `sampler.set_epoch`
            the engine calls on a single loader; not supported.
    """
    if rank != -1:
        raise NotImplementedError(
            "BalancedDualLoader does not support DDP: engine/trainer.py calls "
            "train_loader.sampler.set_epoch on a single sampler, and this loader drives two. "
            "Train on a single device (device: '0')."
        )

    n_gt, n_pl = split_batch(batch, ratio)
    if len(gt_dataset) < n_gt:
        raise ValueError(
            f"GT pool has {len(gt_dataset)} images but each batch needs {n_gt}. "
            "build_dataloader would silently clamp the batch to the pool size and break the "
            "ratio, so this fails instead. Lower `batch`, or raise `ratio`."
        )
    if len(pl_dataset) < n_pl:
        raise ValueError(
            f"PL pool has {len(pl_dataset)} images but each batch needs {n_pl}. "
            "build_dataloader would silently clamp the batch to the pool size and break the "
            "ratio, so this fails instead. Lower `batch`, or lower `ratio`."
        )

    if workers > 0:
        w_pl = max(1, round(workers * n_pl / batch))
        w_gt = max(1, workers - w_pl)
    else:
        w_gt = w_pl = 0

    gt_loader = _build_loader(gt_dataset, n_gt, w_gt, seed, _GT_SEED_OFFSET)
    pl_loader = _build_loader(pl_dataset, n_pl, w_pl, seed, _PL_SEED_OFFSET)

    steps_per_epoch = len(pl_loader)
    dataset = BalancedConcatDataset(
        [gt_dataset, pl_dataset], images_per_epoch=steps_per_epoch * batch
    )
    loader = BalancedDualLoader(gt_loader, pl_loader, dataset, n_gt, n_pl)

    LOGGER.info(
        f"balanced loader: {n_gt} GT + {n_pl} PL per batch, {steps_per_epoch} batches/epoch "
        f"(one pass over PL; GT cycles) — pools: {len(gt_dataset)} GT / {len(pl_dataset)} PL, "
        f"seed {seed}, workers {w_gt}+{w_pl}"
    )
    return loader
