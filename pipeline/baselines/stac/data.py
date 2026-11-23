# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ALD contributors: modified extensions; distribution notice dated 2026-11-23.
# See THIRD_PARTY_NOTICES.md for provenance and modification scope.
"""data.py — how STAC data reaches the model: the weighted dataset, and the balanced loader.

  STACDataset      a YOLODataset that stamps one constant w_image on every sample it emits
  balanced loader  two single-source STACDataset instances behind two InfiniteDataLoaders,
                   merged into a fixed labeled/pseudo-labeled batch

trainer.py wires both. Mirrors student_training/data.py's split so the two are directly comparable.


═══ STACDataset ════════════════════════════════════════════════════════════════════════════

STACDataset is a YOLODataset that stamps one constant weight on every sample.

Why STACDataset is ~60 lines where student_training's Phase2YOLODataset is ~370
---------------------------------------------------------------------
STAC's three choices, taken together, collapse the whole per-image/per-box weighting apparatus
into a single constant per data source:

  - balanced loading builds ONE dataset instance per source (GT-only and PL-only), so an
    instance is never mixed;
  - a mosaic composes its other three tiles from `self.dataset` — the instance that built the
    transform — so a mosaic is single-source too, structurally;
  - the loss weights the whole sample by lambda_u, with no per-box confidence.

So the GT instance stamps w_image = 1.0 and the PL instance stamps w_image = lambda_u, and that
value is EXACT for a mosaicked sample rather than an approximation of one.

What that removes, relative to the student training dataset this was copied from:

  - `_Phase2Mosaic` / `_Phase2RandomPerspective` / `_inject_weight_carry`. Nothing rides through
    augmentation any more, so ultralytics' own Mosaic and RandomPerspective are used UNTOUCHED.
    That is worth stating plainly: `_Phase2Mosaic._cat_labels` was a method-body copy of
    ultralytics' internals, and the copy silently omitted the 'mosaic_border' key for months,
    letterboxing the 2x mosaic canvas down to imgsz instead of cropping it and training every
    mosaicked object at half its true scale. There is no body copy here, so that class of bug
    cannot occur and nothing needs re-verifying on an ultralytics upgrade.
  - the `mosaic_w_image: primary | average` axis — the question it answers ("what weight does a
    mosaic of GT and PL tiles get?") cannot arise when mosaics are single-source.
  - per-box weights, the coordinate matcher against confidence_weights.json, and
    `match_tol`. STAC has no per-box confidence.
  - the mixup/cutmix/copy_paste guard. It existed because those transforms compose boxes in ways
    the per-box carry could not follow; with nothing to carry, they need no guard. They stay at
    ultralytics' 0.0 defaults regardless.


═══ The balanced loader ════════════════════════════════════════════════════════════════════

STAC's balanced labeled/unlabeled batching, via TWO dataloaders.

What STAC does
--------------
STAC enforces the labeled:unlabeled ratio STRUCTURALLY rather than statistically.
`DataFrom2List` (google-research/ssl_detection, detection/data.py) zips
`itertools.cycle(labeled_idxs)` against one shuffled pass of the unlabeled indices and yields one
labeled + one unlabeled datapoint per step — a hard 1:1, with epoch length `len(unlabeled)`.

Without it, a plain concatenation of the two sets behind one shuffled loader gives a BINOMIAL
ratio per batch. At this project's 5/95 split most batches would be almost entirely
pseudo-labeled, and the realized weighting would depend on whatever ratio the sampler happened
to produce rather than on lambda_u.

What this module does
---------------------
Two `STACDataset` instances (GT-only, PL-only), each behind its own `InfiniteDataLoader`, pulled
in lockstep and merged into one batch:

    batch = merge_batches(next(gt_loader.iterator), next(pl_loader.iterator))

`InfiniteDataLoader.iterator` is persistent and infinite (its batch_sampler is wrapped in
`_RepeatSampler`, which loops `while True: yield from iter(self.sampler)`), so the smaller pool
cycles forever with no epoch boundary — exactly STAC's `itertools.cycle`.

Two deliberate departures from the paper:

  - The configured `batch` is SPLIT (16 -> 8 GT + 8 PL), not doubled. The merged batch is exactly
    `batch`, so VRAM, `accumulate`, the `nbs` weight-decay scaling and the LR schedule are
    unchanged against every other run in this repo. STAC pairs one labeled with one unlabeled
    datapoint, which would double the effective batch.
  - The ratio is a parameter (`ratio`, default 0.5 = STAC's 1:1) rather than structurally fixed.

Mosaic purity is free
---------------------
Each dataset builds its OWN transform pipeline — YOLODataset.build_transforms -> v8_transforms(self)
-> Mosaic(dataset=self). Both of `Mosaic.get_indexes`' paths (the dataset's own `buffer`, and
`randint(0, len(dataset) - 1)`) are therefore confined to the single instance that built the
transform, so a mosaic can only ever compose tiles from ONE source. That is what makes
STACDataset's single constant `w_image` exact rather than an approximation, and it covers
MixUp / CutMix / dataset-sourced CopyPaste too (all at 0.0 by default regardless).

Reproducibility
---------------
`ultralytics.data.build.build_dataloader` hardcodes
`generator.manual_seed(6148914691236517205 + RANK)` — identical for every loader it builds, and
not derived from `args.seed` at all. `_build_loader` below owns the seed instead: two loaders
sharing one generator seed would draw correlated permutations, and a seed that ignores
`args.seed` means "seed 42" would not actually pin the data order. The generator drives both the
shuffle order (read by the RandomSampler) and mosaic tile selection (Mosaic.get_indexes uses the
stdlib `random` module in the worker process; torch derives each worker's base seed from the
loader generator, and ultralytics' `seed_worker` reseeds `random`/`numpy` from
`torch.initial_seed()` there). Keeping `worker_init_fn=seed_worker` is what makes mosaic
composition reproducible — do not drop it.

Version note: `_build_loader` mirrors ultralytics 8.4.x's `data/build.py:build_dataloader`. It is
bypassed rather than called because that helper exposes no seed parameter, and the generator is
consumed inside `InfiniteDataLoader.__init__` (which calls `super().__iter__()` eagerly), so it
cannot be re-seeded after construction. Re-verify against build.py on an ultralytics upgrade.

Copied from student_training/data.py. `split_source_paths` is gone: there is no combined
`train:` list to partition here — the two sources are two config keys.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from ultralytics.data import YOLODataset
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLOConcatDataset
from ultralytics.utils import LOGGER

class STACDataset(YOLODataset):
    """YOLODataset that attaches a constant `w_image` to every sample it yields.

    One instance per source: 1.0 for the labeled half, lambda_u for the pseudo-labeled half.

    Parameters
    ----------
    w_image : the weight every sample from this dataset carries into the loss. The loss
        multiplies the sample's entire detection loss by it — foreground and background anchors
        alike — which is what STAC's `lambda_u * l_u` means.
    """

    def __init__(self, *args: Any, w_image: float = 1.0, **kwargs: Any) -> None:
        self.w_image = float(w_image)
        super().__init__(*args, **kwargs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample = super().__getitem__(i)
        sample["w_image"] = torch.tensor(self.w_image, dtype=torch.float32)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate, then stack `w_image` into a (batch_size,) tensor.

        YOLODataset.collate_fn decides what to do with a key by a hardcoded name list and leaves
        anything it does not recognise as a TUPLE of per-sample tensors, which the loss cannot
        index. Stacking here is what makes `batch["w_image"]` a tensor.
        """
        new_batch = YOLODataset.collate_fn(batch)
        if "w_image" in new_batch:
            new_batch["w_image"] = torch.stack(list(new_batch["w_image"]))
        return new_batch


# Distinct offsets so the two loaders draw independent permutations from one base seed.
_GT_SEED_OFFSET = 0
_PL_SEED_OFFSET = 1


def split_batch(batch: int, ratio: float = 0.5) -> tuple[int, int]:
    """Split a batch size into (n_gt, n_pl), with `ratio` of the slots going to PL.

    The GT count is FLOORED, so an odd batch gives the extra slot to PL — the larger pool.
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
    `batch_idx` from 0, so the PL half must be shifted by the GT half's image count before the
    two are concatenated, or the loss would attribute PL boxes to GT images — and, here, would
    give them the GT samples' w_image of 1.0.

    Everything else is a plain concatenation: tensors on dim 0 (which covers `w_image`, already
    a (bs,) tensor thanks to STACDataset.collate_fn) and sequence-valued keys (`im_file`,
    `ori_shape`, `resized_shape`, `ratio_pad`) by sequence concat.

    Call this BEFORE preprocess_batch, never after: multi_scale draws ONE random size per call,
    so preprocessing the halves separately would resize them differently and the concat would
    fail. Merging first also keeps the /255 and .to(device) passes single.
    """
    if gt_batch.keys() != pl_batch.keys():
        only_gt = sorted(set(gt_batch) - set(pl_batch))
        only_pl = sorted(set(pl_batch) - set(gt_batch))
        raise ValueError(
            f"merge_batches: batches have different keys (GT-only: {only_gt}, PL-only: {only_pl})."
        )
    if "img" not in gt_batch:
        raise ValueError("merge_batches: batches have no 'img' key; are these collated batches?")
    if gt_batch["img"].shape[1:] != pl_batch["img"].shape[1:]:
        raise ValueError(
            f"merge_batches: image shapes differ (GT {tuple(gt_batch['img'].shape)}, "
            f"PL {tuple(pl_batch['img'].shape)}). Both datasets must be built with rect=False "
            "and the same imgsz so every sample is square imgsz."
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


def _build_loader(
    dataset: Any,
    batch: int,
    workers: int,
    seed: int,
    seed_offset: int,
) -> InfiniteDataLoader:
    """Build one single-source InfiniteDataLoader. See the module docstring's version note.

    Differs from ultralytics' build_dataloader in exactly two ways: the generator is seeded from
    `seed + seed_offset` rather than a hardcoded constant, and drop_last is unconditional — a
    short tail batch would silently change the GT:PL ratio for that step.
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

    The two sub-loaders index the real GT and PL datasets directly, so this exists only to
    satisfy what ultralytics' engine reads off `train_loader.dataset`:

      - `close_mosaic(hyp)` — inherited from YOLOConcatDataset, which fans out to both
        sub-datasets and re-runs their build_transforms.
      - `labels` — DetectionTrainer.plot_training_labels reads it whenever args.plots is on, and
        ConcatDataset has no such attribute.
      - `collate_fn` — overridden to STACDataset's. Unreachable today (each sub-loader collates
        for itself), but the inherited YOLODataset one leaves `w_image` as a tuple, so anyone who
        later puts a single loader over this class would hand the loss the wrong type.
      - `__len__` — DELIBERATELY images consumed per epoch, not the sum of the pool sizes. The
        engine computes `iterations = ceil(len(train_loader.dataset) / max(batch, nbs)) * epochs`
        and feeds it to build_optimizer, where it decides the `optimizer: "auto"` choice. With an
        epoch defined as one pass over PL (GT cycling), the pool sum would misreport that count.
    """

    def __init__(self, datasets: Sequence[Any], images_per_epoch: int) -> None:
        super().__init__(list(datasets))
        self._images_per_epoch = int(images_per_epoch)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return STACDataset.collate_fn(batch)

    def __len__(self) -> int:
        """Images consumed per epoch — see the class docstring. NOT the total pool size."""
        return self._images_per_epoch

    @property
    def labels(self) -> list[dict]:
        """All sub-datasets' labels, concatenated. Read by plot_training_labels."""
        return [label for dataset in self.datasets for label in dataset.labels]


class BalancedDualLoader:
    """Two single-source InfiniteDataLoaders pulled in lockstep, yielding one merged batch.

    Quacks like the InfiniteDataLoader `engine/trainer.py` expects: `len()`, iteration,
    `.dataset`, `.reset()` and `.num_workers`. `.sampler` is DDP-only and raises — DDP is
    rejected at construction instead.

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

        Pulls from each loader's PERSISTENT infinite iterator rather than iterating the loaders,
        so whichever pool is shorter simply cycles. `.iterator` is read fresh on every pull
        because `reset()` replaces it — caching a reference would keep feeding from workers
        holding the pre-close_mosaic transforms.

        Deliberately side-effect-free until the first `next()`: the engine calls `iter()` twice
        per epoch (once for the plain `enumerate`, then again for the TQDM-wrapped one) and
        discards the first without consuming it.
        """
        for _ in range(self.steps_per_epoch):
            yield merge_batches(next(self.gt_loader.iterator), next(self.pl_loader.iterator))

    def reset(self) -> None:
        """Rebuild both iterators. Called after close_mosaic rebuilds the transforms."""
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
    ratio: float,
    seed: int,
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
            "ratio, so this fails instead."
        )
    if len(pl_dataset) < n_pl:
        raise ValueError(
            f"PL pool has {len(pl_dataset)} images but each batch needs {n_pl}. "
            "build_dataloader would silently clamp the batch to the pool size and break the "
            "ratio, so this fails instead."
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
        f"STAC balanced loader: {n_gt} GT + {n_pl} PL per batch, {steps_per_epoch} batches/epoch "
        f"(one pass over PL; GT cycles) — pools: {len(gt_dataset)} GT / {len(pl_dataset)} PL, "
        f"seed {seed}, workers {w_gt}+{w_pl}"
    )
    return loader
