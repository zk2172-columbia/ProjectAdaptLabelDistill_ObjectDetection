"""trainer.py — the student training trainers: one spec, one base trainer, two opt-in loader variants.

  Phase2Spec                    one experiment arm: how weights are built and how they are applied
  Phase2DetectionTrainer        the default — one combined GT+PL dataset behind one shuffled loader
  BalancedDualDetectionTrainer  opt-in: a fixed GT:PL ratio in every batch  (`balanced_loader: true`)
  PureMosaicDetectionTrainer    opt-in PROTOTYPE: single-source mosaics, default batching

The two variants differ from the base class in exactly one respect each — the train dataloader
and the train dataset respectively — and inherit everything else. train.py's ARMS registry picks
between them per arm.


═══ Phase2Spec and Phase2DetectionTrainer ══════════════════════════════════════════════════


An arm is one Phase2Spec passed through `overrides`, wiring the matching Phase2YOLODataset flags
and Phase2DetectionLoss config. It carries both halves so the dataset cannot emit weight keys the
loss does not consume; train.py's ARMS registry holds one per experiment.


═══ BalancedDualDetectionTrainer ═══════════════════════════════════════════════════════════


Selected per arm by `balanced_loader: true` in config.yaml's experiments: block (or
`--balanced-loader` for every GT+PL arm at once). train.py picks this class instead of
Phase2DetectionTrainer and supplies `pl_img_path` / `balance_ratio`; the default is off, so an arm
that does not mention it trains exactly as before. Because a balanced run writes to
`<arm>-balanced`, the same arm can be run both ways under one --run-name — that pair is the A/B
this class exists to make possible.

What it changes
---------------
ONLY the train dataloader. Everything else — Phase2Spec, the weighted loss, get_model, the
w_eff.csv diagnostics, the validator, the val loader — is inherited from Phase2DetectionTrainer
unchanged, so an arm run through this trainer differs from the same arm run through the normal one
in exactly one respect: batch composition.

Instead of one Phase2YOLODataset over the combined GT+PL image list behind one shuffled loader,
it builds TWO single-source datasets and merges a fixed slice of each into every batch (default
8 GT + 8 PL at batch=16). See data.py for the mechanism, the STAC correspondence, and
why two dataset instances make every mosaic single-source for free.

Usage (standalone — the normal path is train.py, which builds these overrides for you)
--------------------------------------------------------------------------------------
    from trainer import BalancedDualDetectionTrainer, Phase2Spec

    trainer = BalancedDualDetectionTrainer(overrides={
        ...,                                     # the same overrides train.py builds
        "data": "<...>/combined_dataset.yaml",   # must contain BOTH sources in `train:`
        "spec": Phase2Spec(),
        "weights_json_path": "<...>/confidence_weights.json",
        "lambda1": 1.0,
        "pl_img_path": "<...>/label/images",     # REQUIRED — which `train:` entry is PL
        "balance_ratio": 0.5,                    # optional, default 0.5
    })
    trainer.train()

Two notes on the loss under a balanced loader. Both are behaviour changes to be aware of when
reading results, not bugs:

  - A fixed 50/50 ratio massively over-samples the small GT pool relative to the combined loader
    (GT is ~5% of the images but half of every batch). That is the intended STAC behaviour, but it
    means an epoch is no longer one pass over the data and GT sees many more gradient steps.

Single-GPU only; DDP raises.


═══ PureMosaicDetectionTrainer ═════════════════════════════════════════════════════════════


PROTOTYPE. This is a diagnostic arm for one question — how much of the balanced loader's effect is
mosaic purity rather than batch composition — NOT a pipeline default and not a candidate to become
one. It must stay strictly opt-in: leave `--pure-mosaic` off the `steps:` args in
config.yaml, and leave `pure_mosaic:` out of its `experiments:` blocks. Only a config
written specifically to measure this should ever set it. If the measurement turns out to be worth
keeping, that is a decision to raise explicitly, not to reach by letting the default drift.

Opt-in, per arm: `pure_mosaic: true` in config.yaml's experiments: block, or
`--pure-mosaic` for every GT+PL arm at once. Off by default. Runs land under `<arm>-puremosaic`,
so the same arm can be run both ways under one --run-name and compared. Mutually exclusive with
`balanced_loader` (which already makes mosaics pure, as a side effect of splitting the loader).

Why this exists
---------------
The balanced loader changes TWO things at once relative to the default combined loader:

  1. batch composition — a fixed GT:PL ratio in every batch instead of a binomial draw, and the
     small GT pool is massively over-sampled (half of every batch at ~5% of the images);
  2. mosaic purity — a mosaic can no longer compose GT and PL tiles into one sample, because each
     source has its own dataset instance and therefore its own Mosaic.

A gt_pl_* vs gt_pl_*-balanced comparison therefore cannot say which of the two produced the gap.
This trainer isolates (2): mosaics become single-source while batch composition stays EXACTLY the
default binomial shuffle over the combined image list. The triple

    <arm>              mixed mosaics, binomial batches      (Phase2DetectionTrainer)
    <arm>-puremosaic   pure mosaics,  binomial batches      (this class)
    <arm>-balanced     pure mosaics,  fixed-ratio batches   (BalancedDualDetectionTrainer)

separates the two effects: puremosaic - plain is the mosaic-purity effect alone, and
balanced - puremosaic is what fixed-ratio batching adds on top.

What it changes
---------------
ONLY `build_dataset` for mode="train", and only its RETURN TYPE: instead of one Phase2YOLODataset
over the combined GT+PL image list, it returns a `PureMosaicConcatDataset` holding two
single-source Phase2YOLODatasets. Everything else — the Phase2Spec, the weighted loss, get_model,
the w_eff.csv diagnostics, the validator, the val loader — is inherited from Phase2DetectionTrainer
unchanged.

Crucially the LOADER is untouched: ultralytics' own DetectionTrainer.get_dataloader wraps whatever
build_dataset returns in `build_dataloader(..., shuffle=True, ...)`, so the sampler, its hardcoded
generator seed, drop_last and the worker setup are bit-for-bit the ones the default arm uses. An
epoch is still one pass over every image, and GT:PL per batch is still binomial at the corpus
ratio. Only which pool a mosaic's other three tiles may come from has changed.

Why a ConcatDataset makes mosaics pure
--------------------------------------
`ConcatDataset.__getitem__` delegates the whole call to the sub-dataset that owns the index, and
each Phase2YOLODataset builds its OWN transform pipeline — YOLODataset.build_transforms ->
v8_transforms(self) -> Mosaic(dataset=self). Both of `Mosaic.get_indexes`' paths (the dataset's own
`buffer`, and `randint(0, len(self.dataset) - 1)`) are therefore confined to the single instance
that built the transform, so a mosaic can only ever compose tiles from ONE source. The same
argument covers MixUp / CutMix / dataset-sourced CopyPaste, which the config pins to 0.0 anyway.
This is the identical mechanism data.py relies on; the difference is that there the two
datasets sit behind two loaders, and here behind one.

This does NOT collapse the `mosaic_w_image` axis: a pure-PL mosaic still composes four PL images
with four different mean(w_sam3) values, so "primary" and "average" still differ. What changes is
that GT's hard 1.0 can no longer leak into a PL sample's weight or vice versa.

Usage (standalone — the normal path is train.py, which builds these overrides for you)
--------------------------------------------------------------------------------------
    from trainer import Phase2Spec, PureMosaicDetectionTrainer

    trainer = PureMosaicDetectionTrainer(overrides={
        ...,                                     # the same overrides train.py builds
        "data": "<...>/combined_dataset.yaml",   # must contain BOTH sources in `train:`
        "spec": Phase2Spec(),
        "weights_json_path": "<...>/confidence_weights.json",
        "lambda1": 1.0,
        "pl_img_path": "<...>/label/images",     # REQUIRED — which `train:` entry is PL
    })
    trainer.train()
"""

from __future__ import annotations

import csv
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.data.dataset import YOLOConcatDataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from data import (
    DEFAULT_SEED,
    Phase2YOLODataset,
    build_balanced_dual_loader,
    split_batch,
    split_source_paths,
)
from loss import Phase2DetectionModel

# Extra overrides each trainer consumes and strips before ultralytics sees them.
_PHASE2_KEYS = ("weights_json_path", "lambda1", "zerogate", "spec")
_BALANCED_KEYS = ("pl_img_path", "balance_ratio")
_PURE_MOSAIC_KEYS = ("pl_img_path",)

# Augmentations that compose boxes in ways Phase2YOLODataset does NOT carry per-box weights
# through. (mosaic + affine ARE carried, so they are allowed.)
_UNSUPPORTED_AUGS = ("mixup", "cutmix", "copy_paste")

_W_EFF_HEADER = ("epoch", "mean_w_eff", "mean_w_cls_eff")


@dataclass(frozen=True)
class Phase2Spec:
    """One experiment arm: how weights are built (dataset) and how they are applied (loss).

    Dataset side:
        per_box        carry per-box 'box_weights' through augmentation
        per_image      emit the per-image scalar 'w_image'
        mosaic_w_image "primary" | "average" — how a mosaicked sample gets its w_image

    Loss side:
        loc_weight_src what weights box (IoU) + DFL — "image" | "box" | "none"
        cls_weight_src what weights the cls (BCE) loss tensor — "image" | "box" | "none"
    """

    per_box: bool = False
    per_image: bool = True
    mosaic_w_image: str = "primary"
    loc_weight_src: str = "image"
    cls_weight_src: str = "image"
    stock_loss: bool = False

    def __post_init__(self) -> None:
        if self.mosaic_w_image not in ("primary", "average"):
            raise ValueError(
                f"mosaic_w_image must be 'primary' or 'average'; got {self.mosaic_w_image!r}"
            )
        if self.loc_weight_src == "box" or self.cls_weight_src == "box":
            if not self.per_box:
                raise ValueError("a 'box' weight source requires per_box=True")
        if (self.loc_weight_src == "image" or self.cls_weight_src == "image") and not self.per_image:
            raise ValueError("an 'image' weight source requires per_image=True")
        if self.stock_loss and (
            self.per_box
            or self.per_image
            or self.loc_weight_src != "none"
            or self.cls_weight_src != "none"
        ):
            # Stock v8DetectionLoss has no weighting hooks, so any weight configured here would
            # be computed and then silently discarded. Fail instead of pretending it applied.
            raise ValueError(
                "stock_loss=True cannot be combined with any weighting: set per_box=False, "
                "per_image=False and loc/cls_weight_src='none'"
            )

    def dataset_flags(self) -> dict[str, Any]:
        return {
            "per_box": self.per_box,
            "per_image": self.per_image,
            "mosaic_w_image": self.mosaic_w_image,
        }

    def loss_cfg(self) -> dict[str, Any]:
        """Loss knobs as PRIMITIVES ONLY — this dict is stored on the model and pickled into
        best.pt. See Phase2DetectionModel's docstring for why a dataclass must not go there."""
        return {
            "loc_weight_src": self.loc_weight_src,
            "cls_weight_src": self.cls_weight_src,
        }


class Phase2DetectionTrainer(DetectionTrainer):
    """DetectionTrainer using Phase2DetectionModel + Phase2YOLODataset, configured by a Phase2Spec.

    Extra overrides (stripped before ultralytics sees them):
      spec              : Phase2Spec    — the arm
      weights_json_path : str | None    — path to pseudo-labeling confidence_weights.json
      lambda1           : float         — global PL loss scale (default 1.0)
      zerogate          : float         — PL boxes with raw w_sam3 below this get weight 0.0

    Logs per-epoch effective weights to <save_dir>/w_eff.csv for every arm — see _write_w_eff.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ) -> None:
        overrides = dict(overrides or {})
        self._phase2_params: dict[str, Any] = {
            k: overrides.pop(k, None) for k in _PHASE2_KEYS
        }
        if self._phase2_params["lambda1"] is None:
            self._phase2_params["lambda1"] = 1.0
        if self._phase2_params["zerogate"] is None:
            self._phase2_params["zerogate"] = 0.0
        self.spec: Phase2Spec = self._phase2_params["spec"] or Phase2Spec()
        super().__init__(cfg, overrides, _callbacks)

        self._w_eff_sum = 0.0
        self._w_cls_eff_sum = 0.0
        self._w_eff_n = 0
        self._w_eff_started = False
        self.add_callback("on_train_batch_end", self._accumulate_w_eff)
        self.add_callback("on_fit_epoch_end", self._write_w_eff)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return the model whose criterion implements this arm's loss.

        stock_loss arms get ultralytics' own DetectionModel, so training runs against the
        INSTALLED v8DetectionLoss rather than this package's reimplementation of it. Phase2's
        loss reduces to stock bit-for-bit with no weighting configured (pinned by the golden
        harness), but a baseline arm should not depend on that equivalence holding — using the
        real class means an ultralytics change can never silently move the baseline.
        """
        model_cls = DetectionModel if self.spec.stock_loss else Phase2DetectionModel
        model = model_cls(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if not self.spec.stock_loss:
            # Set before load() so a criterion built at any point sees the right config.
            # Primitives only — this is pickled into the checkpoint.
            model._phase2_loss_cfg = self.spec.loss_cfg()
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Return a Phase2YOLODataset; the weights are applied for train mode only.

        Val mode keeps the arm's dataset flags but passes neutral weights, so every w_image and
        every box weight is 1.0 and the loss reduces to stock v8DetectionLoss.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)

        if mode == "train":
            if self.spec.per_box:
                self._assert_box_safe_augment()
            weights_json_path = self._phase2_params["weights_json_path"]
            lambda1 = self._phase2_params["lambda1"]
            zerogate = self._phase2_params["zerogate"]
            if zerogate and not self.spec.per_box:
                # The gate is a per-box weight; nothing else can carry it.
                raise ValueError("zerogate requires a per_box arm")
        else:
            weights_json_path, lambda1, zerogate = None, 1.0, 0.0

        return Phase2YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=(mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=getattr(self.args, "task", "detect"),
            classes=getattr(self.args, "classes", None),
            data=self.data,
            fraction=getattr(self.args, "fraction", 1.0) if mode == "train" else 1.0,
            weights_json_path=weights_json_path,
            lambda1=lambda1,
            zerogate=zerogate,
            **self.spec.dataset_flags(),
        )

    def _build_source_dataset(
        self, img_path: list[str] | str, batch: int, *, weights_json_path: str | None
    ):
        """Build one SINGLE-SOURCE train dataset with an explicit weights JSON.

        Used by the loader subclasses that split the combined dataset back into its GT and PL
        halves (trainer.py, trainer.py). Reuses build_dataset's body rather
        than copying it — that body encodes a dozen ultralytics constructor arguments plus the
        arm's dataset flags, and a copy would drift the first time either changes. The only thing
        that varies per source is which weights JSON is passed, which build_dataset reads from
        `_phase2_params`, so it is swapped for the duration of the call.

        Deliberately calls `Phase2DetectionTrainer.build_dataset` UNBOUND: a subclass whose
        build_dataset returns a multi-source dataset (PureMosaicDetectionTrainer) would otherwise recurse
        into itself here.

        Passing weights_json_path=None for the GT half is not merely equivalent to passing the
        real JSON: today GT vs PL identity is *stem membership* in that JSON
        (data.py:_build_image_weights), so a stem collision between the two sources would
        silently reweight a GT image. Splitting the sources structurally removes that hazard.
        """
        saved = self._phase2_params["weights_json_path"]
        self._phase2_params["weights_json_path"] = weights_json_path
        try:
            return Phase2DetectionTrainer.build_dataset(self, img_path, mode="train", batch=batch)
        finally:
            self._phase2_params["weights_json_path"] = saved

    def _assert_box_safe_augment(self) -> None:
        """Fail if an unsupported box-composing augmentation is enabled (weights not carried)."""
        enabled = {k: getattr(self.args, k, 0.0) for k in _UNSUPPORTED_AUGS
                   if getattr(self.args, k, 0.0)}
        if enabled:
            raise ValueError(
                "Phase2DetectionTrainer does not carry per-box weights through "
                f"{tuple(_UNSUPPORTED_AUGS)}; set these to 0 in the config's training: block. "
                f"Enabled: {enabled}. (mosaic + affine ARE supported.) "
                "See README.md, \"Augmentation\"."
            )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    # ── w_eff diagnostics ─────────────────────────────────────────────────────────────────

    def _accumulate_w_eff(self, _trainer) -> None:
        """Accumulate the criterion's per-batch effective weights (no-op before the first loss)."""
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        w_eff = getattr(criterion, "last_w_eff", None)
        if w_eff is None:
            return
        self._w_eff_sum += float(w_eff)
        w_cls_eff = getattr(criterion, "last_w_cls_eff", None)
        if w_cls_eff is not None:
            self._w_cls_eff_sum += float(w_cls_eff)
        self._w_eff_n += 1

    def _write_w_eff(self, _trainer) -> None:
        """Append this epoch's mean effective weights to <save_dir>/w_eff.csv, then reset.

        Logged for every arm EXCEPT stock_loss ones: v8DetectionLoss exposes no last_w_eff, so
        _accumulate_w_eff never fires and no file is written. Nothing is lost — a stock arm
        applies no weighting, so both columns would be a constant 1.0 by definition.

        Under an absolute normalizer the reported box/cls/dfl values scale
        with each batch's PL fraction, so training curves are not comparable across arms or
        against baseline; these columns are what let you tell "the model improved" apart from
        "that epoch was PL-heavy" (a visible instance: w_eff steps at the close_mosaic boundary,
        where mosaic tile-averaging stops pulling PL weights toward GT's 1.0). On the relative-
        normalizer arms the weight cancels by construction and both columns are exactly 1.0.
        Compare arms on val mAP, not loss.
        """
        if self._w_eff_n == 0 or RANK not in (-1, 0):
            return
        mean_w_eff = self._w_eff_sum / self._w_eff_n
        mean_w_cls_eff = self._w_cls_eff_sum / self._w_eff_n
        self._w_eff_sum, self._w_cls_eff_sum, self._w_eff_n = 0.0, 0.0, 0

        path = self.save_dir / "w_eff.csv"
        if not self._w_eff_started:
            self._w_eff_started = True
            self._rotate_stale_w_eff(path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_W_EFF_HEADER)
            writer.writerow([self.epoch, f"{mean_w_eff:.6f}", f"{mean_w_cls_eff:.6f}"])

    @staticmethod
    def _rotate_stale_w_eff(path: Path) -> None:
        """Move aside a w_eff.csv whose header does not match the current schema.

        output.exist_ok is True, so a re-run can land in a directory holding a CSV written with a
        different set of columns (e.g. the pre-refactor 2-column one). Appending would produce a
        ragged file, so the old one is renamed rather than corrupted or discarded.
        """
        if not path.exists():
            return
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header == list(_W_EFF_HEADER):
            return
        path.replace(path.with_name("w_eff.prev.csv"))


class BalancedDualDetectionTrainer(Phase2DetectionTrainer):
    """Phase2DetectionTrainer whose TRAIN loader draws a fixed GT:PL ratio from two datasets.

    Extra overrides (stripped before ultralytics sees them):
      pl_img_path   : str | list  — the PL image directory/directories. Required: it is what
                                    partitions the combined yaml's `train:` list into the two
                                    sources. Typically <yolo_data>/<run_name>/label/images.
      balance_ratio : float       — fraction of each batch drawn from PL (default 0.5 = STAC 1:1).

    The loader seed is NOT a new key — it comes from `self.args.seed` (training.seed: 42 in
    config.yaml), so the data order stays in lockstep with the `init_seeds(seed + 1 + RANK)`
    ultralytics already applies to weight init and main-process augmentation.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ) -> None:
        overrides = dict(overrides or {})
        self._pl_img_path = overrides.pop("pl_img_path", None)
        ratio = overrides.pop("balance_ratio", None)
        self._balance_ratio = 0.5 if ratio is None else float(ratio)
        if self._pl_img_path is None:
            raise ValueError(
                f"{type(self).__name__} requires a `pl_img_path` override — the PL image "
                "directory, used to split the combined dataset's `train:` list into its GT and PL "
                f"halves. Known extra keys: {_BALANCED_KEYS}."
            )
        super().__init__(cfg, overrides, _callbacks)

    # ── dataloader ─────────────────────────────────────────────────────────────────────────

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"
    ):
        """Return the balanced dual loader for train; the stock single loader for val.

        Validation is deliberately untouched: it runs on the GT val split alone, has no PL half to
        balance against, and the parent already builds it with neutral weights so the reported
        metrics stay comparable across arms.
        """
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        if RANK != -1:
            raise NotImplementedError(
                f"{type(self).__name__} does not support DDP — engine/trainer.py calls "
                "train_loader.sampler.set_epoch on a single sampler, and this loader drives two. "
                "Train on a single device (device: '0')."
            )

        gt_paths, pl_paths = split_source_paths(dataset_path, self._pl_img_path)
        n_gt, n_pl = split_batch(batch_size, self._balance_ratio)

        # Each dataset gets its OWN source's per-batch count as batch_size: it sizes that
        # dataset's mosaic buffer (BaseDataset.max_buffer_length = min(ni, batch_size * 8, 1000)),
        # and each loader really does draw that many images per step.
        gt_dataset = self._build_source_dataset(gt_paths, n_gt, weights_json_path=None)
        pl_dataset = self._build_source_dataset(
            pl_paths, n_pl, weights_json_path=self._phase2_params["weights_json_path"]
        )

        seed = getattr(self.args, "seed", None)
        return build_balanced_dual_loader(
            gt_dataset,
            pl_dataset,
            batch=batch_size,
            workers=self.args.workers,
            ratio=self._balance_ratio,
            seed=DEFAULT_SEED if seed is None else int(seed),
            rank=rank,
        )

    # _build_source_dataset (one single-source train dataset with an explicit weights JSON) is
    # inherited from Phase2DetectionTrainer — trainer.py needs the same helper.


class PureMosaicConcatDataset(YOLOConcatDataset):
    """Two single-source Phase2YOLODatasets behind one dataset interface.

    Indexed directly by the ordinary shuffled loader (unlike balanced_loader's
    BalancedConcatDataset, which is never indexed), so it must get two things right that plain
    YOLOConcatDataset does not:

      - `collate_fn` — the inherited one is YOLODataset's, which leaves student training's `w_image` and
        `box_weights` as raw tuples and would break the loss. Phase2YOLODataset.collate_fn
        tensorises both.
      - `labels` — DetectionTrainer.plot_training_labels reads it whenever args.plots is on, and
        ConcatDataset has no such attribute.

    `close_mosaic` is inherited from YOLOConcatDataset, which fans out to both sub-datasets and
    therefore re-runs Phase2YOLODataset.build_transforms (re-injecting the weight-carrying
    transform overrides). `__len__` is ConcatDataset's — the total image count, which is correct
    here: an epoch is one pass over every image, exactly as in the default arm.
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """student training's collate, not YOLODataset's — see the class docstring."""
        return Phase2YOLODataset.collate_fn(batch)

    @property
    def labels(self) -> list[dict]:
        """All sub-datasets' labels, concatenated. Read by plot_training_labels."""
        return [label for dataset in self.datasets for label in dataset.labels]


class PureMosaicDetectionTrainer(Phase2DetectionTrainer):
    """Phase2DetectionTrainer whose TRAIN dataset is split by source, so mosaics never mix.

    Extra overrides (stripped before ultralytics sees them):
      pl_img_path : str | list — the PL image directory/directories. Required: it is what
                                 partitions the combined yaml's `train:` list into the two sources.
                                 Typically <yolo_data>/<run_name>/label/images.

    Adds no seed knob: the loader is ultralytics' own, so data order comes from the same generator
    seed every non-balanced student training run already uses.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ) -> None:
        overrides = dict(overrides or {})
        self._pl_img_path = overrides.pop("pl_img_path", None)
        if self._pl_img_path is None:
            raise ValueError(
                f"{type(self).__name__} requires a `pl_img_path` override — the PL image "
                "directory, used to split the combined dataset's `train:` list into its GT and PL "
                f"halves. Known extra keys: {_PURE_MOSAIC_KEYS}."
            )
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Two single-source datasets concatenated for train; the stock single one for val.

        Validation is deliberately untouched: it runs on the GT val split alone, has no PL half to
        separate, and augmentation is off there anyway, so there is no mosaic to keep pure.
        """
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)

        gt_paths, pl_paths = split_source_paths(img_path, self._pl_img_path)

        # Both sub-datasets get the FULL batch size, unlike the balanced loader's n_gt / n_pl
        # split: one batch here can be up to `batch` images from either source, and batch_size
        # only sizes that dataset's mosaic buffer
        # (BaseDataset.max_buffer_length = min(ni, batch_size * 8, 1000)).
        gt_dataset = self._build_source_dataset(gt_paths, batch, weights_json_path=None)
        pl_dataset = self._build_source_dataset(
            pl_paths, batch, weights_json_path=self._phase2_params["weights_json_path"]
        )
        return PureMosaicConcatDataset([gt_dataset, pl_dataset])
