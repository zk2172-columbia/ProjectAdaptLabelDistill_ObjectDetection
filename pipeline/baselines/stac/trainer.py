"""trainer.py — STACTrainer: the student trainer.

One class, because STAC is one configuration. It differs from ultralytics' DetectionTrainer in
exactly three places:

  build_dataset   returns a STACDataset carrying a constant w_image
  get_dataloader  returns the balanced dual loader for train (the stock one for val)
  get_model       returns STACDetectionModel, whose criterion is STACDetectionLoss

plus the w_eff.csv diagnostic callbacks.

Extra overrides, consumed here and stripped before ultralytics sees them:

  gt_img_path  : str | list — the LABELED image directory/directories
  pl_img_path  : str | list — the PSEUDO-LABELED image directory/directories
  lambda_u     : float      — the weight every PL sample carries into the loss
  balance_ratio: float      — PL's share of each batch (0.5 = STAC's 1:1)

The two sources are passed EXPLICITLY rather than partitioned out of a combined dataset yaml's
`train:` list (which is what student_training does). Splitting a list by path prefix is a guess that can
silently mis-partition and still "work"; two config keys cannot.

The loader seed is not an override: it comes from `self.args.seed`, so the data order stays in
lockstep with the `init_seeds(seed + 1 + RANK)` ultralytics already applies to weight init and
main-process augmentation. One seed, everywhere.

Copied from student_training/trainer.py + student_training/trainer_balanced.py, dropping Phase2Spec and its six
flags, the arm registry they configured, `_assert_box_safe_augment` (nothing carries per-box
weights, so mixup/cutmix/copy_paste need no guard), and `_build_source_dataset`'s unbound-call
recursion workaround (no subclass here returns a multi-source dataset).
"""

from __future__ import annotations

import csv
from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from data import STACDataset, build_balanced_dual_loader, split_batch
from loss import STACDetectionModel

# Keys consumed here and stripped before ultralytics sees them.
_STAC_KEYS = ("gt_img_path", "pl_img_path", "lambda_u", "balance_ratio")

_W_EFF_HEADER = ("epoch", "mean_w_eff")


class STACTrainer(DetectionTrainer):
    """DetectionTrainer with STAC's balanced batching and lambda_u-weighted unsupervised loss."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ) -> None:
        overrides = dict(overrides or {})
        params = {k: overrides.pop(k, None) for k in _STAC_KEYS}

        missing = [k for k in ("gt_img_path", "pl_img_path") if not params[k]]
        if missing:
            raise ValueError(
                f"STACTrainer requires {missing} — the labeled and pseudo-labeled image "
                f"directories. Known extra keys: {_STAC_KEYS}."
            )
        self._gt_img_path = params["gt_img_path"]
        self._pl_img_path = params["pl_img_path"]
        self._lambda_u = 1.0 if params["lambda_u"] is None else float(params["lambda_u"])
        self._balance_ratio = 0.5 if params["balance_ratio"] is None else float(params["balance_ratio"])

        super().__init__(cfg, overrides, _callbacks)

        self._w_eff_sum = 0.0
        self._w_eff_n = 0
        self._w_eff_started = False
        self.add_callback("on_train_batch_end", self._accumulate_w_eff)
        self.add_callback("on_fit_epoch_end", self._write_w_eff)

    # ── model ──────────────────────────────────────────────────────────────────────────────

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = STACDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    # ── data ───────────────────────────────────────────────────────────────────────────────

    def build_dataset(
        self,
        img_path: str | list[str],
        mode: str = "train",
        batch: int | None = None,
        w_image: float = 1.0,
    ):
        """Build one single-source STACDataset.

        `w_image` is 1.0 everywhere except the pseudo-labeled train dataset, which gets
        lambda_u. Validation runs on the GT val split, so its 1.0 makes the loss reduce to stock
        v8DetectionLoss there — val numbers stay comparable to any other run's.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return STACDataset(
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
            w_image=w_image,
        )

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"
    ):
        """The balanced dual loader for train; the stock single loader for val.

        `dataset_path` is ignored for train — the two sources come from gt_img_path/pl_img_path,
        which is the whole point of passing them explicitly. Validation is untouched: it runs on
        the GT val split alone, has no pseudo-labeled half to balance against, and augmentation
        is off there anyway.
        """
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        if RANK != -1:
            raise NotImplementedError(
                "STACTrainer does not support DDP — engine/trainer.py calls "
                "train_loader.sampler.set_epoch on a single sampler, and this loader drives two. "
                "Train on a single device (device: '0')."
            )

        n_gt, n_pl = split_batch(batch_size, self._balance_ratio)

        # Each dataset gets its OWN source's per-batch count as batch_size: it sizes that
        # dataset's mosaic buffer (BaseDataset.max_buffer_length = min(ni, batch_size * 8, 1000)),
        # and each loader really does draw that many images per step.
        gt_dataset = self.build_dataset(self._gt_img_path, "train", n_gt, w_image=1.0)
        pl_dataset = self.build_dataset(self._pl_img_path, "train", n_pl, w_image=self._lambda_u)

        return build_balanced_dual_loader(
            gt_dataset,
            pl_dataset,
            batch=batch_size,
            workers=self.args.workers,
            ratio=self._balance_ratio,
            seed=int(getattr(self.args, "seed", 0)),
            rank=rank,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    # ── w_eff diagnostics ──────────────────────────────────────────────────────────────────

    def _accumulate_w_eff(self, _trainer) -> None:
        """Accumulate the criterion's per-batch effective weight (no-op before the first loss)."""
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        w_eff = getattr(criterion, "last_w_eff", None)
        if w_eff is None:
            return
        self._w_eff_sum += float(w_eff)
        self._w_eff_n += 1

    def _write_w_eff(self, _trainer) -> None:
        """Append this epoch's mean effective weight to <save_dir>/w_eff.csv, then reset.

        Under an unweighted normalizer the reported box/cls/dfl values scale with each batch's
        pseudo-label fraction, so training curves are not comparable against a GT-only baseline
        without this column: it is what separates "the model improved" from "that epoch was
        PL-heavy". Expect a visible step at the close_mosaic boundary. Compare against baselines
        on val mAP, not on loss.
        """
        if self._w_eff_n == 0 or RANK not in (-1, 0):
            return
        mean_w_eff = self._w_eff_sum / self._w_eff_n
        self._w_eff_sum, self._w_eff_n = 0.0, 0

        path = self.save_dir / "w_eff.csv"
        if not self._w_eff_started:
            self._w_eff_started = True
            self._rotate_stale_w_eff(path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_W_EFF_HEADER)
            writer.writerow([self.epoch, f"{mean_w_eff:.6f}"])

    @staticmethod
    def _rotate_stale_w_eff(path: Path) -> None:
        """Move aside a w_eff.csv whose header does not match, rather than append a ragged file."""
        if not path.exists():
            return
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header == list(_W_EFF_HEADER):
            return
        path.replace(path.with_name("w_eff.prev.csv"))
