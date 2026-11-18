"""Phase2DetectionLoss / Phase2DetectionModel — the weighted student training loss.

Two weight sources, selected per arm by loc_weight_src / cls_weight_src:
"image" uses the per-image scalar w_image, "box" the per-anchor weight gathered from its owning
box, "none" leaves the term unweighted.

    arm                        loc_weight_src   cls_weight_src
    ──────────────────────────────────────────────────────────
    gt_pl_unweighted_absolute      image            image
    gt_pl_perbox                   box              box

Z is always the UNWEIGHTED target_scores.sum(), so weighting is absolute: it does not cancel on a
homogeneous all-PL batch, a GT-only batch reduces bit-for-bit to stock v8DetectionLoss, and a
zero-weight box still dilutes the batch by contributing to Z and to no numerator.

The BCE TARGET is never weighted — the weight multiplies the resulting loss tensor instead.
BCE's gradient w.r.t. the logit is (sigmoid(logit) - target), so scaling the target would move the
classifier's fixed point from align_metric to w·align_metric, training the model to under-report
confidence rather than merely training it less hard. At w = 0 that distinction is what makes a
gated box an ignore region rather than a "predict zero confidence here" target.

cls_weight_src="image" also down-weights background anchors on PL images, which is intended: PL
annotations are incomplete. A per-box weight structurally cannot reach them, so a gated box's
still-visible pixels are trained as hard background everywhere TAL did not assign them to it.

Background anchors: target_gt_idx is arbitrary there, so the gathered per-box weight is reset to
1.0 via fg_mask before use.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

# Neutral fallback: reduces to stock v8DetectionLoss. Phase2DetectionTrainer always sets an
# explicit config, so this only applies to a model reconstructed outside the trainer (e.g. from a
# checkpoint) — where training with the WRONG weighting would be far worse than training with none.
DEFAULT_LOSS_CFG: dict[str, Any] = {
    "loc_weight_src": "none",
    "cls_weight_src": "none",
}


class Phase2DetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with configurable per-image / per-box confidence weighting.

    Attributes:
        last_w_eff (torch.Tensor): detached scalar diagnostic — the batch's effective
            LOCALIZATION weight, target_scores_w.sum() / Z.
        last_w_cls_eff (torch.Tensor): detached scalar diagnostic — the batch's effective
            CLASSIFICATION weight, (target_scores · w_cls).sum() / Z.

    Both are 1.0 for an all-GT batch. Logged per epoch to <save_dir>/w_eff.csv: absolute
    normalization makes the reported box/cls/dfl scale with each batch's PL fraction, so training
    curves are not comparable across arms. Compare arms on val mAP, not loss.
    """

    def __init__(
        self,
        model: Any,
        *args: Any,
        loc_weight_src: str = "image",
        cls_weight_src: str = "image",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, *args, **kwargs)
        for name, src in (("loc_weight_src", loc_weight_src), ("cls_weight_src", cls_weight_src)):
            if src not in ("image", "box", "none"):
                raise ValueError(f"{name} must be 'image', 'box' or 'none'; got {src!r}")
        self.loc_weight_src = loc_weight_src
        self.cls_weight_src = cls_weight_src

    @property
    def _uses_box_weights(self) -> bool:
        return "box" in (self.loc_weight_src, self.cls_weight_src)

    @staticmethod
    def _select(src: str, w_image: torch.Tensor | None, c: torch.Tensor | None) -> torch.Tensor | None:
        """Pick the weight tensor for one loss term, or None if it is unweighted.

        Both are already shaped to broadcast against (bs, n_anchors, nc): w_image is
        (bs, 1, 1) and c is (bs, n_anchors, 1).
        """
        if src == "image":
            return w_image
        if src == "box":
            return c
        return None

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple:
        """Compute box/cls/dfl losses with the configured confidence weighting."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # ── Build targets, appending box_weights as a trailing 7th column when needed ────
        # preprocess scatters the weight to the same (b, j) slot as gt_labels/gt_bboxes, and its
        # xywh2xyxy slice (out[..., 1:5]) touches only the bbox columns, leaving it untouched.
        has_box_w = self._uses_box_weights and "box_weights" in batch
        cols = [batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]]
        if has_box_w:
            cols.append(batch["box_weights"].view(-1, 1))
        targets = torch.cat(cols, 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        if has_box_w:
            gt_labels, gt_bboxes, gt_weights = targets.split((1, 4, 1), 2)  # cls, xyxy, w
        else:
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            gt_weights = None
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ── Build the two weight tensors ─────────────────────────────────────────────────
        # w_image: (bs, 1, 1) — 1.0 for GT, lambda1 for PL. Explicit device= because
        # preprocess_batch moving every tensor key is ultralytics-version-sensitive.
        w_image = None
        if "image" in (self.loc_weight_src, self.cls_weight_src) and "w_image" in batch:
            w_image = batch["w_image"].to(device=self.device, dtype=dtype)[:, None, None]

        # c: (bs, n_anchors, 1) — each anchor's owning-box weight, guarded at background [D6].
        c = None
        if gt_weights is not None:
            c = gt_weights.squeeze(-1).gather(1, target_gt_idx)  # (bs, n_anchors)
            c = torch.where(fg_mask, c, torch.ones_like(c)).unsqueeze(-1)

        w_loc = self._select(self.loc_weight_src, w_image, c)
        w_cls = self._select(self.cls_weight_src, w_image, c)

        target_scores_w = target_scores if w_loc is None else target_scores * w_loc
        target_scores_sum = max(target_scores.sum(), 1)

        # Diagnostics only — never participate in the graph.
        self.last_w_eff = (target_scores_w.sum() / target_scores_sum).detach()
        target_scores_cls = target_scores if w_cls is None else target_scores * w_cls
        self.last_w_cls_eff = (target_scores_cls.sum() / target_scores_sum).detach()

        # Cls loss (BCE) — UNWEIGHTED target, then the weight applied to the loss tensor.
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if w_cls is not None:
            bce_loss = bce_loss * w_cls
        loss[1] = bce_loss.sum() / target_scores_sum

        # Box loss (IoU + DFL) — weighted via target_scores_w.sum(-1).
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores_w,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class Phase2DetectionModel(DetectionModel):
    """DetectionModel whose criterion is a Phase2DetectionLoss configured by _phase2_loss_cfg.

    _phase2_loss_cfg must hold ONLY primitives (str / bool). Ultralytics pickles the model object
    into best.pt, and train.py's _export_standard_checkpoint makes that checkpoint portable by
    swapping __class__ to stock DetectionModel — it does not strip instance attributes. A
    dataclass or custom object here would therefore be pickled into every "standard" checkpoint
    and silently re-introduce a student_training import dependency at load time, which is exactly what the
    standard export exists to remove.

    Phase2DetectionTrainer.get_model sets this per arm; the class-level default is the neutral
    stock-equivalent config (see DEFAULT_LOSS_CFG).
    """

    _phase2_loss_cfg: dict[str, Any] = DEFAULT_LOSS_CFG

    def init_criterion(self) -> Phase2DetectionLoss:
        return Phase2DetectionLoss(self, **self._phase2_loss_cfg)
