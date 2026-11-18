"""loss.py — STACDetectionLoss: v8DetectionLoss with STAC's lambda_u on the unsupervised half.

The paper's objective (STAC eq. 4):

    l = l_s(x_s, p*, t*)  +  lambda_u * l_u(A(x_u, s*), q*)

i.e. the supervised loss at full strength, plus the unsupervised loss on the pseudo-labeled
images scaled by one global constant. This class implements exactly that, and nothing else:

    Z            = target_scores.sum()            # ALWAYS the UNWEIGHTED sum
    w            = batch["w_image"] -> (bs, 1, 1) # 1.0 for GT samples, lambda_u for PL samples
    L_cls        = (BCE(pred_scores, target_scores) * w).sum() / Z
    L_box, L_dfl = BboxLoss(..., target_scores * w, Z, ...)

Three properties worth being explicit about.

**The normalizer is the unweighted sum.** Z does not see w, so a uniform weight scales the loss
by exactly that weight: doubling lambda_u exactly doubles the pseudo-label half's gradient
contribution, and a batch of GT alone reduces bit-for-bit to stock v8DetectionLoss. The
alternative (Z from the weighted scores) makes a batch-wide constant cancel between numerator
and denominator, which would turn lambda_u into a literal no-op on a homogeneous batch — and
with the balanced loader every mosaic sample IS homogeneous.

**The BCE target is never scaled — only the resulting loss tensor.** BCE's gradient w.r.t. the
logit is (sigmoid(logit) - target), so scaling the target would move the classifier's fixed
point from align_metric to w*align_metric and train the model to systematically UNDER-report
confidence on pseudo-labeled images. Weighting the loss preserves the fixed point and only
reduces gradient magnitude.

**The weight reaches background anchors too, and that is the point.** Because w is per SAMPLE
rather than per box, it multiplies the BCE of every anchor on a pseudo-labeled image, including
the negatives that belong to no box. That is what "scale the unsupervised loss by lambda_u"
means, and it is the half that matters most here: at tau = 0.9 the teacher misses a great deal,
and every miss is an anchor asserting "background" that the student would otherwise be trained
on at full strength. A per-box weight structurally cannot reach those anchors; this one does.

Copied from student_training/loss.py with its configuration flags (loc_weight_src,
cls_weight_src), the per-box weight column and its gather/fg_mask handling, and the arm registry
they served all removed — STAC needs exactly one of the configurations they spanned.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class STACDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss scaled per sample by `batch["w_image"]`, under an unweighted normalizer.

    Attributes:
        last_w_eff (torch.Tensor): detached scalar diagnostic — the batch's realized effective
            weight, (target_scores * w).sum() / Z. 1.0 for an all-GT batch, lambda_u for an
            all-PL one, and in between for a mixed batch in proportion to how much foreground
            score mass the assigner put on each half. Written per epoch to w_eff.csv by
            STACTrainer, because an unweighted normalizer makes the reported box/cls/dfl values
            scale with each batch's PL fraction: this column is what separates "the model
            improved" from "that epoch was PL-heavy". Compare against baselines on val mAP, not
            on loss.

    If `w_image` is absent from the batch (validation builds the dataset without it), the
    weighting is skipped and this is stock v8DetectionLoss.
    """

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple:
        """Compute box/cls/dfl with the per-sample STAC weight applied."""
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

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
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

        # (bs, 1, 1) so it broadcasts against (bs, n_anchors, nc). Explicit device= as well as
        # dtype: preprocess_batch moves every tensor key today, but that is version-sensitive
        # and this costs nothing.
        w = None
        if "w_image" in batch:
            w = batch["w_image"].to(device=self.device, dtype=dtype)[:, None, None]

        target_scores_w = target_scores if w is None else target_scores * w
        # THE line that makes lambda_u absolute: Z comes from the UNWEIGHTED scores.
        target_scores_sum = max(target_scores.sum(), 1)

        # Diagnostic only — detached, never in the graph.
        self.last_w_eff = (target_scores_w.sum() / target_scores_sum).detach()

        # Cls (BCE): UNWEIGHTED target, weight applied to the resulting loss tensor.
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if w is not None:
            bce_loss = bce_loss * w
        loss[1] = bce_loss.sum() / target_scores_sum

        # Box (CIoU) + DFL: BboxLoss consumes target_scores_w purely as a per-anchor multiplier,
        # and both terms reduce by a plain sum, so passing the weighted copy IS per-anchor loss
        # multiplication — no BboxLoss subclass needed.
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


class STACDetectionModel(DetectionModel):
    """DetectionModel whose criterion is STACDetectionLoss.

    Carries no configuration at all — there is one loss, and the per-sample weight arrives with
    the batch rather than with the model. That is deliberate: ultralytics pickles the model
    object into best.pt, and train_student.py's checkpoint export makes it portable by swapping
    __class__ back to stock DetectionModel WITHOUT stripping instance attributes, so anything
    stored here would ride along into every "standard" checkpoint.
    """

    def init_criterion(self) -> STACDetectionLoss:
        return STACDetectionLoss(self)
