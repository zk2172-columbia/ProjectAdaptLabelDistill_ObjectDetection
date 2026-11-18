"""coco_eval.py — the one COCOeval implementation every arm in this repo is scored with.

Reports six numbers, and only these six: **mAP50, mAP50-95, APs, APm, APl, AR**, from COCOeval on
one fixed held-out split. See pipeline/EVALUATION.md for the standard this implements; this
module is the standard's executable form. Nothing else in the repo should construct a COCOeval.

BACKEND: faster-coco-eval, NOT pycocotools
------------------------------------------
`faster_coco_eval.COCOeval_faster` is a C++ reimplementation of pycocotools' `COCOeval` with the
same API, the same 12-entry `stats` vector in the same order, and the same `eval["precision"]`
layout. It replaced pycocotools here for speed: 11x on this repo's eval set (127.3s -> 11.5s
across 63 stored prediction files), which matters because SAM3 emits ~200 queries per prompt per
image and the teacher is now scored through this same module.

The swap changes no number this repo has reported. Re-scoring all 63 stored `predictions.json`
files under both backends against the same GT: worst difference **0.00e+00** across all 12 stats
AND across every per-class AP50-95, over detection counts from 17k to 100k. That is the same bar
this module cleared when it replaced the four one-off scripts (EVALUATION.md §6).

Exact agreement is not, however, guaranteed in general — the two accumulate in different orders,
so results can differ in the last bit. A synthetic fixture in tests/ lands on 0.40069306930693066
under one backend and 0.4006930693069306 under the other: 5.6e-17, one ULP. The 63 real files
agreeing exactly is luck rather than a property. So the equivalence test asserts a 1e-12 tolerance
— four orders of magnitude tighter than any real divergence, since a changed matching rule or area
band moves metrics in the third decimal (EVALUATION.md §1) — and a future faster-coco-eval upgrade
that changes behaviour fails loudly rather than silently shifting every number in the repo.

One difference is real and is a FIX, not a regression. pycocotools' `_summarize()` hardcodes
`maxDets=100` as a default and `_summarizeDets` does not pass one through for `stats[0]`, so its
headline mAP50-95 ignores `params.maxDets[2]` entirely. faster-coco-eval indexes `maxDets[2]`
correctly. This module DOES set maxDets as of 2026-08-27 — `COCO_MAX_DETS = [1, 100, 300]`, see
that constant — which is only possible because of this difference.

What pycocotools' `stats[0]` then returns depends on the list, and BOTH cases are traps:
`[1, 10, 300]` leaves 100 absent, the index lookup fails, and it returns **-1** (loudly wrong);
`[1, 100, 300]`, the value in use here, finds 100 at index 1 and returns a plausible-looking
**AP@100** (silently wrong). The backend-equivalence test therefore recomputes pycocotools'
headline off `eval["precision"]` at maxDet index 2, unconditionally — never off `stats[0]`, and
never behind an `== -1` guard.

pycocotools is still a dependency, used only by the equivalence test.

DEPENDENCY NOTE
---------------
`faster-coco-eval` is NOT a hard dependency of ultralytics — it is declared as an optional extra
(`faster-coco-eval>=1.6.7; extra == "extra"`), so an ultralytics reinstall without that extra
removes it. It needs pinning wherever this pipeline's dependencies get recorded. Verified against
faster-coco-eval 1.7.2 / pycocotools 2.0.11 / ultralytics 8.4.21.

WHY NOT ULTRALYTICS' mAP
-----------------------
`model.val()` still runs here — it is how detections are GENERATED — but its `DetMetrics` numbers
are deliberately NOT recorded. The two disagree, by an amount that depends on the model, so a
table mixing them is not a comparison. `ultralytics/utils/metrics.py::compute_ap` appends a
sentinel at (recall=1.0, precision=0.0) and integrates the linear interpolation across the recall
range the model never reaches; COCOeval sets precision to exactly 0 there. Per class, ultralytics
credits the triangle under that ramp:

    gap  ~=  0.5 * (1 - R) * P          R = max achieved recall, P = precision there

Measured on this repo's held-out split, that formula predicts the observed gap to ~1 part in 40:
2-3 points for a normal-recall model, 26 points for a low-recall one (EVALUATION.md §1). Because
the error grows as recall falls, ultralytics systematically flatters exactly the regime
semi-supervised arms land in when pseudo-labels are sparse — the worst measured cell reports
AP 0.468 for a class it finds 6% of. COCOeval is also what the literature we compare against
reports, and the only one of the two that computes size bands at all.

So `score_checkpoint` returns COCOeval numbers alone. `n_detections` is reported instead, because
it is what distinguishes a genuinely low-recall arm from a broken one.

WHY val()'s OWN predictions.json
--------------------------------
Detections come from `val(save_json=True)`, re-scored — never from a second `model.predict()`
pass. An earlier version of the phase-2 scoring script did run a separate predict pass and
measured ~2,000 different detections dataset-wide and a ~3 point mAP50-95 shift against what
val() itself reported. Reusing val()'s own output makes the size bands describe exactly the
detection set the headline numbers came from, by construction rather than by argument.

FEEDING COCOeval THE COMPLETE DETECTION SET
-------------------------------------------
COCOeval builds its PR curve by sorting detections by score descending and accumulating. Low-score
detections are what extend the curve to the right; dropping them TRUNCATES achieved recall, and
under the COCO convention everything past that point scores zero. Thresholding can therefore only
lower AP — measured, a 0.5 cutoff costs a low-recall model 55% of its mAP (EVALUATION.md §2).

Reusing val()'s output is only correct if val() is in fact emitting everything, so the facts this
module depends on, verified against ultralytics 8.4.21:

  * `conf` is filled in as 0.001 ONLY when it is None (engine/validator.py:123) — a falsy check
    would have turned this module's 0.0 back into 0.001, and it is not one. conf is therefore
    PINNED in EvalSpec at `CONF_FLOOR`, 0.0 as of 2026-08-27, and passed explicitly on every call
    — never inherited from a training config, where a stray `conf: 0.25` would silently truncate.
  * NMS runs multi_label=True (detect/val.py:119), so one box may emit several class rows. All are
    kept; that is correct for COCO.
  * `max_det` (default 300) is applied inside NMS (detect/val.py:122) and is a real per-image
    ceiling; `score_checkpoint` reports images that saturate it, since that is otherwise
    invisible. COCOeval's AP cap is aligned with it as of 2026-08-27 (COCO_MAX_DETS[2] = 300 per
    (image, category), previously 100), so for AP max_det is now the binding constraint rather
    than being masked by a tighter scoring cap; AR deliberately stays at 100 (see THE AP/AR CAP
    SPLIT). What max_det drops is the lowest-scoring tail where precision is ~0. Measured on this
    repo's worst-saturating cell (125 of 1,045 images at the cap), raising
    max_det 300 -> 1000 fed COCOeval 41,496 more detections and moved every reported metric by
    <= 2.4e-4. 300 is kept because it is ultralytics' default and therefore what every historical
    number in this repo was measured under.
  * `scale_preds()` runs BEFORE `pred_to_json()` (detect/val.py:200-204), so boxes are already
    xywh top-left in ORIGINAL image pixels. No coordinate conversion here, and none should be
    added.
  * `category_id = class_index + 1` via `class_map` (detect/val.py:85-91) — unless ultralytics
    decides the dataset `is_coco`/`is_lvis` from its path STRING, which swaps in an 80->91 remap
    and rotates every class. `_build_detections` asserts the category ids against the GT's, so
    that shows up as an error rather than a collapsed mAP.
  * predictions.json is written only if there is at least one detection (engine/validator.py:262),
    so a model that finds nothing produces no file at all — reported as such, not as a missing
    file.

NMS and max_det are part of the DETECTOR, not scoring choices, and stay on. The `tau` in the STAC
pipeline is unrelated to any of this: it filters pseudo-labels used for TRAINING and never touches
evaluation.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml
from faster_coco_eval import COCO, COCOeval_faster

# The reported metrics, in report order, and where each lives in COCOeval.stats.
# IoU is averaged over 0.50:0.05:0.95 except mAP50. See EVALUATION.md §4.
#
# THE AP/AR CAP SPLIT (2026-08-27). Every AP metric is at maxDets=300; every AR metric is at
# maxDets=100. That is deliberate, not an oversight: 300 matches the detector's max_det so AP is
# not truncated below what the detector emits, while AR stays at COCO's canonical AR@100, which is
# what "AR" means in the literature and what SAM3's own evaluator reports for the teacher.
#
# COCO_MAX_DETS = [1, 100, 300] is what makes this a SINGLE pass. COCOeval's _summarizeDets reads
# maxDets[2] for stats[0..5] (all AP) and maxDets[0]/[1]/[2] for stats[6]/[7]/[8] (AR). So with
# 100 sitting at index 1, stats[7] IS AR@100 while every AP entry is at 300 -- no second
# accumulate(). Verified against a [1, 10, 100] reference run: AR/ARs/ARm/ARl agree to the
# printed digit.
#
# AR (added 2026-08-23) is COCO's Average Recall averaged over IoU 0.50:0.05:0.95 and over all
# areas -- stats[7], the companion to mAP50-95's stats[0]. It answers the question AP alone
# cannot: how much of the ground truth the arm finds at all, independent of how well it ranks what
# it finds. Two arms with the same AP and different AR are different models, and on a pseudo-label
# pipeline that difference is usually the whole story -- a teacher that misses objects teaches the
# student to miss them, and that shows up in recall before it shows up in AP.
#
# ARs/ARm/ARl are recorded for symmetry with APs/APm/APl but are not in REPORTED, so the headline
# table stays readable; read them from the payload when a size band is in question. They are NOT
# in STAT_INDEX: COCOeval only ever computes a size-banded AR at maxDets[2] (stats[9..11] = @300),
# so holding them at 100 means reading eval["recall"] at maxDet index 1 directly -- see
# _summarize. Like AP, a size band with no GT objects returns -1, not 0.
STAT_INDEX = {
    "mAP50": 1, "mAP50-95": 0, "APs": 3, "APm": 4, "APl": 5,
    "AR": 7,
}
REPORTED = ("mAP50", "mAP50-95", "APs", "APm", "APl", "AR")
# Recorded in the payload alongside the reported set, not in it.
ALSO_RECORDED = ("ARs", "ARm", "ARl")

# The detector floor. Not a tunable: see FEEDING COCOeval THE COMPLETE DETECTION SET above.
#
# CHANGED 2026-08-27 from 0.001 to 0.0. 0.001 was never a principled floor -- it is just the value
# ultralytics fills in when conf is None (engine/validator.py:123), inherited rather than chosen.
# The argument in FEEDING COCOeval THE COMPLETE DETECTION SET has no stopping point above zero:
# every detection dropped can only truncate achieved recall, and COCO scores everything past the
# truncation as zero. So the floor is now the actual floor.
#
# ULTRALYTICS ACCEPTS IT, verified against 8.4.21: validator.py:123 substitutes its 0.001 only
# `if self.args.conf is None`, not on a falsy check, so 0.0 survives; NMS filters `> conf_thres`
# (utils/nms.py:67) and asserts `0 <= conf_thres <= 1`, so 0.0 admits every box with a nonzero
# score and none with a zero one.
#
# COST, measured on the trash eval split (all-label-yolo, 1,045 images): 313,500 detections at 0.0
# against 38,254 at 0.001, an 8.2x larger detection set. That is more NMS work per image, more
# json on disk and a slower accumulate(); it does NOT change what max_det=300 admits per image,
# which still binds first on saturating images.
#
# EFFECT on the numbers, same cell: mAP50 +0.003, mAP50-95 +0.002, APs +0.001, APm +0.002,
# APl +0.004, AR +0.017. AP moves little because the added tail is low-precision; AR moves most,
# because those detections do reach further down the ground truth.
#
# CONSEQUENCE: this is NOT a re-scoring change. Detections below 0.001 are absent from every
# predictions.json written before this date, so score_predictions() cannot recover them -- a
# pre-existing predictions.json re-scored today is still a conf=0.001 measurement no matter what
# this constant says. Bringing an old run onto this standard costs a fresh score_checkpoint(),
# i.e. GPU. Compare `min_score` in a payload against this value to tell the two apart.
CONF_FLOOR = 0.0

# COCOeval's own per-(image, category) detection cap, applied inside accumulate(). CHANGED
# 2026-08-27 from COCOeval's default [1, 10, 100] to [1, 100, 300], so the scoring cap matches the
# detector's max_det=300 rather than silently truncating below it. (100 sits at index 1 rather
# than COCOeval's 10 so that AR can be read at 100 in the same pass -- see THE AP/AR CAP SPLIT.)
#
# Every metric this module reports reads maxDets index 2 -- mAP50-95 (stats[0]), mAP50, APs/APm/APl
# and per-class AP, not just AR -- so this moves ALL of them, and by more than a rounding error.
# Measured on the trash eval split at conf 0.0: mAP50 +0.016, mAP50-95 +0.007, APs +0.010,
# APl +0.001, AR +0.026..0.029. The gain concentrates in small objects, because what the 100-cap
# dropped was the low-scoring tail and on this data that tail is mostly small boxes.
#
# CONSEQUENCE: numbers produced before this date are NOT comparable to numbers produced after it.
# Any table that mixes the two reads a ~+0.016 mAP50 scoring artifact as a real gain. Re-score
# rather than mix -- score_predictions() does the maxDets part from a stored predictions.json with
# no GPU. It does NOT do the CONF_FLOOR part, which landed the same day and needs fresh inference;
# see that constant.
#
# TRAP: pycocotools cannot express this. Its `_summarizeDets` does not pass a maxDets through for
# stats[0], and `_summarize`'s default of 100 is then absent from the list, so headline mAP50-95
# comes back -1 while the other metrics compute normally. faster-coco-eval (the backend here)
# indexes maxDets[2] correctly. This is why tests/test_coco_eval.py's backend-equivalence test
# reads the headline through an explicit _summarize() call on the pycocotools side.
COCO_MAX_DETS = [1, 100, 300]


@dataclass(frozen=True)
class EvalSpec:
    """Everything that determines a score, other than the checkpoint itself.

    `conf` is pinned at the detector floor (`CONF_FLOOR`, 0.0) and is NOT read from training
    configs — see the module docstring. It is a field rather than a constant only so a test can
    demonstrate that raising it lowers AP; production callers leave it alone.

    imgsz/batch/device SHOULD be passed as whatever the arm trained with, so scoring never
    silently happens at a different resolution than training used.
    """

    gt_json: Path
    data_yaml: Path
    split: str = "test"
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    conf: float = CONF_FLOOR
    iou: float = 0.7        # NMS IoU — part of the detector
    max_det: int = 300      # per-image cap, applied inside NMS

    @classmethod
    def from_config(cls, block: dict, *, base_dir: Path | None = None, **defaults) -> "EvalSpec":
        """Build a spec from an `evaluation:` config block.

        Reads gt_json / dataset_yaml / split / imgsz / batch / device only. `conf`, `iou` and
        `max_det` are deliberately NOT read from config: they define the detection set COCOeval
        sees, and a config that quietly narrowed it would produce a number that looks fine and
        compares against nothing. Pass **defaults for imgsz/batch/device fallbacks (what the arm
        trained with).
        """
        def _p(v):
            if v is None:
                return None
            p = Path(v)
            return p if p.is_absolute() or base_dir is None else (base_dir / p).resolve()

        gt_json = _p(block.get("gt_json"))
        data_yaml = _p(block.get("dataset_yaml") or block.get("data_yaml"))
        if gt_json is None:
            raise SystemExit(
                "evaluation.gt_json is unset — there is no ground truth to score against.\n"
                "  Point it at ground_truths/<dataset>/eval/annotations.coco.json "
                "(built by data/prepare_ground_truth.py)."
            )
        if data_yaml is None:
            raise SystemExit("evaluation.dataset_yaml is unset — there is no dataset to run val() on.")

        return cls(
            gt_json=gt_json,
            data_yaml=data_yaml,
            split=block.get("split", "test"),
            imgsz=block.get("imgsz", defaults.get("imgsz")) or 640,
            batch=block.get("batch", defaults.get("batch")) or 16,
            device=str(block.get("device", defaults.get("device")) or "0"),
        )

    def preflight(self) -> None:
        """Raise unless both inputs exist AND agree on the class list. Call BEFORE training starts.

        A typo'd path is worth one line now rather than hours of GPU followed by a failed eval on
        every arm — and a class-order disagreement is worth far more than that, because it does not
        fail at all: it scores every prediction against the wrong class and reports a quietly
        collapsed mAP. See `_check_class_agreement` for what is compared and why.
        """
        for label, p in (("evaluation.gt_json", self.gt_json), ("evaluation.dataset_yaml", self.data_yaml)):
            if not Path(p).is_file():
                raise SystemExit(
                    f"{label} not found: {p}\n"
                    "  Fix the path, or set evaluation.enabled: false / pass --skip-test-eval."
                )
        _check_class_agreement(Path(self.gt_json), Path(self.data_yaml))


# ── Class-order agreement ─────────────────────────────────────────────────────
#
# Every class-order guard in this pipeline fires when an artifact is BUILT:
# prepare_ground_truth.py and data/coco_to_yolo.py remap a source export into the class list's
# order and fail loudly on a name with no counterpart. Nothing re-checked it at CONSUME time, and
# the two files that meet here are chosen by hand in two different config keys (NEW_DATASET.md §4).
# The record of what a given export was built with is the frozen ground_truths/<ds>/classes.yaml,
# which the preflight below reads back.
#
# `_build_detections` already rejects a predicted category_id that is not a GT category id, but
# that is an ids-only check: a class list rotated into the same id range passes it clean. Names are
# what actually pin the mapping down.


def _names_from_gt(gt_json: Path) -> list[str] | None:
    """GT category names ordered by category_id. COCO id = classes.yaml id + 1, so this is
    classes.yaml order."""
    try:
        with open(gt_json) as f:
            cats = json.load(f).get("categories")
    except (json.JSONDecodeError, OSError):
        return None            # a malformed GT is _build_detections' error to report, not ours
    if not cats:
        return None
    return [str(c["name"]) for c in sorted(cats, key=lambda c: c["id"])]


def _names_from_data_yaml(data_yaml: Path) -> list[str] | None:
    """`names:` from a YOLO dataset yaml, index-ordered. Accepts both the list and the
    {id: name} mapping form ultralytics allows."""
    try:
        with open(data_yaml) as f:
            names = (yaml.safe_load(f) or {}).get("names")
    except (yaml.YAMLError, OSError):
        return None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def _frozen_names(gt_json: Path) -> tuple[Path, list[str]] | None:
    """The classes.yaml prepare_ground_truth.py froze beside this export, if it is there.

    gt_json is ground_truths/<ds>/<split>/annotations.coco.json, so the frozen copy is two levels
    up. Absent for a GT that predates the exporter or was built by hand — not an error, just one
    fewer thing to check against.
    """
    path = gt_json.parent.parent / "classes.yaml"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            classes = (yaml.safe_load(f) or {}).get("classes")
    except (yaml.YAMLError, OSError):
        return None
    if not classes:
        return None
    return path, [str(c["name"]) for c in sorted(classes, key=lambda c: c["id"])]


def _check_class_agreement(gt_json: Path, data_yaml: Path) -> None:
    """Raise unless gt_json, data_yaml and the frozen classes.yaml name the same classes in the
    same order.

    Matched case-insensitively, the same way prepare_ground_truth.py matches when it builds the
    export. A source that is missing the list entirely is skipped rather than failed: this is a
    guard against disagreement, not a demand that every hand-built GT carry one.
    """
    sources: list[tuple[str, list[str]]] = []
    if (gt := _names_from_gt(gt_json)) is not None:
        sources.append((f"gt_json          {gt_json}", gt))
    if (data := _names_from_data_yaml(data_yaml)) is not None:
        sources.append((f"dataset_yaml     {data_yaml}", data))
    if (frozen := _frozen_names(gt_json)) is not None:
        sources.append((f"frozen classes   {frozen[0]}", frozen[1]))

    if len(sources) < 2:
        return
    ref_label, ref = sources[0]
    disagreeing = [(lbl, n) for lbl, n in sources[1:] if [x.lower() for x in n] != [x.lower() for x in ref]]
    if not disagreeing:
        return

    lines = [f"  {lbl}\n    {names}" for lbl, names in [(ref_label, ref), *disagreeing]]
    raise SystemExit(
        "class lists disagree — evaluation would score predictions against the wrong classes.\n"
        + "\n".join(lines)
        + "\n\n  These must name the same classes in the same order: the class id is a position, "
        "not a label,\n  so a rotated list still localizes boxes correctly and only shows up as a "
        "collapsed mAP.\n"
        "  Most likely classes_file names a different dataset than this export was "
        "built,\n  or gt_json and dataset_yaml were pointed at two different datasets. Re-export "
        "with\n  data/prepare_ground_truth.py, or fix the paths. See NEW_DATASET.md §2."
    )


# ── Provenance ────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def _versions() -> dict:
    out = {}
    # pycocotools is recorded even though it no longer computes anything: the equivalence this
    # module claims (see BACKEND above) is against a specific version of it, and tests/ asserts
    # that equivalence still holds. A stored number should say which pair it was verified under.
    for mod in ("ultralytics", "faster_coco_eval", "pycocotools"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            # pycocotools ships no __version__ (its __init__.py sets only __author__), so the
            # attribute lookup above has always recorded it as null — every historical
            # coco_metrics.json in this repo says "pycocotools": null for that reason, not
            # because it was absent. Fall back to the installed distribution's metadata.
            try:
                out[mod] = importlib.metadata.version(mod.replace("_", "-"))
            except Exception:  # noqa: BLE001 — provenance is best-effort
                out[mod] = None
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────


def _build_detections(preds: list[dict], coco_gt: COCO, *, max_det: int | None = None) -> list[dict]:
    """Map ultralytics' predictions.json onto the GT's image ids, validating as it goes.

    Ultralytics writes its own image_id (the filename stem), so the join is on `file_name` —
    the one field both sides agree on. Both failure modes below are silent if unchecked: an
    unmapped file_name would KeyError mid-comprehension with no clue which file, and a rotated
    category id would simply produce a low mAP that looks like a bad model.
    """
    filename_to_id = {img["file_name"]: img_id for img_id, img in coco_gt.imgs.items()}

    unknown = {p["file_name"] for p in preds if p["file_name"] not in filename_to_id}
    if unknown:
        raise SystemExit(
            f"{len(unknown)} predicted file_name(s) are not in the ground truth, e.g. "
            f"{sorted(unknown)[:3]}.\n"
            "  The detections and the GT describe different image sets — check that the dataset "
            "yaml's split points at the same images as gt_json."
        )

    gt_cats = set(coco_gt.getCatIds())
    pred_cats = {p["category_id"] for p in preds}
    if not pred_cats <= gt_cats:
        raise SystemExit(
            f"detection category_id(s) {sorted(pred_cats - gt_cats)} are not GT categories "
            f"{sorted(gt_cats)}.\n"
            "  Ultralytics emits class_index + 1; if that no longer lines up with the GT's ids, "
            "every class is being scored against the wrong one. See coco_eval.py's docstring."
        )

    if max_det:
        per_image = Counter(p["file_name"] for p in preds)
        saturated = [f for f, n in per_image.items() if n >= max_det]
        if saturated:
            # Informational, not a defect — but read it differently since 2026-08-27. COCOeval's
            # own cap used to be 100 per (image, category) and absorbed most of what max_det
            # dropped; measured on the worst-saturating cell then (125 of 1,045 images at the cap,
            # conf 0.001), raising max_det 300 -> 1000 fed COCOeval 41,496 more detections and
            # moved every reported metric by <= 2.4e-4. COCO_MAX_DETS[2] is now 300, equal to
            # max_det, so that absorption is gone and max_det binds on AP directly. It still only
            # drops the lowest-scoring detections, which land at the tail of the PR curve where
            # precision is ~0, and saturation is now common rather than rare: at CONF_FLOOR = 0.0
            # a dense image saturates almost by definition.
            print(
                f"  note: {len(saturated)} image(s) hit max_det={max_det} "
                f"(e.g. {saturated[:3]}). COCOeval's cap is {COCO_MAX_DETS[2]} per "
                "(image, category), so this is the binding per-image limit on AP, not a cushioned "
                "one — expect saturation at conf 0.0 and compare only against runs at the same cap."
            )

    return [
        {
            "image_id": filename_to_id[p["file_name"]],
            "category_id": p["category_id"],
            "bbox": p["bbox"],
            "score": p["score"],
        }
        for p in preds
    ]


def _summarize(coco_gt: COCO, dets: list[dict], *, quiet: bool = False) -> dict:
    """Run COCOeval over `dets` and pull out the five reported metrics plus per-class AP."""
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        # print_function=print is REQUIRED, not cosmetic: faster-coco-eval defaults it to
        # `logger.info` on a logger left at WARNING, so summarize() would print nothing and the
        # non-quiet path would silently lose its summary table.
        ev = COCOeval_faster(
            coco_gt, coco_gt.loadRes(dets), iouType="bbox", print_function=print,
        )
        # Must be set BEFORE evaluate(): evaluateImg truncates each (image, category) to
        # max(maxDets), and accumulate() then indexes the list. See COCO_MAX_DETS.
        ev.params.maxDets = list(COCO_MAX_DETS)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats

    # Per-class AP50-95 read straight off the precision array rather than re-running evaluate()
    # once per category: identical result, no extra passes. Axes are [IoU, recall, cat, area,
    # maxDet]; area index 0 is 'all' and maxDet index 2 is COCO_MAX_DETS[2], matching stats[0].
    # Entries are -1 where a class has no GT, and those must be excluded rather than averaged in
    # as zero.
    precision = ev.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        per_class[coco_gt.cats[cat_id]["name"]] = float(np.mean(p)) if p.size else float("nan")

    # ARs/ARm/ARl at maxDets=100, matching AR (stats[7]) rather than the @300 AP metrics.
    # stats[9..11] cannot be used: _summarizeDets computes the size-banded AR only at maxDets[2],
    # which is 300. eval["recall"] has axes [IoU, cat, area, maxDet], so index 1 on the last axis
    # is COCO_MAX_DETS[1] == 100 — the same quantity _summarize(0, areaRng=..., maxDets=100)
    # returns. Area indices follow params.areaRngLbl: 0 'all', 1 'small', 2 'medium', 3 'large'.
    # Entries are -1 where a band has no GT and must be excluded, not averaged in as zero.
    recall = ev.eval["recall"]
    mind_100 = COCO_MAX_DETS.index(100)
    ar_by_area = {}
    for name, aind in zip(ALSO_RECORDED, (1, 2, 3)):
        r = recall[:, :, aind, mind_100]
        r = r[r > -1]
        # -1.0, not nan: this is what stats[9..11] returned for an empty band before the split,
        # so downstream readers of ar_by_area see no change in convention.
        ar_by_area[name] = float(np.mean(r)) if r.size else -1.0

    return {
        "metrics": {name: float(s[STAT_INDEX[name]]) for name in REPORTED},
        "mAP75": float(s[2]),
        "ar_by_area": ar_by_area,
        "per_class_mAP50-95": per_class,
    }


def score_predictions(
    gt_json: str | Path,
    predictions_json: str | Path,
    *,
    max_det: int | None = None,
    quiet: bool = False,
) -> dict:
    """Re-score an existing ultralytics predictions.json. No inference, no GPU, no model.

    This is the cheap path: it re-derives the five metrics from detections already on disk, which
    is how a past run is re-scored under the current standard without spending GPU on it.
    """
    gt_json, predictions_json = Path(gt_json), Path(predictions_json)
    coco_gt = COCO(str(gt_json)) if not quiet else _quiet_coco(gt_json)

    preds = json.loads(predictions_json.read_text())
    dets = _build_detections(preds, coco_gt, max_det=max_det)
    if not dets:
        raise SystemExit(f"{predictions_json} contains no detections — nothing to score.")

    payload = _summarize(coco_gt, dets, quiet=quiet)
    payload.update({
        "n_detections": len(dets),
        "n_images": len(coco_gt.imgs),
        "n_gt_boxes": len(coco_gt.anns),
        "min_score": min(p["score"] for p in preds),
        "provenance": {
            "predictions_json": str(predictions_json),
            "gt_json": str(gt_json),
            "gt_sha256": _sha256(gt_json),
            "git_rev": _git_rev(),
            **_versions(),
        },
    })
    return payload


def _quiet_coco(path: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def score_checkpoint(
    weights: str | Path,
    spec: EvalSpec,
    out_dir: str | Path,
    *,
    name: str = "coco_eval",
    quiet: bool = False,
) -> dict:
    """Score one checkpoint on the held-out split. THE entry point — everything else wraps this.

    Runs val(save_json=True) into out_dir/<name>/ purely to generate detections, then re-scores
    that run's own predictions.json with COCOeval. Returns the payload and does not write it;
    callers decide where it lands (see write_metrics).
    """
    from ultralytics import YOLO

    weights, out_dir = Path(weights), Path(out_dir)
    if not weights.is_file():
        raise SystemExit(f"checkpoint not found: {weights}")
    spec.preflight()

    coco_gt = _quiet_coco(spec.gt_json) if quiet else COCO(str(spec.gt_json))

    preds_path = out_dir / name / "predictions.json"
    if preds_path.exists():
        raise FileExistsError(f"Use a fresh evaluation directory; refusing stale predictions: {preds_path}")
    YOLO(str(weights)).val(
        data=str(spec.data_yaml),
        split=spec.split,
        imgsz=spec.imgsz,
        batch=spec.batch,
        device=spec.device,
        # The three that decide what COCOeval gets to see. Passed explicitly, always.
        conf=spec.conf,
        iou=spec.iou,
        max_det=spec.max_det,
        save_json=True,     # the entire point: this writes the predictions.json re-scored below
        plots=False,
        verbose=not quiet,
        project=str(out_dir),
        name=name,
        exist_ok=True,
    )

    preds_path = out_dir / name / "predictions.json"
    if not preds_path.is_file():
        # validator.py:262 only writes the file when jdict is non-empty.
        raise SystemExit(
            f"val() produced no detections for {weights} — no predictions.json was written.\n"
            f"  (Looked in {preds_path}.) A model that detects nothing at conf={spec.conf} is "
            "broken, not merely low-recall."
        )

    preds = json.loads(preds_path.read_text())
    dets = _build_detections(preds, coco_gt, max_det=spec.max_det)

    payload = _summarize(coco_gt, dets, quiet=quiet)
    payload.update({
        "weights": str(weights),
        "n_detections": len(dets),
        "n_images": len(coco_gt.imgs),
        "n_gt_boxes": len(coco_gt.anns),
        # The floor these detections actually reach. Should sit at spec.conf; a much higher value
        # means something upstream filtered them and the score is on a truncated set.
        "min_score": min(p["score"] for p in preds),
        "provenance": {
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "gt_json": str(spec.gt_json),
            "gt_sha256": _sha256(spec.gt_json),
            "data_yaml": str(spec.data_yaml),
            "split": spec.split,
            "predictions_json": str(preds_path),
            "val_kwargs": {k: v for k, v in asdict(spec).items() if k not in ("gt_json", "data_yaml")},
            "git_rev": _git_rev(),
            **_versions(),
        },
    })
    return payload


# ── Reporting ─────────────────────────────────────────────────────────────────


def write_metrics(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def wandb_summary(payload: dict, prefix: str = "eval") -> dict:
    """Flatten a payload into W&B summary keys: eval/mAP50, ..., eval/mAP50-95/<class>."""
    out = {f"{prefix}/{k}": v for k, v in payload["metrics"].items()}
    out[f"{prefix}/n_detections"] = payload["n_detections"]
    out.update({f"{prefix}/mAP50-95/{c}": v for c, v in payload["per_class_mAP50-95"].items()})
    return out


def format_table(rows: list[tuple[str, dict]], label_header: str = "run") -> str:
    """Render scored rows as the five-column table. `rows` is [(label, payload), ...]."""
    width = max([len(label_header)] + [len(r[0]) for r in rows]) if rows else len(label_header)
    header = f"{label_header:<{width}}  " + "  ".join(f"{m:>9}" for m in REPORTED) + f"  {'n_det':>8}"
    lines = [header, "-" * len(header)]
    for label, payload in rows:
        m = payload["metrics"]
        lines.append(
            f"{label:<{width}}  "
            + "  ".join(f"{m[k]:>9.4f}" for k in REPORTED)
            + f"  {payload['n_detections']:>8,}"
        )
    return "\n".join(lines)


def print_report(label: str, payload: dict) -> None:
    """One checkpoint's five numbers, plus the context needed to read them."""
    m = payload["metrics"]
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {label} — {payload['n_images']:,} images, {payload['n_gt_boxes']:,} GT boxes")
    print(bar)
    print(f"  mAP50       {m['mAP50']:.4f}")
    print(f"  mAP50-95    {m['mAP50-95']:.4f}")
    print(f"  APs         {m['APs']:.4f}   (GT area < 32^2)")
    print(f"  APm         {m['APm']:.4f}   (32^2 - 96^2)")
    print(f"  APl         {m['APl']:.4f}   (area > 96^2)")
    print(f"  {payload['n_detections']:,} detections, min score {payload.get('min_score', float('nan')):.5f}")
    print(bar)
