#!/usr/bin/env python3
"""rescore_dumps.py — put the SAM3 teacher runs on the 2026-08-27 evaluation standard.

The standard changed on the YOLO side that day (see yolo/coco_eval.py):

    COCO_MAX_DETS  [1, 10, 100] -> [1, 100, 300]   every AP metric now scores at 300
    CONF_FLOOR     0.001        -> 0.0             the detector floor is now the actual floor

For the YOLO students both changes need a fresh val() pass, because a predictions.json written at
conf 0.001 simply does not contain the detections below it (that is what a fresh val() pass is
for, and why it wants a GPU). **For the SAM3 teacher neither inference nor a GPU is needed**, and
the reason is worth stating precisely rather than assumed:

  * conf is ALREADY 0.0 in every dump. SAM3 does not threshold: the config's postprocessor is
    `PostProcessImage(max_dets_per_img=-1)` with `detection_threshold` left at its -1.0 default,
    so every one of the model's fixed 200 queries per (image, category) is written out. Verified
    directly on the dumps: exactly 200 detections per (image, category) with no exceptions, and
    scores reaching 0.0 exactly. There is no lower floor to drop to.
  * maxdets=300 never truncates either, for the same reason: COCOeval's maxDets is a per
    (image, category) cap, and SAM3 emits 200 < 300. So 300 admits the complete detection set,
    which is the whole point of the change; it is 100 (pycocotools' default, what the original
    runs scored under) that was cutting the emitted set in half.

So the only thing that moves is the scoring cap, and it moves over detections already on disk.

WHAT IT SCORES
--------------
Every SAM3 teacher run under eval_results/sam3-teacher/ that has both a config_resolved.yaml (for
its GT path) and a dumped coco_predictions_bbox.json. Restrict with --runs.

BACKEND
-------
faster-coco-eval, NOT pycocotools, and this is a deliberate departure from the asymmetry recorded
in ../README.md. That asymmetry existed because the scoring happened inside SAM3's own vendored
harness, which was not worth patching. This re-score happens outside it, so the teacher and the
students can be put through the identical scorer at identical settings.

pycocotools genuinely cannot express this standard, which is the second reason:
`_summarizeDets` does not pass a maxDets through for stats[0], and `_summarize` defaults to 100.
With maxDets=[1, 100, 300] it finds 100 at index 1 and returns a plausible-looking **AP@100** —
silently the wrong number. --cross-check therefore recomputes pycocotools' headline off
eval["precision"] at maxDet index 2, never off stats[0], and asserts the two backends agree.

OUTPUT
------
`coco_metrics_maxdet300.json` beside each run's existing `coco_metrics.json`, which is left in
place: the pre-standard numbers stay readable next to the new ones, which is the only way to tell
a real effect from a scoring artifact. Plus a combined `sam3_teacher_maxdet300.json` in --out-dir.

USAGE
-----
    PY=/home/zklab/miniconda3/envs/ald/bin/python

    $PY rescore_dumps.py                                  # every run with a dump
    $PY rescore_dumps.py --runs ft-sam3-visdrone          # one run
    $PY rescore_dumps.py --cross-check                    # also verify against pycocotools
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import yaml
from faster_coco_eval import COCO, COCOeval_faster

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent.parent / "eval_results" / "sam3-teacher"

# Identical to yolo/coco_eval.py's COCO_MAX_DETS. See that module for why 100 sits at index 1:
# it lets a single accumulate() serve AP@300 (stats[0..5]) and the canonical AR@100 (stats[7]).
COCO_MAX_DETS = [1, 100, 300]

STAT_INDEX = {"mAP50": 1, "mAP50-95": 0, "APs": 3, "APm": 4, "APl": 5, "AR": 7}
REPORTED = ("mAP50", "mAP50-95", "APs", "APm", "APl", "AR")


def discover(run_dir: Path) -> tuple[Path, Path] | None:
    """Return (gt_path, dump_path) for a run, or None if it is not a scoreable run."""
    cfg_path = run_dir / "config_resolved.yaml"
    dumps = sorted(run_dir.glob("dumps/*/*/coco_predictions_bbox.json"))
    if not cfg_path.is_file() or not dumps:
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    meter = cfg["trainer"]["meters"]["val"]["roboflow100"]["detection"]
    gt_path = Path(meter["pred_file_evaluators"][0]["gt_path"])
    assert len(dumps) == 1, f"{run_dir.name}: expected one dump, found {len(dumps)}"
    return gt_path, dumps[0]


def summarize(coco_gt: COCO, dets: list[dict], *, quiet: bool = True) -> dict:
    """COCOeval over `dets` at COCO_MAX_DETS, in the same shape yolo/coco_eval.py reports."""
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        ev = COCOeval_faster(
            coco_gt, coco_gt.loadRes(dets), iouType="bbox", print_function=print,
        )
        # Must be set BEFORE evaluate(): evaluateImg truncates each (image, category) to
        # max(maxDets), and accumulate() then indexes the list.
        ev.params.maxDets = list(COCO_MAX_DETS)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats

    # Axes are [IoU, recall, cat, area, maxDet]; area 0 is 'all', maxDet index 2 is 300.
    # -1 marks a class with no GT and must be excluded, not averaged in as zero.
    precision = ev.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        per_class[coco_gt.cats[cat_id]["name"]] = float(np.mean(p)) if p.size else float("nan")

    return {
        "metrics": {name: float(s[STAT_INDEX[name]]) for name in REPORTED},
        "mAP75": float(s[2]),
        "per_class_mAP50-95": per_class,
    }


def cross_check(gt_path: Path, dets: list[dict], reference: dict) -> dict:
    """Recompute the headline with pycocotools and assert the two backends agree.

    Reads eval["precision"] at maxDet index 2 directly. NEVER stats[0]: with 100 present at
    index 1 of maxDets, pycocotools' _summarize default finds it and returns AP@100.
    """
    from pycocotools.coco import COCO as PycocoCOCO
    from pycocotools.cocoeval import COCOeval as PycocoEval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = PycocoCOCO(str(gt_path))
        ev = PycocoEval(gt, gt.loadRes(list(dets)), iouType="bbox")
        ev.params.maxDets = list(COCO_MAX_DETS)
        ev.evaluate()
        ev.accumulate()

    p = ev.eval["precision"][:, :, :, 0, 2]
    ap = float(np.mean(p[p > -1]))
    delta = abs(ap - reference["metrics"]["mAP50-95"])
    assert delta < 1e-6, f"backend disagreement on mAP50-95: {delta:.2e}"
    return {"pycocotools_mAP50-95": ap, "delta": delta}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--runs", nargs="*", default=None, help="run directory names; default all")
    ap.add_argument("--out-dir", type=Path, default=None, help="where the combined json goes")
    ap.add_argument("--cross-check", action="store_true", help="also verify against pycocotools")
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.results_root.iterdir() if d.is_dir())
    if args.runs:
        wanted = set(args.runs)
        run_dirs = [d for d in run_dirs if d.name in wanted]
        missing = wanted - {d.name for d in run_dirs}
        assert not missing, f"no such run(s): {sorted(missing)}"

    combined = {}
    for run_dir in run_dirs:
        found = discover(run_dir)
        if found is None:
            print(f"[skip] {run_dir.name}: no config_resolved.yaml or no dump")
            continue
        gt_path, dump_path = found

        t0 = time.time()
        dets = json.loads(dump_path.read_text())
        coco_gt = COCO(str(gt_path))
        payload = summarize(coco_gt, dets)

        per_img = {}
        for d in dets:
            per_img[d["image_id"]] = per_img.get(d["image_id"], 0) + 1
        payload.update(
            run=run_dir.name,
            n_detections=len(dets),
            n_images=len(coco_gt.imgs),
            max_detections_per_image=max(per_img.values()),
            min_score=min(d["score"] for d in dets),
            provenance={
                "gt": str(gt_path),
                "predictions": str(dump_path),
                "config_resolved": str(run_dir / "config_resolved.yaml"),
                "max_dets": COCO_MAX_DETS,
                "conf": 0.0,
                "backend": f"faster_coco_eval",
                "scored_by": "evaluation/sam3/rescore_dumps.py",
            },
        )
        if args.cross_check:
            payload["provenance"]["cross_check"] = cross_check(gt_path, dets, payload)

        (run_dir / "coco_metrics_maxdet300.json").write_text(json.dumps(payload, indent=2) + "\n")
        combined[run_dir.name] = payload

        m = payload["metrics"]
        print(
            f"[ok] {run_dir.name:38s} AP {m['mAP50-95']:.4f}  AP50 {m['mAP50']:.4f}  "
            f"APs {m['APs']:.4f}  APm {m['APm']:.4f}  APl {m['APl']:.4f}  "
            f"AR {m['AR']:.4f}  ({time.time() - t0:.0f}s)"
        )

    out_dir = args.out_dir or args.results_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sam3_teacher_maxdet300.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'sam3_teacher_maxdet300.json'} ({len(combined)} runs)")


if __name__ == "__main__":
    main()
