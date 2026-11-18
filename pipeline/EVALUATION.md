# Fixed evaluation protocol

- Fixed held-out COCO annotations; preserve image IDs and class order.
- Infer at conf=0, detector NMS IoU=0.7 and max_det=300 per image.
- Use faster-coco-eval 1.7.2 with maxDets=[1,100,300].
- AP metrics read maxDet index 2 (300); AR and size-banded AR read index 1 (100).
- The pycocotools equivalence test reads precision at index 2 directly: its default stats[0]
  summary is AP@100, not the AP@300 measured here.
- An empty size band is reported as -1, not a zero-accuracy band.
- Empty detections fail explicitly. Unknown images/categories fail rather than being dropped.
- Checkpoint inference must use a fresh output directory; existing predictions are rejected.
- Training-selected validation and final held-out evaluation are different splits.

`tests/test_coco_eval.py` covers backend equivalence, the 100/300 cap distinction and invalid
input behavior. Cached historical prediction re-scoring cannot restore detections discarded
by an earlier inference threshold; it must not be presented as fresh inference.
