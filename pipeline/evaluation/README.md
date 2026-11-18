# Evaluation

The supported student protocol is documented in [EVALUATION.md](../EVALUATION.md).
YOLO student/baseline scoring uses fixed confidence 0.0, NMS IoU 0.7, max_det 300,
COCO AP at maxDets 300, and AR at maxDets 100. Required scoring failures stop the run.

`yolo/eval_coco.py --help` describes checkpoint and cached-prediction scoring.
`sam3/run_eval.py --help` describes the upstream teacher evaluation wrapper.
Teacher and student metrics must only be compared after checking identical dataset hashes,
category mappings and effective COCO parameters; sharing a dataset name does not prove parity.
The six-cell cached regression in `results/` covers original YOLO11m ALD student results.
