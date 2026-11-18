import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"pipeline/evaluation/yolo"))
import coco_eval
from pycocotools.coco import COCO as ReferenceCOCO
from pycocotools.cocoeval import COCOeval

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        gt={"info":{},"images":[{"id":1,"file_name":"image.jpg","width":2000,"height":100}],
            "categories":[{"id":1,"name":"object"}],"annotations":[]}
        self.preds=[]
        for i in range(120):
            bbox=[i*15,10,10,10]
            gt["annotations"].append({"id":i+1,"image_id":1,"category_id":1,"bbox":bbox,"area":100,"iscrowd":0})
            self.preds.append({"file_name":"image.jpg","image_id":1,"category_id":1,"bbox":bbox,"score":1-i*.001})
        self.gt=self.root/"gt.json";self.gt.write_text(json.dumps(gt))
        self.pred=self.root/"pred.json";self.pred.write_text(json.dumps(self.preds))
    def tearDown(self):self.tmp.cleanup()
    def test_backend_equivalence_and_ap_cap(self):
        result=coco_eval.score_predictions(self.gt,self.pred,quiet=True)
        with contextlib.redirect_stdout(io.StringIO()):
            gt=ReferenceCOCO(str(self.gt));dt=gt.loadRes(self.preds)
            ev=COCOeval(gt,dt,"bbox");ev.params.maxDets=[1,100,300]
            ev.evaluate();ev.accumulate();ev.summarize()
        precision=ev.eval["precision"][:,:,:,0,2]
        ref=float(precision[precision>-1].mean())
        self.assertAlmostEqual(result["metrics"]["mAP50-95"],ref,places=12)
        self.assertGreater(result["metrics"]["mAP50-95"],.999)
        self.assertLess(ev.stats[0],.9)
        self.assertAlmostEqual(result["metrics"]["AR"],100/120,places=12)
    def test_bad_category_rejected(self):
        self.preds[0]["category_id"]=99;self.pred.write_text(json.dumps(self.preds))
        with self.assertRaises(SystemExit):coco_eval.score_predictions(self.gt,self.pred,quiet=True)
    def test_wrong_image_rejected(self):
        self.preds[0]["file_name"]="other.jpg";self.pred.write_text(json.dumps(self.preds))
        with self.assertRaises(SystemExit):coco_eval.score_predictions(self.gt,self.pred,quiet=True)
    def test_empty_detection_explicit_failure(self):
        self.pred.write_text("[]")
        with self.assertRaises(SystemExit):coco_eval.score_predictions(self.gt,self.pred,quiet=True)
