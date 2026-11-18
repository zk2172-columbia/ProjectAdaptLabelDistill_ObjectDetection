"""Offline regression tests; no pretrained downloads, GPU jobs or network services."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['WANDB_MODE'] = 'disabled'
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir())/'ald-test-settings'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pipeline/student_training'))
sys.path.insert(0, str(ROOT/'pipeline'))
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from confidence import load_confidence
from data import Phase2YOLODataset
from loss import Phase2DetectionLoss
import train
import utils as student_utils
import pipeline_common
spec = importlib.util.spec_from_file_location('teacher_utils', ROOT/'pipeline/teacher_adaptation/utils.py')
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)
torch.set_num_threads(2)


def box(cls=0, confidence=.8):
    return dict(cx=.5, cy=.5, w=.4, h=.4, class_id=cls, w_sam3=confidence)


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weight_path = self.root/'weights.json'
        for split in ['gt','pl']:
            (self.root/split/'images').mkdir(parents=True)
            (self.root/split/'labels').mkdir()
            for i in range(4):
                Image.new('RGB',(64,64), color=(40+30*i,60,90)).save(self.root/split/'images'/f'{split}{i}.jpg')
                (self.root/split/'labels'/f'{split}{i}.txt').write_text('0 0.5 0.5 0.4 0.4\n')
        self.metadata = {str(self.root/'pl/images'/f'pl{i}.jpg'): [box(confidence=.2 if i%2 else .8)] for i in range(4)}
        self.weight_path.write_text(json.dumps(self.metadata))
    def tearDown(self):
        self.tmp.cleanup()
    def dataset(self, augment=False):
        hyp=get_cfg(DEFAULT_CFG)
        hyp.mosaic=1.0 if augment else 0.0
        hyp.mixup=hyp.cutmix=hyp.copy_paste=0.0
        return Phase2YOLODataset(img_path=[str(self.root/'gt/images'),str(self.root/'pl/images')],
            imgsz=64,batch_size=2,augment=augment,hyp=hyp,rect=False,stride=32,
            data={'names':{0:'object'},'nc':1,'channels':3},
            weights_json_path=self.weight_path,pseudo_label_paths=[str(self.root/'pl/images')],
            lambda1=.5,zerogate=.5,per_box=True,per_image=False)
    def test_source_coverage_and_weight_values(self):
        ds=self.dataset()
        for i,label in enumerate(ds.labels):
            expected=1. if Path(label['im_file']).stem.startswith('gt') else (.4 if int(Path(label['im_file']).stem[-1])%2==0 else 0.)
            self.assertTrue(torch.allclose(ds[i]['box_weights'], torch.tensor([expected])))
    def test_missing_pl_metadata_rejected(self):
        self.metadata.pop(next(iter(self.metadata)))
        self.weight_path.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError,'coverage'):self.dataset()
    def test_box_count_mismatch_rejected(self):
        self.metadata[next(iter(self.metadata))]=[]
        self.weight_path.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError,'count'):self.dataset()
    def test_coordinate_mismatch_rejected(self):
        for v in self.metadata.values():v[0]['cx']=.1
        self.weight_path.write_text(json.dumps(self.metadata))
        ds=self.dataset()
        with self.assertRaisesRegex(ValueError,'confidence match'):ds[4]
    def test_class_matching_uses_class_id(self):
        ds=self.dataset()
        weights=[box(0,.2),box(1,.9)]
        self.assertEqual(ds._weight_for_box([.5,.5,.4,.4],weights,0),0.)
        self.assertEqual(ds._weight_for_box([.5,.5,.4,.4],weights,1),.45)
    def test_ambiguous_boxes_rejected(self):
        ds=self.dataset()
        with self.assertRaisesRegex(ValueError,'found 2'):ds._weight_for_box([.5,.5,.4,.4],[box(),box()],0)
    def test_stem_collision_rejected(self):
        self.weight_path.write_text(json.dumps({'a/image.jpg':[box()],'b/image.png':[box()]}))
        with self.assertRaisesRegex(ValueError,'stem'):load_confidence(self.weight_path)
    def test_duplicate_json_key_rejected(self):
        self.weight_path.write_text('{"a.jpg":[],"a.jpg":[]}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):load_confidence(self.weight_path)
    def test_nonfinite_confidence_rejected(self):
        self.weight_path.write_text(json.dumps({'a.jpg':[box(confidence=float('nan'))]}))
        with self.assertRaisesRegex(ValueError,'Non-finite'):load_confidence(self.weight_path)
    def test_mosaic_affine_preserves_weight_alignment(self):
        ds=self.dataset(augment=True)
        # Distinct allowed scalars let us detect weighting contamination through augmentation.
        for i in range(16):
            sample=ds[i%len(ds)]
            self.assertEqual(len(sample['cls']),len(sample['box_weights']))
            for w in sample['box_weights'].tolist():
                self.assertTrue(any(abs(w-v)<1e-6 for v in (0.,.4,1.)))


class MergeTests(unittest.TestCase):
    def test_exact_merge(self):
        base={'detector.w':torch.zeros(2),'other':torch.ones(1)}
        merged,n,skip=teacher.merge_checkpoint_state(base,{'w':torch.ones(2)})
        self.assertEqual(n,1);self.assertEqual(skip,[])
        self.assertTrue(torch.equal(merged['detector.w'],torch.ones(2)))
        self.assertTrue(torch.equal(base['detector.w'],torch.zeros(2)))
    def test_zero_replacements_rejected(self):
        with self.assertRaisesRegex(ValueError,'zero'):teacher.merge_checkpoint_state({'detector.w':torch.zeros(2)},{'w':torch.ones(3)})
    def test_partial_mismatch_requires_explicit_allowlist(self):
        base={'detector.w':torch.zeros(2)};ft={'w':torch.ones(2),'unused':torch.ones(1)}
        with self.assertRaisesRegex(ValueError,'unmatched'):teacher.merge_checkpoint_state(base,ft)
        self.assertEqual(teacher.merge_checkpoint_state(base,ft,allowed_skipped=['detector.unused'])[1],1)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model=DetectionModel('yolo11n.yaml',nc=1,verbose=False)
        cls.model.args=get_cfg(DEFAULT_CFG)
        cls.model.train()
    def batch(self,empty=False):
        return {'batch_idx':torch.zeros(0 if empty else 2),
            'cls':torch.zeros((0 if empty else 2,1)),
            'bboxes':torch.empty((0,4)) if empty else torch.tensor([[.5,.5,.4,.4],[.2,.2,.1,.1]]),
            'box_weights':torch.ones(0 if empty else 2)}
    def test_neutral_loss_matches_stock_values_and_gradients(self):
        predictions=self.model(torch.rand(2,3,64,64))
        batch=self.batch()
        stock=v8DetectionLoss(self.model)
        ours=Phase2DetectionLoss(self.model,loc_weight_src='box',cls_weight_src='box')
        a=stock(predictions,batch)[0];b=ours(predictions,batch)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        ga=torch.autograd.grad(a.sum(),parameters,retain_graph=True,allow_unused=True)
        gb=torch.autograd.grad(b.sum(),parameters,allow_unused=True)
        for x,y in zip(ga,gb):
            if x is None:self.assertIsNone(y)
            else:torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_empty_target_batch_finite_backward(self):
        predictions=self.model(torch.rand(2,3,64,64))
        value=Phase2DetectionLoss(self.model,loc_weight_src='box',cls_weight_src='box')(predictions,self.batch(True))[0]
        self.assertTrue(torch.isfinite(value).all());value.sum().backward()
    def test_missing_weight_tensor_rejected(self):
        batch=self.batch();batch.pop('box_weights')
        with self.assertRaisesRegex(ValueError,'requires box_weights'):
            Phase2DetectionLoss(self.model,loc_weight_src='box',cls_weight_src='box')(self.model(torch.rand(2,3,64,64)),batch)


class OrchestrationTests(unittest.TestCase):
    def test_relative_dataset_paths_are_relative_to_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.yaml';p.write_text('train: images\n')
            self.assertEqual(student_utils.dataset_train_paths(p),[str(Path(d)/'images')])
    def test_evaluation_failure_is_fatal_and_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            fake=types.SimpleNamespace(train=lambda:None,best=Path(d)/'best.pt',save_dir=Path(d))
            with patch.object(train,'_export_standard_checkpoint',return_value=Path(d)/'standard.pt'),patch.object(train,'run_test_eval',side_effect=ValueError('bad IDs')):
                with self.assertRaisesRegex(RuntimeError,'required evaluation failed'):
                    train.run_experiment('gt_only',{'spec':'stock'},None,trainer_cls=lambda **kw:fake,eval_cfg={'enabled':True})
            self.assertEqual(json.loads((Path(d)/'evaluation_status.json').read_text())['status'],'failed')
    def test_unknown_phase_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml';p.write_text('steps:\n  - name: valid\n    script: never.py\n')
            with self.assertRaisesRegex(ValueError,'Unknown'):
                pipeline_common.run_phase(p,'test',['typo'])


if __name__=='__main__':unittest.main()
