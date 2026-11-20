import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
class TeacherCLITests(unittest.TestCase):
    def run_merge(self,shape,local_base=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'run/checkpoints/checkpoint.pt';raw.parent.mkdir(parents=True)
            base=root/'base.pt';merged=root/'merged.pt'
            torch.save({'model':{'detector.w':torch.zeros(2)}},base)
            torch.save({'model':{'w':torch.ones(shape)}},raw)
            cfg={'dataset':'test','sam3':{'repo_root':str(ROOT/'SAM3'),'template_config':str(ROOT/'experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml'),'generated_config_relpath':'configs/roboflow_v100/test.yaml'},'output':{'experiment_dir':str(root/'run'),'merged_checkpoint_path':str(merged),'keep_raw_checkpoint':False}}
            if local_base: cfg['sam3']['base_checkpoint_path']='base.pt'
            path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            code=f"import sys;sys.path.insert(0,{str(ROOT/'pipeline/teacher_adaptation')!r});import merge_checkpoint as m;m.download_ckpt_from_hf=lambda:{str(base)!r};sys.argv=['merge_checkpoint.py','--config',{str(path)!r}];m.main()"
            if local_base:
                code=f"import sys;sys.path.insert(0,{str(ROOT/'pipeline/teacher_adaptation')!r});import merge_checkpoint as m;m.download_ckpt_from_hf=lambda:(_ for _ in ()).throw(AssertionError('Unexpected network fallback'));sys.argv=['merge_checkpoint.py','--config',{str(path)!r}];m.main()"
            result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=45)
            self.assertTrue(raw.exists(),'Raw resumable checkpoint must always be preserved')
            if shape==2:
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads(merged.with_suffix('.merge.json').read_text())
                self.assertTrue(report['readback_verified']);self.assertEqual(report['replaced'],1)
                self.assertTrue(torch.equal(torch.load(merged,weights_only=True)['model']['detector.w'],torch.ones(2)))
            else:
                self.assertNotEqual(result.returncode,0);self.assertFalse(merged.exists())
                self.assertIn('zero parameters',result.stderr)
    def test_local_base_uses_config_relative_path_without_download(self):self.run_merge(2,local_base=True)
    def test_success_preserves_raw_and_verifies_output(self):self.run_merge(2)
    def test_bad_merge_preserves_raw_and_creates_no_output(self):self.run_merge(3)
    def test_teacher_config_generation_uses_given_paths_and_preserves_hydra_header(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'config.yaml'
            cfg={'dataset':'synthetic','sam3':{'repo_root':str(root/'sam3'),'template_config':str(ROOT/'experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml'),'generated_config_relpath':'configs/roboflow_v100/test.yaml'},'output':{'experiment_dir':str(root/'run'),'merged_checkpoint_path':str(root/'merged.pt')},'overrides':{'paths.bpe_path':str(ROOT/'SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),'paths.roboflow_vl_100_root':str(root)}}
            path.write_text(yaml.safe_dump(cfg))
            command=[sys.executable,str(ROOT/'pipeline/teacher_adaptation/generate_ft_config.py'),'--config',str(path)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            output=root/'sam3/sam3/train/configs/roboflow_v100/test.yaml'
            self.assertTrue(output.read_text().startswith('# @package _global_'))
            self.assertEqual(yaml.safe_load(output.read_text())['all_roboflow_supercategories'],['synthetic'])
            # Refuse to erase a previous run's configuration.
            result=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
