"""External teacher configs must keep source packages read-only and preserve upstream dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ExternalTeacherConfigTests(unittest.TestCase):
    def make_config(self, root, generated="run/configs/sam3.yaml"):
        config = {
            "dataset": "synthetic",
            "sam3": {
                "repo_root": str(ROOT / "SAM3"),
                "template_config": str(ROOT / "experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml"),
                "generated_config_path": generated,
                "use_cluster": 0,
                "num_gpus": 1,
            },
            "output": {"experiment_dir": "run", "merged_checkpoint_path": "run/merged.pt"},
            "overrides": {
                "paths.bpe_path": str(ROOT / "SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                "paths.roboflow_vl_100_root": str(root / "data"),
                "trainer.max_epochs": 1,
                "trainer.logging.wandb_writer.should_log": False,
            },
        }
        path = root / "phase.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def generate(self, config):
        return subprocess.run(
            [sys.executable, str(ROOT / "pipeline/teacher_adaptation/generate_ft_config.py"),
             "--config", str(config)], capture_output=True, text=True, timeout=20)

    def test_external_config_generation_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_configs = ROOT / "SAM3/sam3/train/configs"
            before = set(source_configs.rglob("*"))
            path = self.make_config(root)
            result = self.generate(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = root / "run/configs/sam3.yaml"
            original = generated.read_bytes()
            self.assertTrue(original.startswith(b"# @package _global_"))
            cfg = yaml.safe_load(original)
            self.assertEqual(cfg["all_roboflow_supercategories"], ["synthetic"])
            self.assertEqual(cfg["paths"]["experiment_log_dir"], str(root / "run"))
            self.assertEqual(set(source_configs.rglob("*")), before)
            self.assertNotEqual(self.generate(path).returncode, 0)
            self.assertEqual(generated.read_bytes(), original)

    def test_external_path_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.generate(self.make_config(root, "outside.yaml"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output.experiment_dir/configs", result.stderr)
            self.assertFalse((root / "outside.yaml").exists())

    def test_external_path_cannot_escape_through_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run/configs").mkdir(parents=True)
            (root / "outside").mkdir()
            (root / "run/configs/link").symlink_to(root / "outside", target_is_directory=True)
            result = self.generate(self.make_config(root, "run/configs/link/sam3.yaml"))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "outside/sam3.yaml").exists())

    def test_external_config_root_symlink_cannot_redirect_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            (root / "outside").mkdir()
            (root / "run/configs").symlink_to(root / "outside", target_is_directory=True)
            result = self.generate(self.make_config(root))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "outside/sam3.yaml").exists())

    def test_external_dispatch_uses_adapter_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'pipeline/teacher_adaptation')!r})
import run_finetune
def checked_run(cmd, **kwargs):
    assert cmd[1] == {str(ROOT / 'scripts/run_sam3_training.py')!r}, cmd
    assert cmd[cmd.index('--config') + 1] == {str(root / 'run/configs/sam3.yaml')!r}
    assert cmd[cmd.index('--num-gpus') + 1] == '1'
    assert cmd[cmd.index('--use-cluster') + 1] == '0'
    assert kwargs['check'] is True
run_finetune.subprocess.run = checked_run
sys.argv = ['run_finetune.py', '--config', {str(config)!r}]
run_finetune.main()
"""
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_adapter_composes_real_template_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            result = self.generate(config)
            self.assertEqual(result.returncode, 0, result.stderr)
            code = f"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT / 'SAM3')!r})
sys.path.insert(0, {str(ROOT / 'scripts')!r})
from sam3.train import train as upstream
from hydra import compose
from omegaconf import OmegaConf
import run_sam3_training
called = []
def inspect(args):
    cfg = compose(config_name=args.config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved['trainer']['max_epochs'] == 1
    assert resolved['roboflow_train']['supercategory'] == 'synthetic'
    assert resolved['paths']['experiment_log_dir'] == {str(root / 'run')!r}
    assert args.num_gpus == 1 and args.use_cluster is False
    called.append(True)
upstream.main = inspect
run_sam3_training.run(Path({str(root / 'run/configs/sam3.yaml')!r}), Path({str(ROOT / 'SAM3')!r}))
assert called == [True]
"""
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                                         "WANDB_MODE": "disabled"}, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
