"""One-command planning, artifact handoff and failure containment without GPU work."""
import importlib.util
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_ald", ROOT / "scripts/run_ald.py")
ald = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ald)


class UnifiedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "dataset"
        for name in ("train", "label", "val", "eval"):
            folder = self.data / name
            folder.mkdir(parents=True)
            filename = name + ".jpg"
            (folder / filename).write_bytes(b"fixture only; never decoded")
            (folder / "_annotations.coco.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": filename}],
                "annotations": [],
                "categories": [{"id": 1, "name": "object"}],
            }))
        self.classes = self.root / "classes.yaml"
        self.classes.write_text("classes:\n  - id: 0\n    name: object\n    text_prompt: object\n")
        self.base = self.root / "base.pt"
        self.base.write_bytes(b"trusted test fixture; never loaded")
        self.output = self.root / "run"
        self.argv = ["--split-root", str(self.data), "--classes-file", str(self.classes),
                     "--base-checkpoint", str(self.base), "--output", str(self.output),
                     "--student-model", "yolo11n.yaml", "--teacher-epochs", "1", "--student-epochs", "1"]
        self.fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                       get_device_name=lambda _: "mock"),
            __version__="mock", version=types.SimpleNamespace(cuda="mock"))

    def test_dry_run_writes_nothing_and_does_not_configure_or_start_training(self):
        before = set(self.root.rglob("*"))
        with patch.object(ald, "configure", side_effect=AssertionError("must not configure")), \
             patch.object(ald, "run_stage", side_effect=AssertionError("must not run")):
            self.assertEqual(ald.main(self.argv + ["--dry-run"]), 0)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_existing_output_rejected_before_mutation(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("unchanged")
        with self.assertRaises(SystemExit):
            ald.main(self.argv + ["--dry-run"])
        self.assertEqual(marker.read_text(), "unchanged")

    def test_overlapping_split_is_rejected(self):
        p = self.data / "val/_annotations.coco.json"
        d = json.loads(p.read_text())
        d["images"][0]["file_name"] = "train.jpg"
        p.write_text(json.dumps(d))
        with self.assertRaises(SystemExit):
            ald.main(self.argv + ["--dry-run"])

    def test_category_name_mapping_allows_different_coco_id_order(self):
        self.classes.write_text("classes:\n  - id: 0\n    name: zebra\n  - id: 1\n    name: apple\n")
        for name in ("train", "label", "val", "eval"):
            p = self.data / name / "_annotations.coco.json"
            d = json.loads(p.read_text())
            d["categories"] = [{"id": 10, "name": "apple"}, {"id": 20, "name": "zebra"}]
            p.write_text(json.dumps(d))
        self.assertEqual(ald.main(self.argv + ["--dry-run"]), 0)
        self.assertFalse(self.output.exists())

    def test_generation_wires_local_base_to_adaptation_and_merged_teacher_to_label(self):
        args = ald.parser().parse_args(self.argv)
        ald.validate_inputs(args)
        self.output.mkdir()
        config = ald.configure(args)
        teacher = ald.yaml.safe_load((config / "teacher_adaptation.yaml").read_text())
        pseudo = ald.yaml.safe_load((config / "pseudo_labeling.yaml").read_text())
        student = ald.yaml.safe_load((config / "student_training.yaml").read_text())
        self.assertEqual(teacher["sam3"]["base_checkpoint_path"], str(self.base))
        self.assertEqual(teacher["overrides"]["trainer.model.checkpoint_path"], str(self.base))
        self.assertFalse(teacher["overrides"]["trainer.model.load_from_HF"])
        self.assertTrue(Path(teacher["sam3"]["generated_config_path"]).is_relative_to(self.output))
        self.assertEqual(pseudo["sam3"]["checkpoint_path"],
                         str(self.output / "artifacts/run/teacher/merged_checkpoint.pt"))
        self.assertEqual(student["training"]["epochs"], 1)
        self.assertFalse(student["training"]["amp"])

    def test_stage_failure_stops_before_label_and_preserves_failure_status(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch}), \
             patch.object(ald, "run_stage", side_effect=RuntimeError("adapt failed")) as stage:
            self.assertEqual(ald.main(self.argv), 1)
        self.assertEqual(stage.call_count, 1)
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stages"][0]["stage"], "adapt")
        self.assertIn("adapt failed", status["error"])

    def test_success_requires_deployable_checkpoint_and_metrics(self):
        def stage(command, log, env, timeout):
            log.write_text("mock stage")
            if "--stages" in command and command[-1] == "student_training":
                folder = self.output / "artifacts/run/weights/gt_pl_perbox"
                folder.mkdir(parents=True)
                (folder / "best_standard.pt").write_bytes(b"mock checkpoint")
                (folder / "coco_metrics.json").write_text('{"metrics": {"mAP50": 0.0}}')
        with patch.dict(sys.modules, {"torch": self.fake_torch}), patch.object(ald, "run_stage", side_effect=stage):
            self.assertEqual(ald.main(self.argv), 0)
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual([x["stage"] for x in status["stages"]], ["adapt", "label", "distill"])
        self.assertEqual(status["status"], "completed")
        self.assertTrue((self.output / "result.json").exists())

    def test_timeout_kills_descendant_even_when_leader_exits_first(self):
        marker = self.root / "grandchild.pid"
        child = ("import signal,time,os,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                 "pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid()));time.sleep(30)")
        leader = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(30)"
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                ald.run_stage([sys.executable, "-c", leader], self.root / "tree-timeout.log", dict(os.environ), 1)
            pid = int(marker.read_text())
            for _ in range(20):
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("Descendant survived timeout cleanup")
        finally:
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_timeout_terminates_child_process(self):
        marker = self.root / "child.pid"
        code = "import os,time,pathlib;pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid()));time.sleep(30)"
        with self.assertRaises(subprocess.TimeoutExpired):
            ald.run_stage([sys.executable, "-c", code], self.root / "timeout.log", dict(os.environ), 1)
        pid = int(marker.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
