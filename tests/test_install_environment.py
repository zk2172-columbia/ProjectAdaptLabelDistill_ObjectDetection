"""Safety and backend resolution checks for fresh-environment installation."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ald_installer", ROOT / "scripts/install_environment.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallEnvironmentTests(unittest.TestCase):
    def test_dry_run_does_not_create_target(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "fresh"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/install_environment.py"),
                                     "--backend", "cpu", "--venv", str(target), "--dry-run"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(target.exists())
            self.assertIn("requirements-cpu.txt", result.stdout)

    def test_existing_environment_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)
            marker = target / "keep.txt"
            marker.write_text("keep")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/install_environment.py"),
                                     "--venv", str(target), "--dry-run"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), "keep")

    def test_dangling_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "venv"
            target.symlink_to(Path(d) / "missing")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/install_environment.py"),
                                     "--venv", str(target), "--dry-run"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(target.is_symlink())
            self.assertFalse((Path(d) / "missing").exists())

    def test_resolve_only_does_not_install_runtime_or_setup_source(self):
        commands = installer.installation_commands(ROOT, Path("/tmp/test-ald"), "cu130",
                                                    True, False, sys.executable)
        self.assertIn("--dry-run", commands[-1])
        self.assertIn("--ignore-installed", commands[-1])
        self.assertTrue(any("requirements-cu130.txt" in s for s in commands[-1]))
        self.assertFalse(any("setup_sam3.py" in s for command in commands for s in command))
        self.assertFalse(any("-e" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
