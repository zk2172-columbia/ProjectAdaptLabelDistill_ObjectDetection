"""Environment diagnostics must never copy credential values into training logs."""
import ast
import logging
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SafeTrainingLogTests(unittest.TestCase):
    def test_environment_values_never_reach_logger(self):
        source = ROOT / "SAM3/sam3/train/utils/train_utils.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "log_env_variables")
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"logging": logging, "os": os}
        exec(compile(module, str(source), "exec"), namespace)
        synthetic = {
            "HF_TOKEN": "synthetic-hf-value-do-not-log",
            "ORDINARY_SETTING": "synthetic-secret-in-unexpected-key",
            "AWS_SECRET_ACCESS_KEY": "synthetic-aws-value-do-not-log",
        }
        with patch.dict(os.environ, synthetic, clear=True), self.assertLogs(level="INFO") as captured:
            namespace["log_env_variables"]()
        output = "\n".join(captured.output)
        for value in synthetic.values():
            self.assertNotIn(value, output)
        for key in synthetic:
            self.assertIn(key, output)


if __name__ == "__main__":
    unittest.main()
