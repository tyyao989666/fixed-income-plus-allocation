import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.resume_v58_rng_fixed import ROOT, corrected_children
import subprocess


class RoutingTests(unittest.TestCase):
    def test_child_preserves_arguments_and_checks(self):
        command = [sys.executable, "-u", "-X", "utf8", str(ROOT / "scripts/train_v57_multitask.py"), "--parent", "corporate_action_v58", "--branch", "multitask_v58"]
        with patch.object(subprocess, "run") as actual:
            with corrected_children():
                subprocess.run(command, cwd=ROOT, check=True)
            self.assertIs(subprocess.run, actual)
            passed = actual.call_args.args[0]
            self.assertEqual(passed[5], "--child")
            self.assertEqual(passed[6:], command[4:])
            self.assertTrue(actual.call_args.kwargs["check"])

    def test_unrelated_commands_unchanged(self):
        command = [sys.executable, "-u", "-X", "utf8", "unrelated.py"]
        with patch.object(subprocess, "run") as actual:
            with corrected_children():
                subprocess.run(command)
            self.assertEqual(actual.call_args.args[0], command)


if __name__ == "__main__":
    unittest.main()
