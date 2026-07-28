import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
    def test_power_shell_scripts_check_native_exit_codes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for script_name in [
            "run_full_experiments.ps1",
            "download_cic_iot_diad.ps1",
        ]:
            script = (root / "scripts" / script_name).read_text(encoding="utf-8")
            self.assertIn("Invoke-Step", script)
            self.assertIn("$LASTEXITCODE", script)


if __name__ == "__main__":
    unittest.main()
