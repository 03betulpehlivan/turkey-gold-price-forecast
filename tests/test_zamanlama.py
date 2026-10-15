"""
Zamanlama betikleri yapisal testleri: scripts/zamanli_calistir.ps1, scripts/gorev_kur.ps1
Calistirma: python -m unittest tests.test_zamanlama -v
PowerShell CALISTIRILMAZ; betik metni sinanir.
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAP = (ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")
SETUP = (ROOT / "scripts" / "gorev_kur.ps1").read_text(encoding="utf-8")


class TestWrapper(unittest.TestCase):

    def test_maps_tasks_to_existing_scripts(self):
        self.assertIn('ValidateSet("sabah", "aksam")', WRAP)
        for name in ("sabah_fiziki.ps1", "gunluk_tahmin.ps1"):
            self.assertIn(f"scripts\\{name}", WRAP)
            self.assertTrue((ROOT / "scripts" / name).exists())

    def test_transcript_always_closed_and_exit_code_propagated(self):
        self.assertLess(WRAP.index("Start-Transcript"), WRAP.index("try {"))
        start = WRAP.index("finally {") + len("finally {")
        block = WRAP[start:WRAP.index("}", start)]                 # yalnizca finally blogunun ici
        self.assertIn("Stop-Transcript", block)
        self.assertIn("$kod = $LASTEXITCODE", WRAP)
        self.assertTrue(WRAP.rstrip().endswith("exit $kod"))

    def test_logs_go_to_logs_folder(self):
        self.assertRegex(WRAP, r'logs\\\{0\}_\{1\}\.log')


class TestSetup(unittest.TestCase):

    def test_two_tasks_with_expected_weekday_times(self):
        kur = re.findall(r'^Kur "([^"]+)" "(\w+)" @\(([^)]*)\)', SETUP, re.M)
        tasks = {n: (g, re.findall(r'"([\d:]+)"', t)) for n, g, t in kur}
        self.assertEqual(tasks, {"AltinTahmin-Sabah": ("sabah", ["12:30", "13:30"]),
                                 "AltinTahmin-Aksam": ("aksam", ["22:30"])})
        self.assertIn('@("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")', SETUP)

    def test_sabah_times_inside_protocol_window(self):
        # LIVE_PHYS_V1 penceresi 05:00-11:00 UTC = 08:00-14:00 TSI
        for t in re.findall(r'"(\d\d:\d\d)"', SETUP.split('Kur "AltinTahmin-Sabah"')[1].split("\n")[0]):
            h, m = map(int, t.split(":"))
            self.assertTrue(8 * 60 <= h * 60 + m < 14 * 60, t)

    def test_safe_settings_and_no_credentials(self):
        for s in ("-StartWhenAvailable", "-MultipleInstances IgnoreNew", "-ExecutionTimeLimit"):
            self.assertIn(s, SETUP)
        for bad in ("-Password", "-User ", "EVDS_API_KEY", "-RunLevel Highest"):
            self.assertNotIn(bad, SETUP)
        self.assertIn("zamanli_calistir.ps1", SETUP)
        self.assertIn("-Force", SETUP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
