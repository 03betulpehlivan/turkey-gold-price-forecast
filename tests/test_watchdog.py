"""
Bekci testleri: src/watchdog.py ve .github/workflows/watchdog.yml
Calistirma: python -m unittest tests.test_watchdog -v
"""

from datetime import datetime
from pathlib import Path
from unittest import mock
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import watchdog as wd  # noqa: E402

MON_1420 = datetime.fromisoformat("2026-10-19T11:20:00+00:00")      # pazartesi 14:20 Istanbul
MON_2315 = datetime.fromisoformat("2026-10-19T20:15:00+00:00")


def hb(sabah_at, aksam_at="2026-10-16T22:31:00+03:00", sabah_overall="TAMAM", aksam_overall="TAMAM"):
    return {"sabah": {"at": sabah_at, "overall": sabah_overall, "kod": 0, "problems": []},
            "aksam": {"at": aksam_at, "overall": aksam_overall, "kod": 0, "problems": []}}


class TestCheck(unittest.TestCase):
    def test_ran_today_ok(self):
        self.assertEqual(wd.check(hb("2026-10-19T12:41:00+03:00"), "sabah", MON_1420), [])

    def test_missed_run_detected(self):
        p = wd.check(hb("2026-10-16T12:41:00+03:00"), "sabah", MON_1420)          # son: cuma
        self.assertEqual(len(p), 1)
        self.assertIn("bugun calismadi", p[0])
        self.assertIn("16.10 12:41", p[0])

    def test_error_detected_but_warning_is_fine(self):
        self.assertIn("HATA", wd.check(hb("2026-10-19T12:41:00+03:00", sabah_overall="HATA"), "sabah", MON_1420)[0])
        self.assertEqual(wd.check(hb("2026-10-19T12:41:00+03:00", sabah_overall="UYARI"), "sabah", MON_1420), [])

    def test_weekend_and_too_early_not_checked(self):
        sat = datetime.fromisoformat("2026-10-17T11:20:00+00:00")
        early = datetime.fromisoformat("2026-10-19T09:50:00+00:00")                # 12:50, 40 dk dolmadi
        self.assertEqual(wd.check(hb("2026-10-16T12:41:00+03:00"), "sabah", sat), [])
        self.assertEqual(wd.check(hb("2026-10-16T12:41:00+03:00"), "sabah", early), [])

    def test_missing_file_or_task(self):
        self.assertIn("dosyasi yok", wd.check(None, "aksam", MON_2315)[0])
        self.assertIn("hic kalp atisi yok", wd.check({"sabah": hb("x")["sabah"]}, "aksam", MON_2315)[0])

    def test_auto_task_and_utc_handling(self):
        self.assertEqual(wd.auto_task(MON_1420), "sabah")
        self.assertEqual(wd.auto_task(MON_2315), "aksam")
        late_utc = hb("2026-10-19T09:41:00+00:00")                                 # ayni an, UTC yazimi
        self.assertEqual(wd.check(late_utc, "sabah", MON_1420), [])


class TestMain(unittest.TestCase):
    def test_exit_codes(self):
        tmp = Path(tempfile.mkdtemp(prefix="wd_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = tmp / "heartbeat.json"
        p.write_text(json.dumps(hb("2026-10-19T12:41:00+03:00", aksam_at="2026-10-16T22:31:00+03:00")), encoding="utf-8")
        with mock.patch.object(wd, "HEARTBEAT_PATH", p), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(wd.main(["--simdi", MON_1420.isoformat()]), 0)
            self.assertEqual(wd.main(["--simdi", MON_2315.isoformat()]), 1)             # pazartesi aksami yok
            self.assertEqual(wd.main(["--gorev", "ikisi", "--simdi", MON_2315.isoformat()]), 1)
        self.assertIn("SORUN", out.getvalue())

    def test_stdlib_only(self):
        src = (PROJECT_ROOT / "src" / "watchdog.py").read_text(encoding="utf-8")
        src.encode("ascii")
        for bad in ("import pandas", "import numpy", "from src."):
            self.assertNotIn(bad, src)


class TestWorkflow(unittest.TestCase):
    def test_schedule_and_hardening(self):
        w = (PROJECT_ROOT / ".github" / "workflows" / "watchdog.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "20 11 * * 1-5"', w)                                   # 14:20 Istanbul
        self.assertIn('cron: "15 20 * * 1-5"', w)                                   # 23:15 Istanbul
        self.assertIn("python3 -m src.watchdog", w)
        self.assertNotIn("pip install", w)
        self.assertIn("persist-credentials: false", w)
        self.assertRegex(w, r"permissions:\s*\n\s*contents: read")


if __name__ == "__main__":
    unittest.main(verbosity=2)
