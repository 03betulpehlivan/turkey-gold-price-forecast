"""
Fiziki altin bagimsiz sinavi testleri: src/models/physical_exam.py
Calistirma: python -m unittest tests.test_physical_exam -v
Sentetik dunyalar test_physical_eval ile ortak; git sahte fonksiyonla. Gercek data/ ve research/ kullanilmaz.
"""

from pathlib import Path
from unittest import mock
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import physical_exam as px  # noqa: E402
from tests.test_physical_eval import make_world  # noqa: E402


def clean_git(paths):
    return {"ok": True, "commit": "abc123", "dirty": "", "prereg_tracked": True}


class Base(unittest.TestCase):
    kind = "bagli"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="px_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        phys, panel = make_world(self.kind, seed=11, end="2024-06-28")
        self.phys_path, self.panel_path = self.tmp / "physical_gold_daily.csv", self.tmp / "market_panel.csv"
        phys.to_csv(self.phys_path, index=False, date_format="%Y-%m-%d")
        self.panel = panel
        panel.to_csv(self.panel_path, index=False, date_format="%Y-%m-%d")
        r = self.tmp / "research"
        p = mock.patch.multiple(px, PREREG_PATH=r / "preregistration" / "final_exam_phys_v1.json",
                                REGISTRY_PATH=r / "registry.jsonl", RESULT_DIR=r / "final_exam_phys_v1",
                                PHYS_PATH=self.phys_path, PANEL_PATH=self.panel_path, GIT=clean_git)
        p.start()
        self.addCleanup(p.stop)

    def events(self):
        return [e["event"] for e in px.registry()]


class TestGuards(Base):

    def test_prepare_idempotent_and_locked(self):
        self.assertEqual(px.prepare()["status"], "olusturuldu")
        self.assertEqual(px.prepare()["status"], "zaten hazir")
        with mock.patch.object(px, "EXAM_START", "2022-01-01"):
            with self.assertRaisesRegex(px.ExamError, "degistirilemez"):
                px.prepare()

    def test_refusals(self):
        with self.assertRaisesRegex(px.ExamError, "On kayit yok"):
            px.run()
        px.prepare()
        with mock.patch.object(px, "GIT", lambda p: {"ok": True, "commit": "x", "dirty": "", "prereg_tracked": False}):
            with self.assertRaisesRegex(px.ExamError, "Git"):
                px.run()
        self.phys_path.write_text(self.phys_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(px.ExamError, "on kayitla ayni degil"):
            px.run()
        self.assertEqual(self.events(), ["refused", "prepared", "refused", "refused"])


class TestRun(Base):

    def setUp(self):
        super().setUp()
        drop = pd.to_datetime(["2023-03-15", "2023-08-09", "2024-02-21"])       # bu gunlerin kur/ons verisi yok
        self.panel[~self.panel["Date"].isin(drop)].to_csv(self.panel_path, index=False, date_format="%Y-%m-%d")
        self.drop = [d.strftime("%Y-%m-%d") for d in drop]
        px.prepare()
        self.r = quiet_run()

    def test_exam_set_freshness_and_verdict(self):
        pred = pd.read_csv(px.RESULT_DIR / "predictions.csv", parse_dates=["target_date"])
        self.assertGreaterEqual(pred["target_date"].min(), pd.Timestamp("2023-01-01"))
        self.assertTrue(set(self.drop) <= set(self.r["stale_rows_excluded"]))
        self.assertFalse(pred["target_date"].dt.strftime("%Y-%m-%d").isin(self.drop).any())
        self.assertEqual(self.r["primary"]["verdict"], "DAHA_IYI")
        self.assertGreater(self.r["primary"]["improvement_pct"], 5)

    def test_single_run_registry_and_hash(self):
        with mock.patch.object(px, "RESULT_DIR", self.tmp / "baska"):
            with self.assertRaises(px.ExamAlreadyRun):
                px.run()
        self.assertEqual(self.events(), ["prepared", "started", "completed", "refused"])
        done = [e for e in px.registry() if e["event"] == "completed"][0]
        self.assertEqual(done["result_md5"], hashlib.md5((px.RESULT_DIR / "result.json").read_bytes()).hexdigest())
        res = json.loads((px.RESULT_DIR / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(res["git_commit"], "abc123")
        self.assertIn("Bosluk-Ihtiyatli", res["secondary"])


class TestUnrelatedWorld(Base):
    kind = "ilgisiz"

    def test_no_false_win(self):
        px.prepare()
        r = quiet_run()
        self.assertNotEqual(r["primary"]["verdict"], "DAHA_IYI")


def quiet_run():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return px.run()


_REAL = [PROJECT_ROOT / "research", PROJECT_ROOT / "data" / "processed" / "physical"]
_BEFORE = {}


def _h(folder):
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()} if folder.exists() else {}


def setUpModule():
    for d in _REAL:
        _BEFORE[d] = _h(d)


def tearDownModule():
    for d in _REAL:
        if _h(d) != _BEFORE[d]:
            raise AssertionError(f"Testler gercek klasoru degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
