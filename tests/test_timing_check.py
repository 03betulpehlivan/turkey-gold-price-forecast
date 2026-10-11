"""
Zamanlama kontrolu testleri: src/analysis/timing_check.py
=========================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_timing_check -v

Iki sentetik dunya:
  asenkron : USD/TRY kapanisi ABD seansindaki dolar sokunu kacirir, ertesi gun yansitir -> TUTARLI
  esanli   : USD/TRY ayni sokla ayni gun hareket eder                             -> TUTARSIZ
  ters     : USD/TRY ertesi gun ters yonde tepki verir (isaret beklenene aykiri)  -> TUTARLI olamaz
Gercek data/ klasoru kullanilmaz.
"""

from pathlib import Path
import hashlib
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import timing_check as tc  # noqa: E402
from src.models import train as tr  # noqa: E402


def make_panel(kind: str, seed: int = 0, end: str = "2026-06-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", end)
    n = len(days)
    a = rng.normal(0, 0.004, n)              # Avrupa seansi dolar haberi (kur kapanisindan once)
    u = rng.normal(0, 0.004, n)              # ABD seansi dolar haberi (kur kapanisindan SONRA)
    usd = a + u
    r = {"dxy": usd + rng.normal(0, 0.002, n),
         "gold_usd_oz": -0.8 * usd + rng.normal(0, 0.006, n),
         "silver": -1.0 * usd + rng.normal(0, 0.01, n),
         "sp500": -0.8 * usd + rng.normal(0, 0.006, n),
         "vix": 3.0 * usd + rng.normal(0, 0.03, n),
         "bist100": 0.5 * a + rng.normal(0, 0.01, n),          # yalnizca Turkiye saatindeki bilgi
         "brent": rng.normal(0, 0.015, n), "eurtry": rng.normal(0, 0.006, n),
         "us10y": rng.normal(0, 0.03, n)}
    u_prev = np.concatenate([[0.0], u[:-1]])
    if kind == "asenkron":
        r["usdtry"] = 0.8 * (a + u_prev) + rng.normal(0, 0.004, n)
    elif kind == "ters":                     # ertesi gun TERS yonde tepki: saat kaymasiyla aciklanamaz
        r["usdtry"] = 0.8 * a - 0.8 * u_prev + rng.normal(0, 0.004, n)
    else:
        r["usdtry"] = 0.8 * usd + rng.normal(0, 0.004, n)
    lv = {"dxy": 95, "gold_usd_oz": 1200, "silver": 17, "sp500": 2000, "vix": 16, "bist100": 800,
          "brent": 60, "eurtry": 3.0, "usdtry": 2.3}
    p = pd.DataFrame({"Date": days})
    for k, l in lv.items():
        p[k] = l * np.exp(np.cumsum(r[k]))
    p["us10y"] = 2.0 + np.cumsum(r["us10y"])
    p["theoretical_gram_try"] = p["gold_usd_oz"] * p["usdtry"] / 31.1035
    return p


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {q.name: hashlib.md5(q.read_bytes()).hexdigest() for q in sorted(folder.iterdir()) if q.is_file()}


class TestTimingCheck(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tc_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_world(self, kind, panel=None, name="p"):
        path = self.tmp / f"{name}.csv"
        (panel if panel is not None else make_panel(kind)).to_csv(path, index=False, date_format="%Y-%m-%d")
        return tc.run(path, self.tmp / f"out_{name}")

    def test_asynchronous_world_is_detected(self):
        res = self.run_world("asenkron", name="a")
        v = res["summary"]["verdict"]
        self.assertEqual(v["label"], "TUTARLI")
        self.assertGreaterEqual(v["consistent_us_assets"], 3)
        self.assertTrue(v["control_quiet"])

    def test_synchronous_world_is_not_flagged(self):
        res = self.run_world("esanli", name="s")
        self.assertEqual(res["summary"]["verdict"]["label"], "TUTARSIZ")

    def test_wrong_sign_lead_is_not_called_consistent(self):
        res = self.run_world("ters", name="t")
        self.assertNotEqual(res["summary"]["verdict"]["label"], "TUTARLI")
        self.assertTrue(res["table"].query("role == 'ABD_kapanis'")["outside_band"].sum() >= 3)  # etki var ama ters

    def test_only_pre_test_data_is_used(self):
        base = make_panel("asenkron", seed=1)
        a = self.run_world(None, base, "x")["table"]
        bad = base.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        cols = [c for c in bad.columns if c != "Date"]
        bad.loc[m, cols] = bad.loc[m, cols] * np.linspace(1, 2, int(m.sum()))[:, None]
        b = self.run_world(None, bad, "y")["table"]
        pd.testing.assert_frame_equal(a, b)
        self.assertLess(pd.Timestamp(self.run_world(None, base, "z")["summary"]["period"][1]),
                        pd.Timestamp(tr.TEST_START))

    def test_verdict_rule(self):
        def table(consistent, control_out):
            rows = [{"asset": k, "role": "ABD_kapanis", "consistent": i < consistent, "outside_band": i < consistent}
                    for i, k in enumerate(tc.US_ASSETS)]
            rows.append({"asset": tc.CONTROL, "role": "kontrol", "consistent": False, "outside_band": control_out})
            return pd.DataFrame(rows)
        self.assertEqual(tc.verdict(table(3, False))["label"], "TUTARLI")
        self.assertEqual(tc.verdict(table(3, True))["label"], "KARARSIZ")     # kontrol de tepki veriyor
        self.assertEqual(tc.verdict(table(2, False))["label"], "KARARSIZ")
        self.assertEqual(tc.verdict(table(1, False))["label"], "TUTARSIZ")


_REAL_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed",
              PROJECT_ROOT / "data" / "processed" / "results"]
_REAL_BEFORE = {}


def setUpModule():
    for d in _REAL_DIRS:
        _REAL_BEFORE[d] = file_hashes(d)


def tearDownModule():
    for d in _REAL_DIRS:
        if file_hashes(d) != _REAL_BEFORE[d]:
            raise AssertionError(f"Testler gercek veri klasorunu degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
