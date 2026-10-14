"""
Fiziki tahmin hata analizi testleri: src/analysis/physical_errors.py
Calistirma: python -m unittest tests.test_physical_errors -v
Her analiz, gercegi bilinen sentetik veride sinanir. Gercek data/ kullanilmaz.
"""

from pathlib import Path
from unittest import mock
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import physical_errors as pe_  # noqa: E402
from src.models import physical_eval as pv  # noqa: E402
from src.models import train as tr  # noqa: E402
from tests.test_physical_eval import make_world  # noqa: E402

ROLLS = pd.to_datetime(["2016-03-29", "2016-05-26", "2016-07-27", "2016-11-28", "2017-01-27", "2017-07-27"])
BLIPS = pd.to_datetime(["2016-02-16", "2016-09-14", "2017-04-19", "2017-10-18"])   # tek gunluk, geri donen sapma


def roll_inputs(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2017-12-29")
    spot = 1200 * np.exp(np.cumsum(rng.normal(0, 0.009, len(days))))
    carry = np.zeros(len(days))
    level = 0.0
    for i, d in enumerate(days):
        if d in ROLLS:
            level += 0.015                                  # kontrat degisimi: kalici +%1.5
        level *= 0.995                                       # tasima primi yavasca erir
        carry[i] = level
    gc = spot * np.exp(carry + np.where(days.isin(BLIPS), 0.02, 0.0))   # saat farki: ertesi gun geri doner
    bist = spot * np.exp(rng.normal(0, 0.002, len(days)))    # spot tabanli, saat farki gurultusu
    panel = pd.DataFrame({"Date": days, "gold_usd_oz": gc, "usdtry": 3.0, "theoretical_gram_try": gc * 3 / 31.1035})
    phys = pd.DataFrame({"Date": days, "kapanis_usd_ons": bist, "kapanis_tl_gr": bist * 3 / 31.1035,
                         "agort_tl_gr": bist * 3 / 31.1035, "kaynak": "BIST_KMP"})
    return panel, phys


class TestPieces(unittest.TestCase):

    def test_roll_detector_finds_permanent_steps_only(self):
        panel, phys = roll_inputs()
        found = pe_.detect_rolls(panel, phys)
        self.assertEqual(len(found), len(ROLLS))
        for d in ROLLS:
            self.assertTrue((abs((pd.to_datetime(found["Date"]) - d).dt.days) <= 3).any(), d)
        self.assertTrue((found["step_pct"] > 1.0).all())
        for d in BLIPS:                                        # gecici sapmalar kontrat sayilmaz
            self.assertFalse((abs((pd.to_datetime(found["Date"]) - d).dt.days) <= 3).any(), d)

    def test_calibration_detects_overreaction(self):
        rng = np.random.default_rng(1)
        n = 1500
        s = rng.normal(0, 0.006, n)
        y = s + rng.normal(0, 0.006, n)
        pred = pd.DataFrame({"actual_return": y, "price_t": 100.0, "pred_Zero-Return": 0.0,
                             **{f"pred_{m}": s for m in pe_.MODELS}})
        pred["pred_Bosluk"] = 2 * s                           # bilerek 2 kat abarti
        c = pe_.calibration(pred)
        self.assertAlmostEqual(c["Bosluk"]["mincer_zarnowitz"]["slope"], 0.5, delta=0.08)
        self.assertAlmostEqual(c["XGBoost"]["mincer_zarnowitz"]["slope"], 1.0, delta=0.15)
        lo, hi = c["Bosluk"]["mincer_zarnowitz"]["slope_ci95"]
        self.assertTrue(hi < 1.0)

    def test_concentration_shares(self):
        pred = pd.DataFrame({"actual_return": np.log([2.0] + [1.01] * 99), "price_t": 100.0,
                             **{f"pred_{m}": 0.0 for m in ["Zero-Return"] + pe_.MODELS}})
        c = pe_.concentration(pred)
        self.assertAlmostEqual(c["Zero-Return"]["share_top_1pct"], 100 * 100 / (100 + 99), places=6)

    def test_target_noise_negative_autocorr_for_noisy_close(self):
        rng = np.random.default_rng(2)
        days = pd.bdate_range("2016-01-04", "2022-12-30")
        ag = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(days))))
        close = ag * np.exp(rng.normal(0, 0.006, len(days)))  # kapanista gecici gurultu
        phys = pd.DataFrame({"Date": days, "kapanis_tl_gr": close, "agort_tl_gr": ag})
        t = pe_.target_noise(phys)
        self.assertLess(t["kapanis_tl_gr"]["lag1_autocorr"], -0.15)
        self.assertGreater(t["agort_tl_gr"]["lag1_autocorr"], -0.1)


class TestRunAndLock(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="perr_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        p = mock.patch.object(tr, "OUTPUT_DIR", self.tmp / "results")
        p.start()
        self.addCleanup(p.stop)
        self.phys, self.panel = make_world("bagli", seed=21, end="2024-06-28")
        self.pp, self.pn = self.tmp / "phys.csv", self.tmp / "panel.csv"
        self.phys.to_csv(self.pp, index=False, date_format="%Y-%m-%d")
        self.panel.to_csv(self.pn, index=False, date_format="%Y-%m-%d")
        pv.run(self.pp, self.pn)                                       # gelistirme tahminleri
        self.pred = pv.out_dir() / "predictions.csv"

    def summary(self):
        return json.loads((pe_.out_dir() / "summary.json").read_text(encoding="utf-8"))

    def test_end_to_end_and_locked_data_irrelevant(self):
        r = pe_.run(self.pred, self.pp, self.pn)
        self.assertEqual(r["n"], len(pd.read_csv(self.pred)))
        self.assertLess(pd.Timestamp(r["period"][1]), pe_.LOCK)
        a = self.summary()
        phys2, panel2 = self.phys.copy(), self.panel.copy()
        phys2.loc[phys2["Date"] >= "2023-01-01", ["kapanis_tl_gr", "kapanis_usd_ons"]] *= 2
        panel2.loc[panel2["Date"] >= "2023-01-01", "gold_usd_oz"] *= 0.5
        phys2.to_csv(self.pp, index=False, date_format="%Y-%m-%d")
        panel2.to_csv(self.pn, index=False, date_format="%Y-%m-%d")
        pe_.run(self.pred, self.pp, self.pn)
        b = self.summary()
        self.assertEqual(a, b)

    def test_locked_predictions_refused(self):
        bad = pd.read_csv(self.pred)
        bad.loc[len(bad) - 1, "target_date"] = "2023-01-05"
        bp = self.tmp / "bad.csv"
        bad.to_csv(bp, index=False)
        with self.assertRaisesRegex(pe_.ErrorAnalysisError, "kilitli"):
            pe_.run(bp, self.pp, self.pn)

    def test_overnight_world_shows_calibrated_gap(self):
        r = pe_.run(self.pred, self.pp, self.pn)
        b = r["calibration"]["Bosluk"]["mincer_zarnowitz"]["slope"]
        self.assertAlmostEqual(b, 1.0, delta=0.25)                     # dogru kurulmus dunyada abarti yok


if __name__ == "__main__":
    unittest.main(verbosity=2)
