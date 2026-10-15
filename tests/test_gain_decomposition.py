"""
Kazanc ayristirma testleri: src/analysis/gain_decomposition.py
Calistirma: python -m unittest tests.test_gain_decomposition -v
Sentetik gun ici sira (UTC): kur sabah -> BIST kapanisi 14:00 -> ons kapanisi 18:00 -> gece.
Dunya A: kapanistan sonra ons ve kur hareket eder. Dunya B: kapanistan sonra ons HIC hareket etmez.
"""

from pathlib import Path
from unittest import mock
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import gain_decomposition as gd  # noqa: E402
from src.models import train as tr  # noqa: E402


def world(gold_after_close: float, seed=0, end="2023-06-30"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-04", end)
    gold18, fxm = 1100.0, 1.5
    rows_p, rows_y = [], []
    for d in days:
        gold14 = gold18 * np.exp(rng.normal(0, 0.006))
        fx14 = fxm * np.exp(rng.normal(0.0002, 0.003))
        P = gold14 * fx14 / 31.1035 * np.exp(rng.normal(0, 0.0005))
        rows_p.append({"Date": d, "kapanis_tl_gr": P, "kapanis_usd_ons": gold14})
        yahoo_fx_today = fxm                                            # sabah cubugu (gunun basi)
        gold18 = gold14 * np.exp(gold_after_close * rng.normal(0, 0.006))
        rows_y.append({"Date": d, "gold_usd_oz": gold18 * 1.004, "usdtry": yahoo_fx_today})
        fxm = fx14 * np.exp(rng.normal(0.0003, 0.004))                  # gece kur hareketi
    return pd.DataFrame(rows_p), pd.DataFrame(rows_y)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gd_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        p = mock.patch.object(tr, "OUTPUT_DIR", self.tmp / "results")
        p.start()
        self.addCleanup(p.stop)

    def run_world(self, phys, panel, name="w"):
        pp, pn = self.tmp / f"{name}_phys.csv", self.tmp / f"{name}_panel.csv"
        phys.to_csv(pp, index=False, date_format="%Y-%m-%d")
        panel.to_csv(pn, index=False, date_format="%Y-%m-%d")
        return gd.run(pp, pn)


class TestDecomposition(Base):

    def test_gold_signal_found_when_gold_moves_after_close(self):
        r = self.run_world(*world(1.0, seed=1))
        c = r["comparison"].set_index("model")
        self.assertTrue(c.loc["Ons", "significantly_better"])
        self.assertTrue(c.loc["Kur", "significantly_better"])
        self.assertTrue(r["summary"]["timing_robust_gain"])
        self.assertGreater(c.loc["Ikisi", "improvement_pct"], c.loc["Ons", "improvement_pct"])

    def test_no_false_gold_signal_when_gold_does_not_move_after_close(self):
        r = self.run_world(*world(0.0, seed=2))
        c = r["comparison"].set_index("model")
        self.assertFalse(c.loc["Ons", "significantly_better"])
        self.assertTrue(c.loc["Kur", "significantly_better"])
        self.assertFalse(r["summary"]["timing_robust_gain"])

    def test_inputs_come_from_the_right_days(self):
        phys, panel = world(1.0, seed=3, end="2014-12-31")
        df = gd.build(phys, panel)
        pan = panel.set_index("Date")
        for _, row in df.sample(20, random_state=0).iterrows():
            self.assertEqual(row["gold_t"], pan.loc[row["t"], "gold_usd_oz"])   # ons: t gunu (aksam)
            self.assertEqual(row["fx_T"], pan.loc[row["T"], "usdtry"])          # kur: T gunu (sabah)
        self.assertTrue((df["T"] > df["t"]).all())

    def test_basis_correction_uses_only_past(self):
        phys, panel = world(1.0, seed=4, end="2014-12-31")
        a = gd.build(phys, panel)
        panel2 = panel.copy()
        panel2.loc[panel2["Date"] >= "2013-06-03", "gold_usd_oz"] *= 1.5
        b = gd.build(phys, panel2)
        k = a["t"] < pd.Timestamp("2013-06-03")
        pd.testing.assert_series_equal(a.loc[k, "g_ons"], b.loc[k, "g_ons"])

    def test_locked_data_never_used(self):
        phys, panel = world(1.0, seed=5)
        r1 = self.run_world(phys, panel, "a")["predictions"]
        phys2 = phys.copy()
        phys2.loc[phys2["Date"] >= "2023-01-01", "kapanis_tl_gr"] *= 3
        r2 = self.run_world(phys2, panel, "b")["predictions"]
        pd.testing.assert_frame_equal(r1, r2)
        self.assertLess(pd.to_datetime(r1["T"]).max(), pd.Timestamp("2023-01-01"))
        self.assertEqual(str(gd.LOCK.date()), "2023-01-01")


class TestHolm(unittest.TestCase):

    def test_holm_multiplies_smallest_p_by_three(self):
        rng = np.random.default_rng(9)
        n = 800
        y = rng.normal(0, 0.01, n)
        pred = pd.DataFrame({"T": pd.bdate_range("2015-01-01", periods=n), "P_t": 100.0, "y": y, "pred_RW": 0.0,
                             "pred_Ons": 0.25 * y + rng.normal(0, 0.006, n), "pred_Kur": rng.normal(0, 0.002, n),
                             "pred_Ikisi": 0.5 * y + rng.normal(0, 0.006, n)})
        for m in gd.MODELS:
            pred[f"beta_{m}"] = 1.0
        t = gd.compare(pred).set_index("model")
        lo = t["p_dm"].idxmin()
        self.assertGreater(t.loc[lo, "p_dm"], 0)
        self.assertAlmostEqual(t.loc[lo, "p_holm"] / t.loc[lo, "p_dm"], 3.0, places=9)
        self.assertTrue((t["p_holm"] >= t["p_dm"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
