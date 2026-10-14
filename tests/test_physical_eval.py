"""
Fiziki altin kapanis tahmini testleri: src/models/physical_eval.py
Calistirma: python -m unittest tests.test_physical_eval -v
Sentetik gun ici sira (UTC): kur sabah 03:00 -> fiziki kapanis 15:00 -> ons kapanisi 18:00 -> gece hareketi.
Gercek data/ kullanilmaz.
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

from src.models import physical_eval as pv  # noqa: E402
from src.models import train as tr  # noqa: E402

FAST = mock.patch.object(pv, "generate", wraps=pv.generate)


def make_world(kind: str, seed: int = 0, end: str = "2024-06-28"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-02", end)
    n = len(days)
    G, F, P = np.empty(n), np.empty(n), np.empty(n)
    g18, fx = 1200.0, 2.2
    for i in range(n):
        fx_morning = fx * np.exp(rng.normal(0.0005, 0.004))          # gece kur hareketi -> sabah cubugu
        g15 = g18 * np.exp(rng.normal(0, 0.006))                       # dunku 18:00'den bugun 15:00'e ons
        fx_close = fx_morning * np.exp(rng.normal(0.0002, 0.003))      # gun ici kur
        F[i] = fx_morning
        if kind == "bagli":
            P[i] = g15 * fx_close / 31.1035 * np.exp(rng.normal(0, 0.001))
        else:                                                           # uluslararasi fiyatla ilgisiz
            P[i] = (P[i - 1] if i else 80.0) * np.exp(rng.normal(0.0005, 0.01))
        g18 = g15 * np.exp(rng.normal(0, 0.006))                       # fiziki kapanistan SONRA ons hareketi
        G[i] = g18
        fx = fx_close
    panel = pd.DataFrame({"Date": days, "gold_usd_oz": G, "usdtry": F, "theoretical_gram_try": G * F / 31.1035})
    phys = pd.DataFrame({"Date": days, "kapanis_tl_gr": P, "agort_tl_gr": P, "kapanis_usd_ons": G, "kaynak": "BIST_KMP"})
    return phys, panel


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pv_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        p = mock.patch.object(tr, "OUTPUT_DIR", self.tmp / "results")
        p.start()
        self.addCleanup(p.stop)

    def run_world(self, kind, seed=0, end="2024-06-28"):
        phys, panel = make_world(kind, seed, end)
        pp, pn = self.tmp / f"phys_{kind}.csv", self.tmp / f"panel_{kind}.csv"
        phys.to_csv(pp, index=False, date_format="%Y-%m-%d")
        panel.to_csv(pn, index=False, date_format="%Y-%m-%d")
        return pv.run(pp, pn), phys, panel


class TestWorlds(Base):

    def test_overnight_information_is_found(self):
        res, _, _ = self.run_world("bagli", seed=1)
        self.assertIn("Bosluk", res["summary"]["significantly_better_than_rw"])
        c = res["comparison"].set_index("model")
        self.assertGreater(c.loc["Bosluk", "improvement_pct"], 10)

    def test_unrelated_physical_price_finds_nothing(self):
        res, _, _ = self.run_world("ilgisiz", seed=2)
        self.assertEqual(res["summary"]["significantly_better_than_rw"], [])

    def test_locked_period_never_forecast(self):
        res, _, _ = self.run_world("bagli", seed=3)
        td = pd.to_datetime(res["predictions"]["target_date"])
        self.assertEqual(pv.LOCK_START, "2023-01-01")                  # protokol karari: kod sabitine guvenilmez
        self.assertLess(td.max(), pd.Timestamp("2023-01-01"))
        self.assertGreaterEqual(td.min(), pd.Timestamp(pv.DEV_START))


class TestTimingAndLeakage(Base):

    def test_gold_from_target_day_is_rejected(self):
        phys, panel = make_world("bagli", seed=4, end="2018-12-31")
        df = pv.build_frame(phys, panel)
        pv.audit_timing(df)                                            # dogru kurulum gecer
        bad = df.copy()
        bad["gold_src_next"] = bad["target_date"]                      # ons hedef gunun kendisinden
        with self.assertRaisesRegex(pv.PhysicalEvalError, "Ons"):
            pv.audit_timing(bad)
        self.assertTrue((pd.to_datetime(df["gold_src_next"]) < pd.to_datetime(df["target_date"])).all())

    def test_future_rows_cannot_change_earlier_forecasts(self):
        phys, panel = make_world("bagli", seed=5, end="2023-12-29")
        a = pv.generate(pv.build_frame(phys, panel))
        cut = pd.Timestamp("2020-06-01")
        phys2, panel2 = phys.copy(), panel.copy()
        phys2.loc[phys2["Date"] >= cut, "kapanis_tl_gr"] *= 1.5
        panel2.loc[panel2["Date"] >= cut, ["gold_usd_oz", "usdtry"]] *= 0.7
        b = pv.generate(pv.build_frame(phys2, panel2))
        keep = pd.to_datetime(a["target_date"]) < cut
        cols = [c for c in a.columns if c.startswith("pred_")]
        pd.testing.assert_frame_equal(a.loc[keep, cols].reset_index(drop=True), b.loc[keep, cols].reset_index(drop=True))

    def test_locked_data_cannot_change_dev_results(self):
        phys, panel = make_world("bagli", seed=6, end="2024-06-28")
        a = pv.generate(pv.build_frame(phys, panel))
        phys2 = phys.copy()
        phys2.loc[phys2["Date"] >= pv.LOCK_START, "kapanis_tl_gr"] *= 3
        b = pv.generate(pv.build_frame(phys2, panel))
        pd.testing.assert_frame_equal(a, b)

    def test_timing_warning(self):
        comp = pd.DataFrame({"model": ["Bosluk", "Bosluk-Ihtiyatli"], "improvement_pct": [30.0, 5.0]})
        self.assertIn("UYARI", pv.timing_warning(comp))
        comp["improvement_pct"] = [30.0, 28.0]
        self.assertIn("tutarli", pv.timing_warning(comp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
