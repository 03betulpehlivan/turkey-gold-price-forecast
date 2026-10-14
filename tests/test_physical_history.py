"""
Tarihsel rejim saglamligi testleri: src/models/physical_history.py
Calistirma: python -m unittest tests.test_physical_history -v
Sentetik dunya: 2011-2015 rejiminde fiziki fiyat gece hareketinden KOPUK, diger donemlerde bagli.
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

from src.models import physical_history as ph  # noqa: E402
from src.models import train as tr  # noqa: E402

BREAK = (pd.Timestamp("2011-01-01"), pd.Timestamp("2016-01-01"))


def regime_world(seed=0, start="2004-01-02", end="2023-06-30"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    G, F, P = np.empty(n), np.empty(n), np.empty(n)
    g18, fx, own = 600.0, 1.4, 20.0
    for i, d in enumerate(days):
        fx_morning = fx * np.exp(rng.normal(0.0004, 0.004))
        g15 = g18 * np.exp(rng.normal(0, 0.006))
        fx_close = fx_morning * np.exp(rng.normal(0.0002, 0.003))
        F[i] = fx_morning
        if BREAK[0] <= d < BREAK[1]:
            own = (P[i - 1] if i else own) * np.exp(rng.normal(0.0005, 0.01))   # kopuk rejim
            P[i] = own
        else:
            P[i] = g15 * fx_close / 31.1035 * np.exp(rng.normal(0, 0.001))
        g18 = g15 * np.exp(rng.normal(0, 0.006))
        G[i] = g18
        fx = fx_close
    panel = pd.DataFrame({"Date": days, "gold_usd_oz": G, "usdtry": F, "theoretical_gram_try": G * F / 31.1035})
    phys = pd.DataFrame({"Date": days, "kapanis_tl_gr": P, "agort_tl_gr": P, "kapanis_usd_ons": G, "kaynak": "BIST_KMP"})
    return phys, panel


class TestHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="phist_"))
        cls.patch = mock.patch.object(tr, "OUTPUT_DIR", cls.tmp / "results")
        cls.patch.start()
        cls.phys, cls.panel = regime_world()
        drop = pd.to_datetime(["2008-03-12", "2013-06-19", "2019-11-13"])
        cls.drop = drop
        cls.pp, cls.pn = cls.tmp / "phys.csv", cls.tmp / "hist_panel.csv"
        cls.phys.to_csv(cls.pp, index=False, date_format="%Y-%m-%d")
        cls.panel[~cls.panel["Date"].isin(drop)].to_csv(cls.pn, index=False, date_format="%Y-%m-%d")
        cls.r = ph.run(cls.pp, cls.pn)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def era(self, name):
        e = self.r["eras"]
        return e[(e["era"] == name) & (e["model"] == "Bosluk")].iloc[0]

    def test_break_found_in_the_right_regime(self):
        # Not: kopuk rejim 2016'da bitince sentetik fiyat tek gunde buyuk bir seviye sicramasi yapar;
        # bu tek aykiri gozlem genisleyen pencereli OLS'yi sonraki yillarda bozar. Bu yuzden 2016-2022
        # icin burada iddia yok; yalnizca kirilma oncesi ve kirilma rejimi sinanir.
        self.assertTrue(self.era("2006-2010")["significantly_better"])
        self.assertFalse(self.era("2011-2015")["significantly_better"])
        y = self.r["yearly"].set_index("year")
        self.assertTrue((y.loc[2006:2010, "bosluk_improvement_pct"] > 5).all())
        self.assertTrue((y.loc[2012:2015, "bosluk_improvement_pct"] < 5).all())
        self.assertTrue((y.loc[2006:2010, "beta_gap_median"].between(0.8, 1.2)).all())

    def test_period_bounds_and_freshness(self):
        s = self.r["summary"]
        self.assertGreaterEqual(pd.Timestamp(s["first_target"]), pd.Timestamp(ph.START))
        self.assertLess(pd.Timestamp(s["last_target"]), pd.Timestamp("2023-01-01"))
        self.assertGreaterEqual(s["stale_rows_excluded"], len(self.drop))
        pred = pd.read_csv(ph.out_dir() / "predictions.csv", parse_dates=["target_date"])
        self.assertFalse(pred["target_date"].isin(self.drop).any())

    def test_locked_data_never_read(self):
        a = pd.read_csv(ph.out_dir() / "predictions.csv")
        phys2 = self.phys.copy()
        phys2.loc[phys2["Date"] >= "2023-01-01", "kapanis_tl_gr"] *= 4
        p2 = self.tmp / "phys2.csv"
        phys2.to_csv(p2, index=False, date_format="%Y-%m-%d")
        ph.run(p2, self.pn)
        b = pd.read_csv(ph.out_dir() / "predictions.csv")
        pd.testing.assert_frame_equal(a, b)

    def test_locked_rows_never_reach_the_model(self):
        seen = {}
        real = ph.pv.build_frame

        def spy(phys, panel):
            seen["phys"], seen["panel"] = phys["Date"].max(), panel["Date"].max()
            return real(phys, panel)
        with mock.patch.object(ph.pv, "build_frame", spy):
            ph.run(self.pp, self.pn)
        self.assertLess(seen["phys"], pd.Timestamp("2023-01-01"))
        self.assertLess(seen["panel"], pd.Timestamp("2023-01-01"))

    def test_main_panel_refused(self):
        with self.assertRaisesRegex(ph.HistoryError, "Ana panel"):
            ph.run(self.pp, ph.pv.PANEL_PATH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
