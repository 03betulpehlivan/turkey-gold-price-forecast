"""
Canli protokol geriye donuk test ve aralik testleri: src/models/live_backtest.py
Calistirma: python -m unittest tests.test_live_backtest -v
Sentetik dunya; canli kod (physical_forecast.forecast) birebir cagrilir. Gercek data/ kullanilmaz.
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

from src.live import physical_forecast as lp  # noqa: E402
from src.models import live_backtest as lb  # noqa: E402
from src.models import train as tr  # noqa: E402
from tests.test_physical_history import regime_world  # noqa: E402

PATCH_LOCK = pd.Timestamp("2013-01-01")


class TestBacktest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lbt_"))
        cls.p1 = mock.patch.object(tr, "OUTPUT_DIR", cls.tmp / "results")
        cls.p2 = mock.patch.object(lb, "LOCK", PATCH_LOCK)
        cls.p3 = mock.patch.object(lb, "START", pd.Timestamp("2011-06-01"))
        cls.p4 = mock.patch.object(lb, "PHYS_FROM", pd.Timestamp("2009-06-01"))
        cls.p5 = mock.patch.object(lp, "TRAIN_START", "2009-07-01")
        for p in (cls.p1, cls.p2, cls.p3, cls.p4, cls.p5):
            p.start()
        phys, panel = regime_world(seed=7, start="2009-06-01", end="2013-03-29")
        cls.drop = pd.to_datetime(["2012-03-14", "2012-08-08"])
        cls.phys, cls.panel = phys, panel[~panel["Date"].isin(cls.drop)]
        cls.calls = 0
        real = lp.forecast

        def spy(*a, **k):
            TestBacktest.calls += 1
            return real(*a, **k)
        with mock.patch.object(lp, "forecast", spy):
            cls.res = lb.replay(cls.phys, cls.panel)
        cls.ok = lb.add_intervals(cls.res)

    @classmethod
    def tearDownClass(cls):
        for p in (cls.p1, cls.p2, cls.p3, cls.p4, cls.p5):
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_live_code_called_for_every_day_and_lock(self):
        self.assertEqual(self.calls, len(self.res))
        self.assertLess(pd.to_datetime(self.res["target_date"]).max(), PATCH_LOCK)
        self.assertEqual(pv_lock(), "2023-01-01")

    def test_missing_fx_refused_like_live(self):
        r = self.res.set_index("target_date")
        for d in self.drop:
            self.assertIn("Kur cubugu", r.loc[d, "refused"])

    def test_intervals_use_only_known_past(self):
        ok = self.ok.copy()
        i = len(ok) - 30
        bad = ok.copy()
        bad.loc[i:, "actual"] = bad.loc[i:, "actual"] * 3                 # i ve sonrasi gerceklesmeler bozulur
        again = lb.add_intervals(pd.concat([bad.assign(refused="")], ignore_index=True))
        cols = [c for c in ok.columns if c.startswith(("lo80_", "hi80_", "lo95_", "hi95_"))]
        pd.testing.assert_frame_equal(ok.loc[: i, cols].reset_index(drop=True), again.loc[: i, cols].reset_index(drop=True))

    def test_coverage_reasonable_in_linked_regime(self):
        t = lb.interval_table(self.ok)
        cov = t.set_index(["model", "level"])["coverage"]
        self.assertTrue(0.65 <= cov[("RW", 80)] <= 0.92)
        self.assertTrue(0.88 <= cov[("RW", 95)] <= 0.99)

    def test_previous_day_mode_uses_strictly_earlier_fx_on_same_days(self):
        seen = {}
        real = lp.forecast

        def spy(phys, panel, fx_h, now):
            seen[pd.Timestamp(now.date())] = float(fx_h["close"].iat[0]) if len(fx_h) else None
            return real(phys, panel, fx_h, now)
        with mock.patch.object(lp, "forecast", spy):
            cons = lb.replay(self.phys, self.panel, "onceki_gun")
        fx = self.panel.set_index("Date")["usdtry"]
        for T in list(seen)[100:110]:
            self.assertEqual(seen[T], float(fx[fx.index < T].iat[-1]))
        self.assertEqual(list(cons["target_date"]), list(self.res["target_date"]))      # ayni hedef gunler
        self.assertEqual(list(cons["refused"] != ""), list(self.res["refused"] != ""))   # ayni red gunleri
        with self.assertRaisesRegex(lb.BacktestError, "Bilinmeyen"):
            lb.replay(self.phys, self.panel, "yarin")

    def test_overnight_fx_matters_in_linked_world(self):
        # setUpClass dunyasi 2011-2015'te kopuk rejimde; bu karsilastirma BAGLI bir dunyada yapilir
        from tests.test_physical_eval import make_world
        phys, panel = make_world("bagli", seed=41, end="2016-06-30")
        with mock.patch.multiple(lb, LOCK=pd.Timestamp("2016-06-01"), START=pd.Timestamp("2015-06-01"),
                                 PHYS_FROM=pd.Timestamp("2014-01-01")), \
                mock.patch.object(lp, "TRAIN_START", "2014-02-01"):
            imp = {}
            for mode in lb.FX_MODES:
                d = lb.add_intervals(lb.replay(phys, panel, mode))
                imp[mode] = 1 - np.abs(d["actual"] - d["pred_Bosluk"]).mean() / np.abs(d["actual"] - d["pred_RW"]).mean()
        self.assertGreater(imp["ayni_gun"], 0.05)
        self.assertLess(imp["onceki_gun"], imp["ayni_gun"])           # gece kur bilgisi kaybolur

    def test_main_panel_refused(self):
        with self.assertRaisesRegex(lb.BacktestError, "Genisletilmis"):
            lb.run(self.tmp / "x.csv", lb.pv.PANEL_PATH)


def pv_lock():
    from src.models import physical_eval as pv
    return pv.LOCK_START


if __name__ == "__main__":
    unittest.main(verbosity=2)
