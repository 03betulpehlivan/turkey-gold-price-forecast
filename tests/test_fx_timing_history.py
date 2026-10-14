"""
Yahoo kur cubugu saati (yillara gore) testleri: src/analysis/fx_timing_history.py
Calistirma: python -m unittest tests.test_fx_timing_history -v
Sentetik saatlik kur yolu: Yahoo cubugu 2015 oncesi 20:00 UTC, sonrasi 03:00 UTC; TCMB 15:30 Istanbul.
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

from src.analysis import fx_timing_history as ft  # noqa: E402
from src.models import train as tr  # noqa: E402

SWITCH = 2015


def world(seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2005-01-01", "2026-06-30 23:00", freq="1h", tz="UTC")
    path = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.0015, len(hours)))) * 1.4, index=hours)
    days = pd.bdate_range("2005-01-03", "2026-06-30")
    yh = [20 if d.year < SWITCH else 3 for d in days]
    ytimes = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=h) for d, h in zip(days, yh)])
    hc = ft.tcmb_hour_utc(pd.Series(days))
    ctimes = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=int(np.floor(h))) for d, h in zip(days, hc)])
    yahoo = pd.DataFrame({"Date": days, "usdtry": path.reindex(ytimes).to_numpy()})
    tcmb = pd.DataFrame({"Date": days, ft.TCMB_CODE: path.reindex(ctimes).to_numpy()})
    return yahoo, tcmb


class TestTiming(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fth_"))
        cls.p = mock.patch.object(tr, "OUTPUT_DIR", cls.tmp / "results")
        cls.p.start()
        yahoo, tcmb = world()
        (cls.tmp / "evds").mkdir()
        tcmb.to_csv(cls.tmp / "evds" / "data.csv", index=False, date_format="%Y-%m-%d")
        yahoo.to_csv(cls.tmp / "panel.csv", index=False, date_format="%Y-%m-%d")
        cls.r = ft.run(cls.tmp / "evds", cls.tmp / "panel.csv")

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hour_and_switch_year_recovered(self):
        t = self.r["table"].set_index("year")
        early, late = t.loc[2006:2014], t.loc[2016:2025]
        self.assertTrue((early["verdict"] == "TCMB_SONRASI").all())
        self.assertTrue((late["verdict"] == "SABAH").all())
        # yillik saat tahmini ~104 gozleme dayanir ve gurultuludur; donem ortalamasi yansiz olmali
        self.assertLess(abs(early["est_hour_utc"].mean() - 20), 2)
        self.assertLess(abs(late["est_hour_utc"].mean() - 3), 2)
        self.assertTrue((early["est_hour_utc"].sub(20).abs() < 5).all())
        self.assertTrue((late["est_hour_utc"].sub(3).abs() < 5).all())
        self.assertTrue(self.r["summary"]["validation_2024_2025_morning"])
        self.assertIn(2010, self.r["summary"]["non_morning_years"])

    def test_istanbul_dst_handled(self):
        h = ft.tcmb_hour_utc(pd.Series(pd.to_datetime(["2010-01-15", "2010-07-15", "2020-01-15"])))
        self.assertEqual(list(h), [13.5, 12.5, 12.5])

    def test_missing_series_rejected(self):
        bad = self.tmp / "bad"
        bad.mkdir(exist_ok=True)
        pd.DataFrame({"Date": ["2020-01-02"], "TP.X": [1.0]}).to_csv(bad / "data.csv", index=False)
        with self.assertRaisesRegex(ft.TimingError, "sutunu yok"):
            ft.run(bad, self.tmp / "panel.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)


def bist_world(seed=3):
    """BIST kapanisi 14:00 UTC. Yahoo: <=2015 03:00 (kapanis oncesi), 2016-2022 20:00 (sonrasi). 2010: bagimli getiri."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2005-01-01", "2024-06-30 23:00", freq="1h", tz="UTC")
    inc = rng.normal(0, 0.0015, len(hours))
    yr = hours.year.to_numpy()
    days_idx = (hours - hours[0]).days
    drift = np.zeros(days_idx.max() + 1)
    for i in range(1, len(drift)):
        drift[i] = 0.97 * drift[i - 1] + rng.normal(0, 0.004)
    inc = np.where(yr == 2010, inc + drift[days_idx] / 24, inc)          # 2010: kalici gunluk egilim
    fx = pd.Series(np.exp(np.cumsum(inc)) * 1.4, index=hours)
    days = pd.bdate_range("2005-01-03", "2024-06-28")
    yh = [3 if d.year <= 2015 else 20 for d in days]
    ytimes = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=h) for d, h in zip(days, yh)])
    btimes = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=14) for d in days])
    gold = 600 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    fxb = fx.reindex(btimes).to_numpy()
    phys = pd.DataFrame({"Date": days, "kapanis_tl_gr": gold * fxb / 31.1035, "kapanis_usd_ons": gold})
    yahoo = pd.DataFrame({"Date": days, "usdtry": fx.reindex(ytimes).to_numpy()})
    return yahoo, phys


class TestBistReference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.yahoo, cls.phys = bist_world()
        cls.e = ft.bist_eras(cls.yahoo, cls.phys).set_index("era")

    def test_era_decisions(self):
        self.assertEqual(self.e.loc["2005-2010", "verdict"], "GECERSIZ")          # 2010 bagimli getiri
        self.assertEqual(self.e.loc["2011-2015", "verdict"], "KAPANIS_ONCESI")
        self.assertEqual(self.e.loc["2016-2022", "verdict"], "KAPANIS_SONRASI")

    def test_clean_world_says_no_leak(self):
        y, p = self.yahoo.copy(), self.phys.copy()
        clean = ft.bist_eras(y[y["Date"].dt.year.between(2011, 2015)], p[p["Date"].dt.year.between(2011, 2015)])
        self.assertEqual(list(clean["verdict"]), ["KAPANIS_ONCESI"])

    def test_exam_period_not_used(self):
        t = ft.bist_yearly(self.yahoo, self.phys)
        self.assertLessEqual(t["year"].max(), 2022)
        self.assertEqual(ft.BIST_LAST_YEAR, 2022)
