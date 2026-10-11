"""
Kapanis saati teshisi testleri: src/analysis/fx_close_time.py
Calistirma: python -m unittest tests.test_fx_close_time -v
Sentetik saatlik veri test_sync_prices ile ortak; gunluk kapanislar bilinen saatlerden alinir.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import fx_close_time as fct  # noqa: E402
from src.data import collect_intraday as ci  # noqa: E402
import tests.test_sync_prices as T  # noqa: E402


class TestFxCloseTime(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fct_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.snap = Path(T.quiet(ci.collect_intraday, None, self.tmp / "s", T.fake_fetch_factory())["_path"])

    def test_recovers_known_close_hours(self):
        panel = T.make_panel(self.tmp / "p.csv", fx_close_hour_utc=13)
        res = fct.run(self.snap, panel, self.tmp / "o")
        fx, gold = res["series"]["usdtry"], res["series"]["gold"]
        self.assertEqual((fx["day_offset"], fx["hour_utc"]), (0, 13))
        self.assertEqual(gold["day_offset"], 0)
        self.assertIn(gold["hour_utc"], (21, 22))            # 17:00 New York (yaz/kis)

    def test_recovers_previous_day_close(self):
        days = pd.bdate_range("2023-03-02", "2025-12-30")
        rows = []
        for d in days:
            f_start = pd.Timestamp(d.date(), tz="UTC") - pd.Timedelta(hours=2)    # bitis: d-1 23:00 UTC
            g_start = (pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16)).tz_convert("UTC")
            if f_start in T.FX_PATH.index and g_start in T.GOLD_PATH.index:
                g, f = T.GOLD_PATH[g_start], T.FX_PATH[f_start]
                rows.append({"Date": d, "gold_usd_oz": g, "usdtry": f, "theoretical_gram_try": g * f / 31.1035})
        path = self.tmp / "p2.csv"
        pd.DataFrame(rows).to_csv(path, index=False, date_format="%Y-%m-%d")
        fx = fct.run(self.snap, path, self.tmp / "o2")["series"]["usdtry"]
        self.assertEqual((fx["day_offset"], fx["hour_utc"]), (-1, 23))

    def test_2026_not_used(self):
        panel = T.make_panel(self.tmp / "p.csv", fx_close_hour_utc=13)
        res = fct.run(self.snap, panel, self.tmp / "o")
        self.assertLess(pd.Timestamp(res["period"][1]), pd.Timestamp("2026-01-01"))
        prof = pd.read_csv(self.tmp / "o" / "profile_usdtry.csv")
        self.assertTrue((prof["days"] > 0).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
