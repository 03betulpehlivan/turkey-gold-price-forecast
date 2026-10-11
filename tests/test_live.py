"""
Canli sistem ve esanli seri testleri (LIVE_V1): src/data/sync_dataset.py, src/live/forecast.py
Calistirma: python -m unittest tests.test_live -v
Tamamen sentetik: internet, gercek data/ ve research/ kullanilmaz.
"""

from datetime import datetime, timezone
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

from src.data import collect_intraday as ci  # noqa: E402
from src.data import sync_dataset as sd  # noqa: E402
from src.live import forecast as lf  # noqa: E402
import tests.test_sync_prices as T  # noqa: E402

RETRIEVED = datetime(2026, 10, 1, tzinfo=timezone.utc)


def files_md5(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def snapshot(root, fetch=None, now=RETRIEVED):
    return Path(T.quiet(ci.collect_intraday, None, root, fetch or T.fake_fetch_factory(), now)["_path"])


def dropping_fetch(drop_utc_hours_by_date: dict):
    """Verilen NY tarihlerinde verilen UTC baslangic saatlerindeki ALTIN cubuklarini dusurur."""
    base = T.fake_fetch_factory()

    def f(ticker, period):
        df = base(ticker, period)
        if ticker != "GC=F" or not len(df):
            return df
        idx = df.index.tz_convert("UTC")
        keep = np.ones(len(df), bool)
        for d, hours in drop_utc_hours_by_date.items():
            keep &= ~((idx.normalize() == pd.Timestamp(d, tz="UTC")) & np.isin(idx.hour, hours))
        return df[keep]
    return f


class TestSyncDataset(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sd2_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "snaps"

    def daily(self, root=None):
        sd.run(root or self.root, self.tmp / "out")
        return sd.load_sync_daily(self.tmp / "out" / "sync_daily.csv")

    def test_merge_keeps_latest_and_counts_revisions(self):
        snapshot(self.root)
        base = T.fake_fetch_factory()

        def revised(ticker, period):
            df = base(ticker, period)
            if ticker == "USDTRY=X":
                df = df.copy()
                df.iloc[100:103, :5] = df.iloc[100:103, :5] * 1.001
            return df
        snapshot(self.root, revised, datetime(2026, 10, 2, tzinfo=timezone.utc))
        merged, rep = sd.merge_snapshots(self.root)
        self.assertEqual(rep["series"]["usdtry"]["revised_bars"], 3)
        self.assertEqual(rep["series"]["gold"]["revised_bars"], 0)
        self.assertTrue((merged["usdtry"].iloc[100:103]["snapshot"] == sorted(rep["snapshots"])[-1]).all())

    def test_gold_and_fx_from_the_same_bar(self):
        snapshot(self.root)
        d = self.daily()
        g = pd.read_csv(self.tmp / "out" / "hourly_gold.csv")
        f = pd.read_csv(self.tmp / "out" / "hourly_usdtry.csv")
        g = g.assign(ts=pd.to_datetime(g["ts_utc"], utc=True)).set_index("ts")["close"]
        f = f.assign(ts=pd.to_datetime(f["ts_utc"], utc=True)).set_index("ts")["close"]
        for _, r in d.sample(25, random_state=0).iterrows():
            self.assertAlmostEqual(r["gold_usd_oz"], g[r["bar_start_utc"]])
            self.assertAlmostEqual(r["usdtry"], f[r["bar_start_utc"]])
            self.assertEqual(r["bar_end_utc"] - r["bar_start_utc"], pd.Timedelta(hours=1))

    def test_new_york_dst_rule(self):
        snapshot(self.root)
        d = self.daily().set_index("Date")
        self.assertEqual(d.loc["2023-07-12", "bar_end_utc"], pd.Timestamp("2023-07-12 18:00", tz="UTC"))   # EDT
        self.assertEqual(d.loc["2023-12-13", "bar_end_utc"], pd.Timestamp("2023-12-13 19:00", tz="UTC"))   # EST
        # DST gecis haftalari: 2023-03-13 (yaz saatine gecis sonrasi) ve 2023-11-06 (kisa gecis sonrasi)
        self.assertEqual(d.loc["2023-03-13", "bar_end_utc"].hour, 18)
        self.assertEqual(d.loc["2023-11-06", "bar_end_utc"].hour, 19)
        ny = d["bar_end_utc"].dt.tz_convert("America/New_York")
        self.assertTrue((ny.dt.hour == 14).all())

    def test_missing_14_ny_bar_drops_the_day_never_falls_back(self):
        # 2023-07-12: yalnizca 13-14 NY cubugu (17:00 UTC baslangic) eksik -> gun atilir, 13:00'e dusulmez
        # 2023-07-13: 11:00 NY'den sonraki tum altin cubuklari eksik -> gun atilir
        snapshot(self.root, dropping_fetch({"2023-07-12": [17], "2023-07-13": list(range(15, 24))}))
        res = sd.run(self.root, self.tmp / "out")
        d = sd.load_sync_daily(self.tmp / "out" / "sync_daily.csv").set_index("Date")
        for day in ("2023-07-12", "2023-07-13"):
            self.assertNotIn(pd.Timestamp(day), d.index)
            self.assertIn(day, res["report"]["daily"]["dropped_dates"])
        self.assertIn(pd.Timestamp("2023-07-11"), d.index)

    def test_incomplete_bars_are_not_used(self):
        cut = datetime(2024, 6, 5, 17, 30, tzinfo=timezone.utc)      # 17:00-18:00 cubugu henuz kapanmamis
        snapshot(self.root, now=cut)
        merged, rep = sd.merge_snapshots(self.root)
        for n in ("gold", "usdtry"):
            self.assertLessEqual(merged[n]["ts_utc"].max() + pd.Timedelta(hours=1), pd.Timestamp(cut))
            self.assertGreater(rep["series"][n]["incomplete_bars_dropped"], 0)
        d = self.daily()
        self.assertLess(d["Date"].max(), pd.Timestamp("2024-06-05"))           # 5 Haziran'in 14:00 NY cubugu yok

    def test_archive_preserved_across_runs_and_rolling_window(self):
        snapshot(self.root)
        base = T.fake_fetch_factory()

        def rolled(ticker, period):                                 # Yahoo'nun en eski gunleri dusurmesi
            df = base(ticker, period)
            return df[df.index.tz_convert("UTC") >= pd.Timestamp("2024-01-01", tz="UTC")]
        snapshot(self.root, rolled, datetime(2026, 10, 2, tzinfo=timezone.utc))
        snaps_before = files_md5(self.root)
        d1 = self.daily()
        out1 = files_md5(self.tmp / "out")
        d2 = self.daily()
        self.assertEqual(files_md5(self.root), snaps_before)                 # goruntuler asla degismez
        self.assertEqual({k: v for k, v in files_md5(self.tmp / "out").items() if k != "sync_report.json"},
                         {k: v for k, v in out1.items() if k != "sync_report.json"})
        self.assertLess(d1["Date"].min(), pd.Timestamp("2023-04-01"))         # 2023 verisi eski goruntuden korunur
        pd.testing.assert_frame_equal(d1, d2)


# ---------------------------------------------------------------------------
def make_live_inputs(tmp: Path, end="2025-06-30", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", end)
    n = len(days)
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.9 * h[t - 1] + rng.normal(0, 0.2)
    gold = 1800 * np.exp(np.cumsum(rng.normal(0.0002, 0.009 * np.exp(h))))
    fx = 18 * np.exp(np.cumsum(rng.normal(0.001, 0.004, n)))
    start = (days + pd.Timedelta(hours=17)).tz_localize("UTC")
    end_utc = start + pd.Timedelta(hours=1)
    sync = pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "bar_start_utc": [x.isoformat() for x in start],
                         "bar_end_utc": [x.isoformat() for x in end_utc],
                         "bar_end_ny": [x.tz_convert("America/New_York").isoformat() for x in end_utc],
                         "gold_usd_oz": gold, "usdtry": fx, "theoretical_gram_try": gold * fx / 31.1035})
    pdays = pd.bdate_range("2015-01-01", end)
    m = len(pdays)
    panel = pd.DataFrame({"Date": pdays, "theoretical_gram_try": 100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, m))),
                          "gvz": 15 * np.exp(np.concatenate([np.zeros(m - n), h]))})
    sp, pp = tmp / "sync_daily.csv", tmp / "market_panel.csv"
    sync.to_csv(sp, index=False)
    panel.to_csv(pp, index=False, date_format="%Y-%m-%d")
    return sp, pp


ASOF_OK = pd.Timestamp("2025-06-30 20:00", tz="UTC")          # NY 16:00, ayni gun, cubuk 18:00 UTC'de kapandi


class TestLive(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lf2_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.sp, self.pp = make_live_inputs(self.tmp)
        self.log = self.tmp / "research" / "live" / "forecast_log.csv"

    def issue(self, now=ASOF_OK):
        return lf.issue(self.sp, self.pp, self.log, now_utc=now)

    def frame(self, panel=None):
        sync = sd.load_sync_daily(self.sp)
        panel = pd.read_csv(self.pp, parse_dates=["Date"]) if panel is None else panel
        return lf.build_frame(sync, panel)

    def test_forecast_order_and_basis(self):
        r = self.issue()["row"]
        self.assertTrue(r["q95_low"] < r["q80_low"] < r["point_price"] < r["q80_high"] < r["q95_high"])
        self.assertEqual(r["point_price_rw"], r["price_now"])
        self.assertIn("kuyumcu fiyati degil", r["price_basis"])

    def test_no_target_mixing_point_uses_sync_targets_only(self):
        df = self.frame()
        c = lf.compute(df)
        last_start = list(range(lf.MIN_TRAIN, len(df)))[-1 - ((len(df) - lf.MIN_TRAIN - 1) % 63)]
        self.assertAlmostEqual(c["mu"], float(df["y"].iloc[:last_start - 1].mean()), places=15)
        panel = pd.read_csv(self.pp, parse_dates=["Date"])
        panel["theoretical_gram_try"] *= np.exp(np.linspace(0, 3, len(panel)))   # eski seri tamamen degisir
        c2 = lf.compute(self.frame(panel))
        for k in ("mu", "mu_se", "sigma", "lo80", "hi80", "lo95", "hi95"):
            self.assertEqual(c[k], c2[k])

    def test_only_data_before_forecast_time(self):
        a = lf.compute(self.frame())
        panel = pd.read_csv(self.pp, parse_dates=["Date"])
        m = panel["Date"] >= pd.Timestamp("2025-06-30")          # veri gunu ve sonrasi GVZ
        panel.loc[m, "gvz"] *= 5
        b = lf.compute(self.frame(panel))
        for k in ("mu", "sigma", "lo80", "hi80", "lo95", "hi95"):
            self.assertAlmostEqual(a[k], b[k], places=12)
        df = self.frame()
        self.assertTrue((df["gvz_date"] < df["Date"]).all())
        self.assertTrue(np.isnan(df["y"].iat[-1]))              # bugunun hedefi bilinmiyor

    def test_issue_window_new_york_days(self):
        with self.assertRaises(lf.OutsideIssueWindow):
            self.issue(pd.Timestamp("2025-06-30 17:30", tz="UTC"))          # cubuk kapanmadan
        with self.assertRaises(lf.OutsideIssueWindow):
            self.issue(pd.Timestamp("2025-07-01 05:00", tz="UTC"))          # NY 01:00 ertesi gun
        self.issue(pd.Timestamp("2025-07-01 03:30", tz="UTC"))              # NY 23:30 ayni gun (TSI 06:30)
        self.assertEqual(len(lf.read_log(self.log)), 1)

    def test_issue_window_winter(self):
        bar_end = pd.Timestamp("2025-01-15 19:00", tz="UTC")                  # NY 14:00 EST
        lf.check_issue_window(pd.Timestamp("2025-01-15"), bar_end, pd.Timestamp("2025-01-16 04:30", tz="UTC"))  # NY 23:30
        with self.assertRaises(lf.OutsideIssueWindow):
            lf.check_issue_window(pd.Timestamp("2025-01-15"), bar_end, pd.Timestamp("2025-01-16 05:30", tz="UTC"))

    def test_duplicate_refused_and_scoring_keeps_forecast(self):
        first = self.issue()["row"]
        with self.assertRaises(lf.AlreadyIssued):
            self.issue()
        sync = pd.read_csv(self.sp)
        nxt = sync.iloc[[-1]].copy()
        nxt["Date"] = "2025-07-01"
        nxt["theoretical_gram_try"] = first["price_now"] * 1.004
        pd.concat([sync, nxt]).to_csv(self.sp, index=False)
        before = lf.read_log(self.log)[lf.FORECAST_COLS + ["row_hash"]].astype(str)
        log = lf.score(self.sp, self.log)
        self.assertTrue(log[lf.FORECAST_COLS + ["row_hash"]].astype(str).equals(before))
        self.assertEqual(log.iloc[0]["realized_date"], "2025-07-01")
        self.assertAlmostEqual(log.iloc[0]["rw_abs_err_TL"], first["price_now"] * 0.004)
        self.assertEqual(lf.scorecard(log)["n"], 1)

    def test_log_cannot_be_edited_afterwards(self):
        self.issue()
        text = self.log.read_text(encoding="utf-8")
        log = pd.read_csv(self.log)
        log.loc[0, "point_price"] = float(log.loc[0, "point_price"]) + 5.0      # sonradan duzenleme
        log.to_csv(self.log, index=False)
        tampered = self.log.read_bytes()
        with self.assertRaises(lf.LogTampered):
            lf.score(self.sp, self.log)
        self.assertEqual(self.log.read_bytes(), tampered)                       # hicbir sey yazilmadi
        with self.assertRaises(lf.LogTampered):
            lf.issue(self.sp, self.pp, self.log, now_utc=ASOF_OK)
        self.assertNotEqual(text, tampered.decode("utf-8"))

    def test_row_hash_survives_csv_roundtrip(self):
        self.issue()
        lf.verify_log(lf.read_log(self.log))

    def test_main_panel_not_used_by_default(self):
        self.assertEqual(lf.PANEL_PATH.parent.name, "live")
        self.assertNotEqual(lf.PANEL_PATH, PROJECT_ROOT / "data" / "processed" / "market_panel.csv")

    def test_insufficient_history_raises(self):
        sp, pp = make_live_inputs(self.tmp, end="2023-08-31")
        with self.assertRaisesRegex(ValueError, "Yetersiz"):
            lf.issue(sp, pp, self.log, now_utc=pd.Timestamp("2023-08-31 20:00", tz="UTC"))


def _hashes(folder: Path) -> dict:
    return files_md5(folder) if folder.exists() else {}


_REAL = [PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "research"]
_BEFORE = {}


def setUpModule():
    for d in _REAL:
        _BEFORE[d] = _hashes(d)


def tearDownModule():
    for d in _REAL:
        if _hashes(d) != _BEFORE[d]:
            raise AssertionError(f"Testler gercek klasoru degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
