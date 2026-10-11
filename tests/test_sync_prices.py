"""
Esanli fiyat analizi testleri: src/data/collect_intraday.py, src/analysis/sync_prices.py
=======================================================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_sync_prices -v

Sentetik saatlik veri: altin cubuklari New York, kur cubuklari Londra saat
diliminde gelir (yfinance'e benzer). Gunluk kapanislar bilinen saatlerden
alinir; boylece analiz, kaymayi bilinen bir dogruya karsi olcer.
Gercek data/ klasoru kullanilmaz.
"""

from datetime import datetime, timezone
from pathlib import Path
import contextlib
import hashlib
import io
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

from src.analysis import sync_prices as sp  # noqa: E402
from src.data import collect_intraday as ci  # noqa: E402
from src.models import train as tr  # noqa: E402


def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def hourly_index(start="2023-03-01", end="2026-02-27"):
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    return idx[idx.dayofweek < 5]


IDX = hourly_index()
RNG = np.random.default_rng(5)
GOLD_PATH = pd.Series(1900 * np.exp(np.cumsum(RNG.normal(0, 0.002, len(IDX)))), index=IDX)
FX_PATH = pd.Series(19 * np.exp(np.cumsum(RNG.normal(0.00005, 0.001, len(IDX)))), index=IDX)


def yf_hourly(ticker, idx_utc, closes, tz):
    idx = pd.DatetimeIndex(idx_utc).tz_convert(tz)
    idx.name = "Datetime"
    c = np.asarray(closes, float)
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], [ticker]],
                                      names=["Price", "Ticker"])
    return pd.DataFrame(np.column_stack([c, c, c, c, c, np.zeros(len(c))]), index=idx, columns=cols)


def fake_fetch_factory(fx_shift_minutes=0):
    def fetch(ticker, period):
        if ticker == "GC=F":
            return yf_hourly(ticker, IDX, GOLD_PATH.values, "America/New_York")
        if period == "max":            # kur icin 'max' bos donsun: geri dusus test edilir
            return pd.DataFrame()
        shifted = IDX + pd.Timedelta(minutes=fx_shift_minutes)
        return yf_hourly(ticker, shifted, FX_PATH.values, "Europe/London")
    return fetch


def make_panel(path: Path, fx_close_hour_utc=13, start="2023-03-01", end="2026-02-26"):
    """Gunluk panel: altin kapanisi = bitisi 17:00 NY olan cubuk; kur kapanisi = bitisi fx_close_hour_utc olan cubuk."""
    days = pd.bdate_range(start, end)
    rows = []
    for d in days:
        g_end = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=17)
        g_start = (g_end - pd.Timedelta(hours=1)).tz_convert("UTC")
        if fx_close_hour_utc is None:
            f_start = g_start
        else:
            f_start = pd.Timestamp(d.date(), tz="UTC") + pd.Timedelta(hours=fx_close_hour_utc - 1)
        if g_start not in GOLD_PATH.index or f_start not in FX_PATH.index:
            continue
        g, f = GOLD_PATH[g_start], FX_PATH[f_start]
        rows.append({"Date": d, "gold_usd_oz": g, "usdtry": f, "theoretical_gram_try": g * f / 31.1035})
    pd.DataFrame(rows).to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def snapshot(self, fx_shift=0):
        return Path(quiet(ci.collect_intraday, None, self.tmp / "snaps", fake_fetch_factory(fx_shift))["_path"])


class TestCollector(Base):

    def test_fallback_attempts_are_recorded_and_timestamps_converted(self):
        snap = self.snapshot()
        m = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["series"]["gold"]["period_used"], "max")
        self.assertEqual(m["series"]["usdtry"]["period_used"], "730d")
        self.assertEqual(m["series"]["usdtry"]["attempts"][0], {"period": "max", "rows": 0})
        self.assertEqual(m["series"]["gold"]["index_tz"], "America/New_York")
        g = ci.read_intraday_csv(snap / "gold.csv", "GC=F")
        self.assertEqual(str(g["ts_utc"].dt.tz), "UTC")
        self.assertEqual(g["ts_utc"].iloc[0], IDX[0])
        raw = (snap / "gold.csv").read_text(encoding="utf-8").splitlines()
        self.assertTrue(raw[2].startswith("Datetime"))
        self.assertIn("-05:00", raw[3])    # ham zaman damgasi ofsetiyle korunur

    def test_reader_rejects_wrong_ticker_and_bad_header(self):
        snap = self.snapshot()
        with self.assertRaises(ci.IntradayError):
            ci.read_intraday_csv(snap / "gold.csv", "SI=F")
        bad = self.tmp / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaises(ci.IntradayError):
            ci.read_intraday_csv(bad, "GC=F")

    def test_snapshots_never_overwritten(self):
        now = datetime(2026, 10, 11, tzinfo=timezone.utc)
        a = Path(quiet(ci.collect_intraday, None, self.tmp / "s", fake_fetch_factory(), now)["_path"])
        b = Path(quiet(ci.collect_intraday, None, self.tmp / "s", fake_fetch_factory(), now)["_path"])
        self.assertNotEqual(a, b)


class TestAnalysisUnits(Base):

    def setUp(self):
        super().setUp()
        _, self.hourly = sp.load_intraday(self.snapshot())

    def test_coverage_counts_by_period(self):
        cov = sp.coverage(self.hourly)
        g = cov["gold"]
        self.assertEqual(g["bars"], len(IDX))
        self.assertEqual(g["bars_development"] + g["bars_test_period"] + g["bars_2026"], len(IDX))
        self.assertGreater(g["bars_2026"], 0)
        self.assertEqual(g["minute_offsets"], {"0": len(IDX)})
        self.assertEqual(cov["common_exact_timestamps"]["count"], len(IDX))

    def test_only_identical_timestamps_are_paired(self):
        _, shifted = sp.load_intraday(self.snapshot(fx_shift=30))
        self.assertEqual(len(sp.synchronous_pairs(shifted["gold"], shifted["usdtry"])), 0)
        pairs = sp.synchronous_pairs(self.hourly["gold"], self.hourly["usdtry"])
        self.assertTrue((pairs["gram"] == pairs["gold"] * pairs["usdtry"] / 31.1035).all())

    def test_daily_rule_respects_new_york_dst(self):
        pairs = sp.synchronous_pairs(self.hourly["gold"], self.hourly["usdtry"])
        daily = sp.daily_from_pairs(pairs, 17).set_index("Date")
        summer, winter = pd.Timestamp("2023-07-12"), pd.Timestamp("2023-12-13")
        self.assertEqual(daily.loc[summer, "ts_utc"], pd.Timestamp("2023-07-12 20:00", tz="UTC"))
        self.assertEqual(daily.loc[winter, "ts_utc"], pd.Timestamp("2023-12-13 21:00", tz="UTC"))
        self.assertTrue((daily["bar_end_ny"].dt.hour <= 17).all())

    def test_close_hour_inference_recovers_known_hours(self):
        panel = pd.read_csv(make_panel(self.tmp / "p.csv", fx_close_hour_utc=13), parse_dates=["Date"])
        panel = panel[panel["Date"] < "2023-10-26"]
        g = sp.summarize_match(sp.close_hour_match(panel, "gold_usd_oz", self.hourly["gold"]))
        f = sp.summarize_match(sp.close_hour_match(panel, "usdtry", self.hourly["usdtry"]))
        self.assertGreater(g["share_within_1bp"], 0.99)
        self.assertEqual(set(g["hour_utc_distribution_1bp"]), {"21", "22"})   # 17:00 NY, yaz/kis
        self.assertEqual(f["most_common_hour_utc_among_1bp_matches"], 13)


class TestRun(Base):

    def run_sync(self, fx_close_hour_utc, include_test=False, panel_kwargs=None):
        snap = self.snapshot()
        panel = make_panel(self.tmp / "panel.csv", fx_close_hour_utc, **(panel_kwargs or {}))
        out = self.tmp / "out"
        return quiet(sp.run, snap, panel, out, include_test), out

    def test_synchronous_daily_data_gives_zero_difference(self):
        res, _ = self.run_sync(fx_close_hour_utc=None)
        p = res["summary"]["comparison"]["ny_1700"]["prices"]
        self.assertGreater(p["days_compared"], 100)
        self.assertLess(p["abs_diff_pct_max"], 1e-9)
        self.assertAlmostEqual(res["summary"]["comparison"]["ny_1700"]["returns"]["gram_return_corr_daily_vs_sync"], 1.0)

    def test_asynchronous_daily_data_is_detected(self):
        res, out = self.run_sync(fx_close_hour_utc=13)
        p = res["summary"]["comparison"]["ny_1700"]["prices"]
        self.assertGreater(p["abs_diff_pct_mean"], 0.01)
        for f in ("summary.json", "sync_vs_daily_ny_1700.csv", "sync_vs_daily_ny_1400.csv",
                  "top_deviations_ny_1700.csv", "daily_close_hour_match_gold.csv",
                  "daily_close_hour_match_usdtry.csv", "hour_profile.csv"):
            self.assertTrue((out / f).exists(), f)

    def test_default_uses_development_period_only(self):
        res, _ = self.run_sync(fx_close_hour_utc=13)
        rng = res["summary"]["pairs_range"]
        self.assertLess(pd.Timestamp(rng[1]), pd.Timestamp(tr.TEST_START, tz="UTC"))
        self.assertFalse(res["summary"]["include_test_period"])

    def test_test_and_2026_prices_cannot_change_default_results(self):
        res_a, out_a = self.run_sync(fx_close_hour_utc=13)
        global GOLD_PATH
        saved = GOLD_PATH.copy()
        try:
            GOLD_PATH.loc[GOLD_PATH.index >= pd.Timestamp(tr.TEST_START, tz="UTC")] *= 3.0
            self.tmp = Path(tempfile.mkdtemp(prefix="sync_test_b_"))
            self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
            res_b, out_b = self.run_sync(fx_close_hour_utc=13)
        finally:
            GOLD_PATH[:] = saved
        self.assertEqual(res_a["summary"]["comparison"], res_b["summary"]["comparison"])

    def test_include_test_never_uses_2026(self):
        res, out = self.run_sync(fx_close_hour_utc=13, include_test=True)
        self.assertLess(pd.Timestamp(res["summary"]["pairs_range"][1]), pd.Timestamp(sp.HOLDOUT_START, tz="UTC"))
        merged = pd.read_csv(out / "sync_vs_daily_ny_1700.csv", parse_dates=["Date"])
        self.assertLess(merged["Date"].max(), pd.Timestamp(sp.HOLDOUT_START))
        self.assertGreater(merged["Date"].max(), pd.Timestamp(tr.TEST_START))

    def test_no_development_data_reports_and_exits_3(self):
        global IDX, GOLD_PATH, FX_PATH
        saved = IDX, GOLD_PATH, FX_PATH
        try:
            late = IDX[IDX >= pd.Timestamp("2024-10-14", tz="UTC")]
            IDX, GOLD_PATH, FX_PATH = late, GOLD_PATH[late], FX_PATH[late]
            snap = self.snapshot()
            panel = make_panel(self.tmp / "panel.csv", 13, start="2024-10-14")
            out = self.tmp / "out"
            code = quiet(sp.main, ["--snapshot", str(snap), "--panel", str(panel), "--out", str(out)])
        finally:
            IDX, GOLD_PATH, FX_PATH = saved
        self.assertEqual(code, sp.EXIT_NO_DATA)
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["status"], "no_data_in_period")
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["summary.json"])
        self.assertGreater(s["coverage"]["common_exact_timestamps"]["test_period"], 0)


class TestBoundariesAndCounts(Base):
    """2026 her iki modda disarida; sinir cubuk KAPANIS anina gore; eslesme sayilari tutarli."""

    def run_mode(self, include_test, drop_panel_dates=()):
        snap = self.snapshot()
        panel_path = make_panel(self.tmp / "panel.csv", 13)
        if drop_panel_dates:
            p = pd.read_csv(panel_path)
            p[~p["Date"].isin(drop_panel_dates)].to_csv(panel_path, index=False)
        out = self.tmp / ("out_test" if include_test else "out_dev")
        return quiet(sp.run, snap, panel_path, out, include_test), out

    def test_boundary_bar_is_filtered_by_close_time(self):
        df = pd.DataFrame({"ts_utc": pd.to_datetime(["2025-12-31 21:00", "2025-12-31 22:00", "2025-12-31 23:00",
                                                     "2026-01-01 00:00"], utc=True), "close": [1.0, 2.0, 3.0, 4.0]})
        kept = sp.bars_closed_before(df, pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(list(kept["close"]), [1.0, 2.0])   # 23:00 cubugu 2026-01-01 00:00'da kapanir: disarida

    def test_2026_excluded_in_both_modes(self):
        for include_test, limit in ((False, tr.TEST_START), (True, sp.HOLDOUT_START)):
            with self.subTest(include_test=include_test):
                res, out = self.run_mode(include_test)
                s = res["summary"]
                self.assertLess(pd.Timestamp(s["last_bar_close_utc"]), pd.Timestamp(limit, tz="UTC"))
                self.assertLess(pd.Timestamp(s["last_included_date"]), pd.Timestamp(limit))
                self.assertLess(pd.Timestamp(s["last_included_date"]), pd.Timestamp(sp.HOLDOUT_START))
                for name in ("gold", "usdtry"):
                    mt = pd.read_csv(out / f"daily_close_hour_match_{name}.csv")
                    self.assertLess(pd.to_datetime(mt["matched_bar_end_utc"], utc=True).max(),
                                    pd.Timestamp(limit, tz="UTC"))
                merged = pd.read_csv(out / "sync_vs_daily_ny_1700.csv", parse_dates=["Date"])
                self.assertLess(merged["Date"].max(), pd.Timestamp(limit))

    def test_last_included_date_is_explicit(self):
        self.assertEqual(sp.analysis_window(True)[1], pd.Timestamp("2025-12-31"))
        self.assertEqual(sp.analysis_window(False)[1], pd.Timestamp(tr.TEST_START) - pd.Timedelta(days=1))

    def test_match_counts_are_consistent_and_unmatched_dates_listed(self):
        dropped = ["2024-03-05", "2024-07-10", "2025-02-12"]
        res, out = self.run_mode(True, drop_panel_dates=dropped)
        for rule in sp.RULES:
            with self.subTest(rule=rule):
                p = res["summary"]["comparison"][rule]["prices"]
                self.assertEqual(p["days_compared"] + p["panel_days_without_sync_in_window"],
                                 p["panel_days_in_coverage_window"])
                self.assertEqual(p["unmatched_dates"]["sync_only"], dropped)
                self.assertEqual(p["sync_days_without_panel"], 3)
                merged = pd.read_csv(out / f"sync_vs_daily_{rule}.csv")
                self.assertEqual(len(merged), p["days_compared"])
                um = pd.read_csv(out / f"unmatched_dates_{rule}.csv")
                self.assertEqual(sorted(um.loc[um["side"] == "sync_only", "Date"]), dropped)

    def test_panel_days_before_hourly_coverage_are_not_counted_as_missing(self):
        # Gercek durum: panel 2015'ten, saatlik veri 2024'ten baslar. Kapsam oncesi gunler "eksik" sayilmamali.
        snap = self.snapshot()
        panel_path = make_panel(self.tmp / "panel.csv", 13)
        early = pd.DataFrame({"Date": pd.bdate_range("2022-01-03", "2022-12-30").strftime("%Y-%m-%d"),
                              "gold_usd_oz": 1800.0, "usdtry": 15.0, "theoretical_gram_try": 1800 * 15 / 31.1035})
        pd.concat([early, pd.read_csv(panel_path)]).to_csv(panel_path, index=False)
        res = quiet(sp.run, snap, panel_path, self.tmp / "o", True)
        p = res["summary"]["comparison"]["ny_1700"]["prices"]
        self.assertEqual(p["panel_days_without_sync_in_window"], 0)
        self.assertEqual(p["panel_days_without_sync_total"], len(early))
        self.assertEqual(p["days_compared"], p["panel_days_in_coverage_window"])

    def test_unique_match_detects_ambiguous_slow_series(self):
        hourly = {"usdtry": pd.DataFrame({"ts_utc": IDX[:2000], "close": np.full(2000, 19.0)})}
        panel = pd.DataFrame({"Date": pd.bdate_range("2023-03-02", periods=40), "usdtry": 19.0})
        m = sp.summarize_match(sp.close_hour_match(panel, "usdtry", hourly["usdtry"]))
        self.assertEqual(m["share_within_1bp"], 1.0)
        self.assertEqual(m["share_unique_1bp"], 0.0)          # her saat esit: saat belirlenemez
        self.assertIsNone(m["most_common_hour_utc_among_unique"])


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
