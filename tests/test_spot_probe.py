"""
Spot ons taramasi testleri: src/analysis/spot_probe.py
Calistirma: python -m unittest tests.test_spot_probe -v
Sentetik: spot fiyat + GC=F (tasima primi ve iki kontrat sicramasi). Internet ve gercek data/ kullanilmaz.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import hashlib
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

from src.analysis import spot_probe as sp_  # noqa: E402
from tests.test_sync_prices import yf_hourly, quiet  # noqa: E402

IDX = pd.date_range("2025-06-02", "2026-08-31", freq="1h", tz="UTC")
IDX = IDX[IDX.dayofweek < 5]
RNG = np.random.default_rng(3)
SPOT = pd.Series(3300 * np.exp(np.cumsum(RNG.normal(0, 0.002, len(IDX)))), index=IDX)
carry = np.full(len(IDX), 1.004)
for d in sp_.ROLL_DAYS:                                   # kontrat degisimi: kalici +%1.7 seviye
    carry[IDX >= pd.Timestamp(d, tz="UTC")] *= 1.017
GC = SPOT * carry
GLD_IDX = IDX[(IDX.minute == 0)] + pd.Timedelta(minutes=30)
GLD_IDX = GLD_IDX[(GLD_IDX.hour >= 13) & (GLD_IDX.hour <= 19)]
GLD = SPOT.reindex(GLD_IDX - pd.Timedelta(minutes=30)).to_numpy() * 0.092


def fetch(ticker, period):
    if ticker == "XAUUSD=X":
        return yf_hourly(ticker, IDX, SPOT.values, "Europe/London")
    if ticker in ("GLD", "IAU"):
        return yf_hourly(ticker, GLD_IDX, GLD * (1 if ticker == "GLD" else 0.5), "America/New_York")
    if ticker == "GC=F":
        return yf_hourly(ticker, IDX, GC.values, "America/New_York")
    raise ValueError(ticker)


def write_sync(path: Path):
    end = IDX + pd.Timedelta(hours=1)
    ny = end.tz_convert("America/New_York")
    sel = (ny.hour == 14) & (ny.minute == 0)
    s, e = IDX[sel], end[sel]
    g = GC[sel].to_numpy()
    df = pd.DataFrame({"Date": ny[sel].strftime("%Y-%m-%d"), "bar_start_utc": [x.isoformat() for x in s],
                       "bar_end_utc": [x.isoformat() for x in e],
                       "bar_end_ny": [x.tz_convert("America/New_York").isoformat() for x in e],
                       "gold_usd_oz": g, "usdtry": 40.0, "theoretical_gram_try": g * 40 / 31.1035})
    df.to_csv(path, index=False)
    return path


class TestSpotProbe(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="spot_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.sync = write_sync(self.tmp / "sync_daily.csv")
        self.now = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def probe(self, candidates=None):
        with mock.patch.dict(sp_.CANDIDATES, candidates or {}, clear=bool(candidates)):
            snap = sp_.download(self.tmp / "probe", fetch, self.now)
            return sp_.run(snap, self.sync, self.tmp / "out")

    def test_spot_sources_without_roll_jumps_are_eligible(self):
        s = self.probe()
        for n in ("xauusd", "gld", "iau"):
            r = s["results"][n]
            self.assertTrue(r["roll_ok"], n)
            self.assertTrue(all(z >= sp_.MIN_Z for z in r["roll_day_z"].values()), n)
            self.assertGreater(r["corr_non_roll"], 0.95, n)
            self.assertTrue(r["eligible"], n)
        self.assertIn(s["decision"]["selected"], ("xauusd", "gld", "iau"))

    def test_futures_copy_with_jumps_is_rejected(self):
        s = self.probe({"kotu": "GC=F"})
        r = s["results"]["kotu"]
        self.assertFalse(r["roll_ok"])
        self.assertFalse(r["eligible"])
        self.assertIsNone(s["decision"]["selected"])
        self.assertIn("gecici", s["decision"]["decision"])

    def test_etf_bars_use_nearest_half_hour(self):
        s = self.probe()
        self.assertEqual(set(s["results"]["gld"]["offset_min_counts"]), {"-30"})        # 13:30 NY bitisi
        self.assertEqual(set(s["results"]["xauusd"]["offset_min_counts"]), {"0"})

    def test_download_isolated_and_no_incomplete_bars(self):
        before = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in self.tmp.rglob("*") if p.is_file()}
        snap = sp_.download(self.tmp / "probe", fetch, self.now)
        self.assertTrue(str(snap).startswith(str(self.tmp / "probe")))
        after = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in self.tmp.rglob("*") if p.is_file()}
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)
        m = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({e["status"] for e in m["series"].values()}, {"ok"})

    def test_low_correlation_candidate_rejected_even_without_jumps(self):
        rng = np.random.default_rng(9)
        days = pd.bdate_range("2025-06-02", "2026-08-31")
        r = rng.normal(0, 0.01, len(days))
        jump = np.isin(days, pd.to_datetime(sp_.ROLL_DAYS)) * 0.03
        gc = pd.DataFrame({"Date": days, "gold_usd_oz": 3300 * np.exp(np.cumsum(r + jump))})
        cand = pd.DataFrame({"Date": days, "close": 3300 * np.exp(np.cumsum(0.5 * r + rng.normal(0, 0.003, len(days)))),
                             "offset_min": 0})
        res = sp_.evaluate(cand, gc)
        self.assertTrue(res["roll_ok"])                       # sicrama yok
        self.assertLess(res["corr_non_roll"], 0.95)
        self.assertFalse(res["eligible"])                     # ama korelasyon dusuk

    def test_incomplete_bars_at_retrieval_are_excluded(self):
        self.now = datetime(2026, 8, 28, 17, 30, tzinfo=timezone.utc)   # son gun (28 Agu) 14:00 NY cubugu kapanmamis
        s = self.probe({"xauusd": "XAUUSD=X"})
        n_sync = len(pd.read_csv(self.sync))
        self.assertEqual(s["results"]["xauusd"]["common_days"], n_sync - 1)

    def test_missing_candidate_reported(self):
        def flaky(t, p):
            if t == "XAUUSD=X":
                return pd.DataFrame()
            return fetch(t, p)
        with mock.patch.dict(sp_.CANDIDATES, {"xauusd": "XAUUSD=X", "gld": "GLD"}, clear=True):
            snap = sp_.download(self.tmp / "probe", flaky, self.now)
            s = sp_.run(snap, self.sync, self.tmp / "out")
        self.assertEqual(s["results"]["xauusd"]["note"], "veri yok")
        self.assertEqual(s["decision"]["selected"], "gld")


if __name__ == "__main__":
    unittest.main(verbosity=2)
