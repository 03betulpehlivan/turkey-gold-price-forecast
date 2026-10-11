"""
Teshis analizi testleri: src/analysis/data_diagnostics.py
=========================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_diagnostics -v

En onemli test: son sinav (2026+) ve test donemi verisi bozuldugunda
ongorulebilirlik tanilari HIC degismemeli. Gercek data/ klasoru kullanilmaz.
"""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import data_diagnostics as dd  # noqa: E402
from src.models import train as tr  # noqa: E402


def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def make_panel(start="2015-01-02", end="2026-06-30", seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    walk = lambda s, v, m=0.0: s * np.exp(np.cumsum(rng.normal(m, v, n)))  # noqa: E731
    p = pd.DataFrame({"Date": days, "gold_usd_oz": walk(1200, 0.009), "usdtry": walk(2.3, 0.005, 0.0008)})
    p["theoretical_gram_try"] = p["gold_usd_oz"] * p["usdtry"] / 31.1035
    for name, lvl in (("silver", 17), ("dxy", 90), ("us10y", 2.2), ("vix", 15), ("sp500", 2000),
                      ("bist100", 800), ("brent", 55), ("eurtry", 2.8)):
        p[name] = walk(lvl, 0.01)
        p[f"{name}_age_days"] = 0.0
    return p


def write_snapshot(root: Path) -> Path:
    """Bir Cumartesi cubugu iceren kucuk bir anlik goruntu."""
    snap = root / "20261011T000000Z"
    snap.mkdir(parents=True)
    dates = ["2022-01-03", "2022-01-04", "2022-01-08", "2022-01-10", "2026-02-07"]  # 01-08 ve 2026-02-07 Cumartesi
    lines = ["Price,Adj Close,Close,High,Low,Open,Volume", "Ticker," + ",".join(["USDTRY=X"] * 6), "Date,,,,,,"]
    lines += [f"{d},13.0,13.0,13.1,12.9,13.0,0" for d in dates]
    (snap / "usdtry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (snap / "manifest.json").write_text(json.dumps({"series": {"usdtry": {
        "ticker": "USDTRY=X", "status": "ok", "file": "usdtry.csv", "md5": "x"}}}), encoding="utf-8")
    return snap


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class TestStatistics(unittest.TestCase):

    def test_acf_matches_manual_formula(self):
        x = np.random.default_rng(1).normal(size=500)
        d = x - x.mean()
        manual = [float(d[k:] @ d[:-k] / (d @ d)) for k in range(1, 6)]
        np.testing.assert_allclose(dd.acf(x, 5), manual)

    def test_acf_detects_ar1(self):
        rng = np.random.default_rng(2)
        x = np.zeros(20000)
        for i in range(1, len(x)):
            x[i] = 0.5 * x[i - 1] + rng.normal()
        self.assertAlmostEqual(dd.acf(x, 1)[0], 0.5, delta=0.03)

    def test_cross_corr_sign_convention(self):
        x = pd.Series(np.random.default_rng(3).normal(size=1000))
        y = x.shift(1)                      # y bugun = x dun; y, x'ten sonra gelir
        self.assertAlmostEqual(dd.cross_corr(x, y, 1)[0], 1.0, places=10)
        self.assertLess(abs(dd.cross_corr(x, y, -1)[0]), 0.1)

    def test_roll_window_mask(self):
        dates = pd.Series(pd.bdate_range("2022-01-01", "2022-12-31"))
        m = dd.roll_window_mask(dates)
        self.assertEqual(int(m.sum()), 6 * dd.ROLL_LAST_N)
        jan = dates[(dates.dt.month == 1)]
        self.assertTrue(m[jan.index[-5:]].all())
        self.assertFalse(m[jan.index[:-5]].any())
        self.assertFalse(m[dates.dt.month == 2].any())

    def test_welch_detects_mean_shift(self):
        rng = np.random.default_rng(4)
        r = dd.welch(rng.normal(0.5, 1, 400), rng.normal(0, 1, 400))
        self.assertGreater(r["t"], 3)
        self.assertLess(r["p_normal"], 0.01)

    def test_mde_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "significance.csv"
            pd.DataFrame({"period": ["test", "test"], "model": ["RF", "Zero-Return"],
                          "reference": ["Mean-Return", "Mean-Return"], "n": [548, 548],
                          "mean_abs_err_diff": [-1e-5, 2e-4], "dm_stat": [-0.5, 2.0],
                          "p_value": [0.6, 0.05]}).to_csv(path, index=False)
            out = dd.d1_min_detectable_effect(path)
        self.assertAlmostEqual(out["se"].iloc[0], 2e-5)
        self.assertAlmostEqual(out["mde_80"].iloc[0], 5.6e-5)

    def test_constants_match_training_protocol(self):
        self.assertEqual(pd.Timestamp(dd.ANALYSIS_END) + pd.Timedelta(days=1), pd.Timestamp(tr.TEST_START))


class TestPeriodProtection(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="diag_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.panel = make_panel()

    def run_diag(self, panel: pd.DataFrame, name: str) -> dict:
        path = self.tmp / f"{name}.csv"
        panel.to_csv(path, index=False, date_format="%Y-%m-%d")
        return quiet(dd.run, path, None, self.tmp / "yok.csv", self.tmp / f"out_{name}")

    def test_holdout_rows_are_never_used(self):
        a = self.run_diag(self.panel, "orijinal")
        bad = self.panel.copy()
        m = bad["Date"] >= pd.Timestamp(dd.HOLDOUT_START)
        bad.loc[m, ["gold_usd_oz", "usdtry", "theoretical_gram_try", "silver"]] *= 7.0
        b = self.run_diag(bad, "bozuk_holdout")
        for key in ("d2_effect", "d2_top", "d3", "d4", "d5", "d6"):
            with self.subTest(key=key):
                pd.testing.assert_frame_equal(a[key], b[key])
        self.assertGreater(a["summary"]["holdout_rows_not_analyzed"], 0)
        self.assertLess(pd.Timestamp(a["summary"]["data_quality_period"][1]), pd.Timestamp(dd.HOLDOUT_START))

    def test_predictability_diagnostics_ignore_test_period(self):
        a = self.run_diag(self.panel, "orijinal")
        bad = self.panel.copy()
        m = bad["Date"] > pd.Timestamp(dd.ANALYSIS_END)
        bad.loc[m, ["gold_usd_oz", "usdtry", "dxy", "silver", "eurtry"]] *= np.linspace(1, 3, int(m.sum()))[:, None]
        bad["theoretical_gram_try"] = bad["gold_usd_oz"] * bad["usdtry"] / 31.1035
        b = self.run_diag(bad, "bozuk_test")
        for key in ("d3", "d4", "d5", "d6"):
            with self.subTest(key=key):
                pd.testing.assert_frame_equal(a[key], b[key])
        self.assertLessEqual(a["d6"]["Date"].max(), pd.Timestamp(dd.ANALYSIS_END))

    def test_outputs_written_only_to_output_dir(self):
        path = self.tmp / "panel.csv"
        self.panel.to_csv(path, index=False, date_format="%Y-%m-%d")
        before = file_hashes(self.tmp)
        out = self.tmp / "out"
        quiet(dd.run, path, None, self.tmp / "yok.csv", out)
        self.assertEqual(file_hashes(self.tmp), before)
        for f in ("d3_autocorrelation.csv", "d4_cross_correlation.csv", "d5_calendar_gaps.csv",
                  "d2_roll_window_effect.csv", "d6_rolling_drift.csv", "summary.json"):
            self.assertTrue((out / f).exists(), f)

    def test_weekend_bars_counted_before_holdout_only(self):
        snap = write_snapshot(self.tmp / "snaps")
        out = dd.d5b_weekend_bars(snap)
        row = out.set_index("series").loc["usdtry"]
        self.assertEqual((row["saturday"], row["sunday"], row["rows"]), (1, 0, 4))
        self.assertEqual(row["years"], "2022")

    def test_target_and_gap_definitions(self):
        ret = dd.add_returns(self.panel.head(10))
        p = self.panel["theoretical_gram_try"]
        self.assertAlmostEqual(ret["target"].iloc[3], math.log(p.iloc[4] / p.iloc[3]), places=14)
        self.assertEqual(ret["gap_days"].iloc[4], (self.panel["Date"].iloc[5] - self.panel["Date"].iloc[4]).days)

    def test_missing_panel_exits_1(self):
        code = quiet(dd.main, ["--panel", str(self.tmp / "yok.csv"), "--out", str(self.tmp / "o")])
        self.assertEqual(code, 1)


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
