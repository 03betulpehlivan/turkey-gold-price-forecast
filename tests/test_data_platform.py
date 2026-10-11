"""
Veri platformu testleri: src/data/collect.py ve src/data/build_panel.py
=====================================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_data_platform -v

Ag kullanilmaz: Yahoo yerine yfinance ciktisi bicimini taklit eden sahte bir
indirici kullanilir. Gercek data/ klasoru okunmaz ve yazilmaz; modul sonunda
md5 ile dogrulanir.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock
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

from src.data import build_dataset as bd  # noqa: E402
from src.data import build_panel as bp  # noqa: E402
from src.data import collect as co  # noqa: E402


# ---------------------------------------------------------------------------
# Sahte veri ve indirici
# ---------------------------------------------------------------------------
def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs), buf.getvalue()


def yf_frame(ticker: str, dates, closes, tz=None) -> pd.DataFrame:
    """yfinance.download(auto_adjust=False, multi_level_index=True) bicimi."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    if tz:
        idx = idx.tz_localize(tz) + pd.Timedelta(hours=5)
    closes = np.asarray(closes, dtype=float)
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], [ticker]],
                                      names=["Price", "Ticker"])
    data = np.column_stack([closes, closes, closes * 1.01, closes * 0.99, closes, np.zeros(len(closes))])
    return pd.DataFrame(data, index=idx, columns=cols)


DAYS = pd.bdate_range("2022-01-03", "2022-06-30")
RNG = np.random.default_rng(11)


def fake_series():
    """Takvim farklari, gec baslayan seri ve uzun tatil iceren sahte veri."""
    n = len(DAYS)
    walk = lambda s, v: s * np.exp(np.cumsum(RNG.normal(0, v, n)))  # noqa: E731
    data = {
        "GC=F": (DAYS.delete([10, 50]), None),
        "USDTRY=X": (DAYS.delete([20]), None),
        "SI=F": (DAYS[30:], None),                                   # gec basliyor
        "DX-Y.NYB": (DAYS, None),
        "^TNX": (DAYS, None),
        "^VIX": (DAYS.delete(range(60, 63)), None),                  # 3 gunluk bosluk
        "^GSPC": (DAYS, None),
        "XU100.IS": (DAYS.delete(range(80, 88)), None),              # ~12 takvim gunu tatil
        "BZ=F": (DAYS, None),
        "EURTRY=X": (DAYS, None),
        "^GVZ": (DAYS, None),
    }
    levels = {"GC=F": 1800, "USDTRY=X": 13.5, "SI=F": 23, "DX-Y.NYB": 96, "^TNX": 1.6,
              "^VIX": 18, "^GSPC": 4700, "XU100.IS": 2000, "BZ=F": 80, "EURTRY=X": 15.3,
              "^GVZ": 15.0}
    out = {}
    for t, (d, _) in data.items():
        full = pd.Series(walk(levels[t], 0.01), index=DAYS)
        out[t] = yf_frame(t, d, full[d].to_numpy())
    return out


FAKE = fake_series()


def fake_fetch(ticker, start, end):
    df = FAKE[ticker]
    idx = df.index
    return df[(idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))]


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dp_test_"))
        self.snaps = self.tmp / "raw" / "snapshots"
        self.processed = self.tmp / "processed"
        self.processed.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_collect(self, names=None, fetch=fake_fetch, now=None, start="2022-01-01", end="2022-07-01"):
        names = names or list(co.SERIES)
        return quiet(co.collect, names, start, end, self.snaps, fetch, now)[0]


# ---------------------------------------------------------------------------
# collect.py
# ---------------------------------------------------------------------------
class TestCollect(TempDir):

    def test_snapshot_files_and_manifest(self):
        m = self.run_collect(["gold", "usdtry", "vix"])
        snap = Path(m["_path"])
        saved = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["request"], {"start": "2022-01-01", "end_exclusive": "2022-07-01",
                                            "interval": "1d", "auto_adjust": False})
        for name in ("gold", "usdtry", "vix"):
            e = saved["series"][name]
            self.assertEqual(e["status"], "ok")
            self.assertEqual(e["md5"], hashlib.md5((snap / e["file"]).read_bytes()).hexdigest())
            self.assertEqual(e["rows_valid"], len(FAKE[e["ticker"]]))

    def test_raw_header_matches_existing_raw_format(self):
        snap = Path(self.run_collect(["gold"])["_path"])
        lines = (snap / "gold.csv").read_text(encoding="utf-8").splitlines()[:3]
        self.assertEqual(lines, ["Price,Adj Close,Close,High,Low,Open,Volume",
                                 "Ticker,GC=F,GC=F,GC=F,GC=F,GC=F,GC=F", "Date,,,,,,"])
        parsed, _ = quiet(bd.read_yahoo_csv, snap / "gold.csv", "GC=F")
        np.testing.assert_allclose(parsed["close"], FAKE["GC=F"][("Close", "GC=F")].to_numpy())

    def test_snapshots_are_never_overwritten(self):
        now = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        first = Path(self.run_collect(["gold"], now=now)["_path"])
        before = file_hashes(first)
        second = Path(self.run_collect(["gold"], now=now)["_path"])
        self.assertNotEqual(first, second)
        self.assertEqual(second.name, first.name + "_2")
        self.assertEqual(file_hashes(first), before)

    def test_one_failure_does_not_stop_others(self):
        def flaky(ticker, start, end):
            if ticker == "^VIX":
                raise ConnectionError("baglanti koptu")
            if ticker == "DX-Y.NYB":
                return pd.DataFrame()
            return fake_fetch(ticker, start, end)
        m = self.run_collect(["gold", "vix", "dxy", "usdtry"], fetch=flaky)
        st = {n: e["status"] for n, e in m["series"].items()}
        self.assertEqual(st, {"gold": "ok", "vix": "error", "dxy": "empty", "usdtry": "ok"})
        self.assertIn("ConnectionError", m["series"]["vix"]["error"])
        self.assertFalse((Path(m["_path"]) / "vix.csv").exists())

    def test_wrong_ticker_in_file_is_invalid(self):
        m = self.run_collect(["gold"], fetch=lambda t, s, e: fake_fetch("SI=F", s, e))
        self.assertEqual(m["series"]["gold"]["status"], "invalid")

    def test_timezone_aware_index_is_normalized_and_recorded(self):
        def tzfetch(ticker, start, end):
            df = fake_fetch(ticker, start, end)
            return yf_frame(ticker, df.index, df[("Close", ticker)].to_numpy(), tz="UTC")
        m = self.run_collect(["gold"], fetch=tzfetch)
        e = m["series"]["gold"]
        self.assertEqual(e["status"], "ok")
        self.assertEqual(e["index_tz"], "UTC")
        self.assertTrue(e["index_had_time"])
        self.assertEqual(e["first_date"], str(FAKE["GC=F"].index[0].date()))

    def test_default_end_excludes_today(self):
        self.assertEqual(co.default_end(date(2026, 10, 11)), "2026-10-11")

    def test_argument_validation(self):
        for argv in (["--series", "gold,yok"], ["--start", "2022-02-01", "--end", "2022-01-01"],
                     ["--start", "01/02/2022"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    co.parse_args(argv)

    def test_main_exit_codes(self):
        base = ["--start", "2022-01-01", "--end", "2022-07-01", "--snapshot-root", str(self.snaps)]

        def no_vix(t, s, e):
            if t == "^VIX":
                raise RuntimeError("yok")
            return fake_fetch(t, s, e)

        def no_gold(t, s, e):
            if t == "GC=F":
                raise RuntimeError("yok")
            return fake_fetch(t, s, e)
        with mock.patch.object(co, "fetch_yahoo", no_vix):
            self.assertEqual(quiet(co.main, base + ["--series", "gold,usdtry,vix"])[0], 0)
        with mock.patch.object(co, "fetch_yahoo", no_gold):
            self.assertEqual(quiet(co.main, base + ["--series", "gold,usdtry"])[0], 1)


# ---------------------------------------------------------------------------
# build_panel.py
# ---------------------------------------------------------------------------
class TestPanel(TempDir):

    def setUp(self):
        super().setUp()
        self.snap = Path(self.run_collect()["_path"])
        self.manifest, self.series = bp.load_snapshot(self.snap)
        self.panel, self.cov = bp.build_panel(self.series)

    def test_base_calendar_and_formula_match_build_dataset(self):
        (theo, _), _ = quiet(bd.build, self.series["gold"], self.series["usdtry"])
        pd.testing.assert_frame_equal(self.panel[bp.BASE_COLUMNS].reset_index(drop=True),
                                      theo[bp.BASE_COLUMNS].reset_index(drop=True))
        common = sorted(set(FAKE["GC=F"].index) & set(FAKE["USDTRY=X"].index))
        self.assertEqual(list(self.panel["Date"]), common)

    def test_asof_uses_latest_past_value_only(self):
        vix = self.series["vix"].set_index("Date")["close"]
        for _, row in self.panel.iterrows():
            past = vix[vix.index <= row["Date"]]
            if not np.isnan(row["vix"]):
                self.assertEqual(row["vix"], past.iloc[-1])
                self.assertEqual(row["vix_age_days"], (row["Date"] - past.index[-1]).days)
        self.assertTrue((self.panel["vix_age_days"].dropna() >= 0).all())

    def test_series_starting_later_is_empty_before_start(self):
        first = self.series["silver"]["Date"].iloc[0]
        self.assertTrue(self.panel.loc[self.panel["Date"] < first, "silver"].isna().all())
        self.assertEqual(self.cov["silver"]["rows_before_series_start"],
                         int((self.panel["Date"] < first).sum()))

    def test_long_holiday_values_are_not_carried_beyond_limit(self):
        age = self.panel["bist100_age_days"]
        self.assertLessEqual(age.max(), bp.MAX_STALENESS_DAYS)
        self.assertGreater(self.cov["bist100"]["rows_dropped_stale"], 0)
        self.assertEqual(int(self.panel["bist100"].isna().sum()),
                         self.cov["bist100"]["rows_dropped_stale"] + self.cov["bist100"]["rows_before_series_start"])

    def test_truncation_invariance(self):
        for t in (self.panel["Date"].iloc[25], self.panel["Date"].iloc[70], self.panel["Date"].iloc[-2]):
            with self.subTest(t=str(t.date())):
                cut = {n: s[s["Date"] <= t] for n, s in self.series.items()}
                small, _ = bp.build_panel(cut)
                a = self.panel[self.panel["Date"] == t].reset_index(drop=True)
                b = small[small["Date"] == t].reset_index(drop=True)
                pd.testing.assert_frame_equal(a, b)

    def test_missing_optional_series_gives_empty_columns(self):
        series = {k: v for k, v in self.series.items() if k != "brent"}
        panel, cov = bp.build_panel(series)
        self.assertTrue(panel["brent"].isna().all())
        self.assertNotEqual(cov["brent"]["status"], "ok")

    def test_empty_series_does_not_crash(self):
        series = dict(self.series)
        series["silver"] = series["silver"].iloc[:0]
        panel, cov = bp.build_panel(series)
        self.assertTrue(panel["silver"].isna().all())
        self.assertNotEqual(cov["silver"]["status"], "ok")

    def test_age_columns_have_stable_float_type(self):
        for name in bp.EXTRA_SERIES:
            with self.subTest(name=name):
                self.assertEqual(self.panel[f"{name}_age_days"].dtype, np.float64)
                self.assertEqual(self.panel[name].dtype, np.float64)

    def test_tampered_snapshot_file_is_rejected(self):
        p = self.snap / "vix.csv"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(bp.PanelError, "md5"):
            bp.load_snapshot(self.snap)

    def test_latest_snapshot_skips_failed_required(self):
        def no_gold(t, s, e):
            if t == "GC=F":
                raise RuntimeError("yok")
            return fake_fetch(t, s, e)
        self.run_collect(fetch=no_gold, now=datetime(2099, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(bp.find_latest_snapshot(self.snaps), self.snap)

    def test_compare_with_existing_theoretical_file(self):
        path = self.processed / "theoretical_gold.csv"
        self.panel[bp.BASE_COLUMNS].to_csv(path, index=False, date_format="%Y-%m-%d")
        self.assertEqual(bp.compare_with_existing(self.panel, path)["status"], "AYNI")
        changed = self.panel[bp.BASE_COLUMNS].copy()
        changed.loc[5, "usdtry"] += 0.01
        changed.to_csv(path, index=False, date_format="%Y-%m-%d")
        res = bp.compare_with_existing(self.panel, path)
        self.assertEqual(res["status"], "FARKLI")
        self.assertEqual(res["usdtry_rows_diff_gt_1e-9"], 1)


class TestPanelMain(TempDir):

    def setUp(self):
        super().setUp()
        self.run_collect()
        # Mevcut proje dosyalarini taklit et: panel bunlara dokunmamali
        (self.processed / "model_dataset.csv").write_text("Date,x\n2022-01-03,1\n", encoding="utf-8")
        (self.processed / "theoretical_gold.csv").write_text(
            "Date,gold_usd_oz,usdtry,theoretical_gram_try\n2022-01-03,1800,13.5,781.2\n", encoding="utf-8")
        self.argv = ["--snapshot-root", str(self.snaps), "--processed-dir", str(self.processed)]
        self.protected = {k: v for k, v in file_hashes(self.processed).items()}

    def test_default_mode_writes_nothing(self):
        code, out = quiet(bp.main, self.argv)
        self.assertEqual(code, 0)
        self.assertEqual(file_hashes(self.processed), self.protected)
        self.assertIn("hicbir dosya yazilmadi", out)

    def test_write_then_idempotent_then_force(self):
        self.assertEqual(quiet(bp.main, self.argv + ["--write"])[0], 0)
        panel_path = self.processed / bp.PANEL_FILE
        self.assertTrue(panel_path.exists())
        report = json.loads((self.processed / bp.REPORT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(report["max_staleness_days"], bp.MAX_STALENESS_DAYS)
        h = hashlib.md5(panel_path.read_bytes()).hexdigest()
        code, out = quiet(bp.main, self.argv + ["--write"])
        self.assertEqual(code, 0)
        self.assertIn("zaten ayni", out)
        self.assertEqual(hashlib.md5(panel_path.read_bytes()).hexdigest(), h)
        # Panel degisirse --force gerekir
        code, _ = quiet(bp.main, self.argv + ["--write", "--max-staleness", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(hashlib.md5(panel_path.read_bytes()).hexdigest(), h)
        code, _ = quiet(bp.main, self.argv + ["--write", "--force", "--max-staleness", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(len(list(self.processed.glob("market_panel.backup_*.csv"))), 1)
        # Mevcut proje dosyalari hic degismedi
        for name, digest in self.protected.items():
            self.assertEqual(hashlib.md5((self.processed / name).read_bytes()).hexdigest(), digest)

    def test_no_valid_snapshot_exits_1(self):
        argv = ["--snapshot-root", str(self.tmp / "bos"), "--processed-dir", str(self.processed)]
        self.assertEqual(quiet(bp.main, argv)[0], 1)


# ---------------------------------------------------------------------------
# Gercek proje verisinin korunmasi
# ---------------------------------------------------------------------------
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
