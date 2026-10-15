"""
LIVE_V2 (GLD tabanli teorik canli tahmin) testleri: src/live/forecast_v2.py
Calistirma: python -m unittest tests.test_forecast_v2 -v
Sentetik dunya; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi). Ag erisimi yok.
"""

from pathlib import Path
from unittest import mock
import contextlib
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live import forecast as lf  # noqa: E402
from src.live import forecast_v2 as v2  # noqa: E402

NY = "America/New_York"
K_TRUE = 10.9


def ny(day, hh, mm=0):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {hh:02d}:{mm:02d}").tz_localize(NY).tz_convert("UTC")


def world(start="2025-01-02", n=420, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    gold = 2600 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    fx = 36 * np.exp(np.cumsum(rng.normal(0.0004, 0.002, n)))
    sync = pd.DataFrame({"Date": days, "bar_start_utc": [ny(d, 13) for d in days], "bar_end_utc": [ny(d, 14) for d in days],
                         "gold_usd_oz": gold, "usdtry": fx})
    sync["theoretical_gram_try"] = sync["gold_usd_oz"] * sync["usdtry"] / 31.1035
    bars = []
    for d, g in zip(days, gold):
        for h in range(9, 16):
            jitter = 1.0 if h == 13 else math.exp(rng.normal(0, 0.001))
            bars.append((ny(d, h, 30) - pd.Timedelta(hours=1), g / K_TRUE * jitter * math.exp(rng.normal(0, 0.0004))))
    hourly = pd.DataFrame(bars, columns=["ts_utc", "close"])
    panel = pd.DataFrame({"Date": days, "gvz": 18 + rng.normal(0, 1, n)})
    return sync, hourly, panel


def write_sync(path: Path, sync: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    sync.to_csv(path, index=False)


def write_gld_snapshot(root: Path, hourly: pd.DataFrame, name: str, retrieved: pd.Timestamp, status="ok"):
    snap = root / name
    snap.mkdir(parents=True)
    lines = ["Price,Adj Close,Close,High,Low,Open,Volume", "Ticker,GLD,GLD,GLD,GLD,GLD,GLD", "Datetime,,,,,,"]
    lines += [f"{t.isoformat()},{c},{c},{c},{c},{c},0" for t, c in zip(hourly["ts_utc"], hourly["close"])]
    (snap / "gld.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (snap / "manifest.json").write_text(json.dumps({"retrieved_at_utc": retrieved.isoformat(), "status": status,
                                                    "file": "gld.csv"}), encoding="utf-8")
    return snap


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for f in v2.CODE_FILES:
            (self.tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / f, self.tmp / f)
        self.sync, self.hourly, self.panel = world()
        self.last = self.sync["Date"].iat[-1]
        self.sync_path = self.tmp / "data" / "processed" / "sync" / "sync_daily.csv"
        write_sync(self.sync_path, self.sync.iloc[:-1])                    # son gun "yarin" icin saklanir
        self.gld_root = self.tmp / "data" / "raw" / "snapshots_gld"
        write_gld_snapshot(self.gld_root, self.hourly[self.hourly["ts_utc"] < ny(self.last, 0)],
                           "20260101T000000Z", ny(self.last, 0))
        self.panel_path = self.tmp / "data" / "processed" / "live" / "market_panel.csv"
        self.panel_path.parent.mkdir(parents=True)
        self.panel.to_csv(self.panel_path, index=False)
        p = mock.patch.multiple(v2, ROOT=self.tmp, GLD_ROOT=self.gld_root, SYNC_PATH=self.sync_path,
                                PANEL_PATH=self.panel_path,
                                FRAME_PATH=self.tmp / "data" / "processed" / "sync_v2" / "v2_daily.csv",
                                LOG_PATH=self.tmp / "research" / "live" / "forecast_v2_log.csv",
                                PREREG_PATH=self.tmp / "research" / "preregistration" / "live_v2.json",
                                REGISTRY_PATH=self.tmp / "research" / "registry.jsonl",
                                RESULT_PATH=self.tmp / "research" / "live" / "live_v2_result.json")
        p.start()
        self.addCleanup(p.stop)
        self.prev = self.sync["Date"].iat[-2]
        self.now = ny(self.prev, 15)

    def add_next_day(self):
        write_sync(self.sync_path, self.sync)
        write_gld_snapshot(self.gld_root, self.hourly, "20260102T000000Z", ny(self.last, 23))


class TestSeries(Base):
    def test_gld_bar_is_1330_ny_in_summer_and_winter(self):
        gd = v2.gld_daily(v2.merge_gld())
        self.assertEqual(len(gd), len(self.sync) - 1)
        end_ny = gd["gld_bar_end_utc"].dt.tz_convert(NY)
        self.assertTrue(((end_ny.dt.hour == 13) & (end_ny.dt.minute == 30)).all())
        summer = gd[gd["Date"] == pd.Timestamp("2025-07-01")]["gld_bar_end_utc"].iat[0]
        winter = gd[gd["Date"] == pd.Timestamp("2025-01-15")]["gld_bar_end_utc"].iat[0]
        self.assertEqual((summer.hour, summer.minute), (17, 30))
        self.assertEqual((winter.hour, winter.minute), (18, 30))

    def test_missing_1330_bar_drops_day(self):
        h = self.hourly[self.hourly["ts_utc"] != ny("2025-03-04", 12, 30)]
        gd = v2.gld_daily(h)
        self.assertNotIn(pd.Timestamp("2025-03-04"), set(gd["Date"]))

    def test_other_bars_in_same_hour_are_not_used(self):
        extra = pd.DataFrame({"ts_utc": [ny("2025-03-05", 12, 0), ny("2025-03-05", 13, 0)], "close": [1.0, 2.0]})
        h = pd.concat([self.hourly, extra]).sort_values("ts_utc").reset_index(drop=True)   # 13:00 ve 14:00 bitisli
        gd = v2.gld_daily(h).set_index("Date")
        want = float(self.hourly.loc[self.hourly["ts_utc"] == ny("2025-03-05", 12, 30), "close"].iat[0])
        self.assertEqual(float(gd.loc[pd.Timestamp("2025-03-05"), "gld_close"]), want)
        h2 = h[h["ts_utc"] != ny("2025-03-05", 12, 30)]                    # 13:30 yok, 13:00 bitisli var
        self.assertNotIn(pd.Timestamp("2025-03-05"), set(v2.gld_daily(h2)["Date"]))

    def test_merge_newest_wins_and_incomplete_dropped(self):
        d = self.prev
        bar = ny(d, 12, 30)
        h2 = self.hourly[self.hourly["ts_utc"] <= bar].copy()
        h2.loc[h2["ts_utc"] == bar, "close"] = 999.0
        write_gld_snapshot(self.gld_root, h2, "20260101T120000Z", ny(d, 23))
        h3 = h2.copy()
        h3.loc[h3["ts_utc"] == bar, "close"] = 1.0
        write_gld_snapshot(self.gld_root, h3, "20260101T130000Z", ny(d, 13))   # cubuk bitmeden indirildi
        m = v2.merge_gld()
        self.assertEqual(float(m.loc[m["ts_utc"] == bar, "close"].iat[0]), 999.0)

    def test_k_excludes_roll_windows_and_uses_last_60(self):
        gd = v2.gld_daily(self.hourly)
        k = v2.estimate_k(self.sync, gd)
        self.assertEqual(k["n"], 60)
        self.assertAlmostEqual(k["K"], K_TRUE, delta=0.05)
        bad = self.sync.copy()
        roll = bad["Date"].map(v2.le.in_roll_window)
        bad.loc[roll, "gold_usd_oz"] *= 5                                     # kontrat gunleri K'yi etkilemez
        self.assertEqual(v2.estimate_k(bad, gd)["K"], k["K"])
        self.assertTrue(roll.tail(80).any())

    def test_build_v2_formula(self):
        gd = v2.gld_daily(self.hourly)
        f = v2.build_v2(self.sync, gd, 11.0)
        np.testing.assert_allclose(f["gold_usd_oz"], f["gld_close"] * 11.0)
        np.testing.assert_allclose(f["theoretical_gram_try"], f["gold_usd_oz"] * f["usdtry"] / 31.1035)
        self.assertTrue((f["bar_end_utc"].dt.tz_convert(NY).dt.hour == 14).all())


class TestFlow(Base):
    def test_prepare_issue_score(self):
        with self.assertRaisesRegex(v2.Refused, "On kayit yok"):
            v2.issue(self.now)
        c = v2.prepare()
        self.assertAlmostEqual(c["K"]["K"], K_TRUE, delta=0.05)
        with self.assertRaisesRegex(v2.Refused, "zaten var"):
            v2.prepare()
        r = v2.issue(self.now)["row"]
        self.assertEqual((r["system_version"], r["asof_date"]), ("LIVE_V2", str(self.prev.date())))
        self.assertEqual(r["code_md5"], c["code_md5_combined"])
        log = v2.read_log()
        v2.verify_log(log)
        self.assertFalse((self.tmp / "research" / "live" / "forecast_log.csv").exists())   # LIVE_V1 kaydi yok
        # ayni model: forecast.compute ile birebir
        frame = pd.read_csv(v2.FRAME_PATH, parse_dates=["Date"])
        for col in ("bar_start_utc", "bar_end_utc"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
        cc = lf.compute(lf.build_frame(frame, self.panel))
        self.assertAlmostEqual(r["point_return"], cc["mu"], places=12)
        self.assertAlmostEqual(r["q80_low"], r["price_now"] * math.exp(cc["lo80"]), places=6)
        with self.assertRaises(lf.AlreadyIssued):
            v2.issue(self.now)
        self.add_next_day()
        log = v2.score()
        self.assertEqual(log["realized_date"].iat[0], str(self.last.date()))
        full = v2.build_v2(self.sync, v2.gld_daily(v2.merge_gld()), c["K"]["K"])
        real = float(full["theoretical_gram_try"].iat[-1])
        self.assertAlmostEqual(float(log["realized_price"].iat[0]), real, places=9)
        self.assertAlmostEqual(float(log["abs_err_TL"].iat[0]), abs(real - r["point_price"]), places=9)
        self.assertAlmostEqual(float(log["rw_abs_err_TL"].iat[0]), abs(real - r["price_now"]), places=9)
        self.assertEqual(float(log["in80"].iat[0]), float(r["q80_low"] <= real <= r["q80_high"]))
        for col in ("point_price", "q80_low", "q95_high", "row_hash"):
            self.assertEqual(str(log[col].iat[0]), str(v2.read_log()[col].iat[0]))
        v2.verify_log(v2.read_log())
        self.assertEqual(len(v2.score()), 1)                                  # ikinci puanlama degistirmez

    def test_issue_window_and_code_lock(self):
        v2.prepare()
        with self.assertRaises(lf.OutsideIssueWindow):
            v2.issue(ny(self.prev, 13, 50))                                  # 14:00 cubugu kapanmadan
        with self.assertRaises(lf.OutsideIssueWindow):
            v2.issue(ny(self.prev, 15) + pd.Timedelta(days=1))               # ertesi NY gunu
        f = self.tmp / "src" / "live" / "forecast.py"
        f.write_bytes(f.read_bytes() + b"# degisti\n")
        with self.assertRaisesRegex(v2.Refused, "Kod on kayittan farkli"):
            v2.issue(self.now)

    def test_tampering_detected(self):
        v2.prepare()
        v2.issue(self.now)
        log = v2.read_log()
        log.loc[0, "point_price"] = 1.0
        log.to_csv(v2.LOG_PATH, index=False)
        with self.assertRaises(lf.LogTampered):
            v2.issue(self.now)

    def test_fetch_writes_separate_root(self):
        seen = []

        def fake(ticker, period):
            seen.append(ticker)
            idx = pd.DatetimeIndex([ny("2026-10-15", 9, 30), ny("2026-10-15", 10, 30)])
            cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["GLD"]],
                                              names=["Price", "Ticker"])
            return pd.DataFrame(np.full((2, 6), 380.0), index=idx.rename("Datetime"), columns=cols)
        m = v2.fetch_gld(fetch=fake)
        self.assertEqual((m["status"], seen), ("ok", ["GLD"]))
        self.assertTrue((self.gld_root / m["snapshot_dir"] / "gld.csv").exists())
        self.assertNotIn("snapshots_intraday", str(v2.GLD_ROOT))


def fake_rows(n, pr, code=None, issued="2026-11-02T20:00:00+00:00", start="2026-11-02"):
    days = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        p = 6500.0
        real = p * math.exp(rng.normal(0, 0.01))
        row = {c: np.nan for c in v2.FORECAST_COLS}
        row.update({"issued_at_utc": issued, "system_version": "LIVE_V2", "asof_date": str(days[i].date()),
                    "asof_bar_end_utc": "x", "price_basis": v2.PRICE_BASIS, "price_now": p, "point_price": p,
                    "point_price_rw": p, "q80_low": p - 50, "q80_high": p + 50, "q95_low": p - 90, "q95_high": p + 90,
                    "code_commit": "abc", "sync_md5": "x", "gld_bar_end_utc": "x", "K": 10.9,
                    "code_md5": code or pr["code_md5_combined"]})
        row["row_hash"] = v2.row_hash(pd.Series(row))
        row.update({"realized_date": str(days[i + 1].date()), "realized_price": real,
                    "abs_err_TL": abs(real - p), "rw_abs_err_TL": abs(real - p),
                    "in80": float(abs(real - p) <= 50), "in95": float(abs(real - p) <= 90)})
        rows.append(row)
    return pd.DataFrame(rows)[v2.ALL_COLS]


class TestEvaluate(Base):
    def setUp(self):
        super().setUp()
        self.pr = v2.prepare()
        self.pr["created_at_utc"] = "2026-11-01T00:00:00+00:00"

    def test_eligibility(self):
        log = pd.concat([fake_rows(5, self.pr), fake_rows(2, self.pr, code="baska", start="2026-12-01"),
                         fake_rows(1, self.pr, issued="2026-10-20T20:00:00+00:00", start="2026-10-20")],
                        ignore_index=True)
        log.loc[0, "realized_price"] = np.nan
        s, r = v2.eligible(log, self.pr)
        self.assertEqual(len(s), 4)
        self.assertEqual(r, {"surum": 0, "puanlanmamis": 1, "on_kayittan_once": 1, "kod_farkli": 2})

    def test_evaluate_needs_n_and_runs(self):
        with self.assertRaisesRegex(v2.Refused, "120 uygun"):
            v2.evaluate(fake_rows(119, self.pr), self.pr)
        rows = fake_rows(125, self.pr)
        r = v2.evaluate(rows.iloc[::-1].reset_index(drop=True), self.pr)
        self.assertEqual(r["n"], 120)
        self.assertEqual((r["first_asof"], r["last_asof"]), (rows["asof_date"].iat[0], rows["asof_date"].iat[119]))
        self.assertEqual(r["primary"]["point_verdict"], "PRATIKTE_ESDEGER")      # ayni hata
        self.assertIn("A2_christoffersen_80", r["primary"])
        self.assertGreaterEqual(r["secondary_roll_excluded"]["dropped"], 1)

    def test_main_evaluate_one_shot(self):
        json.loads(v2.PREREG_PATH.read_text(encoding="utf-8"))
        pr = json.loads(v2.PREREG_PATH.read_text(encoding="utf-8"))
        pr["created_at_utc"] = "2026-11-01T00:00:00+00:00"
        v2.PREREG_PATH.write_text(json.dumps(pr), encoding="utf-8")
        v2.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fake_rows(120, pr).to_csv(v2.LOG_PATH, index=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(v2.main(["status"]), 0)
            self.assertEqual(v2.main(["evaluate"]), 0)
            self.assertEqual(v2.main(["evaluate"]), 2)
        self.assertIn("uygun puanlanmis: 120 / 120", buf.getvalue())
        self.assertTrue(v2.RESULT_PATH.exists())
        ev = [json.loads(l)["event"] for l in v2.REGISTRY_PATH.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(ev, ["prepared", "evaluated"])


class TestScripts(unittest.TestCase):
    """PowerShell CALISTIRILMAZ; betik metni sinanir."""

    def setUp(self):
        self.v2s = (PROJECT_ROOT / "scripts" / "teorik_v2.ps1").read_text(encoding="utf-8")
        self.wrap = (PROJECT_ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")

    def test_v2_script_touches_only_v2_log(self):
        self.v2s.encode("ascii")
        code = "\n".join(l for l in self.v2s.splitlines() if not l.lstrip().startswith("#"))
        self.assertIn('$kayit = "research/live/forecast_v2_log.csv"', code)
        gits = [l.strip() for l in code.splitlines() if l.strip().startswith("git ")]
        self.assertTrue(gits and all("$kayit" in g for g in gits), gits)
        self.assertNotIn("git push", code)
        self.assertNotIn("forecast_log.csv", code.replace("forecast_v2_log.csv", ""))
        steps = re.findall(r'"src\.live\.forecast_v2", "(\w+)"', code)
        self.assertEqual(steps, ["fetch", "score", "issue"])
        self.assertIn('"issue") @(0, 3, 4)', code)

    def test_wrapper_runs_v2_only_in_evening_without_touching_exit_code(self):
        start = self.wrap.index("teorik_v2.ps1")
        self.assertLess(self.wrap.index("$kod = $LASTEXITCODE"), start)
        block = self.wrap[self.wrap.rindex('if ($Gorev -eq "aksam")', 0, start):self.wrap.index("# Veri yedegi")]
        self.assertNotIn("$kod", block)
        self.assertNotIn("exit", block)
        self.assertIn("catch {", block)


_REAL = [PROJECT_ROOT / "research", PROJECT_ROOT / "data"]
_BEFORE = {}


def _h(folder):
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()} if folder.exists() else {}


def setUpModule():
    for d in _REAL:
        _BEFORE[d] = _h(d)


def tearDownModule():
    for d in _REAL:
        if _h(d) != _BEFORE[d]:
            raise AssertionError(f"Testler gercek klasoru degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
