"""
Canli aralik eklentisi testleri: src/live/physical_intervals.py (LIVE_PHYS_INT_V1)
Calistirma: python -m unittest tests.test_physical_intervals -v
Sentetik dunya; once LIVE_PHYS_V1 tahmini, sonra araliklar. Gercek data/ ve research/ kullanilmaz.
"""

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

from src.live import physical_forecast as lp  # noqa: E402
from src.live import physical_intervals as pi  # noqa: E402
from tests.test_physical_eval import make_world  # noqa: E402

T = pd.Timestamp("2016-07-01")
NOW = pd.Timestamp("2016-07-01 07:00", tz="UTC")


def world():
    phys, panel = make_world("bagli", seed=51, end="2016-06-30")
    days = list(panel["Date"]) + [T]
    fx = list(panel["usdtry"]) + [float(panel["usdtry"].iat[-1]) * 1.001]
    fx_h = pd.DataFrame({"ts_utc": [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days], "close": fx})
    return phys, panel, fx_h


class TestIntervals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="pi_"))
        cls.p = [mock.patch.object(lp, "TRAIN_START", "2014-02-01"),
                 mock.patch.object(lp, "REGISTRY_PATH", cls.tmp / "registry.jsonl"),
                 mock.patch.object(pi, "REGISTRY_PATH", cls.tmp / "registry.jsonl")]
        for p in cls.p:
            p.start()
        cls.phys, cls.panel, cls.fx = world()
        cls.v1_log, cls.v1_pre = cls.tmp / "v1.csv", cls.tmp / "v1.json"
        cls.log, cls.pre = cls.tmp / "int.csv", cls.tmp / "int.json"
        lp.prepare(cls.v1_pre)
        pi.prepare(cls.pre)
        lp.issue(cls.phys, cls.panel, cls.fx, NOW, cls.v1_log, cls.v1_pre)
        cls.calls = []
        real = lp.forecast

        def spy(phys, panel, fx_h, now):
            cls.calls.append((pd.Timestamp(now.date()), phys["Date"].max(), panel["Date"].max()))
            return real(phys, panel, fx_h, now)
        with mock.patch.object(lp, "forecast", spy):
            cls.row = pi.issue(cls.phys, cls.panel, cls.fx, NOW, lp.read_log(cls.v1_log), cls.log, cls.pre)

    @classmethod
    def tearDownClass(cls):
        for p in cls.p:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_bounds_nested_and_centered_on_v1(self):
        r, v1 = self.row, lp.read_log(self.v1_log).iloc[0]
        for m in pi.MODELS:
            self.assertEqual(r[f"pred_{m}"], float(v1[f"pred_{m}"]))
            self.assertTrue(r[f"lo95_{m}"] < r[f"lo80_{m}"] < r[f"pred_{m}"] < r[f"hi80_{m}"] < r[f"hi95_{m}"])
        self.assertGreaterEqual(r["n_cal"], pi.MIN_CAL)
        self.assertEqual(r["v1_forecast_id"], "LIVE_PHYS_V1-2016-07-01")

    def test_calibration_uses_only_past_published_days(self):
        cal_calls = [c for c in self.calls if c[0] < T]
        self.assertTrue(cal_calls)
        for D, pmax, nmax in cal_calls:
            self.assertLess(pmax, D)                                   # fiziki seri D'den once kesik
            self.assertLess(nmax, D)                                   # panel (ons) D'den once kesik
        self.assertLess(pd.Timestamp(self.row["cal_last"]), T)

    def test_days_without_hourly_bar_are_skipped(self):
        fx = self.fx[~self.fx["ts_utc"].isin([pd.Timestamp("2016-06-15 04:00", tz="UTC")])]
        cal = pi.calibration(self.phys, self.panel, fx, T)
        self.assertNotIn(pd.Timestamp("2016-06-15"), set(cal["target_date"]))

    def test_refusals(self):
        with self.assertRaisesRegex(pi.Refused, "zaten kayitli"):
            pi.issue(self.phys, self.panel, self.fx, NOW, lp.read_log(self.v1_log), self.log, self.pre)
        with self.assertRaisesRegex(pi.Refused, "V1 tahmini yok"):
            pi.issue(self.phys, self.panel, self.fx, NOW, lp.read_log(self.v1_log).iloc[0:0], self.tmp / "x.csv", self.pre)
        bad = lp.read_log(self.v1_log).copy()
        bad.loc[0, "pred_Bosluk"] = float(bad.loc[0, "pred_Bosluk"]) + 1
        with self.assertRaisesRegex(pi.Refused, "ayni degil"):
            pi.issue(self.phys, self.panel, self.fx, NOW, bad, self.tmp / "y.csv", self.pre)
        with self.assertRaisesRegex(pi.Refused, "On kayit yok"):
            pi.issue(self.phys, self.panel, self.fx, NOW, lp.read_log(self.v1_log), self.tmp / "z.csv", self.tmp / "yok.json")
        with self.assertRaisesRegex(pi.Refused, "Yetersiz kalibrasyon"):
            pi.bounds(pd.DataFrame(columns=["target_date"]), {})

    def test_score_and_tamper(self):
        log_copy = self.tmp / "int_copy.csv"
        shutil.copy(self.log, log_copy)
        real = self.row["pred_Bosluk"]
        phys2 = pd.concat([self.phys, pd.DataFrame({"Date": [T], "kapanis_tl_gr": [real]})], ignore_index=True)
        log = pi.score(phys2, log_copy)
        self.assertEqual(float(log["in80_Bosluk"].iat[0]), 1.0)
        self.assertEqual(pi.card(log)["scored"], 1)
        d = pd.read_csv(log_copy)
        d.loc[0, "lo80_Bosluk"] = 1.0
        d.to_csv(log_copy, index=False)
        with self.assertRaises(pi.LogTampered):
            pi.score(phys2, log_copy)

    def test_prepare_locked_and_evaluate(self):
        self.assertEqual(pi.prepare(self.pre), "zaten hazir")
        with mock.patch.object(pi, "N_CAL", 400):
            with self.assertRaisesRegex(pi.Refused, "degistirilemez"):
                pi.prepare(self.pre)
        md5 = json.loads(self.pre.read_text())["code_md5"]
        rng = np.random.default_rng(0)
        n = 130
        a = 100 * np.exp(rng.normal(0, 0.01, n))
        log = pd.DataFrame({"target_date": pd.bdate_range("2017-01-02", periods=n).strftime("%Y-%m-%d"),
                            "code_md5": [md5] * n, "realized_price": a})
        for m in pi.MODELS:
            log[f"lo80_{m}"], log[f"hi80_{m}"] = 100 * np.exp(-0.0128), 100 * np.exp(0.0128)
            log[f"lo95_{m}"], log[f"hi95_{m}"] = 100 * np.exp(-0.0196), 100 * np.exp(0.0196)
        with self.assertRaisesRegex(pi.Refused, "120"):
            pi.evaluate(log.head(50), {"code_md5": md5})
        ev = pi.evaluate(log, {"code_md5": md5})
        self.assertEqual(ev["n"], 120)
        self.assertTrue(0.7 < ev["results"]["Bosluk_80"]["coverage"] < 0.9)
        self.assertTrue(ev["results"]["Bosluk_80"]["calibrated"])

    def test_code_md5_covers_dependencies_and_is_eol_independent(self):
        a = pi.code_md5()
        src = Path(pi.ist.__file__).read_bytes()
        with mock.patch.object(Path, "read_bytes", lambda self: src.replace(b"\n", b"\r\n") if self.name == "interval_study.py"
                               else open(self, "rb").read()):
            self.assertEqual(pi.code_md5(), a)                         # CRLF donusumu ozeti degistirmez
        with mock.patch.object(Path, "read_bytes", lambda self: src + b"# degisti\n" if self.name == "interval_study.py"
                               else open(self, "rb").read()):
            self.assertNotEqual(pi.code_md5(), a)                      # bagimli dosya degisirse ozet degisir


_REAL = [PROJECT_ROOT / "research", PROJECT_ROOT / "data" / "processed"]
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
