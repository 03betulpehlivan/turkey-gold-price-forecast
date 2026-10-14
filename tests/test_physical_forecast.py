"""
Canli sabah fiziki tahmini testleri: src/live/physical_forecast.py (LIVE_PHYS_V1)
Calistirma: python -m unittest tests.test_physical_forecast -v
Sentetik dunya test_physical_eval ile ortak; EVDS sahte istemciyle. Gercek data/ ve research/ kullanilmaz.
"""

from pathlib import Path
from unittest import mock
import hashlib
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

from src.data import evds  # noqa: E402
from src.live import physical_forecast as lp  # noqa: E402
from tests.test_physical_eval import make_world  # noqa: E402

T = pd.Timestamp("2024-07-01")                                  # Pazartesi
NOW = pd.Timestamp("2024-07-01 07:00", tz="UTC")                # 10:00 TSI


def world():
    phys, panel = make_world("bagli", seed=31, end="2024-06-28")
    days = list(panel["Date"]) + [T]
    fx = list(panel["usdtry"]) + [float(panel["usdtry"].iat[-1]) * 1.002]
    fx_h = pd.DataFrame({"ts_utc": [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days], "close": fx})
    return phys, panel, fx_h


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lp_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.phys, self.panel, self.fx = world()
        self.log = self.tmp / "research" / "live" / "physical_log.csv"
        self.prereg = self.tmp / "research" / "preregistration" / "live_phys_v1.json"
        p = mock.patch.multiple(lp, REGISTRY_PATH=self.tmp / "research" / "registry.jsonl")
        p.start()
        self.addCleanup(p.stop)
        lp.prepare(self.prereg)

    def issue(self, phys=None, panel=None, fx=None, now=NOW):
        return lp.issue(self.phys if phys is None else phys, self.panel if panel is None else panel,
                        self.fx if fx is None else fx, now, self.log, self.prereg)


class TestIssue(Base):

    def test_forecast_fields_and_models(self):
        r = self.issue()
        self.assertEqual((r["target_date"], r["last_phys_date"], r["horizon_days"]), ("2024-07-01", "2024-06-28", 1))
        self.assertEqual(r["gold_src_date"], "2024-06-28")
        self.assertAlmostEqual(r["pred_RW"], float(self.phys["kapanis_tl_gr"].iat[-1]))
        self.assertAlmostEqual(r["pred_Prim-Capa"], r["M_morning"] * math.exp(r["prim_ewma"]))
        self.assertAlmostEqual(r["gap"], math.log(r["M_morning"]) - math.log(r["last_phys_price"]))
        self.assertGreater(r["beta_full"], 0.7)                         # bagli dunyada gece hareketi yansir
        self.assertEqual(r["code_md5"], lp.code_md5())

    def test_refusals(self):
        cases = {
            "Erken": dict(now=pd.Timestamp("2024-07-01 04:30", tz="UTC")),
            "Gec": dict(now=pd.Timestamp("2024-07-01 11:30", tz="UTC")),
            "Hafta sonu": dict(now=pd.Timestamp("2024-06-29 07:00", tz="UTC")),
            "saatlik arsivde yok": dict(fx=self.fx.iloc[:-1]),
            "EVDS guncel degil": dict(phys=self.phys[self.phys["Date"] < "2024-06-24"]),
            "zaten yayimlanmis": dict(phys=pd.concat([self.phys, self.phys.tail(1).assign(Date=T)])),
            "panel guncel degil": dict(panel=self.panel[self.panel["Date"] < "2024-06-27"],
                                       phys=self.phys),
        }
        for msg, kw in cases.items():
            with self.subTest(msg):
                with self.assertRaisesRegex(lp.Refused, msg):
                    self.issue(**kw)
        self.issue()
        with self.assertRaisesRegex(lp.Refused, "zaten kayitli"):
            self.issue()
        with self.assertRaisesRegex(lp.Refused, "On kayit yok"):
            lp.issue(self.phys, self.panel, self.fx, NOW, self.tmp / "baska.csv", self.tmp / "yok.json")

    def test_training_never_sees_target_or_future(self):
        tr_, _ = lp.training_frame(self.phys, self.panel, T)
        self.assertLess(pd.to_datetime(tr_["target_date"]).max(), T)
        self.assertEqual(lp.TRAIN_START, "2011-01-01")
        later = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03"])
        phys2 = pd.concat([self.phys, self.phys.tail(3).assign(Date=later, kapanis_tl_gr=1.0)], ignore_index=True)
        panel2 = pd.concat([self.panel, self.panel.tail(3).assign(Date=later)], ignore_index=True)   # kur/ons da var
        tr2, _ = lp.training_frame(phys2, panel2, T)
        pd.testing.assert_frame_equal(tr_, tr2)                        # hedef gun ve sonrasi egitime girmez

    def test_evening_gold_of_target_day_not_used(self):
        panel2 = pd.concat([self.panel, self.panel.tail(1).assign(Date=T, gold_usd_oz=9999.0)], ignore_index=True)
        r = self.issue(panel=panel2)
        self.assertEqual(r["gold_src_date"], "2024-06-28")
        self.assertNotEqual(r["gold_usd"], 9999.0)


class TestScoreAndLog(Base):

    def test_score_fills_and_keeps_forecast(self):
        r = self.issue()
        real = r["pred_RW"] * 1.01
        phys2 = pd.concat([self.phys, pd.DataFrame({"Date": [T], "kapanis_tl_gr": [real]})], ignore_index=True)
        before = lp.read_log(self.log)[lp.FORECAST_COLS + ["row_hash"]].astype(str)
        log = lp.score(phys2, self.log)
        self.assertTrue(log[lp.FORECAST_COLS + ["row_hash"]].astype(str).equals(before))
        self.assertAlmostEqual(float(log["abs_err_RW"].iat[0]), real - r["pred_RW"])
        c = lp.card(log)
        self.assertEqual((c["scored"], c["pending"]), (1, 0))
        self.assertIn("vs_RW_pct", c["Bosluk"])

    def test_pending_not_scored(self):
        self.issue()
        log = lp.score(self.phys, self.log)
        self.assertEqual(lp.card(log)["pending"], 1)

    def test_tampering_detected(self):
        self.issue()
        d = pd.read_csv(self.log)
        d.loc[0, "pred_Bosluk"] = float(d.loc[0, "pred_Bosluk"]) + 1
        d.to_csv(self.log, index=False)
        with self.assertRaises(lp.LogTampered):
            lp.score(self.phys, self.log)


class TestProtocol(Base):

    def test_prepare_locked(self):
        self.assertEqual(lp.prepare(self.prereg), "zaten hazir")
        with mock.patch.object(lp, "ROLL_N", 100):
            with self.assertRaisesRegex(lp.Refused, "degistirilemez"):
                lp.prepare(self.prereg)

    def test_evaluate_threshold_holm_and_version_filter(self):
        rng = np.random.default_rng(0)
        n = 130
        md5 = json.loads(self.prereg.read_text())["code_md5"]
        rw = 10 + rng.normal(0, 1, n) ** 2
        log = pd.DataFrame({"target_date": pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
                            "code_md5": ["eski"] * 5 + [md5] * 125, "realized_price": 100.0,
                            "abs_err_RW": rw, "abs_err_Bosluk": rw * rng.uniform(0.75, 1.05, n),
                            "abs_err_Bosluk-Kayan": rw * rng.uniform(0.9, 1.1, n), "abs_err_Prim-Capa": rw * 1.3})
        with self.assertRaisesRegex(lp.Refused, "120"):
            lp.evaluate(log.head(100), json.loads(self.prereg.read_text()))
        ev = lp.evaluate(log, json.loads(self.prereg.read_text()))
        self.assertEqual(ev["n"], 120)
        self.assertEqual(ev["first"], log["target_date"].iat[5])          # eski surum tahminleri sayilmaz
        r = ev["results"]
        raw = {m: r[m]["dm"]["p_value"] for m in r}
        lo = min(raw, key=raw.get)
        self.assertGreater(raw[lo], 0)                                   # Holm'un etkisi gorulebilsin
        self.assertAlmostEqual(r[lo]["p_holm"] / raw[lo], 3.0, places=9)    # en kucuk p: 3 karsilastirma ile carpilir
        for m in r:
            self.assertGreaterEqual(r[m]["p_holm"], raw[m])
        self.assertTrue(ev["results"]["Bosluk"]["better"])
        self.assertFalse(ev["results"]["Prim-Capa"]["better"])
        self.assertGreaterEqual(ev["results"]["Bosluk-Kayan"]["p_holm"], ev["results"]["Bosluk"]["p_holm"])


class TestRefresh(unittest.TestCase):

    def test_refresh_keeps_base_and_writes_live_copy(self):
        from tests.test_physical_gold import FakeEVDS
        tmp = Path(tempfile.mkdtemp(prefix="lpr_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        base = pd.DataFrame({"Date": pd.to_datetime(["2018-06-28", "2018-06-29"]), "kapanis_tl_gr": [180.0, 184.0],
                             "agort_tl_gr": [180.0, 184.0], "kapanis_usd_ons": [1250.0, 1252.0], "kaynak": "IAB_arsiv"})
        bp, out = tmp / "base.csv", tmp / "live" / "physical_gold_daily.csv"
        base.to_csv(bp, index=False, date_format="%Y-%m-%d")
        h = hashlib.md5(bp.read_bytes()).hexdigest()
        with mock.patch.object(evds, "OUT_ROOT", tmp / "evds"):
            r = lp.refresh(evds.Client("k", get=FakeEVDS()), bp, out, now=pd.Timestamp("2018-08-31", tz="UTC").to_pydatetime())
        self.assertEqual(hashlib.md5(bp.read_bytes()).hexdigest(), h)
        live = pd.read_csv(out, parse_dates=["Date"])
        self.assertEqual(set(live["kaynak"]), {"IAB_arsiv", "BIST_KMP"})
        self.assertEqual(r["last_date"], "2018-08-31")


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
