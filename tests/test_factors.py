"""
Faz C testleri: src/models/factors.py
=====================================
Calistirma (proje kokunden):
  python -m unittest tests.test_factors -v

Uc sentetik senaryo:
  null    : faktorlerin altinla iliskisi yok   -> hicbir grup kabul edilmemeli
  getiri  : yarinin altin getirisi bugunun dolar endeksi getirisine bagli -> dolar_faiz bulunmali
  oynaklik: yarinin altin oynakligi bugunun VIX seviyesine bagli; VIX soklari kisa omurlu
            oldugu icin bu bilgi GARCH'a gore EK bilgidir                 -> risk (oynaklik) bulunmali
Gercek data/ klasoru kullanilmaz.
"""

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
from src.models import factors as fc  # noqa: E402
from src.models import train as tr  # noqa: E402

EXTRAS = {"silver": 17.0, "dxy": 95.0, "us10y": 2.0, "vix": 16.0, "sp500": 2000.0,
          "bist100": 800.0, "brent": 60.0, "eurtry": 3.0}


def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def make_world(scenario: str, seed: int = 0, end: str = "2025-12-31"):
    """Ortak is gunu takviminde altin, kur ve 8 faktor; senaryoya gore gercek etki eklenir."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", end)
    n = len(days)
    fac_r = {k: rng.normal(0, 0.01, n) for k in EXTRAS if k not in ("vix", "us10y")}
    # VIX soklari kisa omurlu (AR katsayisi 0.5): GARCH bunlari gecmis getirilerden
    # ogrenemez, VIX ise onceden haber verir -> GARCH'a gore EK bilgi tasiyan senaryo.
    log_vix = np.empty(n)
    log_vix[0] = np.log(16)
    for t in range(1, n):
        log_vix[t] = np.log(16) + 0.5 * (log_vix[t - 1] - np.log(16)) + rng.normal(0, 0.3)
    us10y = 2.0 + np.cumsum(rng.normal(0, 0.03, n))
    r_gold = np.empty(n)
    for t in range(n):
        sd = 0.009
        if scenario == "oynaklik" and t > 0:
            sd = 0.009 * np.exp(1.2 * (log_vix[t - 1] - np.log(16)))
        r_gold[t] = rng.normal(0.0002, sd)
        if scenario == "getiri" and t > 0:
            r_gold[t] += -0.6 * fac_r["dxy"][t - 1]
    gold = 1200 * np.exp(np.cumsum(r_gold))
    fx = 2.3 * np.exp(np.cumsum(rng.normal(0.0008, 0.003, n)))
    _, model = quiet(bd.build, pd.DataFrame({"Date": days, "close": gold}), pd.DataFrame({"Date": days, "close": fx}))
    panel = pd.DataFrame({"Date": days, "gold_usd_oz": gold, "usdtry": fx,
                          "theoretical_gram_try": gold * fx / 31.1035})
    for k, lvl in EXTRAS.items():
        if k == "vix":
            panel[k] = np.exp(log_vix)
        elif k == "us10y":
            panel[k] = us10y
        else:
            panel[k] = lvl * np.exp(np.cumsum(fac_r[k]))
        panel[f"{k}_age_days"] = 0.0
    return model, panel


def write_world(tmp: Path, scenario: str, seed: int = 0, end: str = "2025-12-31"):
    model, panel = make_world(scenario, seed, end)
    mpath, ppath = tmp / "model_dataset.csv", tmp / "market_panel.csv"
    model.to_csv(mpath, index=False, date_format="%Y-%m-%d")
    panel.to_csv(ppath, index=False, date_format="%Y-%m-%d")
    return mpath, ppath


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class TestFeatures(unittest.TestCase):

    def setUp(self):
        _, self.panel = make_world("null", end="2026-03-31")

    def test_definitions(self):
        f = fc.build_factor_features(self.panel)
        p = self.panel
        t = 300
        self.assertAlmostEqual(f.loc[t, "dxy_r1"], np.log(p["dxy"][t] / p["dxy"][t - 1]), places=14)
        self.assertAlmostEqual(f.loc[t, "sp500_r5"], np.log(p["sp500"][t] / p["sp500"][t - 5]), places=14)
        self.assertAlmostEqual(f.loc[t, "us10y_d5"], p["us10y"][t] - p["us10y"][t - 5], places=14)
        self.assertAlmostEqual(f.loc[t, "vix_log"], np.log(p["vix"][t]), places=14)

    def test_2026_rows_dropped_and_no_look_ahead(self):
        f = fc.build_factor_features(self.panel)
        self.assertLess(f["Date"].max(), pd.Timestamp(fc.HOLDOUT_START))
        for t in (10, 800, len(f) - 2):
            cut = fc.build_factor_features(self.panel.iloc[:t + 1])
            pd.testing.assert_series_equal(f.drop(columns="Date").iloc[t], cut.drop(columns="Date").iloc[t],
                                           check_names=False)

    def test_groups_are_new_information_only(self):
        for g, feats in fc.GROUPS.items():
            self.assertFalse(set(feats) & set(fc.CONTROL_FEATURES), g)
        self.assertEqual(len(fc.GROUPS["hepsi"]), 16)

    def test_missing_panel_values_raise(self):
        tmp = Path(tempfile.mkdtemp(prefix="fc_nan_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        mpath, ppath = write_world(tmp, "null")
        panel = pd.read_csv(ppath)
        panel.loc[500, "dxy"] = np.nan
        panel.to_csv(ppath, index=False)
        with self.assertRaisesRegex(ValueError, "eksik"):
            quiet(fc.load_inputs, mpath, ppath)


class TestScenarios(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fc_sc_"))
        cls.res = {}
        for sc in ("null", "getiri", "oynaklik"):
            d = cls.tmp / sc
            d.mkdir()
            mpath, ppath = write_world(d, sc, seed=1)
            with mock.patch.object(tr, "OUTPUT_DIR", d / "results"):
                cls.res[sc] = quiet(fc.run, mpath, ppath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def accepted(self, sc, task):
        d = self.res[sc]["decision"]
        return set(d[(d["task"] == task) & d["accepted"]]["group"])

    def test_null_world_accepts_nothing(self):
        self.assertEqual(self.accepted("null", "returns"), set())
        self.assertEqual(self.accepted("null", "vol"), set())

    def test_return_signal_is_found_in_the_right_group(self):
        acc = self.accepted("getiri", "returns")
        self.assertIn("dolar_faiz", acc)
        self.assertNotIn("emtia", acc)
        self.assertNotIn("turkiye", acc)

    def test_volatility_signal_is_found_in_the_right_group(self):
        acc = self.accepted("oynaklik", "vol")
        self.assertIn("risk", acc)
        self.assertNotIn("emtia", acc)

    def test_evaluation_stays_in_development_period(self):
        for sc, r in self.res.items():
            dates = pd.to_datetime(r["walk"]["returns"]["prediction_date"])
            self.assertGreaterEqual(dates.min(), pd.Timestamp(fc.DEV_START))
            self.assertLess(dates.max(), pd.Timestamp(tr.TEST_START))
            self.assertFalse(r["summary"]["test_period_used"])


class TestProtection(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fc_prot_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mpath, self.ppath = write_world(self.tmp, "null", seed=2)

    def test_test_period_values_cannot_change_forecasts(self):
        df = quiet(fc.load_inputs, self.mpath, self.ppath)
        a = fc.walk_forward(df)
        bad = df.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        cols = ["gold_return", tr.TARGET, tr.PRICE_COL] + fc.GROUPS["hepsi"] + fc.CONTROL_FEATURES
        bad.loc[m, cols] = bad.loc[m, cols] * 5.0
        b = fc.walk_forward(bad)
        pd.testing.assert_frame_equal(a["returns"], b["returns"])
        pd.testing.assert_frame_equal(a["vol"], b["vol"])

    def test_preregistration_written_before_computation_and_outputs_isolated(self):
        out_root = self.tmp / "results"
        before = file_hashes(self.tmp)
        with mock.patch.object(tr, "OUTPUT_DIR", out_root), \
                mock.patch.object(fc, "walk_forward", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(fc.run, self.mpath, self.ppath)
        target = out_root / "experiments" / "phase_c"
        prereg = json.loads((target / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(set(prereg["groups"]), set(fc.GROUPS))
        self.assertFalse((target / "decision.csv").exists())
        self.assertEqual(file_hashes(self.tmp), before)

    def test_decision_rule(self):
        metrics = pd.DataFrame({"model": ["control", "a", "b"], "MAE": [1.0, 0.9, 0.95]})
        sig = pd.DataFrame({"group": ["a", "b"], "dm_stat": [-3.0, -3.0], "p_holm": [0.01, 0.2],
                            "win_share": [0.7, 0.9]})
        d = fc.decide(metrics, sig, "returns").set_index("group")
        self.assertTrue(d.loc["a", "accepted"])
        self.assertFalse(d.loc["b", "accepted"])


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
