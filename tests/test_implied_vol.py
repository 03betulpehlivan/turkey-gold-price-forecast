"""
GVZ / bilesen oynaklik testleri: src/models/implied_vol.py
==========================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_implied_vol -v

Iki sentetik dunya:
  bilgili : GVZ, yarinin altin oynakligini onceden haber verir -> comp_gvz bulunmali
  gurultu : GVZ altinla ilgisiz                              -> hicbir GVZ yontemi kabul edilmemeli
Gercek data/ klasoru kullanilmaz.
"""

from pathlib import Path
from unittest import mock
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

from src.data import build_dataset as bd  # noqa: E402
from src.models import implied_vol as ivx  # noqa: E402
from src.models import train as tr  # noqa: E402


def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def make_world(kind: str, seed: int = 0, end: str = "2026-03-31"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", end)
    n = len(days)
    h = np.zeros(n)
    for t in range(1, n):                       # kisa omurlu oynaklik soklari
        h[t] = 0.5 * h[t - 1] + rng.normal(0, 0.35)
    sd_next = 0.009 * np.exp(h)                 # sd_next[t]: r_gold[t+1] icin
    r_gold = np.empty(n)
    r_gold[0] = rng.normal(0, 0.009)
    r_gold[1:] = rng.normal(0.0002, sd_next[:-1])
    if kind == "bilgili":
        gvz = sd_next * math.sqrt(252) * 100 * np.exp(rng.normal(0, 0.05, n))
    else:
        gvz = 15 * np.exp(rng.normal(0, 0.3, n))
    gold = 1200 * np.exp(np.cumsum(r_gold))
    fx = 2.3 * np.exp(np.cumsum(rng.normal(0.0008, 0.003, n)))
    _, model = quiet(bd.build, pd.DataFrame({"Date": days, "close": gold}), pd.DataFrame({"Date": days, "close": fx}))
    model = model[model["Date"] < pd.Timestamp("2026-01-01")]
    panel = pd.DataFrame({"Date": days, "gvz": gvz})
    panel.loc[panel["Date"] >= pd.Timestamp("2026-01-01"), "gvz"] = np.nan   # 2026 okunursa hata verir
    return model, panel


def write_world(tmp: Path, kind: str, seed: int = 0):
    model, panel = make_world(kind, seed)
    mpath, ppath = tmp / "model_dataset.csv", tmp / "market_panel.csv"
    model.to_csv(mpath, index=False, date_format="%Y-%m-%d")
    panel.to_csv(ppath, index=False, date_format="%Y-%m-%d")
    return mpath, ppath


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class TestFormulas(unittest.TestCase):

    def test_gvz_to_daily(self):
        self.assertAlmostEqual(ivx.gvz_to_daily([100 * math.sqrt(252) * 0.01])[0], 0.01, places=12)

    def test_component_sigma(self):
        self.assertAlmostEqual(ivx.component_sigma(3.0, 4.0, 0.0)[()], 5.0)
        self.assertAlmostEqual(ivx.component_sigma(3.0, 4.0, 1.0)[()], 7.0)
        self.assertAlmostEqual(ivx.component_sigma(3.0, 4.0, -1.0)[()], 1.0)

    def test_decision_rule_requires_holm_significance(self):
        metrics = pd.DataFrame({"method": ["garch", "a", "b"], "pinball": [1.0, 0.9, 0.9],
                                "coverage_80": [0.8, 0.8, 0.8], "coverage_95": [0.95, 0.95, 0.95]})
        sig = pd.DataFrame({"method": ["a", "b"], "dm_stat": [-3.0, -3.0], "p_holm": [0.01, 0.2],
                            "win_share": [0.7, 0.7]})
        with mock.patch.object(ivx, "CANDIDATES", ["a", "b"]):
            d = ivx.decide(metrics, sig).set_index("method")
        self.assertTrue(d.loc["a", "accepted"])
        self.assertFalse(d.loc["b", "accepted"])

    def test_timing_note(self):
        d = pd.DataFrame({"method": ["comp_gvz", "comp_gvz_lag"], "accepted": [True, False]})
        self.assertIn("zamanlama", ivx.timing_note(d))
        d["accepted"] = [True, True]
        self.assertIn("aciklanamaz", ivx.timing_note(d))


class TestWorlds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ivx_w_"))
        cls.res = {}
        for kind in ("bilgili", "gurultu"):
            d = cls.tmp / kind
            d.mkdir()
            mpath, ppath = write_world(d, kind, seed=3)
            with mock.patch.object(tr, "OUTPUT_DIR", d / "results"):
                cls.res[kind] = quiet(ivx.run, mpath, ppath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def accepted(self, kind):
        d = self.res[kind]["decision"]
        return set(d.loc[d["accepted"], "method"])

    def test_informative_gvz_is_found(self):
        self.assertIn("comp_gvz", self.accepted("bilgili"))

    def test_noise_gvz_is_not_accepted(self):
        self.assertFalse(self.accepted("gurultu") & {"comp_gvz", "comp_gvz_lag", "garch_gvz_adj"})

    def test_evaluation_stays_in_development_period(self):
        for r in self.res.values():
            dates = pd.to_datetime(r["walk"]["predictions"]["prediction_date"])
            self.assertGreaterEqual(dates.min(), pd.Timestamp(ivx.DEV_START))
            self.assertLess(dates.max(), pd.Timestamp(tr.TEST_START))


class TestProtection(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ivx_p_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mpath, self.ppath = write_world(self.tmp, "gurultu", seed=4)

    def test_2026_rows_are_dropped_even_if_model_dataset_contains_them(self):
        # model_dataset ileride 2026 verisiyle uretilirse: 2026 satirlari atilmali, panelde 2026 GVZ
        # bos oldugu icin yanlis "eksik GVZ" hatasi da verilmemeli.
        model, _ = make_world("gurultu", seed=4)
        days = pd.bdate_range("2015-01-01", "2026-03-31")
        rng = np.random.default_rng(9)
        g = 1200 * np.exp(np.cumsum(rng.normal(0, 0.009, len(days))))
        f = 2.3 * np.exp(np.cumsum(rng.normal(0.0008, 0.003, len(days))))
        _, full = quiet(bd.build, pd.DataFrame({"Date": days, "close": g}), pd.DataFrame({"Date": days, "close": f}))
        self.assertGreater(full["Date"].max(), pd.Timestamp("2026-01-01"))
        full.to_csv(self.mpath, index=False, date_format="%Y-%m-%d")
        df = quiet(ivx.load_inputs, self.mpath, self.ppath)
        self.assertLess(df["Date"].max(), pd.Timestamp("2026-01-01"))

    def test_gvz_lag_is_previous_panel_day(self):
        df = quiet(ivx.load_inputs, self.mpath, self.ppath)
        panel = pd.read_csv(self.ppath, parse_dates=["Date"]).set_index("Date")["gvz"]
        d = df["Date"].iloc[500]
        prev = panel.index[panel.index.get_loc(d) - 1]
        self.assertEqual(df["gvz_lag1"].iloc[500], panel[prev])

    def test_missing_gvz_column_or_values_raise(self):
        p = pd.read_csv(self.ppath)
        p.drop(columns="gvz").to_csv(self.ppath, index=False)
        with self.assertRaisesRegex(ValueError, "gvz"):
            quiet(ivx.load_inputs, self.mpath, self.ppath)
        p.loc[800, "gvz"] = np.nan
        p.to_csv(self.ppath, index=False)
        with self.assertRaisesRegex(ValueError, "eksik"):
            quiet(ivx.load_inputs, self.mpath, self.ppath)

    def test_test_period_values_cannot_change_forecasts(self):
        df = quiet(ivx.load_inputs, self.mpath, self.ppath)
        a = ivx.walk_forward(df)["predictions"]
        bad = df.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        cols = ["gold_return", "gold_usd_return", "usdtry_return", tr.TARGET, tr.PRICE_COL, "gvz", "gvz_lag1"]
        bad.loc[m, cols] = bad.loc[m, cols] * 4.0
        b = ivx.walk_forward(bad)["predictions"]
        pd.testing.assert_frame_equal(a, b)

    def test_preregistration_before_computation_and_isolation(self):
        out_root = self.tmp / "results"
        before = file_hashes(self.tmp)
        with mock.patch.object(tr, "OUTPUT_DIR", out_root), \
                mock.patch.object(ivx, "walk_forward", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(ivx.run, self.mpath, self.ppath)
        target = out_root / "experiments" / "implied_vol"
        prereg = json.loads((target / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(prereg["candidates"], ivx.CANDIDATES)
        self.assertFalse((target / "decision.csv").exists())
        self.assertEqual(file_hashes(self.tmp), before)


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
