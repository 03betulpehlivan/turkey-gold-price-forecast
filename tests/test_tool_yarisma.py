"""
Model yarismasi testleri: src/tool/yarisma.py
Calistirma: python -m unittest tests.test_tool_yarisma -v
Sentetik fiyat serileri ve sahte Chronos; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi).
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

from src.tool import yarisma as y  # noqa: E402


def world(seed=2, start="2008-01-02", end="2024-06-28"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, end)
    v, r = 1e-4, np.empty(len(d))
    for i in range(len(d)):
        w = 2e-6 if d[i].year < 2018 else 5e-6
        r[i] = 0.0004 + np.sqrt(v) * rng.standard_t(5) * np.sqrt(3 / 5)
        v = w + 0.08 * (r[i] - 0.0004) ** 2 + 0.9 * v
    return pd.Series(50 * np.exp(np.cumsum(r)), index=d)


FAST = ("Sabit aralik", "Gecmis benzer", "EWMA", "EWMA+ACI", "XGBoost-kantil")


class FakeChronos:
    def predict_quantiles(self, ctx, prediction_length, quantile_levels):
        out = []
        for c in ctx:
            p = float(np.asarray(c)[-1])
            row = [[p * (1 + 0.01 * (q - 0.5)) for q in quantile_levels]] * prediction_length
            out.append(row)
        return np.array(out), None


class TestNoFuture(unittest.TestCase):
    def setUp(self):
        self.p = world(end="2016-12-30")
        self.d = y.hazirla(self.p)

    def test_forecast_ignores_later_prices(self):
        for h in (1, 5):
            a = y.tahminler(self.d, h, FAST + ("GARCH-t",))
            cut = int(a["idx"][300])
            p2 = self.p.copy()
            p2.iloc[cut + 1:] *= np.exp(np.cumsum(np.full(len(p2) - cut - 1, 0.01)))
            b = y.tahminler(y.hazirla(p2), h, FAST + ("GARCH-t",))
            for m in a["Q"]:
                np.testing.assert_allclose(a["Q"][m][:301], b["Q"][m][:301], rtol=1e-9, atol=1e-12, err_msg=f"{m} h={h}")

    def test_forecast_does_not_see_its_own_target(self):
        h = 3
        a = y.tahminler(self.d, h, FAST + ("GARCH-t+ACI", "GJR-t"))
        j = 200
        i0 = int(a["idx"][j])
        p2 = self.p.copy()
        p2.iloc[i0 + h] *= 1.08                                                   # yalnizca o tahminin hedef gunu
        b = y.tahminler(y.hazirla(p2), h, FAST + ("GARCH-t+ACI", "GJR-t"))
        for m in a["Q"]:
            np.testing.assert_allclose(a["Q"][m][j], b["Q"][m][j], rtol=1e-9, err_msg=m)


    def test_yearly_ml_training_uses_only_realized_targets(self):
        h = 5
        a = y.tahminler(self.d, h, ("XGBoost-kantil",))
        yil = pd.DatetimeIndex(self.d["tarih"][a["idx"]]).year
        j = int(np.flatnonzero(yil == 2015)[0])                                   # yilin ilk tahmin gunu (egitim ani)
        b0 = int(a["idx"][j])
        for k_ in range(1, h + 1):                                                # b0'dan sonraki h gun: henuz bilinmiyor
            p2 = self.p.copy()
            p2.iloc[b0 + k_] *= 1.10
            b = y.tahminler(y.hazirla(p2), h, ("XGBoost-kantil",))
            np.testing.assert_allclose(a["Q"]["XGBoost-kantil"][j], b["Q"]["XGBoost-kantil"][j], rtol=1e-9, err_msg=str(k_))


class TestNewModels(unittest.TestCase):
    YENI = ("ARIMA-GARCH", "HAR", "Rejim (Markov)", "Birlesim")

    def setUp(self):
        self.p = world(end="2015-12-31")
        self.d = y.hazirla(self.p)

    def test_no_future_and_own_target(self):
        import warnings
        warnings.simplefilter("ignore")
        h = 3
        a = y.tahminler(self.d, h, self.YENI)
        j = 150
        i0 = int(a["idx"][j])
        p2 = self.p.copy()
        p2.iloc[i0 + 1:] *= np.exp(np.cumsum(np.full(len(p2) - i0 - 1, 0.01)))   # bilgi gununden sonraki her sey
        b = y.tahminler(y.hazirla(p2), h, self.YENI)
        for m in self.YENI:
            np.testing.assert_allclose(a["Q"][m][: j + 1], b["Q"][m][: j + 1], rtol=1e-7, atol=1e-10, err_msg=m)

    def test_refit_day_uses_only_realized_targets(self):
        import warnings
        warnings.simplefilter("ignore")
        h = 3
        for m in ("HAR", "ARIMA-GARCH", "Rejim (Markov)"):
            a = y.tahminler(self.d, h, (m,))
            per = pd.DatetimeIndex(self.d["tarih"][a["idx"]]).to_period("Q")
            j = int(np.flatnonzero(per == pd.Period("2014Q2"))[0])                # ay ve ceyrek basi (yeniden tahmin gunu)
            b0 = int(a["idx"][j])
            for k_ in range(1, h + 1):
                p2 = self.p.copy()
                p2.iloc[b0 + k_] *= 1.10
                b = y.tahminler(y.hazirla(p2), h, (m,))
                np.testing.assert_allclose(a["Q"][m][j], b["Q"][m][j], rtol=1e-7, err_msg=f"{m} +{k_}")

    def test_combination_is_mean_of_members(self):
        h = 1
        t = y.tahminler(self.d, h, ("Birlesim",) + y.BIRLESIM)
        np.testing.assert_allclose(t["Q"]["Birlesim"], np.mean([t["Q"][u] for u in y.BIRLESIM], axis=0))
        self.assertEqual(set(y.tahminler(self.d, h, ("Birlesim",))["Q"]), {"Birlesim"})

    def test_regime_and_har_reasonable_coverage(self):
        import warnings
        warnings.simplefilter("ignore")
        d = y.hazirla(world(end="2019-12-31"))
        t = y.tahminler(d, 1, ("HAR", "Rejim (Markov)"))
        yy = d["lp"][t["idx"] + 1] - d["lp"][t["idx"]]
        for m in ("HAR", "Rejim (Markov)"):
            s = y.puanla(t["Q"][m], yy)
            self.assertAlmostEqual(s["tutma80"], 80, delta=5, msg=m)
            self.assertAlmostEqual(s["tutma95"], 95, delta=3, msg=m)


class TestScoring(unittest.TestCase):
    def test_interval_score(self):
        self.assertAlmostEqual(float(y.aralik_puani(-1.0, 1.0, 0.0, 0.2)), 2.0)
        self.assertAlmostEqual(float(y.aralik_puani(-1.0, 1.0, 2.0, 0.2)), 2.0 + 10 * 1.0)

    def test_puanla_and_winner_rule(self):
        Q = np.tile(np.log([0.97, 0.98, 1.0, 1.02, 1.03]), (4, 1))
        s = y.puanla(Q, np.log(np.array([1.0, 1.01, 1.025, 0.95])))
        self.assertAlmostEqual(s["tutma80"], 50.0)
        self.assertAlmostEqual(s["tutma95"], 75.0)
        tab = {"genis": {"tutma80": 80.5, "tutma95": 95.1, "puan80": 5.0, "puan95": 9.0},
               "dar_ama_iskalar": {"tutma80": 70.0, "tutma95": 90.0, "puan80": 4.0, "puan95": 8.0},
               "dar_ve_tutar": {"tutma80": 79.0, "tutma95": 94.0, "puan80": 4.5, "puan95": 8.5}}
        self.assertEqual(y.kazanan(tab), "dar_ve_tutar")                          # hedefi tutturanlar arasinda en iyi puan

    def test_quantiles_ordered(self):
        d = y.hazirla(world(end="2014-12-31"))
        t = y.tahminler(d, 1, ("XGBoost-kantil", "Gecmis benzer", "EWMA"))
        for m, Q in t["Q"].items():
            self.assertTrue((np.diff(Q, axis=1) >= -1e-12).all(), m)


class TestRun(unittest.TestCase):
    def test_periods_and_structure(self):
        res = y.yarisma(world(), ufuklar=(1,), modeller=FAST)
        u = res["ufuklar"]["1"]
        self.assertTrue(u["donem"]["secim"][1] < "2023-01-01")                  # secimde 2023 hedefi yok
        self.assertTrue(u["donem"]["final"][0] >= "2023-01-01")
        self.assertIn(u["kazanan"], FAST)
        self.assertEqual(set(u["final"]), set(FAST))
        self.assertEqual({r["yil"] for r in u["yillar"]["EWMA"]}, set(range(2012, 2025)))
        e = u["secim"]["EWMA+ACI"]
        self.assertAlmostEqual(e["tutma80"], 80, delta=2)                         # ACI hedefe oturtur

    def test_chronos_fake_and_cli(self):
        p = world(end="2014-12-31")
        d = y.hazirla(p)
        t = y.tahminler(d, 3, ("Chronos",), chronos_pipeline=FakeChronos())
        self.assertEqual(t["Q"]["Chronos"].shape[1], 5)
        np.testing.assert_allclose(t["Q"]["Chronos"][:, 2], 0.0, atol=1e-12)     # medyan = bilgi gunu fiyati
        self.assertTrue((t["Q"]["Chronos"][:, 0] < t["Q"]["Chronos"][:, 1]).all())  # %95 alt sinir, %80'in altinda
        tmp = Path(tempfile.mkdtemp(prefix="yar_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(y.k, "load_series", lambda: world()), mock.patch.object(y, "MODELLER", FAST), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(y.main(["--ufuk", "1", "--out", str(tmp)]), 0)
        r = json.loads((tmp / "yarisma_1.json").read_text(encoding="utf-8"))
        self.assertIn("1", r["ufuklar"])
        self.assertIn("kazanan", out.getvalue())


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
