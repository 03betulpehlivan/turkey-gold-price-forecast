"""
Tahmin karnesi testleri: src/tool/karne.py
Calistirma: python -m unittest tests.test_tool_karne -v
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

from src.tool import karne as k  # noqa: E402


def world(seed=1, start="2010-01-04", end="2022-12-30", calm=False):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, end)
    w, a, b = (1e-6, 0.0, 0.0) if calm else (2e-6, 0.08, 0.9)
    v = 1e-4 if calm else w / (1 - a - b)
    r = np.empty(len(d))
    for i in range(len(d)):
        r[i] = 0.0003 + np.sqrt(v) * rng.normal()
        if not calm:
            v = w + a * (r[i] - 0.0003) ** 2 + b * v
    return pd.Series(100 * np.exp(np.cumsum(r)), index=d)


class FakeChronos:
    """Son fiyat etrafinda sabit +-%1 / %2 yuzdelikler; baglamin son degerini kaydeder."""
    def __init__(self):
        self.last = []

    def predict_quantiles(self, ctx, prediction_length, quantile_levels):
        out = []
        for c in ctx:
            p = float(np.asarray(c)[-1])
            self.last.append(p)
            out.append([[p * 0.98, p * 0.99, p, p * 1.01, p * 1.02]])
        return np.array(out), None


class TestBasics(unittest.TestCase):
    def test_parse_and_split(self):
        self.assertEqual(k.parse_years("2012-2019"), (2012, 2019))
        self.assertEqual(k.parse_years("2020"), (2020, 2020))
        with self.assertRaises(ValueError):
            k.parse_years("2020-2010")
        df = k.frame(world())
        tr, te = k.split(df, (2012, 2019), (2020, 2022))
        self.assertTrue((tr["T"].dt.year.between(2012, 2019)).all())
        self.assertTrue((te["T"].dt.year.between(2020, 2022)).all())
        with self.assertRaisesRegex(ValueError, "SONRA"):
            k.split(df, (2015, 2019), (2014, 2016))
        with self.assertRaisesRegex(ValueError, "kisa"):
            k.split(df[df["T"].dt.year >= 2019].iloc[:100], (2019, 2019), (2020, 2020))

    def test_frame_alignment(self):
        p = world()
        df = k.frame(p)
        self.assertEqual(df["T"].iat[0], p.index[1])
        self.assertAlmostEqual(df["r1"].iat[5], np.log(p.iat[6] / p.iat[5]))
        self.assertAlmostEqual(df["r"].iat[5], np.log(p.iat[5] / p.iat[4]))       # bilgi gunu getirisi

    def test_p_move(self):
        self.assertAlmostEqual(float(k.p_move(0.0, 0.01, 1.0)), 0.3173, places=3)  # +-1 sigma disi
        self.assertLess(float(k.p_move(0.0, 0.001, 1.0)), 1e-6)


class TestNoFuture(unittest.TestCase):
    def test_forecasts_ignore_future_prices(self):
        p = world()
        a = k.run(p, (2012, 2019), (2020, 2021))
        p2 = p.copy()
        p2[p2.index >= "2021-07-01"] *= 3.0
        b = k.run(p2, (2012, 2019), (2020, 2021))
        df = k.frame(p)
        tr, te = k.split(df, (2012, 2019), (2020, 2021))
        full = df.loc[tr.index.min(): te.index.max()]
        df2 = k.frame(p2)
        tr2, te2 = k.split(df2, (2012, 2019), (2020, 2021))
        full2 = df2.loc[tr2.index.min(): te2.index.max()]
        early = (te["T"] < "2021-06-30").to_numpy()
        for m in k.MODELS:
            o1 = k.FUNCS[m](tr, te, full, p)
            o2 = k.FUNCS[m](tr2, te2, full2, p2)
            for c in ("point", "lo80", "hi80", "lo95", "hi95"):
                np.testing.assert_allclose(o1[c].to_numpy()[early], o2[c].to_numpy()[early], rtol=1e-9, err_msg=f"{m} {c}")
        self.assertEqual(set(a["modeller"]), set(k.MODELS))

    def test_forecast_does_not_see_its_own_target(self):
        p = world()
        df = k.frame(p)
        tr, te = k.split(df, (2012, 2019), (2020, 2020))
        full = df.loc[tr.index.min(): te.index.max()]
        j = te.index[100]
        p2 = p.copy()
        p2[df.loc[j, "T"]] *= 1.05                                               # yalnizca o gunun gerceklesen fiyati
        df2 = k.frame(p2)
        tr2, te2 = k.split(df2, (2012, 2019), (2020, 2020))
        full2 = df2.loc[tr2.index.min(): te2.index.max()]
        for m in k.MODELS:
            o1 = k.FUNCS[m](tr, te, full, p).set_index(te.index)
            o2 = k.FUNCS[m](tr2, te2, full2, p2).set_index(te2.index)
            for c in ("point", "lo80", "hi80"):
                self.assertAlmostEqual(o1.loc[j, c], o2.loc[j, c], places=9, msg=f"{m} {c}")
        for m in k.PENCERELI:                                                    # aylik yeniden egitimle de
            for w in (250, 0):
                o1 = k.FUNCS[m](tr, te, full, p, pencere=w).set_index(te.index)
                o2 = k.FUNCS[m](tr2, te2, full2, p2, pencere=w).set_index(te2.index)
                self.assertAlmostEqual(o1.loc[j, "lo80"], o2.loc[j, "lo80"], places=9, msg=f"{m} pencere {w}")

    def test_training_years_only_set_parameters(self):
        p = world()
        df = k.frame(p)
        tr, te = k.split(df, (2012, 2019), (2020, 2020))
        full = df.loc[tr.index.min(): te.index.max()]
        o = k.m_rw(tr, te, full, p)
        self.assertAlmostEqual(o["sigma"].iat[0], float(np.std(tr["r1"])))
        np.testing.assert_allclose(o["point"].to_numpy(), te["P"].to_numpy())


def shift_world(seed=2, end="2022-12-30"):
    """2018 sonrasi oynaklik sertlesir; kalin kuyruklu soklar."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2008-01-02", end)
    v, r = 1e-4, np.empty(len(d))
    for i in range(len(d)):
        w = 2e-6 if d[i].year < 2018 else 6e-6
        r[i] = 0.0004 + np.sqrt(v) * rng.standard_t(5) * np.sqrt(3 / 5)
        v = w + 0.08 * (r[i] - 0.0004) ** 2 + 0.9 * v
    return pd.Series(50 * np.exp(np.cumsum(r)), index=d)


class TestImprovements(unittest.TestCase):
    def test_distribution_helpers(self):
        self.assertAlmostEqual(k._ppf(0.975, np.nan), 1.959964, places=5)
        self.assertGreater(k._ppf(0.975, 5.0), 1.96)                              # kalin kuyruk
        for nu in (np.nan, 6.0):
            self.assertAlmostEqual(k._cdf(k._ppf(0.9, nu), nu), 0.9, places=9)
        self.assertEqual(np.shape(k._ppf(np.array([0.9, 0.975]), np.array([np.nan, 5.0]))), (2,))

    def test_window_and_aci_fix_coverage_after_regime_shift(self):
        p = shift_world()
        base = k.run(p, (2012, 2017), (2018, 2022), models=("GARCH",))["modeller"]["GARCH"]
        best = k.run(p, (2012, 2017), (2018, 2022), models=("GARCH-t+ACI",), pencere=500)["modeller"]["GARCH-t+ACI"]
        self.assertLess(abs(best["aralik80_tutma_pct"] - 80), abs(base["aralik80_tutma_pct"] - 80))
        self.assertLess(abs(best["aralik95_tutma_pct"] - 95), abs(base["aralik95_tutma_pct"] - 95))
        self.assertAlmostEqual(best["aralik80_tutma_pct"], 80, delta=2)

    def test_window_choice_ignores_test_years(self):
        p = shift_world()
        a = k.pencere_secimi(p, (2012, 2017), (2018, 2019), pencereler=(250, 0))
        p2 = p.copy()
        p2[p2.index >= "2018-01-01"] *= np.exp(np.cumsum(np.full((p2.index >= "2018-01-01").sum(), 0.01)))
        b = k.pencere_secimi(p2, (2012, 2017), (2018, 2019), pencereler=(250, 0))
        self.assertEqual(a["secilen_pencere"], b["secilen_pencere"])
        self.assertEqual([r["dogrulama_sapma"] for r in a["pencereler"]], [r["dogrulama_sapma"] for r in b["pencereler"]])
        self.assertEqual(a["dogrulama"], [2016, 2017])
        self.assertEqual(len(a["pencereler"]), 2)

    def test_aci_widens_after_misses(self):
        te = pd.DataFrame({"P": [100.0] * 4, "P1": [110.0, 110.0, 110.0, 100.0]})
        o = pd.DataFrame({"point": [100.0] * 4, "mu": [0.0] * 4, "sigma": [0.01] * 4, "nu": [np.nan] * 4,
                          "lo80": 0.0, "hi80": 0.0, "lo95": 0.0, "hi95": 0.0})
        a = k.aci(te, o, gamma=0.05)
        w = (a["hi80"] - a["lo80"]).to_numpy()
        self.assertAlmostEqual(w[0], 100 * (np.exp(k.Z80 * 0.01) - np.exp(-k.Z80 * 0.01)), places=6)   # ilk gun nominal
        self.assertTrue(w[1] > w[0] and w[2] > w[1])                             # iskaladikca genisler


class TestScore(unittest.TestCase):
    def test_garch_coverage_and_reliability(self):
        res = k.run(world(seed=3), (2012, 2018), (2019, 2022), models=("Yarin=bugun", "GARCH"))
        g = res["modeller"]["GARCH"]
        self.assertAlmostEqual(g["aralik80_tutma_pct"], 80, delta=5)
        self.assertAlmostEqual(g["aralik95_tutma_pct"], 95, delta=3)
        self.assertEqual(g["rw_gore_pct"], 0.0)
        self.assertEqual(sum(r["gun"] for r in g["guvenilirlik"]), g["gun"])
        self.assertEqual({y["yil"] for y in res["yillar"]["GARCH"]}, {2019, 2020, 2021, 2022})

    def test_score_math(self):
        te = pd.DataFrame({"P": [100.0, 100.0], "P1": [101.0, 98.0], "r1": [np.log(1.01), np.log(0.98)]})
        o = pd.DataFrame({"point": [100.0, 100.0], "lo80": [99.5, 99.0], "hi80": [101.5, 101.0],
                          "lo95": [99.0, 97.0], "hi95": [102.0, 103.0], "mu": [0.0, 0.0], "sigma": [0.001, 0.05]})
        s = k.score(te, o, rw_mae=1.5)
        self.assertAlmostEqual(s["ortalama_hata_tl"], 1.5)
        self.assertAlmostEqual(s["aralik80_tutma_pct"], 50.0)
        self.assertAlmostEqual(s["aralik95_tutma_pct"], 100.0)
        self.assertAlmostEqual(s["hep_sakin_isabet_pct"], 50.0)                  # 2. gun %2 hareket


class TestChronosAndTomorrow(unittest.TestCase):
    def test_chronos_uses_context_up_to_info_day(self):
        p = world()
        fake = FakeChronos()
        res = k.run(p, (2012, 2019), (2020, 2020), models=("Chronos",), chronos_pipeline=fake)
        df = k.frame(p)
        te = df[df["T"].dt.year == 2020]
        np.testing.assert_allclose(fake.last, te["P"].to_numpy())                 # baglamin son degeri = bilgi gunu
        self.assertAlmostEqual(res["modeller"]["Chronos"]["aralik80_genislik_pct"], 2.0, places=6)

    def test_tomorrow_equals_forecast_for_a_known_day(self):
        p = world()
        df = k.frame(p)
        tr = df[(df["T"].dt.year >= 2012) & (df["T"].dt.year <= 2021)]
        te = df[df.index > tr.index.max()]                                       # 2022; son satirin hedefi biliniyor
        full = df.loc[tr.index.min():]
        for m in k.MODELS:
            c = k.tomorrow(p.iloc[:-1], (2012, 2021), m)
            o = k.FUNCS[m](tr, te, full, p).iloc[-1]
            self.assertAlmostEqual(c["tahmin"], float(o["point"]), places=6, msg=m)
            self.assertAlmostEqual(c["aralik80"][0], float(o["lo80"]), places=6, msg=m)

    def test_tomorrow_card(self):
        p = world(end="2026-10-16")
        c = k.tomorrow(p, (2012, 2026))
        self.assertEqual(c["bilgi_gunu"], "2026-10-16")
        self.assertAlmostEqual(c["tahmin"], float(p.iloc[-1]))
        self.assertLess(c["aralik95"][0], c["aralik80"][0])
        self.assertLess(c["aralik80"][1], c["aralik95"][1])
        self.assertTrue(0 <= c["hareketli_olasiligi_pct"] <= 100)
        for m in ("Yarin=bugun", "ARIMA", "XGBoost"):
            self.assertTrue(np.isfinite(k.tomorrow(p, (2012, 2026), m)["tahmin"]), m)


class TestMain(unittest.TestCase):
    def test_cli_writes_json(self):
        p = world()
        tmp = Path(tempfile.mkdtemp(prefix="karne_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(k, "load_series", lambda: p), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(k.main(["--egitim", "2012-2018", "--test", "2019-2020", "--out", str(tmp)]), 0)
            self.assertEqual(k.main(["yarin", "--egitim", "2012-2022"]), 0)
        r = json.loads((tmp / "karne_2012-2018_2019-2020.json").read_text(encoding="utf-8"))
        self.assertEqual(set(r["modeller"]), set(k.MODELS))
        self.assertIn("olasilikla", out.getvalue())

    def test_missing_data_message(self):
        with mock.patch.object(k, "PATHS", [Path(tempfile.gettempdir()) / "yok_123.csv"]):
            with self.assertRaisesRegex(FileNotFoundError, "Fiziki altin verisi yok"):
                k.load_series()


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
