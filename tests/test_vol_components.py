"""
TL oynakligi bilesen modeli testleri: src/analysis/vol_components.py
Calistirma: python -m unittest tests.test_vol_components -v
Sentetik dunya: ons GARCH oynakligi + iki kur krizi; GVZ yalnizca ons oynakligini bilir. Gercek veri kullanilmaz.
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

from src.analysis import vol_components as vc  # noqa: E402
from src.analysis import volatility_study as vs  # noqa: E402

CRISES = (("2018-08-01", "2018-09-30"), ("2021-11-15", "2022-01-15"))


def world(seed=1, start="2006-01-02", end="2022-12-30"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, end)
    n = len(d)
    w, a, b = 1e-6, 0.06, 0.92
    v, g, sg = w / (1 - a - b), np.empty(n), np.empty(n)
    for i in range(n):
        sg[i] = v
        g[i] = np.sqrt(v) * rng.normal()
        v = w + a * g[i] ** 2 + b * v
    sf = np.full(n, 0.003)
    for s, e in CRISES:
        sf[(d >= s) & (d <= e)] = 0.03
    f = sf * rng.normal(size=n)
    r = g + f + rng.normal(0, 0.001, n)
    phys = pd.Series(20 * np.exp(np.cumsum(r)), index=d)
    panel = pd.DataFrame({"Date": d, "gold_usd_oz": 1000 * np.exp(np.cumsum(g)), "usdtry": 2 * np.exp(np.cumsum(f))})
    gvz = pd.Series(100 * np.sqrt(np.roll(sg, -1) * 252), index=d)
    return phys, gvz, panel


class TestComponents(unittest.TestCase):
    def setUp(self):
        self.p, self.g, self.pn = world()

    def test_ewma_recursion_includes_day(self):
        c = vc.ewma_components(self.pn.iloc[:60].reset_index(drop=True))
        g = np.diff(np.log(self.pn["gold_usd_oz"].to_numpy()[:60]))
        f = np.diff(np.log(self.pn["usdtry"].to_numpy()[:60]))
        vf = f[:20].var()
        for i in range(31):
            vf = vc.LAMBDA * vf + (1 - vc.LAMBDA) * f[i] ** 2
        self.assertAlmostEqual(c["var_kur"].iat[30], vf)
        self.assertEqual(c["comp_date"].iat[30], self.pn["Date"].iat[31])        # i. getiri = i+1. gun sonu

    def test_attach_uses_t_and_staleness(self):
        df = vs.frame(self.p, self.g)
        comp = vc.ewma_components(self.pn)
        m = vc.attach(df, comp)
        row = m.iloc[500]
        want = comp.loc[comp["comp_date"] == row["t"], "var_kur"].iat[0]
        self.assertEqual(row["var_kur"], want)
        t0 = m["t"].iat[700]
        gap = comp[comp["comp_date"] != t0]                                     # o gun panelde yok
        m3 = vc.attach(df, gap)
        prev = comp.loc[comp["comp_date"] < t0, "var_kur"].iat[-1]
        self.assertEqual(m3.loc[m3["t"] == t0, "var_kur"].iat[0], prev)          # onceki gun, ertesi degil
        far = comp[comp["comp_date"] < "2010-01-01"]                            # 2010 sonrasi bilesen yok
        m2 = vc.attach(df, far)
        self.assertTrue(m2.loc[m2["t"] > "2010-01-15", "var_kur"].isna().all())

    def test_load_panel_lock(self):
        tmp = Path(tempfile.mkdtemp(prefix="vc_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pd.DataFrame({"Date": ["2022-12-30", "2023-01-02", "2022-12-29"], "gold_usd_oz": [1.0, 2.0, 0.0],
                      "usdtry": [1.0, 1.0, 1.0]}).to_csv(tmp / "p.csv", index=False)
        self.assertEqual(list(vc.load_panel(tmp / "p.csv")["Date"].dt.year), [2022])


class TestForecasts(unittest.TestCase):
    def setUp(self):
        self.p, self.g, self.pn = world()
        self.df = vc.attach(vs.frame(self.p, self.g), vc.ewma_components(self.pn))

    def test_no_future_information(self):
        a = vc.forecasts(self.df, (2015,))
        pn = self.pn.copy()
        pn.loc[pn["Date"] >= "2016-01-01", "usdtry"] *= np.exp(np.cumsum(np.full((pn["Date"] >= "2016-01-01").sum(), 0.05)))
        b_df = vc.attach(vs.frame(self.p, self.g), vc.ewma_components(pn))
        b = vc.forecasts(b_df, (2015,))
        for m in vc.MODELS:
            np.testing.assert_allclose(a[m].to_numpy(), b[m].to_numpy(), err_msg=m)

    def test_calibration_uses_training_only(self):
        a = vc.forecasts(self.df, (2015,))
        tr = self.df[self.df["T"] < "2015-01-01"]
        raw = (tr["var_ons"] + tr["var_kur"] + 2 * tr["kov"])
        ok = raw.notna()
        c = (tr.loc[ok, "r"] ** 2).mean() / raw[ok].mean()
        row = self.df[self.df["T"] == a["T"].iat[5]].iloc[0]
        self.assertAlmostEqual(a["BilesenEWMA"].iat[5], c * (row["var_ons"] + row["var_kur"] + 2 * row["kov"]))

    def test_fx_crises_fixed_by_components(self):
        pred = vc.forecasts(self.df, range(2012, 2023))
        y = {r["year"]: r for r in vs.yearly(pred, vc.MODELS)}
        for Y in (2018, 2021):
            self.assertLess(y[Y]["GVZ_gain"], 0)                                 # GVZ kur krizini gormez
            self.assertGreater(y[Y]["BilesenGVZ_gain"], y[Y]["GVZ_gain"] + 0.5)
            self.assertGreater(y[Y]["BilesenEWMA_gain"], 0.3)


class TestMain(unittest.TestCase):
    def test_outputs(self):
        p, g, pn = world()
        tmp = Path(tempfile.mkdtemp(prefix="vc_out_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(vc.ld, "load_prices", lambda: p), mock.patch.object(vc.vs, "load_gvz", lambda: g), \
                mock.patch.object(vc, "load_panel", lambda: pn), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vc.main(["--out", str(tmp)]), 0)
        res = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("KULLANILMAZ", res["nature"])
        self.assertEqual([m["model"] for m in res["models"]], list(vc.MODELS))
        self.assertEqual({m["model"] for m in res["vs_garch"]}, {"BilesenEWMA", "BilesenGVZ", "Birlesik2"})


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
