"""
Oynaklik tahmin calismasi testleri: src/analysis/volatility_study.py
Calistirma: python -m unittest tests.test_volatility_study -v
Sentetik GARCH ve sabit oynaklik dunyalari; gecici klasorler. Gercek data/ ve research/ kullanilmaz.
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

from src.analysis import volatility_study as vs  # noqa: E402


def world(garch=True, seed=1, start="2006-01-02", end="2014-12-31"):
    d = pd.bdate_range(start, end)
    n, rng = len(d), np.random.default_rng(seed)
    w, a, b = 1e-6, 0.08, 0.9
    v, r, s = w / (1 - a - b), np.empty(n), np.empty(n)
    for i in range(n):
        s[i] = v if garch else w / (1 - a - b)
        r[i] = np.sqrt(s[i]) * rng.normal()
        if garch:
            v = w + a * r[i] ** 2 + b * v
    p = pd.Series(30 * np.exp(np.cumsum(r)), index=d)
    g = pd.Series(100 * np.sqrt(np.roll(s, -1) * 252) * np.exp(rng.normal(0, 0.1, n)), index=d)
    return p, g


class TestFrame(unittest.TestCase):
    def test_alignment_staleness_and_lock(self):
        d = pd.to_datetime(["2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30", "2023-01-02"])
        p = pd.Series([10.0, 11.0, 12.0, 13.0, 99.0], index=d)
        g = pd.Series([20.0, 21.0, 50.0], index=pd.to_datetime(["2022-12-20", "2022-12-28", "2023-01-02"]))
        f = vs.frame(p, g)
        self.assertEqual(len(f), 3)                                              # 2023 satiri yok
        self.assertTrue((f["T"] < pd.Timestamp("2023-01-01")).all())
        self.assertTrue(np.isnan(f["gvz"].iat[0]))                               # t=12-27: son GVZ 7 gun once
        self.assertEqual(list(f["gvz"].iloc[1:]), [21.0, 21.0])                  # t gunu veya oncesi, T degil
        np.testing.assert_allclose(f["r"], np.diff(np.log([10.0, 11.0, 12.0, 13.0])))

    def test_load_gvz_lock(self):
        tmp = Path(tempfile.mkdtemp(prefix="vs_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pd.DataFrame({"Date": ["2022-12-30", "2023-01-03", "2021-01-04"], "gvz": [20.0, 30.0, 0.0]}).to_csv(tmp / "p.csv", index=False)
        s = vs.load_gvz(tmp / "p.csv")
        self.assertEqual(list(s.values), [20.0])


class TestForecasts(unittest.TestCase):
    def setUp(self):
        self.p, self.g = world()
        self.df = vs.frame(self.p, self.g)

    def test_simple_estimators_use_only_past(self):
        pred = vs.forecasts(self.df, (2012,))
        r2 = self.df["r"].to_numpy() ** 2
        i = int(self.df.index[self.df["T"] == pred["T"].iat[10]][0])
        self.assertAlmostEqual(pred["Kayan20"].iat[10], r2[i - 20:i].mean())
        v = r2[:vs.ROLL].mean()
        for k in range(i):
            v = vs.LAMBDA * v + (1 - vs.LAMBDA) * r2[k]
        self.assertAlmostEqual(pred["EWMA"].iat[10], v)
        self.assertAlmostEqual(pred["Sabit"].iat[0], r2[(self.df["T"] < "2012-01-01").to_numpy()].mean())

    def test_no_future_information(self):
        a = vs.forecasts(self.df, (2012,))
        b_df = self.df.copy()
        late = b_df["T"] >= pd.Timestamp("2013-01-01")
        b_df.loc[late, "r"] *= 5.0
        b_df.loc[late, "gvz"] *= 3.0
        pd.testing.assert_frame_equal(a, vs.forecasts(b_df, (2012,)))
        c_df = self.df.copy()
        mid = (c_df["T"] >= "2012-07-01") & (c_df["T"] < "2013-01-01")
        c_df.loc[mid, "r"] *= 5.0                                                 # test yilinin ikinci yarisi
        c = vs.forecasts(c_df, (2012,))
        early = (a["T"] < pd.Timestamp("2012-06-01")).to_numpy()
        for m in vs.MODELS:
            np.testing.assert_allclose(a.loc[early, m].to_numpy(), c.loc[early, m].to_numpy(), err_msg=m)

    def test_forecast_for_a_day_ignores_that_days_return(self):
        a = vs.forecasts(self.df, (2012,))
        k = int(self.df.index[self.df["T"] == a["T"].iat[100]][0])
        d = self.df.copy()
        d.loc[k, "r"] *= 8.0                                                     # yalnizca o gunun getirisi
        b = vs.forecasts(d, (2012,))
        for m in vs.MODELS:
            np.testing.assert_allclose(a[m].iloc[:101].to_numpy(), b[m].iloc[:101].to_numpy(), err_msg=m)
        self.assertNotAlmostEqual(a["GARCH"].iat[101], b["GARCH"].iat[101])       # ertesi gun etkilenir

    def test_qlike(self):
        self.assertAlmostEqual(float(vs.qlike(np.array([0.02]), np.array([0.0004]))[0]), np.log(0.0004) + 1.0)


class TestWorlds(unittest.TestCase):
    def test_volatility_clustering_detected(self):
        p, g = world(garch=True, end="2018-12-31")
        res = vs.run_all(p, g, eval_years=range(2012, 2019))
        m = {r["model"]: r for r in res["models"]}
        for k in ("EWMA", "GARCH", "GVZ", "Birlesik"):
            self.assertGreater(m[k]["qlike_gain_vs_sabit"], 0, k)
            self.assertLess(m[k]["dm_p_vs_sabit"], 0.05, k)
        self.assertLess(m["Sabit"]["christoffersen80_p"], 0.05)                  # sabit aralik ihlalleri kumelenir
        self.assertGreater(m["GARCH"]["christoffersen80_p"], 0.05)
        self.assertEqual(len(res["yearly"]), 7)

    def test_no_false_discovery_when_volatility_is_constant(self):
        p, g = world(garch=False, end="2018-12-31", seed=4)
        res = vs.run_all(p, g, eval_years=range(2012, 2019))
        for r in res["models"]:
            if r["model"] != "Sabit":
                self.assertFalse(r["qlike_gain_vs_sabit"] > 0 and r["dm_p_vs_sabit"] < 0.05, r["model"])

    def test_main_writes_outputs(self):
        p, g = world()
        tmp = Path(tempfile.mkdtemp(prefix="vs_out_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(vs.ld, "load_prices", lambda: p), mock.patch.object(vs, "load_gvz", lambda: g), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vs.main(["--out", str(tmp)]), 0)
        res = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("KULLANILMAZ", res["nature"])
        self.assertEqual([m["model"] for m in res["models"]], list(vs.MODELS))
        self.assertTrue((tmp / "table.csv").exists())


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
