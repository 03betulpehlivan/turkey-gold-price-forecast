"""
Haftalik/aylik trend ve makro calismasi testleri: src/analysis/trend_study.py
Calistirma: python -m unittest tests.test_trend_study -v
Sentetik veri; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi).
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

from src.analysis import trend_study as ts  # noqa: E402


def world(seed=2, start="2005-06-01", end="2022-12-30"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, end)
    n = len(d)
    p = pd.Series(20 * np.exp(np.cumsum(rng.normal(0.0006, 0.011, n))), index=d)
    panel = pd.DataFrame({"Date": d, "usdtry": 1.4 * np.exp(np.cumsum(rng.normal(0.0004, 0.006, n))),
                          "gold_usd_oz": 600 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))),
                          "us10y": 3 + np.cumsum(rng.normal(0, 0.03, n))})
    months = pd.period_range("2004-01", "2022-12", freq="M")
    macro = pd.DataFrame({"month": [str(m) for m in months],
                          "tufe": 100 * np.exp(np.cumsum(rng.normal(0.007, 0.004, len(months)))),
                          "politika_tr": np.clip(10 + np.cumsum(rng.normal(0, 0.5, len(months))), 1, 50),
                          "politika_us": 1.0})
    return p, panel, macro


class TestMacroTiming(unittest.TestCase):
    def setUp(self):
        _, _, self.macro = world()
        self.mm = ts.macro_monthly(self.macro)

    def test_publication_lag(self):
        f = ts.asof_macro(pd.Series(pd.to_datetime(["2020-03-04", "2020-03-05", "2020-03-31"])), self.mm)
        self.assertEqual(list(f["cpi_month"]), ["2020-01", "2020-02", "2020-02"])
        m2 = self.macro.copy()
        m2.loc[m2["month"] == "2020-02", "tufe"] *= 2                      # subat TUFE'si
        m2.loc[m2["month"] == "2020-03", "politika_tr"] += 20               # mart faizi
        g = ts.asof_macro(pd.Series(pd.to_datetime(["2020-03-04", "2020-03-31"])), ts.macro_monthly(m2))
        self.assertEqual(f["reel"].iat[0], g["reel"].iat[0])                # 4 mart: subat TUFE'si henuz yok
        self.assertNotEqual(f["reel"].iat[2], g["reel"].iat[1])             # 31 mart: subat TUFE'si yayimlandi

    def test_real_rate_definition(self):
        f = ts.asof_macro(pd.Series(pd.to_datetime(["2015-06-30"])), self.mm)
        m = self.macro.set_index("month")
        enf = 100 * (m.loc["2015-05", "tufe"] / m.loc["2014-05", "tufe"] - 1)
        self.assertAlmostEqual(f["reel"].iat[0], m.loc["2015-05", "politika_tr"] - enf)

    def test_lock(self):
        m = pd.concat([self.macro, pd.DataFrame({"month": ["2023-01"], "tufe": [1e9], "politika_tr": [99.0], "politika_us": [1.0]})])
        self.assertNotIn(pd.Period("2023-01", "M"), ts.macro_monthly(m).index)


class TestPeriods(unittest.TestCase):
    def setUp(self):
        self.p, self.panel, self.macro = world()
        self.mm = ts.macro_monthly(self.macro)

    def test_period_ends_and_target(self):
        for h, k in (("hafta", 13), ("ay", 12)):
            df = ts.periods(self.p, h, self.panel, self.mm)
            self.assertTrue((df["d"].isin(self.p.index)).all())
            nxt = df["d"].shift(-1)
            same = df["d_next"].iloc[:-1] == nxt.iloc[:-1]
            self.assertTrue(same.all() or same.mean() > 0.95)               # ozellik eksigi bosluk yaratabilir
            r = np.log(self.p.loc[df["d_next"]].to_numpy() / self.p.loc[df["d"]].to_numpy())
            np.testing.assert_allclose(df["y"].to_numpy(), r)
            self.assertTrue((df["d_next"] < pd.Timestamp("2023-01-01")).all())
        m = ts.periods(self.p, "ay", self.panel, self.mm)
        self.assertTrue((m["d"].dt.to_period("M") != m["d_next"].dt.to_period("M")).all())

    def test_missing_panel_day_uses_previous_not_next(self):
        a = ts.periods(self.p, "ay", self.panel, self.mm)
        d0 = a["d"].iat[40]
        pn = self.panel[self.panel["Date"] != d0].copy()                    # o gun panelde yok (ABD tatili gibi)
        nxt = pn.loc[pn["Date"] > d0, "Date"].min()
        pn.loc[pn["Date"] == nxt, "usdtry"] *= 5                             # ertesi gun cok farkli
        b = ts.periods(self.p, "ay", pn, self.mm)
        prev = self.panel.loc[self.panel["Date"] < d0, "usdtry"].iat[-1]
        before = a["d"].iat[39]
        want = np.log(prev / self.panel.loc[self.panel["Date"] == before, "usdtry"].iat[0])
        self.assertAlmostEqual(float(b.loc[b["d"] == d0, "fx1"].iat[0]), float(want))

    def test_features_use_no_future(self):
        a = ts.periods(self.p, "ay", self.panel, self.mm)
        cut = pd.Timestamp("2015-06-30")
        pn = self.panel.copy()
        pn.loc[pn["Date"] > cut, ["usdtry", "gold_usd_oz", "us10y"]] *= 3
        p2 = self.p.copy()
        p2[p2.index > cut] *= 2
        b = ts.periods(p2, "ay", pn, self.mm)
        mask_a = a["d"] <= cut
        pd.testing.assert_frame_equal(a.loc[mask_a, ts.FEATURES].reset_index(drop=True),
                                      b.loc[b["d"] <= cut, ts.FEATURES].reset_index(drop=True))


def signal_frame(n=264, beta=0.6, seed=5):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2000-01-31", periods=n + 1, freq="ME")
    X = rng.normal(size=(n, len(ts.FEATURES)))
    y = 0.03 * (beta * X[:, 2] + rng.normal(0, 1, n))
    df = pd.DataFrame(X, columns=ts.FEATURES)
    df["d"], df["d_next"], df["P"], df["y"] = d[:-1], d[1:], 100.0, y
    return df


class TestModels(unittest.TestCase):
    def test_signal_detected(self):
        pred = ts.walk_forward(signal_frame(beta=1.0), range(2012, 2022))
        s = ts.summarize(pred)
        r = {x["model"]: x for x in s["returns"]}
        dd = {x["model"]: x for x in s["direction"]}
        self.assertGreater(r["Dogrusal"]["vs_rw_pct"], 5)
        self.assertLess(r["Dogrusal"]["dm_p_vs_rw"], 0.05)
        self.assertGreater(dd["Lojistik"]["brier_skill_vs_iklim"], 0.05)
        self.assertLess(dd["Lojistik"]["dm_p_vs_iklim"], 0.05)

    def test_no_false_discovery(self):
        pred = ts.walk_forward(signal_frame(beta=0.0, seed=8), range(2012, 2022))
        s = ts.summarize(pred)
        lg = [x for x in s["direction"] if x["model"] == "Lojistik"][0]
        self.assertFalse(lg["brier_skill_vs_iklim"] > 0 and lg["dm_p_vs_iklim"] < 0.05)

    def test_walk_forward_uses_only_past(self):
        df = signal_frame()
        a = ts.walk_forward(df, (2015,))
        b_df = df.copy()
        b_df.loc[b_df["d_next"] >= "2016-01-01", ts.FEATURES + ["y"]] = 9.0
        pd.testing.assert_frame_equal(a, ts.walk_forward(b_df, (2015,)))
        self.assertTrue((a["d_next"].dt.year == 2015).all())
        tr = df[df["d_next"] < "2015-01-01"]
        self.assertAlmostEqual(a["p_Iklim"].iat[0], float((tr["y"] > 0).mean()))
        self.assertAlmostEqual(a["g_Ortalama"].iat[0], float(tr["y"].mean()))


class TestEndToEnd(unittest.TestCase):
    def test_run_and_main(self):
        p, panel, macro = world()
        res = ts.run_all(p, panel, macro, eval_years=(2020, 2021, 2022))
        self.assertIn("KULLANILMAZ", res["nature"])
        self.assertEqual(res["ay"]["n"], 36)
        self.assertEqual({r["model"] for r in res["ay"]["direction"]}, {"Iklim", "Lojistik", "Hep_yukari"})
        tmp = Path(tempfile.mkdtemp(prefix="ts_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(ts.ld, "load_prices", lambda: p), mock.patch.object(ts, "load_panel", lambda: panel), \
                mock.patch.object(ts.mc, "load", lambda: macro), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ts.main(["--out", str(tmp)]), 0)
        self.assertEqual(json.loads((tmp / "summary.json").read_text(encoding="utf-8"))["ay"]["n"], 132)
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
