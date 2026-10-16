"""
LSTM (src/models/lstm_np.py) ve sizinti gosterimi (src/analysis/leakage_demo.py) testleri
Calistirma: python -m unittest tests.test_leakage_demo -v
Sentetik veri; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi).
"""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import leakage_demo as ld  # noqa: E402
from src.models.lstm_np import LSTM  # noqa: E402

FAST = {"hidden": 4, "epochs": 2, "batch": 256, "lr": 1e-3}


def rw_prices(start="2006-01-02", end="2013-12-31", seed=3):
    d = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.Series(30 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, len(d)))), index=d)


class TestLSTM(unittest.TestCase):
    def test_gradient_check(self):
        rng = np.random.default_rng(0)
        X, t = rng.normal(size=(4, 5, 2)), rng.normal(size=4)
        m = LSTM(2, 3, seed=1)
        _, g = m.loss_grad(X, t)
        worst = 0.0
        for k, P in m.p.items():
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + 1e-6
                lp, _ = m.loss_grad(X, t)
                P[idx] = old - 1e-6
                lm, _ = m.loss_grad(X, t)
                P[idx] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
        self.assertLess(worst, 1e-5)

    def test_learns_and_is_deterministic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(800, 8, 1))
        y = 0.6 * X[:, -1, 0] - 0.3 * X[:, -2, 0]
        a, b = LSTM(1, 8, seed=0), LSTM(1, 8, seed=0)
        h = a.fit(X, y, epochs=20, lr=1e-2, seed=0)
        b.fit(X, y, epochs=20, lr=1e-2, seed=0)
        self.assertLess(h[-1], 0.1 * h[0])
        np.testing.assert_array_equal(a.predict(X[:50]), b.predict(X[:50]))


class TestData(unittest.TestCase):
    def test_lock_and_cleaning(self):
        tmp = Path(tempfile.mkdtemp(prefix="ld_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        df = pd.DataFrame({"Date": ["2005-12-30", "2006-01-03", "2006-01-02", "2010-05-05", "2022-12-30", "2023-01-02"],
                           "kapanis_tl_gr": [1.0, 2.0, 3.0, 0.0, 4.0, 999.0]})
        df.to_csv(tmp / "p.csv", index=False)
        s = ld.load_prices(tmp / "p.csv")
        self.assertEqual([str(d.date()) for d in s.index], ["2006-01-02", "2006-01-03", "2022-12-30"])
        self.assertNotIn(999.0, s.values)

    def test_windows(self):
        X, y = ld.windows(np.arange(25.0), 20)
        self.assertEqual(X.shape, (5, 20))
        np.testing.assert_array_equal(X[0], np.arange(20.0))
        self.assertEqual(y[0], 20.0)
        self.assertTrue(np.all(X[:, -1] + 1 == y))                              # hedef pencerede yok


class TestSetups(unittest.TestCase):
    def setUp(self):
        self.p = rw_prices()

    def test_level_setups_show_meaningless_r2(self):
        for chrono in (False, True):
            rows = {r["model"]: r for r in ld.setup_level(self.p, chrono, FAST, models=("Dogrusal",))}
            self.assertGreater(rows["Yarin=bugun"]["r2_level"], 0.95)            # R2 seviyede anlamsiz derecede yuksek
            self.assertEqual(rows["Yarin=bugun"]["vs_rw_pct"], 0.0)

    def test_walk_forward_uses_only_past(self):
        a = ld.walk_forward_returns(self.p, (2012,), FAST, models=("Dogrusal", "LSTM"))
        q = self.p.copy()
        q[q.index >= "2013-01-01"] *= 3.0                                        # gelecek bozulur
        b = ld.walk_forward_returns(q, (2012,), FAST, models=("Dogrusal", "LSTM"))
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a["date"].dt.year == 2012).all())
        c = self.p.copy()
        c[(c.index >= "2012-06-01") & (c.index < "2013-01-01")] *= 1.5          # test yilinin kendisi
        d = ld.walk_forward_returns(c, (2012,), FAST, models=("Dogrusal",))
        first = d["date"] < pd.Timestamp("2012-05-01")
        np.testing.assert_allclose(a.loc[first, "Dogrusal"].to_numpy(), d.loc[first, "Dogrusal"].to_numpy())

    def test_scaling_uses_training_period_only(self):
        a = ld.walk_forward_returns(self.p, (2011,), FAST, models=("Dogrusal",))
        self.assertAlmostEqual(float(a["Ortalama"].iat[0]),
                               float(np.diff(np.log(self.p[self.p.index < "2011-01-01"].to_numpy()))[ld.WIN:].mean()))

    def test_summary_and_outputs(self):
        res = ld.run_all(self.p, eval_years=(2012, 2013), lstm_cfg=FAST, models=("Dogrusal", "LSTM"))
        k3 = {r["model"]: r for r in res["K3_dogru_walk_forward"]}
        self.assertEqual(k3["Yarin=bugun"]["vs_rw_pct"], 0.0)
        self.assertEqual(k3["Yarin=bugun"]["oos_r2_return"], 0.0)
        for m in ("Ortalama", "Dogrusal", "LSTM"):
            self.assertIn("dm_p", k3[m])
            self.assertLess(abs(k3[m]["vs_rw_pct"]), 10)                         # rastgele yuruyuste kazanc yok
        self.assertIn("KULLANILMAZ", res["nature"])
        self.assertEqual(res["config"]["eval_years"], [2012, 2013])
        tmp = Path(tempfile.mkdtemp(prefix="ld_out_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(ld, "load_prices", lambda: self.p), \
                mock.patch.object(ld, "run_all", lambda prices: res), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ld.main(["--out", str(tmp)]), 0)
        self.assertEqual(json.loads((tmp / "summary.json").read_text(encoding="utf-8"))["K3_n"], res["K3_n"])
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
