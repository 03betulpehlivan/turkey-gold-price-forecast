"""
Merkez tahmin calismasi testleri: src/tool/merkez.py
Calistirma: python -m unittest tests.test_tool_merkez -v
"""

from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tool import merkez as mz  # noqa: E402
from src.tool import yarisma as yr  # noqa: E402


def dunya(phi=0.0, seed=11, end="2025-06-30"):
    """GARCH oynakligi; phi > 0 ise yarinki getiri bugunkune baglidir (ongorulebilir yapi)."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2008-01-02", end)
    v, r = 1e-4, np.zeros(len(d))
    for i in range(1, len(d)):
        e = np.sqrt(v) * rng.standard_t(6) * np.sqrt(4 / 6)
        r[i] = phi * r[i - 1] + e
        v = 2e-6 + 0.07 * e ** 2 + 0.9 * v
    return pd.Series(50 * np.exp(np.cumsum(r)), index=d)


class TestMerkez(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.rw = mz.calisma(dunya(0.0), 1, "ARIMA-GARCH", chronos_ok=False)
        cls.ar = mz.calisma(dunya(0.25, seed=12), 1, "ARIMA-GARCH", chronos_ok=False)

    def test_random_walk_no_signal(self):
        f = self.rw["formul"]
        p = dict(zip(f["ozellikler"], f["p"]))
        self.assertGreater(p["son_getiri"], 0.01)                                 # rastgele yuruyuste iliski yok
        fin = self.rw["yontemler"]["Ozel formul"]["final"]
        self.assertLess(abs(fin["yarin_bugune_gore_pct"]), 3.0)                   # yarin = bugun ile ayni
        self.assertGreater(self.rw["yontemler"]["Araligin ortasi"]["final"]["p_degeri"], 0.05)   # anlamli fark yok

    def test_predictable_world_detected(self):
        f = self.ar["formul"]
        p = dict(zip(f["ozellikler"], f["p"]))
        b = dict(zip(f["ozellikler"], f["katsayi"]))
        self.assertLess(min(p["son_getiri"], p["model_orta"]), 0.01)              # iliski bulunur (biri ya da digeri)
        fin = self.ar["yontemler"]["Ozel formul"]["final"]
        self.assertGreater(fin["yarin_bugune_gore_pct"], 1.0)
        self.assertLess(fin["p_degeri"], 0.05)
        self.assertGreater(fin["yon_isabeti_pct"], 55)

    def test_structure_and_midpoint(self):
        r = self.rw
        self.assertEqual(set(r["yontemler"]), set(mz.YONTEM))
        self.assertEqual({y["yil"] for y in r["yillar"]} >= {2012, 2023}, True)
        self.assertEqual(len(r["formul"]["katsayi"]), len(mz.OZELLIK))
        for m in mz.YONTEM:
            self.assertIn("final", r["yontemler"][m])

    def test_formula_trained_only_on_selection(self):
        p = dunya(0.0, seed=13)
        a = mz.calisma(p, 1, "ARIMA-GARCH", chronos_ok=False)["formul"]["katsayi"]
        p2 = p.copy()
        p2[p2.index >= "2023-06-01"] *= np.exp(np.cumsum(np.full((p2.index >= "2023-06-01").sum(), 0.01)))
        b = mz.calisma(p2, 1, "ARIMA-GARCH", chronos_ok=False)["formul"]["katsayi"]
        np.testing.assert_allclose(a, b, rtol=1e-9)

    def test_nw_matches_ols_point_estimate(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(500), rng.normal(size=500)])
        y = 0.5 + 2 * X[:, 1] + rng.normal(size=500)
        r = mz.ekk_nw(X, y, lag=3)
        self.assertAlmostEqual(r["beta"][1], 2, delta=0.15)
        self.assertLess(r["p"][1], 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
