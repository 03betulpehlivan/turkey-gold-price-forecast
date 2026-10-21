"""
Kronos denemesi testleri: src/tool/kronos_deneme.py (sahte tahminciyle; Kronos ve PyTorch gerekmez)
Calistirma: python -m unittest tests.test_tool_kronos -v
"""

from pathlib import Path
from unittest import mock
import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tool import kronos_deneme as kd  # noqa: E402
from tests.test_tool_yarisma import world  # noqa: E402


class FakeKronos:
    """Son kapanistan baslayan rastgele yuruyus senaryolari; gordugu baglami kaydeder."""
    def __init__(self):
        self.son_baglam, self.ilk_hedef = [], []

    def predict_batch(self, dfs, xts, yts, pred_len, T, top_p, sample_count, verbose):
        self.top_p = top_p
        out = []
        for df, xt, yt in zip(dfs, xts, yts):
            self.son_baglam.append(pd.Timestamp(xt.iloc[-1]))
            self.ilk_hedef.append(pd.Timestamp(yt.iloc[0]))
            p0 = float(df["close"].iloc[-1])
            assert (df["open"] == df["close"]).all() and len(df) == kd.GERI
            r = np.random.normal(0, 0.01, pred_len)
            out.append(pd.DataFrame({"close": p0 * np.exp(np.cumsum(r))}))
        return out


class TestKronos(unittest.TestCase):
    def test_context_ends_at_info_day_and_quantiles(self):
        p = world(end="2014-12-31")
        idx = np.array([900, 1200, 1500])
        f = FakeKronos()
        q = kd.kronos_yuzdelik(f, p, idx, ornek=8, parti=2, ilerleme=False)
        self.assertEqual(set(q), {1, 2, 3, 4, 5})
        self.assertEqual(q[3].shape, (3, 5))
        for h in q:
            self.assertTrue((np.diff(q[h], axis=1) >= 0).all())
        self.assertEqual(sorted(set(f.son_baglam)), sorted(p.index[idx]))          # baglam bilgi gununde biter
        self.assertTrue(all(h > b for h, b in zip(f.ilk_hedef, f.son_baglam)))
        self.assertGreater(np.mean(q[5][:, 4] - q[5][:, 0]), np.mean(q[1][:, 4] - q[1][:, 0]))
        self.assertEqual(f.top_p, 1.0)                                              # kuyruklar kesilmez

    def test_comparison_structure(self):
        p = world(end="2024-06-28")
        res = kd.deneme(p, FakeKronos(), ornek=5, adim=60)
        self.assertEqual(set(res["ufuklar"]), {"1", "3", "5"})
        r = res["ufuklar"]["3"]
        self.assertEqual(set(r), {"secim", "final"})
        self.assertIn("p_degeri", r["final"]["kronos_arima_garch_a_gore"])
        self.assertIn("kapanis", res["not"])

    def test_missing_install_message(self):
        with mock.patch.dict(os.environ, {"KRONOS_DIR": ""}):
            with self.assertRaisesRegex(RuntimeError, "KRONOS_DIR"):
                kd.yukle()


if __name__ == "__main__":
    unittest.main(verbosity=2)
