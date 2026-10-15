"""
Oynakliga duyarli aralik testleri: src/models/interval_study.py
Calistirma: python -m unittest tests.test_interval_study -v
Sentetik artiklar: 80 gunluk sakin (%0.4) / calkantili (%1.6) bloklar; bazi gunlerde dunku sonuc yayimlanmamis.
"""

from pathlib import Path
from unittest import mock
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import interval_study as ist  # noqa: E402
from src.models import train as tr  # noqa: E402


def frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    td = pd.bdate_range("2012-01-02", periods=n)
    vol = np.where((np.arange(n) // 80) % 2 == 0, 0.004, 0.016)
    e = rng.normal(0, vol)
    pred = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    last = [td[i - 1] if i % 10 else td[i - 2] for i in range(n)]
    last[0] = td[0] - pd.Timedelta(days=1)
    df = pd.DataFrame({"target_date": td, "last_date": pd.to_datetime(last), "actual": pred * np.exp(e)})
    for m in ist.MODELS:
        df[f"pred_{m}"] = pred
    return df


class TestIntervals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = frame()
        cls.iv = ist.intervals_for(cls.df, "Bosluk")
        cls.t = pd.DataFrame(ist.score(cls.iv, "Bosluk")).set_index(["method", "level"])

    def test_scaling_removes_clustering_and_improves_winkler(self):
        t = self.t
        self.assertLess(t.loc[("Statik", 80), "christoffersen_p"], 0.01)
        self.assertGreater(t.loc[("Olcekli", 80), "christoffersen_p"], t.loc[("Statik", 80), "christoffersen_p"])
        self.assertLess(t.loc[("Olcekli", 80), "winkler_TL"], t.loc[("Statik", 80), "winkler_TL"])
        self.assertTrue(0.74 <= t.loc[("Olcekli", 80), "coverage"] <= 0.86)
        self.assertTrue(0.92 <= t.loc[("Olcekli", 95), "coverage"] <= 0.98)

    def test_only_known_results_used(self):
        r = 900
        bad = self.df.copy()
        bad.loc[r:, "actual"] *= 1.3
        iv2 = ist.intervals_for(bad, "Bosluk")
        cols = [c for c in self.iv.columns if c.startswith(("lo", "hi"))]
        pd.testing.assert_frame_equal(self.iv.loc[:r, cols], iv2.loc[:r, cols])

    def test_interval_invariant_to_everything_unknown_at_issue(self):
        td = pd.to_datetime(self.df["target_date"]).to_numpy()
        m = ist.known_counts(td, pd.to_datetime(self.df["last_date"]).to_numpy())
        cols = [c for c in self.iv.columns if c.startswith(("lo", "hi"))]
        for i in (600, 605, 700, 705):                   # 600/700: dunku sonuc yayimlanmamis gunler
            bad = self.df.copy()
            bad.loc[m[i]:, "actual"] = bad.loc[m[i]:, "actual"] * 10      # o sabah bilinmeyen HER SEY
            iv2 = ist.intervals_for(bad, "Bosluk")
            pd.testing.assert_series_equal(self.iv.loc[i, cols], iv2.loc[i, cols], check_names=False)

    def test_standardization_uses_previous_volatility(self):
        e = np.log(self.df["actual"] / self.df["pred_Bosluk"]).to_numpy()
        v = ist.ewma_var(e)
        j = 500
        self.assertAlmostEqual(v[j], ist.EWMA_LAMBDA * v[j - 1] + (1 - ist.EWMA_LAMBDA) * e[j] ** 2)
        src = Path(ist.__file__).read_text(encoding="utf-8")
        self.assertIn("z = e / s_prev", src)
        s_prev = np.sqrt(np.concatenate([[np.mean(e[:ist.INIT_N] ** 2)], v[:-1]]))
        self.assertAlmostEqual(s_prev[j], np.sqrt(v[j - 1]))

    def test_unpublished_yesterday_not_known(self):
        td = pd.to_datetime(self.df["target_date"]).to_numpy()
        m = ist.known_counts(td, pd.to_datetime(self.df["last_date"]).to_numpy())
        self.assertEqual(m[20], 19)            # i % 10 == 0: son yayimlanan T-2 (satir 18) -> 0..18 bilinir, dunku (19) degil
        self.assertEqual(m[21], 21)            # normal gun: 0..20 bilinir
        self.assertTrue((m <= np.arange(len(m))).all())

    def test_aci_direction(self):
        self.assertGreater(ist.aci_update(1.0, True, 0.2), 1.0)
        self.assertLess(ist.aci_update(1.0, False, 0.2), 1.0)

    def test_locked_input_refused(self):
        tmp = Path(tempfile.mkdtemp(prefix="ist_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        df = self.df.copy()
        df.loc[len(df) - 1, "target_date"] = pd.Timestamp("2023-01-03")
        p = tmp / "f.csv"
        df.to_csv(p, index=False, date_format="%Y-%m-%d")
        with mock.patch.object(tr, "OUTPUT_DIR", tmp / "results"):
            with self.assertRaisesRegex(ist.IntervalStudyError, "kilitli"):
                ist.run({"ayni_gun": p})


if __name__ == "__main__":
    unittest.main(verbosity=2)
