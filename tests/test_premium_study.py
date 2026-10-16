"""
Fiziki prim calismasi testleri: src/analysis/premium_study.py
Calistirma: python -m unittest tests.test_premium_study -v
Sentetik prim dunyalari (kalici, ortalamaya donen, gurultulu); gecici klasorler. Gercek veri kullanilmaz.
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

from src.analysis import premium_study as ps  # noqa: E402


def world(phi_daily=1.0, noise=0.0, step=0.08, seed=1, start="2006-01-02", end="2022-12-30"):
    """Gunluk prim: AR(1) (phi_daily; 1 = rastgele yuruyus) + bagimsiz olcum gurultusu (yuzde puan)."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, end)
    n = len(d)
    p = np.empty(n)
    p[0] = 0.0
    for i in range(1, n):
        p[i] = phi_daily * p[i - 1] + rng.normal(0, step)
    obs = p + rng.normal(0, noise, n)
    theo = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=d)
    phys = theo * np.exp(obs / 100)
    return phys, theo


class TestBasics(unittest.TestCase):
    def test_alignment_lock_and_weekly_rule(self):
        d = pd.to_datetime(["2022-12-26", "2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30", "2023-01-02",
                            "2023-01-03"])
        theo = pd.Series(100.0, index=d)
        phys = pd.Series([101.0, 101.0, 101.0, np.nan, 102.0, 500.0, 500.0], index=d).dropna()
        dp = ps.daily_premium(phys, theo.drop(pd.Timestamp("2022-12-27")))
        self.assertEqual(len(dp), 3)                                             # ortak gunler, 2023 yok
        self.assertAlmostEqual(dp.iat[0], 100 * math.log(1.01))
        w = ps.weekly_premium(dp)
        self.assertEqual(len(w), 1)                                              # 3 gun -> hafta sayilir
        self.assertEqual(w.index[0], pd.Timestamp("2022-12-30"))
        self.assertEqual(len(ps.weekly_premium(dp.iloc[:2])), 0)                 # 2 gun -> sayilmaz

    def test_ar1_and_half_life(self):
        rng = np.random.default_rng(3)
        x = np.empty(4000)
        x[0] = 0
        for i in range(1, len(x)):
            x[i] = 1.0 + 0.5 * (x[i - 1] - 1.0) + rng.normal(0, 0.1)
        a = ps.ar1(x)
        self.assertAlmostEqual(a["phi"], 0.5, delta=0.05)
        self.assertAlmostEqual(a["mu"], 1.0, delta=0.05)
        self.assertAlmostEqual(a["half_life"], math.log(0.5) / math.log(a["phi"]))
        self.assertEqual(ps.ar1(np.exp(0.01 * np.arange(50)))["half_life"], float("inf"))   # phi >= 1
        lo, hi = ps.bootstrap_phi(x, reps=300)
        self.assertEqual([lo, hi], ps.bootstrap_phi(x, reps=300))                 # sabit tohum: tekrarlanabilir
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class TestNoise(unittest.TestCase):
    def test_measurement_noise_fakes_daily_reversion_weekly_reduces_it(self):
        phys, theo = world(phi_daily=1.0, noise=1.0, step=0.1)                   # gercek prim kalici (phi = 1)
        d = ps.daily_premium(phys, theo)
        w = ps.weekly_premium(d)
        daily_phi, weekly_phi = ps.ar1(d.to_numpy())["phi"], ps.ar1(w.to_numpy())["phi"]
        self.assertLess(daily_phi, 0.85)                                         # sahte hizli donus
        self.assertGreater(weekly_phi, daily_phi + 0.1)                          # haftalik ortalama azaltir
        self.assertGreater(weekly_phi, 0.9)


class TestForecast(unittest.TestCase):
    def test_mean_reversion_detected(self):
        phys, theo = world(phi_daily=0.93, noise=0.0, step=0.15)                  # haftalik phi ~ 0.7
        res = ps.run_all(phys, theo)
        f = {r["model"]: r for r in res["forecast_next_week"]}
        self.assertGreater(f["AR1"]["vs_rw_pct"], 3)
        self.assertLess(f["AR1"]["dm_p_vs_rw"], 0.05)
        lo, hi = res["ar1_weekly"]["phi_ci95"]
        self.assertLess(hi, 1.0)
        self.assertLess(res["ar1_weekly"]["half_life"], 10)

    def test_no_false_discovery_for_persistent_premium(self):
        phys, theo = world(phi_daily=1.0, noise=0.0, step=0.1, seed=7)
        res = ps.run_all(phys, theo)
        f = {r["model"]: r for r in res["forecast_next_week"]}
        self.assertFalse(f["AR1"]["vs_rw_pct"] > 0 and f["AR1"]["dm_p_vs_rw"] < 0.05)
        self.assertLess(f["Ortalama"]["vs_rw_pct"], 0)

    def test_walk_forward_uses_only_past(self):
        phys, theo = world(phi_daily=0.95, step=0.15)
        w = ps.weekly_premium(ps.daily_premium(phys, theo))
        a = ps.walk_forward(w, (2015,))
        w2 = w.copy()
        w2[w2.index >= "2016-01-01"] += 50.0
        pd.testing.assert_frame_equal(a, ps.walk_forward(w2, (2015,)))
        self.assertTrue((a["d_next"].dt.year == 2015).all())
        tr = w[w.index < "2015-01-01"]
        self.assertAlmostEqual(a["Ortalama"].iat[0], float(tr.iloc[1:].mean()))


class TestMain(unittest.TestCase):
    def test_outputs(self):
        phys, theo = world(phi_daily=0.95, step=0.15)
        tmp = Path(tempfile.mkdtemp(prefix="ps_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(ps.ld, "load_prices", lambda: phys), mock.patch.object(ps, "load_theoretical", lambda: theo), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.main(["--out", str(tmp)]), 0)
        res = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("KULLANILMAZ", res["nature"])
        self.assertEqual(len(res["yearly"]), 17)
        self.assertEqual(res["config"]["eval_years"], list(range(2012, 2023)))


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
