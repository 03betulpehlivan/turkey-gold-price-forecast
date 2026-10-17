"""
Guc analizi testleri: src/analysis/power_study.py
Calistirma: python -m unittest tests.test_power_study -v
Sentetik hata serileri; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi).
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

from src.analysis import power_study as ps  # noqa: E402


def errors(n=2000, signal=0.6, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * 0.008
    g = signal * r + rng.normal(0, 0.006, n)
    price = np.linspace(20, 1000, n)
    return np.abs(r - g) * price, np.abs(r) * price, price


class TestPowerCurve(unittest.TestCase):
    def test_monotone_size_and_mde(self):
        em, er, p = errors()
        res = ps.power_curve(em, er, p)
        pw = [c["power"] for c in res["curve"]]
        self.assertEqual(pw, sorted(pw))                                             # etki buyudukce guc artar
        self.assertLessEqual(res["size_at_zero"], 0.1)                               # yanlis alarm dusuk
        self.assertEqual(res["curve"][-1]["power"], 1.0)
        want = min(c["effect_pct"] for c in res["curve"] if c["power"] >= 0.8 and c["effect_pct"] > 0)
        self.assertEqual(res["mde80_pct"], want)
        self.assertGreaterEqual(res["mde80_holm3_pct"], res["mde80_pct"])            # Holm daha ihtiyatli

    def test_scale_invariance_and_observed(self):
        em, er, p = errors()
        a = ps.power_curve(em, er, p)
        b = ps.power_curve(em * 7, er * 7, p * 7)
        self.assertEqual(a["curve"], b["curve"])
        rel = (em - er) / p
        self.assertAlmostEqual(a["observed"]["full_period_improvement_pct"], -100 * rel.mean() / (er / p).mean())

    def test_windows(self):
        w = ps.windows(250, 120, 10)
        self.assertEqual(len(w), 14)
        self.assertEqual((w[0][0], w[-1][-1]), (0, 249))


class TestIntervals(unittest.TestCase):
    def test_kupiec_size_and_power(self):
        k = ps.kupiec_power(reps=1500)
        r80 = {x["true_miss_rate"]: x["power"] for x in k["80"]}
        self.assertAlmostEqual(r80[0.20], 0.05, delta=0.025)                        # dogru kapsamda yanlis alarm
        self.assertGreater(r80[0.40], 0.95)
        self.assertGreater(r80[0.30], r80[0.25])

    def test_christoffersen_and_markov(self):
        rng = np.random.default_rng(3)
        h = ps.markov_hits(20000, 0.2, 0.5, rng)
        self.assertAlmostEqual(h.mean(), 0.2, delta=0.02)
        self.assertAlmostEqual(h[1:][h[:-1] == 1].mean(), 0.5, delta=0.03)
        c = {x["p_violation_after_violation"]: x["power"] for x in ps.christoffersen_power(reps=1500)}
        self.assertAlmostEqual(c[0.2], 0.05, delta=0.03)
        self.assertGreater(c[0.6], c[0.4])


class TestData(unittest.TestCase):
    def test_theoretical_lock_and_past_only(self):
        tmp = Path(tempfile.mkdtemp(prefix="pw_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        d = pd.bdate_range("2010-01-04", "2023-03-31")
        rng = np.random.default_rng(2)
        p = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(d))))
        pd.DataFrame({"Date": d, "theoretical_gram_try": p}).to_csv(tmp / "a.csv", index=False)
        a = ps.theoretical_dev_errors(tmp / "a.csv")
        self.assertLess(a["date"].max(), pd.Timestamp("2023-01-01"))
        self.assertGreaterEqual(a["date"].min(), pd.Timestamp("2012-01-01"))
        p2 = p.copy()
        p2[d >= "2018-01-01"] *= 3
        pd.DataFrame({"Date": d, "theoretical_gram_try": p2}).to_csv(tmp / "b.csv", index=False)
        b = ps.theoretical_dev_errors(tmp / "b.csv")
        early = a["date"] < "2017-12-29"
        pd.testing.assert_frame_equal(a[early], b[early])                           # gecmis tahmin gelecegi gormez


class TestMain(unittest.TestCase):
    def test_outputs(self):
        em, er, p = errors(800)
        fr = pd.DataFrame({"date": pd.bdate_range("2012-01-02", periods=800), "model": em, "ref": er, "price": p})
        tmp = Path(tempfile.mkdtemp(prefix="pw_out_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(ps, "physical_dev_errors", lambda: fr), mock.patch.object(ps, "theoretical_dev_errors", lambda: fr), \
                mock.patch.object(ps, "REPS", 300), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ps.main(["--out", str(tmp)]), 0)
        res = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("DEGISTIRMEZ", res["nature"])
        self.assertEqual(set(res["kupiec"]), {"80", "95"})


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
