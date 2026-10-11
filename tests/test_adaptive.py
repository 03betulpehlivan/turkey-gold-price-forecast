"""
ACI testleri: src/models/adaptive.py
====================================
Calistirma (proje kokunden):
  python -m unittest tests.test_adaptive -v

En onemli test: veri yari yolda aniden iki kat oynaklasir (sentetik kriz).
Sabit aralik kapsami duser; ACI kapsami hedefe geri getirmelidir.
Gercek data/ klasoru kullanilmaz.
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

from src.models import adaptive as ad  # noqa: E402
from src.models import train as tr  # noqa: E402
from tests.test_implied_vol import write_world, quiet, file_hashes  # noqa: E402


class TestACIMechanics(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.z = rng.standard_normal(5000)

    def run_aci(self, y, gamma):
        n = len(y)
        return ad.aci_sequence(y, np.zeros(n), np.ones(n), [self.z] * n, 0.05, gamma)

    def test_update_uses_only_past_outcomes(self):
        y = np.array([10.0, 0.0, 0.0])                   # ilk gun kesin tasma
        lo, hi, k, miss = self.run_aci(y, 0.1)
        self.assertEqual(k[0], 1.0)
        self.assertTrue(miss[0])
        self.assertAlmostEqual(k[1], np.exp(0.1 * (1.0 - 0.05)))     # ilk gunun hatasi ikinci gunu etkiler
        self.assertAlmostEqual(k[2], k[1] * np.exp(0.1 * (0.0 - 0.05)))
        self.assertAlmostEqual(hi[1] / hi[0], k[1])

    def test_future_outcomes_do_not_change_past_intervals(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(400)
        lo_a, hi_a, _, _ = self.run_aci(y, 0.02)
        y2 = y.copy()
        y2[300:] *= 10                                     # gelecek degisir
        lo_b, hi_b, _, _ = self.run_aci(y2, 0.02)
        np.testing.assert_array_equal(lo_a[:301], lo_b[:301])
        np.testing.assert_array_equal(hi_a[:301], hi_b[:301])

    def test_calibrated_stationary_data_keeps_nominal_coverage(self):
        y = np.random.default_rng(2).standard_normal(4000)
        _, _, _, miss = self.run_aci(y, 0.005)
        self.assertAlmostEqual(1 - miss.mean(), 0.95, delta=0.012)

    def test_aci_recovers_coverage_after_volatility_shift(self):
        rng = np.random.default_rng(3)
        y = np.concatenate([rng.standard_normal(1500), 2.0 * rng.standard_normal(1500)])   # sentetik kriz
        fixed_lo, fixed_hi = np.quantile(self.z, [0.025, 0.975])
        fixed_cov_after = np.mean((y[1500:] >= fixed_lo) & (y[1500:] <= fixed_hi))
        _, _, k, miss = self.run_aci(y, 0.02)
        self.assertLess(fixed_cov_after, 0.70)               # sabit aralik cok altina duser
        # Kalici donem (gecis bittikten sonra): kapsam hedefe doner
        self.assertAlmostEqual(1 - miss[2000:].mean(), 0.95, delta=0.02)
        # Sok sonrasi toplam: teleskopik toplamdan ortalama(hata) - a = (log k_T - log k_s) / (eta T)
        expected = 0.95 - (np.log(k[-1]) - np.log(k[1500])) / (0.02 * 1500)
        self.assertAlmostEqual(1 - miss[1500:].mean(), expected, delta=0.002)
        self.assertGreater(k[-1], 1.6)                       # olcek gercekten ~2 kata yaklasti

    def test_scale_can_grow_beyond_training_extremes(self):
        y = np.full(200, 20.0)                              # egitimde hic gorulmeyen buyuklukte sapma
        lo, hi, k, miss = self.run_aci(y, 0.1)
        self.assertGreater(hi[-1], np.max(self.z) * 2)      # aralik egitimdeki en uc artigin otesine acilir
        self.assertFalse(miss[-1])                          # ve sonunda kapsar
        self.assertTrue(np.isfinite(lo).all() and np.isfinite(hi).all())


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="aci_int_"))
        cls.mpath, cls.ppath = write_world(cls.tmp, "bilgili", seed=5)
        cls.out_root = cls.tmp / "results"
        with mock.patch.object(tr, "OUTPUT_DIR", cls.out_root):
            cls.res = quiet(ad.run, cls.mpath, cls.ppath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_outputs_and_period(self):
        target = self.out_root / "experiments" / "adaptive"
        for f in ("metrics.csv", "annual.csv", "significance.csv", "decision.csv", "alpha_paths.csv",
                  "summary.json", "preregistration.json"):
            self.assertTrue((target / f).exists(), f)
        dates = pd.to_datetime(self.res["walk"]["predictions"]["prediction_date"])
        self.assertGreaterEqual(dates.min(), pd.Timestamp(ad.DEV_START))
        self.assertLess(dates.max(), pd.Timestamp(tr.TEST_START))

    def test_state_is_carried_across_windows(self):
        pred = self.res["walk"]["predictions"]
        paths = self.res["walk"]["alpha_paths"]
        first_of_w2 = pred.index[pred["window"] == 2][0]
        self.assertNotEqual(paths["aci_garch_g020_95"].iloc[first_of_w2], 1.0)

    def test_all_candidates_have_ordered_quantiles(self):
        pred = self.res["walk"]["predictions"]
        for m in ad.CANDIDATES:
            q = [pred[f"{m}_q{t}"] for t in (0.025, 0.10, 0.90, 0.975)]
            self.assertTrue((q[0] <= q[1]).mean() > 0.99 and (q[2] <= q[3]).mean() > 0.99, m)
            self.assertTrue((q[1] < q[2]).all(), m)


class TestProtection(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aci_p_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mpath, self.ppath = write_world(self.tmp, "gurultu", seed=6)

    def test_test_period_values_cannot_change_forecasts(self):
        from src.models import implied_vol as ivx
        df = quiet(ivx.load_inputs, self.mpath, self.ppath)
        a = ad.walk_forward(df)["predictions"]
        bad = df.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        cols = ["gold_return", "gold_usd_return", "usdtry_return", tr.TARGET, tr.PRICE_COL, "gvz", "gvz_lag1"]
        bad.loc[m, cols] = bad.loc[m, cols] * 4.0
        b = ad.walk_forward(bad)["predictions"]
        pd.testing.assert_frame_equal(a, b)

    def test_preregistration_before_computation_and_isolation(self):
        out_root = self.tmp / "results"
        before = file_hashes(self.tmp)
        with mock.patch.object(tr, "OUTPUT_DIR", out_root), \
                mock.patch.object(ad, "walk_forward", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(ad.run, self.mpath, self.ppath)
        target = out_root / "experiments" / "adaptive"
        prereg = json.loads((target / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(prereg["candidates"], ad.CANDIDATES)
        self.assertFalse((target / "decision.csv").exists())
        self.assertEqual(file_hashes(self.tmp), before)

    def test_decision_rule(self):
        metrics = pd.DataFrame({"method": ["garch", "a", "b", "c"], "pinball": [1.0, 1.01, 1.0, 1.0],
                                "coverage_80": [0.8, 0.8, 0.8, 0.8], "coverage_95": [0.95, 0.95, 0.95, 0.95],
                                "kupiec_80_p": [0.5] * 4, "kupiec_95_p": [0.5, 0.5, 0.01, 0.5]})
        annual = pd.DataFrame({"method": ["a", "a", "b", "c", "c"], "year": [1, 2, 1, 1, 2],
                               "coverage_95": [0.93, 0.95, 0.95, 0.91, 0.96]})
        sig = pd.DataFrame({"method": ["a", "b", "c"], "dm_stat": [1.0, 0.0, 0.0], "p_holm": [0.5, 1, 1]})
        with mock.patch.object(ad, "CANDIDATES", ["a", "b", "c"]):
            d = ad.decide(metrics, annual, sig).set_index("method")
        self.assertTrue(d.loc["a", "accepted"])        # hafif kotu ama anlamli degil: kabul
        self.assertFalse(d.loc["b", "accepted"])       # Kupiec reddi
        self.assertFalse(d.loc["c", "accepted"])       # bir yil 0.91 < 0.92


class TestConfirm(unittest.TestCase):
    """Teyit testi: on kayit, donem ve 2026 sinirlari, tek seferlik calistirma."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aci_c_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mpath, self.ppath = write_world(self.tmp, "bilgili", seed=7)
        pt = mock.patch.object(tr, "OUTPUT_DIR", self.tmp / "results")
        pt.start()
        self.addCleanup(pt.stop)
        self.cdir = self.tmp / "results" / "experiments" / "adaptive_confirm"

    def test_preregistration_written_before_computation(self):
        with mock.patch.object(ad, "walk_forward", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(ad.run_confirm, self.mpath, self.ppath)
        prereg = json.loads((self.cdir / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(prereg["method"], ad.CONFIRM_METHOD)
        self.assertEqual(prereg["hypothesis"], ad.CONFIRM_HYPOTHESIS)
        self.assertFalse((self.cdir / "result.json").exists())

    def test_test_period_only_and_no_2026_target(self):
        res = quiet(ad.run_confirm, self.mpath, self.ppath)
        dates = pd.to_datetime(res["predictions"]["prediction_date"])
        self.assertGreaterEqual(dates.min(), pd.Timestamp(tr.TEST_START))
        # Sentetik verinin son satiri 2025-12-31; hedefi 2026'nin ilk gunu -> degerlendirmeye girmemeli
        model = pd.read_csv(self.mpath, parse_dates=["Date"])
        self.assertEqual(model["Date"].max(), pd.Timestamp("2025-12-31"))
        self.assertNotIn(pd.Timestamp("2025-12-31"), set(dates))
        self.assertFalse([c for c in res["predictions"].columns if c.startswith(("aci_garch", "comp_gvz_lag_q"))])

    def test_frozen_method_matches_development_code(self):
        res = quiet(ad.run_confirm, self.mpath, self.ppath)
        from src.models import implied_vol as ivx
        df = quiet(ivx.load_inputs, self.mpath, self.ppath)
        ref = ad.walk_forward(df, start=tr.TEST_START, end=ad.HOLDOUT_START)["predictions"]
        for tau in (0.025, 0.10, 0.90, 0.975):
            np.testing.assert_array_equal(res["predictions"][f"aci_gvzlag_g020_q{tau}"], ref[f"aci_gvzlag_g020_q{tau}"])

    def test_second_run_refused_and_force_rerun_logged(self):
        quiet(ad.run_confirm, self.mpath, self.ppath)
        first = (self.cdir / "result.json").read_bytes()
        with self.assertRaises(ad.ConfirmAlreadyRun):
            quiet(ad.run_confirm, self.mpath, self.ppath)
        self.assertEqual((self.cdir / "result.json").read_bytes(), first)
        quiet(ad.run_confirm, self.mpath, self.ppath, force=True)
        self.assertEqual((self.cdir / "result_run1.json").read_bytes(), first)
        log = (self.cdir / "run_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(log), 2)

    def test_main_flags(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ad.main(["--force-rerun"])
        with mock.patch.object(ad, "run_confirm", side_effect=ad.ConfirmAlreadyRun("var")):
            self.assertEqual(quiet(ad.main, ["--confirm"]), 2)

    def test_verdict_rules(self):
        good = {"coverage_80": 0.80, "coverage_95": 0.95, "kupiec_80_p": 0.5, "kupiec_95_p": 0.5}
        annual = pd.DataFrame({"year": [2023, 2024, 2025], "n": [46, 252, 250], "coverage_95": [0.85, 0.93, 0.95]})
        ok = ad.confirm_verdict(good, good, annual, {"dm_stat": 1.0, "p_value": 0.3})
        self.assertTrue(ok["hypothesis_supported"])          # 46 gunluk 2023 karara girmez
        self.assertEqual(ok["years_checked"], [2024, 2025])
        worse = ad.confirm_verdict(good, good, annual, {"dm_stat": 2.5, "p_value": 0.01})
        self.assertFalse(worse["hypothesis_supported"])
        bad_year = annual.assign(coverage_95=[0.95, 0.91, 0.95])
        self.assertFalse(ad.confirm_verdict(good, good, bad_year, {"dm_stat": 0, "p_value": 1})["hypothesis_supported"])


_REAL_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed",
              PROJECT_ROOT / "data" / "processed" / "results"]
_REAL_BEFORE = {}


def setUpModule():
    for d in _REAL_DIRS:
        _REAL_BEFORE[d] = file_hashes(d)


def tearDownModule():
    for d in _REAL_DIRS:
        if file_hashes(d) != _REAL_BEFORE[d]:
            raise AssertionError(f"Testler gercek veri klasorunu degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
