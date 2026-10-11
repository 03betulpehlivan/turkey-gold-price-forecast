"""
Aralik/oynaklik testleri: src/models/intervals.py
=================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_intervals -v

Sentetik veri GARCH(1,1) sureciyle uretilir (gercek oynaklik kumelenmesi
icerir). Gercek data/ klasoru kullanilmaz.
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

from src.data import build_dataset as bd  # noqa: E402
from src.models import intervals as iv  # noqa: E402
from src.models import train as tr  # noqa: E402


def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def garch_returns(n, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    rng = np.random.default_rng(seed)
    r, v = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(v) * rng.standard_normal()
        v = omega + alpha * r[t] ** 2 + beta * v
    return r


def make_dataset(seed=0) -> pd.DataFrame:
    days = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(days)
    gold = 1200 * np.exp(np.cumsum(garch_returns(n, seed=seed) + 0.0002))
    fx = 2.3 * np.exp(np.cumsum(np.random.default_rng(seed + 1).normal(0.0008, 0.003, n)))
    g = pd.DataFrame({"Date": days, "close": gold})
    f = pd.DataFrame({"Date": days, "close": fx})
    _, model = quiet(bd.build, g, f)
    return model


_CACHE = {}


def loaded(tmp: Path) -> pd.DataFrame:
    if "df" not in _CACHE:
        path = tmp / "model_dataset.csv"
        make_dataset().to_csv(path, index=False, date_format="%Y-%m-%d")
        _CACHE["df"] = quiet(tr.load_data, path)
    return _CACHE["df"].copy()


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class TestFormulas(unittest.TestCase):

    def test_pinball(self):
        y, q = np.array([1.0, -1.0]), np.array([0.0, 0.0])
        np.testing.assert_allclose(iv.pinball(y, q, 0.9), [0.9, 0.1])
        np.testing.assert_allclose(iv.pinball(y, q, 0.1), [0.1, 0.9])

    def test_kupiec(self):
        hits = np.zeros(1000, bool)
        hits[:200] = True
        self.assertAlmostEqual(iv.kupiec_p(hits, 0.20), 1.0)
        hits[:400] = True
        self.assertLess(iv.kupiec_p(hits, 0.20), 1e-6)

    def test_summarize_uses_exceedance_rates(self):
        # %80 araliktan tam %20, %95 araliktan tam %5 tasma: iki Kupiec p de 1 olmali
        n = 1000
        y = np.zeros(n)
        y[:200] = 5.0          # %80 araligin disi, %95 araligin ici
        y[:50] = 50.0          # %95 araligin da disi
        pred = pd.DataFrame({"y": y, "price_t": np.full(n, 100.0)})
        for tau, q in zip(iv.QUANTILES, (-10.0, -1.0, 1.0, 10.0)):
            pred[f"static_q{tau}"] = q
        s = iv.summarize(pred, "static")
        self.assertAlmostEqual(s["coverage_80"], 0.80)
        self.assertAlmostEqual(s["coverage_95"], 0.95)
        self.assertGreater(s["kupiec_80_p"], 0.999)   # 1 - 80/100 kayan noktada tam 0.2 degil
        self.assertGreater(s["kupiec_95_p"], 0.999)

    def test_christoffersen_detects_clustering(self):
        rng = np.random.default_rng(1)
        clustered = np.zeros(1000, bool)
        for s in range(0, 1000, 100):
            clustered[s:s + 10] = True
        self.assertLess(iv.christoffersen_p(clustered), 1e-6)
        self.assertGreater(iv.christoffersen_p(rng.random(1000) < 0.1), 0.01)

    def test_ljung_box_detects_dependence(self):
        x = garch_returns(3000, seed=2)
        self.assertLess(iv.ljung_box(np.abs(x))[2], 1e-6)
        self.assertGreater(iv.ljung_box(np.random.default_rng(3).normal(size=3000))[2], 0.001)

    def test_ewma_recursion(self):
        r = np.array([0.01, -0.02, 0.03] + [0.0] * 30)
        s = iv.ewma_sigma(r, lam=0.94, init_n=3)
        v0 = np.mean(r[:3] ** 2)
        v = [0.94 * v0 + 0.06 * 0.01 ** 2]
        v.append(0.94 * v[0] + 0.06 * 0.02 ** 2)
        np.testing.assert_allclose(s[:2] ** 2, v)

    def test_garch_recursion_matches_arch(self):
        from arch import arch_model
        r = garch_returns(3000, seed=4)
        params = iv.fit_garch(r)
        mine = iv.garch_sigma(r, params) * 100                   # sigma for r[t+1]
        p = pd.Series({"mu": params["mu"], "omega": params["omega"],
                       "alpha[1]": params["alpha"], "beta[1]": params["beta"]})
        ref = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fix(p).conditional_volatility
        np.testing.assert_allclose(mine[1000:-1], np.asarray(ref)[1001:], rtol=1e-6)

    def test_sigma_estimators_do_not_look_ahead(self):
        r = garch_returns(1500, seed=5)
        params = iv.fit_garch(r[:1000])
        full = {"ewma": iv.ewma_sigma(r), "roll": iv.rolling_sigma(r, 60), "garch": iv.garch_sigma(r, params)}
        for t in (100, 700, 1400):
            cut = {"ewma": iv.ewma_sigma(r[:t + 1]), "roll": iv.rolling_sigma(r[:t + 1], 60),
                   "garch": iv.garch_sigma(r[:t + 1], params)}
            for k in full:
                with self.subTest(est=k, t=t):
                    self.assertAlmostEqual(full[k][t], cut[k][t], places=15)


class TestWalkForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="iv_test_"))
        cls.df = loaded(cls.tmp)
        cls.res = iv.walk_forward_intervals(cls.df)
        cls.pred = cls.res["predictions"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_evaluation_stays_before_test_period(self):
        self.assertGreaterEqual(pd.to_datetime(self.pred["prediction_date"]).min(), pd.Timestamp(iv.DEV_START))
        self.assertLess(pd.to_datetime(self.pred["target_date"]).max(), pd.Timestamp(tr.TEST_START))
        for w in self.res["windows"]:
            self.assertLess(w["train_last_date"], w["eval_first_date"])

    def test_test_period_values_cannot_change_forecasts(self):
        bad = self.df.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        bad.loc[m, ["gold_return", tr.TARGET, tr.PRICE_COL]] *= 9.0
        other = iv.walk_forward_intervals(bad)["predictions"]
        pd.testing.assert_frame_equal(self.pred, other)

    def test_quantiles_ordered_and_centered_on_mean_return(self):
        for m in iv.METHODS:
            q = [self.pred[f"{m}_q{t}"] for t in iv.QUANTILES]
            for a, b in zip(q, q[1:]):
                self.assertTrue((a < b).all(), m)
        first = self.res["windows"][0]
        self.assertAlmostEqual(self.pred["mu"].iloc[0], first["mu"])
        w1 = tr.eval_positions(self.df, iv.DEV_START, tr.TEST_START)[0]
        self.assertAlmostEqual(first["mu"], float(self.df[tr.TARGET].iloc[:w1 - 1].mean()), places=15)

    def test_coverage_is_reasonable_and_garch_beats_static(self):
        mt = iv.metrics_table(self.pred)
        dev = mt[mt["period"].str.startswith("dev_")].set_index("method")
        for m in ("static", "garch", "ewma"):
            self.assertAlmostEqual(dev.loc[m, "coverage_80"], 0.80, delta=0.04)
            self.assertAlmostEqual(dev.loc[m, "coverage_95"], 0.95, delta=0.03)
        self.assertLess(dev.loc["garch", "pinball"], dev.loc["static", "pinball"])

    def test_calendar_factor_estimated_from_training_only(self):
        w = self.res["windows"][0]
        self.assertTrue(np.isfinite(w["garch_cal_factor"]) and w["garch_cal_factor"] > 0)
        fri = self.pred["friday"].to_numpy()
        ratio = self.pred["garch_cal_sigma"] / self.pred["garch_sigma"]
        first = self.pred["window"] == 1
        np.testing.assert_allclose(ratio[first & fri], w["garch_cal_factor"])
        np.testing.assert_allclose(ratio[first & ~fri], 1.0)


class TestDecisionAndRun(unittest.TestCase):

    def test_decision_rule(self):
        metrics = pd.DataFrame({
            "period": ["dev_x"] * 3, "method": ["static", "good", "badcov"],
            "pinball": [1.0, 0.9, 0.8], "coverage_80": [0.80, 0.81, 0.70], "coverage_95": [0.95, 0.95, 0.90]})
        windows = pd.DataFrame({"window": [1, 1, 1, 2, 2, 2], "method": ["static", "good", "badcov"] * 2,
                                "pinball": [1, 0.9, 0.8, 1, 0.9, 0.8]})
        sig = pd.DataFrame({"method": ["good", "badcov"], "dm_stat": [-3.0, -3.0], "p_holm": [0.01, 0.01]})
        with mock.patch.object(iv, "METHODS", ["static", "good", "badcov"]):
            d = iv.decide(metrics, windows, sig).set_index("method")
        self.assertTrue(d.loc["good", "accepted"])
        self.assertFalse(d.loc["badcov", "accepted"])
        self.assertFalse(d.loc["badcov", "coverage_ok"])

    def test_run_writes_only_to_experiment_folder(self):
        tmp = Path(tempfile.mkdtemp(prefix="iv_run_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        data_dir = tmp / "processed"
        data_dir.mkdir()
        make_dataset(seed=7).to_csv(data_dir / "model_dataset.csv", index=False, date_format="%Y-%m-%d")
        out_root = data_dir / "results"
        before = file_hashes(data_dir)
        with mock.patch.object(tr, "DATA_DIR", data_dir), mock.patch.object(tr, "OUTPUT_DIR", out_root):
            res = quiet(iv.run, data_dir / "model_dataset.csv")
        self.assertEqual(file_hashes(data_dir), before)
        target = out_root / "experiments" / "intervals"
        for f in ("metrics.csv", "decision.csv", "significance.csv", "predictions.csv", "summary.json",
                  "volatility_clustering.csv", "windows.csv", "annual.csv", "window_metrics.csv"):
            self.assertTrue((target / f).exists(), f)
        self.assertEqual(sorted(p.name for p in out_root.iterdir()), ["experiments"])
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["test_period_used"])
        self.assertEqual(set(res["significance"]["reference"]), {"static"})


class TestConfirm(unittest.TestCase):
    """Adim 2c teyit testi: on kayit, donem korumasi, tek seferlik calistirma."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="iv_confirm_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.data_dir = self.tmp / "processed"
        self.data_dir.mkdir()
        self.data = self.data_dir / "model_dataset.csv"
        make_dataset(seed=9).to_csv(self.data, index=False, date_format="%Y-%m-%d")
        self.out_root = self.data_dir / "results"
        self.patches = [mock.patch.object(tr, "DATA_DIR", self.data_dir),
                        mock.patch.object(tr, "OUTPUT_DIR", self.out_root)]
        for pt in self.patches:
            pt.start()
            self.addCleanup(pt.stop)
        self.cdir = self.out_root / "experiments" / "intervals_confirm"

    def test_preregistration_written_before_computation(self):
        with mock.patch.object(iv, "walk_forward_intervals", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(iv.run_confirm, self.data)
        prereg = json.loads((self.cdir / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(prereg["hypothesis"], iv.CONFIRM_HYPOTHESIS)
        self.assertEqual(set(prereg["code_md5"]), {"intervals.py", "train.py"})
        self.assertFalse((self.cdir / "result.json").exists())

    def test_confirm_uses_test_period_only_and_frozen_method(self):
        res = quiet(iv.run_confirm, self.data)
        pred = res["predictions"]
        self.assertGreaterEqual(pd.to_datetime(pred["prediction_date"]).min(), pd.Timestamp(tr.TEST_START))
        self.assertLess(pd.to_datetime(pred["target_date"]).max(), pd.Timestamp(iv.HOLDOUT_START))
        self.assertFalse([c for c in pred.columns if c.startswith(("ewma", "rolling", "garch_cal"))])
        df = quiet(tr.load_data, self.data)
        ref = iv.walk_forward_intervals(df, start=tr.TEST_START, end=None)["predictions"]
        for tau in iv.QUANTILES:
            np.testing.assert_array_equal(pred[f"garch_q{tau}"], ref[f"garch_q{tau}"])
        r = res["result"]
        self.assertEqual(r["hypothesis_supported"], bool(r["dm"]["dm_stat"] < 0 and r["dm"]["p_value"] < 0.05))
        self.assertEqual(len(res["metrics"]), 2)
        self.assertTrue((self.cdir / "result.json").exists())

    def test_second_run_refused_and_force_rerun_is_logged(self):
        quiet(iv.run_confirm, self.data)
        first = (self.cdir / "result.json").read_bytes()
        with self.assertRaises(iv.ConfirmAlreadyRun):
            quiet(iv.run_confirm, self.data)
        self.assertEqual(quiet(iv.main, ["--confirm"]), 2)
        self.assertEqual((self.cdir / "result.json").read_bytes(), first)
        quiet(iv.run_confirm, self.data, force=True)
        self.assertEqual((self.cdir / "result_run1.json").read_bytes(), first)
        log = (self.cdir / "run_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(log), 2)
        self.assertEqual(json.loads((self.cdir / "preregistration.json").read_text(encoding="utf-8"))["run_number"], 2)

    def test_holdout_rows_stop_the_test(self):
        days = pd.bdate_range("2015-01-01", "2026-03-31")
        n = len(days)
        g = pd.DataFrame({"Date": days, "close": 1200 * np.exp(np.cumsum(garch_returns(n, seed=1)))})
        f = pd.DataFrame({"Date": days, "close": 2.3 * np.exp(np.cumsum(np.full(n, 0.0008)))})
        _, model = quiet(bd.build, g, f)
        model.to_csv(self.data, index=False, date_format="%Y-%m-%d")
        with self.assertRaisesRegex(ValueError, "2026"):
            quiet(iv.run_confirm, self.data)
        self.assertFalse((self.cdir / "result.json").exists())

    def test_force_rerun_requires_confirm(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            iv.main(["--force-rerun"])


class TestBootstrap(unittest.TestCase):

    def test_equal_losses_give_zero_interval(self):
        loss = np.random.default_rng(0).random(500)
        self.assertEqual(iv.block_bootstrap_ci(loss, loss, reps=200), (0.0, 0.0))

    def test_uniform_improvement_and_determinism(self):
        lr = np.random.default_rng(1).random(500) + 0.1
        lo, hi = iv.block_bootstrap_ci(0.8 * lr, lr, reps=200)
        self.assertAlmostEqual(lo, 20.0, places=9)
        self.assertAlmostEqual(hi, 20.0, places=9)
        noisy = lr * np.random.default_rng(2).uniform(0.6, 1.1, 500)
        self.assertEqual(iv.block_bootstrap_ci(noisy, lr, reps=300), iv.block_bootstrap_ci(noisy, lr, reps=300))
        lo, hi = iv.block_bootstrap_ci(noisy, lr, reps=300)
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0)


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
