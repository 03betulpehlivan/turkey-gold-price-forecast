"""
Egitim ve degerlendirme testleri: src/models/train.py (Faz A)
=============================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_train -v
  python -m unittest discover -s tests -v

Kurallar:
  - Yalnizca unittest (+ unittest.mock) ve projede kurulu paketler.
  - Testler hizli olsun diye agir modeller (RF, XGBoost) yerine yalnizca
    baseline'lar ve Linear Regression kullanilir (get_models yamalanir).
    Degerlendirme mantigi model turunden bagimsizdir.
  - Gercek data/ klasoru okunmaz ve yazilmaz; veri gecici klasorde
    build_dataset.build ile uretilir. Modul sonunda gercek data/raw,
    data/processed ve data/processed/results dosyalarinin degismedigi
    md5 ile dogrulanir.
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
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataset as bd  # noqa: E402
from src.models import train as tr  # noqa: E402


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def light_models():
    return {
        "Zero-Return": "zero",
        "Mean-Return": "mean",
        "Last-Return": "last",
        "Linear Regression": LinearRegression(),
    }


def make_model_dataset(start="2015-01-01", end="2025-12-31", seed=7) -> pd.DataFrame:
    """build_dataset.build ile sentetik ama gercek semali model_dataset uretir."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    gold = 1186.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n)))
    fx = 2.33 * np.exp(np.cumsum(rng.normal(0.0010, 0.005, n)))
    g = pd.DataFrame({"Date": days, "close": gold})
    f = pd.DataFrame({"Date": days, "close": fx}).drop(index=[3, 40, 41])  # tatil benzeri
    (_, model), _ = quiet(bd.build, g, f.reset_index(drop=True))
    return model


_DATASET = None


def dataset() -> pd.DataFrame:
    global _DATASET
    if _DATASET is None:
        _DATASET = make_model_dataset()
    return _DATASET.copy()


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="train_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, df: pd.DataFrame, name="model_dataset.csv") -> Path:
        path = self.tmp / name
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def load(self, df: pd.DataFrame) -> pd.DataFrame:
        return quiet(tr.load_data, self.write(df))[0]


# ---------------------------------------------------------------------------
# 1) Veri sozlesmesi: load_data
# ---------------------------------------------------------------------------
class TestLoadDataContract(TempDir):

    def test_clean_data_drops_exactly_warmup_rows(self):
        raw = dataset()
        df = self.load(raw)
        self.assertEqual(tr.WARMUP_ROWS, 19)
        self.assertEqual(len(df), len(raw) - tr.WARMUP_ROWS)
        self.assertEqual(df["Date"].iloc[0], pd.Timestamp(raw["Date"].iloc[tr.WARMUP_ROWS]))
        self.assertEqual(list(df.index), list(range(len(df))))
        self.assertFalse(df[tr.FEATURES + [tr.TARGET]].isna().any().any())

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            quiet(tr.load_data, self.tmp / "yok.csv")

    def test_unsorted_dates_raise_instead_of_silent_sort(self):
        raw = dataset()
        raw.iloc[[100, 101]] = raw.iloc[[101, 100]].to_numpy()
        with self.assertRaisesRegex(ValueError, "kronolojik sirali degil"):
            self.load(raw)

    def test_duplicate_dates_raise(self):
        raw = dataset()
        raw.loc[101, "Date"] = raw.loc[100, "Date"]
        with self.assertRaisesRegex(ValueError, "mukerrer"):
            self.load(raw)

    def test_missing_column_raises(self):
        with self.assertRaisesRegex(ValueError, "Eksik sutunlar"):
            self.load(dataset().drop(columns="usdtry_return"))

    def test_nan_in_required_column_raises(self):
        raw = dataset()
        raw.loc[500, "gold_usd_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.load(raw)

    def test_non_positive_price_raises(self):
        raw = dataset()
        raw.loc[500, "usdtry"] = -1.0
        with self.assertRaisesRegex(ValueError, "gecersiz deger"):
            self.load(raw)

    def test_inconsistent_return_columns_raise(self):
        for col in ("gold_return", "gold_usd_return", "usdtry_return"):
            with self.subTest(col=col):
                raw = dataset()
                raw.loc[500, col] += 1e-6
                with self.assertRaisesRegex(ValueError, col):
                    self.load(raw)

    def test_identity_violation_raises(self):
        # Ilk satirin getirileri fiyattan dogrulanamaz; ozdeslik kontrolu yakalar
        raw = dataset()
        raw.loc[0, "gold_usd_return"] += 1e-4
        with self.assertRaisesRegex(ValueError, "gold_usd_return \\+ usdtry_return"):
            self.load(raw)

    def test_return_consistency_tolerates_float_noise(self):
        raw = dataset()
        raw["gold_return"] = raw["gold_return"] + 1e-13
        self.assertEqual(len(self.load(raw)), len(raw) - tr.WARMUP_ROWS)

    def test_nan_after_warmup_raises(self):
        original = tr.build_features

        def with_gap(df):
            out = original(df)
            out.loc[200, "return_mean_5"] = np.nan
            return out

        with mock.patch.object(tr, "build_features", side_effect=with_gap):
            with self.assertRaisesRegex(ValueError, "Isinma donemi"):
                self.load(dataset())


# ---------------------------------------------------------------------------
# 2) Hedef hizalama
# ---------------------------------------------------------------------------
class TestTargetAlignment(TempDir):

    def setUp(self):
        super().setUp()
        self.df = self.load(dataset())

    def test_leakage_checks_pass_on_clean_data(self):
        _, out = quiet(tr.run_leakage_checks, self.df)
        self.assertEqual(out.count("[OK]"), 4)

    def test_shifted_target_detected(self):
        bad = self.df.copy()
        bad[tr.TARGET] = bad["gold_return"]
        with self.assertRaises(ValueError):
            quiet(tr.run_leakage_checks, bad)

    def test_prediction_rows_aligned_with_next_observation(self):
        with mock.patch.object(tr, "get_models", light_models):
            res, _ = quiet(tr.walk_forward_evaluate, self.df)
        pred = res["predictions"]
        by_date = self.df.set_index("Date")
        p_t = by_date.loc[pred["prediction_date"], tr.PRICE_COL].to_numpy()
        p_next = by_date.loc[pd.to_datetime(pred["target_date"].iloc[:-1]), tr.PRICE_COL].to_numpy()
        np.testing.assert_allclose(pred["price_t"], p_t)
        np.testing.assert_allclose(pred["actual_return"].iloc[:-1], np.log(p_next / p_t[:-1]), atol=1e-12)
        nxt = self.df["Date"].shift(-1)
        expected_target_dates = nxt[self.df["Date"].isin(pred["prediction_date"])].tolist()
        self.assertEqual(pd.to_datetime(pred["target_date"]).tolist()[:-1], expected_target_dates[:-1])


# ---------------------------------------------------------------------------
# 3) Kronolojik donem sinirlari
# ---------------------------------------------------------------------------
class TestPeriodBoundaries(TempDir):

    def setUp(self):
        super().setUp()
        self.df = self.load(dataset())

    def test_protocol_dates_are_ordered(self):
        self.assertLess(pd.Timestamp(tr.TRAIN_END), pd.Timestamp(tr.VALIDATION_START))
        self.assertLess(pd.Timestamp(tr.VALIDATION_START), pd.Timestamp(tr.TEST_START))

    def test_test_positions_match_previous_definition(self):
        pos = tr.eval_positions(self.df, tr.TEST_START)
        legacy = self.df.index[self.df["Date"] >= pd.Timestamp(tr.TEST_START)].tolist()
        self.assertEqual(pos, legacy)

    def test_validation_positions_stay_before_test(self):
        pos = tr.eval_positions(self.df, tr.VALIDATION_START, tr.TEST_START)
        dates = self.df["Date"].iloc[pos]
        next_dates = self.df["Date"].shift(-1).iloc[pos]
        self.assertGreaterEqual(dates.min(), pd.Timestamp(tr.VALIDATION_START))
        self.assertLess(next_dates.max(), pd.Timestamp(tr.TEST_START))
        # Hedefi test donemine tasan son satir dogrulamada degil
        last_pre_test = self.df.index[self.df["Date"] < pd.Timestamp(tr.TEST_START)][-1]
        self.assertNotIn(last_pre_test, pos)
        self.assertEqual(pos[-1], last_pre_test - 1)

    def test_validation_and_test_do_not_overlap(self):
        val = set(tr.eval_positions(self.df, tr.VALIDATION_START, tr.TEST_START))
        test = set(tr.eval_positions(self.df, tr.TEST_START))
        self.assertFalse(val & test)

    def test_too_short_validation_period_raises(self):
        short = make_model_dataset(end="2022-01-20")
        df = self.load(short)
        with mock.patch.object(tr, "get_models", light_models):
            with self.assertRaisesRegex(ValueError, "yetersiz"):
                quiet(tr.walk_forward_evaluate, df, tr.VALIDATION_START, tr.TEST_START, "dogrulama")


# ---------------------------------------------------------------------------
# 4) Walk-forward sirasi ve egitim sinirlari
# ---------------------------------------------------------------------------
class TestWalkForward(TempDir):

    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="train_wf_"))
        try:
            path = tmp / "model_dataset.csv"
            dataset().to_csv(path, index=False, date_format="%Y-%m-%d")
            cls.df = quiet(tr.load_data, path)[0]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with mock.patch.object(tr, "get_models", light_models):
            cls.test_res = quiet(tr.walk_forward_evaluate, cls.df)[0]
            cls.val_res = quiet(tr.walk_forward_evaluate, cls.df, tr.VALIDATION_START,
                                tr.TEST_START, "dogrulama")[0]

    def test_windows_partition_positions_in_order(self):
        for res, args in ((self.test_res, (tr.TEST_START,)),
                          (self.val_res, (tr.VALIDATION_START, tr.TEST_START))):
            pos = tr.eval_positions(self.df, *args)
            windows = tr.make_windows(pos, tr.RETRAIN_EVERY)
            self.assertEqual([p for w in windows for p in w], pos)
            self.assertTrue(all(len(w) <= tr.RETRAIN_EVERY for w in windows))
            self.assertEqual(len(res["windows"]), len(windows))

    def test_training_always_ends_before_evaluation(self):
        for res in (self.test_res, self.val_res):
            for w in res["windows"]:
                self.assertLess(w["train_last_date"], w["eval_first_date"])
                self.assertEqual(w["n_purged"], 1)

    def test_validation_training_never_sees_test_period(self):
        for w in self.val_res["windows"]:
            self.assertLess(pd.Timestamp(w["train_last_date"]), pd.Timestamp(tr.TEST_START))
            self.assertLess(pd.Timestamp(w["eval_last_date"]), pd.Timestamp(tr.TEST_START))

    def test_first_validation_window_trains_on_training_period_only(self):
        first = self.val_res["windows"][0]
        self.assertLessEqual(pd.Timestamp(first["train_last_date"]), pd.Timestamp(tr.TRAIN_END))

    def test_every_model_has_one_prediction_per_row(self):
        for res in (self.test_res, self.val_res):
            pred = res["predictions"]
            self.assertTrue(pd.to_datetime(pred["prediction_date"]).is_monotonic_increasing)
            self.assertFalse(pd.to_datetime(pred["prediction_date"]).duplicated().any())
            for name in light_models():
                self.assertEqual(pred[f"pred_{name}"].notna().sum(), len(pred))
            self.assertTrue((res["metrics"]["n_obs"] == len(pred)).all())

    def test_mean_baseline_uses_purged_training_mean(self):
        w1 = tr.eval_positions(self.df, tr.TEST_START)[0]
        expected = float(self.df[tr.TARGET].iloc[:w1 - 1].mean())  # son satir purge
        got = self.test_res["predictions"]["pred_Mean-Return"].iloc[0]
        self.assertAlmostEqual(got, expected, places=15)

    def test_metric_columns_keep_legacy_order(self):
        legacy = ["MAE_return", "RMSE_return", "direction_accuracy_pct", "MAE_TL",
                  "RMSE_TL", "skill_vs_zero", "model"]
        self.assertEqual(list(self.test_res["metrics"].columns[:len(legacy)]), legacy)
        self.assertEqual(list(self.test_res["metrics"].columns[len(legacy):]),
                         ["skill_vs_mean", "n_obs", "n_directional", "up_fraction_pct"])


# ---------------------------------------------------------------------------
# 5) Ozelliklerin kesme degismezligi
# ---------------------------------------------------------------------------
class TestFeatureTruncationInvariance(unittest.TestCase):

    def test_features_at_t_do_not_depend_on_future_rows(self):
        full = tr.build_features(dataset())
        for t in (25, 400, 1500, len(full) - 2):
            with self.subTest(t=t):
                cut = tr.build_features(dataset().iloc[:t + 1])
                pd.testing.assert_series_equal(
                    full.loc[t, tr.FEATURES], cut.loc[t, tr.FEATURES], check_names=False)


# ---------------------------------------------------------------------------
# 6) Diebold-Mariano testi
# ---------------------------------------------------------------------------
class TestDieboldMariano(unittest.TestCase):

    def test_equal_losses(self):
        e = np.random.default_rng(1).normal(0, 0.01, 300)
        r = tr.diebold_mariano(e, e)
        self.assertEqual((r["dm_stat"], r["p_value"], r["n"]), (0.0, 1.0, 300))

    def test_equal_absolute_losses_with_opposite_signs(self):
        e = np.random.default_rng(2).normal(0, 0.01, 300)
        r = tr.diebold_mariano(e, -e)
        self.assertEqual((r["dm_stat"], r["p_value"]), (0.0, 1.0))

    def test_consistently_lower_loss_is_significant(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 0.01, 500)
        better = ref * 0.8 + rng.normal(0, 0.0005, 500)
        r = tr.diebold_mariano(better, ref)
        self.assertLess(r["dm_stat"], 0)
        self.assertLess(r["p_value"], 0.01)
        r2 = tr.diebold_mariano(ref, better)
        self.assertAlmostEqual(r2["dm_stat"], -r["dm_stat"], places=12)
        self.assertAlmostEqual(r2["p_value"], r["p_value"], places=12)

    def test_constant_difference_is_degenerate(self):
        r = tr.diebold_mariano(np.full(50, 0.01), np.full(50, 0.02))
        self.assertEqual(r["dm_stat"], -math.inf)
        self.assertEqual(r["p_value"], 0.0)

    def test_known_value_with_zero_lag(self):
        # d = |e1| - |e2| = [1, 2, 3, 4]: ortalama 2.5, ortalamanin varyansi 1.25/4
        r = tr.diebold_mariano([1, 2, 3, 4], [0, 0, 0, 0], lag=0)
        stat = 2.5 / math.sqrt(1.25 / 4)
        self.assertAlmostEqual(r["dm_stat"], stat, places=12)
        self.assertAlmostEqual(r["p_value"], math.erfc(stat / math.sqrt(2)), places=12)
        self.assertEqual(r["hac_lag"], 0)

    def test_default_lag_rule(self):
        e = np.random.default_rng(4).normal(0, 1, 555)
        self.assertEqual(tr.diebold_mariano(e, e * 1.1)["hac_lag"], 5)
        e = np.random.default_rng(4).normal(0, 1, 100)
        self.assertEqual(tr.diebold_mariano(e, e * 1.1)["hac_lag"], 4)

    def test_hac_widens_variance_for_autocorrelated_differences(self):
        rng = np.random.default_rng(5)
        d = np.zeros(400)
        for i in range(1, 400):
            d[i] = 0.8 * d[i - 1] + rng.normal()
        d += 0.3
        e_ref = np.zeros(400) + 10.0
        e_mod = e_ref + d  # |e_mod| - |e_ref| = d (hepsi pozitif)
        r0 = tr.diebold_mariano(e_mod, e_ref, lag=0)
        r8 = tr.diebold_mariano(e_mod, e_ref, lag=8)
        self.assertLess(abs(r8["dm_stat"]), abs(r0["dm_stat"]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            tr.diebold_mariano([1, 2, 3], [1, 2])
        with self.assertRaises(ValueError):
            tr.diebold_mariano([1, np.nan], [1, 2])
        with self.assertRaises(ValueError):
            tr.diebold_mariano([1], [2])


# ---------------------------------------------------------------------------
# 7) Metrik tanimlari
# ---------------------------------------------------------------------------
class TestMetricDefinitions(unittest.TestCase):

    def setUp(self):
        self.actual = np.array([0.01, -0.02, 0.0, 0.03, -0.01, 0.02])
        self.pred_df = pd.DataFrame({
            "prediction_date": pd.date_range("2024-01-01", periods=6),
            "actual_return": self.actual,
            "pred_Zero-Return": np.zeros(6),
            "pred_Mean-Return": np.full(6, 0.001),
            "pred_Model": np.array([0.005, 0.004, -0.01, 0.0, -0.002, 0.01]),
        })

    def test_direction_accuracy_excludes_zero_predictions_and_zero_actuals(self):
        pred = self.pred_df["pred_Model"].to_numpy()
        # Gecerli gunler: 0, 1, 4, 5 (2: gercek 0, 3: tahmin 0); dogru: 0, 4, 5
        self.assertEqual(tr.directional_count(self.actual, pred), 4)
        self.assertAlmostEqual(tr.direction_accuracy(self.actual, pred), 75.0)
        self.assertTrue(np.isnan(tr.direction_accuracy(self.actual, np.zeros(6))))

    def test_up_fraction_equals_always_up_direction_accuracy(self):
        self.assertAlmostEqual(tr.up_fraction(self.actual), 60.0)
        self.assertAlmostEqual(tr.direction_accuracy(self.actual, np.full(6, 1e-9)), 60.0)

    def test_skill_vs_mean(self):
        rows = [dict(tr.compute_metrics(self.actual, self.pred_df[f"pred_{n}"].to_numpy(),
                                        np.full(6, 100.0)), model=n)
                for n in ("Zero-Return", "Mean-Return", "Model")]
        out = tr.add_reference_metrics(pd.DataFrame(rows), self.pred_df)
        mae = lambda c: float(np.mean(np.abs(self.actual - self.pred_df[c])))
        expected = round(1 - mae("pred_Model") / mae("pred_Mean-Return"), 4)
        self.assertEqual(out.set_index("model").loc["Mean-Return", "skill_vs_mean"], 0.0)
        self.assertAlmostEqual(out.set_index("model").loc["Model", "skill_vs_mean"], expected)
        self.assertEqual(out["up_fraction_pct"].iloc[0], 60.0)

    def test_annual_table_marks_insufficient_years(self):
        out = tr.annual_comparison(self.pred_df, "test")
        self.assertEqual(set(out["model"]), {"Zero-Return", "Mean-Return", "Model"})
        self.assertTrue((out["status"] == "yetersiz_gozlem").all())
        self.assertTrue(out["MAE_return"].isna().all())

    def test_significance_table_compares_against_both_references(self):
        sig = tr.significance_table(self.pred_df, "test")
        pairs = set(zip(sig["model"], sig["reference"]))
        self.assertIn(("Model", "Mean-Return"), pairs)
        self.assertIn(("Model", "Zero-Return"), pairs)
        self.assertNotIn(("Mean-Return", "Mean-Return"), pairs)


# ---------------------------------------------------------------------------
# 8) Uctan uca: main() ciktilari (gecici klasorde)
# ---------------------------------------------------------------------------
class TestMainArtifacts(TempDir):

    def test_main_produces_phase_a_artifacts(self):
        data_dir = self.tmp / "processed"
        data_dir.mkdir()
        dataset().to_csv(data_dir / "model_dataset.csv", index=False, date_format="%Y-%m-%d")
        before = file_hashes(data_dir)
        out_dir = data_dir / "results"
        with mock.patch.object(tr, "DATA_DIR", data_dir), \
                mock.patch.object(tr, "OUTPUT_DIR", out_dir), \
                mock.patch.object(tr, "get_models", light_models):
            quiet(tr.main)
        self.assertEqual(file_hashes(data_dir), before)  # girdi degismedi
        for name in ("model_comparison.csv", "predictions.csv", "holdout_comparison.csv",
                     "subperiod_analysis.csv", "feature_importance.csv",
                     "evaluation_summary.json", "validation_comparison.csv",
                     "validation_predictions.csv", "significance.csv",
                     "annual_comparison.csv", "run_config.json"):
            self.assertTrue((out_dir / name).exists(), name)
        cfg = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
        for key in ("target", "forecast_horizon", "periods", "models", "random_seeds",
                    "environment", "dataset", "run_timestamp_utc", "selection"):
            self.assertIn(key, cfg)
        self.assertEqual(cfg["periods"]["test"]["start"], tr.TEST_START)
        self.assertIn("dogrulama", cfg["selection"]["selected_on"])
        self.assertEqual(len(cfg["dataset"]["md5"]), 32)
        sig = pd.read_csv(out_dir / "significance.csv")
        self.assertEqual(set(sig["period"]), {"validation", "test"})
        annual = pd.read_csv(out_dir / "annual_comparison.csv")
        self.assertEqual(set(annual["model"]), set(light_models()))


# ---------------------------------------------------------------------------
# 9) Faz B: bilesen ozellikleri ve izole deney
# ---------------------------------------------------------------------------
BASE_11 = ["gold_return", "gold_usd_return", "usdtry_return", "return_lag_1",
           "return_lag_2", "return_lag_5", "return_lag_10", "return_mean_5",
           "return_mean_20", "return_volatility_20", "day_of_week"]
NEW_6 = ["fx_mean_20", "fx_mean_60", "gold_usd_mean_20", "gold_usd_mean_60",
         "fx_vol_20", "gold_usd_vol_20"]


def phase_b_models():
    """Hizli test modelleri: kabul kurali en az iki ML modeli gerektirir."""
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    return {
        "Zero-Return": "zero",
        "Mean-Return": "mean",
        "Last-Return": "last",
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    }


class TestComponentFeatures(unittest.TestCase):

    def setUp(self):
        self.raw = dataset()
        self.feat = tr.build_features(self.raw)

    def test_formulas_match_manual_values(self):
        fx = self.raw["usdtry_return"].to_numpy()
        gd = self.raw["gold_usd_return"].to_numpy()
        f = self.feat
        t = 100
        self.assertAlmostEqual(f.loc[t, "fx_mean_20"], fx[t - 19:t + 1].mean(), places=15)
        self.assertAlmostEqual(f.loc[t, "gold_usd_mean_20"], gd[t - 19:t + 1].mean(), places=15)
        self.assertAlmostEqual(f.loc[t, "fx_vol_20"], np.std(fx[t - 19:t + 1], ddof=1), places=15)
        self.assertAlmostEqual(f.loc[t, "gold_usd_vol_20"], np.std(gd[t - 19:t + 1], ddof=1), places=15)
        t = 200  # tam 60'lik pencere
        self.assertAlmostEqual(f.loc[t, "fx_mean_60"], fx[t - 59:t + 1].mean(), places=15)
        self.assertAlmostEqual(f.loc[t, "gold_usd_mean_60"], gd[t - 59:t + 1].mean(), places=15)
        t = 30   # kisa gecmis: min_periods=20, satir 0..30
        self.assertAlmostEqual(f.loc[t, "fx_mean_60"], fx[:t + 1].mean(), places=15)
        self.assertAlmostEqual(f.loc[t, "gold_usd_mean_60"], gd[:t + 1].mean(), places=15)

    def test_first_valid_row_is_19(self):
        for col in NEW_6:
            with self.subTest(col=col):
                self.assertTrue(self.feat[col].iloc[:19].isna().all())
                self.assertFalse(self.feat[col].iloc[19:].isna().any())

    def test_truncation_invariance(self):
        for t in (19, 45, 70, 400, len(self.feat) - 2):
            with self.subTest(t=t):
                cut = tr.build_features(self.raw.iloc[:t + 1])
                pd.testing.assert_series_equal(self.feat.loc[t, NEW_6], cut.loc[t, NEW_6],
                                               check_names=False)


class TestFeatureSets(TempDir):

    def test_default_feature_list_is_unchanged(self):
        self.assertEqual(tr.FEATURES, BASE_11)
        self.assertEqual(tr.FEATURE_SETS["base"], BASE_11)
        self.assertEqual(tr.FEATURE_SETS["component"], BASE_11 + NEW_6)

    def test_compact_set_contents(self):
        compact = tr.FEATURE_SETS["compact"]
        self.assertEqual(compact, ["gold_usd_return", "usdtry_return", *NEW_6, "day_of_week"])
        self.assertEqual(len(compact), 9)
        self.assertNotIn("gold_return", compact)
        self.assertFalse([c for c in compact if c.startswith("return_")])

    def test_warmup_and_sample_identical_for_all_sets(self):
        raw = dataset()
        df = self.load(raw)
        self.assertEqual(tr.WARMUP_ROWS, 19)
        self.assertEqual(len(df), len(raw) - 19)
        for name, cols in tr.FEATURE_SETS.items():
            with self.subTest(set=name):
                self.assertFalse(df[cols].isna().any().any())

    def test_default_walk_forward_uses_base_set(self):
        df = self.load(dataset())
        with mock.patch.object(tr, "get_models", light_models):
            a = quiet(tr.walk_forward_evaluate, df)[0]["predictions"]
            b = quiet(tr.walk_forward_evaluate, df, None, None, "test",
                      tr.FEATURE_SETS["base"])[0]["predictions"]
        pd.testing.assert_frame_equal(a, b)
        self.assertNotIn("window", a.columns)  # legacy predictions.csv semasi korunur

    def test_default_main_does_not_run_experiment(self):
        data_dir = self.tmp / "processed"
        data_dir.mkdir()
        dataset().to_csv(data_dir / "model_dataset.csv", index=False, date_format="%Y-%m-%d")
        out_dir = data_dir / "results"
        with mock.patch.object(tr, "DATA_DIR", data_dir), \
                mock.patch.object(tr, "OUTPUT_DIR", out_dir), \
                mock.patch.object(tr, "get_models", light_models):
            quiet(tr.main)
        self.assertFalse((out_dir / "experiments").exists())
        legacy = pd.read_csv(out_dir / "model_comparison.csv")
        self.assertEqual(list(legacy.columns[:7]), ["MAE_return", "RMSE_return",
                         "direction_accuracy_pct", "MAE_TL", "RMSE_TL", "skill_vs_zero", "model"])


class TestPhaseBExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="phase_b_"))
        cls.data_dir = cls.tmp / "processed"
        cls.out_dir = cls.data_dir / "results"
        cls.out_dir.mkdir(parents=True)
        dataset().to_csv(cls.data_dir / "model_dataset.csv", index=False, date_format="%Y-%m-%d")
        # Mevcut sonuc dosyalarini taklit et: deney bunlara dokunmamali
        (cls.out_dir / "model_comparison.csv").write_text("eski,sonuc\n1,2\n", encoding="utf-8")
        (cls.out_dir / "run_config.json").write_text('{"eski": true}', encoding="utf-8")
        cls.before = file_hashes(cls.out_dir)
        with mock.patch.object(tr, "DATA_DIR", cls.data_dir), \
                mock.patch.object(tr, "OUTPUT_DIR", cls.out_dir), \
                mock.patch.object(tr, "get_models", phase_b_models):
            cls.res = quiet(tr.run_phase_b_main)[0]
        cls.exp_dir = cls.out_dir / "experiments" / "phase_b"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_outputs_only_in_experiment_folder(self):
        self.assertEqual(file_hashes(self.out_dir), self.before)
        for name in ("metrics.csv", "window_metrics.csv", "annual.csv", "significance.csv",
                     "decision.csv", "experiment_summary.json", "experiment_log.jsonl",
                     "validation_predictions_base.csv", "validation_predictions_component.csv",
                     "validation_predictions_compact.csv"):
            self.assertTrue((self.exp_dir / name).exists(), name)

    def test_cli_flag_runs_experiment(self):
        with mock.patch.object(tr, "run_phase_b_main") as run:
            tr.main(["--experiment", "phase_b"])
        run.assert_called_once()

    def test_test_period_not_used(self):
        summary = json.loads((self.exp_dir / "experiment_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["test_period_used"])
        for fs in tr.PHASE_B_SETS:
            pred = self.res["results"][fs]["predictions"]
            self.assertLess(pd.to_datetime(pred["target_date"]).max(), pd.Timestamp(tr.TEST_START))

    def test_baselines_identical_across_sets(self):
        base = self.res["results"]["base"]["predictions"]
        for fs in ("component", "compact"):
            pred = self.res["results"][fs]["predictions"]
            for b in ("Zero-Return", "Mean-Return", "Last-Return"):
                with self.subTest(set=fs, baseline=b):
                    np.testing.assert_array_equal(base[f"pred_{b}"], pred[f"pred_{b}"])

    def test_feature_sets_actually_change_model_inputs(self):
        base = self.res["results"]["base"]["predictions"]
        for fs in ("component", "compact"):
            pred = self.res["results"][fs]["predictions"]
            with self.subTest(set=fs):
                self.assertFalse(np.allclose(base["pred_Linear Regression"],
                                             pred["pred_Linear Regression"]))

    def test_sample_and_windows_identical_across_sets(self):
        ref = [(w["n_train"], w["n_eval"], w["eval_first_date"]) for w in self.res["results"]["base"]["windows"]]
        for fs in ("component", "compact"):
            got = [(w["n_train"], w["n_eval"], w["eval_first_date"]) for w in self.res["results"][fs]["windows"]]
            self.assertEqual(got, ref)
        summary = self.res["summary"]
        self.assertEqual({k: v["n_features"] for k, v in summary["feature_sets"].items()},
                         {"base": 11, "component": 17, "compact": 9})
        self.assertEqual(summary["warmup_rows"], 19)

    def test_window_metrics_use_correct_rows(self):
        res = self.res["results"]["base"]
        pred = res["predictions"]
        table = self.res["windows"]
        base_rows = table[(table["feature_set"] == "base") & (table["model"] == "Linear Regression")]
        self.assertEqual(int(base_rows["n_obs"].sum()), len(pred))
        dates = pd.to_datetime(pred["prediction_date"])
        for _, row in base_rows.iterrows():
            m = (dates >= pd.Timestamp(row["eval_first_date"])) & (dates <= pd.Timestamp(row["eval_last_date"]))
            expected = float(np.mean(np.abs(pred.loc[m, "actual_return"] - pred.loc[m, "pred_Linear Regression"])))
            self.assertEqual(int(m.sum()), row["n_obs"])
            self.assertAlmostEqual(row["MAE_return"], expected, places=15)

    def test_decision_table_covers_candidates_and_ml_models(self):
        d = self.res["decision"]
        self.assertEqual(set(d["feature_set"]), {"component", "compact"})
        self.assertEqual(set(d["model"]), {"Linear Regression", "Ridge", "Tree"})
        for fs, v in self.res["verdicts"].items():
            self.assertEqual(v["accepted"], int(d[d["feature_set"] == fs]["passed"].sum()) >= 2)


class TestPhaseBRules(unittest.TestCase):

    def test_holm_adjust(self):
        np.testing.assert_allclose(tr.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(tr.holm_adjust([0.5, 0.9]), [1.0, 1.0])

    def test_required_window_wins(self):
        self.assertEqual(tr.required_window_wins(8), 5)
        self.assertEqual(tr.required_window_wins(10), 7)
        self.assertEqual(tr.required_window_wins(6), 4)

    def _fake_results(self, cand_scale, worse_2023=False):
        """8 pencere x 4 satir; 2022 ve 2023. Aday hatasi = base hatasi * olcek."""
        dates = pd.bdate_range("2022-12-01", periods=32)  # 2022 ve 2023'u kapsar
        actual = np.full(32, 0.01)
        err_base = np.full(32, 0.004)
        err_cand = err_base * cand_scale
        if worse_2023:
            err_cand = np.where(dates.year == 2023, err_base * 1.5, err_cand)
        wid = np.repeat(np.arange(1, 9), 4)

        def res(err):
            pred = pd.DataFrame({"prediction_date": dates, "actual_return": actual,
                                 "pred_M1": actual - err, "pred_M2": actual - err})
            return {"predictions": pred, "window_of_row": wid}
        return {"base": res(err_base), "component": res(err_cand), "compact": res(err_base)}

    def test_rule_accepts_uniform_improvement(self):
        d = tr.evaluate_acceptance(self._fake_results(0.9), ["M1", "M2"])
        comp = d[d["feature_set"] == "component"]
        self.assertTrue(comp["passed"].all())
        self.assertEqual(int(comp["windows_won"].iloc[0]), 8)
        # Ayni hatalar: daha dusuk degil; esitlik pencere kazanimi sayilmaz
        same = d[d["feature_set"] == "compact"]
        self.assertFalse(same["passed"].any())
        self.assertTrue((same["windows_won"] == 0).all())

    def test_fixture_spans_both_years(self):
        dates = self._fake_results(0.9)["base"]["predictions"]["prediction_date"]
        self.assertEqual(set(dates.dt.year), {2022, 2023})

    def test_missing_year_counts_as_not_satisfied(self):
        res = self._fake_results(0.9)
        for r in res.values():
            r["predictions"]["prediction_date"] = pd.bdate_range("2022-01-03", periods=32)
        d = tr.evaluate_acceptance(res, ["M1", "M2"])
        comp = d[d["feature_set"] == "component"]
        self.assertTrue(comp["year_2023_not_worse"].isna().all())
        self.assertFalse(comp["passed"].any())

    def test_rule_rejects_when_one_year_is_worse(self):
        d = tr.evaluate_acceptance(self._fake_results(0.5, worse_2023=True), ["M1", "M2"])
        comp = d[d["feature_set"] == "component"]
        self.assertTrue(comp["mae_lower_than_base"].all())
        self.assertFalse(comp["year_2023_not_worse"].any())
        self.assertFalse(comp["passed"].any())


# ---------------------------------------------------------------------------
# Gercek proje verisinin korunmasi
# ---------------------------------------------------------------------------
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
