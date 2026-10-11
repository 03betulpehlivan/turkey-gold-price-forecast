"""
Nokta tahmini degerlendirme testleri: src/models/point_eval.py
=============================================================
Calistirma (proje kokunden):
  python -m unittest tests.test_point_eval -v

Iki sentetik dunya:
  kur    : kur getirisi ertesi gune tasinir (AR(1)) -> FX-AR1 rastgele yuruyusu anlamli gecmeli
  bos    : hicbir ongorulebilirlik yok             -> hicbir model "anlamli iyi" cikmamali
RF/XGB testlerde kucuk agac sayisiyla calisir (uretim ayarlari degismez).
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

from src.data import build_dataset as bd  # noqa: E402
from src.models import point_eval as pe  # noqa: E402
from src.models import train as tr  # noqa: E402

FAST = [mock.patch.dict(tr.RF_PARAMS, {"n_estimators": 15}), mock.patch.dict(tr.XGB_PARAMS, {"n_estimators": 15})]


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def make_world(kind: str, seed: int = 0, end: str = "2025-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", end)
    n = len(days)
    r_gold = rng.normal(0.0002, 0.003, n)
    r_fx = np.empty(n)
    r_fx[0] = 0.0
    phi = 0.5 if kind == "kur" else 0.0     # belirgin kalicilik: 8'li Holm altinda da bulunabilmeli
    for t in range(1, n):
        r_fx[t] = 0.0004 + phi * (r_fx[t - 1] - 0.0004) + rng.normal(0, 0.006)
    gold = 1200 * np.exp(np.cumsum(r_gold))
    fx = 2.3 * np.exp(np.cumsum(r_fx))
    _, model = quiet(bd.build, pd.DataFrame({"Date": days, "close": gold}), pd.DataFrame({"Date": days, "close": fx}))
    return model


def write(tmp: Path, kind: str, seed: int = 0, end: str = "2025-12-31") -> Path:
    path = tmp / "model_dataset.csv"
    make_world(kind, seed, end).to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


class TestMetrics(unittest.TestCase):

    def test_point_metrics_known_values(self):
        P = np.array([100.0, 100.0, 100.0, 100.0])
        a = np.log(np.array([101.0, 99.0, 102.0, 100.5]) / 100)     # gercek: 101, 99, 102, 100.5
        p = np.log(np.array([100.7, 99.0, 101.0, 100.5]) / 100)     # tahmin: hatalar 0.3, 0, 1, 0
        m = pe.point_metrics(a, p, P)
        self.assertAlmostEqual(m["MAE_TL"], 0.325)
        self.assertAlmostEqual(m["MedAE_TL"], 0.15)
        self.assertAlmostEqual(m["RMSE_TL"], np.sqrt((0.09 + 1.0) / 4))
        self.assertAlmostEqual(m["share_err_le_0.4TL_pct"], 75.0)
        self.assertAlmostEqual(m["share_err_le_1TL_pct"], 100.0)    # 1.0 TL esik dahil
        self.assertAlmostEqual(m["direction_accuracy_pct"], 100.0)
        self.assertAlmostEqual(m["MAE_TL_down_days"], 0.0)
        self.assertAlmostEqual(m["bias_TL"], (-0.3 + 0 - 1 + 0) / 4)

    def test_alignment_check_detects_shifted_target(self):
        df = make_world("bos", end="2016-06-30")
        self.assertEqual(pe.verify_price_alignment(df)["rows_checked"], len(df) - 1)
        bad = df.copy()
        bad[tr.TARGET] = bad[tr.TARGET].shift(1).fillna(0.0)
        with self.assertRaisesRegex(ValueError, "hizalamasi"):
            pe.verify_price_alignment(bad)

    def test_rule_requires_every_condition(self):
        base = {"diff_TL": -1.0, "dm_stat": -3.0, "p_holm": 0.01, "diff_ci95_high": -0.1, "win_share": 0.7}
        rows = [base, {**base, "p_holm": 0.2}, {**base, "diff_ci95_high": 0.1}, {**base, "win_share": 0.5},
                {**base, "dm_stat": 1.0}, {**base, "diff_TL": 0.5}]
        out = pe.apply_rule(pd.DataFrame(rows))
        self.assertEqual(out["significantly_better"].tolist(), [True, False, False, False, False, False])

    def test_bootstrap_ci(self):
        e = np.random.default_rng(1).random(400)
        self.assertEqual(pe.block_bootstrap_diff_ci(e, e, reps=200), (0.0, 0.0))
        lo, hi = pe.block_bootstrap_diff_ci(e - 0.5, e, reps=200)
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, -0.5)


class TestWorlds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for p in FAST:
            p.start()
        cls.tmp = Path(tempfile.mkdtemp(prefix="pe_w_"))
        cls.res = {}
        for kind in ("kur", "bos"):
            d = cls.tmp / kind
            d.mkdir()
            path = write(d, kind, seed=2)
            with mock.patch.object(tr, "OUTPUT_DIR", d / "results"):
                cls.res[kind] = quiet(pe.run, path)

    @classmethod
    def tearDownClass(cls):
        for p in FAST:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fx_persistence_is_found(self):
        self.assertIn("FX-AR1", self.res["kur"]["summary"]["significantly_better_than_random_walk"])

    def test_no_signal_world_finds_nothing(self):
        self.assertEqual(self.res["bos"]["summary"]["significantly_better_than_random_walk"], [])

    def test_all_models_on_identical_rows_and_dev_period(self):
        pred = self.res["kur"]["predictions"]
        for m in list(tr.get_models()) + pe.NEW_MODELS:
            self.assertIn(f"pred_{m}", pred.columns)
            self.assertFalse(pred[f"pred_{m}"].isna().any(), m)
        self.assertGreaterEqual(pd.to_datetime(pred["prediction_date"]).min(), pd.Timestamp(pe.DEV_START))
        self.assertLess(pd.to_datetime(pred["target_date"]).max(), pd.Timestamp(tr.TEST_START))

    def test_existing_models_unchanged(self):
        df = quiet(tr.load_data, self.tmp / "kur" / "model_dataset.csv")
        wf = quiet(tr.walk_forward_evaluate, df, start=pe.DEV_START, end=tr.TEST_START, label="x")
        for m in ("Zero-Return", "Mean-Return", "Linear Regression"):
            np.testing.assert_array_equal(self.res["kur"]["predictions"][f"pred_{m}"], wf["predictions"][f"pred_{m}"])

    def test_artifact_warning_is_reported(self):
        c = self.res["kur"]["comparison_rw"].set_index("model")
        self.assertIn("saat damgasi", c.loc["Bilesen-Gecikme", "artifact_warning"])
        self.assertEqual(c.loc["FX-AR1", "artifact_warning"], "")


class TestProtection(unittest.TestCase):

    def setUp(self):
        for p in FAST:
            p.start()
            self.addCleanup(p.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="pe_p_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_test_period_values_cannot_change_predictions(self):
        df = quiet(tr.load_data, write(self.tmp, "kur", seed=3))
        a = pe.generate(df)["predictions"]
        bad = df.copy()
        m = bad["Date"] >= pd.Timestamp(tr.TEST_START)
        cols = list(dict.fromkeys(["gold_return", "gold_usd_return", "usdtry_return", tr.TARGET, tr.PRICE_COL]
                                  + tr.FEATURES))
        bad.loc[m, cols] = bad.loc[m, cols] * 3.0
        b = pe.generate(bad)["predictions"]
        pd.testing.assert_frame_equal(a, b)

    def test_preregistration_before_computation_and_isolation(self):
        path = write(self.tmp, "bos", seed=4)
        out_root = self.tmp / "results"
        before = file_hashes(self.tmp)
        with mock.patch.object(tr, "OUTPUT_DIR", out_root), \
                mock.patch.object(pe, "generate", side_effect=RuntimeError("durdu")):
            with self.assertRaises(RuntimeError):
                quiet(pe.run, path)
        target = out_root / "experiments" / "point_eval"
        prereg = json.loads((target / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(prereg["new_models"], pe.NEW_MODELS)
        self.assertFalse((target / "metrics.csv").exists())
        self.assertEqual(file_hashes(self.tmp), before)

    def test_2026_rows_are_dropped(self):
        path = write(self.tmp, "bos", seed=5, end="2026-03-31")
        with mock.patch.object(tr, "OUTPUT_DIR", self.tmp / "results"):
            res = quiet(pe.run, path)
        self.assertLess(pd.to_datetime(res["predictions"]["target_date"]).max(), pd.Timestamp(tr.TEST_START))


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
