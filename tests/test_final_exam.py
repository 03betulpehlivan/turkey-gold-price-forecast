"""
Son sinav testleri: src/models/final_exam.py
Calistirma: python -m unittest tests.test_final_exam -v
Sentetik anlik goruntu, model veri seti ve panel; git kontrolu sahte fonksiyonla. Gercek data/ ve git kullanilmaz.
"""

from datetime import datetime, timezone
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
from src.data import build_panel as bp  # noqa: E402
from src.data import collect as co  # noqa: E402
from src.models import final_exam as fe  # noqa: E402
from tests.test_data_platform import yf_frame  # noqa: E402

DAYS = pd.bdate_range("2015-01-01", "2026-03-31")
RNG = np.random.default_rng(42)
N = len(DAYS)
H = np.zeros(N)
for _t in range(1, N):
    H[_t] = 0.97 * H[_t - 1] + RNG.normal(0, 0.15)
PATHS = {"GC=F": 1200 * np.exp(np.cumsum(RNG.normal(0.0002, 0.009 * np.exp(H)))),
         "USDTRY=X": 2.3 * np.exp(np.cumsum(RNG.normal(0.0008, 0.004, N))),
         "^GVZ": 15 * np.exp(H)}


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def fetch(ticker, start, end):
    return yf_frame(ticker, DAYS, PATHS[ticker])


def clean_git(paths):
    return {"ok": True, "commit": "abc123def456", "dirty": "", "prereg_tracked": True}


class Base(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fe_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        snaps = self.tmp / "snapshots"
        quiet(co.collect, ["gold", "usdtry", "gvz"], "2015-01-01", "2026-04-01", snaps, fetch,
              datetime(2026, 10, 11, 17, 41, 48, tzinfo=timezone.utc))
        manifest, series = bp.load_snapshot(snaps / fe.SNAPSHOT)
        proc = self.tmp / "processed"
        proc.mkdir()
        old = {k: v[v["Date"] <= "2025-12-31"] for k, v in series.items()}
        _, model_old = quiet(bd.build, old["gold"], old["usdtry"])
        model_old.to_csv(proc / "model_dataset.csv", index=False, date_format="%Y-%m-%d")
        panel, _ = bp.build_panel(series)
        panel.to_csv(proc / "market_panel.csv", index=False, date_format="%Y-%m-%d")
        (proc / "market_panel_report.json").write_text(json.dumps({"snapshot_dir": fe.SNAPSHOT}), encoding="utf-8")
        research = self.tmp / "research"
        self.patches = mock.patch.multiple(
            fe, SNAPSHOT_ROOT=snaps, RESEARCH_DIR=research,
            PREREG_PATH=research / "preregistration" / "final_exam_v1.json",
            REGISTRY_PATH=research / "registry.jsonl", RESULT_DIR=research / "final_exam_v1",
            EXAM_DATA_PATH=proc / "final_exam" / "model_dataset_2015_2026.csv",
            EXAM_THEO_PATH=proc / "final_exam" / "theoretical_2015_2026.csv",
            EXISTING_MODEL_PATH=proc / "model_dataset.csv", PANEL_PATH=proc / "market_panel.csv",
            PANEL_REPORT_PATH=proc / "market_panel_report.json", GIT=clean_git)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.proc = proc

    def events(self):
        return [e["event"] for e in fe.registry_entries()]


class TestPrepare(Base):

    def test_prepare_builds_separate_dataset_and_prereg(self):
        r = quiet(fe.prepare)
        self.assertEqual(r["status"], "olusturuldu")
        self.assertTrue(fe.EXAM_DATA_PATH.exists() and fe.EXAM_THEO_PATH.exists())
        old = pd.read_csv(self.proc / "model_dataset.csv")
        new = pd.read_csv(fe.EXAM_DATA_PATH)
        self.assertGreater(len(new), len(old))
        prereg = json.loads(fe.PREREG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(prereg["parameters"]["protocol_id"], "FINAL_EXAM_V1")
        self.assertIn("final_exam.py", prereg["code_md5"])
        self.assertIn("market_panel.csv", prereg["data_md5"])
        self.assertEqual(self.events(), ["prepared"])

    def test_prepare_rejects_mismatch_with_existing_dataset(self):
        m = pd.read_csv(self.proc / "model_dataset.csv")
        m.loc[100, "usdtry"] *= 1.001
        m.to_csv(self.proc / "model_dataset.csv", index=False)
        with self.assertRaisesRegex(fe.ExamError, "ayni degil"):
            quiet(fe.prepare)
        self.assertFalse(fe.PREREG_PATH.exists())

    def test_prepare_is_idempotent_but_protocol_is_locked(self):
        quiet(fe.prepare)
        self.assertEqual(quiet(fe.prepare)["status"], "zaten hazir")
        with mock.patch.object(fe, "EQUIV_MARGIN_PCT", 5.0):
            with self.assertRaisesRegex(fe.ExamError, "protokol degistirilemez"):
                quiet(fe.prepare)

    def test_panel_from_other_snapshot_rejected(self):
        (self.proc / "market_panel_report.json").write_text(json.dumps({"snapshot_dir": "baska"}), encoding="utf-8")
        with self.assertRaisesRegex(fe.ExamError, "farkli bir anlik"):
            quiet(fe.prepare)


class TestRunGuards(Base):

    def test_refuses_without_prereg(self):
        with self.assertRaisesRegex(fe.ExamError, "On kayit yok"):
            quiet(fe.run)
        self.assertEqual(self.events(), ["refused"])

    def test_refuses_without_git_commit(self):
        quiet(fe.prepare)
        with mock.patch.object(fe, "GIT", lambda p: {"ok": True, "commit": "x", "dirty": "", "prereg_tracked": False}):
            with self.assertRaisesRegex(fe.ExamError, "Git"):
                quiet(fe.run)
        with mock.patch.object(fe, "GIT", lambda p: {"ok": True, "commit": "x", "dirty": " M src/models/train.py",
                                                     "prereg_tracked": True}):
            with self.assertRaisesRegex(fe.ExamError, "Git"):
                quiet(fe.run)
        self.assertEqual(self.events(), ["prepared", "refused", "refused"])

    def test_refuses_when_data_changed_after_prereg(self):
        quiet(fe.prepare)
        txt = fe.EXAM_DATA_PATH.read_text(encoding="utf-8")
        fe.EXAM_DATA_PATH.write_text(txt + "\n", encoding="utf-8")
        with self.assertRaisesRegex(fe.ExamError, "on kayitla ayni degil"):
            quiet(fe.run)

    def test_refuses_when_parameters_changed_after_prereg(self):
        quiet(fe.prepare)
        with mock.patch.object(fe, "NONINF_MARGIN_PCT", 10.0):
            with self.assertRaisesRegex(fe.ExamError, "on kayitla ayni degil"):
                quiet(fe.run)


class TestRun(Base):

    def setUp(self):
        super().setUp()
        quiet(fe.prepare)
        self.result = quiet(fe.run)

    def test_single_run_only_even_with_other_output_dir(self):
        with mock.patch.object(fe, "RESULT_DIR", self.tmp / "baska_klasor"):
            with self.assertRaises(fe.ExamAlreadyRun):
                quiet(fe.run)
        self.assertEqual(self.events(), ["prepared", "started", "completed", "refused"])
        self.assertEqual(quiet(fe.main, ["run"]), 2)

    def test_exam_set_is_2026_targets_including_last_2025_origin(self):
        pred = pd.read_csv(fe.RESULT_DIR / "predictions.csv", parse_dates=["prediction_date", "target_date"])
        self.assertGreaterEqual(pred["target_date"].min(), pd.Timestamp("2026-01-01"))
        self.assertEqual(pred["prediction_date"].min(), pd.Timestamp("2025-12-31"))
        theo = pd.read_csv(fe.EXAM_THEO_PATH, parse_dates=["Date"])
        self.assertEqual(len(pred), int((theo["Date"] >= "2026-01-01").sum()))
        self.assertEqual(self.result["n_forecasts"], len(pred))

    def test_result_contents_and_registry_consistency(self):
        r = json.loads((fe.RESULT_DIR / "result.json").read_text(encoding="utf-8"))
        self.assertIn(r["point"]["verdict"], {"DAHA_IYI", "DAHA_KOTU", "PRATIKTE_ESDEGER", "AYIRT_EDILEMEDI"})
        self.assertIn("all_passed", r["interval"]["verdict"])
        self.assertEqual(r["git_commit"], "abc123def456")
        done = [e for e in fe.registry_entries() if e["event"] == "completed"][0]
        self.assertEqual(done["result_md5"], hashlib.md5((fe.RESULT_DIR / "result.json").read_bytes()).hexdigest())
        for w in r["window_audit"]:
            self.assertLess(w["train_last_date"], w["eval_first_date"])
            self.assertLessEqual(w["train_last_target_date"], w["eval_first_date"])


class TestNoLookAhead(Base):

    def test_future_rows_cannot_change_earlier_forecasts(self):
        quiet(fe.prepare)
        df = fe.load_exam_frame()
        a = fe.generate(df)["predictions"]
        cut = pd.Timestamp("2026-02-16")
        bad = df.copy()
        m = bad["Date"] >= cut
        cols = ["gold_return", "gold_usd_return", "usdtry_return", "target_next_return", "theoretical_gram_try",
                "gvz", "gvz_lag1"]
        bad.loc[m, cols] = bad.loc[m, cols] * 3.0
        b = fe.generate(bad)["predictions"]
        keep = pd.to_datetime(a["prediction_date"]) < cut
        qcols = [c for c in a.columns if "_q" in c] + ["pred_Mean-Return"]
        pd.testing.assert_frame_equal(a.loc[keep, qcols].reset_index(drop=True), b.loc[keep, qcols].reset_index(drop=True))


class TestStatistics(unittest.TestCase):

    def test_wilson(self):
        lo, hi = fe.wilson(95, 100)
        self.assertAlmostEqual(lo, 0.8882, places=3)
        self.assertAlmostEqual(hi, 0.9785, places=3)

    def test_hln_factor_and_t_pvalue(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(200) + 0.05, rng.random(200)
        r = fe.dm_hln(a, b)
        self.assertAlmostEqual(r["dm_stat_hln"], r["dm_stat_nw"] * np.sqrt(199 / 200))
        self.assertLess(r["p_value"], 0.2)

    def test_bootstrap_identical_losses(self):
        e = np.random.default_rng(1).random(150)
        r = fe.paired_bootstrap(e, e, reps=200)
        self.assertEqual(r["diff_ci95"], [0.0, 0.0])
        self.assertEqual(r["pct_ci95"], [0.0, 0.0])

    def test_point_verdict_categories(self):
        dm_sig, dm_ns = {"p_value": 0.01}, {"p_value": 0.4}
        self.assertEqual(fe.point_verdict({"diff_ci95": [-2, -1], "pct_ci95": [-5, -3]}, dm_sig), "DAHA_IYI")
        self.assertEqual(fe.point_verdict({"diff_ci95": [1, 2], "pct_ci95": [3, 5]}, dm_sig), "DAHA_KOTU")
        self.assertEqual(fe.point_verdict({"diff_ci95": [-0.1, 0.1], "pct_ci95": [-1.5, 1.5]}, dm_ns), "PRATIKTE_ESDEGER")
        self.assertEqual(fe.point_verdict({"diff_ci95": [-1, 1], "pct_ci95": [-4, 4]}, dm_ns), "AYIRT_EDILEMEDI")
        self.assertEqual(fe.point_verdict({"diff_ci95": [-2, -1], "pct_ci95": [-5, -3]}, dm_ns), "AYIRT_EDILEMEDI")

    def test_interval_verdict(self):
        good = {"kupiec_80_p": 0.5, "kupiec_95_p": 0.5, "christoffersen_80_p": 0.5}
        self.assertTrue(fe.interval_verdict(good, {"pct_ci95": [-3, 1.5]})["all_passed"])
        self.assertFalse(fe.interval_verdict(good, {"pct_ci95": [-3, 2.5]})["all_passed"])
        self.assertFalse(fe.interval_verdict({**good, "kupiec_95_p": 0.01}, {"pct_ci95": [-3, 1]})["all_passed"])
        self.assertFalse(fe.interval_verdict({**good, "christoffersen_80_p": float("nan")}, {"pct_ci95": [-3, 1]})["all_passed"])
        self.assertNotIn("KANITLANDI", fe.interval_verdict(good, {"pct_ci95": [-3, 1]})["statement"].replace("KANITLANMIS", ""))


def _hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


_REAL = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "research"]
_BEFORE = {}


def setUpModule():
    for d in _REAL:
        _BEFORE[d] = _hashes(d)


def tearDownModule():
    for d in _REAL:
        if _hashes(d) != _BEFORE[d]:
            raise AssertionError(f"Testler gercek klasoru degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
