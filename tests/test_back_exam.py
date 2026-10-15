"""
Geriye donuk sinav testleri: src/models/back_exam.py (BACK_EXAM_ONS_V1)
Calistirma: python -m unittest tests.test_back_exam -v
Sentetik dunya; gecici klasorler. Gercek data/ ve research/ kullanilmaz (md5 korumasi). Ag erisimi yok.
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

from src.models import back_exam as be  # noqa: E402


def world(b_true=0.8, seed=1, start="2000-06-01", end="2005-03-31", gold_start="2000-08-30", noise=0.004, drift=0.0005):
    """Fiziki seri (IAB kapanisi) + Yahoo ons (IAB kapanisindan sonra). Gece ons hareketi g, ertesi gune b_true ile yansir."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    days = days[rng.random(len(days)) > 0.02]
    n = len(days)
    usd = 270 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    g = rng.normal(0, 0.006, n)
    gold_close = usd * np.exp(0.002 + g)
    step = b_true * g[:-1] + rng.normal(drift, noise, n - 1)
    P = 9.0 * np.exp(np.concatenate([[0.0], np.cumsum(step)]))
    phys = pd.DataFrame({"Date": days, "kapanis_tl_gr": P, "kapanis_usd_ons": usd})
    k = days >= pd.Timestamp(gold_start)
    gold = pd.DataFrame({"Date": days[k], "close": gold_close[k]})
    return phys, gold


def write_snapshot(root: Path, gold: pd.DataFrame, name="20261016T100000Z", status="ok"):
    snap = root / name
    snap.mkdir(parents=True)
    lines = ["Price,Adj Close,Close,High,Low,Open,Volume", "Ticker,GC=F,GC=F,GC=F,GC=F,GC=F,GC=F", "Date,,,,,,"]
    lines += [f"{d.date()},{c},{c},{c},{c},{c},0" for d, c in zip(gold["Date"], gold["close"])]
    (snap / "gold.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (snap / "manifest.json").write_text(json.dumps({"series": {"gold": {"status": status}}}), encoding="utf-8")
    return snap


class TestBuildRows(unittest.TestCase):
    def setUp(self):
        self.phys, self.gold = world()

    def test_basic_rules(self):
        r = be.build_rows(self.phys, self.gold, be.DATA_END)
        self.assertLess(r["T"].max(), pd.Timestamp(be.DATA_END))
        self.assertTrue(((r["T"] - r["t"]).dt.days <= be.MAX_GAP).all())
        self.assertGreaterEqual(r["t"].min(), pd.Timestamp("2000-08-30"))
        g = self.gold.set_index("Date")["close"]
        self.assertTrue(np.allclose(r["gold_t"], r["t"].map(g)))            # ayni gun t
        self.assertTrue(np.allclose(r["y"], np.log(r["P_T"] / r["P_t"])))

    def test_stale_and_nonpositive_excluded(self):
        gold = self.gold.copy()
        i = 200
        gold.loc[i, "close"] = gold.loc[i - 1, "close"]
        phys = self.phys.copy()
        bad = self.phys["Date"] == gold.loc[300, "Date"]
        phys.loc[bad, "kapanis_usd_ons"] = 0.0
        r = be.build_rows(phys, gold, be.DATA_END)
        self.assertNotIn(gold.loc[i, "Date"], set(r["t"]))
        self.assertNotIn(gold.loc[300, "Date"], set(r["t"]) | set(r["T"]))

    def test_basis_uses_only_past(self):
        r1 = be.build_rows(self.phys, self.gold, "2003-01-01")
        r2 = be.build_rows(self.phys, self.gold, be.DATA_END)
        common = r1.iloc[:-1]
        pd.testing.assert_frame_equal(common, r2.iloc[:len(common)])
        gold = self.gold.copy()
        gold.loc[gold["Date"] >= "2003-06-02", "close"] *= 1.5             # gelecek degisir
        r3 = be.build_rows(self.phys, gold, be.DATA_END)
        early = r2["t"] < pd.Timestamp("2003-06-02")
        pd.testing.assert_series_equal(r2.loc[early, "g_ons"], r3.loc[early, "g_ons"])

    def test_basis_is_median_of_previous_20(self):
        r = be.build_rows(self.phys, self.gold, be.DATA_END)
        # build_rows ilk 20 satiri (taban yok) dusurur; ham seri bagimsiz olarak yeniden kurulur
        p = self.phys.sort_values("Date").reset_index(drop=True)
        g = self.gold.set_index("Date")["close"]
        t = p["Date"].iloc[:-1].reset_index(drop=True)
        ok = ((p["Date"].shift(-1) - p["Date"]).dt.days.iloc[:-1] <= 4).to_numpy() & t.isin(g.index).to_numpy()
        tt = t[ok].reset_index(drop=True)
        rr = np.log(tt.map(g).to_numpy() / p.set_index("Date").loc[tt, "kapanis_usd_ons"].to_numpy())
        for k in (20, 57, 300, 800):
            want = rr[k] - np.median(rr[k - 20:k])
            row = r.index[r["t"] == tt[k]][0]
            self.assertAlmostEqual(r.loc[row, "g_ons"], want, places=12)
        self.assertEqual(list(r["t"].iloc[:5]), list(tt.iloc[20:25]))             # ilk 20 satir tabansiz dusurulur

    def test_max_gap(self):
        phys = self.phys[~self.phys["Date"].between("2002-03-05", "2002-03-12")]
        r = be.build_rows(phys, self.gold, be.DATA_END)
        self.assertTrue(((r["T"] - r["t"]).dt.days <= 4).all())
        self.assertNotIn(pd.Timestamp("2002-03-13"), set(r["T"][r["t"] < pd.Timestamp("2002-03-05")]))


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        phys, gold = world()
        self.df = be.build_rows(phys, gold, be.DATA_END)

    def test_eval_range_and_min_train(self):
        p = be.walk_forward(self.df)
        self.assertGreaterEqual(p["T"].min(), pd.Timestamp(be.EVAL_START))
        self.assertLessEqual(p["T"].max(), pd.Timestamp(be.EVAL_END))
        self.assertGreaterEqual(int(p["n_train"].min()), be.MIN_TRAIN)
        self.assertEqual(len(p), p["T"].nunique())

    def test_no_future_information(self):
        p = be.walk_forward(self.df)
        first_T = p["T"].iat[0]
        w0 = int(self.df.index[self.df["T"] == first_T][0])
        bad = self.df.copy()
        bad.loc[w0 - 1:, "y"] = 5.0                                         # purge satiri ve sonrasi bozulur
        q = be.walk_forward(bad)
        n1 = int((p["n_train"] == p["n_train"].iat[0]).sum())
        np.testing.assert_allclose(p["pred_Ons"].iloc[:n1], q["pred_Ons"].iloc[:n1])
        np.testing.assert_allclose(p["pred_Egilim"].iloc[:n1], q["pred_Egilim"].iloc[:n1])
        tr0 = self.df.loc[:w0 - 2]
        self.assertAlmostEqual(p["pred_Egilim"].iat[0], tr0["y"].mean())
        self.assertTrue((q["n_train"].iloc[:n1] == w0 - 1).all())

    def test_late_start_shifts_first_evaluation(self):
        phys, gold = world(gold_start="2000-11-20")
        df = be.build_rows(phys, gold, be.DATA_END)
        self.assertLess(int((df["T"] < pd.Timestamp(be.EVAL_START)).sum()), be.MIN_TRAIN)
        p = be.walk_forward(df)
        self.assertGreater(p["T"].min(), pd.Timestamp("2001-01-15"))
        self.assertEqual(int(p["n_train"].iat[0]), be.MIN_TRAIN)

    def test_refuses_without_rows(self):
        with self.assertRaises(be.Refused):
            be.walk_forward(self.df[self.df["T"] < pd.Timestamp("2000-12-01")])


class TestDevCoefficients(unittest.TestCase):
    def test_lock_and_estimate(self):
        phys, gold = world(start="2010-06-01", end="2024-06-28", gold_start="2010-06-01", seed=3)
        panel = gold.rename(columns={"close": "gold_usd_oz"})
        d1 = be.dev_coefficients(phys, panel)
        phys2, panel2 = phys.copy(), panel.copy()
        phys2.loc[phys2["Date"] >= "2023-01-01", "kapanis_tl_gr"] = 1.0
        panel2.loc[panel2["Date"] >= "2023-01-01", "gold_usd_oz"] = 1e6
        self.assertEqual(d1, be.dev_coefficients(phys2, panel2))
        self.assertLess(d1["last_T"], "2023-01-01")
        self.assertGreaterEqual(d1["first_T"], "2011-01-01")
        self.assertAlmostEqual(d1["b"], 0.8, delta=0.1)


class TestEvaluate(unittest.TestCase):
    def test_signal_detected(self):
        phys, gold = world(b_true=0.8)
        pred = be.walk_forward(be.build_rows(phys, gold, be.DATA_END))
        r = be.evaluate(pred, {"a": 0.0005, "b": 0.8})
        for k in ("Ons_vs_RW", "Ons_vs_Egilim"):
            self.assertEqual(r["primary_walk_forward"][k]["verdict"], "DAHA_IYI")
            self.assertLessEqual(r["primary_walk_forward"][k]["p_value"], r["primary_walk_forward"][k]["p_used"])
        self.assertTrue(r["hypothesis_supported"])
        pw = r["primary_walk_forward"]
        adj = be.holm({k: v["p_value"] for k, v in pw.items()})
        for k in pw:
            self.assertEqual(pw[k]["p_used"], adj[k])
        self.assertEqual(r["secondary_frozen_dev"]["Donuk_vs_RW"]["verdict"], "DAHA_IYI")
        self.assertEqual([y["year"] for y in r["yearly"]], [2001, 2002, 2003, 2004])

    def test_drift_alone_is_not_ons_information(self):
        """Gece ons etkisi yok, ama guclu TL surukleme var: Ons RW'yi gecebilir, Egilim'i GECEMEZ."""
        phys, gold = world(b_true=0.0, seed=5)
        pred = be.walk_forward(be.build_rows(phys, gold, be.DATA_END))
        r = be.evaluate(pred, {"a": 0.0, "b": 0.8})
        self.assertGreater(r["descriptive_Egilim_vs_RW"]["improvement_pct"], 0)
        self.assertNotEqual(r["primary_walk_forward"]["Ons_vs_Egilim"]["verdict"], "DAHA_IYI")
        self.assertFalse(r["hypothesis_supported"])
        self.assertEqual(r["secondary_frozen_dev"]["Donuk_vs_RW"]["verdict"], "DAHA_KOTU")   # yanlis donuk b zarar verir

    def test_strong_drift_beats_rw_but_not_supported(self):
        phys, gold = world(b_true=0.0, seed=8, drift=0.003)
        pred = be.walk_forward(be.build_rows(phys, gold, be.DATA_END))
        r = be.evaluate(pred, {"a": 0.003, "b": 0.0})
        self.assertEqual(r["primary_walk_forward"]["Ons_vs_RW"]["verdict"], "DAHA_IYI")
        self.assertNotEqual(r["primary_walk_forward"]["Ons_vs_Egilim"]["verdict"], "DAHA_IYI")
        self.assertFalse(r["hypothesis_supported"])

    def test_holm(self):
        self.assertEqual(be.holm({"x": 0.01, "y": 0.04}), {"x": 0.02, "y": 0.04})
        self.assertEqual(be.holm({"x": 0.03, "y": 0.02}), {"x": 0.04, "y": 0.04})
        self.assertEqual(be.holm({"x": 0.6, "y": 0.9}), {"x": 1.0, "y": 1.0})

    def test_verdict_rules(self):
        v = lambda d, p, pv: be.point_verdict({"diff_ci95": d, "pct_ci95": p}, {"p_value": pv})
        self.assertEqual(v([-2, -0.5], [-9, -1], 0.01), "DAHA_IYI")
        self.assertEqual(v([-2, -0.5], [-9, -1], 0.2), "AYIRT_EDILEMEDI")
        self.assertEqual(v([0.5, 2], [1, 9], 0.01), "DAHA_KOTU")
        self.assertEqual(v([-0.1, 0.1], [-1.9, 1.9], 0.6), "PRATIKTE_ESDEGER")
        self.assertEqual(v([-0.1, 0.3], [-1.0, 3.5], 0.4), "AYIRT_EDILEMEDI")
        self.assertEqual(v([-0.3, 0.1], [-3.5, 1.0], 0.4), "AYIRT_EDILEMEDI")


class TestFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="be_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for f in be.CODE_FILES:
            (self.tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / f, self.tmp / f)
        phys, gold = world()
        dphys, dgold = world(start="2010-06-01", end="2024-06-28", gold_start="2010-06-01", seed=3)
        allphys = pd.concat([phys, dphys[dphys["Date"] > phys["Date"].max()]], ignore_index=True)
        self.phys_path = self.tmp / "phys.csv"
        allphys.to_csv(self.phys_path, index=False)
        self.panel_path = self.tmp / "panel.csv"
        dgold.rename(columns={"close": "gold_usd_oz"}).to_csv(self.panel_path, index=False)
        self.snaps = self.tmp / "data" / "raw" / "snapshots_backward"
        write_snapshot(self.snaps, gold, "20261016T090000Z", status="error")
        self.snap = write_snapshot(self.snaps, gold, "20261016T100000Z")
        p = mock.patch.multiple(be, ROOT=self.tmp, SNAP_ROOT=self.snaps, PHYS_PATH=self.phys_path,
                                HIST_PANEL=self.panel_path, DATA_DIR=self.tmp / "data" / "processed" / "back_exam",
                                PREREG_PATH=self.tmp / "research" / "preregistration" / "back_exam_ons_v1.json",
                                REGISTRY_PATH=self.tmp / "research" / "registry.jsonl",
                                RESULT_DIR=self.tmp / "research" / "back_exam_ons_v1")
        p.start()
        self.addCleanup(p.stop)

    def events(self):
        return [json.loads(l)["event"] for l in be.REGISTRY_PATH.read_text(encoding="utf-8").splitlines()]

    def test_latest_ok_snapshot(self):
        self.assertEqual(be.latest_snapshot(), self.snap)

    def test_prepare_then_one_shot_run(self):
        c = be.prepare()
        self.assertEqual(c["gold_snapshot"], self.snap.name)
        self.assertEqual(set(c["code_md5"]), set(be.CODE_FILES))
        self.assertGreater(c["counts"]["eval_candidates"], 900)
        self.assertNotIn("MAE", json.dumps(c))                               # on kayitta sonuc yok
        ds = pd.read_csv(be.DATA_DIR / "dataset.csv", parse_dates=["t", "T"])
        self.assertLess(ds["T"].max(), pd.Timestamp("2005-01-01"))
        self.assertEqual(c["parameters"]["data_end"], "2005-01-01")
        self.assertEqual(c["parameters"]["eval_targets"], ["2001-01-01", "2004-12-31"])
        with self.assertRaisesRegex(be.Refused, "zaten var"):
            be.prepare()
        with self.assertRaisesRegex(be.Refused, "Git"):
            be.run(git_fn=lambda paths: (False, "commit edilmemis"))
        self.assertEqual(self.events(), ["prepared"])
        r = be.run(git_fn=lambda paths: (True, "abc123"))
        self.assertTrue(r["hypothesis_supported"])
        for f in ("result.json", "predictions.csv", "yearly.csv"):
            self.assertTrue((be.RESULT_DIR / f).exists())
        self.assertEqual(self.events(), ["prepared", "started", "completed"])
        with self.assertRaisesRegex(be.Refused, "tekrar"):
            be.run(git_fn=lambda paths: (True, "abc123"))

    def test_run_refuses_changed_data_or_code(self):
        be.prepare()
        ds = be.DATA_DIR / "dataset.csv"
        orig = ds.read_bytes()
        ds.write_bytes(orig + b"\n")
        with self.assertRaisesRegex(be.Refused, "verisi"):
            be.run(git_fn=lambda paths: (True, "x"))
        ds.write_bytes(orig)
        f = self.tmp / "src" / "models" / "back_exam.py"
        f.write_bytes(f.read_bytes() + b"# degisti\n")
        with self.assertRaisesRegex(be.Refused, "Kod"):
            be.run(git_fn=lambda paths: (True, "x"))
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n").replace(b"# degisti\r\n", b""))
        self.assertEqual(be.run(git_fn=lambda paths: (True, "x"))["protocol_id"], be.PROTOCOL_ID)   # CRLF farksiz

    def test_dataset_round_trip_reproduces_predictions(self):
        be.prepare()
        df = pd.read_csv(be.DATA_DIR / "dataset.csv", parse_dates=["t", "T"], float_precision="round_trip")
        direct = be.build_rows(pd.read_csv(self.phys_path, parse_dates=["Date"]), be.load_gold(self.snap), be.DATA_END)
        a, b = be.walk_forward(df), be.walk_forward(direct)
        np.testing.assert_array_equal(a["pred_Ons"].to_numpy(), b["pred_Ons"].to_numpy())
        np.testing.assert_array_equal(a["pred_Egilim"].to_numpy(), b["pred_Egilim"].to_numpy())

    def test_main_fetch_uses_separate_root(self):
        seen = {}

        def fake_collect(names, start, end, snapshot_root=None, **kw):
            seen.update(names=names, start=start, end=end, root=snapshot_root)
            return {"snapshot_dir": "x", "series": {"gold": {"status": "ok"}}}
        with mock.patch.object(be.co, "collect", fake_collect), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(be.main(["fetch"]), 0)
        self.assertEqual(seen, {"names": ["gold"], "start": "2000-01-01", "end": "2005-01-01", "root": self.snaps})
        self.assertNotEqual(Path(seen["root"]).name, "snapshots")


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
