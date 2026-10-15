"""
LIVE_V1 degerlendirme on kaydi testleri: src/live/forecast_eval.py
Calistirma: python -m unittest tests.test_forecast_eval -v
Sentetik kayitlar; gecici kok klasor. Gercek research/ ve data/processed/ kullanilmaz (md5 korumasi).
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

from src.live import forecast as lf  # noqa: E402
from src.live import forecast_eval as le  # noqa: E402

CREATED = "2026-10-16T10:00:00+00:00"


def make_log(n, ratio=1.0, hits80=None, hits95=None, start="2026-10-15", commit="aaa", version="LIVE_V1", seed=7):
    """ratio: model hatasi / RW hatasi. hits: 1 = gerceklesen aralik DISINDA."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n + 1)
    h80 = np.zeros(n) if hits80 is None else np.asarray(hits80)
    h95 = np.zeros(n) if hits95 is None else np.asarray(hits95)
    rows = []
    for i in range(n):
        p = 6500.0 + i
        real = p * math.exp(rng.normal(0, 0.01))
        point = p + (real - p) * (1 - ratio)
        row = {"issued_at_utc": f"{days[i].date()}T19:00:00+00:00", "system_version": version,
               "asof_date": str(days[i].date()), "asof_bar_end_utc": f"{days[i].date()} 18:00:00+00:00",
               "price_basis": lf.PRICE_BASIS, "price_now": p, "point_price": point, "point_return": 0.0001,
               "point_return_se": 0.0004, "point_price_rw": p, "q80_low": p - 50, "q80_high": p + 50,
               "q95_low": p - 90, "q95_high": p + 90, "aci_k80": 1.0, "aci_k95": 1.0, "gvz_lag1": 18.0,
               "train_days": 480, "code_commit": commit, "sync_md5": "x"}
        row["row_hash"] = lf.row_hash(pd.Series(row))
        row.update({"realized_date": str(days[i + 1].date()), "realized_price": real,
                    "abs_err_TL": abs(real - point), "rw_abs_err_TL": abs(real - p),
                    "in80": float(1 - h80[i]), "in95": float(1 - h95[i])})
        rows.append(row)
    return pd.DataFrame(rows)[lf.ALL_COLS]


def iid_hits(n, rate, seed):
    return (np.random.default_rng(seed).random(n) < rate).astype(int)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="le_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for f in le.FORECAST_FILES + le.EVAL_FILES:
            (self.tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / f, self.tmp / f)
        p = mock.patch.multiple(le, ROOT=self.tmp, REGISTRY_PATH=self.tmp / "research" / "registry.jsonl",
                                PREREG_PATH=self.tmp / "research" / "preregistration" / "live_v1.json",
                                RESULT_PATH=self.tmp / "research" / "live" / "live_v1_result.json")
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(lf, "LOG_PATH", self.tmp / "research" / "live" / "forecast_log.csv")
        q.start()
        self.addCleanup(q.stop)

    def prereg(self, created=CREATED):
        le.prepare()
        d = json.loads(le.PREREG_PATH.read_text(encoding="utf-8"))
        d["created_at_utc"] = created
        return d

    def md5_fn(self, prereg):
        good = prereg["forecast_code_md5"]
        return lambda c: good if c == "aaa" else {k: "farkli" for k in good}


class TestRollWindow(unittest.TestCase):
    def test_known_roll_days_and_boundaries(self):
        inside = ["2025-07-30", "2025-08-01", "2026-07-29", "2026-07-31", "2026-11-24", "2026-11-30",
                  "2026-12-01", "2026-12-02", "2027-01-25", "2027-02-01", "2027-02-02", "2027-03-31", "2026-08-03", "2026-08-04"]
        outside = ["2026-07-15", "2026-07-23", "2026-08-05", "2026-10-20", "2026-11-23", "2026-12-03",
                   "2027-01-04", "2027-02-03", "2026-10-30", "2026-06-30"]
        for d in inside:
            with self.subTest(d=d):
                self.assertTrue(le.in_roll_window(d))
        for d in outside:
            with self.subTest(d=d):
                self.assertFalse(le.in_roll_window(d))

    def test_realized_bar_is_14_ny_with_dst(self):
        self.assertEqual(le.realized_bar_utc("2026-10-16"), pd.Timestamp("2026-10-16 18:00", tz="UTC"))
        self.assertEqual(le.realized_bar_utc("2026-11-02"), pd.Timestamp("2026-11-02 19:00", tz="UTC"))


class TestPrepare(Base):
    def test_creates_once_and_logs(self):
        self.assertEqual(le.prepare(), "olusturuldu")
        self.assertEqual(le.prepare(), "zaten hazir")
        d = json.loads(le.PREREG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(d["parameters"]["eval_n"], 120)
        self.assertEqual(set(d["forecast_code_md5"]), set(le.FORECAST_FILES))
        self.assertEqual(set(d["eval_code_md5"]), set(le.EVAL_FILES))
        ev = [json.loads(l) for l in le.REGISTRY_PATH.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(e["protocol_id"], e["event"]) for e in ev], [("LIVE_V1", "prepared")])

    def test_line_endings_do_not_matter_but_content_does(self):
        le.prepare()
        f = self.tmp / "src" / "models" / "adaptive.py"
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(le.prepare(), "zaten hazir")
        f.write_bytes(f.read_bytes() + b"# degisti\r\n")
        with self.assertRaisesRegex(le.Refused, "degistirilemez"):
            le.prepare()

    def test_md5_matches_committed_lf_bytes(self):
        data = (PROJECT_ROOT / "src" / "live" / "forecast.py").read_bytes()
        self.assertEqual(le.norm_md5(data), hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest())


class TestEligible(Base):
    def test_reasons_and_order(self):
        pr = self.prereg()
        good = make_log(5)
        bad_ver = make_log(1, version="LIVE_V0", seed=1)
        early = make_log(1, start="2026-10-01", seed=2)
        unscored = make_log(1, start="2026-11-02", seed=3)
        unscored.loc[:, "realized_price"] = np.nan
        wrong = make_log(2, start="2026-12-07", commit="bbb", seed=4)
        nocommit = make_log(1, start="2026-12-14", commit="", seed=5)
        log = pd.concat([good.iloc[::-1], bad_ver, early, unscored, wrong, nocommit], ignore_index=True)
        s, r = le.eligible(log, pr, self.md5_fn(pr))
        self.assertEqual(list(s["asof_date"]), list(good["asof_date"]))
        self.assertEqual(r, {"surum": 1, "erken_tarih": 1, "puanlanmamis": 1, "sonuc_on_kayittan_once": 0,
                             "kod_farkli": 3})

    def test_outcome_known_before_prereg_is_excluded(self):
        pr = self.prereg(created="2026-10-16T18:30:00+00:00")      # 15.10 tahmininin sonucu 18:00 UTC'de belli
        s, r = le.eligible(make_log(3), pr, self.md5_fn(pr))
        self.assertEqual(r["sonuc_on_kayittan_once"], 1)
        self.assertEqual(list(s["asof_date"]), ["2026-10-16", "2026-10-19"])


class TestEvaluate(Base):
    def run_eval(self, log, pr=None):
        pr = pr or self.prereg()
        return le.evaluate(log, pr, self.md5_fn(pr))

    def test_needs_eval_n(self):
        with self.assertRaisesRegex(le.Refused, "120 uygun"):
            self.run_eval(make_log(119))

    def test_point_verdicts(self):
        h80, h95 = iid_hits(120, 0.2, 1), iid_hits(120, 0.05, 2)
        cases = {0.6: "DAHA_IYI", 1.0: "PRATIKTE_ESDEGER", 1.5: "DAHA_KOTU"}
        for ratio, want in cases.items():
            with self.subTest(ratio=ratio):
                r = self.run_eval(make_log(120, ratio=ratio, hits80=h80, hits95=h95))
                self.assertEqual(r["primary"]["point_verdict"], want)
                self.assertAlmostEqual(r["primary"]["bootstrap"]["pct"], 100 * (ratio - 1), places=6)

    def test_point_verdict_rules(self):
        def v(diff_ci, pct_ci, p):
            return le.point_verdict({"diff_ci95": diff_ci, "pct_ci95": pct_ci}, {"p_value": p}, 0.05, 2.0)
        self.assertEqual(v([-2.0, -0.5], [-9.0, -1.0], 0.01), "DAHA_IYI")
        self.assertEqual(v([-2.0, -0.5], [-9.0, -1.0], 0.20), "AYIRT_EDILEMEDI")      # GA tamam, p degil
        self.assertEqual(v([0.5, 2.0], [1.0, 9.0], 0.01), "DAHA_KOTU")
        self.assertEqual(v([-0.1, 0.1], [-1.9, 1.9], 0.60), "PRATIKTE_ESDEGER")
        self.assertEqual(v([-0.1, 0.3], [-1.0, 3.5], 0.40), "AYIRT_EDILEMEDI")      # ust sinir marji asiyor
        self.assertEqual(v([-0.3, 0.1], [-3.5, 1.0], 0.40), "AYIRT_EDILEMEDI")      # alt sinir marji asiyor

    def test_interval_criteria(self):
        n = 120
        ok = self.run_eval(make_log(n, hits80=iid_hits(n, 0.2, 3), hits95=iid_hits(n, 0.05, 4)))["primary"]
        self.assertTrue(ok["A1_kupiec"] and ok["A2_christoffersen_80"])
        low = self.run_eval(make_log(n, hits80=iid_hits(n, 0.4, 5), hits95=iid_hits(n, 0.05, 6)))["primary"]
        self.assertFalse(low["A1_kupiec"])
        clustered = np.zeros(n, int)
        for start in range(0, n, 20):
            clustered[start:start + 4] = 1                               # %20 ihlal, ama kumelenmis
        cl = self.run_eval(make_log(n, hits80=clustered, hits95=iid_hits(n, 0.05, 7)))["primary"]
        self.assertAlmostEqual(cl["cov80"], 0.8)
        self.assertTrue(cl["A1_kupiec"])
        self.assertFalse(cl["A2_christoffersen_80"])

    def test_uses_first_n_and_secondary_roll(self):
        log = make_log(130, hits80=iid_hits(130, 0.2, 8), hits95=iid_hits(130, 0.05, 9))
        r = self.run_eval(log.iloc[::-1].reset_index(drop=True))
        self.assertEqual((r["n"], r["first_asof"], r["last_asof"]),
                         (120, log["asof_date"].iat[0], log["asof_date"].iat[119]))
        s = log.head(120)
        flag = s["asof_date"].map(le.in_roll_window) | s["realized_date"].map(le.in_roll_window)
        self.assertGreater(int(flag.sum()), 0)
        self.assertEqual(r["secondary_roll_excluded"]["dropped"], int(flag.sum()))
        self.assertEqual(r["secondary_roll_excluded"]["n"], 120 - int(flag.sum()))

    def test_refuses_changed_eval_code_or_tampered_log(self):
        pr = self.prereg()
        log = make_log(120)
        (self.tmp / "src" / "live" / "forecast_eval.py").write_text("# degisti\n", encoding="utf-8")
        with self.assertRaisesRegex(le.Refused, "Degerlendirme kodu"):
            le.evaluate(log, pr, self.md5_fn(pr))
        shutil.copyfile(PROJECT_ROOT / "src" / "live" / "forecast_eval.py", self.tmp / "src" / "live" / "forecast_eval.py")
        log.loc[3, "point_price"] = 1.0
        with self.assertRaises(lf.LogTampered):
            le.evaluate(log, pr, self.md5_fn(pr))


class TestMain(Base):
    def call(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = le.main(list(args))
        return code, buf.getvalue()

    def test_status_and_one_shot_evaluate(self):
        code, out = self.call("status")
        self.assertEqual(code, 3)
        self.assertIn("On kayit yok", out)
        self.assertEqual(self.call("prepare")[0], 0)
        pr = json.loads(le.PREREG_PATH.read_text(encoding="utf-8"))
        pr["created_at_utc"] = CREATED
        le.PREREG_PATH.write_text(json.dumps(pr, indent=2), encoding="utf-8")
        log = make_log(120, hits80=iid_hits(120, 0.2, 10), hits95=iid_hits(120, 0.05, 11))
        lf.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(lf.LOG_PATH, index=False)
        with mock.patch.object(le, "commit_md5s", self.md5_fn(pr)):
            code, out = self.call("status")
            self.assertEqual(code, 0)
            self.assertIn("uygun puanlanmis: 120 / 120", out)
            self.assertNotIn("MAE", out)                                   # ara basari gosterilmez
            self.assertEqual(self.call("evaluate")[0], 0)
            self.assertTrue(le.RESULT_PATH.exists())
            code, out = self.call("evaluate")
            self.assertEqual(code, 2)
            self.assertIn("zaten yapildi", out)
        ev = [json.loads(l)["event"] for l in le.REGISTRY_PATH.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(ev, ["prepared", "evaluated"])

    def test_defaults_read_at_runtime(self):
        self.assertEqual(le.prepare(), "olusturuldu")
        self.assertTrue(str(le.PREREG_PATH).startswith(str(self.tmp)))
        self.assertTrue((self.tmp / "research" / "preregistration" / "live_v1.json").exists())


_REAL = [PROJECT_ROOT / "research", PROJECT_ROOT / "data" / "processed"]
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
