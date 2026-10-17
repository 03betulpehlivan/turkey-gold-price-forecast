"""
Veritabani (src/db/store.py) ve REST API (src/api/app.py) testleri
Calistirma: python -m unittest tests.test_db_api -v
Sentetik kayitlar (gercek ozet fonksiyonlariyla), gecici SQLite. Gercek data/ ve research/ kullanilmaz.
SQL Server gerektirmez; SQL Server'a ozgu adim (veritabani olusturma) sahte motorla sinanir.
"""

from pathlib import Path
from unittest import mock
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message=".*httpx2.*")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app as api  # noqa: E402
from src.db import store  # noqa: E402
from src.live import forecast as lf  # noqa: E402
from src.live import forecast_v2 as v2  # noqa: E402
from src.live import physical_forecast as lp  # noqa: E402
from src.live import physical_intervals as pi  # noqa: E402

DAYS = ["2026-10-15", "2026-10-16", "2026-10-19"]


def _row(cols, text, hash_fn, values):
    r = {c: ("x" if c in text else 1.0) for c in cols}
    r.update(values)
    r["row_hash"] = hash_fn(pd.Series(r))
    return r


def make_root(root: Path):
    live = root / "research" / "live"
    live.mkdir(parents=True)
    phys, ivl = [], []
    for i, d in enumerate(DAYS):
        preds = {"RW": 6500.0 + i, "Bosluk": 6510.0 + i, "Bosluk-Kayan": 6511.0 + i, "Prim-Capa": 6490.0 + i}
        r = _row(lp.FORECAST_COLS, lp.TEXT, lp.row_hash,
                 {"target_date": d, "issued_at_utc": f"{d}T09:55:00+00:00", **{f"pred_{m}": v for m, v in preds.items()}})
        q = _row(pi.FORECAST_COLS, pi.TEXT, pi.row_hash,
                 {"target_date": d, **{f"pred_{m}": v for m, v in preds.items()},
                  **{f"{b}{lvl}_{m}": v + s * w for m, v in preds.items() for lvl, w in ((80, 60), (95, 100))
                     for b, s in (("lo", -1), ("hi", 1))}})
        if i < 2:                                                                # son gun puanlanmamis
            real = 6520.0 + i
            r.update(realized_price=real, **{f"abs_err_{m}": abs(real - v) for m, v in preds.items()})
            q.update(realized_price=real, **{f"in{lvl}_{m}": 1.0 for m in preds for lvl in (80, 95)})
        phys.append(r)
        ivl.append(q)
    pd.DataFrame(phys).reindex(columns=lp.ALL_COLS).to_csv(live / "physical_log.csv", index=False)
    pd.DataFrame(ivl).reindex(columns=pi.ALL_COLS).to_csv(live / "physical_intervals_log.csv", index=False)
    for mod, ver, name in ((lf, "LIVE_V1", "forecast_log.csv"), (v2, "LIVE_V2", "forecast_v2_log.csv")):
        rows = []
        for i, d in enumerate(DAYS[:2]):
            r = _row(mod.FORECAST_COLS, mod.TEXT_COLS, mod.row_hash,
                     {"system_version": ver, "asof_date": d, "issued_at_utc": f"{d}T19:30:00+00:00",
                      "price_now": 6600.0, "point_price": 6605.0, "point_price_rw": 6600.0,
                      "q80_low": 6500.0, "q80_high": 6700.0, "q95_low": 6450.0, "q95_high": 6750.0})
            if i == 0:
                r.update(realized_date=DAYS[1], realized_price=6610.0, abs_err_TL=5.0, rw_abs_err_TL=10.0, in80=1.0, in95=1.0)
            rows.append(r)
        pd.DataFrame(rows).reindex(columns=mod.ALL_COLS).to_csv(live / name, index=False)
    (root / "research" / "registry.jsonl").write_text(
        json.dumps({"protocol_id": "LIVE_V1", "event": "prepared", "at_utc": "2026-10-15T19:53:47+00:00", "prereg_md5": "a"}) + "\n",
        encoding="utf-8")
    ex = root / "research" / "back_exam_ons_v1"
    ex.mkdir(parents=True)
    (ex / "result.json").write_text('{"hypothesis_supported": false, "p": NaN}', encoding="utf-8")
    pp = root / "data" / "processed" / "live"
    pp.mkdir(parents=True)
    pd.DataFrame({"Date": ["2026-10-14", "2026-10-15", "2026-10-16"], "kapanis_tl_gr": [6500.0, 0.0, 6520.0],
                  "agort_tl_gr": 1.0, "kapanis_usd_ons": 1.0, "kaynak": "BIST"}).to_csv(pp / "physical_gold_daily.csv", index=False)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="db_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        make_root(self.tmp)
        self.url = f"sqlite:///{(self.tmp / 'test.sqlite').as_posix()}"
        p = mock.patch.dict(os.environ, {"ALTIN_DB_URL": self.url})
        p.start()
        self.addCleanup(p.stop)

    def load(self):
        return store.load_all(root=self.tmp)


class TestLoad(Base):
    def test_counts_and_mapping(self):
        r = self.load()
        self.assertEqual(r["counts"], {"forecast": 12 + 4 + 4, "protocol_event": 1, "exam_result": 1, "price": 2})
        self.assertEqual(r["missing_price_series"], ["teorik_v1", "teorik_v2"])
        eng = store.get_engine()
        with eng.connect() as c:
            fz = store.q_forecasts(c, "LIVE_PHYS_V1", "Bosluk")
            self.assertEqual([f["asof_date"] for f in fz], list(reversed(DAYS)))
            first = fz[-1]
            self.assertEqual((first["point"], first["lo80"], first["hi95"], first["realized"]), (6510.0, 6450.0, 6610.0, 6520.0))
            self.assertEqual((first["abs_err"], first["in80"]), (10.0, 1.0))
            self.assertIsNone(fz[0]["realized"])
            th = store.q_forecasts(c, "LIVE_V2", "RW")
            self.assertEqual(th[-1]["abs_err"], 10.0)
            self.assertIsNone(th[-1]["lo80"])
            self.assertEqual(th[-1]["target_date"], DAYS[1])
            self.assertEqual(store.q_exams(c)[0]["result"], {"hypothesis_supported": False, "p": None})
            self.assertEqual([p["value"] for p in store.q_prices(c, "fiziki_kapanis")], [6500.0, 6520.0])   # 0 atilir
        eng.dispose()

    def test_idempotent_and_audited(self):
        a = self.load()
        b = self.load()
        self.assertEqual(a, b)
        eng = store.get_engine()
        with eng.connect() as c:
            self.assertEqual(len(store.q_forecasts(c)), 20)
            last = store.q_last_load(c)
            self.assertEqual(last["id"], 2)
            self.assertEqual(last["counts"]["forecast"], 20)
        eng.dispose()

    def test_tampered_log_writes_nothing(self):
        self.load()
        cases = {"forecast_v2_log.csv": "point_price", "forecast_log.csv": "q80_low",
                 "physical_log.csv": "pred_Bosluk", "physical_intervals_log.csv": "lo80_RW"}
        for name, col in cases.items():
            with self.subTest(name):
                p = self.tmp / "research" / "live" / name
                orig = p.read_bytes()
                log = pd.read_csv(p)
                log.loc[0, col] = 1.0
                log.to_csv(p, index=False)
                with self.assertRaisesRegex(store.StoreError, "butunlugu"):
                    self.load()
                p.write_bytes(orig)
        eng = store.get_engine()
        with eng.connect() as c:
            self.assertEqual(len(store.q_forecasts(c)), 20)                     # eski ayna yerinde
            self.assertEqual(store.q_last_load(c)["id"], 1)
        eng.dispose()

    def test_scorecard_and_latest(self):
        self.load()
        eng = store.get_engine()
        with eng.connect() as c:
            sc = store.q_scorecard(c, "LIVE_PHYS_V1")
            m = {x["model"]: x for x in sc["models"]}
            self.assertEqual(sc["issued_days"], 3)
            self.assertEqual(m["RW"]["n"], 2)
            self.assertAlmostEqual(m["Bosluk"]["vs_rw_pct"], 100 * (1 - m["Bosluk"]["mae"] / m["RW"]["mae"]))
            self.assertIsNone(m["RW"]["vs_rw_pct"])
            self.assertIn("KULLANILMAZ", sc["note"])
            lt = store.q_latest(c, "LIVE_PHYS_V1")
            self.assertEqual({x["asof_date"] for x in lt}, {DAYS[-1]})
            self.assertEqual(len(lt), 4)
        eng.dispose()


class TestPerformance(Base):
    def test_cumulative_difference_vs_rw(self):
        self.load()
        eng = store.get_engine()
        with eng.connect() as c:
            p = store.q_performance(c, "LIVE_PHYS_V1")
            self.assertEqual((p["n"], p["dates"]), (2, DAYS[:2]))                 # 3. gun puanlanmamis
            self.assertEqual(set(p["series"]), {"Bosluk", "Bosluk-Kayan", "Prim-Capa"})
            self.assertEqual(p["series"]["Bosluk"]["daily"], [-10.0, -10.0])
            self.assertEqual(p["series"]["Bosluk"]["cumulative"], [-10.0, -20.0])
            self.assertEqual(p["series"]["Prim-Capa"]["cumulative"], [10.0, 20.0])
            t = store.q_performance(c, "LIVE_V2")
            self.assertEqual((t["n"], t["series"]["Mean-Return"]["cumulative"]), (1, [-5.0]))
            self.assertIn("Betimleyici", t["note"])
        eng.dispose()

    def test_empty_and_api(self):
        eng = store.get_engine()
        store.META.create_all(eng)
        with eng.connect() as c:
            self.assertEqual(store.q_performance(c, "LIVE_V1")["n"], 0)
        eng.dispose()
        self.load()
        cl = TestClient(api.app, base_url="http://127.0.0.1")
        self.assertEqual(cl.get("/api/performance", params={"protocol": "LIVE_PHYS_V1"}).json()["n"], 2)
        self.assertEqual(cl.get("/api/performance", params={"protocol": "XYZ"}).status_code, 404)

    def test_dashboard_has_performance_and_turkish_messages(self):
        html = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "src" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="performans"', html)
        self.assertIn("/api/performance?protocol=", js)
        self.assertIn("turkce(c.detail)", js)
        self.assertNotIn("innerHTML", js)


class TestConfig(unittest.TestCase):
    def test_url_read_at_runtime_and_redacted(self):
        with mock.patch.dict(os.environ, {"ALTIN_DB_URL": "mssql+pyodbc://u:gizli@srv/AltinTahmin?driver=X"}):
            self.assertIn("mssql", store.db_url())
            self.assertNotIn("gizli", store.redact(store.db_url()))
        with mock.patch.dict(os.environ, {"ALTIN_DB_URL": ""}):
            self.assertTrue(store.db_url().endswith("data/processed/db/altin.sqlite"))

    def test_mssql_init_creates_database_via_master(self):
        calls = []

        class Conn:
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def execute(self, stmt, params=None):
                calls.append(str(stmt))

                class R:
                    def scalar(self_inner): return None
                return R()

        class Eng:
            def connect(self): return Conn()
            def dispose(self): pass
        seen = {}

        def fake_create(url, **kw):
            seen.setdefault("urls", []).append(str(url))
            seen.setdefault("kw", []).append(kw)
            return Eng()
        with mock.patch.object(store.sa, "create_engine", fake_create), mock.patch.object(store.META, "create_all"):
            store.init_db("mssql+pyodbc://@(localdb)\\MSSQLLocalDB/AltinTahmin?driver=ODBC+Driver+17+for+SQL+Server")
        self.assertIn("master", seen["urls"][0])
        self.assertEqual(seen["kw"][0], {"isolation_level": "AUTOCOMMIT"})
        self.assertTrue(any("CREATE DATABASE [AltinTahmin]" in s for s in calls))
        self.assertEqual(seen["kw"][1].get("fast_executemany"), True)


class TestApi(Base):
    def setUp(self):
        super().setUp()
        self.c = TestClient(api.app, base_url="http://127.0.0.1")

    def test_empty_database(self):
        r = self.c.get("/api/health").json()
        self.assertIn(r["status"], ("veritabani_yok", "bos"))

    def test_endpoints(self):
        self.load()
        self.assertEqual(self.c.get("/api/health").json()["status"], "ok")
        lt = self.c.get("/api/forecasts/latest", params={"protocol": "LIVE_V1"}).json()
        self.assertEqual(lt["asof_date"], DAYS[1])
        self.assertEqual({f["model"] for f in lt["forecasts"]}, {"Mean-Return", "RW"})
        f = self.c.get("/api/forecasts", params={"protocol": "LIVE_PHYS_V1", "start": DAYS[1]}).json()["forecasts"]
        self.assertEqual({x["asof_date"] for x in f}, {DAYS[1], DAYS[2]})
        self.assertEqual(self.c.get("/api/prices", params={"series": "fiziki_kapanis"}).json()["prices"][0]["value"], 6500.0)
        self.assertEqual(self.c.get("/api/exams").json()["exams"][0]["protocol_id"], "BACK_EXAM_ONS_V1")
        self.assertEqual(self.c.get("/api/scorecard", params={"protocol": "LIVE_V2"}).json()["issued_days"], 2)
        self.assertEqual(len(self.c.get("/api/protocols").json()["events"]), 1)

    def test_validation_and_read_only(self):
        self.load()
        self.assertEqual(self.c.get("/api/forecasts", params={"protocol": "XYZ"}).status_code, 404)
        self.assertEqual(self.c.get("/api/prices", params={"series": "baska"}).status_code, 404)
        self.assertEqual(self.c.get("/api/forecasts", params={"limit": 0}).status_code, 422)
        self.assertEqual(self.c.get("/api/forecasts/latest").status_code, 422)
        methods = {m for r in api.app.routes for m in getattr(r, "methods", set())}
        self.assertEqual(methods - {"GET", "HEAD"}, set())
        self.assertIn("yatirim tavsiyesi degildir", api.app.description)

    def test_foreign_host_rejected(self):
        evil = TestClient(api.app, base_url="http://evil.example")             # DNS rebinding denemesi
        self.assertEqual(evil.get("/api/health").status_code, 400)
        self.assertEqual(TestClient(api.app, base_url="http://localhost").get("/api/health").status_code, 200)
        with mock.patch.dict(os.environ, {"ALTIN_ALLOWED_HOSTS": "a.local, b.local"}):
            self.assertEqual(api.allowed_hosts(), ["a.local", "b.local"])

    def test_security_headers_and_csp(self):
        self.load()
        for path in ("/", "/api/health", "/static/app.js"):
            h = self.c.get(path).headers
            with self.subTest(path):
                self.assertEqual(h["x-content-type-options"], "nosniff")
                self.assertEqual(h["x-frame-options"], "DENY")
                self.assertEqual(h["referrer-policy"], "no-referrer")
                self.assertIn("script-src 'self';", h["content-security-policy"])
                self.assertNotIn("unsafe-inline", h["content-security-policy"])
                self.assertIn("frame-ancestors 'none'", h["content-security-policy"])
        self.assertNotIn("content-security-policy", self.c.get("/docs").headers)    # Swagger harici dosya kullanir

    def test_static_files_and_no_inline_code(self):
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/app.css").status_code, 200)
        for bad in ("/static/../db/store.py", "/static/%2e%2e/db/store.py", "/static/missing.js"):
            self.assertEqual(self.c.get(bad).status_code, 404, bad)
        html = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html))       # satir ici betik yok
        self.assertNotIn("<style", html)
        self.assertNotIn(" style=", html)
        self.assertIsNone(re.search(r"\son[a-z]+=", html))                        # satir ici olay isleyici yok
        js = (PROJECT_ROOT / "src" / "web" / "app.js").read_text(encoding="utf-8")
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function", "document.write"):
            self.assertNotIn(bad, js)

    def test_dashboard_served(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("/static/app.js", r.text)
        self.assertIn("yat\u0131r\u0131m tavsiyesi de\u011fildir", r.text)
        self.assertIn("/api/forecasts/latest", self.c.get("/static/app.js").text)


class TestHardening(unittest.TestCase):
    def test_invalid_database_name_refused(self):
        for bad in ("Altin]; DROP DATABASE x--", "1abc", "a b"):
            with self.assertRaises(store.StoreError):
                store.init_db(f"mssql+pyodbc://@srv/{bad}?driver=X")

    def test_launcher_binds_localhost_only(self):
        ps = (PROJECT_ROOT / "scripts" / "pano.ps1").read_text(encoding="utf-8")
        ps.encode("ascii")
        self.assertIn("--host 127.0.0.1", ps)
        self.assertIn("--no-server-header", ps)
        self.assertNotIn("0.0.0.0", ps)


class TestScheduledLoad(unittest.TestCase):
    """PowerShell CALISTIRILMAZ; betik metni sinanir."""

    def test_wrapper_loads_db_without_touching_exit_code(self):
        w = (PROJECT_ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")
        i = w.index("-m src.db.store load")
        self.assertLess(w.index("$kod = $LASTEXITCODE"), i)
        self.assertLess(i, w.index("yedekle.ps1"))                            # yedek, veritabanini da kapsar
        block = w[w.rindex("try {", 0, i):w.index("# Veri yedegi")]
        self.assertNotIn("$kod", block)
        self.assertNotIn("exit", block)
        self.assertIn("catch {", block)


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
