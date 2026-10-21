"""
Ileriye donuk karne testleri: src/tool/canli_karne.py, /api/tahmin/canli-karne, sabah gorevi entegrasyonu
Calistirma: python -m unittest tests.test_tool_canli_karne -v
Gecici klasorler ve sentetik fiyatlar; gercek research/ ve data/ kullanilmaz (md5 korumasi).
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import hashlib
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
from src.tool import canli_karne as ck  # noqa: E402
from src.tool import service as ts  # noqa: E402


def ufuk(bilgi="2026-10-20", P=6560.0):
    return {"bilgi_gunu": bilgi, "son_kapanis": P,
            "ufuklar": [{"ufuk_gun": h, "model": "ARIMA-GARCH", "kullanilan": "ARIMA-GARCH",
                         "aralik80": [P * (1 - 0.01 * h), P * (1 + 0.01 * h)], "aralik95": [P * (1 - 0.02 * h), P * (1 + 0.02 * h)]}
                        for h in (1, 3, 5)]}


SABAH = datetime(2026, 10, 21, 9, 45, tzinfo=timezone.utc)                     # 12:45 Istanbul, ertesi is gunu


class TestKaydet(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ck_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.p = self.tmp / "log.csv"

    def test_records_once_before_deadline(self):
        r = ck.kaydet(ufuk(), now=SABAH, path=self.p)
        self.assertEqual(r["eklenen"], 3)
        once = self.p.read_bytes()
        self.assertEqual(ck.kaydet(ufuk(), now=SABAH, path=self.p)["eklenen"], 0)        # ayni gun ikinci kez yok
        self.assertEqual(self.p.read_bytes(), once)
        log = ck.oku(self.p)
        self.assertEqual(list(log["ufuk_gun"]), ["1", "3", "5"])
        self.assertTrue(ck.dogrula(log).all())
        ck.kaydet(ufuk("2026-10-21"), now=datetime(2026, 10, 22, 9, 0, tzinfo=timezone.utc), path=self.p)
        self.assertTrue(self.p.read_bytes().startswith(once))                       # yalnizca sonuna eklenir
        self.assertNotIn(b"\r\n", self.p.read_bytes())

    def test_late_and_weekend_refused(self):
        gec = datetime(2026, 10, 21, 11, 0, tzinfo=timezone.utc)                    # 14:00 Istanbul
        r = ck.kaydet(ufuk(), now=gec, path=self.p)
        self.assertEqual(r["eklenen"], 0)
        self.assertIn("gec", r["neden"])
        self.assertFalse(self.p.exists())
        cts = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)                     # cumartesi, cuma kapanisi
        self.assertEqual(ck.kaydet(ufuk("2026-10-23"), now=cts, path=self.p)["neden"], "hafta sonu")
        self.assertEqual(ck.son_zaman("2026-10-23"), pd.Timestamp("2026-10-26 14:00", tz="Europe/Istanbul"))

    def test_tampered_row_excluded(self):
        ck.kaydet(ufuk(), now=SABAH, path=self.p)
        txt = self.p.read_text(encoding="utf-8").replace("6494.4000", "6000.0000", 1)
        self.p.write_text(txt, encoding="utf-8")
        log = ck.oku(self.p)
        self.assertEqual(int((~ck.dogrula(log)).sum()), 1)
        d = pd.bdate_range("2026-10-01", "2026-10-30")
        c = ck.karne(pd.Series(6560.0, index=d), self.p)
        self.assertEqual((c["kayit"], c["bozuk"]), (2, 1))


class TestKarne(unittest.TestCase):
    def test_scoring_uses_h_trading_days_later(self):
        tmp = Path(tempfile.mkdtemp(prefix="ck2_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = tmp / "log.csv"
        ck.kaydet(ufuk(), now=SABAH, path=p)
        d = pd.bdate_range("2026-10-01", "2026-10-23")                               # 3 gun sonrasi var, 5 gun yok
        prices = pd.Series(6560.0, index=d)
        prices[pd.Timestamp("2026-10-21")] = 6700.0                                # 1 gun: %80 disi (+%2,1)
        prices[pd.Timestamp("2026-10-23")] = 6600.0                                # 3 gun: icinde
        c = ck.karne(prices, p)
        u = {x["ufuk_gun"]: x for x in c["ufuklar"]}
        self.assertEqual((u[1]["puanlanan"], u[1]["tuttu80"], u[1]["tuttu95"]), (1, 0, 0))
        self.assertEqual((u[3]["puanlanan"], u[3]["tuttu80"]), (1, 1))
        self.assertEqual((u[5]["puanlanan"], u[5]["kayit"]), (0, 1))
        son = {r["ufuk_gun"]: r for r in c["son_kayitlar"]}
        self.assertEqual(son[3]["hedef_gun"], "2026-10-23")
        self.assertIsNone(son[5]["gercek"])

    def test_empty(self):
        c = ck.karne(pd.Series(dtype=float), Path(tempfile.gettempdir()) / "yok_ck.csv")
        self.assertEqual(c["kayit"], 0)


class TestIntegration(unittest.TestCase):
    def test_api(self):
        tmp = Path(tempfile.mkdtemp(prefix="ck3_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        log, csvp = tmp / "log.csv", tmp / "p.csv"
        ck.kaydet(ufuk(), now=SABAH, path=log)
        d = pd.bdate_range("2023-01-02", "2026-10-28")
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "kapanis_tl_gr": np.linspace(2000, 6600, len(d))}).to_csv(csvp, index=False)
        ts._canli_karne.cache_clear()
        with mock.patch.object(ck, "LOG", log), mock.patch.object(ts.k, "PATHS", [csvp]):
            r = TestClient(api.app, base_url="http://127.0.0.1").get("/api/tahmin/canli-karne")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kayit"], 3)

    def test_morning_task_and_page(self):
        w = (PROJECT_ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")
        i = w.index("src.tool.canli_karne kaydet")
        self.assertLess(i, w.index("src.db.store load"))
        blok = w[w.rindex('if ($Gorev -eq "sabah")', 0, i): w.index("src.db.store load")]
        self.assertIn('git commit -q -m "Ileriye donuk karne: 1/3/5 gun araliklari" -- $ck', blok)
        self.assertIn("catch {", blok)
        self.assertNotRegex(blok, r"git add (-A|\.|--all)")
        js = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
        self.assertIn("/api/tahmin/canli-karne", js)
        self.assertNotIn("innerHTML", js)


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
