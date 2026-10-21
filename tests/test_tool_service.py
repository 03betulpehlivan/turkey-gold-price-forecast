"""
Pano tahmin servisi ve ana ekran testleri: src/tool/service.py, /api/tahmin/*, src/web/index.html + ana.js
Calistirma: python -m unittest tests.test_tool_service -v
Sentetik fiyat dosyasi gecici klasorde; kayitli karne dosyalari yalnizca OKUNUR.
"""

from pathlib import Path
from unittest import mock
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
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
from src.tool import karne as k  # noqa: E402
from src.tool import service as ts  # noqa: E402


def write_prices(path, end="2026-10-21", seed=7):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2010-01-04", end)
    r = 0.0008 + 0.01 * rng.standard_t(6, len(d)) * np.sqrt(4 / 6)
    pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "kapanis_tl_gr": np.round(100 * np.exp(np.cumsum(r)), 4)}).to_csv(path, index=False)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ts_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.csv = self.tmp / "physical_gold_daily.csv"
        write_prices(self.csv)
        p = mock.patch.object(k, "PATHS", [self.csv])
        p.start()
        self.addCleanup(p.stop)
        ts._kart.cache_clear()
        ts._bu_yil.cache_clear()


class TestService(Base):
    def test_window_from_committed_choice_and_fallback(self):
        self.assertEqual(ts.secilen_pencere(), 750)                               # research/tool/pencere_*.json
        self.assertEqual(ts.secilen_pencere(self.tmp), ts.VARSAYILAN_PENCERE)
        (self.tmp / "pencere_2012-2017_2018-2022.json").write_text(json.dumps({"secilen_pencere": 500}), encoding="utf-8")
        self.assertEqual(ts.secilen_pencere(self.tmp), 500)

    def test_card(self):
        c = ts.kart(pencere=250)
        self.assertEqual(c["bilgi_gunu"], "2026-10-21")
        self.assertEqual(c["model"], k.ODAK)
        self.assertEqual(c["egitim"], [2012, 2025])
        self.assertLess(c["aralik95"][0], c["aralik80"][0])
        self.assertLess(c["aralik80"][0], c["son_kapanis"])
        self.assertLess(c["son_kapanis"], c["aralik80"][1])
        self.assertTrue(0 <= c["hareketli_olasiligi_pct"] <= 100)

    def test_cache_and_refresh_on_new_data(self):
        a = ts.kart(pencere=250)
        t0 = time.time()
        self.assertEqual(ts.kart(pencere=250), a)
        self.assertLess(time.time() - t0, 0.05)                                   # onbellekten
        write_prices(self.csv, end="2026-10-22")
        os.utime(self.csv, (time.time() + 5, time.time() + 5))
        self.assertEqual(ts.kart(pencere=250)["bilgi_gunu"], "2026-10-22")          # yeni veri, yeni hesap

    def test_this_year_matches_engine(self):
        d = ts.bu_yil(pencere=250)
        self.assertEqual(d["yil"], 2026)
        g = pd.DataFrame(d["gunler"])
        self.assertTrue(g["tarih"].str.startswith("2026").all())
        self.assertEqual(d["gun"], len(g))
        in80 = ((g["gercek"] >= g["lo80"]) & (g["gercek"] <= g["hi80"])).sum()
        self.assertEqual(d["aralik80_tuttu"], int(in80))
        ref = k.gunluk(k.load_series(self.csv), (2012, 2025), (2026, 2026), k.ODAK, 250)
        np.testing.assert_allclose(g["lo80"].to_numpy(), ref["lo80"].to_numpy())

    def test_scorecard_from_committed_file(self):
        c = ts.karne()
        self.assertEqual((c["model"], c["pencere"], c["test"]), (k.ODAK, 750, [2018, 2022]))
        src = json.loads((PROJECT_ROOT / c["kaynak"]).read_text(encoding="utf-8"))
        self.assertEqual(c["karsilastirma"][-1]["aralik80"], src["modeller"][k.ODAK]["aralik80_tutma_pct"])
        self.assertEqual([r["model"] for r in c["karsilastirma"]], ["Yarin=bugun", "GARCH", k.ODAK])
        with self.assertRaisesRegex(FileNotFoundError, "Karne dosyasi yok"):
            ts.karne(self.tmp)

    def test_missing_prices(self):
        with mock.patch.object(k, "PATHS", [self.tmp / "yok.csv"]):
            with self.assertRaisesRegex(FileNotFoundError, "Fiyat verisi yok"):
                ts.kart()


class TestYarismaService(Base):
    def setUp(self):
        super().setUp()
        ts._ufuklar.cache_clear()
        ts._bu_yil_yarisma.cache_clear()
        p = mock.patch.object(ts.yr, "chronos_var", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def test_summary_from_committed_file(self):
        d = ts.yarisma()
        self.assertEqual([u["ufuk_gun"] for u in d["ufuklar"]], [1, 3, 5])
        src = json.loads((PROJECT_ROOT / d["kaynak"]).read_text(encoding="utf-8"))
        for u in d["ufuklar"]:
            s = src["ufuklar"][str(u["ufuk_gun"])]
            self.assertEqual(u["kazanan"], s["kazanan"])
            self.assertEqual(u["final"]["tutma95"], s["final"][s["kazanan"]]["tutma95"])
            puanlar = [r["puan"] for r in u["modeller"]]
            self.assertEqual(puanlar, sorted(puanlar))
            w = next(r for r in u["modeller"] if r["model"] == u["kazanan"])
            self.assertTrue(w["uygun"])                                            # kazanan hedefe yakin olmali
            self.assertEqual(min(r["puan"] for r in u["modeller"] if r["uygun"]), w["puan"])

    def test_horizon_cards_with_fallback(self):
        d = ts.ufuklar()
        self.assertEqual([u["ufuk_gun"] for u in d["ufuklar"]], [1, 3, 5])
        for u in d["ufuklar"]:
            self.assertLess(u["aralik95"][0], u["aralik80"][0])
            self.assertLess(u["aralik80"][1], u["aralik95"][1])
            if u["model"] in ("Chronos", "Birlesim-2"):
                self.assertTrue(u["yedek"])
                self.assertEqual(u["kullanilan"], "ARIMA-GARCH")
        w = [u["aralik80"][1] - u["aralik80"][0] for u in d["ufuklar"]]
        self.assertLess(w[0], w[1])
        self.assertLess(w[1], w[2])                                               # ufuk uzadikca aralik genisler
        self.assertTrue(0 <= d["hareketli_olasiligi_pct"] <= 100)

    def test_this_year_with_winner(self):
        d = ts.bu_yil_yarisma()
        self.assertEqual(d["yil"], 2026)
        g = pd.DataFrame(d["gunler"])
        self.assertTrue(g["tarih"].str.startswith("2026").all())
        in80 = ((g["gercek"] >= g["lo80"]) & (g["gercek"] <= g["hi80"])).sum()
        self.assertEqual(d["aralik80_tuttu"], int(in80))

    def test_missing_result_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "Yarisma sonucu yok"):
            ts.yarisma_dosyasi(self.tmp)


class TestApiAndPage(Base):
    def client(self):
        return TestClient(api.app, base_url="http://127.0.0.1")

    def test_endpoints(self):
        cl = self.client()
        with mock.patch.object(ts, "secilen_pencere", lambda tool_dir=None: 250 if tool_dir is None else 750):
            r = cl.get("/api/tahmin/yarin")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["pencere"], 250)
            self.assertEqual(cl.get("/api/tahmin/bu-yil").json()["yil"], 2026)
        self.assertEqual(cl.get("/api/tahmin/karne").json()["model"], k.ODAK)
        self.assertIn("script-src 'self'", r.headers["content-security-policy"])

    def test_yarisma_endpoints(self):
        cl = self.client()
        with mock.patch.object(ts.yr, "chronos_var", lambda: False):
            ts._ufuklar.cache_clear()
            self.assertEqual(len(cl.get("/api/tahmin/ufuklar").json()["ufuklar"]), 3)
            self.assertEqual(cl.get("/api/tahmin/yarisma").json()["ufuklar"][0]["ufuk_gun"], 1)
            self.assertEqual(cl.get("/api/tahmin/bu-yil-yarisma").status_code, 200)

    def test_503_when_data_missing(self):
        with mock.patch.object(k, "PATHS", [self.tmp / "yok.csv"]):
            r = self.client().get("/api/tahmin/yarin")
        self.assertEqual(r.status_code, 503)
        self.assertIn("Fiyat verisi yok", r.json()["detail"])

    def test_pages(self):
        cl = self.client()
        ana = cl.get("/")
        self.assertEqual(ana.status_code, 200)
        self.assertIn("/static/ana.js", ana.text)
        self.assertIn('href="/detay"', ana.text)
        detay = cl.get("/detay")
        self.assertIn("/static/app.js", detay.text)
        self.assertIn('href="/"', detay.text)
        js = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
        for u in ("/api/tahmin/ufuklar", "/api/tahmin/yarisma", "/api/tahmin/bu-yil-yarisma"):
            self.assertIn(u, js)
        self.assertNotIn("innerHTML", js)
        self.assertNotIn("style:", js)                                            # CSP: satir ici stil yok
        self.assertEqual(cl.get("/static/ana.css").status_code, 200)


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
