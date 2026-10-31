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
import re
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

    def test_summary_has_fixed_interval_comparison(self):
        d = ts.yarisma()
        src = json.loads((PROJECT_ROOT / d["kaynak"]).read_text(encoding="utf-8"))
        for u in d["ufuklar"]:
            s = src["ufuklar"][str(u["ufuk_gun"])]
            self.assertNotEqual(u["kazanan"], ts.SABIT)
            self.assertEqual(u["sabit"]["final"], s["final"][ts.SABIT])          # kazananin degil, basit yontemin sonucu
            self.assertEqual(u["sabit"]["yillar"], s["yillar"][ts.SABIT])
            self.assertEqual([r["yil"] for r in u["sabit"]["yillar"]], [r["yil"] for r in u["yillar"]])
        u1 = d["ufuklar"][0]
        self.assertLess(u1["final"]["genislik80"], u1["sabit"]["final"]["genislik80"])   # kayitli sonuc: daha dar aralik

    def test_fixed_interval_missing_is_none(self):
        res = json.loads(ts.yarisma_dosyasi().read_text(encoding="utf-8"))
        for u in res["ufuklar"].values():
            for part in ("secim", "final", "yillar"):
                u[part].pop(ts.SABIT, None)
            u.pop("anlamlilik_sabit_araliga_gore", None)
        (self.tmp / "yarisma_1-3-5.json").write_text(json.dumps(res), encoding="utf-8")
        d = ts.yarisma(self.tmp)
        self.assertEqual(len(d["ufuklar"]), 3)
        for u in d["ufuklar"]:
            self.assertIsNone(u["sabit"]["final"])
            self.assertIsNone(u["sabit"]["yillar"])

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
        for u in d["ufuklar"]:                                                     # etiket: tatiller atlanir
            self.assertEqual(u["hedef_gun_yaklasik"], ts.hedef_gun(d["bilgi_gunu"], u["ufuk_gun"]))

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


class TestHedefGun(Base):
    def test_skips_weekends_and_holidays(self):
        self.assertEqual(ts.hedef_gun("2026-10-27", 1), "2026-10-30")            # arife (28) ve bayram (29) atlanir
        self.assertEqual(ts.hedef_gun("2026-10-27", 3), "2026-11-03")            # ... ve hafta sonu
        self.assertEqual(ts.hedef_gun("2026-10-27", 5), "2026-11-05")
        self.assertEqual(ts.hedef_gun("2026-10-30", 1), "2026-11-02")
        self.assertEqual(ts.hedef_gun("2026-03-18", 1), "2026-03-23")            # arife (19) + bayram (20) + hafta sonu
        for arife in ("2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26", "2026-10-28"):
            self.assertIn(arife, ts.TATILLER)                                    # fiyat dosyasinda kapanis yok
        self.assertEqual(ts.hedef_gun("2026-05-25", 1), "2026-06-01")            # arife + Kurban Bayrami + hafta sonu
        self.assertEqual(ts.hedef_gun("2026-10-23", 1), "2026-10-26")            # yalnizca hafta sonu
        for t in ts.TATILLER:
            self.assertLess(pd.Timestamp(t).weekday(), 5, t)                     # listede yalnizca hafta ici gunler

    def test_horizon_labels_cross_holiday(self):
        rng = np.random.default_rng(3)
        d = pd.bdate_range("2021-01-04", "2026-10-27")
        r = rng.normal(0, 0.01, len(d))
        f = self.tmp / "fiyat_tatil.csv"
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "kapanis_tl_gr": 1000 * np.exp(np.cumsum(r))}).to_csv(f, index=False)
        out = ts._ufuklar.__wrapped__((str(f), 0), str(ts.yarisma_dosyasi()), False)
        self.assertEqual(out["bilgi_gunu"], "2026-10-27")
        self.assertEqual([u["hedef_gun_yaklasik"] for u in out["ufuklar"]], ["2026-10-30", "2026-11-03", "2026-11-05"])

    def test_scorecard_waiting_rows_name_expected_close(self):
        d = ts.canli_karne()
        self.assertEqual(d["son_fiyat_gunu"], str(k.load_series().index[-1].date()))
        self.assertTrue(d["son_kayitlar"])
        for r in d["son_kayitlar"]:
            if r["gercek"] is None:
                self.assertEqual(r["beklenen_gun"], ts.hedef_gun(r["bilgi_gunu"], r["ufuk_gun"]))
            else:
                self.assertIsNone(r["beklenen_gun"])


class TestMerkez(Base):
    def test_merkez_summary_and_missing(self):
        sahte = {"ufuklar": {str(h): {"yontemler": {
            "Araligin ortasi": {"final": {"hata_tl": 80.0 + h, "hata_pct": 1.2, "yarin_bugune_gore_pct": 1.5, "p_degeri": 0.05, "gun": 900}},
            "Yarin = bugun": {"final": {"hata_tl": 85.0 + h, "hata_pct": 1.3, "gun": 900}}}} for h in (5, 1, 3)}}
        (self.tmp / "merkez_tahmin.json").write_text(json.dumps(sahte), encoding="utf-8")
        d = ts.merkez(self.tmp)
        self.assertEqual([u["ufuk_gun"] for u in d["ufuklar"]], [1, 3, 5])
        self.assertEqual(d["ufuklar"][0]["hata_tl"], 81.0)
        self.assertEqual(d["ufuklar"][0]["rw_hata_tl"], 86.0)
        self.assertIn("(%80 araligin alt siniri + ust siniri) / 2", d["formul"])
        (self.tmp / "merkez_tahmin.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Merkez tahmin sonucu yok"):
            ts.merkez(self.tmp)
        js = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
        self.assertIn("/api/tahmin/merkez", js)
        self.assertIn("(f.aralik80[0] + f.aralik80[1]) / 2", js)


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


    def test_main_page_sections_and_csp_rules(self):
        html = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
        css = (PROJECT_ROOT / "src" / "web" / "ana.css").read_text(encoding="utf-8")
        for sid in ("tahmin", "aralik", "tuttu", "kriz", "nasil", "model", "veri", "denenen", "sozluk"):
            self.assertIn(f'<section id="{sid}"', html)                         # bolum var
            self.assertIn(f'href="#{sid}"', html)                                # ust menude baglantisi var
        for i in ("cetvel", "ufuk-sec", "duzey-sec", "tekrar", "oynat", "panel-aralik", "panel-merkez",
                  "yillar", "dar-genis", "sakin-calkanti", "guven", "canli", "yarisma",
                  "gosteri", "gosteri-cetvel", "gosteri-ileri", "yaris", "yaris-satirlar", "yaris-sec"):
            self.assertIn(f'id="{i}"', html)
            self.assertTrue(i in js, f"{i} ana.js icinde kullanilmiyor")          # JS bu ogeyi dolduruyor
        self.assertIsNone(re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html))     # satir ici betik yok
        self.assertNotIn("<style", html)
        self.assertNotIn(" style=", html)
        self.assertIsNone(re.search(r"\son[a-z]+=", html))                     # satir ici olay isleyici yok
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function", "document.write",
                    "style:", 'setAttribute("style"', "localStorage"):
            self.assertNotIn(bad, js)
        self.assertIn("TL eksik", js)                                            # merkez tahmin sapmasi, yonuyle
        self.assertIn("TL fazla", js)
        self.assertIn("r.gercek - (r.lo80 + r.hi80) / 2", js)                    # sapma = gerceklesen - merkez
        self.assertIn("u.sabit", js)                                             # sabit aralik karsilastirmasi
        self.assertIn("prefers-reduced-motion", css)                             # hareket azaltma tercihi
        self.assertIn("prefers-reduced-motion", js)
        self.assertIn('href="/detay"', html)
        for metin in (html.lower(), js.lower()):                                 # "secim donemi" siyasi secimle karisiyordu
            self.assertNotIn("se\u00e7im d\u00f6nemi", metin)
        self.assertIn("haz\u0131rl\u0131k d\u00f6nemi", js)
        self.assertIn('NE[m] ? el("small", {}, NE[m])', js)                      # her modelin sade aciklamasi tabloda
        ne = js[js.index("const NE = {"): js.index("};", js.index("const NE = {"))]
        res = json.loads(ts.yarisma_dosyasi().read_text(encoding="utf-8"))
        for m in res["ufuklar"]["1"]["secim"]:
            self.assertIn(f'"{m}": "', ne, f"{m} icin sade aciklama yok")
        self.assertIn("r.tutma80", js)                                           # yarisma tablosunda tutma oranlari
        self.assertNotIn('"Yar\u0131n"', js)                                    # dugme "Yarin" degil: tarih gecmiste kalabilir
        self.assertIn("r.beklenen_gun", js)                                      # bekleyen satir hangi kapanisi bekliyor
        self.assertIn("f.hedef_gun_yaklasik", js)
        self.assertNotIn("deprem", html)                                         # dogrulanmamis ornek kaldirildi
        self.assertNotIn("kapan\u0131\u015f g\u00fcnleri doldurulmaz", html)  # belirsiz ifade duzeltildi
        self.assertIn("arife", html)                                             # arifede kapanis olusmadigi yazili


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
