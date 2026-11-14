"""
src/tool/site.py testleri: herkese acik sabit site (GitHub Pages). Gecici klasorlerde calisir.
"""
from pathlib import Path
from unittest import mock
import json
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool import site  # noqa: E402

JS = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
HTML = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")


class Yanit:
    def __init__(self, kod, veri):
        self.status_code, self._v, self.text = kod, veri, json.dumps(veri)

    def json(self):
        return self._v


class SahteIstemci:
    def __init__(self, bozuk=None):
        self.bozuk = bozuk

    def get(self, url):
        u = url.rsplit("/", 1)[1]
        if u == self.bozuk:
            return Yanit(500, {"detail": "veri yok"})
        return Yanit(200, {"uc": u, "bilgi_gunu": "2026-11-13", "son_kapanis": 6537.4})


class TestSite(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="site_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "_site"

    def test_every_page_endpoint_is_published(self):
        """ana.js'nin istedigi her /api/tahmin/ ucu sitede olmali (yeni uc eklenirse bu test hatirlatir)."""
        self.assertEqual(set(re.findall(r'"/api/tahmin/([\w-]+)"', JS)), set(site.UCLAR))
        self.assertNotRegex(site.betik(JS), r'["\'`]/api/')

    def test_build_writes_relative_static_site(self):
        d = site.uret(self.out, SahteIstemci())
        self.assertEqual(d["bilgi_gunu"], "2026-11-13")
        html = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="static/ana.css"', html)
        self.assertIn('src="static/ana.js"', html)
        self.assertNotIn('"/static/', html)
        self.assertNotIn('href="/detay"', html)                                     # sunucu gerektiren baglantilar yok
        self.assertNotIn('href="/docs"', html)
        self.assertIn("Gram_Altin_Proje_Raporu.pdf", html)
        js = (self.out / "static" / "ana.js").read_text(encoding="utf-8")
        for u in site.UCLAR:
            self.assertIn(f'"api/tahmin/{u}.json"', js)
            self.assertEqual(json.loads((self.out / "api" / "tahmin" / f"{u}.json").read_text(encoding="utf-8"))["uc"], u)
        self.assertTrue((self.out / ".nojekyll").exists())
        self.assertEqual((self.out / "static" / "ana.css").read_bytes(), (PROJECT_ROOT / "src" / "web" / "ana.css").read_bytes())

    def test_source_files_unchanged(self):
        site.uret(self.out, SahteIstemci())
        self.assertEqual((PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8"), JS)
        self.assertEqual((PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8"), HTML)

    def test_failed_endpoint_keeps_previous_site(self):
        site.uret(self.out, SahteIstemci())
        once = {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        with self.assertRaises(site.SiteHatasi):
            site.uret(self.out, SahteIstemci(bozuk="canli-karne"))
        self.assertEqual({p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}, once)   # yarim site yok

    def test_unconverted_api_call_refused(self):
        with self.assertRaises(site.SiteHatasi):
            site.betik('al("/api/tahmin/yeni-uc")')

    def test_real_api_end_to_end(self):
        """Gercek API (src/api/app.py) sentetik fiyatlarla: bes ucun hepsi uretilir."""
        from fastapi.testclient import TestClient
        from src.api import app as api
        from src.tool import service as ts
        csvp = self.tmp / "p.csv"
        d = pd.bdate_range("2018-07-02", "2026-11-13")
        rng = np.random.default_rng(3)
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"),
                      "kapanis_tl_gr": 6500 * np.exp(np.cumsum(rng.normal(0, 0.01, len(d))) - 0.5)}).to_csv(csvp, index=False)
        for f in (ts._ufuklar, ts._bu_yil_yarisma, ts._canli_karne):
            f.cache_clear()
        with mock.patch.object(ts.k, "PATHS", [csvp]), mock.patch.object(ts.yr, "chronos_var", return_value=False):
            r = site.uret(self.out, TestClient(api.app, base_url="http://127.0.0.1"))
        for f in (ts._ufuklar, ts._bu_yil_yarisma, ts._canli_karne):
            f.cache_clear()
        self.assertEqual(r["bilgi_gunu"], "2026-11-13")
        u = json.loads((self.out / "api" / "tahmin" / "ufuklar.json").read_text(encoding="utf-8"))
        self.assertEqual([x["ufuk_gun"] for x in u["ufuklar"]], [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
