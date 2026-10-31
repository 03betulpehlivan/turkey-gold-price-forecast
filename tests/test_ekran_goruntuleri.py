"""
Ekran goruntuleri araci testi: src/ekran_goruntuleri.py
Calistirma: python -m unittest tests.test_ekran_goruntuleri -v
Tarayici acilmaz; secicilerin sayfada, goruntulerin raporda ve depoda oldugu kontrol edilir.
"""

from pathlib import Path
import contextlib
import io
import re
import sys
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ekran_goruntuleri as eg  # noqa: E402


class TestEkranGoruntuleri(unittest.TestCase):
    def test_selectors_exist_on_page(self):
        html = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "src" / "web" / "ana.js").read_text(encoding="utf-8")
        for sec in list(eg.EKRANLAR.values()) + eg.HAZIR:
            ilk = sec.split()[0]
            if ilk.startswith("#"):
                self.assertIn(f'id="{ilk[1:]}"', html, sec)
            else:
                self.assertRegex(html, rf'class="[^"]*\b{re.escape(ilk[1:])}\b', sec)
        self.assertIn('class="serit"', html)                                    # gizlenen ust menu
        for fn in ("function gosteriDur(", "function gosteriGit("):            # aracin cagirdigi sayfa islevleri
            self.assertIn(fn, js)

    def test_report_uses_every_screenshot_and_files_are_committed(self):
        report = (PROJECT_ROOT / "docs" / "danisman_raporu.md").read_text(encoding="utf-8")
        for ad in eg.EKRANLAR:
            self.assertIn(f"figures/{ad}.png", report, ad)
            p = PROJECT_ROOT / "docs" / "figures" / f"{ad}.png"
            self.assertTrue(p.exists(), f"{p} yok: pano acikken python -m src.ekran_goruntuleri")
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", ad)

    def test_missing_playwright_is_reported(self):
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"playwright": None, "playwright.sync_api": None}), \
                contextlib.redirect_stdout(out):
            self.assertEqual(eg.main([]), 1)
        self.assertIn("pip install playwright", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
