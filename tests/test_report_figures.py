"""
Rapor grafikleri testi: src/report_figures.py
Calistirma: python -m unittest tests.test_report_figures -v
Grafikler kayitli sonuc dosyalarindan gecici klasore cizilir; depodaki docs/figures degismez.
"""

from pathlib import Path
import contextlib
import inspect
import io
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import report_figures as rf  # noqa: E402


class TestFigures(unittest.TestCase):
    def test_all_figures_render_from_committed_results(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rf.main(["--out", str(tmp)]), 0)
        for name in rf.FIGURES:
            p = tmp / f"{name}.png"
            self.assertTrue(p.exists(), name)
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(p.stat().st_size, 10000)

    def test_committed_figures_exist_for_report(self):
        report = (PROJECT_ROOT / "docs" / "danisman_raporu.md").read_text(encoding="utf-8")
        for name in rf.FIGURES:
            self.assertTrue((PROJECT_ROOT / "docs" / "figures" / f"{name}.png").exists(), name)
            self.assertIn(f"figures/{name}.png", report)

    def test_report_and_readme_use_plain_terms(self):
        for name in ("docs/danisman_raporu.md", "README.md"):
            text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("se\u00e7im d\u00f6nemi", text.lower(), name)       # siyasi secimle karisiyordu
            self.assertIn("Haz\u0131rl\u0131k d\u00f6nemi", text, name)
            self.assertIn("S\u0131nav d\u00f6nemi", text, name)
        report = (PROJECT_ROOT / "docs" / "danisman_raporu.md").read_text(encoding="utf-8")
        self.assertNotIn("yine %95 g\u00fcvenilir", report)                     # %80 genislik ile %95 tutma karismasin
        self.assertIn("arife", report)

    def test_width_figure_reports_80_percent_hit_rate(self):
        src = inspect.getsource(rf.fig_yarisma)
        self.assertIn('["tutma80"]', src)                                        # cubuklar %80 genislik: yazi da %80
        self.assertNotIn('["tutma95"]', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
