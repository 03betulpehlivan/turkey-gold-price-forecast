"""
Arastirma bulgulari testleri: src/findings.py, /api/findings ve pano bolumu
Calistirma: python -m unittest tests.test_findings -v
Depodaki kayitli sonuc dosyalari yalnizca OKUNUR; hata durumlari gecici klasorde sinanir.
"""

from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message=".*httpx2.*")
from fastapi.testclient import TestClient  # noqa: E402

from src import findings as fd  # noqa: E402
from src.api import app as api  # noqa: E402


class TestCollect(unittest.TestCase):
    def test_committed_results_are_all_readable(self):
        items = {it["id"]: it for it in fd.collect()}
        self.assertEqual(set(items), set(fd.FILES))
        for k, it in items.items():
            with self.subTest(k):
                self.assertTrue(it["available"], it.get("error"))
                self.assertTrue(it["values"])
        v = items["oynaklik"]["values"]
        self.assertLessEqual(v["garch_years_better"], v["years"])
        self.assertIn("hafta", items["trend"]["values"])
        self.assertIsInstance(items["geriye_donuk"]["values"]["supported"], bool)

    def test_values_match_source_files(self):
        items = {it["id"]: it for it in fd.collect()}
        src = json.loads((PROJECT_ROOT / fd.FILES["prim"]).read_text(encoding="utf-8"))
        self.assertEqual(items["prim"]["values"]["phi"], src["ar1_weekly"]["phi"])
        src = json.loads((PROJECT_ROOT / fd.FILES["geriye_donuk"]).read_text(encoding="utf-8"))
        self.assertEqual(items["geriye_donuk"]["values"]["ons_vs_rw_p"], src["primary_walk_forward"]["Ons_vs_RW"]["p_used"])

    def test_back_exam_uses_holm_adjusted_p(self):
        c = {"improvement_pct": 1.0, "improvement_ci95": [-1, 3], "p_value": 0.01, "p_used": 0.02}
        v = fd._geriye({"hypothesis_supported": False, "n": 10, "period": ["2001-01-01", "2004-12-31"],
                        "primary_walk_forward": {"Ons_vs_RW": c, "Ons_vs_Egilim": dict(c, p_used=0.03)}})
        self.assertEqual((v["ons_vs_rw_p"], v["ons_vs_egilim_p"]), (0.02, 0.03))

    def test_missing_broken_and_nan(self):
        tmp = Path(tempfile.mkdtemp(prefix="fd_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = tmp / fd.FILES["prim"]
        p.parent.mkdir(parents=True)
        p.write_text('{"ar1_weekly": {"phi": NaN}}', encoding="utf-8")            # alan eksik + NaN
        g = tmp / fd.FILES["oynaklik"]
        g.parent.mkdir(parents=True)
        g.write_text("{bozuk", encoding="utf-8")
        items = {it["id"]: it for it in fd.collect(tmp)}
        self.assertFalse(items["sizinti"]["available"])
        self.assertFalse(items["prim"]["available"])
        self.assertIn("KeyError", items["prim"]["error"])
        self.assertFalse(items["oynaklik"]["available"])
        self.assertEqual(fd._clean({"a": float("nan"), "b": [float("inf"), 1.0]}), {"a": None, "b": [None, 1.0]})


class TestApiAndDashboard(unittest.TestCase):
    def test_endpoint(self):
        r = TestClient(api.app, base_url="http://127.0.0.1").get("/api/findings")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["items"]), len(fd.FILES))
        self.assertIn("script-src 'self'", r.headers["content-security-policy"])

    def test_dashboard_section(self):
        html = (PROJECT_ROOT / "src" / "web" / "index.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "src" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="bulgular"', html)
        self.assertIn('al("/api/findings")', js)
        for k in fd.FILES:
            self.assertIn(f"{k}: v =>", js)
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
