"""
Yeniden uretim denetimi testleri: src/repro.py
Calistirma: python -m unittest tests.test_repro -v
Sahte calismalar ve gecici klasorler; gercek analizler calistirilmaz.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import repro  # noqa: E402

RESULT = {"created_at_utc": "x", "a": 1.0, "b": [1, 2.5, float("nan")], "c": {"2012": 0.1, "ok": True, "s": "metin"}}


class TestCompare(unittest.TestCase):
    def test_equal_with_nan_tolerance_and_ignored_keys(self):
        actual = {"created_at_utc": "baska", "a": 1.0 + 1e-13, "b": [1, 2.5, float("nan")],
                  "c": {"2012": 0.1, "ok": True, "s": "metin"}}
        self.assertEqual(repro.compare(RESULT, actual), [])

    def test_differences_reported_with_path(self):
        actual = {"a": 1.001, "b": [1, 2.5], "c": {"2012": 0.1, "ok": False, "yeni": 1}}
        d = repro.compare(RESULT, actual)
        self.assertTrue(any(x.startswith("/a:") for x in d))
        self.assertIn("/b: uzunluk 3 != 2", d)
        self.assertTrue(any("/c/ok" in x for x in d))
        self.assertIn("/c/s: anahtar eksik", d)
        self.assertIn("/c/yeni: anahtar fazla", d)

    def test_normalize_matches_json_round_trip(self):
        actual = repro.normalize({"phi": {2012: np.float64(0.4)}, "n": np.int64(3)})
        self.assertEqual(actual, {"phi": {"2012": 0.4}, "n": 3})
        self.assertEqual(repro.compare({"phi": {"2012": 0.4}}, repro.normalize({"phi": {2012: 0.4}})), [])

    def test_bool_is_not_number(self):
        self.assertNotEqual(repro.compare({"x": True}, {"x": 1}), [])


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rp_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "r").mkdir()
        (self.tmp / "r" / "s.json").write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
        self.art = {"iyi": ("r/s.json", lambda: dict(RESULT, created_at_utc="yeni")),
                    "kotu": ("r/s.json", lambda: dict(RESULT, a=2.0)),
                    "yok": ("r/yok.json", lambda: RESULT),
                    "coker": ("r/s.json", lambda: 1 / 0)}

    def test_statuses(self):
        st = {n: repro.check(n, self.tmp, self.art)["status"] for n in self.art}
        self.assertEqual(st, {"iyi": "AYNI", "kotu": "FARKLI", "yok": "EKSIK", "coker": "EKSIK"})

    def test_computed_result_is_normalized_before_compare(self):
        art = {"np": ("r/s.json", lambda: {"a": np.float64(1.0), "b": [1, 2.5, float("nan")],
                                          "c": {2012: np.float64(0.1), "ok": True, "s": "metin"}})}
        self.assertEqual(repro.check("np", self.tmp, art)["status"], "AYNI")

    def test_never_writes_results(self):
        before = {p: p.read_bytes() for p in self.tmp.rglob("*") if p.is_file()}
        for n in self.art:
            repro.check(n, self.tmp, self.art)
        self.assertEqual(before, {p: p.read_bytes() for p in self.tmp.rglob("*") if p.is_file()})

    def test_main_exit_codes(self):
        with mock.patch.object(repro, "ROOT", self.tmp), contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(repro, "ARTIFACTS", {"iyi": self.art["iyi"]}):
                self.assertEqual(repro.main([]), 0)
            with mock.patch.object(repro, "ARTIFACTS", {"iyi": self.art["iyi"], "kotu": self.art["kotu"]}):
                self.assertEqual(repro.main([]), 1)

    def test_registered_artifacts_point_to_committed_results(self):
        for name, (rel, fn) in repro.ARTIFACTS.items():
            if name != "bilesen":                                               # bilesen sonucu ilk calistirmada yazilir
                self.assertTrue((PROJECT_ROOT / rel).exists(), rel)
            self.assertTrue(callable(fn))
        self.assertEqual(set(repro.ARTIFACTS), {"sizinti", "oynaklik", "trend", "prim", "bilesen", "geriye_donuk"})


class TestBackExamIsNotRerun(unittest.TestCase):
    def test_no_registry_event_and_data_hash_checked(self):
        src = (PROJECT_ROOT / "src" / "repro.py").read_text(encoding="utf-8")
        body = src[src.index("def _back_exam"):src.index("ARTIFACTS = {")]
        self.assertNotIn("be.run(", body)
        self.assertNotIn("_log", body)
        self.assertIn('pr["data_md5"]["dataset.csv"]', body)


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
