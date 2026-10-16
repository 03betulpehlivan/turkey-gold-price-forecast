"""
Makro gostergeler testleri: src/data/macro.py
Calistirma: python -m unittest tests.test_macro -v
Sahte EVDS yaniti (ag erisimi yok); gecici klasorler. Gercek data/ kullanilmaz.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import evds  # noqa: E402
from src.data import macro  # noqa: E402

KEY = "GIZLI-ANAHTAR-123456"
NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def item(t, tufe, tr, us):
    return {"Tarih": t, "TP_GENENDEKS_T1": tufe, "TP_BISPOLFAIZ_TUR": tr, "TP_BISPOLFAIZ_USA": us, "UNIXTIME": {"$numberLong": "1"}}


class Fake:
    def __init__(self, items):
        self.items, self.headers, self.urls = items, [], []

    def __call__(self, url, headers):
        self.urls.append(url)
        self.headers.append(headers)
        return 200, json.dumps({"totalCount": len(self.items), "items": self.items}).encode("utf-8")


class TestParse(unittest.TestCase):
    def test_monthly_and_daily_formats(self):
        self.assertEqual(macro.parse_month("2015-1"), "2015-01")
        self.assertEqual(macro.parse_month("2015-12"), "2015-12")
        self.assertEqual(macro.parse_month("01-03-2016"), "2016-03")
        for bad in ("2015-13", "2015-Q1", "2015/01", ""):
            with self.assertRaises(macro.MacroError):
                macro.parse_month(bad)

    def test_values_nd_and_duplicates(self):
        df = macro.parse_items([item("2004-2", "110.5", "ND", "1.0"), item("2004-1", "109", "26", None),
                                item("2004-2", "110.5", "ND", "1.0")])
        self.assertEqual(list(df["month"]), ["2004-01", "2004-02"])
        self.assertEqual(df["tufe"].tolist(), [109.0, 110.5])
        self.assertTrue(df["politika_tr"].isna().iat[1])
        self.assertTrue(df["politika_us"].isna().iat[0])
        with self.assertRaisesRegex(macro.MacroError, "farkli"):
            macro.parse_items([item("2004-1", "1", "2", "3"), item("2004-1", "9", "2", "3")])
        with self.assertRaises(macro.MacroError):
            macro.parse_items([])


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mac_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_download_writes_snapshot_and_csv_without_key(self):
        fake = Fake([item(f"2004-{m}", str(100 + m), "20", "1") for m in range(1, 13)])
        client = evds.Client(KEY, get=fake)
        df, cov, snap = macro.download(client, raw_root=self.tmp / "raw", out_path=self.tmp / "m.csv", now=NOW)
        self.assertEqual(len(df), 12)
        self.assertEqual(cov["tufe"]["first"], "2004-01")
        self.assertEqual(cov["tufe"]["last"], "2004-12")
        self.assertIn("series=TP.GENENDEKS.T1-TP.BISPOLFAIZ.TUR-TP.BISPOLFAIZ.USA", fake.urls[0])
        self.assertIn("endDate=16-10-2026", fake.urls[0])
        self.assertEqual(fake.headers[0], {"key": KEY})                      # anahtar baslikta, URL'de degil
        self.assertNotIn(KEY, fake.urls[0])
        for p in list((self.tmp / "raw").rglob("*")) + [self.tmp / "m.csv"]:
            if p.is_file():
                self.assertNotIn(KEY, p.read_text(encoding="utf-8"))
        loaded = macro.load(self.tmp / "m.csv")
        self.assertEqual(loaded["month"].iat[0], "2004-01")
        man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["series"], macro.SERIES)

    def test_truncated_response_refused(self):
        fake = Fake([item("2004-1", "1", "2", "3")] * evds.ROW_LIMIT)
        with self.assertRaisesRegex(macro.MacroError, "kesilmis"):
            macro.download(evds.Client(KEY, get=fake), raw_root=self.tmp, out_path=self.tmp / "m.csv", now=NOW)
        self.assertFalse((self.tmp / "m.csv").exists())

    def test_missing_file_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "macro indir"):
            macro.load(self.tmp / "yok.csv")


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
