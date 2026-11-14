"""
src/tool/bulut_veri.py testleri: bulutta ana (arsiv) dosya olmadan guncel seri; varsa bilgisayardaki davranis.
Sahte EVDS istemcisi kullanilir (ag ve anahtar gerekmez).
"""
from pathlib import Path
from unittest import mock
import hashlib
import shutil
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import evds  # noqa: E402
from src.tool import bulut_veri as bv  # noqa: E402
from tests.test_physical_gold import FakeEVDS  # noqa: E402

NOW = pd.Timestamp("2018-08-31", tz="UTC").to_pydatetime()


class TestGuncelle(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bv_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "live" / "physical_gold_daily.csv"

    def test_without_archive_writes_current_series_only(self):
        with mock.patch.object(evds, "OUT_ROOT", self.tmp / "evds"):
            r = bv.guncelle(evds.Client("k", get=FakeEVDS()), self.tmp / "yok.csv", self.out, now=NOW)
        live = pd.read_csv(self.out, parse_dates=["Date"])
        self.assertEqual(set(live["kaynak"]), {"BIST_KMP"})
        self.assertEqual((r["last_date"], r["arsiv"]), ("2018-08-31", False))
        self.assertTrue(live["Date"].is_monotonic_increasing)
        self.assertFalse((self.tmp / "yok.csv").exists())

    def test_with_archive_behaves_like_pc_refresh(self):
        base = pd.DataFrame({"Date": pd.to_datetime(["2018-06-28", "2018-06-29"]), "kapanis_tl_gr": [180.0, 184.0],
                             "agort_tl_gr": [180.0, 184.0], "kapanis_usd_ons": [1250.0, 1252.0], "kaynak": "IAB_arsiv"})
        bp = self.tmp / "base.csv"
        base.to_csv(bp, index=False, date_format="%Y-%m-%d")
        h = hashlib.md5(bp.read_bytes()).hexdigest()
        with mock.patch.object(evds, "OUT_ROOT", self.tmp / "evds"):
            r = bv.guncelle(evds.Client("k", get=FakeEVDS()), bp, self.out, now=NOW)
        live = pd.read_csv(self.out, parse_dates=["Date"])
        self.assertEqual(set(live["kaynak"]), {"IAB_arsiv", "BIST_KMP"})
        self.assertTrue(r["arsiv"])
        self.assertEqual(hashlib.md5(bp.read_bytes()).hexdigest(), h)                 # ana dosya degismez

    def test_same_rows_as_pc_after_2018(self):
        """Arsivsiz ve arsivli canli kopyanin 2018 sonrasi satirlari birebir ayni olmali."""
        base = pd.DataFrame({"Date": pd.to_datetime(["2018-06-29"]), "kapanis_tl_gr": [184.0], "agort_tl_gr": [184.0],
                             "kapanis_usd_ons": [1252.0], "kaynak": "IAB_arsiv"})
        bp = self.tmp / "base.csv"
        base.to_csv(bp, index=False, date_format="%Y-%m-%d")
        a, b = self.tmp / "a.csv", self.tmp / "b.csv"
        with mock.patch.object(evds, "OUT_ROOT", self.tmp / "evds"):
            bv.guncelle(evds.Client("k", get=FakeEVDS()), bp, a, now=NOW)
            bv.guncelle(evds.Client("k", get=FakeEVDS()), self.tmp / "yok.csv", b, now=NOW)
        pa, pb = pd.read_csv(a), pd.read_csv(b)
        pd.testing.assert_frame_equal(pa[pa["kaynak"] == "BIST_KMP"].reset_index(drop=True), pb)


if __name__ == "__main__":
    unittest.main()
