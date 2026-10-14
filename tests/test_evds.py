"""
EVDS araci testleri: src/data/evds.py
Calistirma: python -m unittest tests.test_evds -v
EVDS'ye BAGLANILMAZ: alan adlari evds 0.4.0 istemcisinin kaynak kodundan alinan sahte yanitlar kullanilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import evds  # noqa: E402

KEY = "GIZLI-ANAHTAR-123"
GROUPS = [{"DATAGROUP_CODE": "bie_altin", "DATAGROUP_NAME": "Alt\u0131n Fiyatlar\u0131", "DATAGROUP_NAME_ENG": "Gold Prices"},
          {"DATAGROUP_CODE": "bie_rez", "DATAGROUP_NAME": "ALTIN REZERVLER\u0130", "DATAGROUP_NAME_ENG": "Reserves"},
          {"DATAGROUP_CODE": "bie_doviz", "DATAGROUP_NAME": "D\u00f6viz Kurlar\u0131", "DATAGROUP_NAME_ENG": "Exchange Rates"}]
SERIES = {"bie_altin": [{"SERIE_CODE": "TP.ALT.SAT", "SERIE_NAME": "Kulce altin satis (TL/gr)", "FREQUENCY_STR": "GUNLUK",
                         "START_DATE": "02-01-1950", "END_DATE": "13-10-2026"},
                        {"SERIE_CODE": "TP.ALT.ALIS", "SERIE_NAME": "Kulce altin alis (TL/gr)", "FREQUENCY_STR": "GUNLUK",
                         "START_DATE": "02-01-1950", "END_DATE": "13-10-2026"}],
          "bie_rez": [{"SERIE_CODE": "TP.REZ.ALT", "SERIE_NAME": "Altin rezervi", "FREQUENCY_STR": "HAFTALIK",
                       "START_DATE": "01-01-2000", "END_DATE": "09-10-2026"}]}
ITEMS = {"items": [{"Tarih": "05-01-2015", "TP_ALT_SAT": "88.10", "TP_ALT_ALIS": "87.90", "UNIXTIME": {"$numberLong": "1"}},
                   {"Tarih": "02-01-2015", "TP_ALT_SAT": "87.50", "TP_ALT_ALIS": None, "UNIXTIME": {"$numberLong": "0"}},
                   {"Tarih": "06-01-2015", "TP_ALT_SAT": "ND", "TP_ALT_ALIS": "88.00", "UNIXTIME": {"$numberLong": "2"}}],
         "totalCount": 3}


class FakeGet:
    def __init__(self, status=200):
        self.status, self.seen = status, []

    def __call__(self, url, headers):
        self.seen.append((url, dict(headers)))
        if self.status != 200:
            return self.status, b"{}"
        if "datagroups/" in url:
            body = GROUPS
        elif "serieList/" in url:
            body = SERIES[url.split("code=")[1]]
        elif "categories/" in url:
            body = [{"CATEGORY_ID": 1, "TOPIC_TITLE_TR": "PIYASA VERILERI"}]
        else:
            body = ITEMS
        return 200, json.dumps(body).encode("utf-8")


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


class TestEvds(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="evds_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.get = FakeGet()
        self.c = evds.Client(KEY, get=self.get)

    def test_turkish_normalization(self):
        for s in ("ALTIN", "Alt\u0131n", "alt\u0131n", "AlTIN", "ALT\u0130N"):
            self.assertEqual(evds.normalize(s), "altin")
        self.assertEqual(evds.normalize("K\u00fcl\u00e7e"), "kulce")

    def test_key_only_in_header_never_in_url_or_files(self):
        cat, snap = evds.run_discover(self.c, evds.DEFAULT_TERMS, self.tmp, datetime(2026, 10, 14, tzinfo=timezone.utc))
        df, rep, snap2 = evds.run_download(self.c, ["TP.ALT.SAT", "TP.ALT.ALIS"], "01-01-2015", "", self.tmp)
        for url, headers in self.get.seen:
            self.assertEqual(headers, {"key": KEY})
            self.assertNotIn(KEY, url)
        for p in self.tmp.rglob("*"):
            if p.is_file():
                self.assertNotIn(KEY, p.read_text(encoding="utf-8", errors="ignore"), p)

    def test_discover_finds_gold_groups_with_turkish_casing(self):
        cat = evds.discover(self.c)
        self.assertEqual(set(cat["DATAGROUP_CODE"]), {"bie_altin", "bie_rez"})
        self.assertEqual(len(cat), 3)
        row = cat.set_index("SERIE_CODE").loc["TP.ALT.SAT"]
        self.assertEqual((row["FREQUENCY"], row["START_DATE"]), ("GUNLUK", "02-01-1950"))

    def test_request_urls_follow_verified_protocol(self):
        self.c.datagroups()
        self.c.series_list("bie_altin")
        self.c.data(["TP.ALT.SAT", "TP.ALT.ALIS"], "01-01-2015", "13-10-2026")
        urls = [u for u, _ in self.get.seen]
        self.assertEqual(urls[0], evds.BASE_URL + "datagroups/mode=0&code=&type=json")
        self.assertEqual(urls[1], evds.BASE_URL + "serieList/type=json&code=bie_altin")
        self.assertTrue(urls[2].startswith(evds.BASE_URL + "series=TP.ALT.SAT-TP.ALT.ALIS&startDate=01-01-2015"
                                                           "&endDate=13-10-2026&type=json"))
        self.assertTrue(evds.BASE_URL.startswith("https://evds3.tcmb.gov.tr/"))

    def test_parse_items(self):
        df = evds.parse_items(ITEMS["items"], ["TP.ALT.SAT", "TP.ALT.ALIS", "TP.YOK"])
        self.assertEqual(list(df["Date"]), list(pd.to_datetime(["2015-01-02", "2015-01-05", "2015-01-06"])))
        self.assertEqual(df.loc[0, "TP.ALT.SAT"], 87.50)
        self.assertTrue(pd.isna(df.loc[0, "TP.ALT.ALIS"]))           # None
        self.assertTrue(pd.isna(df.loc[2, "TP.ALT.SAT"]))            # 'ND'
        self.assertTrue(df["TP.YOK"].isna().all())                   # yanitta olmayan seri
        with self.assertRaisesRegex(evds.EVDSError, "Tarih alani"):
            evds.parse_items([{"X": 1}], ["TP.ALT.SAT"])
        with self.assertRaisesRegex(evds.EVDSError, "Ayristirilamayan"):
            evds.parse_items([{"Tarih": "2015-Q1", "TP_ALT_SAT": "1"}], ["TP.ALT.SAT"])

    def test_download_snapshot_and_coverage(self):
        df, rep, snap = evds.run_download(self.c, ["TP.ALT.SAT", "TP.ALT.ALIS"], "01-01-2015", "", self.tmp)
        m = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["series"], ["TP.ALT.SAT", "TP.ALT.ALIS"])
        self.assertEqual(rep["TP.ALT.SAT"]["observations"], 2)
        self.assertEqual(rep["TP.ALT.SAT"]["missing_or_empty"], 1)
        self.assertEqual(rep["TP.ALT.SAT"]["first_date"], "2015-01-02")
        self.assertTrue((snap / "data.csv").exists() and (snap / "data_raw.json").exists())
        import hashlib
        self.assertEqual(m["raw_md5"], hashlib.md5((snap / "data_raw.json").read_bytes()).hexdigest())

    def test_errors(self):
        with self.assertRaises(evds.EVDSError):
            evds.Client("")
        bad = evds.Client(KEY, get=FakeGet(status=403))
        with self.assertRaisesRegex(evds.EVDSError, "403"):
            bad.datagroups()
        with mock.patch.dict("os.environ", {"EVDS_API_KEY": ""}):
            self.assertEqual(quiet(evds.main, ["kesif"]), 1)

    def test_legacy_ssl_option_enabled(self):
        s = evds._session()
        adapter = s.get_adapter("https://evds3.tcmb.gov.tr/")
        ctx = adapter.poolmanager.connection_pool_kw["ssl_context"]
        self.assertTrue(ctx.options & 0x4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
