"""
Fiziki altin serisi ve EVDS parcali indirme testleri: src/data/physical_gold.py, src/data/evds.py
Calistirma: python -m unittest tests.test_physical_gold -v
Sahte EVDS sunucusu gercekte gozlenen davranisi taklit eder: bos endDate -> 400; >1000 satir -> SON 1000 satir.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import evds  # noqa: E402
from src.data import physical_gold as pg  # noqa: E402

DAYS = pd.bdate_range("1995-07-27", "2026-10-13")
HOLIDAYS = set(DAYS[::97])                                   # bazi gunler bos (tatil)
LEVEL = pd.Series(np.exp(np.linspace(np.log(9000), np.log(7_000_000), len(DAYS))), index=DAYS)   # TL/kg


def value(code, d):
    if d in HOLIDAYS:
        return None
    base = LEVEL[d]
    if code.endswith("AOF.Y") or code.endswith("AGORT02"):
        return f"{base * 0.999:.2f}"
    if "USD" in code or code.endswith("D.KAP.Y") or code.endswith("KAP03"):
        return f"{base / 1000 * 31.1035 / 30:.2f}"
    return f"{base:.2f}"


def in_range(code, d):
    if code.startswith("TP.MK."):
        return d <= pd.Timestamp("2018-06-29")
    return d >= pd.Timestamp("2018-07-02")


class FakeEVDS:
    def __init__(self, jump=0.0):
        self.jump, self.requests = jump, []

    def __call__(self, url, headers):
        q = parse_qs(url.split(evds.BASE_URL, 1)[1], keep_blank_values=True)
        self.requests.append(q)
        if not q.get("endDate", [""])[0]:
            return 400, b'{"error":"endDate zorunlu"}'
        codes = q["series"][0].split("-")
        a = pd.to_datetime(q["startDate"][0], format="%d-%m-%Y")
        b = pd.to_datetime(q["endDate"][0], format="%d-%m-%Y")
        items = []
        for d in DAYS[(DAYS >= a) & (DAYS <= b)]:
            row = {"Tarih": d.strftime("%d-%m-%Y")}
            for c in codes:
                v = value(c, d) if in_range(c, d) else None
                if v is not None and self.jump and d == pd.Timestamp("2018-07-02"):
                    v = f"{float(v) * (1 + self.jump):.2f}"
                row[c.replace(".", "_")] = v
            items.append(row)
        items = items[-evds.ROW_LIMIT:]                       # gercek davranis: son 1000 satir
        return 200, json.dumps({"items": items, "totalCount": len(items)}).encode()


class TestChunking(unittest.TestCase):

    def test_truncation_is_detected_and_all_rows_recovered(self):
        c = evds.Client("k", get=FakeEVDS())
        items, chunks = evds.fetch_chunked(c, ["TP.ALTINPIYASA.KAP02"], "02-07-2018", "13-10-2026", chunk_days=10000)
        df = evds.parse_items(items, ["TP.ALTINPIYASA.KAP02"])
        expected = DAYS[(DAYS >= "2018-07-02") & (DAYS <= "2026-10-13")]
        self.assertEqual(len(df), len(expected))                         # kesilmeden tum gunler
        self.assertTrue(any(ch["split"] for ch in chunks))                # 1000'e ulasan parca bolundu
        self.assertTrue(all(ch["rows"] < evds.ROW_LIMIT for ch in chunks if not ch["split"]))

    def test_empty_end_defaults_to_today_and_server_error_detail_shown(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        g = FakeEVDS()
        c = evds.Client("k", get=g)
        evds.run_download(c, ["TP.ALTINPIYASA.KAP02"], "01-10-2026", "", tmp, datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(g.requests[-1]["endDate"][0], "14-10-2026")
        with self.assertRaisesRegex(evds.EVDSError, "endDate zorunlu"):
            c.data(["TP.ALTINPIYASA.KAP02"], "01-10-2026", "")

    def test_inconsistent_duplicates_raise(self):
        items = [{"Tarih": "02-07-2018", "TP_X": "1"}, {"Tarih": "02-07-2018", "TP_X": "2"}]
        with self.assertRaisesRegex(evds.EVDSError, "farkli degerler"):
            evds.parse_items(items, ["TP.X"])
        same = [{"Tarih": "02-07-2018", "TP_X": "1"}, {"Tarih": "02-07-2018", "TP_X": "1"}]
        self.assertEqual(len(evds.parse_items(same, ["TP.X"])), 1)


class TestPhysicalGold(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pg_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        exp = DAYS[DAYS >= "2015-01-02"]
        theo = pd.DataFrame({"Date": exp, "theoretical_gram_try": LEVEL[exp].to_numpy() / 1000 / 1.01,
                             "gold_usd_oz": LEVEL[exp].to_numpy() / 1000 * 31.1035 / 30})
        self.panel = self.tmp / "market_panel.csv"
        theo.to_csv(self.panel, index=False, date_format="%Y-%m-%d")

    def run_pg(self, jump=0.0):
        c = evds.Client("k", get=FakeEVDS(jump))
        with unittest.mock.patch.object(evds, "OUT_ROOT", self.tmp / "evds"):
            return pg.run(c, None, self.tmp / "out", self.panel, self.tmp / "yok.csv", self.now)

    def test_full_series_units_sources_and_holidays(self):
        r = self.run_pg()
        s, rep = r["series"], r["report"]
        all_days = DAYS[DAYS <= "2026-10-13"]
        self.assertEqual(len(s), len(all_days) - len(HOLIDAYS))
        self.assertEqual(rep["empty_days_dropped"], len(HOLIDAYS))
        first = min(d for d in all_days if d not in HOLIDAYS)
        self.assertEqual(rep["date_range"], [str(first.date()), "2026-10-13"])
        d = s.set_index("Date")
        x = pd.Timestamp("2020-03-02")
        self.assertAlmostEqual(d.loc[x, "kapanis_tl_gr"], round(LEVEL[x], 2) / 1000, places=6)   # TL/kg -> TL/gr
        self.assertEqual(d.loc["2018-06-29", "kaynak"], "IAB_arsiv")
        self.assertEqual(d.loc["2018-07-02", "kaynak"], "BIST_KMP")
        self.assertTrue(rep["boundary"]["within_tolerance"])

    def test_premium_vs_theoretical(self):
        c = self.run_pg()["report"]["comparisons"]["teorik_gunluk_yahoo"]
        self.assertAlmostEqual(c["premium_median_pct"], 1.0, delta=0.01)          # fiziki = teorik x 1.01
        self.assertAlmostEqual(c["usd_ons_diff_median_pct"], 0.0, delta=0.01)
        self.assertEqual(set(c["yearly"]), {str(y) for y in range(2015, 2027)})

    def test_boundary_jump_flagged(self):
        b = self.run_pg(jump=0.10)["report"]["boundary"]
        self.assertFalse(b["within_tolerance"])
        self.assertAlmostEqual(b["change_pct"], 10.0, delta=0.5)

    def test_rebuild_from_snapshots_matches(self):
        a = self.run_pg()
        snaps = sorted((self.tmp / "evds").iterdir())
        b = pg.run(None, [str(s) for s in snaps], self.tmp / "out2", self.panel, self.tmp / "yok.csv", self.now)
        pd.testing.assert_frame_equal(a["series"].reset_index(drop=True), b["series"].reset_index(drop=True),
                                      check_dtype=False)

    def test_existing_theoretical_files_untouched(self):
        before = self.panel.read_bytes()
        self.run_pg()
        self.assertEqual(self.panel.read_bytes(), before)
        self.assertTrue((self.tmp / "out" / "physical_gold_daily.csv").exists())


import unittest.mock  # noqa: E402
import hashlib  # noqa: E402

_REAL = [PROJECT_ROOT / "data" / "raw" / "evds", PROJECT_ROOT / "data" / "processed" / "physical"]
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
