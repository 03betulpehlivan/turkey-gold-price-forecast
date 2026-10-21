"""
Olay takvimi indirici testleri: src/data/olay_indir.py (ag erisimi yok; ornek sayfa parcalari)
Calistirma: python -m unittest tests.test_olay_indir -v
"""

from datetime import date
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import olay_indir as oi  # noqa: E402
from src.tool import yarisma as yr  # noqa: E402

GUNCEL = """
<h4><a id="1">2025 FOMC Meetings</a></h4>
<div class="fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">28-29</div></div>
<div class="fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col-xs-4">30-1*</div></div>
<h4><a id="2">2020 FOMC Meetings</a></h4>
<div class="fomc-meeting"><div class="fomc-meeting__month"><strong>March</strong></div>
<div class="fomc-meeting__date">3 (unscheduled)</div></div>
<div class="fomc-meeting"><div class="fomc-meeting__month"><strong>March</strong></div>
<div class="fomc-meeting__date">15</div></div>
"""
TARIHSEL = """<div class="panel-heading"><h5>January 27-28 Meeting - 2015</h5></div> ...
<h5>March 17-18 Meeting - 2015</h5><h5>October 15 (unscheduled) Meeting - 2015</h5>
<h5>November 2 Conference Call - 2015</h5><h5>December 15-16 Meeting - 2015</h5><h5>June 4 Meeting - 2014</h5>"""


class TestParse(unittest.TestCase):
    def test_decision_day(self):
        self.assertEqual(oi.karar_gunu(2025, "January", "28-29"), date(2025, 1, 29))
        self.assertEqual(oi.karar_gunu(2025, "Apr/May", "30-1*"), date(2025, 5, 1))
        self.assertEqual(oi.karar_gunu(2020, "March", "15"), date(2020, 3, 15))
        self.assertEqual(oi.karar_gunu(2019, "Oct/Nov", "29-30"), date(2019, 11, 30))

    def test_current_page_skips_unscheduled(self):
        t = oi.ayristir_guncel(GUNCEL)
        self.assertIn(date(2025, 1, 29), t)
        self.assertIn(date(2025, 5, 1), t)
        self.assertNotIn(date(2020, 3, 3), t)

    def test_historical_page(self):
        t = oi.ayristir_tarihsel(TARIHSEL, 2015)
        self.assertEqual(t, [date(2015, 1, 28), date(2015, 3, 18), date(2015, 12, 16)])   # plansiz ve telefon yok

    def test_fetch_combines_and_counts(self):
        sayfalar = {oi.GUNCEL: GUNCEL, oi.TARIHSEL.format(yil=2015): TARIHSEL}
        r = oi.fed_tarihleri(2015, 2025, getir=lambda u: sayfalar.get(u, ""))
        self.assertEqual(r["yillar"][2015], 3)
        self.assertEqual(r["yillar"][2025], 2)
        self.assertEqual(r["yillar"][2016], 0)
        self.assertEqual(r["kaynak"][date(2015, 1, 28)], oi.TARIHSEL.format(yil=2015))


class TestWrite(unittest.TestCase):
    def test_merge_keeps_other_types_and_loads(self):
        tmp = Path(tempfile.mkdtemp(prefix="oi_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = tmp / "olay.csv"
        p.write_text(oi.BASLIK + "tarih,tur,kaynak\n2024-02-13,ABD_TUFE,https://www.bls.gov\n", encoding="utf-8")
        n = oi.takvime_ekle([date(2024, 1, 31), date(2024, 3, 20)], {date(2024, 1, 31): "u"}, path=p)
        self.assertEqual(n, 2)
        self.assertEqual(oi.takvime_ekle([date(2024, 1, 31)], {}, path=p), 0)          # ayni tarih iki kez yok
        o = yr.olay_yukle(p)
        self.assertEqual(len(o), 3)
        self.assertEqual(sorted(o["tur"]), ["ABD_TUFE", "FED", "FED"])
        self.assertTrue(p.read_text(encoding="utf-8").startswith("# Olay takvimi"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
