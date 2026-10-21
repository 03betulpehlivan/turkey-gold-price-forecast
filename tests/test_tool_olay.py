"""
Olay takvimi modeli testleri: src/tool/yarisma.py (GARCH-t+olay, olay_gunleri, olay_yukle)
Calistirma: python -m unittest tests.test_tool_olay -v
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tool import yarisma as y  # noqa: E402


def olay_dunyasi(etki=3.0, seed=5, end="2018-12-31"):
    """GARCH dunyasi; her ~20 islem gununde bir olay. Olay gunu ve ertesi gun varyans 'etki' katina cikar."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2008-01-02", end)
    olay = d[::20]
    ev = np.zeros(len(d), bool)
    for t in olay:
        i = d.get_loc(t)
        ev[i] = True
        if i + 1 < len(d):
            ev[i + 1] = True
    v, r = 1e-4, np.empty(len(d))
    for i in range(len(d)):
        s = np.sqrt(v * (etki if ev[i] else 1.0))
        r[i] = s * rng.standard_t(6) * np.sqrt(4 / 6)
        v = 2e-6 + 0.06 * r[i] ** 2 + 0.9 * v
    return pd.Series(50 * np.exp(np.cumsum(r)), index=d), pd.DataFrame({"tarih": olay, "tur": "FED"})


class TestOlay(unittest.TestCase):
    def test_load_and_window(self):
        tmp = Path(tempfile.mkdtemp(prefix="olay_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = tmp / "o.csv"
        p.write_text("# aciklama\ntarih,tur,kaynak\n2026-10-24,FED,x\n2026-10-21,TCMB_PPK,y\n", encoding="utf-8")
        o = y.olay_yukle(p)
        self.assertEqual(len(o), 2)
        p.write_text("# bos\ntarih,tur,kaynak\n", encoding="utf-8")
        self.assertIsNone(y.olay_yukle(p))
        prices = pd.Series(1.0, index=pd.bdate_range("2026-10-19", "2026-10-30"))
        w = y.olay_gunleri(y.hazirla(prices), o)
        gun = [str(t.date()) for t in prices.index[w]]
        self.assertEqual(gun, ["2026-10-21", "2026-10-22", "2026-10-26", "2026-10-27"])     # cumartesi -> pazartesi

    def test_detects_event_volatility(self):
        warnings.simplefilter("ignore")
        p, o = olay_dunyasi(etki=3.0)
        res = y.yarisma(p, ufuklar=(1,), modeller=("GARCH-t", "GARCH-t+olay", "Sabit aralik"), olaylar=o)
        s = res["ufuklar"]["1"]["secim"]
        self.assertLess(s["GARCH-t+olay"]["puan80"] + s["GARCH-t+olay"]["puan95"], s["GARCH-t"]["puan80"] + s["GARCH-t"]["puan95"])
        d = y.hazirla(p)
        idx = y.kokenler(d, 1)
        f = y.m_garch_olay(d, idx, 1, {"idx_all": idx}, o)
        self.assertGreater(np.median(f["kappa"][f["kappa"] > 0]), 0.8)

    def test_no_effect_no_gain(self):
        warnings.simplefilter("ignore")
        p, o = olay_dunyasi(etki=1.0, seed=9)
        d = y.hazirla(p)
        idx = y.kokenler(d, 1)
        f = y.m_garch_olay(d, idx, 1, {"idx_all": idx}, o)
        self.assertLess(np.mean(f["kappa"]), 0.3)
        self.assertNotIn("GARCH-t+olay", y.tahminler(d, 1, ("GARCH-t+olay",))["Q"])          # takvim yoksa girmez

    def test_kappa_uses_only_realized_returns(self):
        warnings.simplefilter("ignore")
        p, o = olay_dunyasi(etki=3.0, end="2015-12-31")
        h = 3
        d = y.hazirla(p)
        a = y.tahminler(d, h, ("GARCH-t+olay",), olaylar=o)
        per = pd.DatetimeIndex(d["tarih"][a["idx"]]).to_period("M")
        j = int(np.flatnonzero(per == pd.Period("2014-05"))[0])
        b0 = int(a["idx"][j])
        for k_ in range(1, h + 1):
            p2 = p.copy()
            p2.iloc[b0 + k_] *= 1.10
            b = y.tahminler(y.hazirla(p2), h, ("GARCH-t+olay",), olaylar=o)
            np.testing.assert_allclose(a["Q"]["GARCH-t+olay"][j], b["Q"]["GARCH-t+olay"][j], rtol=1e-9, err_msg=str(k_))
            d2 = y.hazirla(p2)
            idx = y.kokenler(d, h)
            ka = y.m_garch_olay(d, idx, h, {"idx_all": idx}, o)["kappa"]
            kb = y.m_garch_olay(d2, idx, h, {"idx_all": idx}, o)["kappa"]
            jj = int(np.flatnonzero(idx == b0)[0])
            self.assertAlmostEqual(ka[jj], kb[jj], places=12, msg=f"kappa +{k_}")                # o ayin kappa'si


if __name__ == "__main__":
    unittest.main(verbosity=2)
