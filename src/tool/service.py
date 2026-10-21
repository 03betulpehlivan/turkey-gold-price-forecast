"""
Pano icin tahmin servisi (salt okunur)
=====================================
Web panosunun ana ekranini besler:
  kart()      bir sonraki islem gunu icin aralik tahmini ve hareketli gun olasiligi
  bu_yil()    icinde bulunulan yilin gunluk tahminleri ve gerceklesen fiyatlar (gercek, ileriye donuk sinama)
  karne()     kayitli basari karnesi (research/tool/), testi 2018-2022
Model ve pencere onceden sabittir: GARCH-t + ACI, pencere dogrulamada secilen (research/tool/pencere_*.json).
Hesaplar gunluk fiyat dosyasi degismedikce onbellekte tutulur. Hicbir dosyaya yazmaz.
"""

from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np

from src.tool import karne as k

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "research" / "tool"
EGITIM, TEST = (2012, 2017), (2018, 2022)
VARSAYILAN_PENCERE = 750


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {a: _clean(b) for a, b in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def secilen_pencere(tool_dir: Path = None) -> int:
    p = Path(tool_dir or TOOL_DIR) / f"pencere_{EGITIM[0]}-{EGITIM[1]}_{TEST[0]}-{TEST[1]}.json"
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["secilen_pencere"])
    except (OSError, KeyError, ValueError):
        return VARSAYILAN_PENCERE


def _data_key():
    p = next((q for q in k.PATHS if q.exists()), None)
    if p is None:
        raise FileNotFoundError("Fiyat verisi yok (data/processed/...physical_gold_daily.csv).")
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=4)
def _kart(key, pencere):
    prices = k.load_series(Path(key[0]))
    egitim = (EGITIM[0], prices.index[-1].year - 1)
    c = k.tomorrow(prices, egitim, k.ODAK, k.ESIK, pencere)
    c["sonraki_gun_tahmini"] = True
    return c


@lru_cache(maxsize=4)
def _bu_yil(key, pencere):
    prices = k.load_series(Path(key[0]))
    y = prices.index[-1].year
    g = k.gunluk(prices, (EGITIM[0], y - 1), (y, y), k.ODAK, pencere)
    gercek = g["gercek"].to_numpy(float)
    in80 = (gercek >= g["lo80"]) & (gercek <= g["hi80"])
    in95 = (gercek >= g["lo95"]) & (gercek <= g["hi95"])
    rows = [{"tarih": str(t.date()), "gercek": float(a), "lo80": float(b), "hi80": float(c), "lo95": float(d), "hi95": float(e)}
            for t, a, b, c, d, e in zip(g["T"], gercek, g["lo80"], g["hi80"], g["lo95"], g["hi95"])]
    return {"yil": int(y), "gun": int(len(g)), "aralik80_tuttu": int(in80.sum()), "aralik95_tuttu": int(in95.sum()),
            "aralik80_pct": float(100 * in80.mean()), "aralik95_pct": float(100 * in95.mean()),
            "gunler": rows}


def kart(pencere: int = None) -> dict:
    w = secilen_pencere() if pencere is None else pencere
    return _clean(dict(_kart(_data_key(), w)))


def bu_yil(pencere: int = None) -> dict:
    w = secilen_pencere() if pencere is None else pencere
    d = dict(_bu_yil(_data_key(), w))
    d["model"], d["pencere"] = k.ODAK, w
    return _clean(d)


def karne(tool_dir: Path = None) -> dict:
    tool_dir = Path(tool_dir or TOOL_DIR)
    w = secilen_pencere(tool_dir)
    p = tool_dir / f"karne_{EGITIM[0]}-{EGITIM[1]}_{TEST[0]}-{TEST[1]}_pencere{w}.json"
    if not p.exists():
        raise FileNotFoundError(f"Karne dosyasi yok: {p.name} (python -m src.tool.karne --egitim 2012-2017 --test 2018-2022 --pencere {w})")
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["modeller"]
    keep = [x for x in ("Yarin=bugun", "GARCH", k.ODAK) if x in m]
    return _clean({"egitim": d["egitim"], "test": d["test"], "test_gun": d["test_gun"], "pencere": w, "model": k.ODAK,
                   "esik_pct": d["esik_pct"], "kaynak": f"research/tool/{p.name}",
                   "karsilastirma": [{"model": x, "aralik80": m[x]["aralik80_tutma_pct"], "aralik95": m[x]["aralik95_tutma_pct"],
                                      "hata_pct": m[x]["ortalama_hata_pct"], "hata_tl": m[x]["ortalama_hata_tl"]} for x in keep],
                   "guvenilirlik": m[k.ODAK]["guvenilirlik"], "yillar": d["yillar"][k.ODAK],
                   "hareketli": {"isabet": m[k.ODAK]["hareketli_isabet_pct"], "hep_sakin": m[k.ODAK]["hep_sakin_isabet_pct"]}})
