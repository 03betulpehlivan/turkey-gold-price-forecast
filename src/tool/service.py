"""
Pano icin tahmin servisi (salt okunur)
=====================================
Web panosunun ana ekranini besler:
  kart()      bir sonraki islem gunu icin aralik tahmini ve hareketli gun olasiligi
  bu_yil()    icinde bulunulan yilin gunluk tahminleri ve gerceklesen fiyatlar (gercek, ileriye donuk sinama)
  karne()     kayitli basari karnesi (research/tool/), testi 2018-2022
  ufuklar()   1, 3 ve 5 gun sonrasi icin araliklar; her ufukta model yarismasinin kazanani (research/tool/yarisma_*.json)
  hedef_gun() h islem gunu sonrasi (hafta sonu ve resmi tatiller atlanir; yalnizca ekran etiketi)
  yarisma()   yarisma ozeti: modeller, kazananlar, hic gorulmemis 2023-2026 sonuclari ve anlamlilik;
              karsilastirma icin 'Sabit aralik' yonteminin final sonucu ve yillik tutma/genislik degerleri
Model ve pencere onceden sabittir: GARCH-t + ACI, pencere dogrulamada secilen (research/tool/pencere_*.json).
Hesaplar gunluk fiyat dosyasi degismedikce onbellekte tutulur. Hicbir dosyaya yazmaz.
"""

from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from src.tool import karne as k
from src.tool import yarisma as yr

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "research" / "tool"
EGITIM, TEST = (2012, 2017), (2018, 2022)
VARSAYILAN_PENCERE = 750
SABIT = "Sabit aralik"                                    # yarismadaki basit karsilastirma yontemi

# Borsa Istanbul'un TAM GUN kapali oldugu hafta ici resmi tatiller. Yalnizca EKRANDAKI hedef tarih etiketi ve
# karnede "hangi gunun kapanisi bekleniyor" bilgisi icin kullanilir; tahmin ve puanlama her zaman fiyat
# dosyasindaki gercek islem gunlerini sayar. Arife (yarim gun) islem gunu sayilir.
# 2027 dini bayramlari (Ramazan, Kurban) resmi takvim aciklaninca eklenmeli.
TATILLER = frozenset({
    "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
    "2025-07-15", "2025-10-29",
    "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
    "2026-07-15", "2026-10-29",
    "2027-01-01", "2027-04-23", "2027-05-19", "2027-07-15", "2027-08-30", "2027-10-29",
})


def hedef_gun(son, h: int) -> str:
    """son tarihten sonraki h. islem gunu: hafta sonu ve TATILLER atlanir."""
    t, n = pd.Timestamp(son), 0
    while n < int(h):
        t += pd.Timedelta(days=1)
        if t.weekday() < 5 and str(t.date()) not in TATILLER:
            n += 1
    return str(t.date())


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


def yarisma_dosyasi(tool_dir: Path = None) -> Path:
    tool_dir = Path(tool_dir or TOOL_DIR)
    for name in ("yarisma_1-3-5_chronos.json", "yarisma_1-3-5.json"):
        if (tool_dir / name).exists():
            return tool_dir / name
    raise FileNotFoundError("Yarisma sonucu yok (python -m src.tool.yarisma --chronos)")


def _yarisma_json(tool_dir: Path = None) -> dict:
    return json.loads(yarisma_dosyasi(tool_dir).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _ufuklar(key, dosya, chronos_ok):
    prices = k.load_series(Path(key[0]))
    res = json.loads(Path(dosya).read_text(encoding="utf-8"))
    panel = yr.panel_yukle()
    out = []
    for h, u in sorted(res["ufuklar"].items(), key=lambda kv: int(kv[0])):
        c = yr.canli(prices, int(h), u["kazanan"], panel=panel, chronos_ok=chronos_ok)
        f = u["final"].get(u["kazanan"], {}) if u.get("final") else {}
        c["final"] = {"tutma80": f.get("tutma80"), "tutma95": f.get("tutma95"), "gun": f.get("gun")}
        c["hedef_gun_yaklasik"] = hedef_gun(prices.index[-1], int(h))          # tatiller atlanir (yalnizca etiket)
        out.append(c)
    s = out[0]["gunluk_sigma_pct"] if out else float("nan")
    hareket = float(100 * 2 * (1 - float(k._cdf(1.0 / s, np.nan)))) if s and np.isfinite(s) else None
    return {"bilgi_gunu": str(prices.index[-1].date()), "son_kapanis": float(prices.iloc[-1]), "ufuklar": out,
            "hareketli_olasiligi_pct": hareket, "esik_pct": 1.0, "kaynak": f"research/tool/{Path(dosya).name}"}


def ufuklar() -> dict:
    return _clean(dict(_ufuklar(_data_key(), str(yarisma_dosyasi()), yr.chronos_var())))


@lru_cache(maxsize=4)
def _bu_yil_yarisma(key, dosya, chronos_ok):
    prices = k.load_series(Path(key[0]))
    res = json.loads(Path(dosya).read_text(encoding="utf-8"))
    model = res["ufuklar"]["1"]["kazanan"]
    y = prices.index[-1].year
    g = yr.gunluk_seri(prices, 1, model, f"{y}-01-01", chronos_ok=chronos_ok)
    g = g[pd.DatetimeIndex(g["tarih"]).year == y]
    gercek = g["gercek"].to_numpy(float)
    in80 = (gercek >= g["lo80"].to_numpy()) & (gercek <= g["hi80"].to_numpy())
    in95 = (gercek >= g["lo95"].to_numpy()) & (gercek <= g["hi95"].to_numpy())
    rows = [{"tarih": str(pd.Timestamp(a).date()), "gercek": float(b), "lo80": float(c), "hi80": float(dd), "lo95": float(e), "hi95": float(f)}
            for a, b, c, dd, e, f in zip(g["tarih"], gercek, g["lo80"], g["hi80"], g["lo95"], g["hi95"])]
    return {"yil": int(y), "gun": int(len(g)), "aralik80_tuttu": int(in80.sum()), "aralik95_tuttu": int(in95.sum()),
            "aralik80_pct": float(100 * in80.mean()) if len(g) else None, "aralik95_pct": float(100 * in95.mean()) if len(g) else None,
            "model": str(g["model"].iat[0]) if len(g) else model, "gunler": rows}


def bu_yil_yarisma() -> dict:
    return _clean(dict(_bu_yil_yarisma(_data_key(), str(yarisma_dosyasi()), yr.chronos_var())))


@lru_cache(maxsize=4)
def _canli_karne(key, log_key):
    from src.tool import canli_karne as ck
    prices = k.load_series(Path(key[0]))
    d = ck.karne(prices, Path(log_key[0]))
    d["son_fiyat_gunu"] = str(prices.index[-1].date())                      # en son yayimlanan kapanis
    for r in d["son_kayitlar"]:                                              # bekleyen satir: hangi gunun kapanisi?
        r["beklenen_gun"] = None if r["gercek"] is not None else hedef_gun(r["bilgi_gunu"], r["ufuk_gun"])
    return d


def canli_karne() -> dict:
    from src.tool import canli_karne as ck
    p = ck.LOG
    log_key = (str(p), p.stat().st_mtime_ns if p.exists() else 0)
    return _clean(dict(_canli_karne(_data_key(), log_key)))


def merkez(tool_dir: Path = None) -> dict:
    """Merkez tahmin (araligin ortasi) icin hic gorulmemis 2023-2026 ortalama hatasi ve 'yarin = bugun' hatasi."""
    p = Path(tool_dir or TOOL_DIR) / "merkez_tahmin.json"
    if not p.exists():
        raise FileNotFoundError("Merkez tahmin sonucu yok (python -m src.tool.merkez)")
    d = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for h, u in sorted(d["ufuklar"].items(), key=lambda kv: int(kv[0])):
        f = u["yontemler"]["Araligin ortasi"]["final"]
        r = u["yontemler"]["Yarin = bugun"]["final"]
        out.append({"ufuk_gun": int(h), "hata_tl": f["hata_tl"], "hata_pct": f["hata_pct"], "rw_hata_tl": r["hata_tl"],
                    "iyilesme_pct": f["yarin_bugune_gore_pct"], "p_degeri": f["p_degeri"], "gun": f["gun"]})
    return _clean({"kaynak": "research/tool/merkez_tahmin.json", "ufuklar": out,
                   "formul": "merkez tahmin = (%80 araligin alt siniri + ust siniri) / 2"})


def yarisma(tool_dir: Path = None) -> dict:
    p = yarisma_dosyasi(tool_dir)
    res = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for h, u in sorted(res["ufuklar"].items(), key=lambda kv: int(kv[0])):
        w = u["kazanan"]
        modeller = sorted(({"model": m, "tutma80": s["tutma80"], "tutma95": s["tutma95"], "genislik80": s["genislik80"],
                            "puan": s["puan80"] + s["puan95"],
                            "uygun": bool(abs(s["tutma80"] - 80) <= yr.HEDEF[80] and abs(s["tutma95"] - 95) <= yr.HEDEF[95])}
                           for m, s in u["secim"].items()), key=lambda r: r["puan"])
        anl = u.get("anlamlilik_sabit_araliga_gore", {})
        sabit = {"final": u["final"].get(SABIT) if u.get("final") else None,
                 "yillar": u.get("yillar", {}).get(SABIT)}                   # karsilastirma: basit yontem
        out.append({"ufuk_gun": int(h), "kazanan": w, "modeller": modeller, "donem": u["donem"],
                    "final": u["final"].get(w) if u.get("final") else None, "anlamlilik": anl,
                    "yillar": u["yillar"][w], "sabit": sabit})
    return _clean({"kaynak": f"research/tool/{p.name}", "pencere_gun": res.get("pencere_gun"), "ufuklar": out,
                   "hedef_sapma": {"80": yr.HEDEF[80], "95": yr.HEDEF[95]},
                   "model_sayisi": len(out[0]["modeller"]) if out else 0})


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
