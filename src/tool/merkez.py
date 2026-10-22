"""
Merkez Tahmin: araliktan tek bir fiyat tahmini cikarilabilir mi?
===============================================================
Kullanim (proje kokunden):
  python -m src.tool.merkez                 # 1, 3 ve 5 gun; Chronos kuruluysa yarismanin kazananlariyla
  python -m src.tool.merkez --ufuk 1

Soru: modelin verdigi aralik (alt sinir, ust sinir, orta nokta) ile gerceklesen fiyat arasinda, tek bir fiyat
  tahmini icin kullanilabilecek anlamli bir iliski var mi?
Yontem:
  Her bilgi gunu i icin: bugunku fiyat P, modelin orta degeri (medyan), %80 araligin genisligi, araligin
  kaykinligi (ust kisim alt kisimdan ne kadar uzun) ve son gunun getirisi.
  Ozel formul, h gunluk log getiriyi bu bilgilerle aciklayan dogrusal bir denklemdir (EKK); katsayilar YALNIZCA
  secim doneminde (2012-2022) tahmin edilir, Newey-West standart hatalariyla anlamlilik verilir.
  Karsilastirilanlar: "yarin = bugun", araligin ortasi ((alt + ust) / 2), modelin orta degeri ve ozel formul.
  Final donemde (2023-2026) ortalama hata (TL ve %), yon isabeti, yillara gore dokum ve "yarin = bugun"e gore
  DM testi raporlanir. Ozel formulun final donemi hic gormedigi bir sinavdir.
Ciktilar: research/tool/merkez_tahmin.json
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.models import final_exam as fe
from src.tool import karne as k
from src.tool import yarisma as yr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "tool" / "merkez_tahmin.json"
OZELLIK = ("sabit", "model_orta", "son_getiri", "genislik", "kayiklik")
YONTEM = ("Yarin = bugun", "Araligin ortasi", "Modelin orta degeri", "Ozel formul")


def ozellikler(d: dict, idx: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Q: log getiri yuzdelikleri (yr.QS sirasiyla). Hepsi bilgi gunu i'de bilinir."""
    med = Q[:, 2]
    gen = Q[:, 3] - Q[:, 1]
    kay = (Q[:, 3] - med) - (med - Q[:, 1])
    son = np.nan_to_num(d["r"][idx], nan=0.0)
    return np.column_stack([np.ones(len(idx)), med, son, gen, kay])


def ekk_nw(X: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """Kucuk kareler + Newey-West (Bartlett) standart hatalari."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    t = beta / se
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return {"beta": beta, "se": se, "t": t, "p": p}


def tahminler_h(d, h, model, chronos_ok):
    kullanilan = model if (chronos_ok or model not in ("Chronos", "Birlesim-2")) else "ARIMA-GARCH"
    istek = (kullanilan,) if kullanilan != "Birlesim-2" else ("ARIMA-GARCH", "Chronos", "Birlesim-2")
    t = yr.tahminler(d, h, istek)
    return t["idx"], t["Q"][kullanilan], kullanilan


def calisma(prices: pd.Series, h: int, model: str, chronos_ok: bool) -> dict:
    d = yr.hazirla(prices)
    idx, Q, kullanilan = tahminler_h(d, h, model, chronos_ok)
    y = d["lp"][idx + h] - d["lp"][idx]
    P = np.exp(d["lp"][idx])
    bilgi, hedef = d["tarih"][idx], d["tarih"][idx + h]
    secim = np.asarray((bilgi >= pd.Timestamp(yr.SECIM[0])) & (hedef < pd.Timestamp(yr.SECIM[1])))
    final = np.asarray(bilgi >= pd.Timestamp(yr.FINAL))
    X = ozellikler(d, idx, Q)
    reg = ekk_nw(X[secim], y[secim], lag=max(h, 5))
    orta = np.log((np.exp(Q[:, 1]) + np.exp(Q[:, 3])) / 2)                # (alt + ust) / 2, log getiri olarak
    tah = {"Yarin = bugun": np.zeros(len(idx)), "Araligin ortasi": orta, "Modelin orta degeri": Q[:, 2],
           "Ozel formul": X @ reg["beta"]}
    son_fiyat = float(prices.iloc[-1])

    def ozet(m):
        out = {}
        for ad, mask in (("secim", secim), ("final", final)):
            if not mask.any():
                continue
            hata = np.abs(np.exp(y[mask]) - np.exp(tah[m][mask]))          # fiyat oraninda mutlak hata
            yon_ok = np.sign(tah[m][mask]) == np.sign(y[mask])
            yon_ok = yon_ok[(tah[m][mask] != 0) & (y[mask] != 0)]
            r = {"gun": int(mask.sum()), "hata_pct": float(100 * hata.mean()), "hata_tl": float(son_fiyat * hata.mean()),
                 "yon_isabeti_pct": float(100 * yon_ok.mean()) if len(yon_ok) else None}
            if m != "Yarin = bugun":
                h0 = np.abs(np.exp(y[mask]) - 1.0)
                dm = fe.dm_hln(hata, h0)
                r["yarin_bugune_gore_pct"] = float(100 * (1 - hata.mean() / h0.mean()))
                r["p_degeri"] = dm["p_value"]
            out[ad] = r
        return out

    yillar = pd.DatetimeIndex(bilgi).year
    yil_tab = []
    for Y in np.unique(yillar):
        m_ = yillar == Y
        row = {"yil": int(Y), "gun": int(m_.sum())}
        for m in YONTEM:
            row[m] = float(son_fiyat * np.mean(np.abs(np.exp(y[m_]) - np.exp(tah[m][m_]))))
        yil_tab.append(row)
    return {"ufuk_gun": h, "model": kullanilan, "fiyat_olcegi_tl": son_fiyat,
            "formul": {"ozellikler": list(OZELLIK), "katsayi": reg["beta"].tolist(), "std_hata": reg["se"].tolist(),
                       "t": reg["t"].tolist(), "p": reg["p"].tolist(), "egitim": "secim donemi 2012-2022"},
            "yontemler": {m: ozet(m) for m in YONTEM}, "yillar": yil_tab}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merkez tahmin calismasi.")
    ap.add_argument("--ufuk", default="1,3,5")
    args = ap.parse_args([] if argv is None else argv)
    res_y = json.loads((ROOT / "research" / "tool" / "yarisma_1-3-5_chronos.json").read_text(encoding="utf-8"))
    ok = yr.chronos_var()
    prices = k.load_series()
    print("=" * 76 + "\nMERKEZ TAHMIN - araliktan tek fiyat tahmini\n" + "=" * 76)
    sonuc = {}
    for h in (int(x) for x in args.ufuk.split(",")):
        r = calisma(prices, h, res_y["ufuklar"][str(h)]["kazanan"], ok)
        sonuc[str(h)] = r
        f = r["formul"]
        print(f"\n  UFUK {h} GUN | model {r['model']} | formul katsayilari (2012-2022):")
        for ad, b, t, p in zip(f["ozellikler"], f["katsayi"], f["t"], f["p"]):
            print(f"    {ad:12s} {b:+.5f}  t {t:+.2f}  p {p:.3f}{'  <- anlamli' if p < 0.05 and ad != 'sabit' else ''}")
        print(f"  {'yontem':22s} {'secim hata TL':>14s} {'final hata TL':>14s} {'final yon %':>12s} {'yarin=bugune gore':>18s} {'p':>8s}")
        for m in YONTEM:
            s, fi = r["yontemler"][m].get("secim", {}), r["yontemler"][m].get("final", {})
            yb = "" if m == "Yarin = bugun" else f"%{fi['yarin_bugune_gore_pct']:+.1f}"
            pp = "" if m == "Yarin = bugun" else f"{fi['p_degeri']:.4f}"
            yon = "-" if fi.get("yon_isabeti_pct") is None else f"{fi['yon_isabeti_pct']:.1f}"
            print(f"  {m:22s} {s['hata_tl']:14.1f} {fi['hata_tl']:14.1f} {yon:>12s} {yb:>18s} {pp:>8s}")
    out = {"ufuklar": sonuc, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "not": "hata TL, son kapanis fiyati olceginde; secim 2012-2022 (formul burada egitildi), final 2023-2026"}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n  Yazildi: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
