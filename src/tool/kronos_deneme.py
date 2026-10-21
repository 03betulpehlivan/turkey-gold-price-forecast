"""
Kronos Denemesi: finans icin onceden egitilmis temel model, ARIMA-GARCH'a karsi
==============================================================================
Kurulum (bir kez):
  git clone https://github.com/shiyu-coder/Kronos.git %USERPROFILE%\\Kronos
  setx KRONOS_DIR "%USERPROFILE%\\Kronos"          (yeni terminalde gecerli olur)
  pip install einops safetensors
Kullanim (proje kokunden):
  python -m src.tool.kronos_deneme                  # Kronos-mini, haftalik tahmin gunleri, 20 senaryo
  python -m src.tool.kronos_deneme --boyut small --ornek 30

Duzen: yarisma ile ayni tahmin gunlerinden her 5. gun (haftalik). Her gun icin Kronos'a o gune kadarki son 512
  gunun fiyati verilir ve 5 gunluk 20 bagimsiz senaryo uretilir; 1, 3 ve 5 gun sonrasinin yuzdelikleri bu
  senaryolardan hesaplanir. Ayni gunlerde ARIMA-GARCH (yarismanin 1 gun kazanani) ayni olcutle puanlanir.
Veri siniri (onemli): fiziki gram altin icin yalnizca KAPANIS fiyati var. Kronos mum grafikleriyle (acilis, en
  yuksek, en dusuk, kapanis) egitildi; burada dordu de kapanisa esitlenir. Sonuc bu uyumsuzlugu da yansitir.
Ciktilar: research/tool/kronos_deneme.json. Secim 2012-2022 ve final 2023-2026 ayri; DM testi ARIMA-GARCH'a gore.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.tool import karne as k
from src.tool import yarisma as yr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "tool" / "kronos_deneme.json"
BOYUT = {"mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
         "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512)}
GERI, ADIM, ORNEK, UFUK = 512, 5, 20, 5
TOP_P = 1.0      # belirsizlik olcerken kuyruklar kesilmez (Kronos'un varsayilan 0.9'u araligi yapay daraltir)


def yukle(boyut="mini"):
    kd = os.environ.get("KRONOS_DIR")
    if not kd or not (Path(kd) / "model").exists():
        raise RuntimeError("KRONOS_DIR tanimli degil ya da Kronos deposu yok (docstring'deki kurulum adimlari).")
    if kd not in sys.path:
        sys.path.insert(0, kd)
    from model import Kronos, KronosTokenizer, KronosPredictor
    m_id, t_id, ctx = BOYUT[boyut]
    return KronosPredictor(Kronos.from_pretrained(m_id), KronosTokenizer.from_pretrained(t_id), max_context=ctx)


def kronos_yuzdelik(predictor, prices: pd.Series, idx: np.ndarray, ornek=ORNEK, parti=64, tohum=0, ilerleme=True,
                    top_p=TOP_P) -> dict:
    """Her bilgi gunu i icin son GERI gunun (i dahil) fiyatindan UFUK gunluk 'ornek' senaryo; h = 1..UFUK
    icin log getiri yuzdelikleri (yarisma.QS). Bilgi gununden sonraki hicbir fiyat kullanilmaz."""
    try:
        import torch
        tohumla = torch.manual_seed
    except ImportError:                                               # yalnizca testlerde (sahte tahminci)
        tohumla = np.random.seed
    v, tarih = prices.to_numpy(float), prices.index
    yollar = np.empty((len(idx), ornek, UFUK))
    t0 = time.time()
    for a in range(0, len(idx), parti):
        blok = idx[a:a + parti]
        dfs, xts, yts = [], [], []
        for i in blok:
            p = v[i - GERI + 1: i + 1]
            dfs.append(pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 0.0, "amount": 0.0}))
            xts.append(pd.Series(tarih[i - GERI + 1: i + 1]))
            yts.append(pd.Series(pd.bdate_range(tarih[i] + pd.offsets.BDay(1), periods=UFUK)))
        for s in range(ornek):
            tohumla(tohum + s)
            out = predictor.predict_batch(dfs, xts, yts, pred_len=UFUK, T=1.0, top_p=top_p, sample_count=1, verbose=False)
            for j, df in enumerate(out):
                yollar[a + j, s] = df["close"].to_numpy(float)[:UFUK]
        if ilerleme and a == 0:
            sure = (time.time() - t0) * len(idx) / max(len(blok), 1)
            print(f"  Tahmini sure: yaklasik {sure / 60:.0f} dakika ({len(idx)} gun, {ornek} senaryo)", flush=True)
    P = v[idx][:, None]
    return {h: np.quantile(np.log(yollar[:, :, h - 1] / P), yr.QS, axis=1).T for h in range(1, UFUK + 1)}


def deneme(prices: pd.Series, predictor, ornek=ORNEK, adim=ADIM, ufuklar=(1, 3, 5), top_p=TOP_P) -> dict:
    d = yr.hazirla(prices)
    idx = yr.kokenler(d, max(ufuklar))
    idx = idx[(idx >= GERI) & (np.arange(len(idx)) % adim == 0)]
    kq = kronos_yuzdelik(predictor, prices, idx, ornek, top_p=top_p)
    sonuc = {}
    for h in ufuklar:
        tam = yr.kokenler(d, h)
        ag = yr.tahminler(d, h, ("ARIMA-GARCH",))["Q"]["ARIMA-GARCH"]
        ag = ag[np.searchsorted(tam, idx)]
        y = d["lp"][idx + h] - d["lp"][idx]
        bilgi, hedef = d["tarih"][idx], d["tarih"][idx + h]
        secim = np.asarray((bilgi >= pd.Timestamp(yr.SECIM[0])) & (hedef < pd.Timestamp(yr.SECIM[1])))
        final = np.asarray(bilgi >= pd.Timestamp(yr.FINAL))
        r = {}
        for ad, m in (("secim", secim), ("final", final)):
            if m.any():
                r[ad] = {"Kronos": yr.puanla(kq[h][m], y[m]), "ARIMA-GARCH": yr.puanla(ag[m], y[m]),
                         "kronos_arima_garch_a_gore": yr.anlamlilik(kq[h][m], ag[m], y[m])}
        sonuc[str(h)] = r
    return {"ornek": ornek, "adim_gun": adim, "geri_gun": GERI, "top_p": top_p, "tahmin_gunu": int(len(idx)), "ufuklar": sonuc,
            "not": "fiziki altin icin yalnizca kapanis var; acilis/en yuksek/en dusuk kapanisa esitlendi"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kronos denemesi (ARIMA-GARCH'a karsi).")
    ap.add_argument("--boyut", choices=list(BOYUT), default="mini")
    ap.add_argument("--ornek", type=int, default=ORNEK)
    ap.add_argument("--adim", type=int, default=ADIM)
    ap.add_argument("--top_p", type=float, default=TOP_P, help="1.0 = kuyruklar kesilmez (onerilen)")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 74 + f"\nKRONOS DENEMESI - Kronos-{args.boyut}, her {args.adim}. gun, {args.ornek} senaryo, top_p {args.top_p}\n" + "=" * 74)
    res = deneme(k.load_series(), yukle(args.boyut), args.ornek, args.adim, top_p=args.top_p)
    res["boyut"], res["created_at_utc"] = args.boyut, datetime.now(timezone.utc).isoformat(timespec="seconds")
    for h, r in res["ufuklar"].items():
        for donem in ("secim", "final"):
            if donem not in r:
                continue
            a, b, s = r[donem]["Kronos"], r[donem]["ARIMA-GARCH"], r[donem]["kronos_arima_garch_a_gore"]
            print(f"  {h} gun, {donem:6s}: Kronos %80 {a['tutma80']:.1f} %95 {a['tutma95']:.1f} puan {a['puan80'] + a['puan95']:.2f}"
                  f" | ARIMA-GARCH %80 {b['tutma80']:.1f} %95 {b['tutma95']:.1f} puan {b['puan80'] + b['puan95']:.2f}"
                  f" | Kronos farki %{s['iyilesme_pct']:+.1f}, p {s['p_degeri']:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"kronos_deneme_{args.boyut}_adim{args.adim}_ornek{args.ornek}_topp{args.top_p:g}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n  Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
