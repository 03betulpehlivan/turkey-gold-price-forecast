"""
Zamanlama Kontrolu: Bilesen-Gecikme sinyali olcum yan etkisi mi?
================================================================
Kullanim (proje kokunden):
  python -m src.analysis.timing_check

Soru: Bugunku ons getirisinin yarinki USD/TRY getirisini "tahmin etmesi"
(D4: -0.096) gercek bir piyasa gecikmesi mi, yoksa Yahoo'nun gunluk USD/TRY
kapanisinin ABD seansi bitmeden olusmasindan dogan bir saat damgasi kaymasi mi?

Ayirt edici ongoru (asenkronluk hipotezi, H_async)
--------------------------------------------------
  USD/TRY gunluk kapanisi ABD kapanisindan ONCE olusuyorsa, ABD'de gec kapanan
  TUM varliklarin t gunu getirisi, t+1 gunu USD/TRY getirisiyle EKONOMIK OLARAK
  BEKLENEN YONDE iliskili gorunur (ABD seansindaki dolar haberi kura ertesi gun
  yansir). Turkiye saatiyle kapanan BIST 100 ise bu etkiyi gostermemelidir (kontrol).

  Beklenen isaretler, corr(x[t], usdtry[t+1]):
    dxy +   (dolar guclenir -> TL zayiflar)      gold_usd -  (altin yukselir ~ dolar zayif)
    silver - sp500 -  (risk istahi -> TL guclenir) vix +     (risk kacisi -> TL zayiflar)

Karar kurali (sonuclardan ONCE sabit)
-------------------------------------
  "H_async ile TUTARLI": 5 ABD varligindan EN AZ 3'u beklenen isarette ve
  |corr| > 1.96/sqrt(n) iken BIST 100 kontrolu bandin icinde.
  "TUTARSIZ": ABD varliklarindan beklenen isarette bant disinda olan 1 veya hic yoksa.
  Aksi halde "KARARSIZ".
  Bu bir teshistir (kanit derecesi); Bilesen-Gecikme zaten model seciminin disindadir.

Donem: yalnizca TEST_START oncesi (2015 .. 2023-10-25). 2026 ve test donemi kullanilmaz.
"""

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.analysis import data_diagnostics as dd
from src.models import train as tr

OUT_DIR = dd.OUT_DIR / "timing"
US_ASSETS = {"dxy": +1, "gold_usd": -1, "silver": -1, "sp500": -1, "vix": +1}
CONTROL = "bist100"
MIN_CONSISTENT = 3


def lead_lag_table(ret: pd.DataFrame) -> pd.DataFrame:
    """Her varlik icin ayni gun ve ertesi gun USD/TRY ile korelasyon."""
    rows = []
    for name, sign in list(US_ASSETS.items()) + [(CONTROL, 0)]:
        x, fx = ret[f"r_{name}"], ret["r_usdtry"]
        same, n0 = dd.cross_corr(x, fx, 0)
        nxt, n1 = dd.cross_corr(x, fx, 1)
        prev, _ = dd.cross_corr(x, fx, -1)
        band = 1.96 / math.sqrt(n1) if n1 else np.nan
        rows.append({"asset": name, "role": "kontrol" if name == CONTROL else "ABD_kapanis",
                     "expected_sign_next_day": sign, "corr_same_day": same, "corr_next_day_fx": nxt,
                     "corr_prev_day_fx": prev, "n": n1, "band_95": band,
                     "outside_band": bool(abs(nxt) > band),
                     "consistent": bool(sign != 0 and np.sign(nxt) == sign and abs(nxt) > band)})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame) -> dict:
    us = table[table["role"] == "ABD_kapanis"]
    k = int(us["consistent"].sum())
    control_quiet = not bool(table.loc[table["asset"] == CONTROL, "outside_band"].iloc[0])
    if k >= MIN_CONSISTENT and control_quiet:
        label = "TUTARLI"
        text = ("ABD'de gec kapanan varliklar ertesi gunun kurunu beklenen yonde 'tahmin ediyor', BIST 100 "
                "etmiyor: Bilesen-Gecikme kazanci buyuk olasilikla saat damgasi kaymasindan dogan bir olcum "
                "yan etkisidir ve gercek piyasada kullanilamaz.")
    elif k <= 1:
        label = "TUTARSIZ"
        text = ("ABD varliklarinda ertesi gun etkisi yaygin degil: saat damgasi kaymasi bu sinyali "
                "aciklamiyor; gercek bir gecikme olabilir (ayri teyit gerekir).")
    else:
        label = "KARARSIZ"
        text = "Kanit karisik; sinyalin kaynagi belirlenemedi. Bilesen-Gecikme secim disinda kalmali."
    return {"label": label, "consistent_us_assets": k, "of": len(us), "control_quiet": control_quiet,
            "text": text}


def run(panel_path: Path = dd.PANEL_PATH, out_dir: Path = OUT_DIR) -> dict:
    panel = dd.load_panel(panel_path)                         # 2026 burada atilir
    ret = dd.upto(dd.add_returns(panel), dd.ANALYSIS_END)     # test oncesi
    missing = [c for c in [f"r_{a}" for a in list(US_ASSETS) + [CONTROL]] + ["r_usdtry"] if c not in ret]
    if missing:
        raise ValueError(f"Panelde gerekli seriler yok: {missing}")
    table = lead_lag_table(ret)
    v = verdict(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "lead_lag.csv", index=False)
    summary = {"period": [str(ret["Date"].min().date()), str(ret["Date"].max().date())],
               "rule": {"min_consistent": MIN_CONSISTENT, "of": len(US_ASSETS), "control": CONTROL,
                        "expected_signs": US_ASSETS},
               "verdict": v, "test_period_used": False, "holdout_2026_used": False}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return {"table": table, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bilesen-Gecikme zamanlama kontrolu (salt okunur).")
    ap.add_argument("--panel", type=Path, default=dd.PANEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 70 + "\nZAMANLAMA KONTROLU (salt okunur)\n" + "=" * 70)
    try:
        res = run(args.panel, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"  HATA: {e}")
        return 1
    s = res["summary"]
    pd.set_option("display.width", 160)
    print(f"  Donem: {s['period'][0]} .. {s['period'][1]} (test oncesi)\n")
    print(res["table"].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    v = s["verdict"]
    print(f"\n  Beklenen isarette ve bant disinda olan ABD varligi: {v['consistent_us_assets']}/{v['of']} "
          f"| BIST 100 kontrolu sessiz: {v['control_quiet']}")
    print(f"  SONUC: {v['label']}\n  {v['text']}")
    print(f"\n  Ciktilar: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
