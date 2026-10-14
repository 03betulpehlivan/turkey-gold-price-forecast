"""
Fiziki Altin Gunluk Serisi (TCMB EVDS) ve Teorik Fiyatla Karsilastirma
=====================================================================
Kullanim (proje kokunden; EVDS anahtari ortam degiskeninde):
  $env:EVDS_API_KEY = "..."
  python -m src.data.physical_gold                   # indir (parcali) + seri kur + teorikle karsilastir
  python -m src.data.physical_gold --snapshots data\\raw\\evds\\<arsiv> data\\raw\\evds\\<yeni>   # indirmeden yeniden kur

Kaynak seriler (EVDS katalogunda gercek calistirmada goruldu, 14.10.2026):
  Arsiv (27.07.1995 - 29.06.2018, is gunu):  TP.MK.T.KAP.Y kapanis TL/kg, TP.MK.T.AOF.Y agirlikli ort. TL/kg,
                                             TP.MK.D.KAP.Y kapanis USD/ons
  Guncel (02.07.2018 - bugun, is gunu):     TP.ALTINPIYASA.KAP02 kapanis TL/kg, .AGORT02 agirlikli ort. TL/kg,
                                             .KAP03 kapanis USD/ons
  TL/gr = TL/kg / 1000. KAP05 (TL/gr) seyrek oldugu icin KULLANILMAZ.

Ne olcer: Turkiye'deki organize fiziki altin piyasasinin (EVDS kategori adlarina gore Istanbul Altin
Borsasi arsivi ve Borsa Istanbul kiymetli madenler piyasasi; urun/ayar EVDS metaverisinden ayrica
dogrulanmali) gunluk TOPTAN fiyati. Kuyumcu alis/satis fiyati DEGILDIR (iscilik ve marj icermez).

Sifir degerler: arsivde bazi gunlerde (2009-2010, cogu yarim gun/arife) TL fiyati 0, USD/ons fiyati dolu
geldi (gercek calistirmada gozlendi). 0 "o gun TL islemi yok" olarak BOS sayilir, gunleri raporlanir.
Negatif deger veri hatasidir ve durdurur.

Teorik fiyat (ons x USD/TRY / 31.1035) ile AYNI DEGISKEN SAYILMAZ: ayri seri olarak tutulur.
Prim = fiziki / teorik - 1. Iki fiyatin saatleri farkli oldugundan gunluk prim zaman farkini da icerir.

Cikti (mevcut dosyalara dokunulmaz): data/processed/physical/physical_gold_daily.csv, report.json
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from src.data import evds

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "physical"
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
SYNC_PATH = ROOT / "data" / "processed" / "sync" / "sync_daily.csv"
ARCHIVE = {"TP.MK.T.KAP.Y": "kapanis_tl_kg", "TP.MK.T.AOF.Y": "agort_tl_kg", "TP.MK.D.KAP.Y": "kapanis_usd_ons"}
CURRENT = {"TP.ALTINPIYASA.KAP02": "kapanis_tl_kg", "TP.ALTINPIYASA.AGORT02": "agort_tl_kg",
           "TP.ALTINPIYASA.KAP03": "kapanis_usd_ons"}
ARCHIVE_RANGE = ("27-07-1995", "29-06-2018")
CURRENT_START = "02-07-2018"
BOUNDARY_TOL = 0.03         # arsiv son / guncel ilk kapanis farki bu orani asarsa uyari
JUMP_FLAG = 0.15            # |gunluk log getiri| bu degeri asarsa listelenir (hata degil)


class PhysicalError(Exception):
    pass


def _rename(df: pd.DataFrame, mapping: dict, source: str) -> pd.DataFrame:
    out = df.rename(columns=mapping)[["Date"] + list(mapping.values())].copy()
    out["kaynak"] = source
    return out


def build(archive: pd.DataFrame, current: pd.DataFrame) -> tuple:
    """Iki donemi birlestirir, TL/gr'ye cevirir, sinir ve tutarlilik denetler."""
    a = _rename(archive, ARCHIVE, "IAB_arsiv")
    c = _rename(current, CURRENT, "BIST_KMP")
    a = a[a["Date"] <= pd.Timestamp("2018-06-29")]
    c = c[c["Date"] >= pd.Timestamp("2018-07-02")]
    both = pd.concat([a, c], ignore_index=True).sort_values("Date")
    if both["Date"].duplicated().any():
        raise PhysicalError("Arsiv ve guncel seri ayni tarihte cakisiyor.")
    price_cols = ["kapanis_tl_kg", "agort_tl_kg", "kapanis_usd_ons"]
    if (both[price_cols] < 0).any().any():
        raise PhysicalError("Negatif fiyat var (veri hatasi).")
    zero_tl = both["kapanis_tl_kg"] == 0
    zero_dates = both.loc[zero_tl, "Date"].dt.strftime("%Y-%m-%d").tolist()
    both[price_cols] = both[price_cols].mask(both[price_cols] == 0)          # 0 = islem yok -> bos
    holidays = int(both["kapanis_tl_kg"].isna().sum()) - len(zero_dates)
    both = both.dropna(subset=["kapanis_tl_kg"]).reset_index(drop=True)
    if (both["kapanis_tl_kg"] <= 0).any():
        raise PhysicalError("Sifir veya negatif kapanis fiyati var.")
    out = pd.DataFrame({"Date": both["Date"],
                        "kapanis_tl_gr": both["kapanis_tl_kg"] / 1000.0,
                        "agort_tl_gr": both["agort_tl_kg"] / 1000.0,
                        "kapanis_usd_ons": both["kapanis_usd_ons"],
                        "kaynak": both["kaynak"]})
    r = np.log(out["kapanis_tl_gr"]).diff()
    jumps = out.loc[r.abs() > JUMP_FLAG, "Date"].dt.strftime("%Y-%m-%d").tolist()
    last_a = out[out["kaynak"] == "IAB_arsiv"].tail(1)
    first_c = out[out["kaynak"] == "BIST_KMP"].head(1)
    boundary = None
    if len(last_a) and len(first_c):
        ratio = float(first_c["kapanis_tl_gr"].iat[0] / last_a["kapanis_tl_gr"].iat[0] - 1)
        boundary = {"archive_last": str(last_a["Date"].iat[0].date()), "current_first": str(first_c["Date"].iat[0].date()),
                    "archive_last_tl_gr": float(last_a["kapanis_tl_gr"].iat[0]),
                    "current_first_tl_gr": float(first_c["kapanis_tl_gr"].iat[0]),
                    "change_pct": 100 * ratio, "within_tolerance": bool(abs(ratio) <= BOUNDARY_TOL)}
    report = {"zero_tl_no_trade_days": zero_dates,
              "rows": int(len(out)), "date_range": [str(out["Date"].min().date()), str(out["Date"].max().date())] if len(out) else None,
              "rows_by_source": out["kaynak"].value_counts().to_dict(), "empty_days_dropped": holidays,
              "boundary": boundary, "large_daily_moves": jumps,
              "missing_agort": int(out["agort_tl_gr"].isna().sum()), "missing_usd_ons": int(out["kapanis_usd_ons"].isna().sum())}
    return out, report


def compare(phys: pd.DataFrame, theo: pd.DataFrame, price_col: str, usd_col: str = None) -> dict:
    """Fiziki / teorik - 1 (prim) yillik ozet. Iki serinin saatleri farkli; prim zaman farkini da icerir."""
    m = phys.merge(theo, on="Date", how="inner")
    if m.empty:
        return {"common_days": 0}
    prem = 100 * (m["kapanis_tl_gr"] / m[price_col] - 1)
    yearly = {str(y): {"days": int(len(g)), "median_pct": float(g.median()), "p10_pct": float(g.quantile(0.1)),
                       "p90_pct": float(g.quantile(0.9))} for y, g in prem.groupby(m["Date"].dt.year)}
    out = {"common_days": int(len(m)), "range": [str(m["Date"].min().date()), str(m["Date"].max().date())],
           "premium_median_pct": float(prem.median()), "premium_abs_median_pct": float(prem.abs().median()),
           "premium_last30_median_pct": float(prem.tail(30).median()), "yearly": yearly}
    if usd_col and "kapanis_usd_ons" in m:
        d = 100 * (m["kapanis_usd_ons"] / m[usd_col] - 1)
        out["usd_ons_diff_median_pct"] = float(d.median())
    return out


def run(client=None, snapshots=None, out_dir: Path = OUT_DIR, panel_path: Path = PANEL_PATH,
        sync_path: Path = SYNC_PATH, now=None) -> dict:
    if snapshots:
        frames = []
        for snap in snapshots:
            frames.append(pd.read_csv(Path(snap) / "data.csv", parse_dates=["Date"]))
        archive = next(f for f in frames if "TP.MK.T.KAP.Y" in f.columns)
        current = next(f for f in frames if "TP.ALTINPIYASA.KAP02" in f.columns)
        used = [Path(s).name for s in snapshots]
    else:
        archive, _, s1 = evds.run_download(client, list(ARCHIVE), *ARCHIVE_RANGE, now=now)
        current, _, s2 = evds.run_download(client, list(CURRENT), CURRENT_START, "", now=now)
        used = [s1.name, s2.name]
    phys, rep = build(archive, current)
    comparisons = {}
    if Path(panel_path).exists():
        p = pd.read_csv(panel_path, parse_dates=["Date"])[["Date", "theoretical_gram_try", "gold_usd_oz"]]
        comparisons["teorik_gunluk_yahoo"] = compare(phys, p, "theoretical_gram_try", "gold_usd_oz")
    if Path(sync_path).exists():
        sy = pd.read_csv(sync_path, parse_dates=["Date"])[["Date", "theoretical_gram_try", "gold_usd_oz"]]
        comparisons["teorik_esanli_14ny"] = compare(phys, sy, "theoretical_gram_try", "gold_usd_oz")
    out_dir.mkdir(parents=True, exist_ok=True)
    phys.to_csv(out_dir / "physical_gold_daily.csv", index=False, date_format="%Y-%m-%d")
    report = {"built_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
              "evds_snapshots": used, "series": {"archive": ARCHIVE, "current": CURRENT}, **rep,
              "comparisons": comparisons,
              "note": "Toptan fiziki piyasa fiyati; kuyumcu fiyati degil. Teorik fiyatla ayri seri."}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"series": phys, "report": report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fiziki altin gunluk serisi (EVDS) ve teorik fiyatla karsilastirma.")
    ap.add_argument("--snapshots", nargs=2, default=None, help="Var olan EVDS goruntuleri (arsiv ve guncel).")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 70 + "\nFIZIKI ALTIN GUNLUK SERISI (EVDS)\n" + "=" * 70)
    try:
        client = None if args.snapshots else evds.Client(os.environ.get("EVDS_API_KEY", ""))
        r = run(client, args.snapshots)["report"]
    except (evds.EVDSError, PhysicalError, StopIteration, FileNotFoundError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1
    print(f"  Gun: {r['rows']} | {r['date_range'][0]} .. {r['date_range'][1]} | kaynak: {r['rows_by_source']} "
          f"| bos gun (tatil): {r['empty_days_dropped']}")
    b = r["boundary"]
    if b:
        print(f"  Birlesme: {b['archive_last']} {b['archive_last_tl_gr']:.3f} -> {b['current_first']} "
              f"{b['current_first_tl_gr']:.3f} TL/gr ({b['change_pct']:+.2f}%) | tolerans icinde: {b['within_tolerance']}")
    print(f"  TL fiyati 0 (islem yok, bos sayildi): {len(r['zero_tl_no_trade_days'])} gun {r['zero_tl_no_trade_days']}")
    print(f"  Buyuk gunluk hareket (>%{100 * JUMP_FLAG:.0f}): {r['large_daily_moves'] or 'yok'}")
    for name, c in r["comparisons"].items():
        if not c.get("common_days"):
            continue
        print(f"\n  Prim (fiziki / {name} - 1): {c['common_days']} ortak gun {c['range'][0]} .. {c['range'][1]} | "
              f"medyan %{c['premium_median_pct']:.2f} | son 30 gun medyan %{c['premium_last30_median_pct']:.2f}"
              + (f" | USD/ons farki medyan %{c['usd_ons_diff_median_pct']:.2f}" if "usd_ons_diff_median_pct" in c else ""))
        for y, v in c["yearly"].items():
            print(f"    {y}: {v['days']} gun | medyan %{v['median_pct']:+.2f} | %10-%90: %{v['p10_pct']:+.2f} .. %{v['p90_pct']:+.2f}")
    print(f"\n  Cikti: {OUT_DIR}\n  Not: Toptan fiziki piyasa fiyati; kuyumcu fiyati degil.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
