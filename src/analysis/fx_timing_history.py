"""
Yahoo USD/TRY Gunluk Cubugunun Saati: Yillara Gore Teshis (TCMB gosterge kuru ile)
===================================================================================
Kullanim (proje kokunden):
  $env:EVDS_API_KEY = "..."
  python -m src.data.evds indir --seri TP.DK.USD.A.YTL --baslangic 01-01-2005
  python -m src.analysis.fx_timing_history --evds data\\raw\\evds\\<yukaridaki_goruntu>

Soru: Fiziki sabah tahmini "hedef gunun Yahoo kur cubugunu" kullaniyor. Bu cubuk Borsa Istanbul
kapanisindan ONCE mi olusuyor? 2024-2025 icin saatlik veriyle ~02-05 UTC olculdu; eski yillarda
Yahoo farkli bir saat kullandiysa tarihsel kazancin bir kismi sizinti olabilir.

Yontem: TCMB gosterge kuru (TP.DK.USD.A.YTL) her gun sabit saatte belirlenir (15:30 Istanbul;
bu TCMB'nin bilinen uygulamasidir, EVDS metaverisinden ayrica dogrulanmadi). Yahoo gunluk getirisi
[h(t-1), h(t)] ile TCMB getirisi arasindaki ortusme, iki saat arasindaki farkla orantilidir:
  r_tcmb(t-1) ile ortusme ~ max(0, hc - h), r_tcmb(t) ile ~ 24 - |h - hc|, r_tcmb(t+1) ile ~ max(0, h - hc).
Yalnizca ardisik takvim gunleri arasindaki (tam 24 saatlik) getiriler kullanilir; hafta sonu iceren
getiriler ortusme oranlarini bozar. Bu uc korelasyondan h tahmin edilir: h = hc - 24*lag/toplam + 24*lead/toplam  (hc: TCMB saati, UTC).
On kayitli siniflandirma (her yil): lead > 1.96/sqrt(n) -> "TCMB_SONRASI" (cubuk gun ici/aksam);
h_tahmin < hc - 3 ve lead bant icinde -> "SABAH"; aksi -> "BELIRSIZ".
Dogrulama: 2024 ve 2025 tahmini saatlik teshisle (~02-05 UTC) tutarli olmali.
Yalnizca KUR verisi kullanilir; fiziki fiyatlara ve sinav sonuclarina dokunulmaz.

GERCEK CALISTIRMA SONUCU (14.10.2026): dogrulama TUTARSIZ (tahmini saatler 24'u asiyor). Olasi neden:
EVDS gosterge kurunu gecerlilik gunuyle (ertesi gun) tarihlendiriyor olabilir; 2023 sonrasi kontrollu
kurda getirilerin bagimsizlik varsayimi da bozulabilir. Bu nedenle ikinci mod eklendi:

MOD 2 (--referans bist): Borsa Istanbul kapanisindaki ORTUK kur = (TL/gr x 31.1035) / (USD/ons), iki fiyat
da ayni borsanin ayni gunku kapanisi (data/processed/physical/physical_gold_daily.csv). Soru dogrudan:
Yahoo cubugu BIST kapanisindan ONCE mi? r_yahoo(t) ile r_bist(t+1) korelasyonu bant icindeyse ONCE
(sizinti yok); bandi asarsa SONRA (sizinti riski). Kontrol: r_bist(t+2) ile korelasyon bant icinde olmali;
degilse getiriler bagimli, "GECERSIZ". KARAR rejimlere gore toplu verilir (2005-2010, 2011-2015, 2016-2022;
physical_history ile ayni); yillik tablo yalnizca betimleyicidir (yil basina ~250 gozlemde %5 yanlis alarm
ve dusuk guc sentetik testte goruldu). Yalnizca 2023 oncesi (sinav donemi fiziki verisi kullanilmaz).
Yahoo kur getirileri (t-1 -> t) ile ortuk kur getirileri ardisik ORTAK gunler uzerinden hesaplanir.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
PHYS_PATH = ROOT / "data" / "processed" / "physical" / "physical_gold_daily.csv"
BIST_LAST_YEAR = 2022           # sinav donemi (2023+) fiziki verisi bu teshiste de kullanilmaz
BIST_ERAS = [("2005-2010", 2005, 2010), ("2011-2015", 2011, 2015), ("2016-2022", 2016, 2022)]
TCMB_CODE = "TP.DK.USD.A.YTL"
TCMB_LOCAL = "15:30"
ONE_DAY = 1               # yalnizca ardisik takvim gunleri: her getiri tam 24 saat (hafta sonu/tatil disarida)


class TimingError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "diagnostics" / "fx_timing_history"


def tcmb_hour_utc(dates: pd.Series) -> np.ndarray:
    """15:30 Istanbul yerel saatinin UTC karsiligi (2016 oncesi yaz saati dahil)."""
    local = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + TCMB_LOCAL).dt.tz_localize("Europe/Istanbul")
    u = local.dt.tz_convert("UTC")
    return (u.dt.hour + u.dt.minute / 60).to_numpy(float)


def merged(yahoo: pd.DataFrame, tcmb: pd.DataFrame) -> pd.DataFrame:
    m = yahoo[["Date", "usdtry"]].merge(tcmb[["Date", TCMB_CODE]].dropna(), on="Date").sort_values("Date")
    m = m[(m["usdtry"] > 0) & (m[TCMB_CODE] > 0)].reset_index(drop=True)
    gap = m["Date"].diff().dt.days
    m["r_y"] = np.log(m["usdtry"]).diff()
    m["r_c"] = np.log(m[TCMB_CODE]).diff()
    m.loc[gap != ONE_DAY, ["r_y", "r_c"]] = np.nan              # 24 saatten uzun getiri ortusme oranini bozar
    m["r_c_lag"] = m["r_c"].shift(1)
    m["r_c_lead"] = m["r_c"].shift(-1)
    m["hc"] = tcmb_hour_utc(m["Date"])
    return m


def yearly(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in m.groupby(m["Date"].dt.year):
        g = g.dropna(subset=["r_y", "r_c", "r_c_lag", "r_c_lead"])
        n = len(g)
        if n < 40:
            continue
        lag, same, lead = (float(g["r_y"].corr(g[c])) for c in ("r_c_lag", "r_c", "r_c_lead"))
        tot = max(lag, 0) + max(same, 0) + max(lead, 0)
        hc = float(np.median(g["hc"]))
        h = hc - 24 * max(lag, 0) / tot + 24 * max(lead, 0) / tot if tot > 0.3 else np.nan
        band = 1.96 / np.sqrt(n)
        if lead > band:
            verdict = "TCMB_SONRASI"
        elif np.isfinite(h) and h < hc - 3:
            verdict = "SABAH"
        else:
            verdict = "BELIRSIZ"
        rows.append({"year": int(y), "n": n, "corr_lag": lag, "corr_same": same, "corr_lead": lead, "band95": band,
                     "tcmb_hour_utc": hc, "est_hour_utc": h, "verdict": verdict})
    return pd.DataFrame(rows)


def bist_implied(phys: pd.DataFrame) -> pd.DataFrame:
    p = phys[phys["Date"].dt.year <= BIST_LAST_YEAR].dropna(subset=["kapanis_tl_gr", "kapanis_usd_ons"])
    p = p[(p["kapanis_tl_gr"] > 0) & (p["kapanis_usd_ons"] > 0)]
    return pd.DataFrame({"Date": p["Date"], "fx_bist": p["kapanis_tl_gr"] * 31.1035 / p["kapanis_usd_ons"]})


def _bist_returns(yahoo: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    b = bist_implied(phys)
    m = yahoo[["Date", "usdtry"]].merge(b, on="Date").sort_values("Date").reset_index(drop=True)
    m = m[m["Date"].dt.year <= BIST_LAST_YEAR]
    m["r_y"] = np.log(m["usdtry"]).diff()
    m["r_b"] = np.log(m["fx_bist"]).diff()
    m["r_b_lead"] = m["r_b"].shift(-1)
    m["r_b_lead2"] = m["r_b"].shift(-2)
    return m.dropna(subset=["r_y", "r_b", "r_b_lead", "r_b_lead2"])


def _bist_row(g: pd.DataFrame) -> dict:
    n = len(g)
    same, lead, lead2 = (float(g["r_y"].corr(g[c])) for c in ("r_b", "r_b_lead", "r_b_lead2"))
    band = 1.96 / np.sqrt(n)
    if abs(lead2) > band:
        verdict = "GECERSIZ"
    elif lead > band:
        verdict = "KAPANIS_SONRASI"
    else:
        verdict = "KAPANIS_ONCESI"
    return {"n": n, "corr_same": same, "corr_lead": lead, "corr_lead2_control": lead2, "band95": band, "verdict": verdict}


def bist_yearly(yahoo: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    m = _bist_returns(yahoo, phys)
    return pd.DataFrame([{"year": int(y), **_bist_row(g)} for y, g in m.groupby(m["Date"].dt.year) if len(g) >= 60])


def bist_eras(yahoo: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    m = _bist_returns(yahoo, phys)
    yr = m["Date"].dt.year
    return pd.DataFrame([{"era": name, **_bist_row(m[(yr >= a) & (yr <= b)])} for name, a, b in BIST_ERAS
                         if ((yr >= a) & (yr <= b)).sum() >= 60])


def run_bist(phys_path: Path = PHYS_PATH, panel_path: Path = HIST_PANEL) -> dict:
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    table, eras = bist_yearly(panel, phys), bist_eras(panel, phys)
    if eras.empty:
        raise TimingError("Yeterli ortak gun yok.")
    summary = {"reference": "BIST kapanisinda ortuk USD/TRY", "last_year": BIST_LAST_YEAR,
               "era_verdicts": dict(zip(eras["era"], eras["verdict"])),
               "decision": ("SIZINTI_YOK" if (eras["verdict"] == "KAPANIS_ONCESI").all() else
                            "SIZINTI_RISKI" if (eras["verdict"] == "KAPANIS_SONRASI").any() else "KARARSIZ")}
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    table.to_csv(d / "bist_yearly.csv", index=False)
    eras.to_csv(d / "bist_eras.csv", index=False)
    (d / "bist_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"table": table, "eras": eras, "summary": summary}


def run(evds_snapshot: Path, panel_path: Path = HIST_PANEL) -> dict:
    p = Path(evds_snapshot) / "data.csv"
    if not p.exists():
        raise TimingError(f"EVDS verisi yok: {p}")
    tcmb = pd.read_csv(p, parse_dates=["Date"])
    if TCMB_CODE not in tcmb.columns:
        raise TimingError(f"{TCMB_CODE} sutunu yok; dogru goruntuyu verin.")
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    table = yearly(merged(panel, tcmb))
    if table.empty:
        raise TimingError("Yeterli ortak gun yok.")
    check = table[table["year"].isin([2024, 2025])]
    consistent = bool(len(check) and (check["verdict"] == "SABAH").all())
    summary = {"years": int(len(table)), "verdict_counts": table["verdict"].value_counts().to_dict(),
               "validation_2024_2025_morning": consistent,
               "non_morning_years": table.loc[table["verdict"] != "SABAH", "year"].tolist(),
               "note": "Yalnizca kur verisi; TCMB 15:30 saati bilinen uygulama, EVDS'den ayrica dogrulanmadi."}
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    table.to_csv(d / "yearly.csv", index=False)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"table": table, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Yahoo USD/TRY gunluk cubuk saati, yillara gore (TCMB ile).")
    ap.add_argument("--referans", choices=["tcmb", "bist"], default="bist")
    ap.add_argument("--evds", type=Path, default=None, help="(tcmb modu) TP.DK.USD.A.YTL iceren EVDS goruntu klasoru.")
    ap.add_argument("--panel", type=Path, default=HIST_PANEL)
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 78
    if args.referans == "bist":
        print(f"\n{sep}\nYAHOO USD/TRY CUBUGU BIST KAPANISINDAN ONCE MI? (ORTUK KUR, {BIST_LAST_YEAR} VE ONCESI)\n{sep}")
        try:
            r = run_bist(PHYS_PATH, args.panel)
        except (FileNotFoundError, TimingError) as e:
            print(f"  HATA: {e}")
            return 1
        pd.set_option("display.width", 200)
        print("  Yillik (YALNIZCA betimleyici; tek yil ~%5 yanlis alarm):")
        print(r["table"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n  REJIMLERE GORE (KARAR):")
        print(r["eras"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  KARAR: {r['summary']['decision']}  (SIZINTI_YOK: tum rejimler kapanis oncesi)")
        print(f"  Ciktilar: {out_dir()}\n{sep}")
        return 0
    if args.evds is None:
        ap.error("--referans tcmb icin --evds gerekli.")
    print(f"\n{sep}\nYAHOO USD/TRY GUNLUK CUBUK SAATI - YILLARA GORE (TCMB 15:30 REFERANSI)\n{sep}")
    try:
        r = run(args.evds, args.panel)
    except (FileNotFoundError, TimingError) as e:
        print(f"  HATA: {e}")
        return 1
    pd.set_option("display.width", 200)
    print(r["table"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    s = r["summary"]
    print("  Not: yillik saat tahmini ~100 gozleme dayanir ve +/-4-5 saat gurultuludur; karar siniflandirmadan verilir.")
    print(f"\n  Siniflar: {s['verdict_counts']} | sabah disi yillar: {s['non_morning_years'] or 'yok'}")
    print(f"  Dogrulama (2024-2025 sabah olmali): {'TUTARLI' if s['validation_2024_2025_morning'] else 'TUTARSIZ - yonteme guvenme'}")
    print(f"  Not: {s['note']}\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
