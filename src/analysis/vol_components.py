"""
TL Oynakligi: Bilesen Modeli (KESIF, betimleyici)
=================================================
Kullanim (proje kokunden):  python -m src.analysis.vol_components
Cikti: research/exploratory/vol_components/summary.json ve table.csv

Soru: Oynaklik calismasinda (volatility_study) GVZ, TL kur oynakligini icermedigi icin 2018 ve 2021 kur
  krizlerinde basarisiz oldu. Fiziki TL getirisi yaklasik olarak ons (USD) getirisi + USD/TRY getirisidir;
  oyleyse varyans ~ var(ons) + var(kur) + 2 kov(ons, kur). Bilesenleri ayri tahmin etmek isler mi?
Veri: fiziki gram altin 2006-2022, GVZ ve tarihsel panel (ons USD, USD/TRY; Yahoo). 2023+ OKUNMADAN atilir.
Modeller (sigma^2; hepsi yalnizca GECMIS bilgiyle; t gunu bilgisi, T gunu hedefi):
  Sabit, GARCH, GVZ      volatility_study ile ayni (karsilastirma)
  BilesenEWMA            c * [EWMA var(ons) + EWMA var(kur) + 2 EWMA kov(ons, kur)]
  BilesenGVZ             c * [(GVZ/100)^2/252 + EWMA var(kur) + 2 EWMA kov(ons, kur)]
  Birlesik2              (GARCH + BilesenGVZ) / 2  (onceden sabit agirlik)
  EWMA lambda = 0.94; panel getirileri panel takviminde, t gunu VEYA ONCESI (en cok 5 gun) eslenir.
  c: her test yili icin yalnizca egitim satirlariyla (ortalama r^2 / ortalama bilesen); zamanlama ve
  prim gurultusunu ortalama olarak duzeltir.
Degerlendirme: 2012..2022 (volatility_study ile ayni); QLIKE, Sabit'e ve GARCH'a gore DM, yillara gore
  kazanc (ozellikle 2018 ve 2021), %80/%95 kapsam ve Christoffersen.
Hiperparametreler onceden sabittir. Bu bir hipotez testi degildir; model secimi icin KULLANILMAZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import leakage_demo as ld
from src.analysis import volatility_study as vs
from src.models import final_exam as fe

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "vol_components"
LOCK = ld.LOCK
LAMBDA, MAX_AGE_DAYS, MIN_CAL = 0.94, 5, 250
MODELS = ("Sabit", "GARCH", "GVZ", "BilesenEWMA", "BilesenGVZ", "Birlesik2")


def load_panel(path: Path = None) -> pd.DataFrame:
    df = pd.read_csv(Path(path or HIST_PANEL), usecols=["Date", "gold_usd_oz", "usdtry"], parse_dates=["Date"])
    df = df[(df["Date"] < LOCK) & (df["gold_usd_oz"] > 0) & (df["usdtry"] > 0)].dropna()
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def ewma_components(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel gunu d icin, d dahil getirilerle guncellenmis EWMA var(ons), var(kur), kov (d sonunda bilinen)."""
    g = np.diff(np.log(panel["gold_usd_oz"].to_numpy()))
    f = np.diff(np.log(panel["usdtry"].to_numpy()))
    n0 = 20
    vg, vf, cv = g[:n0].var(), f[:n0].var(), float(np.cov(g[:n0], f[:n0])[0, 1])
    out = np.empty((len(g), 3))
    for i in range(len(g)):
        vg = LAMBDA * vg + (1 - LAMBDA) * g[i] ** 2
        vf = LAMBDA * vf + (1 - LAMBDA) * f[i] ** 2
        cv = LAMBDA * cv + (1 - LAMBDA) * g[i] * f[i]
        out[i] = (vg, vf, cv)
    res = pd.DataFrame(out, columns=["var_ons", "var_kur", "kov"])
    res["comp_date"] = panel["Date"].to_numpy()[1:]
    return res


def attach(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    c = comp.copy()
    c["comp_date"] = pd.to_datetime(c["comp_date"]).astype("datetime64[ns]")
    d = df.copy()
    d["t"] = pd.to_datetime(d["t"]).astype("datetime64[ns]")
    m = pd.merge_asof(d.sort_values("t"), c.sort_values("comp_date"), left_on="t", right_on="comp_date", direction="backward")
    stale = (m["t"] - m["comp_date"]).dt.days > MAX_AGE_DAYS
    m.loc[stale, ["var_ons", "var_kur", "kov"]] = np.nan
    return m.drop(columns="comp_date").sort_values("T").reset_index(drop=True)


def forecasts(df: pd.DataFrame, eval_years=vs.EVAL_YEARS) -> pd.DataFrame:
    base = vs.forecasts(df, eval_years)
    raw_e = (df["var_ons"] + df["var_kur"] + 2 * df["kov"]).to_numpy()
    raw_g = ((df["gvz"] / 100.0) ** 2 / 252.0 + df["var_kur"] + 2 * df["kov"]).to_numpy()
    r2 = df["r"].to_numpy() ** 2
    parts = []
    for Y, part in base.groupby("year", sort=True):
        tr = (df["T"] < pd.Timestamp(f"{Y}-01-01")).to_numpy()
        idx = df.index[df["T"].isin(part["T"])].to_numpy()
        part = part.copy()
        for name, raw in (("BilesenEWMA", raw_e), ("BilesenGVZ", raw_g)):
            ok = tr & np.isfinite(raw)
            c = r2[ok].mean() / raw[ok].mean() if ok.sum() >= MIN_CAL else np.nan
            part[name] = c * raw[idx]
        part["Birlesik2"] = (part["GARCH"] + part["BilesenGVZ"]) / 2
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def vs_garch(pred: pd.DataFrame) -> list:
    p = pred.dropna(subset=list(MODELS))
    r = p["r"].to_numpy()
    g = vs.qlike(r, p["GARCH"].to_numpy())
    out = []
    for m in ("BilesenEWMA", "BilesenGVZ", "Birlesik2"):
        q = vs.qlike(r, p[m].to_numpy())
        out.append({"model": m, "qlike_gain_vs_garch": float(g.mean() - q.mean()),
                    "dm_p_vs_garch": fe.dm_hln(q, g)["p_value"],
                    "years_better_than_garch": int(sum(
                        vs.qlike(gg["r"].to_numpy(), gg[m].to_numpy()).mean() < vs.qlike(gg["r"].to_numpy(), gg["GARCH"].to_numpy()).mean()
                        for _, gg in p.groupby("year")))})
    return out


def run_all(prices: pd.Series, gvz: pd.Series, panel: pd.DataFrame, eval_years=vs.EVAL_YEARS) -> dict:
    df = attach(vs.frame(prices, gvz), ewma_components(panel))
    pred = forecasts(df, eval_years)
    return {"nature": "KESIF (betimleyici); model secimi icin KULLANILMAZ",
            "config": {"lambda": LAMBDA, "max_age_days": MAX_AGE_DAYS, "eval_years": [int(y) for y in sorted(pred["year"].unique())]},
            "coverage": {"components": float(df["var_kur"].notna().mean()), "gvz": float(df["gvz"].notna().mean())},
            "models": vs.summarize(pred, MODELS), "vs_garch": vs_garch(pred), "yearly": vs.yearly(pred, MODELS)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TL oynakligi bilesen modeli (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    print("=" * 78 + "\nTL OYNAKLIGI BILESEN MODELI (KESIF) - fiziki gram altin, 2012-2022\n" + "=" * 78)
    res = run_all(ld.load_prices(), vs.load_gvz(), load_panel())
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = pd.DataFrame(res["models"])
    t.to_csv(out / "table.csv", index=False, float_format="%.6g")
    cols = ["model", "qlike_gain_vs_sabit", "dm_p_vs_sabit", "cov80", "christoffersen80_p", "cov95"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t[cols].round(4).to_string(index=False))
        print("\n  GARCH'a gore:")
        print(pd.DataFrame(res["vs_garch"]).round(4).to_string(index=False))
        print("\n  Yillara gore QLIKE kazanci (Sabit'e gore; pozitif = daha iyi):")
        print(pd.DataFrame(res["yearly"]).round(3).to_string(index=False))
    print(f"\n  Bilesen kapsami %{100 * res['coverage']['components']:.1f} | GVZ kapsami %{100 * res['coverage']['gvz']:.1f} | Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
