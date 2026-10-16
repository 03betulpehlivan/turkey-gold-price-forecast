"""
Oynaklik Tahmin Edilebilir mi? (KESIF, betimleyici)
==================================================
Kullanim (proje kokunden):  python -m src.analysis.volatility_study
Cikti: research/exploratory/volatility_study/summary.json ve table.csv

Soru: Fiyatin YONU tahmin edilemiyor (sizinti gosterimi K3). Peki yarinin OYNAKLIGI tahmin edilebilir mi?
Veri: fiziki gram altin (TL/gr) 2006-2022; GVZ (CBOE altin oynaklik endeksi) tarihsel panelden. 2023 ve
  sonrasi OKUNMADAN atilir. Hedef: ertesi fiziki gunun log getirisinin karesi r^2 (gurultulu ama yansiz vekil).
Modeller (sigma^2 tahmini; hepsi yalnizca GECMIS bilgiyle):
  Sabit       egitim donemi ortalama r^2 (oynaklik tahmin edilemez varsayimi; TABAN)
  Kayan20     son 20 gunun ortalama r^2
  EWMA        RiskMetrics, lambda = 0.94 (tahmin edilen parametre yok)
  GARCH       GARCH(1,1), parametreler her yil yalnizca egitim verisiyle (src/models/intervals.fit_garch)
  GVZ         c * (GVZ_t / 100)^2 / 252; c egitim doneminde olceklenir (GVZ ABD dolari altin oynakligidir,
              TL kur oynakligini icermez; c bunu ortalama olarak duzeltir)
  Birlesik    (GARCH + GVZ) / 2  (onceden sabit agirlik)
Degerlendirme: 2012..2022 her yil, genisleyen pencere. Kayip: QLIKE = log s2 + r^2 / s2 (vekil gurultusune
  dayanikli; Patton 2011). Sabit'e gore QLIKE farki icin DM (Newey-West + HLN). Mincer-Zarnowitz R2
  (r^2'nin s2'ye regresyonu). Normal varsayimli %80/%95 araliklarin kapsami, Kupiec ve Christoffersen p.
Hiperparametreler onceden sabittir; arama yapilmaz. Bu bir hipotez testi degildir; model secimi icin KULLANILMAZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import leakage_demo as ld
from src.models import final_exam as fe
from src.models import intervals as iv

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "volatility_study"
LOCK = ld.LOCK
EVAL_YEARS = tuple(range(2012, 2023))
LAMBDA, ROLL = 0.94, 20
GVZ_MAX_AGE_DAYS = 5
MODELS = ("Sabit", "Kayan20", "EWMA", "GARCH", "GVZ", "Birlesik")
Z = {80: 1.2815515655446004, 95: 1.959963984540054}


def load_gvz(path: Path = None) -> pd.Series:
    df = pd.read_csv(Path(path or HIST_PANEL), usecols=["Date", "gvz"], parse_dates=["Date"])
    df = df[(df["Date"] < LOCK) & (df["gvz"] > 0)].dropna()
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")["gvz"].astype(float)


def frame(prices: pd.Series, gvz: pd.Series = None) -> pd.DataFrame:
    """Satir = hedef gun T; r = log(P_T / P_t); bilgi gunu t. gvz_t: t gunu veya oncesi (en cok 5 gun)."""
    p = prices[prices.index < LOCK]
    df = pd.DataFrame({"t": p.index[:-1], "T": p.index[1:], "r": np.diff(np.log(p.to_numpy()))})
    if gvz is not None and len(gvz):
        g = gvz[gvz.index < LOCK].rename("gvz").rename_axis("gvz_date").reset_index()
        g["gvz_date"] = pd.to_datetime(g["gvz_date"]).astype("datetime64[ns]")
        df["t"] = pd.to_datetime(df["t"]).astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("t"), g.sort_values("gvz_date"), left_on="t", right_on="gvz_date",
                           direction="backward")
        stale = (df["t"] - df["gvz_date"]).dt.days > GVZ_MAX_AGE_DAYS
        df.loc[stale, "gvz"] = np.nan
        df = df.drop(columns="gvz_date")
    else:
        df["gvz"] = np.nan
    return df.reset_index(drop=True)


def forecasts(df: pd.DataFrame, eval_years=EVAL_YEARS) -> pd.DataFrame:
    """Her satir icin sigma^2 tahminleri; T yili icin parametreler yalnizca T yilindan ONCEKI satirlarla."""
    r = df["r"].to_numpy()
    r2 = r ** 2
    # gecmise dayali, parametresiz seriler: deger[i] = i. satirin hedefi icin tahmin (r[0..i-1] ile)
    ewma = np.empty(len(r))
    v = r2[:ROLL].mean()
    for i in range(len(r)):
        ewma[i] = v
        v = LAMBDA * v + (1 - LAMBDA) * r2[i]
    roll = pd.Series(r2).shift(1).rolling(ROLL, min_periods=ROLL).mean().to_numpy()
    out = []
    for Y in eval_years:
        tr = (df["T"] < pd.Timestamp(f"{Y}-01-01")).to_numpy()
        te = ((df["T"] >= pd.Timestamp(f"{Y}-01-01")) & (df["T"] < pd.Timestamp(f"{Y + 1}-01-01"))).to_numpy()
        if te.sum() == 0 or tr.sum() < 500:
            continue
        gp = iv.fit_garch(r[tr])
        sig = iv.garch_sigma(r, gp)                     # sig[i]: r[0..i] ile, r[i+1] icin
        garch = np.concatenate([[np.nan], sig[:-1] ** 2])
        gd2 = (df["gvz"].to_numpy() / 100.0) ** 2 / 252.0
        ok = tr & np.isfinite(gd2)
        c = r2[ok].mean() / gd2[ok].mean() if ok.sum() >= 250 else np.nan
        part = pd.DataFrame({"T": df.loc[te, "T"].to_numpy(), "year": Y, "r": r[te], "Sabit": r2[tr].mean(),
                             "Kayan20": roll[te], "EWMA": ewma[te], "GARCH": garch[te], "GVZ": c * gd2[te]})
        part["Birlesik"] = (part["GARCH"] + part["GVZ"]) / 2
        out.append(part)
    return pd.concat(out, ignore_index=True)


def qlike(r: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return np.log(s2) + r ** 2 / s2


def summarize(pred: pd.DataFrame, models=MODELS) -> list:
    pred = pred.dropna(subset=list(models)).reset_index(drop=True)
    r, r2 = pred["r"].to_numpy(), pred["r"].to_numpy() ** 2
    base = qlike(r, pred["Sabit"].to_numpy())
    rows = []
    for m in models:
        s2 = pred[m].to_numpy()
        q = qlike(r, s2)
        X = np.column_stack([np.ones(len(s2)), s2])
        beta, *_ = np.linalg.lstsq(X, r2, rcond=None)
        resid = r2 - X @ beta
        row = {"model": m, "n": int(len(pred)), "qlike": float(q.mean()), "mse_var": float(np.mean((r2 - s2) ** 2)),
               "mz_r2": float(1 - resid.var() / r2.var()), "mz_slope": float(beta[1]),
               "qlike_gain_vs_sabit": float(base.mean() - q.mean())}
        if m != "Sabit":
            row["dm_p_vs_sabit"] = fe.dm_hln(q, base)["p_value"]
        for lvl in (80, 95):
            miss = (np.abs(r) > Z[lvl] * np.sqrt(s2)).astype(float)
            row[f"cov{lvl}"] = float(1 - miss.mean())
            row[f"kupiec{lvl}_p"] = iv.kupiec_p(miss, 1 - lvl / 100)
            row[f"christoffersen{lvl}_p"] = iv.christoffersen_p(miss)
        rows.append(row)
    return rows


def yearly(pred: pd.DataFrame, models=MODELS) -> list:
    out = []
    for Y, g in pred.dropna(subset=list(models)).groupby("year"):
        base = qlike(g["r"].to_numpy(), g["Sabit"].to_numpy()).mean()
        out.append({"year": int(Y), "n": int(len(g)), **{f"{m}_gain": float(base - qlike(g["r"].to_numpy(), g[m].to_numpy()).mean())
                                                          for m in models if m != "Sabit"}})
    return out


def run_all(prices: pd.Series, gvz: pd.Series = None, eval_years=EVAL_YEARS) -> dict:
    df = frame(prices, gvz)
    pred = forecasts(df, eval_years)
    return {"nature": "KESIF (betimleyici); model secimi icin KULLANILMAZ",
            "data": {"first": str(df["T"].min().date()), "last": str(df["T"].max().date()), "n_returns": int(len(df)),
                     "gvz_coverage": float(df["gvz"].notna().mean())},
            "config": {"lambda": LAMBDA, "roll": ROLL, "eval_years": [int(y) for y in sorted(pred["year"].unique())]},
            "models": summarize(pred), "yearly": yearly(pred)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Oynaklik tahmin edilebilirligi (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    print("=" * 78 + "\nOYNAKLIK TAHMINI (KESIF) - fiziki gram altin, 2006-2022\n" + "=" * 78)
    res = run_all(ld.load_prices(), load_gvz())
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = pd.DataFrame(res["models"])
    t.to_csv(out / "table.csv", index=False, float_format="%.6g")
    cols = ["model", "qlike_gain_vs_sabit", "dm_p_vs_sabit", "mz_r2", "mz_slope", "cov80", "christoffersen80_p", "cov95"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t[[c for c in cols if c in t.columns]].round(4).to_string(index=False))
        print("\n  Yillara gore QLIKE kazanci (Sabit'e gore; pozitif = daha iyi):")
        print(pd.DataFrame(res["yearly"]).round(4).to_string(index=False))
    print(f"\n  n {res['models'][0]['n']} | GVZ kapsami %{100 * res['data']['gvz_coverage']:.1f} | Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
