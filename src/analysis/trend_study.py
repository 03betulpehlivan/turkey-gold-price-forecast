"""
Haftalik ve Aylik Trend, Makro Gostergelerle (KESIF, betimleyici)
================================================================
Kullanim (proje kokunden):  python -m src.analysis.trend_study
Cikti: research/exploratory/trend_study/summary.json ve table.csv

Soru: Gunluk yon tahmin edilemiyor. Daha uzun ufukta (1 hafta, 1 ay) momentum, kur, ons, ABD faizi ve TL
  reel faizi fiziki gram altinin YONUNU veya GETIRISINI tahmin etmeye yardim ediyor mu?
Veri: fiziki gram altin (TL/gr) 2006-2022; tarihsel panel (USD/TRY, ons, ABD 10 yillik faiz); EVDS aylik
  makro (src/data/macro.py: TUFE genel 2003=100, TR politika faizi). 2023 ve sonrasi OKUNMADAN atilir.
Donemler: hafta = her ISO haftasinin (cuma biter) son fiziki gunu; ay = her ayin son fiziki gunu.
  Satir t: bilgi tarihi d_t; hedef y = log(P[t+1] / P[t]); yukari = y > 0.
Ozellikler (d_t'de bilinen; onceden sabit):
  mom1      son donem getirisi                    momk   son k donem getirisi (ay: 12, hafta: 13)
  fx1       USD/TRY son donem log degisimi        ons1   ons (USD) son donem log degisimi
  us10y_d   ABD 10 yillik faiz son donem degisimi
  reel      TR politika faizi - yillik TUFE enflasyonu; reel_d3 = reel'in 3 ay onceye gore degisimi
  YAYIM GECIKMESI: M ayinin TUFE'si M+1 ayinin 5. gununden itibaren; M ayinin politika faizi M+1 ayinin
  1. gununden itibaren kullanilir (tutucu; aciklama tarihinden once kullanilmaz).
Modeller (her test yili 2012..2022 icin yalnizca hedefi o yildan ONCE biten satirlarla; olcekleme egitimle):
  Getiri:  Yarin=bugun (0), Ortalama (egitim ortalamasi), Dogrusal (EKK, standartlastirilmis ozellikler)
  Yon:     Iklim (egitim yukari orani), Lojistik (C=1), Hep-yukari (yalnizca isabet icin kural)
Olcutler: TL mutlak hata, RW'ye ve Ortalama'ya gore iyilesme ve DM; Brier skoru, Iklim'e gore Brier
  becerisi ve DM; log kayip; isabet (0.5 esik). Bu bir hipotez testi degildir; model secimi icin KULLANILMAZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import leakage_demo as ld
from src.data import macro as mc
from src.models import final_exam as fe

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "trend_study"
LOCK = ld.LOCK
EVAL_YEARS = tuple(range(2012, 2023))
HORIZONS = {"hafta": {"rule": "W-FRI", "k": 13}, "ay": {"rule": "M", "k": 12}}
FEATURES = ["mom1", "momk", "fx1", "ons1", "us10y_d", "reel", "reel_d3"]
CPI_DAY, MIN_TRAIN = 5, 60


def load_panel(path: Path = None) -> pd.DataFrame:
    df = pd.read_csv(Path(path or HIST_PANEL), usecols=["Date", "usdtry", "gold_usd_oz", "us10y"], parse_dates=["Date"])
    return df[df["Date"] < LOCK].sort_values("Date").reset_index(drop=True)


def macro_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Ay basina reel faiz; ay = verinin ait oldugu ay (yayim gecikmesi asof() icinde uygulanir)."""
    m = m[m["month"] < LOCK[:7]].sort_values("month").reset_index(drop=True)
    m["enf"] = 100 * (m["tufe"] / m["tufe"].shift(12) - 1)
    m["period"] = pd.PeriodIndex(m["month"], freq="M")
    return m.set_index("period")[["enf", "politika_tr"]]


def asof_macro(dates: pd.Series, mm: pd.DataFrame) -> pd.DataFrame:
    """Her tarih icin YAYIMLANMIS en son enflasyon ve politika faizi; reel = faiz - enflasyon."""
    out = []
    for d in pd.to_datetime(dates):
        p = pd.Period(d, freq="M")
        cpi_m = p - 1 if d.day >= CPI_DAY else p - 2
        pol_m = p - 1
        enf = mm["enf"].get(cpi_m, np.nan)
        pol = mm["politika_tr"].get(pol_m, np.nan)
        enf3 = mm["enf"].get(cpi_m - 3, np.nan)
        pol3 = mm["politika_tr"].get(pol_m - 3, np.nan)
        out.append({"reel": pol - enf, "reel_d3": (pol - enf) - (pol3 - enf3), "cpi_month": str(cpi_m)})
    return pd.DataFrame(out)


def periods(prices: pd.Series, horizon: str, panel: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    rule, k = HORIZONS[horizon]["rule"], HORIZONS[horizon]["k"]
    p = prices[prices.index < LOCK]
    last = p.groupby(p.index.to_period(rule)).apply(lambda s: s.index.max())
    ends = pd.DatetimeIndex(last.to_numpy())
    df = pd.DataFrame({"d": ends, "P": p.loc[ends].to_numpy()})
    pn = panel.set_index("Date").sort_index()
    at = pn.reindex(pn.index.union(ends)).ffill().loc[ends]           # d tarihinde veya oncesinde bilinen deger
    df["fx"], df["ons"], df["us10y"] = at["usdtry"].to_numpy(), at["gold_usd_oz"].to_numpy(), at["us10y"].to_numpy()
    lp = np.log(df["P"])
    df["mom1"] = lp.diff()
    df["momk"] = lp - lp.shift(k)
    df["fx1"] = np.log(df["fx"]).diff()
    df["ons1"] = np.log(df["ons"]).diff()
    df["us10y_d"] = df["us10y"].diff()
    df = pd.concat([df, asof_macro(df["d"], mm)], axis=1)
    df["d_next"], df["y"] = df["d"].shift(-1), lp.shift(-1) - lp
    return df.dropna(subset=FEATURES + ["y", "d_next"]).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, eval_years=EVAL_YEARS) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    out = []
    for Y in eval_years:
        start, end = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        tr = df[df["d_next"] < start]
        te = df[(df["d_next"] >= start) & (df["d_next"] < end)]
        if len(te) == 0 or len(tr) < MIN_TRAIN:
            continue
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
        Xtr, Xte = ((tr[FEATURES] - mu) / sd).to_numpy(), ((te[FEATURES] - mu) / sd).to_numpy()
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        beta = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)[0]
        up_tr = (tr["y"] > 0).astype(int).to_numpy()
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, up_tr)
        part = te[["d", "d_next", "P", "y"]].copy()
        part["year"] = Y
        part["g_RW"], part["g_Ortalama"] = 0.0, float(tr["y"].mean())
        part["g_Dogrusal"] = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
        part["p_Iklim"] = float(up_tr.mean())
        part["p_Lojistik"] = lr.predict_proba(Xte)[:, 1]
        part["p_Hep_yukari"] = 1.0
        out.append(part)
    return pd.concat(out, ignore_index=True)


def summarize(pred: pd.DataFrame) -> dict:
    P1 = pred["P"] * np.exp(pred["y"])
    err = {m: np.abs(P1 - pred["P"] * np.exp(pred[f"g_{m}"])).to_numpy() for m in ("RW", "Ortalama", "Dogrusal")}
    ret = []
    for m, e in err.items():
        row = {"model": m, "mae_tl": float(e.mean()), "vs_rw_pct": float(100 * (1 - e.mean() / err["RW"].mean())),
               "vs_ortalama_pct": float(100 * (1 - e.mean() / err["Ortalama"].mean()))}
        if m != "RW":
            row["dm_p_vs_rw"] = fe.dm_hln(e, err["RW"])["p_value"]
        if m == "Dogrusal":
            row["dm_p_vs_ortalama"] = fe.dm_hln(e, err["Ortalama"])["p_value"]
        ret.append(row)
    up = (pred["y"] > 0).astype(float).to_numpy()
    eps = 1e-12
    brier = {m: (pred[f"p_{m}"].to_numpy() - up) ** 2 for m in ("Iklim", "Lojistik", "Hep_yukari")}
    dirs = []
    for m, b in brier.items():
        p = np.clip(pred[f"p_{m}"].to_numpy(), eps, 1 - eps)
        row = {"model": m, "brier": float(b.mean()), "brier_skill_vs_iklim": float(1 - b.mean() / brier["Iklim"].mean()),
               "hit_pct": float(100 * np.mean((pred[f"p_{m}"].to_numpy() >= 0.5) == (up == 1)))}
        if m != "Hep_yukari":
            row["logloss"] = float(-np.mean(up * np.log(p) + (1 - up) * np.log(1 - p)))
        if m == "Lojistik":
            row["dm_p_vs_iklim"] = fe.dm_hln(b, brier["Iklim"])["p_value"]
        dirs.append(row)
    return {"n": int(len(pred)), "up_rate_pct": float(100 * up.mean()), "returns": ret, "direction": dirs}


def run_all(prices: pd.Series, panel: pd.DataFrame, macro_df: pd.DataFrame, eval_years=EVAL_YEARS) -> dict:
    mm = macro_monthly(macro_df)
    res = {"nature": "KESIF (betimleyici); model secimi icin KULLANILMAZ",
           "config": {"features": FEATURES, "cpi_available_day": CPI_DAY, "policy_rate_lag_months": 1,
                      "eval_years": list(eval_years), "horizons": HORIZONS}}
    for h in HORIZONS:
        pred = walk_forward(periods(prices, h, panel, mm), eval_years)
        res[h] = {**summarize(pred), "first": str(pred["d_next"].min().date()), "last": str(pred["d_next"].max().date())}
    return res


def table(res: dict) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        for kind in ("returns", "direction"):
            for r in res[h][kind]:
                rows.append({"ufuk": h, "tur": kind, "n": res[h]["n"], **r})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Haftalik/aylik trend ve makro (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    print("=" * 78 + "\nHAFTALIK / AYLIK TREND VE MAKRO (KESIF) - fiziki gram altin, 2012-2022\n" + "=" * 78)
    res = run_all(ld.load_prices(), load_panel(), mc.load())
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = table(res)
    t.to_csv(out / "table.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for h in HORIZONS:
            print(f"\n  {h.upper()}: n {res[h]['n']} ({res[h]['first']} .. {res[h]['last']}), yukari orani %{res[h]['up_rate_pct']:.1f}")
            print(pd.DataFrame(res[h]["returns"]).round(4).to_string(index=False))
            print(pd.DataFrame(res[h]["direction"]).round(4).to_string(index=False))
    print(f"\n  Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
