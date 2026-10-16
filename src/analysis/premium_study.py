"""
Fiziki Prim: Ortalamaya Doner mi, Tahmin Edilebilir mi? (KESIF, betimleyici)
===========================================================================
Kullanim (proje kokunden):  python -m src.analysis.premium_study
Cikti: research/exploratory/premium_study/summary.json ve table.csv

Tanim: prim_t = 100 * log(fiziki TL/gr_t / teorik TL/gr_t); teorik = ons x USD/TRY / 31.1035 (tarihsel panel).
Veri: 2006-01-01 .. 2022-12-31. 2023 ve sonrasi OKUNMADAN atilir (2023+ fiziki veri kilitli; FINAL_EXAM_PHYS_V1
  basarisizliginin nedeni olan prim rejimi burada INCELENMEZ).
Olcum uyarisi: teorik fiyatin ons ve kuru farkli saatlere aittir; gunluk prim bu nedenle zamanlama gurultusu
  icerir. Bagimsiz olcum gurultusu otokorelasyonu dusurur ve SAHTE ortalamaya donus gosterir. Bu yuzden ANA
  analiz haftalik ortalama prim (cuma biten hafta, en az 3 gun) uzerindedir; gunluk sonuc yalnizca karsilastirma.
Analizler:
  1. Yillik betimleme: medyan, ceyrekler arasi aralik, |prim| > %1 gun orani.
  2. AR(1): p[w+1] - mu = phi (p[w] - mu) + e; yari omur = ln(0.5) / ln(phi); phi icin blok bootstrap %95 GA.
  3. Ileriye donuk sinama (2012..2022, her yil yalnizca gecmisle): gelecek haftanin ortalama primi icin
     RW (gecen hafta), Ortalama (egitim ortalamasi) ve AR1 (egitimle tahmin) karsilastirilir; mutlak hata,
     DM (Newey-West + HLN).
Onceden sabit: haftalik kural W-FRI, en az 3 gun, bootstrap blok 8 hafta, 2000 tekrar, tohum 20260101.
Bu bir hipotez testi degildir; model secimi icin KULLANILMAZ. Canli Prim-Capa modeli DEGISTIRILMEZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.analysis import leakage_demo as ld
from src.models import final_exam as fe

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "premium_study"
LOCK = ld.LOCK
EVAL_YEARS = tuple(range(2012, 2023))
MIN_DAYS, BLOCK, REPS, SEED = 3, 8, 2000, 20260101


def load_theoretical(path: Path = None) -> pd.Series:
    df = pd.read_csv(Path(path or HIST_PANEL), usecols=["Date", "theoretical_gram_try"], parse_dates=["Date"])
    df = df[(df["Date"] < LOCK) & (df["theoretical_gram_try"] > 0)].dropna()
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")["theoretical_gram_try"].astype(float)


def daily_premium(phys: pd.Series, theo: pd.Series) -> pd.Series:
    j = pd.concat([phys[phys.index < LOCK].rename("phys"), theo[theo.index < LOCK].rename("theo")], axis=1, join="inner")
    return (100 * np.log(j["phys"] / j["theo"])).rename("prim")


def weekly_premium(daily: pd.Series) -> pd.Series:
    g = daily.groupby(daily.index.to_period("W-FRI"))
    w = g.mean()[g.size() >= MIN_DAYS]
    w.index = w.index.end_time.normalize()
    return w.rename("prim")


def ar1(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    a, b = x[:-1], x[1:]
    A = np.column_stack([np.ones(len(a)), a])
    c, phi = np.linalg.lstsq(A, b, rcond=None)[0]
    mu = c / (1 - phi) if phi < 1 else float(np.mean(x))
    hl = math.log(0.5) / math.log(phi) if 0 < phi < 1 else float("inf")
    return {"phi": float(phi), "mu": float(mu), "half_life": float(hl), "n": int(len(b))}


def bootstrap_phi(x: np.ndarray, block: int = BLOCK, reps: int = REPS, seed: int = SEED) -> list:
    x = np.asarray(x, float)
    pairs = np.column_stack([x[:-1], x[1:]])
    n = len(pairs)
    rng = np.random.default_rng(seed)
    phis = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        s = pairs[idx]
        A = np.column_stack([np.ones(n), s[:, 0]])
        phis.append(np.linalg.lstsq(A, s[:, 1], rcond=None)[0][1])
    return [float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5))]


def yearly(daily: pd.Series) -> list:
    out = []
    for Y, s in daily.groupby(daily.index.year):
        out.append({"year": int(Y), "n": int(len(s)), "median": float(s.median()),
                    "iqr": float(s.quantile(0.75) - s.quantile(0.25)), "share_abs_gt_1pct": float((s.abs() > 1).mean())})
    return out


def walk_forward(w: pd.Series, eval_years=EVAL_YEARS) -> pd.DataFrame:
    df = pd.DataFrame({"d": w.index[:-1], "d_next": w.index[1:], "p": w.to_numpy()[:-1], "y": w.to_numpy()[1:]})
    out = []
    for Y in eval_years:
        start, end = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        tr = df[df["d_next"] < start]
        te = df[(df["d_next"] >= start) & (df["d_next"] < end)]
        if len(te) == 0 or len(tr) < 100:
            continue
        A = np.column_stack([np.ones(len(tr)), tr["p"].to_numpy()])
        c, phi = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)[0]
        part = te.copy()
        part["year"] = Y
        part["RW"], part["Ortalama"] = part["p"], float(tr["y"].mean())
        part["AR1"] = c + phi * part["p"]
        part["phi_train"] = phi
        out.append(part)
    return pd.concat(out, ignore_index=True)


def summarize_forecasts(pred: pd.DataFrame) -> list:
    err = {m: np.abs(pred["y"] - pred[m]).to_numpy() for m in ("RW", "Ortalama", "AR1")}
    rows = []
    for m, e in err.items():
        r = {"model": m, "mae_pct_points": float(e.mean()), "vs_rw_pct": float(100 * (1 - e.mean() / err["RW"].mean()))}
        if m != "RW":
            r["dm_p_vs_rw"] = fe.dm_hln(e, err["RW"])["p_value"]
        rows.append(r)
    return rows


def run_all(phys: pd.Series, theo: pd.Series, eval_years=EVAL_YEARS) -> dict:
    d = daily_premium(phys, theo)
    w = weekly_premium(d)
    pred = walk_forward(w, eval_years)
    wa = ar1(w.to_numpy())
    return {"nature": "KESIF (betimleyici); model secimi icin KULLANILMAZ",
            "data": {"first": str(d.index.min().date()), "last": str(d.index.max().date()), "days": int(len(d)),
                     "weeks": int(len(w))},
            "config": {"weekly_rule": "W-FRI", "min_days": MIN_DAYS, "bootstrap": {"block": BLOCK, "reps": REPS, "seed": SEED},
                       "eval_years": [int(y) for y in sorted(pred["year"].unique())]},
            "yearly": yearly(d),
            "ar1_weekly": {**wa, "phi_ci95": bootstrap_phi(w.to_numpy())},
            "ar1_daily_noisy": {**ar1(d.to_numpy()), "note": "zamanlama gurultusu donusu abartir"},
            "forecast_next_week": summarize_forecasts(pred),
            "phi_train_by_year": {int(Y): float(g["phi_train"].iat[0]) for Y, g in pred.groupby("year")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fiziki prim (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    print("=" * 78 + "\nFIZIKI PRIM (KESIF) - 2006-2022\n" + "=" * 78)
    res = run_all(ld.load_prices(), load_theoretical())
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame(res["yearly"]).to_csv(out / "table.csv", index=False, float_format="%.6g")
    a, dn = res["ar1_weekly"], res["ar1_daily_noisy"]
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(res["yearly"]).round(3).to_string(index=False))
        print(f"\n  Haftalik AR(1): phi {a['phi']:.3f} [{a['phi_ci95'][0]:.3f}, {a['phi_ci95'][1]:.3f}] | ortalama %{a['mu']:.2f}"
              f" | yari omur {a['half_life']:.1f} hafta (n {a['n']})")
        print(f"  Gunluk AR(1) (gurultulu, yalnizca karsilastirma): phi {dn['phi']:.3f} | yari omur {dn['half_life']:.1f} gun")
        print("\n  Gelecek haftanin ortalama primi (2012-2022, walk-forward):")
        print(pd.DataFrame(res["forecast_next_week"]).round(4).to_string(index=False))
        print("  Egitimle tahmin edilen phi, yillara gore: " +
              ", ".join(f"{y}: {v:.2f}" for y, v in res["phi_train_by_year"].items()))
    print(f"\n  Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
