"""
Veri Teshis Analizleri (Adim 2a)
================================
Kullanim (proje kokunden):
  python -m src.analysis.data_diagnostics

Girdi : data/processed/market_panel.csv, en yeni anlik goruntu (data/raw/snapshots/),
        varsa data/processed/results/significance.csv
Cikti : yalnizca data/processed/results/diagnostics/ (kendi ciktilarini yeniler)
        Hicbir veri dosyasi degismez, model egitilmez.

Donem kurallari (veri sizintisini ve son sinavin kirlenmesini onlemek icin)
---------------------------------------------------------------------------
  HOLDOUT_START (2026-01-01) ve sonrasi: HICBIR analizde kullanilmaz.
  Ongorulebilirlik tanilari (D3, D4, D5, D6): yalnizca ANALYSIS_END (2023-10-25,
    test donemi oncesi) dahil olmak uzere egitim + dogrulama donemi.
  Veri kalitesi tanilari (D2, hafta sonu cubuklari): HOLDOUT_START oncesi.

Tanilar
-------
  D1  significance.csv'den tespit edilebilir en kucuk MAE farki (%80 guc, %5 anlamlilik)
  D2  Vadeli kontrat degisim penceresi etkisi (altin, gumus, Brent) ve en buyuk hareketler
  D3  Getiri otokorelasyonlari (ons, kur, gram) ve alt donemler
  D4  Esanlilik: ons/kur/dolar endeksi/gumus arasinda bir gunluk capraz korelasyonlar
  D5  Takvim bosluklari ve hedefin bosluk uzunluguna gore dagilimi
  D5b Ham serilerde Cumartesi/Pazar tarihli gunluk cubuklar
  D6  60 gunluk kayan suruklenme (yilliklandirilmis), bilesen bazinda
  D7  Donem ozeti (son sinav icin ayrilan gun sayisi; degerleri okunmaz)
"""

from pathlib import Path
import argparse
import contextlib
import io
import json
import math
import sys

import numpy as np
import pandas as pd

from src.data import build_dataset as bd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
SNAPSHOT_ROOT = ROOT / "data" / "raw" / "snapshots"
SIGNIFICANCE_PATH = ROOT / "data" / "processed" / "results" / "significance.csv"
OUT_DIR = ROOT / "data" / "processed" / "results" / "diagnostics"

HOLDOUT_START = "2026-01-01"
ANALYSIS_END = "2023-10-25"          # train.TEST_START'tan bir gun once
MAX_LAG = 10
ROLL_MONTHS = (1, 3, 5, 7, 9, 11)    # GC/SI aktif ay oncesi (varsayim; dogrulanmadi)
ROLL_LAST_N = 5                       # ayin son 5 islem gunu
SUBPERIODS = [("2015-2018", "2015-01-01", "2018-12-31"),
              ("2019-2021", "2019-01-01", "2021-12-31"),
              ("2022-2023-10", "2022-01-01", ANALYSIS_END)]
RETURN_SOURCES = {          # getiri adi -> panel sutunu
    "gold_usd": "gold_usd_oz", "usdtry": "usdtry", "gram": "theoretical_gram_try",
    "silver": "silver", "dxy": "dxy", "us10y": "us10y", "vix": "vix", "sp500": "sp500",
    "bist100": "bist100", "brent": "brent", "eurtry": "eurtry",
}


def _quiet(func, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args)


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def load_panel(path: Path) -> pd.DataFrame:
    """Paneli okur ve son sinav donemini (HOLDOUT_START ve sonrasi) DUSURUR."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Panel yok: {path}. Once: python -m src.data.build_panel --write")
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [c for c in ("Date", "gold_usd_oz", "usdtry", "theoretical_gram_try") if c not in df.columns]
    if missing:
        raise ValueError(f"Panelde eksik sutunlar: {missing}")
    if not df["Date"].is_monotonic_increasing or df["Date"].duplicated().any():
        raise ValueError("Panel tarihleri kesin artan degil.")
    return df[df["Date"] < pd.Timestamp(HOLDOUT_START)].reset_index(drop=True)


def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel satirlari arasinda log getiri: r[t] = log(X[t] / X[t-1]).

    Iki ardisik satirdan biri bossa getiri bos kalir. Ayrica:
      target[t] = r_gram[t+1] (Faz A/B hedefiyle ayni tanim)
      gap_days[t] = Date[t+1] - Date[t] (takvim gunu)
    """
    out = panel[["Date"]].copy()
    for name, col in RETURN_SOURCES.items():
        if col in panel.columns:
            x = panel[col].astype(float)
            out[f"r_{name}"] = np.log(x / x.shift(1))
    out["target"] = out["r_gram"].shift(-1)
    out["gap_days"] = (panel["Date"].shift(-1) - panel["Date"]).dt.days
    return out


def upto(df: pd.DataFrame, end: str) -> pd.DataFrame:
    return df[df["Date"] <= pd.Timestamp(end)]


# ---------------------------------------------------------------------------
# Istatistik yardimcilari
# ---------------------------------------------------------------------------
def acf(x, max_lag: int = MAX_LAG) -> list:
    """Ornek otokorelasyon: sum((x_t-m)(x_{t-k}-m)) / sum((x_t-m)^2)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    d = x - x.mean()
    denom = float(d @ d)
    if len(x) < 3 or denom == 0:
        return [np.nan] * max_lag
    return [float(d[k:] @ d[:-k]) / denom for k in range(1, max_lag + 1)]


def cross_corr(x: pd.Series, y: pd.Series, k: int) -> tuple:
    """corr(x[t], y[t+k]). k > 0: y, x'ten SONRA gelir. Dondurur (r, n)."""
    pair = pd.concat([x.reset_index(drop=True), y.shift(-k).reset_index(drop=True)], axis=1).dropna()
    if len(pair) < 3:
        return np.nan, len(pair)
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), len(pair)


def welch(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = (a.mean() - b.mean()) / se if se > 0 else np.nan
    p = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan
    return {"n_in": len(a), "n_out": len(b), "mean_in": float(a.mean()), "mean_out": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "t": float(t), "p_normal": float(p)}


def roll_window_mask(dates: pd.Series, months=ROLL_MONTHS, last_n: int = ROLL_LAST_N) -> pd.Series:
    """Panelde, verilen aylarin son `last_n` islem gununu isaretler."""
    d = pd.Series(pd.to_datetime(dates).to_numpy())
    ym = d.dt.to_period("M")
    rank_from_end = d.groupby(ym).rank(method="first", ascending=False)
    return ((rank_from_end <= last_n) & d.dt.month.isin(months)).set_axis(dates.index)


# ---------------------------------------------------------------------------
# Tanilar
# ---------------------------------------------------------------------------
def d1_min_detectable_effect(path: Path) -> pd.DataFrame:
    """se = |ortalama fark / dm_stat|; MDE(%80 guc, iki yonlu %5) ~= 2.80 * se."""
    if not Path(path).exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    s = s[s["reference"] == "Mean-Return"].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        s["se"] = (s["mean_abs_err_diff"] / s["dm_stat"]).abs()
    s["mde_80"] = 2.80 * s["se"]
    return s[["period", "model", "n", "mean_abs_err_diff", "dm_stat", "p_value", "se", "mde_80"]]


def d2_roll_effect(ret: pd.DataFrame) -> tuple:
    """Kontrat degisim penceresinde ortalama getiri farki ve en buyuk hareketler."""
    mask = roll_window_mask(ret["Date"])
    rows, tops = [], []
    for name in ("gold_usd", "silver", "brent"):
        col = f"r_{name}"
        if col not in ret:
            continue
        r = ret[col]
        rows.append({"series": name, **welch(r[mask], r[~mask])})
        top = ret.loc[r.abs().nlargest(20).index, ["Date", col]].rename(columns={col: "log_return"})
        top["series"] = name
        top["in_roll_window"] = mask[top.index].to_numpy()
        top["weekday"] = top["Date"].dt.day_name()
        tops.append(top)
    return pd.DataFrame(rows), (pd.concat(tops, ignore_index=True) if tops else pd.DataFrame())


def d3_autocorrelation(ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = [("egitim+dogrulama", "2015-01-01", ANALYSIS_END)] + SUBPERIODS
    for label, start, end in periods:
        sub = ret[(ret["Date"] >= pd.Timestamp(start)) & (ret["Date"] <= pd.Timestamp(end))]
        for name in ("gold_usd", "usdtry", "gram"):
            x = sub[f"r_{name}"].dropna()
            band = 1.96 / math.sqrt(len(x)) if len(x) else np.nan
            for lag, v in enumerate(acf(x), start=1):
                rows.append({"period": label, "series": name, "lag": lag, "acf": v, "n": len(x),
                             "band_95": band, "outside_band": bool(abs(v) > band) if np.isfinite(v) else False})
    return pd.DataFrame(rows)


def d4_cross_correlation(ret: pd.DataFrame) -> pd.DataFrame:
    pairs = [("gold_usd", "usdtry"), ("usdtry", "gold_usd"), ("dxy", "gold_usd"),
             ("silver", "gold_usd"), ("gold_usd", "silver"), ("eurtry", "usdtry")]
    rows = []
    for a, b in pairs:
        if f"r_{a}" not in ret or f"r_{b}" not in ret:
            continue
        for k in (-1, 0, 1):
            r, n = cross_corr(ret[f"r_{a}"], ret[f"r_{b}"], k)
            rows.append({"x": a, "y": b, "k": k, "definition": f"corr({a}[t], {b}[t{k:+d}])",
                         "corr": r, "n": n, "band_95": 1.96 / math.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def d5_calendar_gaps(ret: pd.DataFrame) -> pd.DataFrame:
    g = ret.dropna(subset=["target", "gap_days"]).copy()
    g["gap"] = g["gap_days"].astype(int).clip(upper=5).map(lambda v: "5+" if v >= 5 else str(v))
    out = g.groupby("gap")["target"].agg(n="size", mean="mean", std="std",
                                         mean_abs=lambda s: s.abs().mean()).reset_index()
    return out


def d5b_weekend_bars(snapshot: Path) -> pd.DataFrame:
    if snapshot is None:
        return pd.DataFrame()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for name, e in manifest["series"].items():
        if e.get("status") != "ok":
            continue
        s = _quiet(bd.read_yahoo_csv, snapshot / e["file"], e["ticker"])
        d = s.loc[s["Date"] < pd.Timestamp(HOLDOUT_START), "Date"]
        wk = d[d.dt.dayofweek >= 5]
        rows.append({"series": name, "ticker": e["ticker"], "rows": int(len(d)),
                     "saturday": int((wk.dt.dayofweek == 5).sum()), "sunday": int((wk.dt.dayofweek == 6).sum()),
                     "years": ",".join(sorted({str(y) for y in wk.dt.year})),
                     "examples": ",".join(str(x.date()) for x in wk.head(3))})
    return pd.DataFrame(rows)


def d6_rolling_drift(ret: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    sub = upto(ret, ANALYSIS_END)
    out = sub[["Date"]].copy()
    for name in ("gold_usd", "usdtry", "gram"):
        out[f"{name}_drift_ann"] = sub[f"r_{name}"].rolling(window, min_periods=window).mean() * 252
    return out


def latest_snapshot(root: Path):
    if not Path(root).exists():
        return None
    snaps = sorted([p for p in Path(root).iterdir() if (p / "manifest.json").exists()], reverse=True)
    return snaps[0] if snaps else None


def save_plots(acf_df: pd.DataFrame, drift: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    full = acf_df[acf_df["period"] == "egitim+dogrulama"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for ax, name in zip(axes, ("gold_usd", "usdtry", "gram")):
        s = full[full["series"] == name]
        ax.bar(s["lag"], s["acf"], color="tab:blue")
        b = s["band_95"].iloc[0]
        ax.axhline(b, color="gray", ls="--", lw=0.8)
        ax.axhline(-b, color="gray", ls="--", lw=0.8)
        ax.set_title(f"ACF {name} (2015 - {ANALYSIS_END})")
        ax.set_xlabel("gecikme")
    plt.tight_layout()
    plt.savefig(out_dir / "acf.png", dpi=110)
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in ("gold_usd", "usdtry"):
        ax.plot(drift["Date"], drift[f"{name}_drift_ann"], lw=0.9, label=name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("60 gunluk kayan suruklenme (yilliklandirilmis log getiri)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_drift.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Ana
# ---------------------------------------------------------------------------
def run(panel_path: Path, snapshot, significance_path: Path, out_dir: Path) -> dict:
    full_panel = pd.read_csv(panel_path, parse_dates=["Date"]) if Path(panel_path).exists() else None
    panel = load_panel(panel_path)
    ret = add_returns(panel)
    pre_test = upto(ret, ANALYSIS_END)

    res = {
        "d1": d1_min_detectable_effect(significance_path),
        "d2_effect": None, "d2_top": None,
        "d3": d3_autocorrelation(pre_test),
        "d4": d4_cross_correlation(pre_test),
        "d5": d5_calendar_gaps(pre_test),
        "d5b": d5b_weekend_bars(snapshot),
        "d6": d6_rolling_drift(ret),
    }
    res["d2_effect"], res["d2_top"] = d2_roll_effect(ret)

    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"d1": "d1_min_detectable_effect.csv", "d2_effect": "d2_roll_window_effect.csv",
             "d2_top": "d2_top_moves.csv", "d3": "d3_autocorrelation.csv",
             "d4": "d4_cross_correlation.csv", "d5": "d5_calendar_gaps.csv",
             "d5b": "d5b_weekend_bars.csv", "d6": "d6_rolling_drift.csv"}
    for key, fname in names.items():
        if res[key] is not None and len(res[key]):
            res[key].to_csv(out_dir / fname, index=False)
    save_plots(res["d3"], res["d6"], out_dir)

    n_holdout = int((full_panel["Date"] >= pd.Timestamp(HOLDOUT_START)).sum()) if full_panel is not None else 0
    summary = {
        "panel_rows_used": int(len(panel)),
        "data_quality_period": [str(panel["Date"].iloc[0].date()), str(panel["Date"].iloc[-1].date())],
        "predictability_period_end": ANALYSIS_END,
        "holdout_start": HOLDOUT_START,
        "holdout_rows_not_analyzed": n_holdout,
        "snapshot": snapshot.name if snapshot is not None else None,
        "assumptions": {"roll_window": f"{list(ROLL_MONTHS)} aylarinin son {ROLL_LAST_N} islem gunu "
                                       "(Yahoo'nun kontrat degisim gunu dogrulanmadi)",
                        "cross_corr": "corr(x[t], y[t+k]); k>0: y sonra"},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    res["summary"] = summary
    return res


def print_report(res: dict):
    pd.set_option("display.width", 160)
    s = res["summary"]
    print(f"  Kullanilan panel: {s['data_quality_period'][0]} .. {s['data_quality_period'][1]} "
          f"({s['panel_rows_used']} gun) | son sinav icin ayrilan, okunmayan: {s['holdout_rows_not_analyzed']} gun")
    print(f"  Ongorulebilirlik tanilari {s['predictability_period_end']} tarihine kadar.\n")
    if len(res["d1"]):
        d = res["d1"].copy()
        print("  D1 Tespit edilebilir en kucuk MAE farki (Mean-Return'e karsi, %80 guc):")
        print(d[["period", "model", "n", "mean_abs_err_diff", "mde_80"]].to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    else:
        print("  D1: significance.csv bulunamadi, atlandi.")
    print("\n  D2 Kontrat degisim penceresi (ortalama gunluk log getiri, iceride - disarida):")
    print(res["d2_effect"][["series", "n_in", "mean_in", "mean_out", "diff", "t", "p_normal"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    d3 = res["d3"][res["d3"]["period"] == "egitim+dogrulama"]
    print("\n  D3 Otokorelasyon (2015 - test oncesi), gecikme 1-5; bant disi olanlar * ile:")
    for name in ("gold_usd", "usdtry", "gram"):
        s3 = d3[d3["series"] == name].head(5)
        vals = "  ".join(f"{r.acf:+.3f}{'*' if r.outside_band else ' '}" for r in s3.itertuples())
        print(f"    {name:9s} {vals}   (bant +/-{s3['band_95'].iloc[0]:.3f})")
    print("\n  D4 Capraz korelasyon corr(x[t], y[t+k]):")
    piv = res["d4"].pivot_table(index=["x", "y"], columns="k", values="corr")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n  D5 Hedefin takvim boslugu uzunluguna gore dagilimi:")
    print(res["d5"].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if len(res["d5b"]):
        print("\n  D5b Hafta sonu tarihli gunluk cubuklar (ham seriler):")
        print(res["d5b"][["series", "rows", "saturday", "sunday", "years"]].to_string(index=False))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Salt okunur veri teshis analizleri.")
    ap.add_argument("--panel", type=Path, default=PANEL_PATH)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--significance", type=Path, default=SIGNIFICANCE_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print("VERI TESHIS ANALIZLERI (salt okunur)")
    print("=" * 60)
    snapshot = args.snapshot or latest_snapshot(SNAPSHOT_ROOT)
    try:
        res = run(args.panel, snapshot, args.significance, args.out)
    except (FileNotFoundError, ValueError, bd.DataError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(res)
    print(f"\n  Ciktilar: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
