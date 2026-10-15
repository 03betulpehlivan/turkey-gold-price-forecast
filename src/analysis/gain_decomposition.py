"""
Sabah Fiziki Tahmin Kazancinin Ayristirilmasi: Gece Ons mu, Gece Kur mu? (KESIF, 2012-2022)
==========================================================================================
Kullanim (proje kokunden):
  python -m src.analysis.gain_decomposition

Soru: Bosluk modelinin 2012-2022 kazanci (ayni gun kur ile ~%24, onceki gun kur ile ~%0) hangi bilgiden
geliyor? Kur kismi Yahoo gunluk kur cubugunun SAATINE baglidir (2012-2022 icin dogrulanamadi). Ons kismi ise
yalnizca sabah kesinlikle bilinen iki degerle olculebilir:
  g_ons = log(Yahoo ons kapanisi, t gunu, ~18:00 UTC; BIST kapanisindan SONRA) - log(BIST USD/ons kapanisi, t)
  g_kur = log(Yahoo USD/TRY cubugu, T gunu)                                     - log(BIST ortuk kuru, t)
  BIST ortuk kuru = BIST TL/gr x 31.1035 / BIST USD/ons (ayni borsa, ayni kapanis).
  Hedef: y = log P[T] - log P[t] (ardisik fiziki gunler, T - t <= 4 takvim gunu).
g_ons Yahoo kur saatinden TAMAMEN bagimsizdir: anlamli kazanc verirse proje iddiasi icin zamanlamadan
bagimsiz bir kanit olur.

Seviye farki duzeltmesi (sonuclardan ONCE sabitlendi): GC=F vadeli ile BIST spot arasinda tasima maliyeti
kadar degisen bir fark vardir; her degiskenden son 20 BILINEN satirin medyani cikarilir (yalnizca gecmis).
Modeller (genisleyen OLS, 63 satirda bir, 1 satir purge, egitim 2011'den): RW, Ons (g_ons), Kur (g_kur),
Ikisi (g_ons + g_kur). Karsilastirma RW'ye gore: HLN-DM + Holm (3) + eslestirilmis blok bootstrap.
Kilit: hedef tarihi >= 2023-01-01 hicbir veri okunmaz.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import final_exam as fe
from src.models import physical_eval as pv
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
LOCK = pd.Timestamp(pv.LOCK_START)
EVAL_START, TRAIN_START = pd.Timestamp("2012-01-01"), pd.Timestamp("2011-01-01")
BASIS_WIN = 20
MAX_GAP = 4
MODELS = {"Ons": ["g_ons"], "Kur": ["g_kur"], "Ikisi": ["g_ons", "g_kur"]}


class DecompError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "diagnostics" / "gain_decomposition"


def build(phys: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    p = phys[(phys["Date"] < LOCK)].dropna(subset=["kapanis_tl_gr", "kapanis_usd_ons"]).sort_values("Date")
    p = p[(p["kapanis_tl_gr"] > 0) & (p["kapanis_usd_ons"] > 0)].reset_index(drop=True)
    pan = panel[panel["Date"] < LOCK].set_index("Date")
    df = pd.DataFrame({"t": p["Date"], "P_t": p["kapanis_tl_gr"], "bist_usd_t": p["kapanis_usd_ons"],
                       "fx_bist_t": p["kapanis_tl_gr"] * 31.1035 / p["kapanis_usd_ons"]})
    df["T"] = df["t"].shift(-1)
    df["P_T"] = df["P_t"].shift(-1)
    df = df.dropna(subset=["T"]).reset_index(drop=True)
    df = df[(df["T"] - df["t"]).dt.days <= MAX_GAP]
    df["gold_t"] = df["t"].map(pan["gold_usd_oz"])                # Yahoo ons, AYNI gun t (aksam kapanisi)
    df["fx_T"] = df["T"].map(pan["usdtry"])                       # Yahoo kur, hedef gun T cubugu
    df = df.dropna(subset=["gold_t", "fx_T"]).reset_index(drop=True)
    df["y"] = np.log(df["P_T"] / df["P_t"])
    for col, raw in (("g_ons", np.log(df["gold_t"] / df["bist_usd_t"])), ("g_kur", np.log(df["fx_T"] / df["fx_bist_t"]))):
        basis = raw.shift(1).rolling(BASIS_WIN, min_periods=BASIS_WIN).median()   # yalnizca gecmis
        df[col] = raw - basis
    return df.dropna(subset=["g_ons", "g_kur"]).reset_index(drop=True)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    pos = df.index[(df["T"] >= EVAL_START) & (df["T"] < LOCK)].tolist()
    if not pos:
        raise DecompError("Degerlendirme satiri yok.")
    out = []
    for w in tr.make_windows(pos, tr.RETRAIN_EVERY):
        ti = df.index[(df["T"] >= TRAIN_START) & (df.index < w[0] - 1)]
        if df.loc[ti, "T"].max() >= df.loc[w[0], "t"] + pd.Timedelta(days=1):
            raise DecompError("Egitim hedefi degerlendirme gunune tasiyor.")
        rec = {"T": df.loc[w, "T"].to_numpy(), "P_t": df.loc[w, "P_t"].to_numpy(), "y": df.loc[w, "y"].to_numpy(),
               "pred_RW": np.zeros(len(w))}
        for m, cols in MODELS.items():
            X = np.column_stack([np.ones(len(ti))] + [df.loc[ti, c].to_numpy() for c in cols])
            beta = np.linalg.lstsq(X, df.loc[ti, "y"].to_numpy(), rcond=None)[0]
            Xw = np.column_stack([np.ones(len(w))] + [df.loc[w, c].to_numpy() for c in cols])
            rec[f"pred_{m}"] = Xw @ beta
            rec[f"beta_{m}"] = np.repeat(beta[1], len(w))
        out.append(pd.DataFrame(rec))
    pred = pd.concat(out, ignore_index=True)
    if pd.to_datetime(pred["T"]).max() >= LOCK:
        raise DecompError("Kilitli doneme tahmin uretildi.")
    return pred


def compare(pred: pd.DataFrame) -> pd.DataFrame:
    p, y = pred["P_t"].to_numpy(float), pred["y"].to_numpy(float)
    err = lambda m: np.abs(p * (np.exp(y) - np.exp(pred[f"pred_{m}"].to_numpy(float))))
    e_rw = err("RW")
    rows, pvals = [], {}
    for m in MODELS:
        e = err(m)
        b, d = fe.paired_bootstrap(e, e_rw), fe.dm_hln(e, e_rw)
        hit = np.mean(np.sign(pred[f"pred_{m}"]) == np.sign(y)) * 100
        rows.append({"model": m, "n": int(len(e)), "MAE_TL": float(e.mean()), "MAE_RW_TL": float(e_rw.mean()),
                     "improvement_pct": -b["pct"], "ci_low_pct": -b["pct_ci95"][1], "ci_high_pct": -b["pct_ci95"][0],
                     "dm_stat": d["dm_stat_hln"], "p_dm": d["p_value"], "direction_hit_pct": float(hit),
                     "beta_last": float(pred[f"beta_{m}"].iat[-1])})
        pvals[m] = d["p_value"]
    t = pd.DataFrame(rows)
    order, running, holm = sorted(pvals, key=pvals.get), 0.0, {}
    for k, m in enumerate(order):
        running = max(running, min(1.0, pvals[m] * (len(order) - k)))
        holm[m] = running
    t["p_holm"] = t["model"].map(holm)
    t["significantly_better"] = (t["p_holm"] < fe.ALPHA) & (t["dm_stat"] < 0) & (t["ci_low_pct"] > 0)
    return t


def yearly(pred: pd.DataFrame) -> pd.DataFrame:
    p, y = pred["P_t"].to_numpy(float), pred["y"].to_numpy(float)
    yr = pd.to_datetime(pred["T"]).dt.year.to_numpy()
    rows = []
    for Y in np.unique(yr):
        k = yr == Y
        e_rw = np.abs(p[k] * (np.exp(y[k]) - 1)).mean()
        row = {"year": int(Y), "n": int(k.sum())}
        for m in MODELS:
            e = np.abs(p[k] * (np.exp(y[k]) - np.exp(pred.loc[k, f"pred_{m}"].to_numpy(float)))).mean()
            row[f"{m}_vs_RW_pct"] = float(100 * (1 - e / e_rw))
        rows.append(row)
    return pd.DataFrame(rows)


def run(phys_path: Path = pv.PHYS_PATH, panel_path: Path = HIST_PANEL) -> dict:
    if Path(panel_path).resolve() == Path(pv.PANEL_PATH).resolve():
        raise DecompError("Genisletilmis panel gerekli (data/processed/history).")
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    df = build(phys[phys["Date"] < LOCK], panel[panel["Date"] < LOCK])
    pred = walk_forward(df)
    comp, yrs = compare(pred), yearly(pred)
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / "predictions.csv", index=False)
    comp.to_csv(d / "comparison.csv", index=False)
    yrs.to_csv(d / "yearly.csv", index=False)
    sig = comp.loc[comp["significantly_better"], "model"].tolist()
    summary = {"nature": "KESIF - gelistirme donemi (hedef < 2023-01-01)",
               "period": [str(pd.Timestamp(pred["T"].min()).date()), str(pd.Timestamp(pred["T"].max()).date())],
               "n": int(len(pred)), "significantly_better_than_rw": sig,
               "timing_robust_gain": bool("Ons" in sig),
               "corr_g_ons_g_kur": float(df["g_ons"].corr(df["g_kur"]))}
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary, "comparison": comp, "yearly": yrs, "predictions": pred}


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Sabah fiziki tahmin kazancinin ayristirilmasi (gelistirme).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 80
    print(f"\n{sep}\nKAZANC AYRISTIRMA: GECE ONS (zamanlamadan bagimsiz) vs GECE KUR (KESIF, 2012-2022)\n{sep}")
    try:
        r = run()
    except (FileNotFoundError, DecompError) as e:
        print(f"  HATA: {e}")
        return 1
    pd.set_option("display.width", 220)
    s = r["summary"]
    print(f"  Donem: {s['period'][0]} .. {s['period'][1]} | {s['n']} tahmin | g_ons-g_kur korelasyonu {s['corr_g_ons_g_kur']:+.3f}\n")
    print(r["comparison"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Yillara gore RW'ye gore iyilesme (%):")
    print(r["yearly"].to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
    verdict = ("EVET: gece ons bilgisi tek basina RW'yi anlamli gecti; kazancin bu kismi Yahoo kur saatinden BAGIMSIZ."
               if s["timing_robust_gain"] else
               "HAYIR: zamanlamadan bagimsiz (yalniz ons) kazanc anlamli degil; iddia canli kayda dayanmali.")
    print(f"\n  Zamanlamadan bagimsiz kazanc: {verdict}\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
