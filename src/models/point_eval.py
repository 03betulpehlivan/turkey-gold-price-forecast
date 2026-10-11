"""
Nokta Tahmini Degerlendirmesi (TL cinsinden)
============================================
Kullanim (proje kokunden):
  python -m src.models.point_eval

Soru: Yarinin teorik gram altin TL fiyatini, "yarin = bugun" diyen rastgele
yuruyusten (Zero-Return) ne kadar daha kucuk hatayla tahmin edebiliyoruz?

Ayrim (karistirilmaz):
  1. Nokta tahmini  : P_hat[t+1] = P[t] * exp(r_hat[t])   (TL hata metrikleri)
  2. Getiri tahmini : r_hat[t]                            (getiri hata metrikleri)
  3. Yon tahmini    : sign(r_hat[t])                      (yukselis/dusus dogrulugu)
  4. Belirsizlik    : intervals.py / adaptive.py          (bu modulde DEGERLENDIRILMEZ)

Not: Log getiri tahmin edildigi icin P[t]*exp(r_hat) fiyatin medyan tahminidir.
MAE icin dogru secimdir; RMSE icin ortalama tahmini exp(s^2/2) kadar farkli olurdu
(gram altinda ~%0.005, birkac kurus): ihmal edilir ve raporda belirtilir.

Veri ve donem
-------------
  model_dataset.csv (hedef ve ozellikler degismedi). Degerlendirme: 2018-01-01 ..
  TEST_START oncesi (gelistirme; KESIF). Test donemi nokta tahmini icin Faz A'da
  zaten goruldugu icin bagimsiz teyit sayilmaz; 2026 burada KULLANILMAZ.
  Mevcut 6 model train.walk_forward_evaluate ile (kodlari degismeden) uretilir;
  yeni adaylar ayni pencere, purge ve satirlarla uretilir ve tarih esitligi dogrulanir.

Yeni adaylar (gerekce: onceki teshis bulgulari)
-----------------------------------------------
  Mean-250        egitimdeki son 250 hedefin ortalamasi (suruklenme rejimden rejime degisir, D6)
  FX-AR1          r = a + b * usdtry_return[t]  (kur getirisinde 1. gecikme ACF +0.092, D3)
  Bilesen-Gecikme r = a + b1*usdtry_return[t] + b2*gold_usd_return[t]
                  (ons[t]-kur[t+1] korelasyonu -0.096, D4). UYARI: bu iliski buyuk olasilikla
                  saat damgasi uyumsuzlugundan dogar; kazanc teorik fiyatin olcum bicimine
                  ozgu olabilir ve gercek piyasada kullanilamayabilir.

Karar kurali (sonuclardan ONCE; preregistration.json)
-----------------------------------------------------
  Bir model "rastgele yuruyusten anlamli olarak iyi" sayilir: MAE_TL dusuk VE
  DM (mutlak TL hata, iki yonlu) Holm p < 0.05 (istatistik < 0) VE blok bootstrap
  %95 GA (MAE_TL farki) tamamen sifirin altinda VE pencerelerin >= %60'inda daha iyi.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models import train as tr

DEV_START = "2018-01-01"
REFERENCE = "Zero-Return"
NEW_MODELS = ["Mean-250", "FX-AR1", "Bilesen-Gecikme"]
ARTIFACT_FLAG = {"Bilesen-Gecikme": "ons[t]-kur[t+1] iliskisi saat damgasi uyumsuzlugundan dogabilir"}
ALPHA = 0.05
MIN_WIN_SHARE = 0.60
TL_THRESHOLDS = (0.40, 1.0, 5.0, 10.0, 25.0)
PCT_THRESHOLDS = (0.10, 0.25, 0.50, 1.0)
BOOT_BLOCK, BOOT_REPS, BOOT_SEED = 20, 2000, 42


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "point_eval"


# ---------------------------------------------------------------------------
# Hizalama kontrolu (gercek veride her calistirmada)
# ---------------------------------------------------------------------------
def verify_price_alignment(df: pd.DataFrame, tol: float = 1e-6) -> dict:
    """P[t] * exp(target[t]) == P[t+1] olmali. Ihlalde ValueError."""
    p = df[tr.PRICE_COL].to_numpy(float)
    y = df[tr.TARGET].to_numpy(float)
    implied, nxt = p[:-1] * np.exp(y[:-1]), p[1:]
    rel = np.abs(implied / nxt - 1)
    if not np.all(rel < tol):
        bad = int(np.argmax(rel))
        raise ValueError(f"Hedef-fiyat hizalamasi bozuk: satir {bad} ({df['Date'].iat[bad].date()}), "
                         f"goreli fark {rel[bad]:.2e}.")
    return {"rows_checked": int(len(rel)), "max_relative_diff": float(rel.max())}


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def point_metrics(actual_ret, pred_ret, price_t) -> dict:
    a, p, P = (np.asarray(x, float) for x in (actual_ret, pred_ret, price_t))
    actual_price, pred_price = P * np.exp(a), P * np.exp(p)
    err = pred_price - actual_price
    abs_tl = np.abs(err)
    ape = 100 * abs_tl / actual_price
    up, down = a > 0, a < 0
    out = {"n": int(len(a)),
           "MAE_TL": float(abs_tl.mean()), "MedAE_TL": float(np.median(abs_tl)),
           "RMSE_TL": float(np.sqrt(np.mean(err ** 2))), "MAPE_pct": float(ape.mean()),
           "MdAPE_pct": float(np.median(ape)), "MAE_return": float(np.mean(np.abs(a - p))),
           "bias_TL": float(err.mean()),
           "direction_accuracy_pct": tr.direction_accuracy(a, p),
           "up_day_recall_pct": float(100 * np.mean(p[up] > 0)) if up.any() else np.nan,
           "down_day_recall_pct": float(100 * np.mean(p[down] < 0)) if down.any() else np.nan,
           "MAE_TL_up_days": float(abs_tl[up].mean()) if up.any() else np.nan,
           "MAE_TL_down_days": float(abs_tl[down].mean()) if down.any() else np.nan,
           "up_fraction_pct": tr.up_fraction(a)}
    big = np.abs(a) >= np.quantile(np.abs(a), 0.90)
    out["MAE_TL_large_moves"] = float(abs_tl[big].mean())
    out["MAE_TL_normal_days"] = float(abs_tl[~big].mean())
    for t in TL_THRESHOLDS:
        out[f"share_err_le_{t:g}TL_pct"] = float(100 * np.mean(abs_tl <= t))
    for t in PCT_THRESHOLDS:
        out[f"share_err_le_{t:g}pct_pct"] = float(100 * np.mean(ape <= t))
    return out


def abs_tl_errors(pred: pd.DataFrame, model: str) -> np.ndarray:
    P = pred["price_t"].to_numpy(float)
    return np.abs(P * np.exp(pred[f"pred_{model}"].to_numpy(float)) - P * np.exp(pred["actual_return"].to_numpy(float)))


def block_bootstrap_diff_ci(e_model, e_ref, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED) -> tuple:
    """Eslestirilmis hareketli blok bootstrap: ort(e_model) - ort(e_ref) icin %95 GA (TL)."""
    d = np.asarray(e_model, float) - np.asarray(e_ref, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    stats = np.empty(reps)
    for i in range(reps):
        idx = (rng.integers(0, n - block + 1, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        stats[i] = d[idx].mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------------------
# Tahmin uretimi
# ---------------------------------------------------------------------------
def new_model_predictions(df: pd.DataFrame, windows: list) -> dict:
    """Yeni adaylar; mevcut walk-forward ile ayni pencere ve purge."""
    y = df[tr.TARGET].to_numpy(float)
    fx = df["usdtry_return"].to_numpy(float)
    gd = df["gold_usd_return"].to_numpy(float)
    out = {m: [] for m in NEW_MODELS}
    for pos in windows:
        ti = np.arange(len(tr._purge_train(df, pos[0])))
        out["Mean-250"].extend(np.full(len(pos), float(y[ti][-250:].mean())))
        m1 = LinearRegression().fit(fx[ti, None], y[ti])
        out["FX-AR1"].extend(m1.predict(fx[pos, None]))
        X = np.column_stack([fx, gd])
        m2 = LinearRegression().fit(X[ti], y[ti])
        out["Bilesen-Gecikme"].extend(m2.predict(X[pos]))
    return {k: np.asarray(v) for k, v in out.items()}


def generate(df: pd.DataFrame, start: str = DEV_START, end: str = None) -> dict:
    end = tr.TEST_START if end is None else end
    with contextlib.redirect_stdout(io.StringIO()):
        wf = tr.walk_forward_evaluate(df, start=start, end=end, label="nokta")
    pred = wf["predictions"].copy()
    positions = tr.eval_positions(df, start, end)
    windows = tr.make_windows(positions, tr.RETRAIN_EVERY)
    dates = df["Date"].iloc[positions].to_numpy()
    if not np.array_equal(pd.to_datetime(pred["prediction_date"]).to_numpy(), dates):
        raise ValueError("Yeni adaylarin satirlari mevcut walk-forward ile eslesmiyor.")
    for name, p in new_model_predictions(df, windows).items():
        pred[f"pred_{name}"] = p
    pred["window"] = wf["window_of_row"]
    return {"predictions": pred, "windows": wf["windows"]}


def model_names(pred: pd.DataFrame) -> list:
    return [c[len("pred_"):] for c in pred.columns if c.startswith("pred_")]


# ---------------------------------------------------------------------------
# Raporlar
# ---------------------------------------------------------------------------
def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"model": m, **point_metrics(pred["actual_return"], pred[f"pred_{m}"], pred["price_t"])}
                         for m in model_names(pred)])


def comparison_table(pred: pd.DataFrame, reference: str) -> pd.DataFrame:
    e_ref = abs_tl_errors(pred, reference)
    rows = []
    for m in model_names(pred):
        if m == reference:
            continue
        e = abs_tl_errors(pred, m)
        dm = tr.diebold_mariano(e, e_ref)          # mutlak TL hatalari; |.| degistirmez
        lo, hi = block_bootstrap_diff_ci(e, e_ref)
        wins = pred.assign(a=e, b=e_ref).groupby("window")[["a", "b"]].mean()
        rows.append({"model": m, "reference": reference,
                     "MAE_TL_model": float(e.mean()), "MAE_TL_reference": float(e_ref.mean()),
                     "diff_TL": float(e.mean() - e_ref.mean()), "diff_ci95_low": lo, "diff_ci95_high": hi,
                     "improvement_pct": float(100 * (1 - e.mean() / e_ref.mean())),
                     "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                     "win_share": float((wins["a"] < wins["b"]).mean()),
                     "artifact_warning": ARTIFACT_FLAG.get(m, "")})
    out = pd.DataFrame(rows)
    out["p_holm"] = tr.holm_adjust(out["p_value"])
    return apply_rule(out)


def apply_rule(out: pd.DataFrame) -> pd.DataFrame:
    """On kayitli kural: dort kosulun hepsi."""
    out = out.copy()
    out["significantly_better"] = ((out["diff_TL"] < 0) & (out["dm_stat"] < 0) & (out["p_holm"] < ALPHA)
                                   & (out["diff_ci95_high"] < 0) & (out["win_share"] >= MIN_WIN_SHARE))
    return out


def grouped_table(pred: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for g, part in pred.groupby(key):
        for m in model_names(pred):
            pm = point_metrics(part["actual_return"], part[f"pred_{m}"], part["price_t"])
            rows.append({label: g, "model": m, "n": pm["n"], "MAE_TL": pm["MAE_TL"], "MedAE_TL": pm["MedAE_TL"],
                         "MAPE_pct": pm["MAPE_pct"], "direction_accuracy_pct": pm["direction_accuracy_pct"]})
    return pd.DataFrame(rows)


def volatility_regime(df: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    """Tahmin gunundeki gecmis 60 gunluk gram oynakligi (yalnizca t ve oncesi) -> uclu dilim.

    Esikler gelistirme donemindeki dagilimdan; yalnizca raporlama icin (model secimi icin degil).
    """
    vol = df.set_index("Date")["gold_return"].rolling(60, min_periods=60).std()
    v = vol.reindex(pd.to_datetime(pred["prediction_date"])).to_numpy()
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    return pd.Series(np.where(v <= q1, "1_sakin", np.where(v <= q2, "2_orta", "3_oynak")), index=pred.index)


def preregistration() -> dict:
    files = {"point_eval.py": Path(__file__), "train.py": Path(tr.__file__)}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "period": {"start": DEV_START, "end_exclusive": tr.TEST_START}, "reference": REFERENCE,
            "existing_models": list(tr.get_models()), "new_models": NEW_MODELS, "artifact_flags": ARTIFACT_FLAG,
            "rule": {"alpha_holm": ALPHA, "min_win_share": MIN_WIN_SHARE, "ci": "blok bootstrap %95 tamamen < 0",
                     "loss": "mutlak TL hata"},
            "nature": "kesif; nokta tahmini icin bagimsiz teyit ancak 2026 ile mumkun",
            "code_md5": {k: hashlib.md5(v.read_bytes()).hexdigest() for k, v in files.items()}}


def save_plot(pred: pd.DataFrame, comp: pd.DataFrame, target: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    e_ref = abs_tl_errors(pred, REFERENCE)
    d = pd.to_datetime(pred["prediction_date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    for m in ["Mean-Return", "FX-AR1", "Bilesen-Gecikme", "Random Forest"]:
        if f"pred_{m}" in pred:
            ax.plot(d, np.cumsum(abs_tl_errors(pred, m) - e_ref), lw=0.9, label=m)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kumulatif |hata| farki (TL)")
    ax.set_title("Rastgele yuruyuse gore kumulatif mutlak TL hata farki (asagi = model daha iyi)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(target / "cumulative_error_vs_random_walk.png", dpi=110)
    plt.close()


def run(data_path=None) -> dict:
    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    prereg = preregistration()
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)              # hesaplamadan ONCE
    df = tr.load_data(data_path)
    df = df[df["Date"] < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    alignment = verify_price_alignment(df)
    gen = generate(df)
    pred = gen["predictions"]
    if pd.to_datetime(pred["target_date"]).max() >= pd.Timestamp(tr.TEST_START):
        raise ValueError("Degerlendirme test donemine tasti; durduruldu.")
    metrics = metrics_table(pred)
    comp_rw = comparison_table(pred, REFERENCE)
    comp_mean = comparison_table(pred, "Mean-Return")
    annual = grouped_table(pred, pd.to_datetime(pred["prediction_date"]).dt.year.rename("year"), "year")
    regime = grouped_table(pred, volatility_regime(df, pred).rename("regime"), "regime")
    for name, frame in (("metrics", metrics), ("comparison_vs_random_walk", comp_rw),
                        ("comparison_vs_mean_return", comp_mean), ("annual", annual), ("regime", regime),
                        ("predictions", pred)):
        frame.to_csv(target / f"{name}.csv", index=False)
    save_plot(pred, comp_rw, target)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_forecasts": int(len(pred)),
               "period": [str(pd.Timestamp(pred["prediction_date"].min()).date()),
                          str(pd.Timestamp(pred["target_date"].max()).date())],
               "alignment_check": alignment,
               "significantly_better_than_random_walk": comp_rw.loc[comp_rw["significantly_better"], "model"].tolist(),
               "nature": "kesif (gelistirme donemi)", "test_period_used": False, "holdout_2026_used": False}
    with open(target / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    with open(target / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at_utc": summary["created_at_utc"],
                            "significant": summary["significantly_better_than_random_walk"],
                            "code_md5": prereg["code_md5"]}, ensure_ascii=False) + "\n")
    return {"metrics": metrics, "comparison_rw": comp_rw, "comparison_mean": comp_mean, "annual": annual,
            "regime": regime, "summary": summary, "predictions": pred}


def print_report(res: dict):
    pd.set_option("display.width", 200)
    s = res["summary"]
    a = s["alignment_check"]
    print(f"  Hizalama kontrolu: {a['rows_checked']} satirda P[t]*exp(hedef) = P[t+1] (max goreli fark {a['max_relative_diff']:.1e})")
    print(f"  Donem: {s['period'][0]} .. {s['period'][1]} | {s['n_forecasts']} tahmin | KESIF (gelistirme donemi)\n")
    m = res["metrics"]
    print("  NOKTA TAHMINI (TL):")
    print(m[["model", "MAE_TL", "MedAE_TL", "RMSE_TL", "MAPE_pct", "MdAPE_pct", "bias_TL"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  YON:")
    print(m[["model", "direction_accuracy_pct", "up_day_recall_pct", "down_day_recall_pct", "up_fraction_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n  HASSASIYET (hata esigin altinda kalan gunlerin yuzdesi):")
    cols = ["model"] + [c for c in m.columns if c.startswith("share_err_le_")]
    print(m[cols].rename(columns=lambda c: c.replace("share_err_le_", "<=").replace("_pct", "")
                         .replace("pct", "%")).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n  BUYUK HAREKET GUNLERI (|getiri| en buyuk %10) ve NORMAL GUNLER, MAE_TL:")
    print(m[["model", "MAE_TL_large_moves", "MAE_TL_normal_days", "MAE_TL_up_days", "MAE_TL_down_days"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  RASTGELE YURUYUSE (yarin = bugun) GORE:")
    c = res["comparison_rw"]
    print(c[["model", "diff_TL", "diff_ci95_low", "diff_ci95_high", "improvement_pct", "p_value", "p_holm",
             "win_share", "significantly_better"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    flagged = c[c["artifact_warning"] != ""]
    for _, r in flagged.iterrows():
        print(f"    UYARI {r['model']}: {r['artifact_warning']}")
    print("\n  OYNAKLIK REJIMINE GORE MAE_TL:")
    print(res["regime"].pivot(index="model", columns="regime", values="MAE_TL").to_string(float_format=lambda v: f"{v:.2f}"))
    sig = s["significantly_better_than_random_walk"]
    print(f"\n  Rastgele yuruyusten anlamli olarak iyi (on kayitli kural): {sig or 'yok'}")
    print("  Not: Kesif. Nokta tahmini icin bagimsiz teyit ancak 2026 verisiyle, on kayitli tek seferlik protokolle mumkun.")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Nokta tahmini degerlendirmesi (TL, gelistirme donemi).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nNOKTA TAHMINI DEGERLENDIRMESI - TEORIK GRAM ALTIN (TL)\n{sep}")
    try:
        res = run()
    except (FileNotFoundError, ValueError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(res)
    print(f"\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
