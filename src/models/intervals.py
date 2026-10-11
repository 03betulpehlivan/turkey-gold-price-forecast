"""
Aralik ve Oynaklik Tahmini (Adim 2b)
====================================
Kullanim (proje kokunden):
  python -m src.models.intervals

Soru: Ertesi gun gram altin log getirisi icin oynaklik modelleri, sabit
oynaklik varsayimina gore daha iyi (dogru kapsamli ve daha dar) tahmin
araliklari uretiyor mu?

Hedef ve veri
-------------
  y[t] = target_next_return[t] = log(P[t+1] / P[t])   (Faz A/B ile ayni hedef)
  Veri: train.load_data() -> model_dataset.csv (2025-12-30'a kadar; 2026 yok).
  Degerlendirme: DEV_START (2018-01-01) .. TEST_START oncesi; hedef tarihi
  TEST_START'tan once olan satirlar. Test donemi ve 2026 KULLANILMAZ.
  Walk-forward: RETRAIN_EVERY gozlemde bir, genisleyen pencere, 1 satir purge
  (train.py ile ayni pencere araclari).

Tahmin yapisi (tum yontemlerde ayni)
------------------------------------
  Merkez:   mu = egitim penceresindeki hedef ortalamasi (Mean-Return).
  Olcek:    sigma[t] = yonteme gore y[t] icin oynaklik tahmini (yalnizca t ve oncesi).
  Kantil:   q_tau[t] = mu + sigma[t] * z_tau
            z_tau: egitim penceresindeki standartlastirilmis artiklarin
            (y - mu) / sigma empirik kantili (filtrelenmis tarihsel simulasyon).
  Aralik:   %80 = [q_0.10, q_0.90], %95 = [q_0.025, q_0.975]

Yontemler (sigma)
-----------------
  static      egitim hedeflerinin standart sapmasi (sabit; referans)
  rolling_20  son 20 gram getirisinin standart sapmasi
  rolling_60  son 60 gram getirisinin standart sapmasi
  ewma        RiskMetrics: v[t] = 0.94 v[t-1] + 0.06 r[t]^2
  garch       GARCH(1,1), arch paketi; egitim penceresinde yeniden tahmin
  *_cal       ayni yontem + takvim carpani: Cuma (ertesi gozlem hafta sonu
              sonrasi) icin egitimden tahmin edilen oynaklik orani

Olcutler
--------
  pinball  dort kantilin ortalama kantil kaybi (dusuk iyi; uygun skor kurali)
  kapsam   gercek degerin aralik icinde kalma orani (%80 ve %95 hedef)
  genislik aralik genisligi (getiri ve TL)
  Kupiec   kosulsuz kapsam LR testi; Christoffersen ihlallerin bagimsizligi testi
  DM       satir bazinda pinball kaybi; her yontem static'e karsi, Holm duzeltmesi

Teyit testi (Adim 2c): python -m src.models.intervals --confirm
--------------------------------------------------------------------
  Dondurulmus GARCH (bu dosyadaki ayarlar degistirilmeden) ile static,
  TEST donemi (2023-10-26 .. 2025-12-30) uzerinde, TEK onceden kayitli
  hipotezle karsilastirilir. On kayit, hesaplamadan ONCE diske yazilir.
  Bir kez calistirilir; tekrar icin --force-rerun gerekir ve her calistirma
  kayda gecer. 2026 (son sinav) verisi kullanilirsa hesaplama durur.

Karar kurali (sonuclardan ONCE tanimlandi)
------------------------------------------
  Bir yontem kabul edilir: (1) pinball static'ten dusuk ve DM (Holm) p < 0.05,
  (2) %80 kapsam 0.80 +/- 0.03 ve %95 kapsam 0.95 +/- 0.015,
  (3) pencerelerin en az %60'inda static'ten dusuk pinball.
  Kabul edilenlerden en dusuk pinball'li olan aday secilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from src.models import train as tr

DEV_START = "2018-01-01"
QUANTILES = (0.025, 0.10, 0.90, 0.975)
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
EWMA_LAMBDA = 0.94
METHODS = ["static", "rolling_20", "rolling_60", "ewma", "ewma_cal", "garch", "garch_cal"]
REFERENCE = "static"
COVERAGE_TOL = {80: 0.03, 95: 0.015}
ALPHA = 0.05
MIN_WINDOW_WIN_SHARE = 0.60
CLUSTER_LAGS = 10

# Teyit testi (Adim 2c) - sonuclardan ONCE sabitlendi
CONFIRM_METHOD = "garch"
CONFIRM_REFERENCE = "static"
CONFIRM_ALPHA = 0.05
HOLDOUT_START = "2026-01-01"
BOOT_BLOCK = 20
BOOT_REPS = 2000
BOOT_SEED = 42
CONFIRM_HYPOTHESIS = (
    "Adim 2b'deki GARCH(1,1) oynaklik yontemi, hicbir ayari degistirilmeden, "
    "test doneminde (2023-10-26 .. 2025-12-30) sabit oynaklik (static) "
    "yontemine gore daha dusuk ortalama pinball kaybi verir. Birincil test: "
    "satir bazinda pinball kaybi uzerinde iki yonlu Diebold-Mariano, "
    "alpha = 0.05, tek karsilastirma (coklu test duzeltmesi yok). Destek "
    "kosulu: dm_stat < 0 ve p < 0.05."
)


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "intervals"


# ---------------------------------------------------------------------------
# Oynaklik tahmin edicileri (hepsi yalnizca t ve oncesini kullanir)
# ---------------------------------------------------------------------------
def ewma_sigma(r: np.ndarray, lam: float = EWMA_LAMBDA, init_n: int = 20) -> np.ndarray:
    """sigma[t]: y[t] = r[t+1] icin; v[t] = lam*v[t-1] + (1-lam)*r[t]^2."""
    v = np.empty(len(r))
    prev = float(np.mean(r[:init_n] ** 2))
    for t, x in enumerate(r):
        prev = lam * prev + (1 - lam) * x * x
        v[t] = prev
    return np.sqrt(v)


def rolling_sigma(r: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(r).rolling(n, min_periods=n).std().to_numpy()


def fit_garch(r_train: np.ndarray) -> dict:
    """GARCH(1,1), sabit ortalama, normal QMLE. Getiriler yuzde olcegiyle (x100) tahmin edilir."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(100 * r_train, mean="Constant", vol="GARCH", p=1, q=1,
                         dist="normal", rescale=False).fit(disp="off")
    p = res.params
    return {"mu": float(p["mu"]), "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "converged": bool(res.convergence_flag == 0),
            "train_var_pct": float(np.var(100 * r_train))}


def garch_sigma(r: np.ndarray, params: dict) -> np.ndarray:
    """sigma[t] (y[t]=r[t+1] icin): v[t] = omega + alpha*(r[t]-mu)^2 + beta*v[t-1], yuzde olcekte."""
    x = 100 * r
    pers = params["alpha"] + params["beta"]
    prev = params["omega"] / (1 - pers) if pers < 1 else params["train_var_pct"]
    v = np.empty(len(x))
    for t, xt in enumerate(x):
        prev = params["omega"] + params["alpha"] * (xt - params["mu"]) ** 2 + params["beta"] * prev
        v[t] = prev
    return np.sqrt(v) / 100


# ---------------------------------------------------------------------------
# Olcutler
# ---------------------------------------------------------------------------
def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def kupiec_p(hits: np.ndarray, expected_rate: float) -> float:
    """Kosulsuz kapsam: ihlal orani beklenen orana esit mi? (LR ~ chi2(1))."""
    n, x = len(hits), int(np.sum(hits))
    if n == 0:
        return np.nan
    phat = x / n
    lr = -2 * (xlogy(n - x, 1 - expected_rate) + xlogy(x, expected_rate)
               - xlogy(n - x, 1 - phat) - xlogy(x, phat))
    return float(stats.chi2.sf(max(lr, 0.0), 1))


def christoffersen_p(hits: np.ndarray) -> float:
    """Ihlallerin bagimsizligi (birinci derece Markov, LR ~ chi2(1))."""
    h = np.asarray(hits, dtype=int)
    if len(h) < 3:
        return np.nan
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    return float(stats.chi2.sf(max(-2 * (l0 - l1), 0.0), 1))


def ljung_box(x: np.ndarray, lags: int = CLUSTER_LAGS) -> tuple:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, d = len(x), x - x.mean()
    denom = float(d @ d)
    rho = [float(d[k:] @ d[:-k]) / denom for k in range(1, lags + 1)]
    q = n * (n + 2) * sum(r * r / (n - k) for k, r in enumerate(rho, start=1))
    return rho, float(q), float(stats.chi2.sf(q, lags))


def summarize(pred: pd.DataFrame, method: str) -> dict:
    y = pred["y"].to_numpy()
    loss = np.mean([pinball(y, pred[f"{method}_q{tau}"].to_numpy(), tau) for tau in QUANTILES], axis=0)
    out = {"n": len(y), "pinball": float(loss.mean())}
    for lvl, (lo, hi) in LEVELS.items():
        ql, qh = pred[f"{method}_q{lo}"].to_numpy(), pred[f"{method}_q{hi}"].to_numpy()
        inside = (y >= ql) & (y <= qh)
        price = pred["price_t"].to_numpy()
        out[f"coverage_{lvl}"] = float(inside.mean())
        out[f"width_{lvl}"] = float(np.mean(qh - ql))
        out[f"width_{lvl}_TL"] = float(np.mean(price * (np.exp(qh) - np.exp(ql))))
        out[f"kupiec_{lvl}_p"] = kupiec_p(~inside, 1 - lvl / 100)
        out[f"christoffersen_{lvl}_p"] = christoffersen_p(~inside)
    return out


def row_loss(pred: pd.DataFrame, method: str) -> np.ndarray:
    y = pred["y"].to_numpy()
    return np.mean([pinball(y, pred[f"{method}_q{tau}"].to_numpy(), tau) for tau in QUANTILES], axis=0)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def _fhs_quantiles(y_tr, mu, sig_tr):
    ok = np.isfinite(sig_tr) & (sig_tr > 0)
    z = (y_tr[ok] - mu) / sig_tr[ok]
    return {tau: float(np.quantile(z, tau)) for tau in QUANTILES}, z, ok


_DEFAULT_END = object()


def walk_forward_intervals(df: pd.DataFrame, start: str = DEV_START, end=_DEFAULT_END) -> dict:
    """Her pencerede: mu, z-kantilleri, GARCH ve takvim carpani YALNIZCA egitimden.

    end varsayilani TEST_START'tir (gelistirme). end=None: start'tan verinin
    sonuna kadar (teyit testi; test donemi).
    """
    end = tr.TEST_START if end is _DEFAULT_END else end
    positions = tr.eval_positions(df, start, end)
    if len(positions) < tr.MIN_EVAL_ROWS:
        raise ValueError(f"Degerlendirme donemi yetersiz: {len(positions)} gozlem.")
    windows = tr.make_windows(positions, tr.RETRAIN_EVERY)

    r = df["gold_return"].to_numpy(float)
    y = df[tr.TARGET].to_numpy(float)
    friday = (df["day_of_week"].to_numpy() == 4)
    base_sig = {"rolling_20": rolling_sigma(r, 20), "rolling_60": rolling_sigma(r, 60),
                "ewma": ewma_sigma(r)}

    rows, info = [], []
    for w, pos in enumerate(windows, start=1):
        train_df = tr._purge_train(df, pos[0])
        tr_idx = np.arange(len(train_df))
        y_tr = y[tr_idx]
        mu = float(y_tr.mean())
        gp = fit_garch(r[tr_idx])
        sig = dict(base_sig)
        sig["static"] = np.full(len(r), float(y_tr.std(ddof=1)))
        sig["garch"] = garch_sigma(r, gp)
        winfo = {"window": w, "train_last_date": str(train_df["Date"].iloc[-1].date()),
                 "eval_first_date": str(df["Date"].iat[pos[0]].date()),
                 "eval_last_date": str(df["Date"].iat[pos[-1]].date()), "n_eval": len(pos),
                 "n_train": len(tr_idx), "mu": mu, **{f"garch_{k}": v for k, v in gp.items()}}
        for base in ("ewma", "garch"):
            _, z, ok = _fhs_quantiles(y_tr, mu, sig[base][tr_idx])
            fri = friday[tr_idx][ok]
            factor = float(np.std(z[fri], ddof=1) / np.std(z[~fri], ddof=1))
            sig[f"{base}_cal"] = sig[base] * np.where(friday, factor, 1.0)
            winfo[f"{base}_cal_factor"] = factor
        info.append(winfo)

        rec = {"window": np.full(len(pos), w), "prediction_date": df["Date"].iloc[pos].to_numpy(),
               "y": y[pos], "price_t": df[tr.PRICE_COL].to_numpy(float)[pos], "mu": np.full(len(pos), mu),
               "friday": friday[pos]}
        for m in METHODS:
            zq, _, _ = _fhs_quantiles(y_tr, mu, sig[m][tr_idx])
            rec[f"{m}_sigma"] = sig[m][pos]
            for tau in QUANTILES:
                rec[f"{m}_q{tau}"] = mu + sig[m][pos] * zq[tau]
        rows.append(pd.DataFrame(rec))
    pred = pd.concat(rows, ignore_index=True)
    pred["target_date"] = tr._get_target_dates(df, [p for w in windows for p in w])
    return {"predictions": pred, "windows": info}


# ---------------------------------------------------------------------------
# Raporlama ve karar
# ---------------------------------------------------------------------------
def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    periods = {f"dev_{DEV_START[:4]}_{pd.Timestamp(tr.TEST_START).year}": pred,
               "validation_2022_2023": pred[pd.to_datetime(pred["prediction_date"]) >= pd.Timestamp(tr.VALIDATION_START)]}
    rows = []
    for label, p in periods.items():
        for m in METHODS:
            rows.append({"period": label, "method": m, **summarize(p, m)})
    return pd.DataFrame(rows)


def window_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, g in pred.groupby("window"):
        for m in METHODS:
            s = summarize(g, m)
            rows.append({"window": int(w), "method": m, "n": s["n"], "pinball": s["pinball"],
                         "coverage_80": s["coverage_80"], "coverage_95": s["coverage_95"]})
    return pd.DataFrame(rows)


def annual_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = pd.to_datetime(pred["prediction_date"]).dt.year
    for yr, g in pred.groupby(years):
        for m in METHODS:
            s = summarize(g, m)
            rows.append({"year": int(yr), "method": m, "n": s["n"], "pinball": s["pinball"],
                         "coverage_80": s["coverage_80"], "coverage_95": s["coverage_95"],
                         "width_80": s["width_80"]})
    return pd.DataFrame(rows)


def significance_table(pred: pd.DataFrame) -> pd.DataFrame:
    ref = row_loss(pred, REFERENCE)
    rows = []
    for m in METHODS:
        if m == REFERENCE:
            continue
        r = tr.diebold_mariano(row_loss(pred, m), ref)   # kayiplar >= 0: |kayip| = kayip
        rows.append({"method": m, "reference": REFERENCE, "n": r["n"], "mean_loss_diff": r["mean_loss_diff"],
                     "dm_stat": r["dm_stat"], "p_value": r["p_value"], "hac_lag": r["hac_lag"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = tr.holm_adjust(out["p_value"])
    return out


def decide(metrics: pd.DataFrame, windows: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    dev = metrics[metrics["period"].str.startswith("dev_")].set_index("method")
    ref_pin = dev.loc[REFERENCE, "pinball"]
    wpiv = windows.pivot(index="window", columns="method", values="pinball")
    s = sig.set_index("method")
    rows = []
    for m in METHODS:
        if m == REFERENCE:
            continue
        lower = bool(dev.loc[m, "pinball"] < ref_pin)
        significant = bool(s.loc[m, "dm_stat"] < 0 and s.loc[m, "p_holm"] < ALPHA)
        cov_ok = all(abs(dev.loc[m, f"coverage_{l}"] - l / 100) <= COVERAGE_TOL[l] for l in LEVELS)
        win_share = float((wpiv[m] < wpiv[REFERENCE]).mean())
        rows.append({"method": m, "pinball": dev.loc[m, "pinball"], "pinball_static": ref_pin,
                     "pinball_improvement_pct": 100 * (1 - dev.loc[m, "pinball"] / ref_pin),
                     "lower_than_static": lower, "dm_holm_significant": significant,
                     "coverage_80": dev.loc[m, "coverage_80"], "coverage_95": dev.loc[m, "coverage_95"],
                     "coverage_ok": cov_ok, "window_win_share": win_share,
                     "windows_ok": win_share >= MIN_WINDOW_WIN_SHARE,
                     "accepted": lower and significant and cov_ok and win_share >= MIN_WINDOW_WIN_SHARE})
    return pd.DataFrame(rows)


def volatility_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Oynaklik kumelenmesi kaniti: yalnizca test oncesi (ANALYSIS_END'e kadar) gram getirisi."""
    r = df.loc[df["Date"] < pd.Timestamp(tr.TEST_START), "gold_return"].to_numpy(float)
    rows = []
    for label, x in (("r", r), ("|r|", np.abs(r)), ("r^2", r * r)):
        rho, q, p = ljung_box(x)
        rows.append({"series": label, "n": len(x), **{f"acf_{k}": v for k, v in enumerate(rho, 1)},
                     "ljung_box_Q10": q, "p_value": p})
    return pd.DataFrame(rows)


def save_plots(pred: pd.DataFrame, metrics: pd.DataFrame, method: str, target: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = pred.tail(250)
    dates = pd.to_datetime(p["target_date"])
    price = p["price_t"].to_numpy()
    realized = price * np.exp(p["y"].to_numpy())
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for lvl, alpha_ in ((95, 0.18), (80, 0.35)):
        lo, hi = LEVELS[lvl]
        ax.fill_between(dates, price * np.exp(p[f"{method}_q{lo}"]), price * np.exp(p[f"{method}_q{hi}"]),
                        alpha=alpha_, color="tab:blue", label=f"%{lvl} aralik ({method})", lw=0)
    ax.plot(dates, realized, color="black", lw=0.8, label="gerceklesen teorik fiyat")
    ax.set_ylabel("TL/gram (teorik)")
    ax.set_title("Ertesi gun icin tahmin araliklari - gelistirme doneminin son 250 gunu (test oncesi)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(target / "interval_example.png", dpi=110)
    plt.close()

    dev = metrics[metrics["period"].str.startswith("dev_")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = np.arange(len(dev))
    ax.bar(xs - 0.2, dev["coverage_80"], 0.4, label="%80 kapsam")
    ax.bar(xs + 0.2, dev["coverage_95"], 0.4, label="%95 kapsam")
    ax.axhline(0.80, ls="--", color="tab:blue", lw=0.8)
    ax.axhline(0.95, ls="--", color="tab:orange", lw=0.8)
    ax.set_xticks(xs, dev["method"], rotation=20)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Gerceklesen kapsam oranlari (kesikli cizgi: hedef)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(target / "coverage.png", dpi=110)
    plt.close()


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "bulunamadi"


def run(data_path: Path = None) -> dict:
    df = tr.load_data(data_path)
    clustering = volatility_clustering(df)
    wf = walk_forward_intervals(df)
    pred = wf["predictions"]
    metrics = metrics_table(pred)
    windows = window_table(pred)
    annual = annual_table(pred)
    sig = significance_table(pred)
    decision = decide(metrics, windows, sig)
    accepted = decision[decision["accepted"]].sort_values("pinball")
    best = accepted["method"].iloc[0] if len(accepted) else None

    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics", metrics), ("window_metrics", windows), ("annual", annual),
                        ("significance", sig), ("decision", decision), ("volatility_clustering", clustering),
                        ("predictions", pred), ("windows", pd.DataFrame(wf["windows"]))):
        frame.to_csv(target / f"{name}.csv", index=False)
    save_plots(pred, metrics, best or "garch", target)
    summary = {
        "experiment": "intervals_step2b",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": tr.TARGET_DEFINITION,
        "evaluation_period": {"start": DEV_START, "end_exclusive": tr.TEST_START,
                              "n_forecasts": int(len(pred)), "n_windows": len(wf["windows"])},
        "test_period_used": False, "holdout_2026_used": False,
        "methods": METHODS, "reference": REFERENCE, "quantiles": list(QUANTILES),
        "decision_rule": {"coverage_tolerance": COVERAGE_TOL, "alpha_holm": ALPHA,
                          "min_window_win_share": MIN_WINDOW_WIN_SHARE},
        "selected_method": best,
        "garch_nonconverged_windows": int(sum(not w["garch_converged"] for w in wf["windows"])),
        "packages": {p: _version(p) for p in ("arch", "numpy", "pandas", "scipy")},
    }
    with open(target / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(target / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_timestamp_utc": summary["run_timestamp_utc"], "selected_method": best,
                            "n_forecasts": len(pred)}) + "\n")
    return {"clustering": clustering, "predictions": pred, "metrics": metrics, "windows": windows,
            "annual": annual, "significance": sig, "decision": decision, "summary": summary,
            "window_info": wf["windows"]}


# ---------------------------------------------------------------------------
# Teyit testi (Adim 2c)
# ---------------------------------------------------------------------------
class ConfirmAlreadyRun(Exception):
    """Teyit testi daha once calistirilmis; --force-rerun olmadan tekrarlanmaz."""


def confirm_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "intervals_confirm"


def block_bootstrap_ci(loss_model, loss_ref, block: int = BOOT_BLOCK, reps: int = BOOT_REPS,
                       seed: int = BOOT_SEED) -> tuple:
    """Hareketli blok bootstrap: pinball iyilesme yuzdesi 100*(1 - ort_model/ort_ref) icin %95 GA.

    Ayni blok indeksleri iki kayip serisine birlikte uygulanir (eslestirilmis).
    """
    lm, lr = np.asarray(loss_model, float), np.asarray(loss_ref, float)
    n = len(lm)
    rng = np.random.default_rng(seed)
    starts_max = n - block
    k = int(math.ceil(n / block))
    stats_ = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, starts_max + 1, k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        stats_[i] = 100 * (1 - lm[idx].mean() / lr[idx].mean())
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))


def _code_md5() -> dict:
    files = {"intervals.py": Path(__file__), "train.py": Path(tr.__file__)}
    return {k: hashlib.md5(v.read_bytes()).hexdigest() for k, v in files.items()}


def preregistration() -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": CONFIRM_HYPOTHESIS,
        "method": CONFIRM_METHOD, "reference": CONFIRM_REFERENCE,
        "period": {"start": tr.TEST_START, "end": "model_dataset.csv'nin sonu (2026 oncesi)"},
        "primary_test": {"name": "Diebold-Mariano, satir bazinda pinball kaybi", "two_sided": True,
                         "alpha": CONFIRM_ALPHA, "multiple_comparison_correction": "yok (tek karsilastirma)"},
        "secondary": ["kapsam %80/%95", "Kupiec", "Christoffersen", "aralik genisligi (TL)",
                      f"blok bootstrap %95 GA (blok={BOOT_BLOCK}, tekrar={BOOT_REPS}, tohum={BOOT_SEED})",
                      "yillik kirilim"],
        "frozen_settings": {"quantiles": list(QUANTILES), "retrain_every": tr.RETRAIN_EVERY,
                            "window": "genisleyen, 1 satir purge", "center": "Mean-Return",
                            "quantile_method": "filtrelenmis tarihsel simulasyon",
                            "garch": "GARCH(1,1), sabit ortalama, normal QMLE, x100 olcek"},
        "code_md5": _code_md5(),
    }


def run_confirm(data_path: Path = None, force: bool = False) -> dict:
    """On kaydi yazar, sonra dondurulmus GARCH'i test doneminde static'e karsi sinar."""
    target = confirm_dir()
    result_path = target / "result.json"
    log_path = target / "run_log.jsonl"
    if result_path.exists() and not force:
        raise ConfirmAlreadyRun(f"Teyit testi zaten calistirildi: {result_path}")
    target.mkdir(parents=True, exist_ok=True)
    previous_runs = sum(1 for _ in open(log_path, encoding="utf-8")) if log_path.exists() else 0
    if result_path.exists():   # --force-rerun: onceki sonuc silinmez, arsivlenir
        result_path.rename(target / f"result_run{previous_runs}.json")

    prereg = preregistration()
    prereg["run_number"] = previous_runs + 1
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)          # hesaplamadan ONCE

    df = tr.load_data(data_path)
    wf = walk_forward_intervals(df, start=tr.TEST_START, end=None)
    pred = wf["predictions"]
    if pd.to_datetime(pred["target_date"]).max() >= pd.Timestamp(HOLDOUT_START):
        raise ValueError("Teyit testine 2026 (son sinav) verisi girdi; durduruldu.")
    keep = ["window", "prediction_date", "target_date", "y", "price_t", "mu", "friday"]
    for m in (CONFIRM_REFERENCE, CONFIRM_METHOD):          # tam sutun adlari: garch_cal vb. alinmaz
        keep += [f"{m}_sigma"] + [f"{m}_q{tau}" for tau in QUANTILES]
    pred = pred[keep]

    lm, lr = row_loss(pred, CONFIRM_METHOD), row_loss(pred, CONFIRM_REFERENCE)
    dm = tr.diebold_mariano(lm, lr)
    ci = block_bootstrap_ci(lm, lr)
    supported = bool(dm["dm_stat"] < 0 and dm["p_value"] < CONFIRM_ALPHA)
    metrics = pd.DataFrame([{"method": m, **summarize(pred, m)} for m in (CONFIRM_REFERENCE, CONFIRM_METHOD)])
    years = pd.to_datetime(pred["prediction_date"]).dt.year
    annual = pd.DataFrame([{"year": int(y), "method": m, **summarize(g, m)}
                           for y, g in pred.groupby(years) for m in (CONFIRM_REFERENCE, CONFIRM_METHOD)])
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_number": prereg["run_number"],
        "hypothesis": CONFIRM_HYPOTHESIS,
        "n_forecasts": int(len(pred)), "n_windows": len(wf["windows"]),
        "period": [str(pd.Timestamp(pred["prediction_date"].min()).date()),
                   str(pd.Timestamp(pred["target_date"].max()).date())],
        "pinball": {CONFIRM_REFERENCE: float(lr.mean()), CONFIRM_METHOD: float(lm.mean())},
        "improvement_pct": float(100 * (1 - lm.mean() / lr.mean())),
        "improvement_pct_ci95_block_bootstrap": list(ci),
        "dm": dm, "hypothesis_supported": supported,
        "garch_nonconverged_windows": int(sum(not w["garch_converged"] for w in wf["windows"])),
        "holdout_2026_used": False,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    metrics.to_csv(target / "metrics.csv", index=False)
    annual.to_csv(target / "annual.csv", index=False)
    pred.to_csv(target / "predictions.csv", index=False)
    pd.DataFrame(wf["windows"]).to_csv(target / "windows.csv", index=False)
    save_confirm_plot(pred, target)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_number": result["run_number"], "created_at_utc": result["created_at_utc"],
                            "hypothesis_supported": supported, "p_value": dm["p_value"],
                            "code_md5": prereg["code_md5"]}) + "\n")
    return {"result": result, "metrics": metrics, "annual": annual, "predictions": pred,
            "preregistration": prereg}


def save_confirm_plot(pred: pd.DataFrame, target: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dates = pd.to_datetime(pred["target_date"])
    price = pred["price_t"].to_numpy()
    realized = price * np.exp(pred["y"].to_numpy())
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for lvl, a in ((95, 0.18), (80, 0.35)):
        lo, hi = LEVELS[lvl]
        ax.fill_between(dates, price * np.exp(pred[f"{CONFIRM_METHOD}_q{lo}"]),
                        price * np.exp(pred[f"{CONFIRM_METHOD}_q{hi}"]), alpha=a, color="tab:blue",
                        lw=0, label=f"%{lvl} aralik (GARCH)")
    ax.plot(dates, realized, color="black", lw=0.8, label="gerceklesen teorik fiyat")
    ax.set_ylabel("TL/gram (teorik)")
    ax.set_title("Teyit testi: ertesi gun tahmin araliklari, test donemi (2023-10 - 2025-12)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(target / "interval_test_period.png", dpi=110)
    plt.close()


def print_confirm(res: dict):
    r = res["result"]
    print(f"  Hipotez (on kayitli): {r['hypothesis']}\n")
    print(f"  Donem: {r['period'][0]} .. {r['period'][1]} | {r['n_forecasts']} tahmin, "
          f"{r['n_windows']} pencere | calistirma no: {r['run_number']}")
    print("\n  Ozet:")
    cols = ["method", "pinball", "coverage_80", "coverage_95", "width_80_TL", "width_95_TL",
            "kupiec_80_p", "kupiec_95_p", "christoffersen_80_p", "christoffersen_95_p"]
    print(res["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  Yillik:")
    print(res["annual"][["year", "method", "n", "pinball", "coverage_80", "coverage_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    lo, hi = r["improvement_pct_ci95_block_bootstrap"]
    print(f"\n  Pinball iyilesmesi: %{r['improvement_pct']:.2f}  (blok bootstrap %95 GA: %{lo:.2f} .. %{hi:.2f})")
    print(f"  Diebold-Mariano: istatistik={r['dm']['dm_stat']:+.3f}, p={r['dm']['p_value']:.4f}, n={r['dm']['n']}")
    verdict = "DESTEKLENDI" if r["hypothesis_supported"] else "DESTEKLENMEDI"
    print(f"\n  SONUC: Hipotez {verdict} (on kayitli olcut: dm_stat < 0 ve p < {CONFIRM_ALPHA}).")
    print("  Not: 2026 verisi kullanilmadi. Fiyat teorik gram altindir.")


def print_report(res: dict):
    pd.set_option("display.width", 170)
    c = res["clustering"].set_index("series")
    print("  Oynaklik kumelenmesi (test oncesi, Ljung-Box 10 gecikme):")
    for s in ("r", "|r|", "r^2"):
        print(f"    {s:4s} acf1={c.loc[s, 'acf_1']:+.3f}  acf5={c.loc[s, 'acf_5']:+.3f}  "
              f"Q={c.loc[s, 'ljung_box_Q10']:.1f}  p={c.loc[s, 'p_value']:.2e}")
    s = res["summary"]["evaluation_period"]
    print(f"\n  Degerlendirme: {s['start']} .. {res['summary']['evaluation_period']['end_exclusive']} (haric), "
          f"{s['n_forecasts']} tahmin, {s['n_windows']} pencere")
    m = res["metrics"]
    for period in m["period"].unique():
        print(f"\n  {period}")
        show = m[m["period"] == period][["method", "pinball", "coverage_80", "coverage_95",
                                         "width_80_TL", "kupiec_80_p", "christoffersen_80_p"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  Karar (static'e karsi):")
    d = res["decision"][["method", "pinball_improvement_pct", "dm_holm_significant", "coverage_ok",
                         "window_win_share", "accepted"]]
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Secilen yontem: {res['summary']['selected_method'] or 'yok (hicbiri kurali saglamadi)'}")
    print("  Not: Test donemi ve 2026 kullanilmadi. Fiyat teorik gram altindir.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aralik/oynaklik tahmini (gelistirme deneyi veya teyit testi).")
    ap.add_argument("--confirm", action="store_true",
                    help="Adim 2c: dondurulmus GARCH'i test doneminde tek hipotezle sina (bir kez).")
    ap.add_argument("--force-rerun", action="store_true",
                    help="--confirm ile: onceki sonucu arsivleyip tekrar calistir (kayda gecer).")
    args = ap.parse_args([] if argv is None else argv)
    if args.force_rerun and not args.confirm:
        ap.error("--force-rerun yalnizca --confirm ile kullanilabilir.")
    sep = "=" * 70
    if args.confirm:
        print(f"\n{sep}\nTEYIT TESTI (Adim 2c) - TEST DONEMI, TEK ON KAYITLI HIPOTEZ\n{sep}")
        try:
            res = run_confirm(force=args.force_rerun)
        except ConfirmAlreadyRun as e:
            print(f"  {e}\n  Sonuclar: {confirm_dir() / 'result.json'}\n"
                  "  Tekrar calistirmak bilimsel olarak gerekceli olmali; gerekiyorsa --force-rerun (kayda gecer).")
            return 2
        except (FileNotFoundError, ValueError) as e:
            print(f"  HATA: {e}")
            return 1
        print_confirm(res)
        print(f"\n  Ciktilar: {confirm_dir()}\n{sep}")
        return 0
    print(f"\n{sep}\nARALIK VE OYNAKLIK TAHMINI (Adim 2b)\n{sep}")
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
