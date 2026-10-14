"""
Fiziki Altin Sabah Tahmini: Hata Analizi (YALNIZCA gelistirme donemi 2016-2022)
==============================================================================
Kullanim (proje kokunden; once python -m src.models.physical_eval calismis olmali):
  python -m src.analysis.physical_errors

Amac: bagimsiz sinavdaki (FINAL_EXAM_PHYS_V1) basarisizligin olasi nedenlerini GELISTIRME verisinde
olcerek ayirt etmek. Sinav donemi (hedef tarihi >= 2023-01-01) bu modulde HIC okunmaz: veriler
yuklenir yuklenmez 2023 oncesine kesilir; tahmin dosyasinda kilitli satir varsa durur.

Hipotezler ve analizler:
  H2 (buyukluk abartisi) : Mincer-Zarnowitz y = a + b*yhat (Newey-West 5); b < 1 abarti demek.
                           |yhat| uclu gruplarinda yon isabeti ve RW'ye gore iyilesme.
  H3 (vadeli kontrat)    : GC=F ile Borsa Istanbul USD/ons (spot tabanli, kontrat yok) arasindaki seviye
                           farkinda KALICI kaymalar = kontrat degisimi. Ons kaynagi bu gunlere denk gelen
                           satirlarin hatasi; son kontrat degisiminden beri gecen gun gruplarina gore yanlilik.
  H1 (prim)              : prim seviyesi ve prim degisimi beslilerinde RW'ye gore iyilesme; gelistirme
                           doneminde |prim| > %1 olan gunlerin payi (H1'in burada ne kadar sinanabildigi).
  Yogunlasma             : toplam mutlak hatanin en kotu %1 ve %5 gunden gelen payi.
  Hedef tanimi           : kapanis ve agirlikli ortalama getirilerinin 1. gecikme otokorelasyonu
                           (negatif: kapanis fiyatinda mikroyapi gurultusu) ve RW hatasi.
  Tazelik                : ons kaynagi onceki fiziki gunden eski olan satirlar (ABD tatili).
Bu modul KESIF amaclidir; buradaki hicbir sonuc bagimsiz basari olarak sunulamaz.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import physical_eval as pv
from src.models import train as tr

LOCK = pd.Timestamp(pv.LOCK_START)
MODELS = ["Bosluk", "Bosluk-Ihtiyatli", "Bosluk+Prim", "XGBoost"]
ROLL_STEP = 0.008          # kalici seviye kaymasi esigi (log) - kontrat degisimi adayi
ROLL_WIN = 5               # oncesi/sonrasi medyan penceresi (gun)
ROLL_MIN_GAP = 15          # iki aday arasi en az gun (ayni olayin tekrarlari birlesir)


class ErrorAnalysisError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "diagnostics" / "physical_errors"


def load_dev(pred_path: Path, phys_path: Path, panel_path: Path) -> tuple:
    pred = pd.read_csv(pred_path, parse_dates=["target_date"])
    if (pred["target_date"] >= LOCK).any():
        raise ErrorAnalysisError("Tahmin dosyasinda kilitli doneme (>= 2023) ait satir var; durduruldu.")
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    return pred, phys[phys["Date"] < LOCK].copy(), panel[panel["Date"] < LOCK].copy()


def newey_west_slope(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict:
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    n = len(y)
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xu[L:].T @ xu[:-L]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    se = float(np.sqrt(V[1, 1]))
    return {"slope": float(beta[1]), "intercept": float(beta[0]), "slope_se": se,
            "slope_ci95": [float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)], "n": int(n)}


def abs_err(pred: pd.DataFrame, model: str) -> np.ndarray:
    p = pred["price_t"].to_numpy(float)
    return np.abs(p * (np.exp(pred["actual_return"].to_numpy(float)) - np.exp(pred[f"pred_{model}"].to_numpy(float))))


def improvement(pred: pd.DataFrame, mask, model: str) -> dict:
    m = np.asarray(mask, bool)
    if m.sum() == 0:
        return {"n": 0}
    e, r = abs_err(pred, model)[m], abs_err(pred, "Zero-Return")[m]
    return {"n": int(m.sum()), "MAE_model": float(e.mean()), "MAE_rw": float(r.mean()),
            "improvement_pct": float(100 * (1 - e.mean() / r.mean()))}


# ---------------------------------------------------------------------------
def calibration(pred: pd.DataFrame) -> dict:
    out = {}
    y = pred["actual_return"].to_numpy(float)
    for m in MODELS:
        yh = pred[f"pred_{m}"].to_numpy(float)
        mz = newey_west_slope(yh, y)
        terc = pd.qcut(np.abs(yh), 3, labels=["kucuk", "orta", "buyuk"])
        groups = {}
        for g in ["kucuk", "orta", "buyuk"]:
            k = np.asarray(terc == g)
            hit = float(np.mean(np.sign(yh[k]) == np.sign(y[k])) * 100)
            groups[g] = {"direction_hit_pct": hit, **improvement(pred, k, m)}
        out[m] = {"mincer_zarnowitz": mz, "by_abs_prediction_tercile": groups}
    return out


def detect_rolls(panel: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    """GC=F / BIST USD-ons seviye farkinda kalici kaymalar (kontrat degisimi adaylari)."""
    m = panel[["Date", "gold_usd_oz"]].merge(phys[["Date", "kapanis_usd_ons"]].dropna(), on="Date").sort_values("Date")
    g = np.log(m["gold_usd_oz"].to_numpy(float)) - np.log(m["kapanis_usd_ons"].to_numpy(float))
    rows = []
    for i in range(ROLL_WIN, len(g) - ROLL_WIN):
        before = np.median(g[i - ROLL_WIN:i])
        after = np.median(g[i:i + ROLL_WIN])
        step = after - before
        if abs(step) >= ROLL_STEP:
            rows.append({"Date": m["Date"].iat[i], "step_pct": 100 * step, "i": i})
    if not rows:
        return pd.DataFrame(columns=["Date", "step_pct"])
    cand = pd.DataFrame(rows)
    keep, last = [], -10 ** 9
    for _, grp in cand.groupby((cand["i"].diff() > ROLL_MIN_GAP).cumsum()):
        best = grp.loc[grp["step_pct"].abs().idxmax()]
        if best["i"] - last >= ROLL_MIN_GAP:
            keep.append(best)
            last = best["i"]
    return pd.DataFrame(keep)[["Date", "step_pct"]].reset_index(drop=True)


def roll_effects(pred: pd.DataFrame, frame: pd.DataFrame, rolls: pd.DataFrame) -> dict:
    f = frame[["target_date", "gold_src_next"]].copy()
    f["target_date"] = pd.to_datetime(f["target_date"])
    p = pred.merge(f, on="target_date", how="left")
    src = pd.to_datetime(p["gold_src_next"])
    rd = pd.DatetimeIndex(pd.to_datetime(rolls["Date"])) if len(rolls) else pd.DatetimeIndex([])
    on_roll = src.isin(rd).to_numpy()
    out = {"roll_days_detected": int(len(rolls)), "rows_with_roll_gold_source": int(on_roll.sum()),
           "by_model": {m: {"roll": improvement(p, on_roll, m), "other": improvement(p, ~on_roll, m)} for m in MODELS}}
    if len(rd):
        pos = np.searchsorted(rd.values, src.values, side="right") - 1
        since = np.where(pos >= 0, (src.values - rd.values[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(int), -1)
        bucket = pd.cut(since, [-2, -1, 5, 20, 40, 1000], labels=["once_yok", "0-5", "6-20", "21-40", "41+"])
        bias = {}
        for b in bucket.categories:
            k = np.asarray(bucket == b)
            if k.sum():
                err = p["pred_Bosluk"].to_numpy(float)[k] - p["actual_return"].to_numpy(float)[k]
                bias[str(b)] = {"n": int(k.sum()), "mean_signed_error_bp": float(1e4 * err.mean())}
        out["bosluk_bias_by_days_since_roll"] = bias
    return out


def premium_effects(pred: pd.DataFrame, frame: pd.DataFrame) -> dict:
    f = frame[["target_date", "prim"]].copy()
    f["target_date"] = pd.to_datetime(f["target_date"])
    f["dprim"] = frame["prim"].diff().to_numpy()
    p = pred.merge(f, on="target_date", how="left")
    out = {"dev_share_abs_premium_gt_1pct": float((p["prim"].abs() > 0.01).mean() * 100),
           "dev_premium_pct_quantiles": {q: float(100 * p["prim"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    for col, name in (("prim", "by_premium_quintile"), ("dprim", "by_premium_change_quintile")):
        qs = pd.qcut(p[col], 5, labels=False, duplicates="drop")
        out[name] = {}
        for q in sorted(qs.dropna().unique()):
            k = np.asarray(qs == q)
            out[name][f"Q{int(q) + 1}"] = {"premium_range_pct": [float(100 * p.loc[k, col].min()), float(100 * p.loc[k, col].max())],
                                            **{m: improvement(p, k, m)["improvement_pct"] for m in ("Bosluk", "Bosluk+Prim")},
                                            "n": int(k.sum())}
    return out


def concentration(pred: pd.DataFrame) -> dict:
    out = {}
    for m in ["Zero-Return"] + MODELS:
        e = np.sort(abs_err(pred, m))[::-1]
        tot = e.sum()
        out[m] = {f"share_top_{k}pct": float(100 * e[: max(1, int(len(e) * k / 100))].sum() / tot) for k in (1, 5)}
    r = abs_err(pred, "Zero-Return")
    cut = np.quantile(r, 0.99)
    keep = r < cut
    out["bosluk_improvement_excluding_top1pct_rw_days"] = improvement(pred, keep, "Bosluk")
    return out


def target_noise(phys: pd.DataFrame) -> dict:
    out = {}
    for col in ("kapanis_tl_gr", "agort_tl_gr"):
        s = phys.set_index("Date")[col].dropna()
        s = s[s.index >= pd.Timestamp(pv.DEV_START)]
        r = np.log(s).diff().dropna()
        out[col] = {"lag1_autocorr": float(r.autocorr(1)), "rw_mape_pct": float(100 * np.mean(np.abs(np.expm1(r)))),
                    "n": int(len(r))}
    return out


def staleness(pred: pd.DataFrame, frame: pd.DataFrame) -> dict:
    f = frame[["target_date", "gold_src_next", "Date"]].copy()
    f["target_date"] = pd.to_datetime(f["target_date"])
    p = pred.merge(f, on="target_date", how="left")
    stale = (pd.to_datetime(p["gold_src_next"]) < pd.to_datetime(p["Date"])).to_numpy()
    return {"stale_rows": int(stale.sum()), "bosluk_stale": improvement(p, stale, "Bosluk"),
            "bosluk_fresh": improvement(p, ~stale, "Bosluk")}


def run(pred_path: Path = None, phys_path: Path = pv.PHYS_PATH, panel_path: Path = pv.PANEL_PATH) -> dict:
    pred_path = pred_path or (pv.out_dir() / "predictions.csv")
    pred, phys, panel = load_dev(pred_path, phys_path, panel_path)
    frame = pv.build_frame(phys, panel)
    frame = frame[pd.to_datetime(frame["target_date"]) < LOCK]
    rolls = detect_rolls(panel, phys)
    res = {"nature": "KESIF - yalnizca gelistirme donemi (hedef < 2023-01-01)",
           "period": [str(pred["target_date"].min().date()), str(pred["target_date"].max().date())],
           "n": int(len(pred)), "calibration": calibration(pred), "rolls": roll_effects(pred, frame, rolls),
           "premium": premium_effects(pred, frame), "concentration": concentration(pred),
           "target_noise": target_noise(phys), "staleness": staleness(pred, frame)}
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    rolls.to_csv(d / "roll_days.csv", index=False, date_format="%Y-%m-%d")
    (d / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {**res, "roll_table": rolls}


def print_report(r: dict):
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    print(f"  {r['nature']}\n  Donem: {r['period'][0]} .. {r['period'][1]} | {r['n']} tahmin\n")
    print("  [H2] Kalibrasyon (gercek = a + b x tahmin; b<1: buyukluk abartili):")
    for m, c in r["calibration"].items():
        mz = c["mincer_zarnowitz"]
        t = c["by_abs_prediction_tercile"]
        print(f"    {m:17s} b={mz['slope']:.2f} [{mz['slope_ci95'][0]:.2f}, {mz['slope_ci95'][1]:.2f}] | |tahmin| ucluleri "
              f"isabet/iyilesme: " + " ".join(f"{g}:%{t[g]['direction_hit_pct']:.0f}/{f(t[g].get('improvement_pct'))}%" for g in t))
    ro = r["rolls"]
    print(f"\n  [H3] Kontrat degisimi adaylari: {ro['roll_days_detected']} gun | ons kaynagi bu gunlere denk gelen satir: "
          f"{ro['rows_with_roll_gold_source']}")
    for m, v in ro["by_model"].items():
        print(f"    {m:17s} kontrat gunu iyilesme %{f(v['roll'].get('improvement_pct'))} (n={v['roll']['n']}) | "
              f"diger %{f(v['other'].get('improvement_pct'))}")
    if "bosluk_bias_by_days_since_roll" in ro:
        print("    Bosluk yanliligi (tahmin - gercek, bp), son degisimden beri gun: " +
              " | ".join(f"{k}: {v['mean_signed_error_bp']:+.1f} (n={v['n']})" for k, v in ro["bosluk_bias_by_days_since_roll"].items()))
    pr = r["premium"]
    print(f"\n  [H1] Gelistirmede |prim| > %1 olan gun payi: %{pr['dev_share_abs_premium_gt_1pct']:.1f} | prim %5-%95: "
          f"%{pr['dev_premium_pct_quantiles'][0.05]:.2f} .. %{pr['dev_premium_pct_quantiles'][0.95]:.2f}")
    for name in ("by_premium_quintile", "by_premium_change_quintile"):
        print(f"    {name}: " + " | ".join(f"{q} [{v['premium_range_pct'][0]:.2f},{v['premium_range_pct'][1]:.2f}]% "
                                         f"Bosluk {f(v['Bosluk'])}%" for q, v in pr[name].items()))
    c = r["concentration"]
    print("\n  [Yogunlasma] Toplam mutlak hatanin en kotu %1 / %5 gunden gelen payi: " +
          " | ".join(f"{m} %{v['share_top_1pct']:.0f}/%{v['share_top_5pct']:.0f}" for m, v in c.items() if m.startswith(("Zero", "Bos", "XG"))))
    ex = c["bosluk_improvement_excluding_top1pct_rw_days"]
    print(f"    En oynak %1 gun haric Bosluk iyilesmesi: %{f(ex.get('improvement_pct'))} (n={ex['n']})")
    tn = r["target_noise"]
    print(f"\n  [Hedef] 1. gecikme otokorelasyonu: kapanis {tn['kapanis_tl_gr']['lag1_autocorr']:+.3f} | agirlikli ortalama "
          f"{tn['agort_tl_gr']['lag1_autocorr']:+.3f}  (negatif: kapanista gurultu)")
    st = r["staleness"]
    print(f"  [Tazelik] Ons kaynagi bayat satir: {st['stale_rows']} | Bosluk iyilesmesi bayat %{f(st['bosluk_stale'].get('improvement_pct'))}"
          f" / taze %{f(st['bosluk_fresh'].get('improvement_pct'))}")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Fiziki sabah tahmini hata analizi (gelistirme donemi).").parse_args([] if argv is None else argv)
    sep = "=" * 78
    print(f"\n{sep}\nFIZIKI ALTIN SABAH TAHMINI - HATA ANALIZI (KESIF, 2016-2022)\n{sep}")
    try:
        r = run()
    except (FileNotFoundError, ErrorAnalysisError, pv.PhysicalEvalError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(r)
    print(f"\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
