"""
Faz C: Faktor Grubu Deneyi (getiri ve oynaklik)
===============================================
Kullanim (proje kokunden):
  python -m src.models.factors

Soru: Piyasa panelindeki faktor gruplari (dolar/faiz, risk istahi, emtia,
Turkiye), gram altin icin (a) ertesi gun getiri tahminine, (b) oynaklik ve
aralik tahminine, tahmin aninda bilinen bilgiyle katki veriyor mu?

Veri ve donem
-------------
  Hedef: target_next_return = log(P[t+1]/P[t])  (degismedi)
  model_dataset.csv (2026 yok) + market_panel.csv (2026 satirlari okunmadan atilir)
  Degerlendirme: 2018-01-01 .. TEST_START oncesi (gelistirme). Test donemi ve 2026 KULLANILMAZ.
  Walk-forward: train.py/intervals.py ile ayni pencereler, genisleyen, 1 satir purge.

Faktor ozellikleri (yalnizca t ve oncesi; panel satirlari arasinda)
--------------------------------------------------------------------
  <seri>_r1 = log(X[t]/X[t-1]),   <seri>_r5 = log(X[t]/X[t-5])
  us10y_d1 = Y[t]-Y[t-1], us10y_d5 = Y[t]-Y[t-5]   (getiri seviyesi; fark alinir)
  vix_log = log(VIX[t])
  Oynaklik gorevinde getiri/fark ozelliklerinin MUTLAK degerleri, vix_log ise oldugu gibi kullanilir.

Gorevler
--------
  (a) Getiri: kontrol = kompakt set + Linear Regression (Faz B). Aday = kontrol + grup.
      Olcut: MAE; DM (mutlak hata), Holm (5 grup).
  (b) Oynaklik: kontrol = Adim 2b GARCH(1,1) (degismedi). Aday = GARCH x exp(b'x),
      b egitimde log(|y-mu|) - log(sigma_garch) uzerine EKK ile tahmin edilir,
      carpan [0.5, 2] araliginda sinirlanir; kantiller filtrelenmis tarihsel simulasyon.
      Olcut: pinball; DM, Holm (5 grup); kapsam toleransi.

Karar kurali (sonuclardan ONCE; ayni kayit preregistration.json'a yazilir)
--------------------------------------------------------------------------
  Grup kabul: olcut kontrolden dusuk VE DM (Holm) p < 0.05 (istatistik < 0)
  VE pencerelerin >= %60'inda kontrolden iyi VE (oynaklik icin) kapsam
  %80 +/- 0.03 ve %95 +/- 0.015.
  Bu deney KESIFTIR: gelistirme donemi onceki fazlarda da kullanildi. Kabul
  edilen bir grup, ancak 2026 verisiyle teyit edilirse kesin bulgu sayilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models import intervals as iv
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
HOLDOUT_START = "2026-01-01"
DEV_START = iv.DEV_START
CONTROL_FEATURES = list(tr.FEATURE_SETS["compact"])
GROUPS = {
    "dolar_faiz": ["dxy_r1", "dxy_r5", "us10y_d1", "us10y_d5"],
    "risk": ["vix_log", "vix_r1", "sp500_r1", "sp500_r5"],
    "emtia": ["silver_r1", "silver_r5", "brent_r1", "brent_r5"],
    "turkiye": ["bist100_r1", "bist100_r5", "eurtry_r1", "eurtry_r5"],
}
GROUPS["hepsi"] = [f for g in ("dolar_faiz", "risk", "emtia", "turkiye") for f in GROUPS[g]]
LEVEL_FEATURES = {"vix_log"}
ADJ_CLIP = (0.5, 2.0)
ALPHA = 0.05
MIN_WIN_SHARE = 0.60
COVERAGE_TOL = iv.COVERAGE_TOL


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "phase_c"


# ---------------------------------------------------------------------------
# Ozellikler
# ---------------------------------------------------------------------------
def build_factor_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel satirlari arasinda yalnizca t ve oncesini kullanan faktor ozellikleri."""
    p = panel[panel["Date"] < pd.Timestamp(HOLDOUT_START)].sort_values("Date").reset_index(drop=True)
    out = p[["Date"]].copy()
    for name in ("dxy", "sp500", "silver", "brent", "bist100", "eurtry", "vix"):
        x = np.log(p[name].astype(float))
        out[f"{name}_r1"] = x - x.shift(1)
        out[f"{name}_r5"] = x - x.shift(5)
    y = p["us10y"].astype(float)
    out["us10y_d1"] = y - y.shift(1)
    out["us10y_d5"] = y - y.shift(5)
    out["vix_log"] = np.log(p["vix"].astype(float))
    return out


def load_inputs(data_path=None, panel_path: Path = PANEL_PATH) -> pd.DataFrame:
    df = tr.load_data(data_path)
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    feats = build_factor_features(panel)
    feats["Date"] = feats["Date"].astype("datetime64[ns]")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    merged = df.merge(feats, on="Date", how="left")
    used = sorted(set(GROUPS["hepsi"]))
    n_nan = merged[used].isna().sum()
    if n_nan.any():
        raise ValueError(f"Faktor ozelliklerinde eksik deger var (panel eksik olabilir): {n_nan[n_nan > 0].to_dict()}")
    return merged


def vol_matrix(df: pd.DataFrame, feats: list) -> np.ndarray:
    cols = [df[f].to_numpy(float) if f in LEVEL_FEATURES else np.abs(df[f].to_numpy(float)) for f in feats]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, start: str = DEV_START, end: str = None) -> dict:
    end = tr.TEST_START if end is None else end
    positions = tr.eval_positions(df, start, end)
    windows = tr.make_windows(positions, tr.RETRAIN_EVERY)
    r = df["gold_return"].to_numpy(float)
    y = df[tr.TARGET].to_numpy(float)
    ret_rows, vol_rows, info = [], [], []
    for w, pos in enumerate(windows, start=1):
        train_df = tr._purge_train(df, pos[0])
        ti = np.arange(len(train_df))
        mu = float(y[ti].mean())
        rec = {"window": np.full(len(pos), w), "prediction_date": df["Date"].iloc[pos].to_numpy(),
               "y": y[pos], "price_t": df[tr.PRICE_COL].to_numpy(float)[pos]}

        # (a) getiri
        ret = dict(rec)
        ret["pred_Mean-Return"] = np.full(len(pos), mu)
        ret["pred_Zero-Return"] = np.zeros(len(pos))
        for name, feats in [("control", CONTROL_FEATURES)] + [(g, CONTROL_FEATURES + f) for g, f in GROUPS.items()]:
            m = LinearRegression().fit(df[feats].to_numpy(float)[ti], y[ti])
            ret[f"pred_{name}"] = m.predict(df[feats].to_numpy(float)[pos])
        ret_rows.append(pd.DataFrame(ret))

        # (b) oynaklik
        gp = iv.fit_garch(r[ti])
        sig_g = iv.garch_sigma(r, gp)
        sigmas = {"garch": sig_g}
        winfo = {"window": w, "train_last_date": str(train_df["Date"].iloc[-1].date()),
                 "eval_first_date": str(df["Date"].iat[pos[0]].date()), "n_eval": len(pos),
                 "garch_converged": gp["converged"]}
        target_log = np.log(np.abs(y[ti] - mu) + 1e-6) - np.log(sig_g[ti])
        for g, feats in GROUPS.items():
            X = vol_matrix(df, feats)
            # Ozellikler EGITIM ortalamasindan sapma olarak kullanilir; aksi halde seviye
            # ozellikleri (ornegin vix_log ~ 2.8) carpani her gun ayni sinira kilitler.
            Xc = X - X[ti].mean(axis=0)
            reg = LinearRegression().fit(Xc[ti], target_log)
            adj = np.clip(np.exp(Xc @ reg.coef_), *ADJ_CLIP)
            # Ortalama carpan egitimde 1 olacak sekilde olceklenir; sabit olcek FHS kantillerine birakilir
            adj = adj / np.exp(np.mean(np.log(adj[ti])))
            sigmas[f"garch+{g}"] = sig_g * adj
            winfo[f"adj_range_{g}"] = [float(adj[pos].min()), float(adj[pos].max())]
        vol = dict(rec)
        for name, sig in sigmas.items():
            zq, _, _ = iv._fhs_quantiles(y[ti], mu, sig[ti])
            vol[f"{name}_sigma"] = sig[pos]
            for tau in iv.QUANTILES:
                vol[f"{name}_q{tau}"] = mu + sig[pos] * zq[tau]
        vol_rows.append(pd.DataFrame(vol))
        info.append(winfo)
    return {"returns": pd.concat(ret_rows, ignore_index=True), "vol": pd.concat(vol_rows, ignore_index=True),
            "windows": info}


# ---------------------------------------------------------------------------
# Degerlendirme ve karar
# ---------------------------------------------------------------------------
def evaluate_returns(pred: pd.DataFrame) -> tuple:
    y = pred["y"].to_numpy()
    err = {n[len("pred_"):]: np.abs(y - pred[n].to_numpy()) for n in pred.columns if n.startswith("pred_")}
    rows = [{"model": k, "MAE": float(v.mean()), "skill_vs_control": float(1 - v.mean() / err["control"].mean()),
             "skill_vs_mean": float(1 - v.mean() / err["Mean-Return"].mean())} for k, v in err.items()]
    sig = []
    for g in GROUPS:
        d = tr.diebold_mariano(y - pred[f"pred_{g}"].to_numpy(), y - pred["pred_control"].to_numpy())
        wins = pred.assign(a=err[g], b=err["control"]).groupby("window")[["a", "b"]].mean()
        sig.append({"group": g, "dm_stat": d["dm_stat"], "p_value": d["p_value"],
                    "win_share": float((wins["a"] < wins["b"]).mean())})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = tr.holm_adjust(sig["p_value"])
    return pd.DataFrame(rows), sig


def evaluate_vol(pred: pd.DataFrame) -> tuple:
    names = ["garch"] + [f"garch+{g}" for g in GROUPS]
    rows = [{"method": n, **iv.summarize(pred, n)} for n in names]
    ctrl = iv.row_loss(pred, "garch")
    sig = []
    for g in GROUPS:
        loss = iv.row_loss(pred, f"garch+{g}")
        d = tr.diebold_mariano(loss, ctrl)
        wins = pred.assign(a=loss, b=ctrl).groupby("window")[["a", "b"]].mean()
        sig.append({"group": g, "dm_stat": d["dm_stat"], "p_value": d["p_value"],
                    "win_share": float((wins["a"] < wins["b"]).mean())})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = tr.holm_adjust(sig["p_value"])
    return pd.DataFrame(rows), sig


def decide(metrics: pd.DataFrame, sig: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    if task == "returns":
        m = metrics.set_index("model")
        base = m.loc["control", "MAE"]
        for _, s in sig.iterrows():
            val = m.loc[s["group"], "MAE"]
            rows.append({"task": task, "group": s["group"], "metric": val, "control": base,
                         "improvement_pct": 100 * (1 - val / base), "lower": bool(val < base),
                         "significant_holm": bool(s["dm_stat"] < 0 and s["p_holm"] < ALPHA),
                         "win_share": s["win_share"], "coverage_ok": True})
    else:
        m = metrics.set_index("method")
        base = m.loc["garch", "pinball"]
        for _, s in sig.iterrows():
            r = m.loc[f"garch+{s['group']}"]
            cov = all(abs(r[f"coverage_{l}"] - l / 100) <= COVERAGE_TOL[l] for l in iv.LEVELS)
            rows.append({"task": task, "group": s["group"], "metric": r["pinball"], "control": base,
                         "improvement_pct": 100 * (1 - r["pinball"] / base), "lower": bool(r["pinball"] < base),
                         "significant_holm": bool(s["dm_stat"] < 0 and s["p_holm"] < ALPHA),
                         "win_share": s["win_share"], "coverage_ok": cov})
    d = pd.DataFrame(rows)
    d["accepted"] = d["lower"] & d["significant_holm"] & (d["win_share"] >= MIN_WIN_SHARE) & d["coverage_ok"]
    return d


def preregistration() -> dict:
    files = {"factors.py": Path(__file__), "intervals.py": Path(iv.__file__), "train.py": Path(tr.__file__)}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "groups": GROUPS, "control_return_features": CONTROL_FEATURES, "control_vol": "GARCH(1,1), Adim 2b",
            "period": {"start": DEV_START, "end_exclusive": tr.TEST_START},
            "rule": {"alpha_holm": ALPHA, "family_size": len(GROUPS), "min_win_share": MIN_WIN_SHARE,
                     "coverage_tolerance": COVERAGE_TOL, "adj_clip": ADJ_CLIP},
            "nature": "kesif; kabul edilen grup 2026 verisiyle teyit edilmeli",
            "code_md5": {k: hashlib.md5(v.read_bytes()).hexdigest() for k, v in files.items()}}


def run(data_path=None, panel_path: Path = PANEL_PATH) -> dict:
    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    prereg = preregistration()
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)              # hesaplamadan ONCE
    df = load_inputs(data_path, panel_path)
    wf = walk_forward(df)
    if pd.to_datetime(wf["returns"]["prediction_date"]).max() >= pd.Timestamp(tr.TEST_START):
        raise ValueError("Degerlendirme test donemine tasti; durduruldu.")
    rm, rs = evaluate_returns(wf["returns"])
    vm, vs = evaluate_vol(wf["vol"])
    decision = pd.concat([decide(rm, rs, "returns"), decide(vm, vs, "vol")], ignore_index=True)
    for name, frame in (("return_metrics", rm), ("return_significance", rs), ("vol_metrics", vm),
                        ("vol_significance", vs), ("decision", decision), ("windows", pd.DataFrame(wf["windows"]))):
        frame.to_csv(target / f"{name}.csv", index=False)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_forecasts": int(len(wf["returns"])), "n_windows": len(wf["windows"]),
               "accepted": decision[decision["accepted"]][["task", "group"]].to_dict(orient="records"),
               "test_period_used": False, "holdout_2026_used": False}
    with open(target / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(target / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at_utc": summary["created_at_utc"], "accepted": summary["accepted"],
                            "code_md5": prereg["code_md5"]}, ensure_ascii=False) + "\n")
    return {"return_metrics": rm, "return_sig": rs, "vol_metrics": vm, "vol_sig": vs,
            "decision": decision, "summary": summary, "walk": wf}


def print_report(res: dict):
    pd.set_option("display.width", 170)
    s = res["summary"]
    print(f"  Degerlendirme: {DEV_START} .. {tr.TEST_START} (haric) | {s['n_forecasts']} tahmin, {s['n_windows']} pencere")
    print("\n  (a) GETIRI - kontrol: kompakt set + Linear Regression")
    print(res["return_metrics"].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n  (b) OYNAKLIK - kontrol: GARCH(1,1)")
    cols = ["method", "pinball", "coverage_80", "coverage_95", "width_80_TL", "christoffersen_80_p"]
    print(res["vol_metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  KARAR (on kayitli kural):")
    d = res["decision"][["task", "group", "improvement_pct", "significant_holm", "win_share", "coverage_ok", "accepted"]]
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    acc = s["accepted"]
    print(f"\n  Kabul edilen: {acc if acc else 'yok'}")
    print("  Not: Kesif deneyi. Kabul edilen bir grup 2026 verisiyle teyit edilmeden kesin bulgu sayilmaz.")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Faz C: faktor grubu deneyi (gelistirme donemi).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 72
    print(f"\n{sep}\nFAZ C - FAKTOR GRUBU DENEYI (getiri ve oynaklik)\n{sep}")
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
