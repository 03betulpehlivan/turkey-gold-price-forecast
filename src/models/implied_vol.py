"""
Zimni Oynaklik ve Bilesen Oynaklik Deneyi (GVZ)
===============================================
Kullanim (proje kokunden):
  python -m src.data.collect                # GVZ dahil 11 seri (internet)
  python -m src.data.build_panel --write --force
  python -m src.models.implied_vol

Soru: Gram altin icin ertesi gun tahmin araliklari, (a) gram oynakligini ons
ve kur bilesenlerinden kurarak ve (b) opsiyon piyasasinin bekledigi altin
oynakligini (CBOE GVZ) kullanarak, Adim 2b GARCH'ina gore iyilesiyor mu?

Veri ve donem
-------------
  Hedef: target_next_return = log(P[t+1]/P[t]) (degismedi).
  model_dataset.csv + market_panel.csv'deki 'gvz' (2026 satirlari okunmadan atilir).
  Degerlendirme: 2018-01-01 .. TEST_START oncesi. Test donemi ve 2026 KULLANILMAZ.
  Pencereler, purge, merkez (Mean-Return) ve kantil yontemi (FHS) intervals.py ile ayni.

Yontemler (sigma, y[t] = r[t+1] icin; yalnizca t ve oncesi)
-----------------------------------------------------------
  garch          KONTROL: gram getirisi uzerinde GARCH(1,1) (Adim 2b)
  comp_garch     sqrt(sg^2 + sf^2 + 2 rho sg sf); sg = ons GARCH, sf = kur GARCH,
                 rho = egitim penceresinde ons-kur getiri korelasyonu
  comp_gvz       ayni bilesen formulu; sg = GVZ[t] / 100 / sqrt(252)
  comp_gvz_lag   ayni; sg = GVZ[t-1] / 100 / sqrt(252)  (zamanlama kontrolu)
  garch_gvz_adj  garch x exp(b * (log GVZ[t] - egitim ortalamasi)), b egitimde EKK,
                 carpan [0.5, 2] ile sinirli ve egitimde geometrik ortalamasi 1

Zamanlama uyarisi
-----------------
  GVZ'nin gunluk kapanisi altinin gunluk kapanisindan sonra olabilir (dogrulanmadi).
  Bu durumda GVZ[t], r[t+1]'e ait hareketlerin bir kismini gormus olabilir.
  comp_gvz_lag bu riski tasimaz. Yalnizca comp_gvz kabul edilir, comp_gvz_lag
  edilmezse kazancin bir kismi zamanlama yan etkisi olabilir; rapor bunu belirtir.

Karar kurali (sonuclardan ONCE; preregistration.json'a yazilir)
---------------------------------------------------------------
  Kabul: pinball kontrolden dusuk VE DM (Holm, 4 aday) p < 0.05 (istatistik < 0)
  VE pencerelerin >= %60'inda kontrolden iyi VE kapsam %80 +/- 0.03, %95 +/- 0.015.
  Kesif deneyidir: kabul edilen yontem test donemi / 2026 ile teyit edilmelidir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
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
CONTROL = "garch"
CANDIDATES = ["comp_garch", "comp_gvz", "comp_gvz_lag", "garch_gvz_adj"]
METHODS = [CONTROL] + CANDIDATES
ADJ_CLIP = (0.5, 2.0)
ALPHA = 0.05
MIN_WIN_SHARE = 0.60
TRADING_DAYS = 252


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "implied_vol"


def gvz_to_daily(gvz) -> np.ndarray:
    """Yillik yuzde oynaklik -> gunluk log getiri standart sapmasi."""
    return np.asarray(gvz, float) / 100.0 / math.sqrt(TRADING_DAYS)


def component_sigma(sg, sf, rho) -> np.ndarray:
    """Var(rg + rf) = sg^2 + sf^2 + 2 rho sg sf (gram = ons x kur log-toplamsal)."""
    sg, sf = np.asarray(sg, float), np.asarray(sf, float)
    return np.sqrt(np.maximum(sg * sg + sf * sf + 2 * rho * sg * sf, 0.0))


def load_inputs(data_path=None, panel_path: Path = PANEL_PATH) -> pd.DataFrame:
    df = tr.load_data(data_path)
    df = df[df["Date"] < pd.Timestamp(HOLDOUT_START)].reset_index(drop=True)   # 2026: son sinav
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    if "gvz" not in panel.columns:
        raise ValueError("Panelde 'gvz' sutunu yok. Once: python -m src.data.collect ve "
                         "python -m src.data.build_panel --write --force")
    panel = panel[panel["Date"] < pd.Timestamp(HOLDOUT_START)].sort_values("Date").reset_index(drop=True)
    panel["gvz_lag1"] = panel["gvz"].shift(1)          # panel satirlari arasinda bir onceki gun
    panel["Date"] = panel["Date"].astype("datetime64[ns]")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    merged = df.merge(panel[["Date", "gvz", "gvz_lag1"]], on="Date", how="left")
    bad = merged[["gvz", "gvz_lag1"]].isna().any(axis=1) | (merged["gvz"] <= 0)
    if bad.any():
        dates = [str(d.date()) for d in merged.loc[bad, "Date"].head(5)]
        raise ValueError(f"GVZ eksik veya gecersiz: {int(bad.sum())} satir (ornek: {dates}).")
    return merged


def walk_forward(df: pd.DataFrame, start: str = DEV_START, end: str = None) -> dict:
    end = tr.TEST_START if end is None else end
    positions = tr.eval_positions(df, start, end)
    windows = tr.make_windows(positions, tr.RETRAIN_EVERY)
    r = df["gold_return"].to_numpy(float)
    rg = df["gold_usd_return"].to_numpy(float)
    rf = df["usdtry_return"].to_numpy(float)
    y = df[tr.TARGET].to_numpy(float)
    gvz_d = gvz_to_daily(df["gvz"])
    gvz_lag_d = gvz_to_daily(df["gvz_lag1"])
    log_gvz = np.log(df["gvz"].to_numpy(float))
    rows, info = [], []
    for w, pos in enumerate(windows, start=1):
        train_df = tr._purge_train(df, pos[0])
        ti = np.arange(len(train_df))
        mu = float(y[ti].mean())
        g_gram, g_gold, g_fx = iv.fit_garch(r[ti]), iv.fit_garch(rg[ti]), iv.fit_garch(rf[ti])
        s_gram = iv.garch_sigma(r, g_gram)
        s_gold = iv.garch_sigma(rg, g_gold)
        s_fx = iv.garch_sigma(rf, g_fx)
        rho = float(np.corrcoef(rg[ti], rf[ti])[0, 1])
        xc = log_gvz - log_gvz[ti].mean()
        target_log = np.log(np.abs(y[ti] - mu) + 1e-6) - np.log(s_gram[ti])
        b = float(LinearRegression().fit(xc[ti, None], target_log).coef_[0])
        adj = np.clip(np.exp(b * xc), *ADJ_CLIP)
        adj = adj / np.exp(np.mean(np.log(adj[ti])))
        sig = {"garch": s_gram,
               "comp_garch": component_sigma(s_gold, s_fx, rho),
               "comp_gvz": component_sigma(gvz_d, s_fx, rho),
               "comp_gvz_lag": component_sigma(gvz_lag_d, s_fx, rho),
               "garch_gvz_adj": s_gram * adj}
        rec = {"window": np.full(len(pos), w), "prediction_date": df["Date"].iloc[pos].to_numpy(),
               "y": y[pos], "price_t": df[tr.PRICE_COL].to_numpy(float)[pos], "mu": np.full(len(pos), mu)}
        for m in METHODS:
            zq, _, _ = iv._fhs_quantiles(y[ti], mu, sig[m][ti])
            rec[f"{m}_sigma"] = sig[m][pos]
            for tau in iv.QUANTILES:
                rec[f"{m}_q{tau}"] = mu + sig[m][pos] * zq[tau]
        rows.append(pd.DataFrame(rec))
        info.append({"window": w, "train_last_date": str(train_df["Date"].iloc[-1].date()),
                     "eval_first_date": str(df["Date"].iat[pos[0]].date()), "n_eval": len(pos),
                     "rho_gold_fx": rho, "gvz_adj_beta": b,
                     "garch_converged": all(g["converged"] for g in (g_gram, g_gold, g_fx))})
    return {"predictions": pd.concat(rows, ignore_index=True), "windows": info}


def evaluate(pred: pd.DataFrame) -> tuple:
    metrics = pd.DataFrame([{"method": m, **iv.summarize(pred, m)} for m in METHODS])
    ctrl = iv.row_loss(pred, CONTROL)
    sig = []
    for m in CANDIDATES:
        loss = iv.row_loss(pred, m)
        d = tr.diebold_mariano(loss, ctrl)
        wins = pred.assign(a=loss, b=ctrl).groupby("window")[["a", "b"]].mean()
        sig.append({"method": m, "dm_stat": d["dm_stat"], "p_value": d["p_value"],
                    "win_share": float((wins["a"] < wins["b"]).mean())})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = tr.holm_adjust(sig["p_value"])
    years = pd.to_datetime(pred["prediction_date"]).dt.year
    annual = pd.DataFrame([{"year": int(yr), "method": m, **{k: v for k, v in iv.summarize(g, m).items()
                                                             if k in ("n", "pinball", "coverage_80", "coverage_95", "width_80_TL")}}
                           for yr, g in pred.groupby(years) for m in METHODS])
    return metrics, sig, annual


def decide(metrics: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    m = metrics.set_index("method")
    base = m.loc[CONTROL, "pinball"]
    rows = []
    for _, s in sig.iterrows():
        r = m.loc[s["method"]]
        cov = all(abs(r[f"coverage_{l}"] - l / 100) <= iv.COVERAGE_TOL[l] for l in iv.LEVELS)
        rows.append({"method": s["method"], "pinball": r["pinball"], "control": base,
                     "improvement_pct": 100 * (1 - r["pinball"] / base), "lower": bool(r["pinball"] < base),
                     "significant_holm": bool(s["dm_stat"] < 0 and s["p_holm"] < ALPHA),
                     "win_share": s["win_share"], "coverage_ok": cov})
    d = pd.DataFrame(rows)
    d["accepted"] = d["lower"] & d["significant_holm"] & (d["win_share"] >= MIN_WIN_SHARE) & d["coverage_ok"]
    return d


def timing_note(decision: pd.DataFrame) -> str:
    acc = set(decision.loc[decision["accepted"], "method"])
    if "comp_gvz" in acc and "comp_gvz_lag" not in acc:
        return ("comp_gvz kabul edildi ama gecikmeli surumu edilmedi: kazancin bir kismi GVZ'nin "
                "altindan sonra kapanmasindan (zamanlama) kaynaklaniyor olabilir.")
    if "comp_gvz_lag" in acc:
        return "Gecikmeli GVZ de kabul edildi: kazanc zamanlama yan etkisiyle aciklanamaz."
    return "GVZ yontemleri kabul edilmedi."


def preregistration() -> dict:
    files = {"implied_vol.py": Path(__file__), "intervals.py": Path(iv.__file__), "train.py": Path(tr.__file__)}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "control": CONTROL, "candidates": CANDIDATES,
            "period": {"start": DEV_START, "end_exclusive": tr.TEST_START},
            "rule": {"alpha_holm": ALPHA, "family_size": len(CANDIDATES), "min_win_share": MIN_WIN_SHARE,
                     "coverage_tolerance": iv.COVERAGE_TOL, "adj_clip": ADJ_CLIP},
            "timing_control": "comp_gvz_lag: GVZ[t-1]",
            "nature": "kesif; kabul edilen yontem test donemi veya 2026 ile teyit edilmeli",
            "code_md5": {k: hashlib.md5(v.read_bytes()).hexdigest() for k, v in files.items()}}


def run(data_path=None, panel_path: Path = PANEL_PATH) -> dict:
    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    prereg = preregistration()
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)              # hesaplamadan ONCE
    df = load_inputs(data_path, panel_path)
    wf = walk_forward(df)
    pred = wf["predictions"]
    if pd.to_datetime(pred["prediction_date"]).max() >= pd.Timestamp(tr.TEST_START):
        raise ValueError("Degerlendirme test donemine tasti; durduruldu.")
    metrics, sig, annual = evaluate(pred)
    decision = decide(metrics, sig)
    note = timing_note(decision)
    for name, frame in (("metrics", metrics), ("significance", sig), ("annual", annual),
                        ("decision", decision), ("windows", pd.DataFrame(wf["windows"])), ("predictions", pred)):
        frame.to_csv(target / f"{name}.csv", index=False)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_forecasts": int(len(pred)), "n_windows": len(wf["windows"]),
               "accepted": decision.loc[decision["accepted"], "method"].tolist(), "timing_note": note,
               "test_period_used": False, "holdout_2026_used": False}
    with open(target / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(target / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at_utc": summary["created_at_utc"], "accepted": summary["accepted"],
                            "code_md5": prereg["code_md5"]}, ensure_ascii=False) + "\n")
    return {"metrics": metrics, "significance": sig, "annual": annual, "decision": decision,
            "summary": summary, "walk": wf}


def print_report(res: dict):
    pd.set_option("display.width", 170)
    s = res["summary"]
    print(f"  Degerlendirme: {DEV_START} .. {tr.TEST_START} (haric) | {s['n_forecasts']} tahmin, {s['n_windows']} pencere")
    cols = ["method", "pinball", "coverage_80", "coverage_95", "width_80_TL", "width_95_TL",
            "kupiec_95_p", "christoffersen_95_p"]
    print(res["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  Yillik %95 kapsam (kriz yillari onemli):")
    piv = res["annual"].pivot(index="year", columns="method", values="coverage_95")[METHODS]
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  KARAR (on kayitli kural, kontrol: garch):")
    d = res["decision"][["method", "improvement_pct", "significant_holm", "win_share", "coverage_ok", "accepted"]]
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Kabul edilen: {s['accepted'] or 'yok'}")
    print(f"  Zamanlama: {s['timing_note']}")
    print("  Not: Kesif deneyi; test donemi ve 2026 kullanilmadi.")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="GVZ ve bilesen oynaklik deneyi (gelistirme donemi).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 72
    print(f"\n{sep}\nZIMNI OYNAKLIK (GVZ) VE BILESEN OYNAKLIK DENEYI\n{sep}")
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
