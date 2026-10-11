"""
Uyarlanabilir Aralik Kalibrasyonu (Adaptive Conformal Inference, ACI)
====================================================================
Kullanim (proje kokunden):
  python -m src.models.adaptive

Soru: Kriz donemlerinde %95 araliklar hedefin altinda kaliyor (GARCH: 2018'de
0.888). Gecmis isabetsizliklere gore araligi cevrimici ayarlayan ACI, kapsami
her yil hedefe yaklastirabilir mi, bunu tahmin kalitesini bozmadan yapabilir mi?

Yontem (Gibbs ve Candes 2021 fikrinin carpimsal olcek surumu)
-------------------------------------------------------------
  Her seviye L (80, 95) icin hedef tasma orani a = 1 - L/100 ve egitim penceresinin
  standartlastirilmis artiklarindan (FHS, intervals.py ile ayni) nominal kantiller
  z_lo = z(a/2), z_hi = z(1 - a/2). t gununun araligi:
      [mu + sigma_t * k_t * z_lo,  mu + sigma_t * k_t * z_hi]
  Olcek, YALNIZCA sonucu bilinen tahminle guncellenir:
      log k_{t+1} = log k_t + eta * (hata_t - a),   hata_t = 1{y_t araligin disinda}
  t gunundeki tahminin hedefi t+1 kapanisinda bilinir; bu nedenle t+1 gunundeki
  tahmin, t gununun hatasiyla guncellenmis k ile yapilir. k'nin ust siniri yoktur:
  aralik, egitimde gorulen en uc degerlerin otesine genisleyebilir (kriz rejimi).
  Teleskopik toplam: ortalama(hata) - a = (log k_T - log k_0) / (eta T); k sinirli kaldikca
  uzun donem tasma orani hedefe yakinsar.
  Durum degerlendirme donemi boyunca surekli tasinir (pencere sinirlarinda sifirlanmaz).

  Not: Ilk tasarim kantil seviyesini kaydiriyordu; bu, araligi egitimdeki en uc
  artikla sinirladigi icin sentetik bir oynaklik soku sonrasi kapsami %86'da
  birakti. Carpimsal surum gercek veride HICBIR sonuc gorulmeden secildi.

Kontrol ve adaylar (sonuclardan ONCE sabitlendi)
------------------------------------------------
  Kontrol: garch (Adim 2b, uyarlanmasiz).
  Aday:    aci_garch_g005, aci_garch_g020, aci_gvzlag_g005, aci_gvzlag_g020
           (taban: garch veya GVZ[t-1] bilesen modeli; eta: 0.005 veya 0.02)
  Bilgi:   comp_gvz_lag (uyarlanmasiz) karsilastirma icin raporlanir.

Karar kurali (preregistration.json'a yazilir)
---------------------------------------------
  Kabul: (1) %80 kapsam 0.80 +/- 0.03 ve %95 kapsam 0.95 +/- 0.015,
         (2) Kupiec p > 0.05 her iki seviyede,
         (3) HER yilin %95 kapsami >= 0.92,
         (4) pinball, kontrolden anlamli olarak KOTU degil (DM istatistigi <= 0 veya
             Holm p >= 0.05; 4 aday).
  Kesif deneyidir; gelistirme donemi onceki deneylerde de kullanildi. Teyit icin
  yontem dondurulup 2026 verisiyle tek seferlik sinanmalidir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from src.models import implied_vol as ivx
from src.models import intervals as iv
from src.models import train as tr

DEV_START = iv.DEV_START
CONTROL = "garch"
BASES = {"garch": "garch", "gvzlag": "comp_gvz_lag"}
GAMMAS = {"g005": 0.005, "g020": 0.02}
CANDIDATES = [f"aci_{b}_{g}" for b in BASES for g in GAMMAS]
REFERENCE_INFO = "comp_gvz_lag"
LEVEL_TAUS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
MIN_YEAR_COVERAGE_95 = 0.92
KUPIEC_MIN_P = 0.05
ALPHA_TEST = 0.05


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "adaptive"


# ---------------------------------------------------------------------------
# ACI cekirdegi
# ---------------------------------------------------------------------------
def aci_sequence(y, mu, sigma, z_by_row, target_miss: float, eta: float) -> tuple:
    """Bir seviye icin carpimsal olcekli ACI. Dondurur: (alt, ust, k_dizisi, hata_dizisi).

    y, mu, sigma: degerlendirme satirlari (zaman sirasinda).
    z_by_row[i]: i. satirin egitim penceresine ait standartlastirilmis artiklar.
    """
    n = len(y)
    lo, hi, k_path, miss = np.empty(n), np.empty(n), np.empty(n), np.empty(n, dtype=bool)
    log_k = 0.0
    for i in range(n):
        zl, zh = np.quantile(z_by_row[i], [target_miss / 2, 1 - target_miss / 2])
        k = float(np.exp(log_k))
        lo[i], hi[i] = mu[i] + sigma[i] * k * zl, mu[i] + sigma[i] * k * zh
        k_path[i] = k
        miss[i] = not (lo[i] <= y[i] <= hi[i])
        log_k += eta * (float(miss[i]) - target_miss)       # sonraki gun icin
    return lo, hi, k_path, miss


def _base_sigmas(df: pd.DataFrame, ti: np.ndarray) -> dict:
    r = df["gold_return"].to_numpy(float)
    rg = df["gold_usd_return"].to_numpy(float)
    rf = df["usdtry_return"].to_numpy(float)
    g_gram, g_fx = iv.fit_garch(r[ti]), iv.fit_garch(rf[ti])
    rho = float(np.corrcoef(rg[ti], rf[ti])[0, 1])
    return {"garch": iv.garch_sigma(r, g_gram),
            "comp_gvz_lag": ivx.component_sigma(ivx.gvz_to_daily(df["gvz_lag1"]), iv.garch_sigma(rf, g_fx), rho)}


def walk_forward(df: pd.DataFrame, start: str = DEV_START, end: str = None) -> dict:
    end = tr.TEST_START if end is None else end
    positions = tr.eval_positions(df, start, end)
    windows = tr.make_windows(positions, tr.RETRAIN_EVERY)
    y = df[tr.TARGET].to_numpy(float)
    frames, z_rows = [], {b: [] for b in BASES.values()}
    for w, pos in enumerate(windows, start=1):
        train_df = tr._purge_train(df, pos[0])
        ti = np.arange(len(train_df))
        mu = float(y[ti].mean())
        sig = _base_sigmas(df, ti)
        rec = {"window": np.full(len(pos), w), "prediction_date": df["Date"].iloc[pos].to_numpy(),
               "y": y[pos], "price_t": df[tr.PRICE_COL].to_numpy(float)[pos], "mu": np.full(len(pos), mu)}
        for base, s in sig.items():
            zq, z, _ = iv._fhs_quantiles(y[ti], mu, s[ti])
            rec[f"{base}_sigma"] = s[pos]
            for tau in iv.QUANTILES:                      # uyarlanmasiz (kontrol ve bilgi)
                rec[f"{base}_q{tau}"] = mu + s[pos] * zq[tau]
            z_rows[base].extend([z] * len(pos))
        frames.append(pd.DataFrame(rec))
    pred = pd.concat(frames, ignore_index=True)
    alpha_paths = {}
    for b, base in BASES.items():
        for g, gamma in GAMMAS.items():
            name = f"aci_{b}_{g}"
            for lvl, (tl, th) in LEVEL_TAUS.items():
                lo, hi, k_path, _ = aci_sequence(pred["y"].to_numpy(), pred["mu"].to_numpy(),
                                                 pred[f"{base}_sigma"].to_numpy(), z_rows[base], 1 - lvl / 100, gamma)
                pred[f"{name}_q{tl}"], pred[f"{name}_q{th}"] = lo, hi
                alpha_paths[f"{name}_{lvl}"] = k_path
    return {"predictions": pred, "alpha_paths": pd.DataFrame(alpha_paths).assign(
        prediction_date=pred["prediction_date"])}


# ---------------------------------------------------------------------------
# Degerlendirme ve karar
# ---------------------------------------------------------------------------
METHODS = [CONTROL, REFERENCE_INFO] + CANDIDATES


def evaluate(pred: pd.DataFrame) -> tuple:
    metrics = pd.DataFrame([{"method": m, **iv.summarize(pred, m)} for m in METHODS])
    years = pd.to_datetime(pred["prediction_date"]).dt.year
    annual = pd.DataFrame([{"year": int(yr), "method": m, **{k: v for k, v in iv.summarize(g, m).items()
                                                             if k in ("n", "pinball", "coverage_80", "coverage_95", "width_95_TL")}}
                           for yr, g in pred.groupby(years) for m in METHODS])
    ctrl = iv.row_loss(pred, CONTROL)
    sig = []
    for m in CANDIDATES:
        d = tr.diebold_mariano(iv.row_loss(pred, m), ctrl)
        sig.append({"method": m, "dm_stat": d["dm_stat"], "p_value": d["p_value"]})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = tr.holm_adjust(sig["p_value"])
    return metrics, annual, sig


def decide(metrics: pd.DataFrame, annual: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    m = metrics.set_index("method")
    s = sig.set_index("method")
    rows = []
    for c in CANDIDATES:
        r = m.loc[c]
        cov_ok = all(abs(r[f"coverage_{l}"] - l / 100) <= iv.COVERAGE_TOL[l] for l in iv.LEVELS)
        kup_ok = bool(r["kupiec_80_p"] > KUPIEC_MIN_P and r["kupiec_95_p"] > KUPIEC_MIN_P)
        min_year = float(annual.loc[annual["method"] == c, "coverage_95"].min())
        worse = bool(s.loc[c, "dm_stat"] > 0 and s.loc[c, "p_holm"] < ALPHA_TEST)
        rows.append({"method": c, "coverage_80": r["coverage_80"], "coverage_95": r["coverage_95"],
                     "coverage_ok": cov_ok, "kupiec_ok": kup_ok, "min_year_coverage_95": min_year,
                     "years_ok": min_year >= MIN_YEAR_COVERAGE_95,
                     "pinball_change_pct": 100 * (r["pinball"] / m.loc[CONTROL, "pinball"] - 1),
                     "not_significantly_worse": not worse})
    d = pd.DataFrame(rows)
    d["accepted"] = d["coverage_ok"] & d["kupiec_ok"] & d["years_ok"] & d["not_significantly_worse"]
    return d


def preregistration() -> dict:
    files = {"adaptive.py": Path(__file__), "implied_vol.py": Path(ivx.__file__),
             "intervals.py": Path(iv.__file__), "train.py": Path(tr.__file__)}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "control": CONTROL, "candidates": CANDIDATES, "gammas": GAMMAS, "bases": BASES,
            "period": {"start": DEV_START, "end_exclusive": tr.TEST_START},
            "rule": {"coverage_tolerance": iv.COVERAGE_TOL, "kupiec_min_p": KUPIEC_MIN_P,
                     "min_year_coverage_95": MIN_YEAR_COVERAGE_95, "non_inferiority": "DM Holm p<0.05 ve kotu ise ret",
                     "update": "log k_{t+1} = log k_t + eta (hata_t - a)"},
            "nature": "kesif; dondurulup 2026 ile teyit edilmeli",
            "code_md5": {k: hashlib.md5(v.read_bytes()).hexdigest() for k, v in files.items()}}


def save_plot(paths: pd.DataFrame, target: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(12, 3.8))
    d = pd.to_datetime(paths["prediction_date"])
    for c in ("aci_garch_g005_95", "aci_gvzlag_g005_95"):
        if c in paths:
            ax.plot(d, paths[c], lw=0.9, label=c.replace("_95", ""))
    ax.axhline(1.0, color="black", lw=0.6, ls="--", label="olcek = 1 (FHS araligi)")
    ax.set_ylabel("k_t (%95 seviyesi)")
    ax.set_title("ACI olcegi: 1'in ustu = genisletilmis aralik (ornegin kriz donemleri)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(target / "aci_alpha_path.png", dpi=110)
    plt.close()


def run(data_path=None, panel_path: Path = ivx.PANEL_PATH) -> dict:
    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    prereg = preregistration()
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)              # hesaplamadan ONCE
    df = ivx.load_inputs(data_path, panel_path)                          # 2026 burada atilir
    wf = walk_forward(df)
    pred = wf["predictions"]
    if pd.to_datetime(pred["prediction_date"]).max() >= pd.Timestamp(tr.TEST_START):
        raise ValueError("Degerlendirme test donemine tasti; durduruldu.")
    metrics, annual, sig = evaluate(pred)
    decision = decide(metrics, annual, sig)
    for name, frame in (("metrics", metrics), ("annual", annual), ("significance", sig),
                        ("decision", decision), ("alpha_paths", wf["alpha_paths"]), ("predictions", pred)):
        frame.to_csv(target / f"{name}.csv", index=False)
    save_plot(wf["alpha_paths"], target)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_forecasts": int(len(pred)), "accepted": decision.loc[decision["accepted"], "method"].tolist(),
               "test_period_used": False, "holdout_2026_used": False}
    with open(target / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(target / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at_utc": summary["created_at_utc"], "accepted": summary["accepted"],
                            "code_md5": prereg["code_md5"]}, ensure_ascii=False) + "\n")
    return {"metrics": metrics, "annual": annual, "significance": sig, "decision": decision,
            "summary": summary, "walk": wf}


def print_report(res: dict):
    pd.set_option("display.width", 170)
    print(f"  Degerlendirme: {DEV_START} .. {tr.TEST_START} (haric) | {res['summary']['n_forecasts']} tahmin")
    cols = ["method", "pinball", "coverage_80", "coverage_95", "width_80_TL", "width_95_TL",
            "kupiec_80_p", "kupiec_95_p", "christoffersen_95_p"]
    print(res["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  Yillik %95 kapsam:")
    print(res["annual"].pivot(index="year", columns="method", values="coverage_95")[METHODS]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  KARAR (on kayitli kural):")
    d = res["decision"][["method", "coverage_80", "coverage_95", "kupiec_ok", "min_year_coverage_95",
                         "pinball_change_pct", "not_significantly_worse", "accepted"]]
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Kabul edilen: {res['summary']['accepted'] or 'yok'}")
    print("  Not: Kesif deneyi; test donemi ve 2026 kullanilmadi.")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="ACI ile uyarlanabilir aralik kalibrasyonu (gelistirme donemi).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 72
    print(f"\n{sep}\nUYARLANABILIR ARALIK KALIBRASYONU (ACI)\n{sep}")
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
