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

Teyit testi: python -m src.models.adaptive --confirm
----------------------------------------------------
  Gelistirmede kabul edilen aci_gvzlag_g020, hicbir ayari degismeden, TEST
  donemi (2023-10-26 .. 2025-12-30) uzerinde tek on kayitli hipotezle sinanir.
  On kayit hesaplamadan ONCE yazilir; bir kez calisir (--force-rerun kayda gecer);
  2026 verisi kullanilirsa hesaplama durur.

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
import math
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

# Teyit testi (gelistirme sonuclarindan sonra, test donemine bakmadan sabitlendi)
CONFIRM_METHOD = "aci_gvzlag_g020"
CONFIRM_CONTROL = "garch"
CONFIRM_MIN_YEAR_N = 100
HOLDOUT_START = "2026-01-01"
CONFIRM_HYPOTHESIS = (
    "Gelistirmede kabul edilen aci_gvzlag_g020 (GVZ[t-1] bilesen oynakligi + carpimsal ACI, "
    "eta = 0.02, olcek degerlendirme basinda 1), hicbir ayari degismeden, test doneminde "
    "(2023-10-26 .. 2025-12-30): (1) %80 kapsam 0.80 +/- 0.03 ve %95 kapsam 0.95 +/- 0.015, "
    "(2) Kupiec p > 0.05 her iki seviyede, (3) en az 100 gozlemli yillarda %95 kapsam >= 0.92, "
    "(4) pinball kaybi garch'tan anlamli olarak kotu degil (iki yonlu DM, p < 0.05 ve kotu ise ret). "
    "Dort kosul birlikte saglanirsa hipotez desteklenir. Tek yontem, tek karsilastirma."
)


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


_DEFAULT_END = object()


def walk_forward(df: pd.DataFrame, start: str = DEV_START, end=_DEFAULT_END) -> dict:
    """end varsayilani TEST_START (gelistirme); end=None verinin sonuna kadar (teyit)."""
    end = tr.TEST_START if end is _DEFAULT_END else end
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


# ---------------------------------------------------------------------------
# Teyit testi
# ---------------------------------------------------------------------------
class ConfirmAlreadyRun(Exception):
    """Teyit testi daha once calistirilmis; --force-rerun olmadan tekrarlanmaz."""


def confirm_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "adaptive_confirm"


def confirm_verdict(summary_m: dict, summary_c: dict, annual_m: pd.DataFrame, dm: dict) -> dict:
    cov_ok = all(abs(summary_m[f"coverage_{l}"] - l / 100) <= iv.COVERAGE_TOL[l] for l in iv.LEVELS)
    kup_ok = bool(summary_m["kupiec_80_p"] > KUPIEC_MIN_P and summary_m["kupiec_95_p"] > KUPIEC_MIN_P)
    full = annual_m[annual_m["n"] >= CONFIRM_MIN_YEAR_N]
    years_ok = bool(len(full) > 0 and (full["coverage_95"] >= MIN_YEAR_COVERAGE_95).all())
    worse = bool(dm["dm_stat"] > 0 and dm["p_value"] < ALPHA_TEST)
    return {"coverage_ok": cov_ok, "kupiec_ok": kup_ok, "years_ok": years_ok,
            "years_checked": [int(y) for y in full["year"]], "not_significantly_worse": not worse,
            "hypothesis_supported": bool(cov_ok and kup_ok and years_ok and not worse)}


def confirm_preregistration() -> dict:
    p = preregistration()
    p.update({"hypothesis": CONFIRM_HYPOTHESIS, "method": CONFIRM_METHOD, "control": CONFIRM_CONTROL,
              "period": {"start": tr.TEST_START, "end": "model_dataset.csv sonu (2026 oncesi)"},
              "min_year_n": CONFIRM_MIN_YEAR_N, "bootstrap": {"block": iv.BOOT_BLOCK, "reps": iv.BOOT_REPS,
                                                             "seed": iv.BOOT_SEED}})
    p.pop("candidates", None)
    return p


def run_confirm(data_path=None, panel_path: Path = ivx.PANEL_PATH, force: bool = False) -> dict:
    target = confirm_dir()
    result_path, log_path = target / "result.json", target / "run_log.jsonl"
    if result_path.exists() and not force:
        raise ConfirmAlreadyRun(f"Teyit testi zaten calistirildi: {result_path}")
    target.mkdir(parents=True, exist_ok=True)
    previous = sum(1 for _ in open(log_path, encoding="utf-8")) if log_path.exists() else 0
    if result_path.exists():
        result_path.rename(target / f"result_run{previous}.json")
    prereg = confirm_preregistration()
    prereg["run_number"] = previous + 1
    with open(target / "preregistration.json", "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)              # hesaplamadan ONCE

    df = ivx.load_inputs(data_path, panel_path)
    # end=HOLDOUT_START: yalnizca HEDEF tarihi (bir sonraki satir) 2026'dan once olan satirlar.
    # Hedef tarihi bilinmeyen son satir da disarida kalir (muhafazakar).
    wf = walk_forward(df, start=tr.TEST_START, end=HOLDOUT_START)
    pred = wf["predictions"]
    if pd.to_datetime(pred["prediction_date"]).max() >= pd.Timestamp(HOLDOUT_START):
        raise ValueError("Teyit testine 2026 (son sinav) verisi girdi; durduruldu.")
    keep = ["window", "prediction_date", "y", "price_t", "mu"]
    for m in (CONFIRM_CONTROL, CONFIRM_METHOD):
        keep += [f"{m}_q{tau}" for tau in iv.QUANTILES]
    pred = pred[keep]
    sm, sc = iv.summarize(pred, CONFIRM_METHOD), iv.summarize(pred, CONFIRM_CONTROL)
    years = pd.to_datetime(pred["prediction_date"]).dt.year
    annual = pd.DataFrame([{"year": int(y), "method": m, **iv.summarize(g, m)}
                           for y, g in pred.groupby(years) for m in (CONFIRM_CONTROL, CONFIRM_METHOD)])
    lm, lc = iv.row_loss(pred, CONFIRM_METHOD), iv.row_loss(pred, CONFIRM_CONTROL)
    dm = tr.diebold_mariano(lm, lc)
    ci = iv.block_bootstrap_ci(lm, lc)
    verdict = confirm_verdict(sm, sc, annual[annual["method"] == CONFIRM_METHOD], dm)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "run_number": prereg["run_number"], "hypothesis": CONFIRM_HYPOTHESIS,
              "n_forecasts": int(len(pred)),
              "period_first_prediction": str(pd.Timestamp(pred["prediction_date"].min()).date()),
              "period_last_prediction": str(pd.Timestamp(pred["prediction_date"].max()).date()),
              "method": {k: sm[k] for k in ("pinball", "coverage_80", "coverage_95", "kupiec_80_p", "kupiec_95_p",
                                            "christoffersen_95_p", "width_80_TL", "width_95_TL")},
              "control": {k: sc[k] for k in ("pinball", "coverage_80", "coverage_95", "kupiec_80_p", "kupiec_95_p",
                                             "christoffersen_95_p", "width_80_TL", "width_95_TL")},
              "pinball_improvement_pct": float(100 * (1 - lm.mean() / lc.mean())),
              "pinball_improvement_ci95_block_bootstrap": list(ci), "dm": dm, **verdict,
              "holdout_2026_used": False}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    annual.to_csv(target / "annual.csv", index=False)
    pred.to_csv(target / "predictions.csv", index=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_number": result["run_number"], "created_at_utc": result["created_at_utc"],
                            "hypothesis_supported": result["hypothesis_supported"],
                            "code_md5": prereg["code_md5"]}) + "\n")
    return {"result": result, "annual": annual, "predictions": pred, "preregistration": prereg}


def print_confirm(res: dict):
    r = res["result"]
    print(f"  Hipotez (on kayitli): {r['hypothesis']}\n")
    print(f"  Donem: {r['period_first_prediction']} .. {r['period_last_prediction']} | {r['n_forecasts']} tahmin "
          f"| calistirma no: {r['run_number']}")
    rows = pd.DataFrame([{"method": CONFIRM_CONTROL, **r["control"]}, {"method": CONFIRM_METHOD, **r["method"]}])
    print(rows.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  Yillik:")
    a = res["annual"][["year", "method", "n", "pinball", "coverage_80", "coverage_95"]]
    print(a.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    lo, hi = r["pinball_improvement_ci95_block_bootstrap"]
    print(f"\n  Pinball degisimi: %{r['pinball_improvement_pct']:.2f} iyilesme (blok bootstrap %95 GA: %{lo:.2f} .. %{hi:.2f})")
    print(f"  DM (yontem - kontrol): istatistik={r['dm']['dm_stat']:+.3f}, p={r['dm']['p_value']:.4f}")
    print(f"  Kosullar: kapsam {r['coverage_ok']} | Kupiec {r['kupiec_ok']} | yillar {r['years_ok']} "
          f"{r['years_checked']} | kotu degil {r['not_significantly_worse']}")
    print(f"\n  SONUC: Hipotez {'DESTEKLENDI' if r['hypothesis_supported'] else 'DESTEKLENMEDI'}.")
    print("  Not: 2026 verisi kullanilmadi. Fiyat teorik gram altindir.")


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
    ap = argparse.ArgumentParser(description="ACI ile uyarlanabilir aralik kalibrasyonu.")
    ap.add_argument("--confirm", action="store_true", help="Dondurulmus aci_gvzlag_g020'yi test doneminde bir kez sina.")
    ap.add_argument("--force-rerun", action="store_true", help="--confirm ile: onceki sonucu arsivleyip tekrar calistir.")
    args = ap.parse_args([] if argv is None else argv)
    if args.force_rerun and not args.confirm:
        ap.error("--force-rerun yalnizca --confirm ile kullanilabilir.")
    sep = "=" * 72
    if args.confirm:
        print(f"\n{sep}\nACI TEYIT TESTI - TEST DONEMI, TEK ON KAYITLI HIPOTEZ\n{sep}")
        try:
            res = run_confirm(force=args.force_rerun)
        except ConfirmAlreadyRun as e:
            print(f"  {e}\n  Tekrar calistirmak bilimsel olarak gerekceli olmali; gerekiyorsa --force-rerun (kayda gecer).")
            return 2
        except (FileNotFoundError, ValueError) as e:
            print(f"  HATA: {e}")
            return 1
        print_confirm(res)
        print(f"\n  Ciktilar: {confirm_dir()}\n{sep}")
        return 0
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
