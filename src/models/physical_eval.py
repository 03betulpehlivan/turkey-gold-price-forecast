"""
Fiziki Altin Kapanis Tahmini: Gelistirme Degerlendirmesi (2016-2022)
====================================================================
Kullanim (proje kokunden):
  python -m src.models.physical_eval

Hedef (teorik fiyattan AYRI): Turkiye organize fiziki altin piyasasi gunluk kapanisi (TL/gr,
data/processed/physical/physical_gold_daily.csv). y = log(P[t'] / P[t]); t ve t' ardisik fiziki islem gunleri.

Tahmin ani: t' gunu SABAHI (EVDS bir gunun kapanisini ertesi gun yayimlar; P[t] o sabah bilinir).
O sabah bilinen uluslararasi bilgi (market_panel.csv, Yahoo gunluk):
  ons   : panel tarihi t'den KESIN once olan son gunun kapanisi (~18:00 UTC; Borsa Istanbul kapanisindan sonra)
  kur   : panel tarihi t' olan USD/TRY gunluk cubugu (fx_close_time teshisi: ~02-05 UTC; BIST acilisindan once)
  Sabah teorik: M[t'] = ons x kur / 31.1035
  Ihtiyatli surum: kur icin de t'den KESIN once olan son deger (saat damgasi riskine karsi alt sinir).

Ozellikler (hepsi tahmin aninda bilinir):
  gap      = log M[t'] - log P[t]        (gece boyunca ons/kur hareketinin fiziki fiyata henuz yansimamis kismi)
  prim     = log P[t] - log M[t]         (dunku fiziki prim; 2023-2024'te belirgin)
  r_onceki = log(P[t] / P[t-1])

Modeller (ayni pencereler, genisleyen, 63 gunde bir, 1 satir purge; train.py araclari):
  Zero-Return (RW, referans), Mean-Return, Bosluk (y = a + b*gap), Bosluk-Ihtiyatli,
  Bosluk+Prim (LR: gap, prim, r_onceki), XGBoost (ayni uc ozellik, kucuk ayar)

Donemler:
  Gelistirme (bu modul): hedef tarihi 2016-01-01 .. 2022-12-31. KESIF.
  2023-01-01 ve sonrasi KILITLI: bu modul o doneme ait tahmin uretmez; ileride on kayitli tek
  seferlik bagimsiz test icin ayrilmistir.
Karsilastirma point_eval.py ile ayni: TL hatalari, blok bootstrap GA, DM + Holm, pencere kazanma orani.
Zamanlama uyarisi: ana Bosluk modeli ihtiyatli surumden belirgin iyi cikarsa kur saat damgasindan
kaynaklanan sizinti olasiligi raporlanir (kur gunluk cubugunun gercek saati kesin degil).
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models import point_eval as pe
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
PHYS_PATH = ROOT / "data" / "processed" / "physical" / "physical_gold_daily.csv"
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
DEV_START, LOCK_START = "2016-01-01", "2023-01-01"
REFERENCE = "Zero-Return"
MODELS = ["Zero-Return", "Mean-Return", "Bosluk", "Bosluk-Ihtiyatli", "Bosluk+Prim", "XGBoost"]
TIMING_FLAG_RATIO = 1.5          # ana iyilesme / ihtiyatli iyilesme bu orani asarsa uyari


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "physical_eval"


class PhysicalEvalError(Exception):
    pass


def build_frame(phys: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Her fiziki gun t icin (t, t') cifti ve sabah bilgisi. Son satirin hedefi yoktur."""
    p = phys[["Date", "kapanis_tl_gr"]].sort_values("Date").reset_index(drop=True).copy()
    p["Date"] = p["Date"].astype("datetime64[ns]")
    pan = panel[["Date", "gold_usd_oz", "usdtry"]].dropna().sort_values("Date").copy()
    pan["Date"] = pan["Date"].astype("datetime64[ns]")

    def asof(dates, col, strict):
        r = pd.merge_asof(pd.DataFrame({"Date": dates}), pan[["Date", col]].rename(columns={"Date": "src"}),
                          left_on="Date", right_on="src", direction="backward", allow_exact_matches=not strict)
        return r[col].to_numpy(float), r["src"].to_numpy()

    d = p["Date"]
    gold, gold_src = asof(d, "gold_usd_oz", strict=True)          # t'den KESIN once
    fx, fx_src = asof(d, "usdtry", strict=False)                  # t' gunu sabah cubugu (yoksa son)
    fx_c, _ = asof(d, "usdtry", strict=True)                       # ihtiyatli: t'den kesin once
    p["M"] = gold * fx / 31.1035
    p["M_cons"] = gold * fx_c / 31.1035
    p["gold_src"], p["fx_src"] = gold_src, fx_src
    lp = np.log(p["kapanis_tl_gr"])
    p["prim"] = lp - np.log(p["M"])
    p["r_onceki"] = lp.diff()
    # satir t: hedef = t' getirisi; ozellikler t' sabahinda bilinen
    p["target_date"] = p["Date"].shift(-1)
    p["y"] = lp.shift(-1) - lp
    p["gap"] = np.log(p["M"].shift(-1)) - lp
    p["gap_cons"] = np.log(p["M_cons"].shift(-1)) - lp
    p["gold_src_next"] = p["gold_src"].shift(-1)
    p["fx_src_next"] = p["fx_src"].shift(-1)
    p["price_t"] = p["kapanis_tl_gr"]
    return p.dropna(subset=["gap", "gap_cons", "prim", "r_onceki"]).reset_index(drop=True)


def audit_timing(df: pd.DataFrame):
    """Ons kaynagi hedef gunden KESIN once, kur kaynagi hedef gunden sonra OLAMAZ."""
    ok = df["y"].notna()
    if (pd.to_datetime(df.loc[ok, "gold_src_next"]) >= pd.to_datetime(df.loc[ok, "target_date"])).any():
        raise PhysicalEvalError("Ons bilgisi hedef gunun kendisinden veya sonrasindan geliyor (sizinti).")
    if (pd.to_datetime(df.loc[ok, "fx_src_next"]) > pd.to_datetime(df.loc[ok, "target_date"])).any():
        raise PhysicalEvalError("Kur bilgisi hedef gunden sonradan geliyor (sizinti).")


def positions(df: pd.DataFrame, start: str = DEV_START, end: str = LOCK_START) -> list:
    """Hedef tarihi [start, end) araligindaki satirlar. end=None: verinin sonuna kadar (yalnizca son sinav)."""
    td = pd.to_datetime(df["target_date"])
    upper = (td < pd.Timestamp(end)) if end else True
    pos = df.index[(td >= pd.Timestamp(start)) & upper & df["y"].notna()].tolist()
    if not pos or pos != list(range(pos[0], pos[-1] + 1)):
        raise PhysicalEvalError("Degerlendirme satirlari kesintisiz degil.")
    return pos


def generate(df: pd.DataFrame, start: str = DEV_START, end: str = LOCK_START) -> pd.DataFrame:
    """Varsayilan: gelistirme donemi; kilitli doneme tahmin URETMEZ. Kilitli donem yalnizca
    physical_exam.py (on kayitli tek seferlik sinav) tarafindan start=LOCK_START, end=None ile cagrilir."""
    from xgboost import XGBRegressor
    pos = positions(df, start, end)
    y = df["y"].to_numpy(float)
    feats = ["gap", "prim", "r_onceki"]
    X = df[feats].to_numpy(float)
    rows = []
    for w, win in enumerate(tr.make_windows(pos, tr.RETRAIN_EVERY), start=1):
        ti = np.arange(win[0] - 1)                     # 1 satir purge: hedefi win[0] gunune uzanan satir disarida
        if pd.to_datetime(df["target_date"].iat[ti[-1]]) > pd.to_datetime(df["Date"].iat[win[0]]):
            raise PhysicalEvalError("Egitim hedefi degerlendirme gununu asiyor.")
        b1 = LinearRegression().fit(df[["gap"]].to_numpy(float)[ti], y[ti])
        b2 = LinearRegression().fit(df[["gap_cons"]].to_numpy(float)[ti], y[ti])
        lr = LinearRegression().fit(X[ti], y[ti])
        xg = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                          random_state=42, n_jobs=1).fit(X[ti], y[ti])
        rows.append(pd.DataFrame({
            "window": w, "prediction_date": df["target_date"].iloc[win].to_numpy(),
            "target_date": df["target_date"].iloc[win].to_numpy(), "price_t": df["price_t"].iloc[win].to_numpy(),
            "actual_return": y[win], "pred_Zero-Return": 0.0, "pred_Mean-Return": float(y[ti].mean()),
            "pred_Bosluk": b1.predict(df[["gap"]].to_numpy(float)[win]),
            "pred_Bosluk-Ihtiyatli": b2.predict(df[["gap_cons"]].to_numpy(float)[win]),
            "pred_Bosluk+Prim": lr.predict(X[win]), "pred_XGBoost": xg.predict(X[win]),
            "beta_gap": float(b1.coef_[0])}))
    pred = pd.concat(rows, ignore_index=True)
    if end and pd.to_datetime(pred["target_date"]).max() >= pd.Timestamp(end):
        raise PhysicalEvalError("Istenen donemin disina tahmin uretildi; durduruldu.")
    return pred


def timing_warning(comp: pd.DataFrame) -> str:
    c = comp.set_index("model")
    main, cons = c.loc["Bosluk", "improvement_pct"], c.loc["Bosluk-Ihtiyatli", "improvement_pct"]
    if main > 0 and (cons <= 0 or main / cons > TIMING_FLAG_RATIO):
        return (f"UYARI: Bosluk (%{main:.1f}) ihtiyatli surumden (%{cons:.1f}) belirgin iyi; kazancin bir kismi "
                "kur gunluk cubugunun saat damgasina bagli olabilir. Saatlik veriyle dogrulanmali.")
    return f"Ana (%{main:.1f}) ve ihtiyatli (%{cons:.1f}) surum tutarli; kur saat damgasi kazanci aciklamiyor."


def run(phys_path: Path = PHYS_PATH, panel_path: Path = PANEL_PATH) -> dict:
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    df = build_frame(phys, panel)
    audit_timing(df)
    pred = generate(df)
    metrics = pe.metrics_table(pred)
    comp = pe.comparison_table(pred, REFERENCE)
    annual = pe.grouped_table(pred, pd.to_datetime(pred["target_date"]).dt.year.rename("year"), "year")
    target = out_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics", metrics), ("comparison_vs_random_walk", comp), ("annual", annual),
                        ("predictions", pred)):
        frame.to_csv(target / f"{name}.csv", index=False)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "target": "fiziki altin kapanisi TL/gr (EVDS; toptan piyasa, kuyumcu degil)",
               "period": [str(pd.Timestamp(pred["target_date"].min()).date()), str(pd.Timestamp(pred["target_date"].max()).date())],
               "n_forecasts": int(len(pred)), "locked_from": LOCK_START, "nature": "kesif (gelistirme)",
               "significantly_better_than_rw": comp.loc[comp["significantly_better"], "model"].tolist(),
               "timing_note": timing_warning(comp), "beta_gap_last_window": float(pred["beta_gap"].iloc[-1])}
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metrics": metrics, "comparison": comp, "annual": annual, "summary": summary, "predictions": pred}


def print_report(res: dict):
    pd.set_option("display.width", 200)
    s = res["summary"]
    print(f"  Hedef: {s['target']}\n  Donem: {s['period'][0]} .. {s['period'][1]} | {s['n_forecasts']} tahmin | "
          f"KESIF | {s['locked_from']} ve sonrasi KILITLI\n")
    m = res["metrics"]
    print(m[["model", "MAE_TL", "MedAE_TL", "RMSE_TL", "MAPE_pct", "direction_accuracy_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  Hassasiyet (hata esigin altinda kalan gun yuzdesi):")
    cols = ["model"] + [c for c in m.columns if c.startswith("share_err_le_") and c.endswith("pct_pct")]
    print(m[cols].rename(columns=lambda c: c.replace("share_err_le_", "<=").replace("pct_pct", "%"))
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n  Rastgele yuruyuse (dunku fiziki kapanis) gore:")
    c = res["comparison"]
    print(c[["model", "improvement_pct", "diff_ci95_low", "diff_ci95_high", "p_holm", "win_share",
             "significantly_better"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Yillara gore MAPE (%):")
    print(res["annual"].pivot(index="year", columns="model", values="MAPE_pct")[MODELS]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Anlamli olarak RW'den iyi: {s['significantly_better_than_rw'] or 'yok'}")
    print(f"  Zamanlama: {s['timing_note']}")
    print(f"  Son penceredeki bosluk katsayisi (b): {s['beta_gap_last_window']:.3f}  (1'e yakin: gece hareketi tam yansiyor)")


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Fiziki altin kapanis tahmini - gelistirme degerlendirmesi.").parse_args(
        [] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nFIZIKI ALTIN KAPANIS TAHMINI - GELISTIRME (2016-2022)\n{sep}")
    try:
        res = run()
    except (FileNotFoundError, PhysicalEvalError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(res)
    print(f"\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
