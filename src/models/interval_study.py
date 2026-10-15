"""
Fiziki Sabah Tahmini icin Oynakliga Duyarli Tahmin Araliklari (KESIF, 2012-2022)
================================================================================
Kullanim (proje kokunden; once python -m src.models.live_backtest calismis olmali):
  python -m src.models.interval_study

Sorun: live_backtest araliklari (son 250 bilinen artigin sabit kantilleri) dogru genislikte ama tasmalar
KUMELENIYOR (Christoffersen p ~ 0): sakin donemde gereksiz genis, calkantili donemde dar.

Yontemler (sonuclardan ONCE sabitlendi), her model ve her duzey (%80, %95) icin:
  Statik   : son 250 BILINEN log artigin (gercek/tahmin) ampirik kantilleri (live_backtest ile ayni).
  Olcekli  : artik e_j, kendinden ONCEKI EWMA oynakligiyla standartlastirilir (z_j = e_j / s_{j-1},
             lambda 0.94); son 250 bilinen z'nin kantilleri bugunku oynaklik tahminiyle (son bilinen
             artiga kadar EWMA) carpilir.
  Olcekli+ACI: Olcekli araligin medyan etrafindaki yari genisligi k ile carpilir; her BILINEN sonucta
             k <- k * exp(eta * (tasma - alfa)), eta 0.02 (teorik fiyat calismasindaki carpimsal surum).
"Bilinen": hedef tarihi, o sabahki son yayimlanmis fiziki kapanis tarihinden buyuk olmayan gunler.
Olcutler: kapsam, ortalama genislik (%), Winkler skoru (TL), Kupiec ve Christoffersen.
Girdi: live_backtest tahmin dosyalari (iki kur surumu). Kilit: hedef tarihi >= 2023-01-01 satir varsa durur.
Bu modul KESIF amaclidir.
"""

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.live import physical_forecast as lp
from src.models import intervals as iv
from src.models import live_backtest as lb
from src.models import physical_eval as pv
from src.models import train as tr

LOCK = pd.Timestamp(pv.LOCK_START)
WINDOW = 250
EWMA_LAMBDA = 0.94
ETA = 0.02
INIT_N = 20
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
METHODS = ["Statik", "Olcekli", "Olcekli+ACI"]
MODELS = lp.MODELS


class IntervalStudyError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "interval_study"


def known_counts(td: np.ndarray, last_date: np.ndarray) -> np.ndarray:
    """Her satir i icin tahmin aninda gerceklesmis (bilinen) onceki satir sayisi m_i (td sirali)."""
    m = np.searchsorted(td, last_date, side="right")
    return np.minimum(m, np.arange(len(td)))


def ewma_var(e: np.ndarray, lam: float = EWMA_LAMBDA, init_n: int = INIT_N) -> np.ndarray:
    """v[j]: e_0..e_j gozlendikten sonraki EWMA varyansi. Baslangic ilk init_n artigin ortalama karesi."""
    v = np.empty(len(e))
    prev = float(np.mean(e[:init_n] ** 2))
    for j, x in enumerate(e):
        prev = lam * prev + (1 - lam) * x * x
        v[j] = prev
    return v


def aci_update(k: float, miss: bool, alpha: float, eta: float = ETA) -> float:
    return k * math.exp(eta * (float(miss) - alpha))


def intervals_for(df: pd.DataFrame, model: str) -> pd.DataFrame:
    td = pd.to_datetime(df["target_date"]).to_numpy()
    last = pd.to_datetime(df["last_date"]).to_numpy()
    m = known_counts(td, last)
    pred = df[f"pred_{model}"].to_numpy(float)
    act = df["actual"].to_numpy(float)
    e = np.log(act / pred)
    v = ewma_var(e)
    s_prev = np.sqrt(np.concatenate([[np.mean(e[:INIT_N] ** 2)], v[:-1]]))   # e_j'den ONCEKI oynaklik
    z = e / s_prev
    out = {}
    for lvl, (lo, hi) in LEVELS.items():
        alpha = 1 - lvl / 100
        L = {mt: np.full(len(df), np.nan) for mt in METHODS}
        H = {mt: np.full(len(df), np.nan) for mt in METHODS}
        k, done = 1.0, 0                      # ACI: islenmis bilinen satir sayisi
        for i in range(len(df)):
            while done < m[i]:                # yeni bilinen sonuclarla ACI guncellemesi
                j = done
                if np.isfinite(L["Olcekli+ACI"][j]):
                    miss = not (L["Olcekli+ACI"][j] <= act[j] <= H["Olcekli+ACI"][j])
                    k = aci_update(k, miss, alpha)
                done += 1
            if m[i] < WINDOW:
                continue
            w = slice(m[i] - WINDOW, m[i])
            ql, qh = np.quantile(e[w], [lo, hi])
            L["Statik"][i], H["Statik"][i] = pred[i] * math.exp(ql), pred[i] * math.exp(qh)
            s_now = math.sqrt(v[m[i] - 1])
            zl, zm, zh = np.quantile(z[w], [lo, 0.5, hi])
            L["Olcekli"][i], H["Olcekli"][i] = pred[i] * math.exp(s_now * zl), pred[i] * math.exp(s_now * zh)
            L["Olcekli+ACI"][i] = pred[i] * math.exp(s_now * (zm + k * (zl - zm)))
            H["Olcekli+ACI"][i] = pred[i] * math.exp(s_now * (zm + k * (zh - zm)))
        for mt in METHODS:
            out[f"lo{lvl}_{mt}"], out[f"hi{lvl}_{mt}"] = L[mt], H[mt]
    return pd.DataFrame({"target_date": df["target_date"].to_numpy(), "actual": act, **out})


def score(iv_df: pd.DataFrame, model: str) -> list:
    rows = []
    a = iv_df["actual"].to_numpy(float)
    for lvl in LEVELS:
        alpha = 1 - lvl / 100
        for mt in METHODS:
            L, H = iv_df[f"lo{lvl}_{mt}"].to_numpy(float), iv_df[f"hi{lvl}_{mt}"].to_numpy(float)
            k = np.isfinite(L) & np.isfinite(H)
            ak, Lk, Hk = a[k], L[k], H[k]
            inside = (ak >= Lk) & (ak <= Hk)
            w = Hk - Lk
            wink = w + (2 / alpha) * ((Lk - ak) * (ak < Lk) + (ak - Hk) * (ak > Hk))
            miss = (~inside).astype(int)
            rows.append({"model": model, "method": mt, "level": lvl, "n": int(k.sum()), "coverage": float(inside.mean()),
                         "width_pct": float(100 * np.mean(w / ak)), "winkler_TL": float(wink.mean()),
                         "kupiec_p": iv.kupiec_p(miss, alpha), "christoffersen_p": iv.christoffersen_p(miss)})
    return rows


def run(mode_files: dict = None) -> dict:
    mode_files = mode_files or {m: lb.out_dir() / f"forecasts_with_intervals_{m}.csv" for m in lb.FX_MODES}
    results, d = {}, out_dir()
    d.mkdir(parents=True, exist_ok=True)
    for mode, path in mode_files.items():
        df = pd.read_csv(path, parse_dates=["target_date", "last_date"]).sort_values("target_date").reset_index(drop=True)
        if (df["target_date"] >= LOCK).any():
            raise IntervalStudyError("Girdide kilitli doneme (>= 2023) ait satir var; durduruldu.")
        rows, yearly = [], []
        for model in MODELS:
            ivd = intervals_for(df, model)
            rows += score(ivd, model)
            if model == "Bosluk":
                yr = pd.to_datetime(ivd["target_date"]).dt.year
                for mt in METHODS:
                    L, H = ivd[f"lo80_{mt}"], ivd[f"hi80_{mt}"]
                    ok = L.notna()
                    cov = ((ivd["actual"] >= L) & (ivd["actual"] <= H))[ok].groupby(yr[ok]).mean()
                    yearly.append(cov.rename(mt))
        table = pd.DataFrame(rows)
        ycov = pd.concat(yearly, axis=1)
        table.to_csv(d / f"intervals_{mode}.csv", index=False)
        ycov.to_csv(d / f"bosluk_yearly_coverage80_{mode}.csv")
        results[mode] = {"table": table, "yearly_cov80_bosluk": ycov}
    best = {m: results[m]["table"].loc[results[m]["table"].groupby(["model", "level"])["winkler_TL"].idxmin(),
                                        ["model", "level", "method"]].to_dict("records") for m in results}
    (d / "summary.json").write_text(json.dumps({"nature": "KESIF - gelistirme donemi", "best_by_winkler": best},
                                               indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Fiziki sabah tahmini - oynakliga duyarli araliklar (gelistirme).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 80
    print(f"\n{sep}\nOYNAKLIGA DUYARLI TAHMIN ARALIKLARI (KESIF, 2012-2022)\n{sep}")
    try:
        r = run()
    except (FileNotFoundError, IntervalStudyError) as e:
        print(f"  HATA: {e}")
        return 1
    pd.set_option("display.width", 220)
    for mode, x in r.items():
        print(f"\n  ##### KUR SURUMU: {mode}")
        t = x["table"]
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n  Bosluk %80 kapsam, yillara gore:")
        print(x["yearly_cov80_bosluk"].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Hedef: kapsam ~0.80/0.95, Christoffersen p > 0.05 (kumelenme yok), Winkler dusuk.\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
