"""
LIVE_PHYS_V1 Geriye Donuk Testi ve Tahmin Araliklari (KESIF, 2012-2022)
=======================================================================
Kullanim (proje kokunden; genisletilmis panel gerekir: data/processed/history/market_panel.csv):
  python -m src.models.live_backtest

Soru: Yarin sabah calisacak canli protokol (src/live/physical_forecast.py, on kayitli) 2012-2022'de her
sabah calissaydi ne olurdu? Prim-Capa ve Bosluk-Kayan gelistirme doneminde hic olculmedi.

Yontem (canli kod BIREBIR kullanilir, DEGISTIRILMEZ):
  Her hedef gun T (fiziki islem gunu) icin: fiziki seri T'den once kesilir; panel T'den once kesilir;
  kur = panelin T gunku Yahoo cubugu (2011-2022'de BIST kapanisindan once oldugu fx_timing_history ile
  gosterildi), saatlik cubuk gibi T 04:00 UTC zamanina yerlestirilir; saat T 07:00 UTC (10:00 TSI).
  physical_forecast.forecast(...) cagrilir; reddetmeler (Refused) nedenleriyle kaydedilir.
  Hiz icin fiziki seri 2010-01-01'den baslatilir (prim EWMA'si, yari omur 20 gun; etkisi yok).

Kur zamanlamasi (DOGRULANMAMIS varsayim, iki surumle sinirlanir):
  ayni_gun   : T gunu Yahoo gunluk cubugu 04:00 UTC'ye yerlestirilir. Panelin usdtry degeri T tarihli GERCEK
               cubuktur (build_dataset.build: altin ve kur ayni tarihte inner join; as-of tasima yok). Cubugun
               07:00 UTC'den once olustugu yalnizca 2024-2025 icin saatlik veriyle gosterildi; 2011-2015 icin
               yalnizca BIST kapanisindan once oldugu biliniyor. Eski yillarda canli sisteme gore IYIMSER olabilir.
  onceki_gun : T'den once son Yahoo cubugu (kesinlikle tahminden once bilinir) -> guvenli ALT SINIR.
  Canli performansin bu ikisi arasinda olmasi beklenir.
Bilinen farklar: canli panel 2015'te basladigi icin canli egitim fiilen 2015'ten; burada 2011'den.
  ABD tatillerinde (panelde gun yok) burada red, canlida saatlik kurla tahmin olabilir.
  Aralik kantillerinde sonlu orneklem konformal duzeltmesi yok (%95'te ~0.4 puan dusuk kapsam olabilir).

Araliklar (her model icin, %80 ve %95): tahmin aninda GERCEKLESTIGI BILINEN son 250 gunun log artiklarinin
  (gercek / tahmin) ampirik kantilleri; hedef tarihi <= o sabahki son fiziki kapanis tarihi olan gunler.
  Olcutler: kapsam, ortalama genislik (TL), Winkler aralik skoru, Kupiec ve Christoffersen.

Kilit: hedef tarihi >= 2023-01-01 hicbir satir uretilmez; veriler 2023 oncesine kesilir.
Bu modul KESIF amaclidir; buradaki hicbir sonuc bagimsiz basari olarak sunulamaz.
"""

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.live import physical_forecast as lp
from src.models import final_exam as fe
from src.models import intervals as iv
from src.models import physical_eval as pv
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
START, LOCK = pd.Timestamp("2012-01-02"), pd.Timestamp(pv.LOCK_START)
PHYS_FROM = pd.Timestamp("2010-01-01")
WINDOW = 250
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
MODELS = lp.MODELS
ERAS = [("2012-2015", "2012-01-01", "2016-01-01"), ("2016-2022", "2016-01-01", "2023-01-01")]


class BacktestError(Exception):
    pass


def out_dir() -> Path:
    return tr.OUTPUT_DIR / "experiments" / "live_backtest"


FX_MODES = ("ayni_gun", "onceki_gun")


def replay(phys: pd.DataFrame, panel: pd.DataFrame, fx_mode: str = "ayni_gun") -> pd.DataFrame:
    if fx_mode not in FX_MODES:
        raise BacktestError(f"Bilinmeyen kur surumu: {fx_mode}")
    phys = phys[(phys["Date"] >= PHYS_FROM) & (phys["Date"] < LOCK)].sort_values("Date").reset_index(drop=True)
    panel = panel[panel["Date"] < LOCK].sort_values("Date").reset_index(drop=True)
    fx_by_day = panel.set_index("Date")["usdtry"]
    rows = []
    targets = phys.loc[(phys["Date"] >= START) & phys["kapanis_tl_gr"].notna(), ["Date", "kapanis_tl_gr"]]
    for T, actual in targets.itertuples(index=False):
        rec = {"target_date": T, "actual": float(actual), "fx_mode": fx_mode}
        bar_start = pd.Timestamp(T.date(), tz="UTC") + pd.Timedelta(hours=lp.FX_BAR_END_UTC - 1)
        prev = fx_by_day[fx_by_day.index < T]
        if fx_mode == "ayni_gun" and T in fx_by_day.index:            # T gunu Yahoo cubugu
            fx_h = pd.DataFrame({"ts_utc": [bar_start], "close": [float(fx_by_day[T])]})
        elif fx_mode == "onceki_gun" and T in fx_by_day.index and len(prev):   # ayni gunler, eski kur
            fx_h = pd.DataFrame({"ts_utc": [bar_start], "close": [float(prev.iat[-1])]})
        else:                                                          # kur yok -> canli kod reddeder
            fx_h = pd.DataFrame({"ts_utc": pd.Series(dtype="datetime64[ns, UTC]"), "close": pd.Series(dtype=float)})
        now = pd.Timestamp(T.date(), tz="UTC") + pd.Timedelta(hours=7)
        try:
            f = lp.forecast(phys[phys["Date"] < T], panel[panel["Date"] < T], fx_h, now)
            rec.update({"refused": "", "last_date": f["last_date"], "P_t": f["P"], "horizon": f["horizon"],
                        **{f"pred_{m}": v for m, v in f["preds"].items()}})
        except lp.Refused as e:
            rec["refused"] = str(e).split(":")[0][:60]
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out) and pd.to_datetime(out["target_date"]).max() >= LOCK:
        raise BacktestError("Kilitli doneme satir uretildi.")
    return out


def add_intervals(res: pd.DataFrame) -> pd.DataFrame:
    ok = res[res["refused"] == ""].copy().reset_index(drop=True)
    td = pd.to_datetime(ok["target_date"]).to_numpy()
    for m in MODELS:
        resid = np.log(ok["actual"].to_numpy(float) / ok[f"pred_{m}"].to_numpy(float))
        for lvl, (lo, hi) in LEVELS.items():
            L, H = np.full(len(ok), np.nan), np.full(len(ok), np.nan)
            for i in range(len(ok)):
                known = np.where(td < np.datetime64(ok["last_date"].iat[i]) + np.timedelta64(1, "D"))[0]
                known = known[known < i][-WINDOW:]
                if len(known) < WINDOW:
                    continue
                q_lo, q_hi = np.quantile(resid[known], [lo, hi])
                L[i] = ok[f"pred_{m}"].iat[i] * math.exp(q_lo)
                H[i] = ok[f"pred_{m}"].iat[i] * math.exp(q_hi)
            ok[f"lo{lvl}_{m}"], ok[f"hi{lvl}_{m}"] = L, H
    return ok


def point_table(ok: pd.DataFrame) -> pd.DataFrame:
    rows, pvals = [], {}
    a = ok["actual"].to_numpy(float)
    e_rw = np.abs(a - ok["pred_RW"].to_numpy(float))
    for m in MODELS:
        p = ok[f"pred_{m}"].to_numpy(float)
        e = np.abs(a - p)
        r = {"model": m, "n": int(len(ok)), "MAE_TL": float(e.mean()), "MedAE_TL": float(np.median(e)),
             "MAPE_pct": float(100 * np.mean(e / a))}
        if m != "RW":
            dirn = np.sign(p - ok["P_t"].to_numpy(float)) == np.sign(a - ok["P_t"].to_numpy(float))
            b, d = fe.paired_bootstrap(e, e_rw), fe.dm_hln(e, e_rw)
            r.update({"direction_hit_pct": float(100 * dirn.mean()), "improvement_pct": -b["pct"],
                      "ci_low_pct": -b["pct_ci95"][1], "ci_high_pct": -b["pct_ci95"][0], "p_dm": d["p_value"],
                      "dm_stat": d["dm_stat_hln"]})
            pvals[m] = d["p_value"]
        rows.append(r)
    t = pd.DataFrame(rows)
    order = sorted(pvals, key=pvals.get)
    running, holm = 0.0, {}
    for k, m in enumerate(order):
        running = max(running, min(1.0, pvals[m] * (len(order) - k)))
        holm[m] = running
    t["p_holm"] = t["model"].map(holm)
    t["significantly_better"] = (t["p_holm"] < fe.ALPHA) & (t["dm_stat"] < 0) & (t["ci_low_pct"] > 0)
    return t


def interval_table(ok: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = ok["actual"].to_numpy(float)
    for m in MODELS:
        for lvl, (lo, hi) in LEVELS.items():
            L, H = ok[f"lo{lvl}_{m}"].to_numpy(float), ok[f"hi{lvl}_{m}"].to_numpy(float)
            k = np.isfinite(L) & np.isfinite(H)
            inside = (a[k] >= L[k]) & (a[k] <= H[k])
            alpha = 1 - lvl / 100
            w = H[k] - L[k]
            winkler = w + (2 / alpha) * ((L[k] - a[k]) * (a[k] < L[k]) + (a[k] - H[k]) * (a[k] > H[k]))
            miss = (~inside).astype(int)
            rows.append({"model": m, "level": lvl, "n": int(k.sum()), "coverage": float(inside.mean()),
                         "mean_width_TL": float(w.mean()), "mean_width_pct": float(100 * np.mean(w / a[k])),
                         "winkler_TL": float(winkler.mean()), "kupiec_p": iv.kupiec_p(miss, alpha),
                         "christoffersen_p": iv.christoffersen_p(miss)})
    return pd.DataFrame(rows)


def yearly(ok: pd.DataFrame) -> pd.DataFrame:
    a = ok["actual"].to_numpy(float)
    g = pd.to_datetime(ok["target_date"]).dt.year
    out = []
    for y, idx in ok.groupby(g).groups.items():
        e_rw = np.abs(a[idx] - ok.loc[idx, "pred_RW"].to_numpy(float)).mean()
        row = {"year": int(y), "n": int(len(idx))}
        for m in MODELS[1:]:
            row[f"{m}_vs_RW_pct"] = float(100 * (1 - np.abs(a[idx] - ok.loc[idx, f"pred_{m}"].to_numpy(float)).mean() / e_rw))
        out.append(row)
    return pd.DataFrame(out)


def run(phys_path: Path = pv.PHYS_PATH, panel_path: Path = HIST_PANEL) -> dict:
    if Path(panel_path).resolve() == Path(pv.PANEL_PATH).resolve():
        raise BacktestError("Genisletilmis panel gerekli (data/processed/history); ana panel 2015'te basliyor.")
    phys = pd.read_csv(phys_path, parse_dates=["Date"])
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for mode in FX_MODES:
        res = replay(phys, panel, mode)
        ok = add_intervals(res)
        pts, ints, yrs = point_table(ok), interval_table(ok), yearly(ok)
        refusals = res.loc[res["refused"] != "", "refused"].value_counts().to_dict()
        for name, frame in (("replay", res), ("forecasts_with_intervals", ok), ("point", pts),
                            ("intervals", ints), ("yearly", yrs)):
            frame.to_csv(d / f"{name}_{mode}.csv", index=False)
        out[mode] = {"summary": {"targets": int(len(res)), "issued": int(len(ok)),
                                 "refused": int((res["refused"] != "").sum()), "refusal_reasons": refusals,
                                 "significantly_better_than_rw": pts.loc[pts["significantly_better"].fillna(False), "model"].tolist(),
                                 "horizon_counts": ok["horizon"].value_counts().to_dict()},
                     "point": pts, "intervals": ints, "yearly": yrs, "forecasts": ok}
    summary = {"nature": "KESIF - canli protokol geriye donuk (2012-2022)",
               "modes": {m: out[m]["summary"] for m in FX_MODES}}
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser(description="LIVE_PHYS_V1 geriye donuk test ve tahmin araliklari (2012-2022).").parse_args(
        [] if argv is None else argv)
    sep = "=" * 80
    print(f"\n{sep}\nLIVE_PHYS_V1 GERIYE DONUK TEST VE TAHMIN ARALIKLARI (KESIF, 2012-2022)\n{sep}")
    try:
        r = run()
    except (FileNotFoundError, BacktestError) as e:
        print(f"  HATA: {e}")
        return 1
    pd.set_option("display.width", 220)
    for mode in FX_MODES:
        x = r[mode]
        s = x["summary"]
        label = "AYNI GUN KURU (2024-25'te dogrulanmis zamanlama; eski yillarda iyimser olabilir)" if mode == "ayni_gun" \
            else "ONCEKI GUN KURU (kesinlikle bilinen bilgi; GUVENLI ALT SINIR)"
        print(f"\n  ##### {label}")
        print(f"  Hedef gun: {s['targets']} | tahmin: {s['issued']} | reddedilen: {s['refused']} {s['refusal_reasons']}")
        print(f"  Ufuk (gun): {s['horizon_counts']}\n\n  NOKTA TAHMINI (canli kodla, RW'ye gore):")
        print(x["point"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n  YILLARA GORE (RW'ye gore iyilesme %):")
        print(x["yearly"].to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
        print("\n  TAHMIN ARALIKLARI (son 250 bilinen gunun artiklari):")
        print(x["intervals"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  Anlamli olarak RW'den iyi: {s['significantly_better_than_rw'] or 'yok'}")
    print(f"\n  Canli performansin iki surum arasinda olmasi beklenir.\n  Ciktilar: {out_dir()}\n{sep}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
