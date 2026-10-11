"""
Gunluk Kapanis Saati Teshisi (USD/TRY ve altin)
===============================================
Kullanim (proje kokunden):
  python -m src.analysis.fx_close_time

Soru: Yahoo'nun "d" tarihli gunluk kapanisi, gunun hangi anina ait?
Her UTC saati h ve gun farki o (-1: onceki gun, 0: ayni gun) icin, gunluk
kapanis ile bitisi (d+o, h:00 UTC) olan saatlik cubugun kapanisi arasindaki
MEDYAN mutlak log farki hesaplanir. En kucuk medyan farkin oldugu (o, h),
gunluk kapanisin ait oldugu anin tahminidir. Medyan, yavas hareket eden
serilerde de (kur) en iyi saati belirleyebilir; tekil eslesme olcutunun
aksine birden cok saatin yakin olmasindan etkilenmez.

On kayitli yorum: (o*, h*) = argmin medyan fark. Ikinci en iyi saatle oran da
raporlanir; oran ~1 ise belirsizdir. Bu bir VERI KALITESI teshisidir; model
degerlendirmez. 2026 cubuklari (kapanisi 2026-01-01 00:00 UTC ve sonrasi) ve
2026 gunluk satirlari KULLANILMAZ. Saatlik veri yalnizca 2024-10'dan itibaren
mevcut oldugundan analiz test donemine duser (veri kalitesi tanisi icin onaylandi).
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import sync_prices as sp

OUT_DIR = sp.OUT_DIR.parent / "fx_close_time"
OFFSETS = (-1, 0)
SERIES = {"usdtry": "usdtry", "gold": "gold_usd_oz"}


def close_profile(daily: pd.DataFrame, col: str, hourly: pd.DataFrame) -> pd.DataFrame:
    """(gun farki, UTC saat) basina medyan |log(gunluk / saatlik)| ve eslesen gun sayisi."""
    h = hourly.copy()
    h["end"] = (h["ts_utc"] + pd.Timedelta(hours=sp.BAR_HOURS)).dt.tz_convert("UTC").dt.tz_localize(None)
    lookup = h.set_index("end")["close"]
    rows = []
    for o in OFFSETS:
        for hr in range(24):
            ends = pd.to_datetime(daily["Date"]) + pd.Timedelta(days=o) + pd.Timedelta(hours=hr)
            vals = lookup.reindex(ends.to_numpy()).to_numpy()
            ok = np.isfinite(vals)
            diff = np.abs(np.log(daily[col].to_numpy(float)[ok] / vals[ok]))
            rows.append({"day_offset": o, "hour_utc": hr, "days": int(ok.sum()),
                         "median_abs_log_diff": float(np.median(diff)) if ok.any() else np.nan,
                         "share_within_1bp": float(np.mean(diff < 1e-4)) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def best_time(profile: pd.DataFrame, min_days: int = 50) -> dict:
    p = profile[profile["days"] >= min_days].sort_values("median_abs_log_diff")
    if len(p) < 2:
        return {"day_offset": None, "hour_utc": None, "note": "yetersiz gozlem"}
    b, s = p.iloc[0], p.iloc[1]
    return {"day_offset": int(b["day_offset"]), "hour_utc": int(b["hour_utc"]),
            "median_abs_log_diff": float(b["median_abs_log_diff"]), "days": int(b["days"]),
            "second_best": {"day_offset": int(s["day_offset"]), "hour_utc": int(s["hour_utc"]),
                            "median_abs_log_diff": float(s["median_abs_log_diff"])},
            "ratio_second_to_best": float(s["median_abs_log_diff"] / b["median_abs_log_diff"])
            if b["median_abs_log_diff"] > 0 else float("inf")}


def run(snapshot: Path, panel_path: Path, out_dir: Path) -> dict:
    _, hourly_full = sp.load_intraday(snapshot)
    end_utc = pd.Timestamp(sp.HOLDOUT_START, tz="UTC")
    hourly = {k: sp.bars_closed_before(v, end_utc) for k, v in hourly_full.items()}
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    panel = panel[panel["Date"] < pd.Timestamp(sp.HOLDOUT_START)]
    first = min(v["ts_utc"].min() for v in hourly.values()).tz_convert("UTC").tz_localize(None).normalize()
    daily = panel[panel["Date"] >= first + pd.Timedelta(days=2)].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"period": [str(daily["Date"].min().date()), str(daily["Date"].max().date())],
              "holdout_2026_used": False, "series": {}}
    for name, col in SERIES.items():
        prof = close_profile(daily, col, hourly[name])
        prof.to_csv(out_dir / f"profile_{name}.csv", index=False)
        result["series"][name] = best_time(prof)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gunluk kapanis saati teshisi (veri kalitesi, 2026 haric).")
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--panel", type=Path, default=sp.PANEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 66 + "\nGUNLUK KAPANIS SAATI TESHISI (veri kalitesi)\n" + "=" * 66)
    snap = args.snapshot or sp.latest_snapshot(sp.INTRADAY_ROOT)
    if snap is None:
        print("  HATA: saatlik anlik goruntu yok (python -m src.data.collect_intraday).")
        return 1
    try:
        res = run(snap, args.panel, args.out)
    except (sp.IntradayError, FileNotFoundError, ValueError) as e:
        print(f"  HATA: {e}")
        return 1
    print(f"  Donem: {res['period'][0]} .. {res['period'][1]} (2026 haric)")
    for name, b in res["series"].items():
        if b.get("hour_utc") is None:
            print(f"  {name:7s} {b.get('note')}")
            continue
        sb = b["second_best"]
        print(f"  {name:7s} en iyi: gun farki {b['day_offset']:+d}, {b['hour_utc']:02d}:00 UTC "
              f"(medyan fark {1e4 * b['median_abs_log_diff']:.2f} bp, {b['days']} gun) | ikinci: "
              f"{sb['day_offset']:+d}, {sb['hour_utc']:02d}:00 UTC ({1e4 * sb['median_abs_log_diff']:.2f} bp) "
              f"| oran {b['ratio_second_to_best']:.2f}")
    print(f"\n  Ciktilar: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
