"""
Esanli Fiyat Analizi (Veri Saglamlastirma, Adim 1)
==================================================
Kullanim (proje kokunden):
  python -m src.data.collect_intraday          # once saatlik veri (internet)
  python -m src.analysis.sync_prices           # yalnizca gelistirme donemi (varsayilan)
  python -m src.analysis.sync_prices --include-test-period   # bilincli karar gerekir

Soru: Gunluk veri setimizdeki teorik gram fiyati P[t], farkli saatlerde
olusmus altin ve kur kapanislarinin birlesimi mi? Ayni saatteki fiyatlarla
kurulan teorik fiyat bundan ne kadar farkli?

Bu calisma bir KAYNAGIN hatali oldugunu gostermez; saatlik ve gunluk veri ayni
ekonomik olcumu temsil etmeyebilir (gunluk kapanis bir uzlasma fiyati olabilir,
saatlik cubuk o saatteki son islem/kotasyondur). Amac, saat damgasi
uyumsuzluguna dair KANIT toplamaktir.

Donem korumasi
--------------
  Varsayilan: yalnizca TEST_START (2023-10-26) oncesi. --include-test-period:
  test donemi de dahil olur. 2026 (HOLDOUT_START ve sonrasi) HICBIR durumda
  kullanilmaz. Kapsam raporu (hangi tarihlerde veri oldugu) her zaman yazilir;
  fiyat degerleri yalnizca izin verilen donemde okunur.

Gunluk esanli seri kurallari (hepsi raporlanir)
------------------------------------------------
  Saatlik cubugun zaman damgasi cubugun baslangicidir; kapanisi baslangic + 1 saat.
  ny_1700: New York takvim gununde, bitisi 17:00 (America/New_York) veya oncesi
           olan SON ortak cubuk (COMEX gunluk oturum kapanisina yakin).
  ny_1400: ayni kural, bitis 14:00 NY veya oncesi (COMEX uzlasma saatine yakin).
  Esanli cift: altin ve kurun AYNI UTC baslangic zamanli cubuklari.

Gunluk kapanisin hangi saate ait oldugu (cikarim)
-------------------------------------------------
  Her gun icin, gunluk kapanis fiyatina en yakin saatlik kapanis (+/- 1 gun
  penceresinde) bulunur; o cubugun bitis saati kaydedilir. Dagilim, Yahoo'nun
  gunluk kapanisinin gunun hangi saatine karsilik geldigine dair kanittir.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import data_diagnostics as dd
from src.data import build_dataset as bd
from src.data.collect_intraday import DEFAULT_ROOT as INTRADAY_ROOT, INTRADAY_SERIES, read_intraday_csv, IntradayError
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "market_panel.csv"
OUT_DIR = ROOT / "data" / "processed" / "results" / "diagnostics" / "sync"
HOLDOUT_START = "2026-01-01"
NY = "America/New_York"
RULES = {"ny_1700": 17, "ny_1400": 14}
MATCH_WINDOW_HOURS = 36
EXIT_NO_DATA = 3


def period_end(include_test: bool) -> pd.Timestamp:
    return pd.Timestamp(HOLDOUT_START if include_test else tr.TEST_START)


def latest_snapshot(root: Path):
    root = Path(root)
    snaps = sorted([p for p in root.iterdir() if (p / "manifest.json").exists()], reverse=True) if root.exists() else []
    return snaps[0] if snaps else None


def load_intraday(snap: Path) -> tuple:
    manifest = json.loads((Path(snap) / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for name, ticker in INTRADAY_SERIES.items():
        e = manifest["series"].get(name, {})
        if e.get("status") != "ok":
            raise IntradayError(f"{name}: anlik goruntude basarili degil ({e.get('status')}).")
        out[name] = read_intraday_csv(Path(snap) / e["file"], ticker)
    return manifest, out


# ---------------------------------------------------------------------------
# Kapsam (fiyat degeri okumadan)
# ---------------------------------------------------------------------------
def coverage(hourly: dict) -> dict:
    test0, hold0 = pd.Timestamp(tr.TEST_START, tz="UTC"), pd.Timestamp(HOLDOUT_START, tz="UTC")
    cov = {}
    for name, df in hourly.items():
        ts = df["ts_utc"]
        cov[name] = {"first_ts_utc": str(ts.min()), "last_ts_utc": str(ts.max()), "bars": int(len(ts)),
                     "bars_development": int((ts < test0).sum()),
                     "bars_test_period": int(((ts >= test0) & (ts < hold0)).sum()),
                     "bars_2026": int((ts >= hold0).sum()),
                     "minute_offsets": {str(k): int(v) for k, v in ts.dt.minute.value_counts().sort_index().items()}}
    common = set(hourly["gold"]["ts_utc"]) & set(hourly["usdtry"]["ts_utc"])
    common = pd.Series(sorted(common), dtype="datetime64[ns, UTC]")
    cov["common_exact_timestamps"] = {
        "count": int(len(common)),
        "first_ts_utc": str(common.min()) if len(common) else None,
        "last_ts_utc": str(common.max()) if len(common) else None,
        "development": int((common < test0).sum()) if len(common) else 0,
        "test_period": int(((common >= test0) & (common < hold0)).sum()) if len(common) else 0,
        "y2026": int((common >= hold0).sum()) if len(common) else 0,
    }
    return cov


def hour_profile(hourly: dict) -> pd.DataFrame:
    rows = []
    for name, df in hourly.items():
        ts = df["ts_utc"]
        for (wd, hr), n in ts.groupby([ts.dt.dayofweek, ts.dt.hour]).size().items():
            rows.append({"series": name, "weekday_utc": int(wd), "hour_utc": int(hr), "bars": int(n)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Esanli seriler
# ---------------------------------------------------------------------------
def synchronous_pairs(gold: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca AYNI UTC baslangic zamanli cubuklar; farkli saatler eslestirilmez."""
    m = gold.rename(columns={"close": "gold"}).merge(fx.rename(columns={"close": "usdtry"}), on="ts_utc", how="inner")
    m["gram"] = m["gold"] * m["usdtry"] / bd.TROY_OUNCE_GRAM
    m["bar_end_ny"] = (m["ts_utc"] + pd.Timedelta(hours=1)).dt.tz_convert(NY)
    return m.sort_values("ts_utc").reset_index(drop=True)


def daily_from_pairs(pairs: pd.DataFrame, cutoff_hour: int) -> pd.DataFrame:
    """NY takvim gunu basina, bitisi cutoff_hour:00 NY veya oncesi olan SON esanli cubuk."""
    p = pairs.copy()
    end = p["bar_end_ny"]
    p["Date"] = pd.to_datetime(end.dt.date).astype("datetime64[ns]")
    p = p[(end.dt.hour < cutoff_hour) | ((end.dt.hour == cutoff_hour) & (end.dt.minute == 0))]
    last = p.groupby("Date").tail(1).sort_values("Date")
    return last[["Date", "ts_utc", "bar_end_ny", "gold", "usdtry", "gram"]].reset_index(drop=True)


def close_hour_match(daily: pd.DataFrame, col: str, hourly: pd.DataFrame) -> pd.DataFrame:
    """Her gun icin gunluk kapanisa en yakin saatlik kapanisin bitis saati (cikarim)."""
    h = hourly.copy()
    h["bar_end_utc"] = h["ts_utc"] + pd.Timedelta(hours=1)
    end = h["bar_end_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    rows = []
    for d, value in zip(daily["Date"], daily[col]):
        center = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)
        lo, hi = center - pd.Timedelta(hours=MATCH_WINDOW_HOURS), center + pd.Timedelta(hours=MATCH_WINDOW_HOURS)
        i0 = np.searchsorted(end, np.datetime64(lo.tz_localize(None), "ns"))
        i1 = np.searchsorted(end, np.datetime64(hi.tz_localize(None), "ns"))
        if i1 <= i0:
            continue
        cand = h.iloc[i0:i1]
        diff = np.abs(np.log(cand["close"].to_numpy() / value))
        k = int(np.argmin(diff))
        be = cand["bar_end_utc"].iloc[k]
        rows.append({"Date": d, "matched_bar_end_utc": be, "hour_utc": be.hour,
                     "hour_ny": be.tz_convert(NY).hour,
                     "day_offset_ny": (be.tz_convert(NY).normalize().tz_localize(None) - pd.Timestamp(d)).days,
                     "abs_log_diff": float(diff[k]), "candidates": int(len(cand))})
    return pd.DataFrame(rows)


def summarize_match(match: pd.DataFrame) -> dict:
    if match.empty:
        return {"days": 0}
    near = match[match["abs_log_diff"] < 1e-4]
    mode = near["hour_utc"].mode() if len(near) else pd.Series(dtype=int)
    return {"days": int(len(match)),
            "share_within_1bp": float((match["abs_log_diff"] < 1e-4).mean()),
            "median_abs_log_diff": float(match["abs_log_diff"].median()),
            "most_common_hour_utc_among_1bp_matches": int(mode.iloc[0]) if len(mode) else None,
            "hour_utc_distribution_1bp": {str(k): int(v) for k, v in near["hour_utc"].value_counts().sort_index().items()}}


def compare_prices(sync_daily: pd.DataFrame, panel: pd.DataFrame) -> tuple:
    m = panel[["Date", "gold_usd_oz", "usdtry", "theoretical_gram_try"]].merge(sync_daily, on="Date", how="inner")
    m["diff_tl"] = m["gram"] - m["theoretical_gram_try"]
    m["diff_pct"] = 100 * (m["gram"] / m["theoretical_gram_try"] - 1)
    m["gold_diff_pct"] = 100 * (m["gold"] / m["gold_usd_oz"] - 1)
    m["usdtry_diff_pct"] = 100 * (m["usdtry_y"] / m["usdtry_x"] - 1) if "usdtry_y" in m else np.nan
    a = m["diff_pct"].abs()
    stats = {"days_compared": int(len(m)),
             "panel_days_without_sync": int(len(set(panel["Date"]) - set(sync_daily["Date"]))),
             "sync_days_without_panel": int(len(set(sync_daily["Date"]) - set(panel["Date"]))),
             "abs_diff_tl_mean": float(m["diff_tl"].abs().mean()) if len(m) else None,
             "abs_diff_tl_median": float(m["diff_tl"].abs().median()) if len(m) else None,
             "abs_diff_pct_mean": float(a.mean()) if len(m) else None,
             "abs_diff_pct_median": float(a.median()) if len(m) else None,
             "abs_diff_pct_p90": float(a.quantile(0.90)) if len(m) else None,
             "abs_diff_pct_p99": float(a.quantile(0.99)) if len(m) else None,
             "abs_diff_pct_max": float(a.max()) if len(m) else None,
             "share_abs_diff_gt_0_5pct": float((a > 0.5).mean()) if len(m) else None}
    top = m.loc[a.sort_values(ascending=False).index[:20],
                ["Date", "theoretical_gram_try", "gram", "diff_pct", "gold_diff_pct", "usdtry_diff_pct"]]
    return m, stats, top


def compare_returns(merged: pd.DataFrame) -> dict:
    """Ortak gunler uzerinde ardisik log getiriler; gunluk ve esanli serinin farki ve kayma izleri."""
    m = merged.sort_values("Date").reset_index(drop=True)
    if len(m) < 5:
        return {"days": int(len(m))}
    r = pd.DataFrame({
        "daily_gram": np.log(m["theoretical_gram_try"]).diff(), "sync_gram": np.log(m["gram"]).diff(),
        "daily_gold": np.log(m["gold_usd_oz"]).diff(), "daily_fx": np.log(m["usdtry_x"]).diff(),
        "sync_gold": np.log(m["gold"]).diff(), "sync_fx": np.log(m["usdtry_y"]).diff()}).dropna()
    d = (r["sync_gram"] - r["daily_gram"]).abs()
    out = {"days": int(len(r)),
           "gram_return_corr_daily_vs_sync": float(r["daily_gram"].corr(r["sync_gram"])),
           "abs_return_diff_mean": float(d.mean()), "abs_return_diff_median": float(d.median()),
           "abs_return_diff_p90": float(d.quantile(0.9))}
    for tag in ("daily", "sync"):
        for k in (-1, 0, 1):
            c, n = dd.cross_corr(r[f"{tag}_gold"], r[f"{tag}_fx"], k)
            out[f"{tag}_corr_gold_t_fx_t{k:+d}"] = c
    out["band_95"] = float(1.96 / np.sqrt(len(r)))
    return out


def save_plots(merged: pd.DataFrame, matches: dict, target: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if len(merged):
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(merged["Date"], merged["diff_pct"], lw=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("% fark")
        ax.set_title("Esanli (ny_1700) teorik gram fiyati - gunluk veri setindeki teorik fiyat")
        plt.tight_layout()
        plt.savefig(target / "sync_vs_daily_pct_diff.png", dpi=110)
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, (name, mt) in zip(axes, matches.items()):
        if len(mt):
            near = mt[mt["abs_log_diff"] < 1e-4]
            ax.bar(near["hour_utc"].value_counts().sort_index().index, near["hour_utc"].value_counts().sort_index().values)
        ax.set_title(f"{name}: gunluk kapanisa esit saatlik kapanisin bitis saati (UTC)")
        ax.set_xlabel("saat (UTC)")
    plt.tight_layout()
    plt.savefig(target / "daily_close_hour_match.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def run(intraday_snapshot: Path, panel_path: Path, out_dir: Path, include_test: bool = False) -> dict:
    manifest, hourly_full = load_intraday(intraday_snapshot)
    cov = coverage(hourly_full)
    end = period_end(include_test)
    end_utc = end.tz_localize("UTC")
    hourly = {k: v[v["ts_utc"] < end_utc].reset_index(drop=True) for k, v in hourly_full.items()}
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    panel["Date"] = panel["Date"].astype("datetime64[ns]")
    panel = panel[panel["Date"] < end].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"intraday_snapshot": Path(intraday_snapshot).name,
               "intraday_retrieved_at_utc": manifest.get("retrieved_at_utc"),
               "attempts": {k: v.get("attempts") for k, v in manifest["series"].items()},
               "period_end_exclusive": str(end.date()), "include_test_period": include_test,
               "holdout_2026_used": False, "coverage": cov, "rules": RULES}
    pairs = synchronous_pairs(hourly["gold"], hourly["usdtry"])
    if pairs.empty:
        summary["status"] = "no_data_in_period"
        summary["message"] = (f"Izin verilen donemde (< {end.date()}) ortak saatlik veri yok. Ortak saatlik "
                              f"veri {cov['common_exact_timestamps']['first_ts_utc']} tarihinde basliyor.")
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return {"summary": summary, "status": "no_data_in_period"}

    results = {}
    for rule, hour in RULES.items():
        sd = daily_from_pairs(pairs, hour)
        merged, stats, top = compare_prices(sd, panel)
        results[rule] = {"prices": stats, "returns": compare_returns(merged)}
        merged.to_csv(out_dir / f"sync_vs_daily_{rule}.csv", index=False)
        top.to_csv(out_dir / f"top_deviations_{rule}.csv", index=False)
        if rule == "ny_1700":
            plot_frame = merged
    matches = {"gold": close_hour_match(panel, "gold_usd_oz", hourly["gold"]),
               "usdtry": close_hour_match(panel, "usdtry", hourly["usdtry"])}
    for name, mt in matches.items():
        mt.to_csv(out_dir / f"daily_close_hour_match_{name}.csv", index=False)
    hour_profile(hourly).to_csv(out_dir / "hour_profile.csv", index=False)
    save_plots(plot_frame, matches, out_dir)

    summary.update(status="ok", synchronous_pairs=int(len(pairs)),
                   pairs_range=[str(pairs["ts_utc"].min()), str(pairs["ts_utc"].max())],
                   comparison=results,
                   daily_close_hour_inference={k: summarize_match(v) for k, v in matches.items()})
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return {"summary": summary, "status": "ok", "matches": matches, "results": results}


def print_report(res: dict):
    s = res["summary"]
    c = s["coverage"]
    print("  Saatlik kapsam (fiyat okunmadan):")
    for name in INTRADAY_SERIES:
        v = c[name]
        print(f"    {name:7s} {v['first_ts_utc']} .. {v['last_ts_utc']} | {v['bars']} cubuk "
              f"(gelistirme {v['bars_development']}, test {v['bars_test_period']}, 2026 {v['bars_2026']})")
    cc = c["common_exact_timestamps"]
    print(f"    ortak     {cc['count']} ayni zaman damgali cubuk (gelistirme {cc['development']}, "
          f"test {cc['test_period']}, 2026 {cc['y2026']})")
    print(f"  Denemeler: {s['attempts']}")
    print(f"  Analiz donemi: < {s['period_end_exclusive']} (test donemi dahil: {s['include_test_period']}; 2026 asla)")
    if res["status"] != "ok":
        print(f"\n  SONUC: {s['message']}")
        print("  Gelistirme doneminde analiz yapilamadi. Test donemini dahil etmek bilincli bir karar gerektirir:")
        print("    python -m src.analysis.sync_prices --include-test-period")
        return
    print(f"  Esanli cift: {s['synchronous_pairs']} ({s['pairs_range'][0]} .. {s['pairs_range'][1]})")
    for rule, r in s["comparison"].items():
        p, q = r["prices"], r["returns"]
        print(f"\n  Kural {rule}: {p['days_compared']} gun karsilastirildi | panelde olup esanli olmayan: "
              f"{p['panel_days_without_sync']} | esanli olup panelde olmayan: {p['sync_days_without_panel']}")
        if p["days_compared"]:
            print(f"    |fark| TL ort {p['abs_diff_tl_mean']:.2f}, medyan {p['abs_diff_tl_median']:.2f} | "
                  f"|fark| % ort {p['abs_diff_pct_mean']:.3f}, medyan {p['abs_diff_pct_median']:.3f}, "
                  f"p90 {p['abs_diff_pct_p90']:.3f}, p99 {p['abs_diff_pct_p99']:.3f}, max {p['abs_diff_pct_max']:.3f}")
        if q.get("days", 0) >= 5:
            print(f"    getiri korelasyonu (gunluk vs esanli gram): {q['gram_return_corr_daily_vs_sync']:.3f} | "
                  f"|getiri farki| medyan {q['abs_return_diff_median']:.5f}")
            print(f"    kayma izi corr(altin[t], kur[t+1]): gunluk {q['daily_corr_gold_t_fx_t+1']:+.3f}, "
                  f"esanli {q['sync_corr_gold_t_fx_t+1']:+.3f} (bant +/-{q['band_95']:.3f})")
    print("\n  Gunluk kapanisin hangi saate ait oldugu (cikarim):")
    for name, m in s["daily_close_hour_inference"].items():
        if m.get("days"):
            print(f"    {name:7s} {m['days']} gun | 1 baz puan icinde eslesen: %{100 * m['share_within_1bp']:.1f} | "
                  f"en sik saat (UTC): {m['most_common_hour_utc_among_1bp_matches']} | dagilim: {m['hour_utc_distribution_1bp']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Altin ve USD/TRY esanli fiyat analizi (salt okunur).")
    ap.add_argument("--snapshot", type=Path, default=None, help="Saatlik anlik goruntu (varsayilan: en yeni).")
    ap.add_argument("--panel", type=Path, default=PANEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--include-test-period", action="store_true",
                    help="Test donemini de dahil et (2026 asla). Veri kalitesi tanisidir; bilincli karar gerekir.")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 66 + "\nESANLI FIYAT ANALIZI (salt okunur)\n" + "=" * 66)
    snap = args.snapshot or latest_snapshot(INTRADAY_ROOT)
    if snap is None:
        print("  HATA: saatlik anlik goruntu yok. Once: python -m src.data.collect_intraday")
        return 1
    try:
        res = run(snap, args.panel, args.out, args.include_test_period)
    except (IntradayError, FileNotFoundError, ValueError) as e:
        print(f"  HATA: {e}")
        return 1
    print_report(res)
    print(f"\n  Ciktilar: {args.out}")
    return 0 if res["status"] == "ok" else EXIT_NO_DATA


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
