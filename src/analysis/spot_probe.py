"""
Spot Ons Kaynagi Taramasi (salt teshis)
=======================================
Kullanim (proje kokunden, internet gerekir):
  python -m src.analysis.spot_probe

Soru: GC=F (surekli vadeli) kontrat degisim gunlerinde sahte sicramalar iceriyor ve spot
fiyattan taşima maliyeti kadar yuksek. Kontrat degisimi OLMAYAN, spotu izleyen bir kaynak
(XAUUSD=X spot, GLD / IAU fiziki altin ETF'leri) saatlik veride kullanilabilir mi?

Yapilan:
  1) Adaylarin saatlik verisi YENI bir anlik goruntuye indirilir:
     data/raw/snapshots_spot_probe/<zaman>/  (mevcut arsive dokunulmaz)
  2) Her aday icin her NY gunu, bitisi 14:00 NY'ye en yakin (+/-30 dk) cubuk alinir
     (ETF cubuklari :30'da baslar). Saat farki raporlanir.
  3) GC=F esanli serisiyle (data/processed/sync/sync_daily.csv) ortak gunlerde log getiriler.
  4) Kontrat testi: GC=F saatlik serisinin kontrat degistirdigi gunlerde (ROLL_DAYS) artik
     r_gc - r_aday, diger gunlerin saglam olcegine (1.4826 x MAD) gore z-skoru.

On kayitli secim kurali (sonuclardan ONCE):
  Aday UYGUN: her ROLL_DAYS gununde z >= 3 (GC'deki sicrama adayda YOK) VE kontrat disi
  gunlerde getiri korelasyonu > 0.95. Uygun adaylar arasinda kapsami en yuksek olan secilir.
  Hicbiri uygun degilse: "gecici cozum" (kontrat gunlerini isaretleyip dislama).
Cikti: data/processed/results/diagnostics/spot_probe/  (salt teshis; model degerlendirmez)
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import sync_prices as sp
from src.data import collect as co
from src.data import collect_intraday as ci
from src.data import sync_dataset as sd

ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = ROOT / "data" / "raw" / "snapshots_spot_probe"
OUT_DIR = ROOT / "data" / "processed" / "results" / "diagnostics" / "spot_probe"
SYNC_PATH = sd.OUT_DIR / "sync_daily.csv"
CANDIDATES = {"xauusd": "XAUUSD=X", "gld": "GLD", "iau": "IAU"}
ROLL_DAYS = ["2025-07-30", "2026-07-29"]          # GC=F saatlik serisinin kontrat degistirdigi gunler (teshis)
TOLERANCE_MIN = 30
MIN_Z, MIN_CORR = 3.0, 0.95


def download(root: Path = PROBE_ROOT, fetch=None, now: datetime = None) -> Path:
    fetch = fetch or ci.fetch_hourly
    snap = co.new_snapshot_dir(Path(root), now)
    manifest = {"retrieved_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                "series": {}}
    for name, ticker in CANDIDATES.items():
        e = {"ticker": ticker, "attempts": []}
        df = None
        for period in ci.PERIOD_ATTEMPTS:
            try:
                d = fetch(ticker, period)
            except Exception as ex:
                e["attempts"].append({"period": period, "result": f"hata: {type(ex).__name__}: {ex}"})
                continue
            e["attempts"].append({"period": period, "rows": 0 if d is None else int(len(d))})
            if d is not None and len(d):
                df = d
                break
        if df is None:
            e["status"] = "veri_yok"
        else:
            path = snap / f"{name}.csv"
            df.to_csv(path)
            e.update(status="ok", file=path.name, md5=hashlib.md5(path.read_bytes()).hexdigest())
        manifest["series"][name] = e
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap


def daily_near_1400(hourly: pd.DataFrame) -> pd.DataFrame:
    """NY gunu basina bitisi 14:00 NY'ye en yakin (+/-TOLERANCE_MIN) tamamlanmis cubuk."""
    h = hourly.copy()
    end = (h["ts_utc"] + pd.Timedelta(hours=sp.BAR_HOURS)).dt.tz_convert(sp.NY)
    h["Date"] = pd.to_datetime(end.dt.date).astype("datetime64[ns]")
    target = end.dt.normalize() + pd.Timedelta(hours=14)
    h["offset_min"] = (end - target).dt.total_seconds() / 60
    h = h[h["offset_min"].abs() <= TOLERANCE_MIN]
    h = h.assign(a=h["offset_min"].abs()).sort_values(["Date", "a", "offset_min"]).drop_duplicates("Date")
    return h[["Date", "close", "offset_min"]].sort_values("Date").reset_index(drop=True)


def evaluate(cand_daily: pd.DataFrame, gc_daily: pd.DataFrame) -> dict:
    m = gc_daily[["Date", "gold_usd_oz"]].merge(cand_daily, on="Date", how="inner").sort_values("Date")
    if len(m) < 30:
        return {"common_days": int(len(m)), "eligible": False, "note": "yetersiz ortak gun"}
    r_gc = np.log(m["gold_usd_oz"]).diff()
    r_c = np.log(m["close"]).diff()
    res = (r_gc - r_c).to_numpy()
    roll_mask = m["Date"].isin(pd.to_datetime(ROLL_DAYS)).to_numpy()
    base = res[~roll_mask & np.isfinite(res)]
    scale = max(1.4826 * np.median(np.abs(base - np.median(base))), 1e-6)   # 0.01 bp alt sinir (sifira bolme yok)
    z = {str(d.date()): float((res[i] - np.median(base)) / scale) if scale > 0 else np.nan
         for i, d in enumerate(m["Date"]) if roll_mask[i]}
    ok_idx = ~roll_mask & np.isfinite(r_gc.to_numpy()) & np.isfinite(r_c.to_numpy())
    corr = float(np.corrcoef(r_gc.to_numpy()[ok_idx], r_c.to_numpy()[ok_idx])[0, 1])
    ratio = (m["gold_usd_oz"] / m["close"]).to_numpy()
    roll_ok = bool(len(z) == len(ROLL_DAYS) and all(np.isfinite(v) and v >= MIN_Z for v in z.values()))
    return {"common_days": int(len(m)), "coverage_of_gc_days": float(len(m) / len(gc_daily)),
            "offset_min_counts": {str(int(k)): int(v) for k, v in cand_daily["offset_min"].value_counts().items()},
            "roll_day_z": z, "roll_ok": roll_ok, "corr_non_roll": corr,
            "median_abs_return_diff_bp": float(1e4 * np.median(np.abs(base))),
            "gc_to_candidate_ratio_first_last": [float(ratio[0]), float(ratio[-1])],
            "eligible": bool(roll_ok and corr > MIN_CORR)}


def choose(results: dict) -> dict:
    ok = {n: r for n, r in results.items() if r.get("eligible")}
    if not ok:
        return {"selected": None, "decision": "gecici cozum: kontrat gunlerini isaretle ve disla"}
    best = max(ok, key=lambda n: ok[n]["coverage_of_gc_days"])
    return {"selected": best, "ticker": CANDIDATES[best], "decision": f"{CANDIDATES[best]} spot kaynak adayi"}


def run(snapshot: Path, sync_path: Path = SYNC_PATH, out_dir: Path = OUT_DIR) -> dict:
    gc = sd.load_sync_daily(sync_path)
    gc["Date"] = gc["Date"].astype("datetime64[ns]")
    manifest = json.loads((Path(snapshot) / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for name, e in manifest["series"].items():
        if e.get("status") != "ok":
            results[name] = {"eligible": False, "note": "veri yok", "attempts": e.get("attempts")}
            continue
        hourly = ci.read_intraday_csv(Path(snapshot) / e["file"], e["ticker"])
        retrieved = pd.Timestamp(manifest["retrieved_at_utc"])
        hourly = hourly[hourly["ts_utc"] + pd.Timedelta(hours=1) <= retrieved]       # tamamlanmamis cubuk yok
        results[name] = evaluate(daily_near_1400(hourly), gc)
    decision = choose(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"snapshot": Path(snapshot).name, "rule": {"min_z": MIN_Z, "min_corr": MIN_CORR, "roll_days": ROLL_DAYS},
               "results": results, "decision": decision}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spot ons kaynagi taramasi (salt teshis).")
    ap.add_argument("--snapshot", type=Path, default=None, help="Var olan tarama goruntusu (indirme yapmaz).")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 66 + "\nSPOT ONS KAYNAGI TARAMASI (salt teshis)\n" + "=" * 66)
    try:
        snap = args.snapshot or download()
        s = run(snap)
    except (FileNotFoundError, ValueError, ci.IntradayError) as e:
        print(f"  HATA: {e}")
        return 1
    for n, r in s["results"].items():
        if "corr_non_roll" not in r:
            print(f"  {n:7s} {r.get('note')}")
            continue
        z = ", ".join(f"{d}: z={v:+.1f}" for d, v in r["roll_day_z"].items())
        print(f"  {n:7s} ortak gun {r['common_days']} (kapsam %{100 * r['coverage_of_gc_days']:.1f}) | korelasyon "
              f"{r['corr_non_roll']:.3f} | kontrat gunleri [{z}] | uygun: {r['eligible']}")
    print(f"\n  KARAR: {s['decision']['decision']}\n  Cikti: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
