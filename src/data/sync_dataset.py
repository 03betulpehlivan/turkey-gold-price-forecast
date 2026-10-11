"""
Saatlik Arsiv ve Esanli Gunluk Seri
===================================
Kullanim (proje kokunden):
  python -m src.data.collect_intraday      # her gun (Yahoo saatlik veriyi ~730 gun tutar)
  python -m src.data.sync_dataset           # arsivi birlestir, esanli gunluk seriyi yaz

Neden: Yahoo gunluk serisinde altin ~18:00 UTC'ye, USD/TRY ~02-05 UTC'ye ait
(fx_close_time teshisi). Mevcut "teorik gram fiyati" bu yuzden ~13-16 saat
farkli anlardaki fiyatlarin carpimi. Bu modul ons ve kuru AYNI saatlik cubuktan alir.

Arsiv: data/raw/snapshots_intraday/ altindaki TUM anlik goruntuler birlestirilir.
  Yahoo en eski saatlik gunu her gun dusurdugu icin eski goruntuler arsivin tek
  kaynagidir; bu modul onlari ASLA silmez veya degistirmez.
  Tamamlanmamis cubuk: bir goruntude kapanisi (baslangic + 1 saat) goruntunun indirme
  zamanindan SONRA olan cubuk o goruntuden atilir (sonraki goruntude tamamlanmis hali gelir).
  Ayni zaman damgasi birden cok goruntude varsa EN YENI goruntunun degeri alinir;
  farkli degerler (revizyon) sayilir ve raporlanir.

Gunluk kural (ny_1400, kesin): New York takvim gunu basina bitisi TAM 14:00 NY olan ortak
  (ayni UTC baslangicli) cubuk. Bu cubuk yoksa (eksik veri, tatil, henuz kapanmamis) gun
  ATILIR ve raporlanir; daha erken bir cubuga dusulmez. Boylece her gozlem ayni saatin
  fiyatidir ve erken calistirmada sonradan degisecek bir fiyat uretilmez.

Cikti (mevcut veri setlerine DOKUNULMAZ):
  data/processed/sync/hourly_gold.csv, hourly_usdtry.csv   (birlesik arsiv)
  data/processed/sync/sync_daily.csv                        (zaman damgali esanli seri)
  data/processed/sync/sync_report.json
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.analysis import sync_prices as sp
from src.data.collect_intraday import DEFAULT_ROOT as INTRADAY_ROOT, INTRADAY_SERIES, read_intraday_csv, IntradayError

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "sync"
RULE_HOUR_NY = 14


def merge_snapshots(root: Path = INTRADAY_ROOT) -> tuple:
    """Tum saatlik anlik goruntuleri birlestirir. Dondurur: ({seri: [ts_utc, close, snapshot]}, rapor)."""
    root = Path(root)
    snaps = sorted([p for p in root.iterdir() if (p / "manifest.json").exists()]) if root.exists() else []
    if not snaps:
        raise IntradayError(f"Saatlik anlik goruntu yok: {root}")
    parts = {n: [] for n in INTRADAY_SERIES}
    used = []
    incomplete = {n: 0 for n in INTRADAY_SERIES}
    for snap in snaps:
        m = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        retrieved = pd.Timestamp(m["retrieved_at_utc"])
        retrieved = retrieved.tz_localize("UTC") if retrieved.tzinfo is None else retrieved.tz_convert("UTC")
        ok = [n for n in INTRADAY_SERIES if m["series"].get(n, {}).get("status") == "ok"]
        for n in ok:
            df = read_intraday_csv(snap / m["series"][n]["file"], INTRADAY_SERIES[n])
            done = df["ts_utc"] + pd.Timedelta(hours=sp.BAR_HOURS) <= retrieved
            incomplete[n] += int((~done).sum())
            df = df[done].copy()
            df["snapshot"] = snap.name
            parts[n].append(df)
        if ok:
            used.append(snap.name)
    merged, report = {}, {"snapshots": used, "series": {}}
    for n, frames in parts.items():
        if not frames:
            raise IntradayError(f"{n}: hicbir anlik goruntude basarili veri yok.")
        allb = pd.concat(frames, ignore_index=True).sort_values(["ts_utc", "snapshot"])
        spread = allb.groupby("ts_utc")["close"].agg(lambda s: float(s.max() - s.min()))
        latest = allb.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
        merged[n] = latest
        report["series"][n] = {"bars": int(len(latest)), "first_ts_utc": str(latest["ts_utc"].min()),
                               "last_ts_utc": str(latest["ts_utc"].max()),
                               "revised_bars": int((spread > 1e-9).sum()),
                               "incomplete_bars_dropped": incomplete[n]}
    return merged, report


def build_sync_daily(hourly: dict) -> tuple:
    pairs = sp.synchronous_pairs(hourly["gold"][["ts_utc", "close"]], hourly["usdtry"][["ts_utc", "close"]])
    daily = sp.daily_from_pairs(pairs, RULE_HOUR_NY)
    end = daily["bar_end_ny"]
    thin = ~((end.dt.hour == RULE_HOUR_NY) & (end.dt.minute == 0))
    out = daily[~thin].rename(columns={"ts_utc": "bar_start_utc", "gold": "gold_usd_oz", "gram": "theoretical_gram_try"})
    out["bar_end_utc"] = out["bar_start_utc"] + pd.Timedelta(hours=sp.BAR_HOURS)
    out = out[["Date", "bar_start_utc", "bar_end_utc", "bar_end_ny", "gold_usd_oz", "usdtry", "theoretical_gram_try"]]
    return out.reset_index(drop=True), {"days": int(len(out)), "dropped_thin_days": int(thin.sum()),
                                        "dropped_dates": [str(d.date()) for d in daily.loc[thin, "Date"]]}


def run(root: Path = INTRADAY_ROOT, out_dir: Path = OUT_DIR) -> dict:
    hourly, rep = merge_snapshots(root)
    daily, drep = build_sync_daily(hourly)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, df in hourly.items():
        df.to_csv(out_dir / f"hourly_{n}.csv", index=False)
    # date_format TUM tarih-saat sutunlarina uygulanir; saatler kaybolmasin diye acik ISO metni yazilir.
    out = daily.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for c in ("bar_start_utc", "bar_end_utc", "bar_end_ny"):
        out[c] = out[c].map(lambda x: x.isoformat())
    out.to_csv(out_dir / "sync_daily.csv", index=False)
    report = {"built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rule": f"NY gunu basina bitisi TAM {RULE_HOUR_NY}:00 NY olan ortak cubuk; yoksa gun atilir",
              **rep, "daily": drep,
              "date_range": [str(daily["Date"].min().date()), str(daily["Date"].max().date())] if len(daily) else None}
    (out_dir / "sync_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report": report, "daily": daily}


def load_sync_daily(path: Path = OUT_DIR / "sync_daily.csv") -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Esanli seri yok: {path} (python -m src.data.sync_dataset)")
    df = pd.read_csv(path, parse_dates=["Date"])
    for c in ("bar_start_utc", "bar_end_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    if not df["Date"].is_monotonic_increasing or df["Date"].duplicated().any():
        raise ValueError("Esanli seri tarihleri kesin artan degil.")
    if not np.isfinite(df["theoretical_gram_try"]).all() or (df["theoretical_gram_try"] <= 0).any():
        raise ValueError("Esanli seride gecersiz fiyat var.")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saatlik arsiv ve esanli gunluk seri.")
    ap.add_argument("--root", type=Path, default=INTRADAY_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 64 + "\nESANLI GUNLUK SERI (ny_1400)\n" + "=" * 64)
    try:
        r = run(args.root, args.out)["report"]
    except (IntradayError, ValueError) as e:
        print(f"  HATA: {e}")
        return 1
    print(f"  Anlik goruntu: {len(r['snapshots'])} | gunler: {r['daily']['days']} "
          f"({r['date_range'][0]} .. {r['date_range'][1]}) | 14:00 NY cubugu olmadigi icin atilan: "
          f"{r['daily']['dropped_thin_days']}")
    for n, s in r["series"].items():
        print(f"  {n:7s} {s['bars']} cubuk | revize edilen: {s['revised_bars']} | tamamlanmamis (atilan): "
              f"{s['incomplete_bars_dropped']}")
    print(f"  Cikti: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
