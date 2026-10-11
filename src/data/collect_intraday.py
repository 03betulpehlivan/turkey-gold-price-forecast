"""
Saatlik Veri Toplama (Yahoo Finance) - esanlilik analizi icin
=============================================================
Kullanim (proje kokunden):
  python -m src.data.collect_intraday

Ne yapar:
  Altin (GC=F) ve USD/TRY (USDTRY=X) icin 1 saatlik cubuklari indirir ve
  YENI bir anlik goruntu klasorune yazar:
    data/raw/snapshots_intraday/<YYYYMMDDTHHMMSSZ>/{gold,usdtry}.csv + manifest.json
  Mevcut hicbir dosyanin uzerine yazilmaz; gunluk anlik goruntuler etkilenmez.

Kapsam varsayilmaz:
  Once period="max" istenir; bos donerse period="730d" denenir. Her denemenin
  sonucu manifest'e yazilir. Gercek kapsam, donen verinin ilk/son zaman
  damgasidir. Yahoo'nun saatlik veri sinirlari bu kodu yazan ortamda
  DOGRULANAMADI.

Zaman damgalari:
  yfinance'in dondurdugu zaman damgalari (saat dilimi ofsetiyle) DEGISTIRILMEDEN
  CSV'ye yazilir. Okuyucu (read_intraday_csv) bunlari UTC'ye cevirir.
  Bir cubugun zaman damgasi cubugun BASLANGICIDIR (yfinance davranisi;
  dogrulanmadi). Kapanis fiyati cubugun sonundaki fiyattir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from src.data import collect as co

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "raw" / "snapshots_intraday"
INTRADAY_SERIES = {"gold": "GC=F", "usdtry": "USDTRY=X"}
PERIOD_ATTEMPTS = ("max", "730d")


class IntradayError(Exception):
    """Saatlik dosya okunamadi veya gecersiz."""


def fetch_hourly(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    return yf.download(ticker, period=period, interval="1h", auto_adjust=False, actions=False,
                       progress=False, multi_level_index=True, threads=False)


def read_intraday_csv(path: Path, expected_ticker: str) -> pd.DataFrame:
    """Saatlik yfinance CSV'sini okur -> [ts_utc, close].

    Uc satirli baslik (Price / Ticker / Datetime veya Date) beklenir; sembol
    dogrulanir. Zaman damgalari ofsetleriyle ayristirilip UTC'ye cevrilir.
    """
    path = Path(path)
    if not path.exists():
        raise IntradayError(f"Dosya yok: {path}")
    with open(path, encoding="utf-8-sig") as fh:
        head = [fh.readline().rstrip("\r\n").split(",") for _ in range(3)]
    if not (head[0][0] == "Price" and head[1][0] == "Ticker" and head[2][0] in ("Datetime", "Date")):
        raise IntradayError(f"{path.name}: taninmayan baslik: {head[0][:3]}")
    wrong = sorted({t for t in head[1][1:] if t != expected_ticker})
    if wrong:
        raise IntradayError(f"{path.name}: beklenen sembol {expected_ticker}, dosyada {wrong}.")
    df = pd.read_csv(path, skiprows=3, header=None, names=["ts"] + head[0][1:], dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    if "Close" not in df.columns or df.empty:
        raise IntradayError(f"{path.name}: 'Close' sutunu yok veya veri bos.")
    ts = pd.to_datetime(df["ts"].str.strip(), utc=True, format="ISO8601", errors="coerce")
    if ts.isna().any():
        raise IntradayError(f"{path.name}: {int(ts.isna().sum())} zaman damgasi ayristirilamadi.")
    raw = df["Close"].str.strip()
    close = pd.to_numeric(raw.where(raw != ""), errors="coerce")
    if ((raw != "") & close.isna()).any():
        raise IntradayError(f"{path.name}: sayi olmayan kapanis degeri var.")
    out = pd.DataFrame({"ts_utc": ts, "close": close.astype(float)}).dropna(subset=["close"])
    if out["ts_utc"].duplicated().any():
        raise IntradayError(f"{path.name}: tekrar eden zaman damgasi var.")
    if not np.isfinite(out["close"]).all() or (out["close"] <= 0).any():
        raise IntradayError(f"{path.name}: sifir, negatif veya sonsuz kapanis.")
    return out.sort_values("ts_utc").reset_index(drop=True)


def collect_intraday(names=None, root: Path = DEFAULT_ROOT, fetch=None, now: datetime = None) -> dict:
    fetch = fetch or fetch_hourly
    names = names or list(INTRADAY_SERIES)
    snap = co.new_snapshot_dir(Path(root), now)
    manifest = {"retrieved_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                "source": "Yahoo Finance via yfinance", "interval": "1h", "auto_adjust": False,
                "yfinance_version": co._version("yfinance"), "snapshot_dir": snap.name, "series": {}}
    for name in names:
        ticker = INTRADAY_SERIES[name]
        entry = {"ticker": ticker, "attempts": [], "status": None}
        print(f"  {name:7s} ({ticker}) ... ", end="")
        raw = None
        for period in PERIOD_ATTEMPTS:
            try:
                df = fetch(ticker, period)
            except Exception as e:
                entry["attempts"].append({"period": period, "result": f"hata: {type(e).__name__}: {e}"})
                continue
            n = 0 if df is None else len(df)
            entry["attempts"].append({"period": period, "rows": int(n)})
            if n:
                raw, entry["period_used"] = df, period
                break
        if raw is None:
            entry["status"] = "empty_or_error"
            print("VERI YOK")
            manifest["series"][name] = entry
            continue
        idx = pd.DatetimeIndex(raw.index)
        entry["index_tz"] = str(idx.tz) if idx.tz is not None else None
        path = snap / f"{name}.csv"
        raw.to_csv(path)
        entry["file"] = path.name
        entry["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        try:
            parsed = read_intraday_csv(path, ticker)
            entry.update(status="ok", rows=int(len(parsed)),
                         first_ts_utc=str(parsed["ts_utc"].iloc[0]), last_ts_utc=str(parsed["ts_utc"].iloc[-1]))
            print(f"ok ({len(parsed)} cubuk, {entry['first_ts_utc']} .. {entry['last_ts_utc']}, "
                  f"period={entry['period_used']}, tz={entry['index_tz']})")
        except IntradayError as e:
            entry.update(status="invalid", error=str(e))
            print(f"GECERSIZ: {e}")
        manifest["series"][name] = entry
    with open(snap / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["_path"] = str(snap)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Altin ve USD/TRY saatlik veri toplama (degismez anlik goruntu).")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 60 + "\nSAATLIK VERI TOPLAMA - Yahoo Finance\n" + "=" * 60)
    m = collect_intraday(root=args.root)
    failed = [n for n, e in m["series"].items() if e["status"] != "ok"]
    print(f"\nAnlik goruntu: {m['_path']}")
    if failed:
        print(f"HATA: basarisiz seriler: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
