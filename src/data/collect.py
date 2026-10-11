"""
Veri Toplama (Yahoo Finance)
============================
Kullanim (proje kokunden):
  python -m src.data.collect                         # tum seriler, 2015-01-01 .. dun
  python -m src.data.collect --series gold,usdtry    # yalnizca secilen seriler
  python -m src.data.collect --start 2015-01-01 --end 2026-10-12

Ne yapar:
  Her calistirma YENI bir anlik goruntu klasoru olusturur:
    data/raw/snapshots/<YYYYMMDDTHHMMSSZ>/<seri>.csv
    data/raw/snapshots/<YYYYMMDDTHHMMSSZ>/manifest.json
  Mevcut hicbir dosyanin uzerine yazilmaz. data/raw/gold_futures.csv ve
  data/raw/usdtry.csv (Faz A/B'nin girdileri) degistirilmez.

Ham veri biciminin korunmasi:
  yfinance.download(..., auto_adjust=False) ciktisi pandas.to_csv ile yazilir;
  bu, mevcut ham dosyalarla ayni uc satirli basliktir ve build_dataset.py'nin
  okuyucusu (read_yahoo_csv) ile dogrulanir.

Tarih araligi:
  --end HARIC tutulur (yfinance davranisi). Varsayilan --end bugundur; boylece
  henuz kapanmamis bugunun kismi gunluk cubugu alinmaz.

Onemli: Sembollerin Yahoo'da mevcut oldugu, tarih kapsamlari ve kapanis
saatleri bu kodu yazan ortamda DOGRULANAMADI. Ilk calistirmanin manifest'i
bu bilgileri kayda gecirir; durum 'ok' olmayan seriler panele alinmaz.

Cikis kodlari:
  0 = zorunlu seriler (gold, usdtry) basarili (istege bagli hatalar raporlanir)
  1 = zorunlu serilerden en az biri basarisiz veya gecersiz arguman
"""

from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

import pandas as pd

from src.data import build_dataset as bd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_ROOT = ROOT / "data" / "raw" / "snapshots"
DEFAULT_START = "2015-01-01"

# ---------------------------------------------------------------------------
# Seri kaydi. Aciklamalar genel piyasa bilgisidir; Yahoo belgelerinden
# dogrulanmadi. 'required' seriler mevcut hedefin (teorik gram altin) girdisidir.
# ---------------------------------------------------------------------------
SERIES = {
    "gold":    {"ticker": "GC=F",     "required": True,
                "description": "COMEX altin vadeli, surekli en yakin kontrat (USD/ons); spot degil"},
    "usdtry":  {"ticker": "USDTRY=X", "required": True,
                "description": "USD/TRY doviz kuru (TRY per USD)"},
    "silver":  {"ticker": "SI=F",     "required": False,
                "description": "COMEX gumus vadeli, surekli kontrat (USD/ons)"},
    "dxy":     {"ticker": "DX-Y.NYB", "required": False,
                "description": "ICE ABD dolar endeksi"},
    "us10y":   {"ticker": "^TNX",     "required": False,
                "description": "ABD 10 yillik Hazine getirisi endeksi (birim dogrulanmadi)"},
    "vix":     {"ticker": "^VIX",     "required": False,
                "description": "CBOE S&P 500 oynaklik endeksi"},
    "sp500":   {"ticker": "^GSPC",    "required": False,
                "description": "S&P 500 endeksi"},
    "bist100": {"ticker": "XU100.IS", "required": False,
                "description": "Borsa Istanbul BIST 100 endeksi (TRY)"},
    "brent":   {"ticker": "BZ=F",     "required": False,
                "description": "Brent petrol vadeli, surekli kontrat (USD/varil)"},
    "eurtry":  {"ticker": "EURTRY=X", "required": False,
                "description": "EUR/TRY doviz kuru"},
}


def default_end(today: date = None) -> str:
    """Varsayilan bitis (haric): bugun. Kismi gunluk cubuk alinmaz."""
    return (today or datetime.now(timezone.utc).date()).isoformat()


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yfinance ile gunluk veri indirir (auto_adjust=False: 'Adj Close' korunur)."""
    import yfinance as yf
    return yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False,
                       actions=False, progress=False, multi_level_index=True, threads=False)


def normalize_daily_index(df: pd.DataFrame) -> tuple:
    """Gunluk indeksi saat ve saat dilimi bilgisi olmadan tarihe indirger.

    yfinance gunluk verisi genelde zaten tarih olarak gelir; saat dilimli
    veya gece yarisi disi saatli gelirse orijinal ilk zaman damgasi raporlanir.
    Dondurur: (df, bilgi sozlugu).
    """
    idx = pd.DatetimeIndex(df.index)
    info = {"index_tz": str(idx.tz) if idx.tz is not None else None,
            "index_had_time": bool(len(idx) and (idx != idx.normalize()).any()),
            "first_raw_timestamp": str(idx[0]) if len(idx) else None}
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = df.copy()
    out.index = pd.DatetimeIndex(idx.normalize(), name="Date")
    return out, info


def new_snapshot_dir(root: Path, now: datetime = None) -> Path:
    """Benzersiz, yeni bir anlik goruntu klasoru olusturur (asla var olani kullanmaz)."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    path = root / stamp
    k = 2
    while path.exists():
        path = root / f"{stamp}_{k}"
        k += 1
    path.mkdir()
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect(names: list, start: str, end: str, snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
            fetch=None, now: datetime = None) -> dict:
    """Secilen serileri indirir, yeni bir anlik goruntuye yazar, manifest dondurur.

    Bir serinin hatasi digerlerini durdurmaz. Her serinin durumu:
      ok      : indirildi, yazildi ve read_yahoo_csv ile dogrulandi
      empty   : kaynak bos veri dondurdu (dosya yazilmaz)
      error   : indirme sirasinda hata
      invalid : dosya yazildi ama dogrulama basarisiz (sebep kaydedilir)
    """
    fetch = fetch or fetch_yahoo
    snap = new_snapshot_dir(Path(snapshot_root), now)
    retrieved = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    manifest = {
        "retrieved_at_utc": retrieved,
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": _version("yfinance"),
        "pandas_version": _version("pandas"),
        "request": {"start": start, "end_exclusive": end, "interval": "1d", "auto_adjust": False},
        "snapshot_dir": snap.name,
        "series": {},
    }
    for name in names:
        meta = SERIES[name]
        entry = {"ticker": meta["ticker"], "required": meta["required"],
                 "description": meta["description"], "status": None}
        print(f"  {name:8s} ({meta['ticker']}) ... ", end="")
        try:
            raw = fetch(meta["ticker"], start, end)
        except Exception as e:  # ag, kaynak veya kutuphane hatasi
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
            print(f"HATA: {entry['error']}")
            manifest["series"][name] = entry
            continue
        if raw is None or len(raw) == 0:
            entry["status"] = "empty"
            print("BOS (kaynak veri dondurmedi)")
            manifest["series"][name] = entry
            continue

        df, idx_info = normalize_daily_index(raw)
        path = snap / f"{name}.csv"
        df.to_csv(path)
        entry.update(file=path.name, md5=_md5(path), rows_raw=int(len(df)), **idx_info)
        try:
            parsed, _ = _quiet(bd.read_yahoo_csv, path, meta["ticker"])
            entry.update(status="ok", rows_valid=int(len(parsed)),
                         first_date=str(parsed["Date"].iloc[0].date()),
                         last_date=str(parsed["Date"].iloc[-1].date()))
            print(f"ok ({len(parsed)} satir, {entry['first_date']} .. {entry['last_date']})")
        except bd.DataError as e:
            entry.update(status="invalid", error=str(e))
            print(f"GECERSIZ: {e}")
        manifest["series"][name] = entry

    with open(snap / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["_path"] = str(snap)
    return manifest


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "bulunamadi"


def _quiet(func, *args):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args), buf.getvalue()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Yahoo Finance'ten gunluk seri toplama (degismez anlik goruntu).")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None, help="Haric bitis tarihi (varsayilan: bugun).")
    ap.add_argument("--series", default=",".join(SERIES),
                    help=f"Virgulle ayrilmis seri adlari. Secenekler: {', '.join(SERIES)}")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    args = ap.parse_args(argv)
    args.names = [s.strip() for s in args.series.split(",") if s.strip()]
    unknown = [s for s in args.names if s not in SERIES]
    if unknown:
        ap.error(f"Bilinmeyen seri: {unknown}")
    args.end = args.end or default_end()
    for label, value in (("--start", args.start), ("--end", args.end)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            ap.error(f"{label} YYYY-MM-DD olmali: {value}")
    if args.start >= args.end:
        ap.error("--start, --end'den once olmali.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print("VERI TOPLAMA - Yahoo Finance")
    print("=" * 60)
    print(f"Aralik: {args.start} .. {args.end} (bitis haric) | seriler: {', '.join(args.names)}\n")
    manifest = collect(args.names, args.start, args.end, args.snapshot_root)

    failed_required = [n for n, e in manifest["series"].items() if e["required"] and e["status"] != "ok"]
    failed_optional = [n for n, e in manifest["series"].items() if not e["required"] and e["status"] != "ok"]
    print(f"\nAnlik goruntu: {manifest['_path']}")
    if failed_optional:
        print(f"UYARI: istege bagli seriler basarisiz (panele alinmaz): {failed_optional}")
    if failed_required:
        print(f"HATA: zorunlu seriler basarisiz: {failed_required}")
        return 1
    print("Zorunlu seriler basarili.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
