"""
Makro Gostergeler (TCMB EVDS, aylik)
====================================
Kullanim (proje kokunden; anahtar YALNIZCA ortam degiskeninde: EVDS_API_KEY):
  python -m src.data.macro indir                      # 01-01-2004'ten bugune
  python -m src.data.macro indir --baslangic 01-01-2004

Seriler (EVDS katalogundan dogrulandi, 16.10.2026):
  tufe         TP.GENENDEKS.T1     Tuketici Fiyat Endeksi, Genel (2003=100), aylik, 2003-01'den
  politika_tr  TP.BISPOLFAIZ.TUR   Turkiye Merkez Bankasi politika faizi (BIS derlemesi), aylik, 2002-02'den
  politika_us  TP.BISPOLFAIZ.USA   ABD Merkez Bankasi politika faizi (BIS derlemesi), aylik

Neden ayri modul: src/data/evds.py yalnizca GUNLUK tarih bicimini kabul eder (bilerek; sabah gorevi kullanir,
  degistirilmez). EVDS aylik serileri "YYYY-M" bicimiyle dondurur; bu modul bu bicimi ayristirir.
Ciktilar:
  data/raw/evds_macro/<zaman>/   ham yanit + manifest (md5)  [mevcut arsive dokunulmaz]
  data/processed/macro/macro_monthly.csv   month (YYYY-MM) + seri sutunlari
Yayim gecikmesi bu modulde UYGULANMAZ (ham deger, ait oldugu ay); analizler gecikmeyi kendisi uygular.
Anahtar hicbir dosyaya veya ekrana yazilmaz.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import re
import sys

import pandas as pd

from src.data import evds

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "evds_macro"
OUT_PATH = ROOT / "data" / "processed" / "macro" / "macro_monthly.csv"
SERIES = {"tufe": "TP.GENENDEKS.T1", "politika_tr": "TP.BISPOLFAIZ.TUR", "politika_us": "TP.BISPOLFAIZ.USA"}
DEFAULT_START = "01-01-2004"
_MONTHLY = re.compile(r"^(\d{4})-(\d{1,2})$")
_DAILY = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")


class MacroError(Exception):
    pass


def parse_month(value: str) -> str:
    v = str(value).strip()
    m = _MONTHLY.match(v)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
    else:
        d = _DAILY.match(v)
        if not d:
            raise MacroError(f"Taninmayan tarih bicimi: {v!r}")
        y, mo = int(d.group(3)), int(d.group(2))
    if not 1 <= mo <= 12:
        raise MacroError(f"Gecersiz ay: {v!r}")
    return f"{y:04d}-{mo:02d}"


def parse_items(items: list, series: dict = None) -> pd.DataFrame:
    series = series or SERIES
    if not items:
        raise MacroError("EVDS bos yanit dondurdu.")
    key = next((k for k in evds.DATE_KEYS if k in items[0]), None)
    if key is None:
        raise MacroError(f"Tarih alani yok; alanlar: {sorted(items[0])}")
    rows = []
    for it in items:
        row = {"month": parse_month(it[key])}
        for name, code in series.items():
            row[name] = pd.to_numeric(it.get(code.replace(".", "_")), errors="coerce")
        rows.append(row)
    df = pd.DataFrame(rows)
    dup = df["month"].duplicated(keep=False)
    if dup.any():
        if (df[dup].groupby("month")[list(series)].nunique(dropna=False) > 1).any().any():
            raise MacroError("Ayni ay icin farkli degerler geldi.")
        df = df.drop_duplicates("month")
    return df.sort_values("month").reset_index(drop=True)


def download(client, start: str = DEFAULT_START, end: str = "", raw_root: Path = None, out_path: Path = None,
             now: datetime = None) -> tuple:
    end = end or (now or datetime.now(timezone.utc)).strftime(evds.DATE_FMT)
    res = client.data(list(SERIES.values()), start, end)
    items = res["items"]
    if len(items) >= evds.ROW_LIMIT:
        raise MacroError(f"Yanit {evds.ROW_LIMIT} satira ulasti; kesilmis olabilir (araligi daraltin).")
    df = parse_items(items)
    cov = {n: {"observations": int(df[n].notna().sum()),
               "first": df.loc[df[n].notna(), "month"].min() if df[n].notna().any() else None,
               "last": df.loc[df[n].notna(), "month"].max() if df[n].notna().any() else None,
               "min": float(df[n].min()) if df[n].notna().any() else None,
               "max": float(df[n].max()) if df[n].notna().any() else None} for n in SERIES}
    snap = evds.save_snapshot("macro", {"items": items}, {"series": SERIES, "start": start, "end": end, "coverage": cov},
                              Path(raw_root or RAW_ROOT), now)
    out = Path(out_path or OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df, cov, snap


def load(path: Path = None) -> pd.DataFrame:
    p = Path(path or OUT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Makro veri yok: {p} (python -m src.data.macro indir)")
    return pd.read_csv(p, dtype={"month": str})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aylik makro gostergeler (TCMB EVDS).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("indir")
    d.add_argument("--baslangic", default=DEFAULT_START, help="gg-aa-yyyy")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 66 + "\nMAKRO GOSTERGELER (TCMB EVDS, AYLIK)\n" + "=" * 66)
    try:
        client = evds.Client(os.environ.get("EVDS_API_KEY", ""))
        df, cov, snap = download(client, args.baslangic)
        for n, c in cov.items():
            print(f"  {n:12s} {SERIES[n]:18s} {c['observations']} ay, {c['first']} .. {c['last']}, aralik {c['min']} .. {c['max']}")
        print(f"  Yazildi: {OUT_PATH}\n  Ham yanit: {snap}")
        return 0
    except (evds.EVDSError, MacroError) as e:
        print(f"  HATA: {e}")
        return 1
    except Exception as e:  # ag/SSL hatalari
        print(f"  BAGLANTI HATASI: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
