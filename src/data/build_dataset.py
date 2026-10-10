"""
Veri Seti Derleme
=================
Ham yfinance CSV dosyalarindan islenmis veri setlerini uretir.
Internet baglantisi gerektirmez; veri indirmez.

Kullanim (proje kokunden):
  python -m src.data.build_dataset                  # yalnizca kontrol, dosya yazmaz
  python -m src.data.build_dataset --write          # referans yoksa yazar; uyumluysa dokunmaz
  python -m src.data.build_dataset --write --force  # fark olsa bile yazar (once yedek alir)

Girdi (yalnizca okunur, asla degistirilmez):
  data/raw/gold_futures.csv   yfinance GC=F ciktisi
  data/raw/usdtry.csv         yfinance USDTRY=X ciktisi

Cikti:
  data/processed/theoretical_gold.csv
  data/processed/model_dataset.csv

Tanimlar (degistirilmez):
  theoretical_gram_try   = gold_usd_oz * usdtry / 31.1035
  gold_return[t]         = log(P[t] / P[t-1])        P = theoretical_gram_try
  gold_usd_return[t]     = log(gold_usd_oz[t] / gold_usd_oz[t-1])
  usdtry_return[t]       = log(usdtry[t] / usdtry[t-1])
  target_next_return[t]  = log(P[t+1] / P[t]) = gold_return[t+1]
  t+1 = bir sonraki satir (bir sonraki ortak islem gunu), takvim gunu degil.
  model_dataset.csv'de ilk satir (getiri yok) ve son satir (hedef yok) bulunmaz.

Not: Bu fiyat teoriktir; fiziksel (kuyumcu) gram altin fiyati degildir.

Cikis kodlari:
  0 = basarili
  1 = girdi veya dogrulama hatasi (hicbir dosya yazilmaz)
  2 = mevcut dosyalarla fark var (yazilmadi)
"""

from pathlib import Path
from datetime import datetime
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"

TROY_OUNCE_GRAM = 31.1035
GOLD_FILE, GOLD_TICKER = "gold_futures.csv", "GC=F"
FX_FILE, FX_TICKER = "usdtry.csv", "USDTRY=X"
THEO_FILE, MODEL_FILE = "theoretical_gold.csv", "model_dataset.csv"

THEO_COLUMNS = ["Date", "gold_usd_oz", "usdtry", "theoretical_gram_try"]
MODEL_COLUMNS = THEO_COLUMNS + [
    "gold_return", "gold_usd_return", "usdtry_return", "target_next_return",
]

DATE_FORMAT = "%Y-%m-%d"
COMPARE_TOL = 1e-9        # mevcut dosyalarla karsilastirma (mutlak fark)
ALIGN_TOL = 1e-12         # hedef hizalama kontrolu
LARGE_MOVE = 0.15         # |log getiri| bundan buyukse uyari (hata degil)
MAX_LIST = 10             # raporda listelenecek en fazla tarih


class DataError(Exception):
    """Girdi veya dogrulama hatasi; hicbir dosya yazilmaz."""


def _section(title: str):
    print(f"\n--- {title} ---")


def _fmt_dates(dates) -> str:
    dates = [pd.Timestamp(d).strftime(DATE_FORMAT) for d in dates]
    if len(dates) <= MAX_LIST:
        return ", ".join(dates)
    return ", ".join(dates[:MAX_LIST]) + f" ... (+{len(dates) - MAX_LIST})"


# ---------------------------------------------------------------------------
# Ham dosya okuma
# ---------------------------------------------------------------------------
def read_yahoo_csv(path: Path, expected_ticker: str) -> pd.DataFrame:
    """yfinance CSV dosyasini okur ve [Date, close] dondurur.

    Desteklenen bicimler (ilk satirlara bakilarak belirlenir, varsayilmaz):
      A) Uc satirli baslik (yfinance cok seviyeli sutun ciktisi):
           Price,Adj Close,Close,High,Low,Open,Volume
           Ticker,GC=F,GC=F,...
           Date,,,,,,
         Ticker satiri beklenen sembolle eslesmek zorundadir.
      B) Tek satirli baslik:  Date,Open,High,Low,Close,...
         Sembol dosyadan dogrulanamaz; uyari verilir.

    Kapanis olarak 'Close' sutunu kullanilir. Bos kapanisli satirlar
    cikarilir ve raporlanir; sifir, negatif veya sonsuz deger hatadir.
    """
    if not path.exists():
        raise DataError(f"Dosya bulunamadi: {path}")

    with open(path, encoding="utf-8-sig") as fh:
        head = [fh.readline().rstrip("\r\n") for _ in range(3)]
    cells = [line.split(",") for line in head]

    if (cells[0][0] == "Price" and cells[1][0] == "Ticker"
            and cells[2][0] == "Date" and all(c == "" for c in cells[2][1:])):
        fields = cells[0][1:]
        tickers = cells[1][1:]
        if len(fields) != len(tickers):
            raise DataError(f"{path.name}: Price ve Ticker satirlarinin sutun sayisi farkli.")
        wrong = sorted({t for t in tickers if t != expected_ticker})
        if wrong:
            raise DataError(
                f"{path.name}: beklenen sembol {expected_ticker}, dosyada {wrong} var."
            )
        df = pd.read_csv(path, skiprows=3, header=None, names=["Date"] + fields,
                         encoding="utf-8-sig", dtype=str, keep_default_na=False)
        print(f"  Bicim: uc satirli yfinance basligi, sembol {expected_ticker} dogrulandi")
    elif cells[0][0] == "Date":
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        print(f"  Bicim: tek satirli baslik; UYARI: sembol dosyadan dogrulanamadi")
    else:
        raise DataError(f"{path.name}: taninmayan baslik bicimi. Ilk satir: {head[0]!r}")

    if "Close" not in df.columns:
        raise DataError(f"{path.name}: 'Close' sutunu yok. Sutunlar: {list(df.columns)}")
    if len(df) == 0:
        raise DataError(f"{path.name}: veri satiri yok.")

    # Tarih: yalnizca YYYY-MM-DD kabul edilir
    raw_dates = df["Date"].str.strip()
    dates = pd.to_datetime(raw_dates, format=DATE_FORMAT, errors="coerce")
    bad = raw_dates[dates.isna()]
    if len(bad):
        raise DataError(
            f"{path.name}: {len(bad)} tarih YYYY-MM-DD biciminde degil "
            f"(ornek: {list(bad[:3])})."
        )

    # Kapanis: bos hucre = eksik gozlem; bos olmayan ama sayi olmayan = hata
    raw_close = df["Close"].str.strip()
    close = pd.to_numeric(raw_close.where(raw_close != ""), errors="coerce")
    unparsable = raw_close[(raw_close != "") & close.isna()]
    if len(unparsable):
        raise DataError(
            f"{path.name}: {len(unparsable)} kapanis degeri sayi degil "
            f"(ornek: {list(unparsable[:3])})."
        )

    out = pd.DataFrame({"Date": dates, "close": close.astype(float)})

    dupes = out["Date"][out["Date"].duplicated(keep=False)].unique()
    if len(dupes):
        raise DataError(f"{path.name}: tekrar eden tarihler: {_fmt_dates(dupes)}")

    if not out["Date"].is_monotonic_increasing:
        print("  Bilgi: tarihler sirali degildi; tarihe gore siralandi.")
        out = out.sort_values("Date").reset_index(drop=True)

    missing = out["Date"][out["close"].isna()]
    if len(missing):
        print(f"  Bilgi: {len(missing)} satirda kapanis bos, cikarildi: {_fmt_dates(missing)}")
        out = out.dropna(subset=["close"]).reset_index(drop=True)

    invalid = out[~np.isfinite(out["close"]) | (out["close"] <= 0)]
    if len(invalid):
        raise DataError(
            f"{path.name}: {len(invalid)} kapanis sifir, negatif veya sonsuz: "
            f"{_fmt_dates(invalid['Date'])}"
        )

    # Bilgi amacli: Adj Close ile Close ayni mi?
    if "Adj Close" in df.columns:
        adj = pd.to_numeric(df["Adj Close"].str.strip().replace("", np.nan), errors="coerce")
        adj = pd.Series(adj.values, index=dates).reindex(out["Date"]).values
        n_diff = int(np.sum(np.abs(adj - out["close"].values) > ALIGN_TOL))
        print(f"  Adj Close ile Close farkli satir: {n_diff}")

    print(f"  Satir: {len(out)} | {out['Date'].iloc[0].date()} - {out['Date'].iloc[-1].date()}")
    return out


# ---------------------------------------------------------------------------
# Derleme
# ---------------------------------------------------------------------------
def build(gold: pd.DataFrame, fx: pd.DataFrame) -> tuple:
    """Iki seriyi birlestirir, teorik fiyati, getirileri ve hedefi hesaplar."""
    g = gold.rename(columns={"close": "gold_usd_oz"})
    f = fx.rename(columns={"close": "usdtry"})

    _section("Birlestirme (ortak tarihler)")
    only_gold = sorted(set(g["Date"]) - set(f["Date"]))
    only_fx = sorted(set(f["Date"]) - set(g["Date"]))
    print(f"  Yalnizca {GOLD_TICKER} tarihinde olup dusen: {len(only_gold)}"
          + (f"  [{_fmt_dates(only_gold)}]" if only_gold else ""))
    print(f"  Yalnizca {FX_TICKER} tarihinde olup dusen: {len(only_fx)}"
          + (f"  [{_fmt_dates(only_fx)}]" if only_fx else ""))

    merged = g.merge(f, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(merged) < 3:
        raise DataError(f"Ortak tarih sayisi yetersiz ({len(merged)}).")
    print(f"  Ortak satir: {len(merged)}")

    theo = merged[["Date", "gold_usd_oz", "usdtry"]].copy()
    # Islem sirasi mevcut dosyalarla bit duzeyinde ayni: (ons * kur) / 31.1035
    theo["theoretical_gram_try"] = theo["gold_usd_oz"] * theo["usdtry"] / TROY_OUNCE_GRAM

    model = theo.copy()
    p = model["theoretical_gram_try"]
    model["gold_return"] = np.log(p / p.shift(1))
    model["gold_usd_return"] = np.log(model["gold_usd_oz"] / model["gold_usd_oz"].shift(1))
    model["usdtry_return"] = np.log(model["usdtry"] / model["usdtry"].shift(1))
    model["target_next_return"] = model["gold_return"].shift(-1)
    # Yalnizca ilk (getiri yok) ve son (hedef yok) satir dusurulur
    model = model.iloc[1:-1].reset_index(drop=True)

    return theo[THEO_COLUMNS], model[MODEL_COLUMNS]


# ---------------------------------------------------------------------------
# Cikti dogrulama
# ---------------------------------------------------------------------------
def validate_outputs(theo: pd.DataFrame, model: pd.DataFrame):
    """Uretilen veri setlerinin ic tutarliligini kontrol eder."""
    _section("Cikti dogrulama")
    for name, df in (("theoretical_gold", theo), ("model_dataset", model)):
        num = df.drop(columns="Date").to_numpy(dtype=float)
        if not np.isfinite(num).all():
            raise DataError(f"{name}: bos veya sonsuz deger var.")
        if not df["Date"].is_monotonic_increasing or df["Date"].duplicated().any():
            raise DataError(f"{name}: tarihler kesin artan sirada degil.")

    if len(model) != len(theo) - 2:
        raise DataError(f"model_dataset satir sayisi {len(model)}, beklenen {len(theo) - 2}.")
    if not model["Date"].reset_index(drop=True).equals(theo["Date"].iloc[1:-1].reset_index(drop=True)):
        raise DataError("model_dataset tarihleri theoretical_gold'un ic satirlariyla eslesmiyor.")

    # Hedef hizalama: target[t] = log(P[t+1]/P[t]), P[t+1] theoretical_gold'dan
    p = theo["theoretical_gram_try"].to_numpy()
    expected_target = np.log(p[2:] / p[1:-1])
    max_dev = float(np.max(np.abs(model["target_next_return"].to_numpy() - expected_target)))
    if max_dev > ALIGN_TOL:
        raise DataError(f"target_next_return hizalamasi hatali (max sapma {max_dev:.3e}).")
    print(f"  [OK] target[t] = log(P[t+1]/P[t]), {len(model)} satir (max sapma {max_dev:.1e})")

    # target[t] == gold_return[t+1] (model_dataset icinde)
    shift_dev = float(np.max(np.abs(
        model["target_next_return"].to_numpy()[:-1] - model["gold_return"].to_numpy()[1:]
    )))
    if shift_dev > ALIGN_TOL:
        raise DataError(f"target ile sonraki gold_return eslesmiyor (max sapma {shift_dev:.3e}).")
    print("  [OK] target[t] = gold_return[t+1]")

    big = model.loc[model["gold_return"].abs() > LARGE_MOVE, ["Date", "gold_return"]]
    if len(big):
        print(f"  Uyari (hata degil): |gold_return| > {LARGE_MOVE} olan {len(big)} gun:")
        for _, r in big.head(MAX_LIST).iterrows():
            print(f"    {r['Date'].date()}  {r['gold_return']:+.4f}")
    else:
        print(f"  |gold_return| > {LARGE_MOVE} olan gun yok")


# ---------------------------------------------------------------------------
# Mevcut dosyalarla karsilastirma
# ---------------------------------------------------------------------------
def compare_with_reference(new: pd.DataFrame, path: Path) -> dict:
    """Uretilen veriyi mevcut dosyayla karsilastirir; hicbir seyi duzeltmez.

    Dondurur: {"status": "missing" | "match" | "differ", "lines": [rapor satirlari]}
    """
    if not path.exists():
        return {"status": "missing", "lines": ["  Mevcut dosya yok."]}

    ref = pd.read_csv(path, encoding="utf-8-sig")
    lines, differ = [], False

    if list(ref.columns) != list(new.columns):
        differ = True
        lines.append(f"  FARK sutunlar: mevcut {list(ref.columns)} | yeni {list(new.columns)}")

    ref_dates = pd.to_datetime(ref["Date"], format=DATE_FORMAT, errors="coerce") \
        if "Date" in ref.columns else pd.Series(dtype="datetime64[ns]")
    if ref_dates.isna().any():
        differ = True
        lines.append(f"  FARK: mevcut dosyada {int(ref_dates.isna().sum())} okunamayan tarih")

    lines.append(f"  Satir: mevcut {len(ref)} | yeni {len(new)}")
    only_ref = sorted(set(ref_dates.dropna()) - set(new["Date"]))
    only_new = sorted(set(new["Date"]) - set(ref_dates.dropna()))
    if only_ref:
        differ = True
        lines.append(f"  FARK yalnizca mevcut dosyada {len(only_ref)} tarih: {_fmt_dates(only_ref)}")
    if only_new:
        differ = True
        lines.append(f"  FARK yalnizca yeni veride {len(only_new)} tarih: {_fmt_dates(only_new)}")

    common_cols = [c for c in new.columns if c != "Date" and c in ref.columns]
    ref_idx = ref.assign(Date=ref_dates).dropna(subset=["Date"]).set_index("Date")
    new_idx = new.set_index("Date")
    common_dates = new_idx.index.intersection(ref_idx.index)
    for col in common_cols:
        a = pd.to_numeric(ref_idx.loc[common_dates, col], errors="coerce").to_numpy(dtype=float)
        b = new_idx.loc[common_dates, col].to_numpy(dtype=float)
        diff = np.abs(a - b)
        n_nan = int(np.isnan(diff).sum())
        max_diff = float(np.nanmax(diff)) if len(diff) and n_nan < len(diff) else 0.0
        n_over = int(np.sum(diff > COMPARE_TOL)) + n_nan
        status = "OK" if n_over == 0 else "FARK"
        if n_over:
            differ = True
            worst = common_dates[int(np.nanargmax(diff))].date() if n_nan < len(diff) else "-"
            lines.append(f"  {status} {col}: {n_over} satir > {COMPARE_TOL:g}, "
                         f"max fark {max_diff:.3e} (en buyuk: {worst})")
        else:
            lines.append(f"  {status} {col}: max fark {max_diff:.3e}")

    return {"status": "differ" if differ else "match", "lines": lines}


# ---------------------------------------------------------------------------
# Yazma
# ---------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path):
    """Once yedek alir (dosya varsa), sonra gecici dosya uzerinden atomik yazar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"  Yedek: {backup}")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, date_format=DATE_FORMAT)
    os.replace(tmp, path)
    print(f"  Yazildi: {path} ({len(df)} satir)")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ham yfinance CSV'lerinden islenmis veri setlerini uretir.")
    ap.add_argument("--write", action="store_true",
                    help="Ciktilari yaz (mevcut dosya yoksa; uyumluysa dokunma).")
    ap.add_argument("--force", action="store_true",
                    help="--write ile birlikte: fark olsa bile yaz (once yedek alir).")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="raw/ ve processed/ klasorlerini iceren dizin (varsayilan: proje/data).")
    args = ap.parse_args(argv)
    if args.force and not args.write:
        ap.error("--force yalnizca --write ile birlikte kullanilabilir.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_dir = args.data_dir / "raw"
    out_dir = args.data_dir / "processed"

    print("=" * 60)
    print("VERI SETI DERLEME")
    print("=" * 60)
    mode = "yaz (zorla)" if args.force else ("yaz" if args.write else "yalnizca kontrol")
    print(f"Mod: {mode}")

    try:
        _section(f"{GOLD_FILE} ({GOLD_TICKER})")
        gold = read_yahoo_csv(raw_dir / GOLD_FILE, GOLD_TICKER)
        _section(f"{FX_FILE} ({FX_TICKER})")
        fx = read_yahoo_csv(raw_dir / FX_FILE, FX_TICKER)
        theo, model = build(gold, fx)
        validate_outputs(theo, model)
    except DataError as e:
        print(f"\n  HATA: {e}")
        print("  Hicbir dosya yazilmadi.")
        return 1

    outputs = [(THEO_FILE, theo), (MODEL_FILE, model)]
    results = {}
    for name, df in outputs:
        _section(f"Karsilastirma: {name}")
        res = compare_with_reference(df, out_dir / name)
        results[name] = res
        print("\n".join(res["lines"]))
        print(f"  Sonuc: {res['status'].upper()}")

    any_differ = any(r["status"] == "differ" for r in results.values())

    _section("Ozet")
    if not args.write:
        if any_differ:
            print("  Mevcut dosyalarla fark var. Hicbir dosya yazilmadi.")
            print("  Farki inceleyin; bilerek degistirmek icin: --write --force")
            return 2
        print("  Uretilen veri mevcut dosyalarla uyumlu (veya dosya yok). Hicbir dosya yazilmadi.")
        return 0

    if any_differ and not args.force:
        print("  Mevcut dosyalarla fark var; --force olmadan yazilmaz. Hicbir dosya yazilmadi.")
        return 2

    for name, df in outputs:
        status = results[name]["status"]
        if status == "match" and not args.force:
            print(f"  {name}: mevcut dosya zaten uyumlu, dokunulmadi.")
            continue
        write_csv(df, out_dir / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

