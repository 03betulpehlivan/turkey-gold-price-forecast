"""
Veri Dogrulama Raporu
=====================
Kullanim:  python -m src.data.validate

Kontrol edilen dosyalar:
  data/processed/theoretical_gold.csv
  data/processed/model_dataset.csv (varsa)

Cikis kodlari:
  0 = Tum kontroller basarili
  1 = En az bir hata bulundu
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

# theoretical_gold.csv icin beklenen sutunlar
TG_COLUMNS = ["Date", "gold_usd_oz", "usdtry", "theoretical_gram_try"]

# model_dataset.csv icin ek beklenen sutunlar
MD_EXTRA = ["gold_return", "gold_usd_return", "usdtry_return", "target_next_return"]

TROY_OUNCE_GRAM = 31.1035
FORMULA_TOLERANCE = 0.01  # TL cinsinden kabul edilebilir sapma


def _section(title: str):
    print(f"\n--- {title} ---")


def validate_file(path: Path, required_cols: list) -> tuple:
    """Tek bir CSV dosyasini dogrular.

    Dondurur: (DataFrame veya None, hata sayisi)
    """
    errors = 0

    _section(path.name)

    if not path.exists():
        print(f"  HATA: Dosya bulunamadi: {path}")
        return None, 1

    # Yukleme
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  HATA: Dosya okunamadi: {e}")
        return None, 1

    print(f"  Satir sayisi: {len(df)}")
    if len(df) == 0:
        print("  HATA: Dosya bos.")
        return None, 1

    # Sutun kontrolu
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  HATA: Eksik sutunlar: {missing}")
        errors += 1
    extra = [c for c in df.columns if c not in required_cols]
    if extra:
        print(f"  Ek sutunlar (bilgi): {extra[:10]}")

    # Tarih parse
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception as e:
            print(f"  HATA: Tarih parse edilemedi: {e}")
            errors += 1
            return df, errors

        print(f"  Tarih araligi: {df['Date'].min().date()} - {df['Date'].max().date()}")

        # Siralama
        if not df["Date"].is_monotonic_increasing:
            print("  UYARI: Tarihler kronolojik sirali degil.")
            errors += 1

        # Mukerrer
        dupes = df["Date"].duplicated().sum()
        if dupes > 0:
            print(f"  HATA: {dupes} mukerrer tarih.")
            errors += 1
        else:
            print("  Mukerrer tarih: 0")

    # NaN ve sonsuz degerler
    nan_counts = df.isna().sum()
    nan_total = nan_counts.sum()
    if nan_total > 0:
        print(f"  UYARI: Toplam {nan_total} NaN deger.")
        for col in nan_counts[nan_counts > 0].index:
            print(f"    {col}: {nan_counts[col]} NaN")
    else:
        print("  NaN deger: 0")

    # Sonsuz degerler
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_total = np.isinf(df[numeric_cols]).sum().sum()
    if inf_total > 0:
        print(f"  HATA: {inf_total} sonsuz (inf) deger.")
        errors += 1

    # Fiyat sutunlarinda sifir/negatif
    price_cols = [c for c in ["gold_usd_oz", "usdtry", "theoretical_gram_try"]
                  if c in df.columns]
    for col in price_cols:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            print(f"  HATA: {col} - {bad} sifir veya negatif deger.")
            errors += 1
        else:
            print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f} (OK)")

    return df, errors


def check_formula(df: pd.DataFrame) -> int:
    """theoretical_gram_try = gold_usd_oz * usdtry / 31.1035 tutarliligini kontrol eder."""
    errors = 0
    if not all(c in df.columns for c in ["gold_usd_oz", "usdtry", "theoretical_gram_try"]):
        return 0

    _section("Formul tutarliligi")
    expected = df["gold_usd_oz"] * df["usdtry"] / TROY_OUNCE_GRAM
    diff = (df["theoretical_gram_try"] - expected).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    violations = (diff > FORMULA_TOLERANCE).sum()

    print(f"  Formul: theoretical_gram_try = gold_usd_oz * usdtry / {TROY_OUNCE_GRAM}")
    print(f"  Ortalama sapma: {mean_diff:.6f} TL")
    print(f"  Maksimum sapma: {max_diff:.6f} TL")
    print(f"  Tolerans ({FORMULA_TOLERANCE} TL) asimi: {violations} satir")

    if violations > 0:
        print(f"  UYARI: {violations} satirda formul tutarsiz.")
        errors += 1
    else:
        print("  Formul tutarliligi: OK")

    return errors


def check_target(df: pd.DataFrame) -> int:
    """target_next_return[t] = log(P[t+1] / P[t]) tutarliligini kontrol eder.

    Veri uretim hattinda son satirin hedefi NaN olarak hesaplanir ve
    model_dataset.csv olusturulurken bu satir silinir. Bu nedenle
    dosyadaki tum target_next_return degerleri gecerli olmalidir.

    Dogrulama: Satirlar 0..N-2 icin target[t] = log(P[t+1]/P[t])
    hesaplanabilir cunku P[t+1] veri setinde mevcuttur. Son satirin (N-1)
    hedefi, veri setinden cikarilmis bir sonraki gune ait oldugu icin
    bu yontemle dogrulanamaz.
    """
    errors = 0
    if "target_next_return" not in df.columns or "theoretical_gram_try" not in df.columns:
        return 0

    _section("Hedef degisken kontrolu")

    # Hedef sutununda NaN olmamali (pipeline sirasinda temizlenmis olmali)
    target_nan = df["target_next_return"].isna().sum()
    if target_nan > 0:
        print(f"  UYARI: target_next_return sutununda {target_nan} NaN deger.")
        print("  Pipeline'da son satir dusurulmemis olabilir.")
        errors += 1

    prices = df["theoretical_gram_try"].values
    n = len(prices)

    if n < 2:
        print("  Dogrulama icin yetersiz satir.")
        return errors

    # Satirlar 0..N-2: target[t] ?= log(prices[t+1] / prices[t])
    # Son satir (N-1): hedef, veri setinde bulunmayan P[N]'e dayandigi
    # icin bu yontemle dogrulanamaz.
    expected = np.log(prices[1:] / prices[:-1])          # N-1 deger, satirlar 0..N-2
    recorded = df["target_next_return"].values[:-1]       # N-1 deger, satirlar 0..N-2

    valid = ~np.isnan(recorded) & ~np.isnan(expected)
    n_checked = int(valid.sum())

    if n_checked == 0:
        print("  Karsilastirma icin yeterli gecerli satir yok.")
        return errors

    diff = np.abs(recorded[valid] - expected[valid])
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"  Dogrulanan satir: {n_checked} / {n}  (son satir haric)")
    print(f"  Ortalama sapma: {mean_diff:.10f}")
    print(f"  Maksimum sapma: {max_diff:.10f}")

    if max_diff > 1e-6:
        print("  UYARI: target_next_return fiyatlarla tutarsiz.")
        errors += 1
    else:
        print("  Hedef degisken tutarliligi: OK")

    return errors


def main():
    print("=" * 55)
    print("VERI DOGRULAMA RAPORU")
    print("=" * 55)

    total_errors = 0

    # theoretical_gold.csv
    tg, err = validate_file(DATA_DIR / "theoretical_gold.csv", TG_COLUMNS)
    total_errors += err
    if tg is not None:
        total_errors += check_formula(tg)

    # model_dataset.csv
    md_path = DATA_DIR / "model_dataset.csv"
    if md_path.exists():
        md, err = validate_file(md_path, TG_COLUMNS + MD_EXTRA)
        total_errors += err
        if md is not None:
            total_errors += check_formula(md)
            total_errors += check_target(md)
    else:
        print(f"\n  model_dataset.csv henuz olusturulmamis, atlanacak.")

    # Ozet
    _section("OZET")
    if total_errors == 0:
        print("  Tum kontroller basarili.")
    else:
        print(f"  {total_errors} sorun bulundu.")

    print()
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
