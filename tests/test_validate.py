"""
Veri dogrulama testleri: src/data/validate.py
=============================================
Calistirma (proje kokunden):
  python -m unittest tests.test_validate -v

Kurallar:
  - Yalnizca unittest (+ unittest.mock) ve projede kurulu pandas/numpy.
  - validate.py degistirilmez; testler MEVCUT davranisi olcer.
    "Mevcut davranis" diye isaretli testler bilinen bosluklari belgeler
    (ornegin NaN degerlerin hata sayilmamasi). Davranis bilerek
    degistirilirse bu testler de bilerek guncellenmelidir.
  - Gercek data/ klasoru okunmaz ve yazilmaz: dosyalar gecici klasorde
    uretilir; main() testlerinde DATA_DIR gecici klasore yonlendirilir.
    Modul sonunda gercek data/raw ve data/processed dosyalarinin
    degismedigi md5 ile dogrulanir.
"""

from pathlib import Path
from unittest import mock
import contextlib
import hashlib
import io
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import validate as vd  # noqa: E402


# ---------------------------------------------------------------------------
# Sentetik veri
# ---------------------------------------------------------------------------
DATES = ["2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07",
         "2015-01-08", "2015-01-09", "2015-01-12", "2015-01-13"]
GOLD = [1186.0, 1203.9000244140625, 1219.300048828125, 1210.5,
        1208.0, 1222.25, 1230.0, 1228.75]
FX = [2.330980062484741, 2.34224009513855, 2.330749988555908, 2.3301,
      2.3355, 2.3400, 2.3290, 2.3412]


def make_theoretical() -> pd.DataFrame:
    """Gecerli theoretical_gold verisi (8 satir)."""
    df = pd.DataFrame({"Date": DATES, "gold_usd_oz": GOLD, "usdtry": FX})
    df["theoretical_gram_try"] = df["gold_usd_oz"] * df["usdtry"] / 31.1035
    return df


def make_model() -> pd.DataFrame:
    """Gecerli model_dataset verisi: ilk ve son satiri dusurulmus 6 satir.

    gold_return[t]        = log(P[t] / P[t-1])
    target_next_return[t] = log(P[t+1] / P[t])
    """
    df = make_theoretical()
    p = df["theoretical_gram_try"]
    df["gold_return"] = np.log(p / p.shift(1))
    df["gold_usd_return"] = np.log(df["gold_usd_oz"] / df["gold_usd_oz"].shift(1))
    df["usdtry_return"] = np.log(df["usdtry"] / df["usdtry"].shift(1))
    df["target_next_return"] = df["gold_return"].shift(-1)
    return df.iloc[1:-1].reset_index(drop=True)


MODEL_COLS = vd.TG_COLUMNS + vd.MD_EXTRA


def quiet(func, *args, **kwargs):
    """Fonksiyonu stdout yakalanarak calistirir; (sonuc, cikti) dondurur."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="validate_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, df: pd.DataFrame, name="data.csv") -> Path:
        path = self.tmp / name
        df.to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------
class TestValidateFile(TempDir):

    def test_valid_theoretical_file_has_no_errors(self):
        (df, err), out = quiet(vd.validate_file, self.write(make_theoretical()), vd.TG_COLUMNS)
        self.assertEqual(err, 0)
        self.assertEqual(len(df), len(DATES))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))
        self.assertIn("Mukerrer tarih: 0", out)
        self.assertIn("NaN deger: 0", out)

    def test_valid_model_file_has_no_errors(self):
        (_, err), _ = quiet(vd.validate_file, self.write(make_model()), MODEL_COLS)
        self.assertEqual(err, 0)

    def test_missing_file(self):
        (df, err), out = quiet(vd.validate_file, self.tmp / "yok.csv", vd.TG_COLUMNS)
        self.assertIsNone(df)
        self.assertEqual(err, 1)
        self.assertIn("Dosya bulunamadi", out)

    def test_zero_byte_file_cannot_be_read(self):
        path = self.tmp / "bos.csv"
        path.write_bytes(b"")
        (df, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS)
        self.assertIsNone(df)
        self.assertEqual(err, 1)
        self.assertIn("Dosya okunamadi", out)

    def test_header_only_file_is_empty(self):
        path = self.tmp / "baslik.csv"
        path.write_text(",".join(vd.TG_COLUMNS) + "\n", encoding="utf-8")
        (df, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS)
        self.assertIsNone(df)
        self.assertEqual(err, 1)
        self.assertIn("Dosya bos", out)

    def test_missing_column_counts_one_error(self):
        path = self.write(make_model().drop(columns=["usdtry_return", "gold_usd_return"]))
        (_, err), out = quiet(vd.validate_file, path, MODEL_COLS)
        self.assertEqual(err, 1)  # eksik sutunlar tek hata olarak sayilir
        self.assertIn("Eksik sutunlar", out)
        self.assertIn("usdtry_return", out)

    def test_extra_column_is_information_only(self):
        path = self.write(make_theoretical().assign(note="x"))
        (_, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS)
        self.assertEqual(err, 0)
        self.assertIn("Ek sutunlar (bilgi)", out)

    def test_unparseable_date_returns_early_with_error(self):
        df = make_theoretical()
        df.loc[3, "Date"] = "tarih-degil"
        (out_df, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertIsNotNone(out_df)
        self.assertEqual(err, 1)
        self.assertIn("Tarih parse edilemedi", out)
        self.assertNotIn("NaN deger", out)  # erken donus: sonraki kontroller calismaz

    def test_unsorted_dates_count_as_error(self):
        df = make_theoretical().iloc[[1, 0, 2, 3]]
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 1)
        self.assertIn("kronolojik sirali degil", out)

    def test_adjacent_duplicate_dates_count_as_one_error(self):
        df = make_theoretical()
        df = pd.concat([df.iloc[:3], df.iloc[[2]], df.iloc[3:]], ignore_index=True)
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        # Bitisik tekrar sirayi bozmaz; yalnizca mukerrer hatasi sayilir
        self.assertEqual(err, 1)
        self.assertIn("1 mukerrer tarih", out)

    def test_unsorted_and_duplicate_dates_count_as_two_errors(self):
        df = make_theoretical()
        df = pd.concat([df.iloc[[1, 0]], df.iloc[[0]], df.iloc[2:]], ignore_index=True)
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 2)

    def test_nan_is_reported_but_not_counted_as_error(self):
        # Mevcut davranis: NaN yalnizca UYARI olarak yazilir, hata sayisini artirmaz
        df = make_theoretical()
        df.loc[2, "usdtry"] = np.nan
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 0)
        self.assertIn("Toplam 1 NaN deger", out)
        self.assertIn("usdtry: 1 NaN", out)

    def test_positive_infinity_counts_as_error(self):
        df = make_theoretical()
        df.loc[2, "usdtry"] = np.inf
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 1)
        self.assertIn("1 sonsuz (inf) deger", out)

    def test_negative_infinity_counts_as_infinity_and_negative(self):
        df = make_theoretical()
        df.loc[2, "usdtry"] = -np.inf
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 2)
        self.assertIn("sonsuz (inf)", out)
        self.assertIn("usdtry - 1 sifir veya negatif", out)

    def test_zero_and_negative_prices_count_per_column(self):
        df = make_theoretical()
        df.loc[1, "gold_usd_oz"] = 0.0
        df.loc[4, "gold_usd_oz"] = -1.0
        df.loc[5, "usdtry"] = -2.0
        (_, err), out = quiet(vd.validate_file, self.write(df), vd.TG_COLUMNS)
        self.assertEqual(err, 2)  # sutun basina bir hata
        self.assertIn("gold_usd_oz - 2 sifir veya negatif", out)
        self.assertIn("usdtry - 1 sifir veya negatif", out)


# ---------------------------------------------------------------------------
# check_formula
# ---------------------------------------------------------------------------
class TestCheckFormula(unittest.TestCase):

    def test_correct_formula_passes(self):
        err, out = quiet(vd.check_formula, make_theoretical())
        self.assertEqual(err, 0)
        self.assertIn("Formul tutarliligi: OK", out)

    def test_deviation_within_tolerance_passes(self):
        df = make_theoretical()
        df.loc[3, "theoretical_gram_try"] += 0.005  # tolerans 0.01 TL
        err, _ = quiet(vd.check_formula, df)
        self.assertEqual(err, 0)

    def test_deviation_above_tolerance_fails(self):
        df = make_theoretical()
        df.loc[3, "theoretical_gram_try"] += 0.02
        err, out = quiet(vd.check_formula, df)
        self.assertEqual(err, 1)
        self.assertIn("1 satirda formul tutarsiz", out)

    def test_many_bad_rows_still_count_as_one_error(self):
        df = make_theoretical()
        df["theoretical_gram_try"] = df["theoretical_gram_try"] * 1.01
        err, out = quiet(vd.check_formula, df)
        self.assertEqual(err, 1)
        self.assertIn(f"{len(df)} satirda formul tutarsiz", out)

    def test_wrong_ounce_constant_detected(self):
        df = make_theoretical()
        df["theoretical_gram_try"] = df["gold_usd_oz"] * df["usdtry"] / 31.1
        err, _ = quiet(vd.check_formula, df)
        self.assertEqual(err, 1)

    def test_works_on_model_dataset(self):
        err, _ = quiet(vd.check_formula, make_model())
        self.assertEqual(err, 0)

    def test_missing_columns_skip_silently(self):
        # Mevcut davranis: gerekli sutunlardan biri yoksa kontrol sessizce atlanir
        err, out = quiet(vd.check_formula, make_theoretical().drop(columns="usdtry"))
        self.assertEqual(err, 0)
        self.assertEqual(out, "")

    def test_nan_price_is_not_flagged(self):
        # Mevcut davranis: NaN fark tolerans karsilastirmasinda ihlal sayilmaz
        df = make_theoretical()
        df.loc[2, "theoretical_gram_try"] = np.nan
        err, _ = quiet(vd.check_formula, df)
        self.assertEqual(err, 0)


# ---------------------------------------------------------------------------
# check_target
# ---------------------------------------------------------------------------
class TestCheckTarget(unittest.TestCase):

    def test_correct_target_passes(self):
        df = make_model()
        err, out = quiet(vd.check_target, df)
        self.assertEqual(err, 0)
        self.assertIn(f"Dogrulanan satir: {len(df) - 1} / {len(df)}", out)
        self.assertIn("Hedef degisken tutarliligi: OK", out)

    def test_target_equal_to_same_day_return_fails(self):
        df = make_model()
        df["target_next_return"] = df["gold_return"]  # bir gun geride: sizinti hatasi
        err, out = quiet(vd.check_target, df)
        self.assertEqual(err, 1)
        self.assertIn("fiyatlarla tutarsiz", out)

    def test_target_shifted_two_rows_fails(self):
        df = make_model()
        df["target_next_return"] = df["target_next_return"].shift(-1).fillna(0.0)
        err, _ = quiet(vd.check_target, df)
        self.assertEqual(err, 1)

    def test_deviation_above_tolerance_fails(self):
        df = make_model()
        df.loc[2, "target_next_return"] += 1e-5  # tolerans 1e-6
        err, _ = quiet(vd.check_target, df)
        self.assertEqual(err, 1)

    def test_deviation_below_tolerance_passes(self):
        df = make_model()
        df.loc[2, "target_next_return"] += 1e-7
        err, _ = quiet(vd.check_target, df)
        self.assertEqual(err, 0)

    def test_last_row_is_not_checked(self):
        # Mevcut davranis: son satirin hedefi P[N]'e dayandigi icin dogrulanmaz
        df = make_model()
        df.loc[len(df) - 1, "target_next_return"] = 0.5
        err, _ = quiet(vd.check_target, df)
        self.assertEqual(err, 0)

    def test_nan_target_in_last_row_counts_as_error(self):
        df = make_model()
        df.loc[len(df) - 1, "target_next_return"] = np.nan
        err, out = quiet(vd.check_target, df)
        self.assertEqual(err, 1)
        self.assertIn("1 NaN deger", out)

    def test_nan_target_in_middle_counts_once(self):
        # NaN bir hata sayilir; o satir karsilastirmadan cikarilir
        df = make_model()
        df.loc[2, "target_next_return"] = np.nan
        err, out = quiet(vd.check_target, df)
        self.assertEqual(err, 1)
        self.assertIn(f"Dogrulanan satir: {len(df) - 2} / {len(df)}", out)

    def test_all_nan_target(self):
        df = make_model()
        df["target_next_return"] = np.nan
        err, out = quiet(vd.check_target, df)
        self.assertEqual(err, 1)
        self.assertIn("yeterli gecerli satir yok", out)

    def test_single_row_is_not_enough(self):
        err, out = quiet(vd.check_target, make_model().iloc[:1])
        self.assertEqual(err, 0)
        self.assertIn("yetersiz satir", out)

    def test_missing_columns_skip_silently(self):
        for col in ("target_next_return", "theoretical_gram_try"):
            with self.subTest(col=col):
                err, out = quiet(vd.check_target, make_model().drop(columns=col))
                self.assertEqual(err, 0)
                self.assertEqual(out, "")

    def test_works_on_file_read_without_date_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.csv"
            make_model().to_csv(path, index=False)
            err, _ = quiet(vd.check_target, pd.read_csv(path))
        self.assertEqual(err, 0)


# ---------------------------------------------------------------------------
# main (DATA_DIR gecici klasore yonlendirilir)
# ---------------------------------------------------------------------------
class TestMain(TempDir):

    def run_main_with_output(self):
        buf = io.StringIO()
        with mock.patch.object(vd, "DATA_DIR", self.tmp), contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                vd.main()
        return ctx.exception.code, buf.getvalue()

    def test_valid_files_exit_0(self):
        self.write(make_theoretical(), "theoretical_gold.csv")
        self.write(make_model(), "model_dataset.csv")
        code, out = self.run_main_with_output()
        self.assertEqual(code, 0)
        self.assertIn("Tum kontroller basarili", out)

    def test_missing_model_dataset_is_skipped(self):
        self.write(make_theoretical(), "theoretical_gold.csv")
        code, out = self.run_main_with_output()
        self.assertEqual(code, 0)
        self.assertIn("model_dataset.csv henuz olusturulmamis", out)

    def test_missing_theoretical_file_exits_1(self):
        code, out = self.run_main_with_output()
        self.assertEqual(code, 1)
        self.assertIn("Dosya bulunamadi", out)

    def test_bad_target_exits_1(self):
        self.write(make_theoretical(), "theoretical_gold.csv")
        md = make_model()
        md["target_next_return"] = md["gold_return"]
        self.write(md, "model_dataset.csv")
        code, out = self.run_main_with_output()
        self.assertEqual(code, 1)
        self.assertIn("1 sorun bulundu", out)

    def test_main_does_not_modify_input_files(self):
        self.write(make_theoretical(), "theoretical_gold.csv")
        self.write(make_model(), "model_dataset.csv")
        before = file_hashes(self.tmp)
        self.run_main_with_output()
        self.assertEqual(file_hashes(self.tmp), before)


# ---------------------------------------------------------------------------
# Gercek proje verisinin korunmasi
# ---------------------------------------------------------------------------
_REAL_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed"]
_REAL_BEFORE = {}


class TestRealDataUntouched(unittest.TestCase):

    def test_validate_points_to_project_data_dir(self):
        # Testler bu klasoru kullanmaz; yalnizca yolun beklenen yer oldugu dogrulanir
        self.assertEqual(vd.DATA_DIR, PROJECT_ROOT / "data" / "processed")


def setUpModule():
    for d in _REAL_DIRS:
        _REAL_BEFORE[d] = file_hashes(d)


def tearDownModule():
    for d in _REAL_DIRS:
        if file_hashes(d) != _REAL_BEFORE[d]:
            raise AssertionError(f"Testler gercek veri klasorunu degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

