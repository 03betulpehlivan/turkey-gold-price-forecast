"""
Veri pipeline testleri: src/data/build_dataset.py (+ validate.py uyumu)
=====================================================================
Calistirma (proje kokunden):
  python -m unittest discover -s tests -v

Kurallar:
  - Yalnizca standart kutuphane (unittest) + projede zaten kurulu pandas/numpy.
  - Gercek data/ klasorune hicbir zaman yazilmaz: her test gecici klasor ve
    --data-dir kullanir. Modul sonunda gercek data/raw ve data/processed
    dosyalarinin degismedigi ayrica dogrulanir.
  - Uretim kodu degistirilmez; testler mevcut davranisi olcer.
"""

from pathlib import Path
import contextlib
import hashlib
import io
import math
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataset as bd  # noqa: E402
from src.data import validate as vd  # noqa: E402


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def yahoo_csv(path: Path, ticker: str, rows, header="multi"):
    """yfinance bicimli CSV yazar. rows = [(YYYY-MM-DD, close), ...].

    close bir sayi ise Adj Close = Close = High = Low = Open = close yazilir;
    bir metin ise (ornegin "" veya "abc") tum fiyat hucrelerine aynen yazilir.
    """
    lines = []
    if header == "multi":
        lines += ["Price,Adj Close,Close,High,Low,Open,Volume",
                  "Ticker," + ",".join([ticker] * 6),
                  "Date,,,,,,"]
    elif header == "single":
        lines += ["Date,Adj Close,Close,High,Low,Open,Volume"]
    for d, c in rows:
        v = repr(float(c)) if isinstance(c, (int, float)) else c
        lines.append(f"{d},{v},{v},{v},{v},{v},0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


GOLD_ROWS = [  # 2015-01-08 yalnizca altinda, 2015-01-01 ve 2015-01-12 yalnizca kurda
    ("2015-01-02", 1186.0), ("2015-01-05", 1203.9000244140625),
    ("2015-01-06", 1219.300048828125), ("2015-01-07", 1210.5),
    ("2015-01-08", 1208.0), ("2015-01-09", 1222.25), ("2015-01-13", 1230.0),
    ("2015-01-14", 1228.75),
]
FX_ROWS = [
    ("2015-01-01", 2.331160068511963), ("2015-01-02", 2.330980062484741),
    ("2015-01-05", 2.34224009513855), ("2015-01-06", 2.330749988555908),
    ("2015-01-07", 2.3301), ("2015-01-09", 2.3355), ("2015-01-12", 2.3400),
    ("2015-01-13", 2.3290), ("2015-01-14", 2.3412),
]
COMMON = ["2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07",
          "2015-01-09", "2015-01-13", "2015-01-14"]


def quiet(func, *args, **kwargs):
    """Fonksiyonu stdout'u yakalayarak calistirir; (sonuc, cikti) dondurur."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def file_hashes(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


class TempDataDir(unittest.TestCase):
    """Her test icin data/raw ve data/processed iceren gecici klasor."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gold_test_"))
        self.raw = self.root / "raw"
        self.out = self.root / "processed"
        self.raw.mkdir()
        self.out.mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write_default_raw(self):
        yahoo_csv(self.raw / bd.GOLD_FILE, bd.GOLD_TICKER, GOLD_ROWS)
        yahoo_csv(self.raw / bd.FX_FILE, bd.FX_TICKER, FX_ROWS)

    def run_main(self, *extra):
        return quiet(bd.main, ["--data-dir", str(self.root), *extra])

    def build_from_raw(self):
        (gold, _), (fx, _) = (quiet(bd.read_yahoo_csv, self.raw / bd.GOLD_FILE, bd.GOLD_TICKER),
                              quiet(bd.read_yahoo_csv, self.raw / bd.FX_FILE, bd.FX_TICKER))
        (theo, model), _ = quiet(bd.build, gold, fx)
        return theo, model


# ---------------------------------------------------------------------------
# 1) Yahoo Finance CSV okuma
# ---------------------------------------------------------------------------
class TestReadYahooCsv(TempDataDir):

    def test_three_line_header_parsed(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", GOLD_ROWS)
        df, out = quiet(bd.read_yahoo_csv, path, "GC=F")
        self.assertEqual(list(df.columns), ["Date", "close"])
        self.assertEqual(len(df), len(GOLD_ROWS))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))
        self.assertEqual(df["Date"].iloc[0], pd.Timestamp("2015-01-02"))
        self.assertEqual(df["close"].iloc[1], 1203.9000244140625)
        self.assertIn("uc satirli yfinance basligi", out)

    def test_wrong_ticker_rejected(self):
        path = self.raw / "fx.csv"
        yahoo_csv(path, "EURTRY=X", FX_ROWS)
        with self.assertRaisesRegex(bd.DataError, "beklenen sembol USDTRY=X"):
            quiet(bd.read_yahoo_csv, path, "USDTRY=X")

    def test_single_line_header_accepted_with_warning(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", GOLD_ROWS, header="single")
        df, out = quiet(bd.read_yahoo_csv, path, "GC=F")
        self.assertEqual(len(df), len(GOLD_ROWS))
        self.assertIn("sembol dosyadan dogrulanamadi", out)

    def test_unknown_header_rejected(self):
        path = self.raw / "g.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(bd.DataError, "taninmayan baslik"):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_missing_file_rejected(self):
        with self.assertRaisesRegex(bd.DataError, "Dosya bulunamadi"):
            quiet(bd.read_yahoo_csv, self.raw / "yok.csv", "GC=F")

    def test_missing_close_column_rejected(self):
        path = self.raw / "g.csv"
        path.write_text("Price,Open,High\nTicker,GC=F,GC=F\nDate,,\n"
                        "2015-01-02,1.0,2.0\n", encoding="utf-8")
        with self.assertRaisesRegex(bd.DataError, "'Close' sutunu yok"):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_header_only_file_rejected(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", [])
        with self.assertRaises(bd.DataError):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_unsorted_dates_are_sorted(self):
        path = self.raw / "g.csv"
        rows = [GOLD_ROWS[2], GOLD_ROWS[0], GOLD_ROWS[1]]
        yahoo_csv(path, "GC=F", rows)
        df, out = quiet(bd.read_yahoo_csv, path, "GC=F")
        self.assertTrue(df["Date"].is_monotonic_increasing)
        self.assertEqual(df["close"].tolist(), [1186.0, 1203.9000244140625, 1219.300048828125])
        self.assertIn("sirali degildi", out)

    def test_duplicate_dates_rejected(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", GOLD_ROWS[:3] + [GOLD_ROWS[1]])
        with self.assertRaisesRegex(bd.DataError, "tekrar eden tarihler"):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_invalid_date_format_rejected(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", [("02/01/2015", 1186.0)] + GOLD_ROWS[1:3])
        with self.assertRaisesRegex(bd.DataError, "YYYY-MM-DD"):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_zero_negative_and_inf_prices_rejected(self):
        for bad in (0.0, -5.0, "inf"):
            with self.subTest(bad=bad):
                path = self.raw / "g.csv"
                yahoo_csv(path, "GC=F", GOLD_ROWS[:2] + [("2015-01-06", bad)])
                with self.assertRaisesRegex(bd.DataError, "sifir, negatif veya sonsuz"):
                    quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_non_numeric_price_rejected(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", GOLD_ROWS[:2] + [("2015-01-06", "abc")])
        with self.assertRaisesRegex(bd.DataError, "sayi degil"):
            quiet(bd.read_yahoo_csv, path, "GC=F")

    def test_empty_price_row_dropped_and_reported(self):
        path = self.raw / "g.csv"
        yahoo_csv(path, "GC=F", GOLD_ROWS[:2] + [("2015-01-06", "")] + GOLD_ROWS[3:4])
        df, out = quiet(bd.read_yahoo_csv, path, "GC=F")
        self.assertEqual(len(df), 3)
        self.assertNotIn(pd.Timestamp("2015-01-06"), set(df["Date"]))
        self.assertIn("kapanis bos, cikarildi", out)


# ---------------------------------------------------------------------------
# 2) Birlestirme, formul ve hedef
# ---------------------------------------------------------------------------
class TestBuild(TempDataDir):

    def setUp(self):
        super().setUp()
        self.write_default_raw()
        self.theo, self.model = self.build_from_raw()

    def test_inner_join_keeps_only_common_dates(self):
        got = self.theo["Date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(got, COMMON)

    def test_dropped_dates_reported_per_source(self):
        (gold, _), (fx, _) = (quiet(bd.read_yahoo_csv, self.raw / bd.GOLD_FILE, "GC=F"),
                              quiet(bd.read_yahoo_csv, self.raw / bd.FX_FILE, "USDTRY=X"))
        _, out = quiet(bd.build, gold, fx)
        self.assertIn("Yalnizca GC=F tarihinde olup dusen: 1  [2015-01-08]", out)
        self.assertIn("Yalnizca USDTRY=X tarihinde olup dusen: 2  [2015-01-01, 2015-01-12]", out)

    def test_values_come_from_the_same_date(self):
        row = self.theo.set_index(self.theo["Date"].dt.strftime("%Y-%m-%d")).loc["2015-01-09"]
        self.assertEqual(row["gold_usd_oz"], 1222.25)
        self.assertEqual(row["usdtry"], 2.3355)

    def test_theoretical_formula(self):
        expected = self.theo["gold_usd_oz"] * self.theo["usdtry"] / 31.1035
        np.testing.assert_array_equal(self.theo["theoretical_gram_try"].to_numpy(),
                                      expected.to_numpy())

    def test_formula_matches_known_project_values(self):
        # Kullanicinin gercek theoretical_gold.csv dosyasindan alinan iki satir
        p = self.theo.set_index(self.theo["Date"].dt.strftime("%Y-%m-%d"))["theoretical_gram_try"]
        self.assertEqual(p["2015-01-02"], 88.88203430825801)
        self.assertEqual(p["2015-01-05"], 90.65934405198438)
        self.assertEqual(p["2015-01-06"], 91.36861044102338)

    def test_columns_and_row_counts(self):
        self.assertEqual(list(self.theo.columns), bd.THEO_COLUMNS)
        self.assertEqual(list(self.model.columns), bd.MODEL_COLUMNS)
        self.assertEqual(len(self.model), len(self.theo) - 2)
        self.assertEqual(self.model["Date"].iloc[0], self.theo["Date"].iloc[1])
        self.assertEqual(self.model["Date"].iloc[-1], self.theo["Date"].iloc[-2])

    def test_returns_are_log_ratios_of_previous_row(self):
        theo = self.theo.reset_index(drop=True)
        for i, row in self.model.iterrows():
            t = i + 1  # model satiri i, theo satiri i+1'e karsilik gelir
            with self.subTest(date=str(row["Date"].date())):
                self.assertAlmostEqual(row["gold_return"], math.log(
                    theo.loc[t, "theoretical_gram_try"] / theo.loc[t - 1, "theoretical_gram_try"]), places=15)
                self.assertAlmostEqual(row["gold_usd_return"], math.log(
                    theo.loc[t, "gold_usd_oz"] / theo.loc[t - 1, "gold_usd_oz"]), places=15)
                self.assertAlmostEqual(row["usdtry_return"], math.log(
                    theo.loc[t, "usdtry"] / theo.loc[t - 1, "usdtry"]), places=15)

    def test_target_is_return_to_next_observation(self):
        theo = self.theo.reset_index(drop=True)
        for i, row in self.model.iterrows():
            t = i + 1
            with self.subTest(date=str(row["Date"].date())):
                self.assertAlmostEqual(row["target_next_return"], math.log(
                    theo.loc[t + 1, "theoretical_gram_try"] / theo.loc[t, "theoretical_gram_try"]), places=15)

    def test_target_skips_calendar_gaps(self):
        # 2015-01-09 (Cuma) hedefi, takvimde bir sonraki gun degil 2015-01-13'tur
        m = self.model.set_index(self.model["Date"].dt.strftime("%Y-%m-%d"))
        p = self.theo.set_index(self.theo["Date"].dt.strftime("%Y-%m-%d"))["theoretical_gram_try"]
        self.assertAlmostEqual(m.loc["2015-01-09", "target_next_return"],
                               math.log(p["2015-01-13"] / p["2015-01-09"]), places=15)

    def test_target_equals_next_gold_return(self):
        np.testing.assert_array_equal(self.model["target_next_return"].to_numpy()[:-1],
                                      self.model["gold_return"].to_numpy()[1:])

    def test_too_few_common_dates_rejected(self):
        gold = pd.DataFrame({"Date": pd.to_datetime(["2015-01-02", "2015-01-05"]), "close": [1.0, 2.0]})
        fx = pd.DataFrame({"Date": pd.to_datetime(["2015-01-02", "2015-01-05"]), "close": [3.0, 4.0]})
        with self.assertRaisesRegex(bd.DataError, "yetersiz"):
            quiet(bd.build, gold, fx)

    def test_validate_outputs_accepts_built_data(self):
        _, out = quiet(bd.validate_outputs, self.theo, self.model)
        self.assertIn("[OK] target[t] = log(P[t+1]/P[t])", out)

    def test_validate_outputs_rejects_shifted_target(self):
        bad = self.model.copy()
        bad["target_next_return"] = bad["gold_return"]  # hedef bir gun kaydirilmis
        with self.assertRaisesRegex(bd.DataError, "hizalamasi hatali"):
            quiet(bd.validate_outputs, self.theo, bad)


# ---------------------------------------------------------------------------
# 3) Mevcut dosyalarla karsilastirma
# ---------------------------------------------------------------------------
class TestCompareWithReference(TempDataDir):

    def setUp(self):
        super().setUp()
        self.write_default_raw()
        self.theo, self.model = self.build_from_raw()
        self.ref = self.out / bd.MODEL_FILE

    def test_missing_reference(self):
        self.assertEqual(bd.compare_with_reference(self.model, self.ref)["status"], "missing")

    def test_identical_reference_matches(self):
        self.model.to_csv(self.ref, index=False, date_format=bd.DATE_FORMAT)
        self.assertEqual(bd.compare_with_reference(self.model, self.ref)["status"], "match")

    def test_small_value_difference_detected(self):
        ref = self.model.copy()
        ref.loc[2, "usdtry"] += 1e-6
        ref.to_csv(self.ref, index=False, date_format=bd.DATE_FORMAT)
        res = bd.compare_with_reference(self.model, self.ref)
        self.assertEqual(res["status"], "differ")
        self.assertTrue(any("FARK usdtry: 1 satir" in line for line in res["lines"]))

    def test_extra_date_detected(self):
        ref = pd.concat([self.model, self.model.tail(1).assign(Date=pd.Timestamp("2016-01-04"))])
        ref.to_csv(self.ref, index=False, date_format=bd.DATE_FORMAT)
        res = bd.compare_with_reference(self.model, self.ref)
        self.assertEqual(res["status"], "differ")
        self.assertTrue(any("yalnizca mevcut dosyada 1 tarih" in line for line in res["lines"]))

    def test_column_difference_detected(self):
        self.model.drop(columns="usdtry_return").to_csv(self.ref, index=False, date_format=bd.DATE_FORMAT)
        res = bd.compare_with_reference(self.model, self.ref)
        self.assertEqual(res["status"], "differ")
        self.assertTrue(any("FARK sutunlar" in line for line in res["lines"]))


# ---------------------------------------------------------------------------
# 4) Calistirma modlari: varsayilan, --write, --force
# ---------------------------------------------------------------------------
class TestMainModes(TempDataDir):

    def setUp(self):
        super().setUp()
        self.write_default_raw()
        self.raw_before = file_hashes(self.raw)

    def tearDown(self):
        # Hicbir modda ham dosyalar degismemeli
        self.assertEqual(file_hashes(self.raw), self.raw_before)

    def write_reference(self):
        theo, model = self.build_from_raw()
        theo.to_csv(self.out / bd.THEO_FILE, index=False, date_format=bd.DATE_FORMAT)
        model.to_csv(self.out / bd.MODEL_FILE, index=False, date_format=bd.DATE_FORMAT)

    def test_default_mode_without_reference_writes_nothing(self):
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(file_hashes(self.out), {})
        self.assertIn("Hicbir dosya yazilmadi", out)

    def test_default_mode_with_matching_reference_writes_nothing(self):
        self.write_reference()
        before = file_hashes(self.out)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("Sonuc: MATCH"), 2)
        self.assertEqual(file_hashes(self.out), before)

    def test_default_mode_with_different_reference_exits_2_and_writes_nothing(self):
        self.write_reference()
        ref = pd.read_csv(self.out / bd.THEO_FILE)
        ref.loc[0, "gold_usd_oz"] += 0.5
        ref.to_csv(self.out / bd.THEO_FILE, index=False)
        before = file_hashes(self.out)
        code, out = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("Sonuc: DIFFER", out)
        self.assertEqual(file_hashes(self.out), before)

    def test_write_creates_missing_files(self):
        code, _ = self.run_main("--write")
        self.assertEqual(code, 0)
        self.assertEqual(set(file_hashes(self.out)), {bd.THEO_FILE, bd.MODEL_FILE})
        model = pd.read_csv(self.out / bd.MODEL_FILE)
        self.assertEqual(list(model.columns), bd.MODEL_COLUMNS)
        self.assertEqual(len(model), len(COMMON) - 2)

    def test_write_leaves_matching_files_untouched(self):
        self.write_reference()
        before = file_hashes(self.out)
        code, out = self.run_main("--write")
        self.assertEqual(code, 0)
        self.assertIn("zaten uyumlu", out)
        self.assertEqual(file_hashes(self.out), before)

    def test_write_without_force_refuses_on_difference(self):
        self.write_reference()
        (self.out / bd.MODEL_FILE).write_text("Date,x\n2015-01-01,1\n", encoding="utf-8")
        before = file_hashes(self.out)
        code, _ = self.run_main("--write")
        self.assertEqual(code, 2)
        self.assertEqual(file_hashes(self.out), before)

    def test_write_force_backs_up_then_overwrites(self):
        self.write_reference()
        (self.out / bd.MODEL_FILE).write_text("Date,x\n2015-01-01,1\n", encoding="utf-8")
        old = hashlib.md5((self.out / bd.MODEL_FILE).read_bytes()).hexdigest()
        code, _ = self.run_main("--write", "--force")
        self.assertEqual(code, 0)
        backups = list(self.out.glob("model_dataset.backup_*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(hashlib.md5(backups[0].read_bytes()).hexdigest(), old)
        self.assertEqual(list(pd.read_csv(self.out / bd.MODEL_FILE).columns), bd.MODEL_COLUMNS)
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_force_without_write_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                bd.parse_args(["--force"])
        self.assertEqual(ctx.exception.code, 2)

    def test_input_error_exits_1_and_writes_nothing(self):
        (self.raw / bd.FX_FILE).unlink()
        self.raw_before = file_hashes(self.raw)
        code, out = self.run_main("--write", "--force")
        self.assertEqual(code, 1)
        self.assertIn("HATA", out)
        self.assertEqual(file_hashes(self.out), {})


# ---------------------------------------------------------------------------
# 5) validate.py ile uyum (ayni tanimlar)
# ---------------------------------------------------------------------------
class TestValidateCompatibility(TempDataDir):

    def setUp(self):
        super().setUp()
        self.write_default_raw()
        code, _ = self.run_main("--write")
        self.assertEqual(code, 0)

    def test_built_files_pass_validate(self):
        (tg, err1), _ = quiet(vd.validate_file, self.out / bd.THEO_FILE, vd.TG_COLUMNS)
        (md, err2), _ = quiet(vd.validate_file, self.out / bd.MODEL_FILE, vd.TG_COLUMNS + vd.MD_EXTRA)
        self.assertEqual((err1, err2), (0, 0))
        self.assertEqual(quiet(vd.check_formula, tg)[0], 0)
        self.assertEqual(quiet(vd.check_formula, md)[0], 0)
        self.assertEqual(quiet(vd.check_target, md)[0], 0)

    def test_validate_detects_shifted_target(self):
        md = pd.read_csv(self.out / bd.MODEL_FILE)
        md["target_next_return"] = md["gold_return"]
        self.assertEqual(quiet(vd.check_target, md)[0], 1)

    def test_validate_detects_missing_column(self):
        path = self.out / bd.MODEL_FILE
        pd.read_csv(path).drop(columns="usdtry_return").to_csv(path, index=False)
        (_, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS + vd.MD_EXTRA)
        self.assertGreaterEqual(err, 1)
        self.assertIn("Eksik sutunlar", out)

    def test_validate_detects_unsorted_and_duplicate_dates(self):
        path = self.out / bd.THEO_FILE
        df = pd.read_csv(path)
        pd.concat([df.iloc[[1, 0]], df.iloc[[0]]]).to_csv(path, index=False)
        (_, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS)
        self.assertGreaterEqual(err, 2)
        self.assertIn("sirali degil", out)
        self.assertIn("mukerrer tarih", out)

    def test_validate_detects_non_positive_price(self):
        path = self.out / bd.THEO_FILE
        df = pd.read_csv(path)
        df.loc[0, "usdtry"] = -1.0
        df.to_csv(path, index=False)
        (_, err), out = quiet(vd.validate_file, path, vd.TG_COLUMNS)
        self.assertGreaterEqual(err, 1)
        self.assertIn("sifir veya negatif", out)


# ---------------------------------------------------------------------------
# Gercek proje verisinin korunmasi
# ---------------------------------------------------------------------------
_REAL_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed"]
_REAL_BEFORE = {}


def setUpModule():
    for d in _REAL_DIRS:
        _REAL_BEFORE[d] = file_hashes(d)


def tearDownModule():
    for d in _REAL_DIRS:
        if file_hashes(d) != _REAL_BEFORE[d]:
            raise AssertionError(f"Testler gercek veri klasorunu degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

