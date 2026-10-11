"""
Gram Altin Fiyat Tahmini - Model Egitim ve Degerlendirme
========================================================
Kullanim:  python -m src.models.train

Veri:      data/processed/model_dataset.csv
Cikti:     data/processed/results/

Yontem:    Kronolojik bolme + genisleyen pencereli walk-forward.
           Her model ayni tarih ve hedef uzerinden degerlendirilir.
           Ozellikler yalnizca tahmin aninda bilinen veriden hesaplanir.

Tarih ve hedef iliskisi
-----------------------
Her satir t, bir tahmin firsatini temsil eder:
  - Tahmin tarihi: t (piyasa kapanisinda bilinen veri)
  - Hedef:  target_next_return[t] = log(P[t+1] / P[t])
  - Hedef tarihi: t+1 (bir sonraki is gunu)
  - Tahmin edilen fiyat:  P_hat[t+1] = P[t] * exp(y_hat[t])

P = theoretical_gram_try = gold_usd_oz * usdtry / 31.1035
Bu fiyat teoriktir; fiziksel altin piyasa primini icermez.

Degerlendirme protokolu (Faz A)
-------------------------------
  Egitim     : veri baslangici .. TRAIN_END (2021-12-31)
  Dogrulama  : VALIDATION_START (2022-01-01) .. TEST_START oncesi
               Hedef tarihi test donemine tasan son satir dogrulamaya alinmaz.
  Test       : TEST_START (2023-10-26) ve sonrasi
  Model, ozellik ve hiperparametre kararlari YALNIZCA dogrulama sonuclarina
  gore verilir. Grafik ve legacy alt donem tablosu icin "en iyi model" de
  dogrulama doneminde secilir. Test sonuclari raporlanir, secimde kullanilmaz.
  Test degerlendirmesi (holdout ve walk-forward) onceki surumle aynidir.

Faz B deneyi (izole)
--------------------
  python -m src.models.train --experiment phase_b
  Uc ozellik setini (base, component, compact) YALNIZCA dogrulama
  doneminde karsilastirir; ciktilar results/experiments/phase_b/ altina
  yazilir. Varsayilan calistirma 'base' setini kullanir ve degismez.

Purge (1 gozlemlik ambargo)
---------------------------
Her egitimde, egitim kumesinin son satiri cikarilir. Bu satirin etiketi
log(P[t0] / P[t0-1])'dir; t0 degerlendirme penceresinin ilk tahmin gunudur.
Model t0 kapanisinda, ilk tahminden hemen once egitiliyorsa P[t0] zaten
bilinir; bu etiket o senaryoda SIZINTI DEGILDIR. Satir yine de cikarilir:
modelin t0 kapanisindan once egitildigi senaryoyu da guvenli kilar ve
onceki sonuclarla uyumu korur. Yani purge bir sizinti duzeltmesi degil,
muhafazakar bir ambargodur.
"""

from datetime import datetime, timezone
import argparse
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

# ---------------------------------------------------------------------------
# Yapilandirma
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "results"   # main() icinde olusturulur (import yan etkisi yok)

# Kronolojik protokol
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"
TEST_START = "2023-10-26"
RETRAIN_EVERY = 63

MIN_TRAIN_ROWS = 50       # her egitim penceresi icin en az satir
MIN_EVAL_ROWS = 20        # dogrulama / test donemi icin en az satir
MIN_YEAR_OBS = 10         # yillik tabloda metrik hesaplamak icin en az gozlem
RETURN_TOL = 1e-9         # getiri sutunlari tutarlilik toleransi (mutlak)

TARGET = "target_next_return"
PRICE_COL = "theoretical_gram_try"

LAG_DAYS = [1, 2, 5, 10]
ROLLING_WINDOWS = [5, 20]
VOLATILITY_WINDOW = 20

FEATURES = [
    "gold_return",       # bugunun gram altin getirisi
    "gold_usd_return",   # bugunun ons altin (USD) getirisi
    "usdtry_return",     # bugunun USD/TRY getirisi
    "return_lag_1",      # dunun gram altin getirisi
    "return_lag_2",
    "return_lag_5",
    "return_lag_10",
    "return_mean_5",     # son 5 gunun ortalama getirisi
    "return_mean_20",
    "return_volatility_20",  # son 20 gunun getiri standart sapmasi
    "day_of_week",       # 0=Pazartesi .. 4=Cuma
]

# Faz B: bilesen ayristirmali suruklenme ve oynaklik ozellikleri.
# fx = usdtry_return (USD/TRY log getirisi), gold_usd = gold_usd_return (ons, USD).
COMPONENT_SHORT_WINDOW = 20
COMPONENT_LONG_WINDOW = 60
COMPONENT_MIN_PERIODS = 20      # 60'lik pencere 20 gozlemden itibaren deger uretir
COMPONENT_FEATURES = [
    "fx_mean_20",          # usdtry_return son 20 gozlem ortalamasi (t dahil)
    "fx_mean_60",          # usdtry_return son min(60, mevcut) gozlem ortalamasi
    "gold_usd_mean_20",    # gold_usd_return son 20 gozlem ortalamasi
    "gold_usd_mean_60",    # gold_usd_return son min(60, mevcut) gozlem ortalamasi
    "fx_vol_20",           # usdtry_return son 20 gozlem std (ddof=1)
    "gold_usd_vol_20",     # gold_usd_return son 20 gozlem std (ddof=1)
]
# Kompakt set: gram getirisi (gold_return) ve ondan turetilen gecikme/pencere
# ozellikleri yok; gold_return = gold_usd_return + usdtry_return oldugu icin
# bilgi kaybi olmadan dogrusal bagimlilik kaldirilir.
COMPACT_FEATURES = [
    "gold_usd_return",
    "usdtry_return",
    *COMPONENT_FEATURES,
    "day_of_week",
]
FEATURE_SETS = {
    "base": list(FEATURES),                           # 11 ozellik, Faz A ile ayni
    "component": list(FEATURES) + COMPONENT_FEATURES,  # 17 ozellik
    "compact": list(COMPACT_FEATURES),                # 9 ozellik
}
ALL_FEATURES = list(dict.fromkeys(f for fs in FEATURE_SETS.values() for f in fs))

REQUIRED_COLS = [
    "Date", "gold_usd_oz", "usdtry", PRICE_COL,
    "gold_return", "gold_usd_return", "usdtry_return", TARGET,
]

RANDOM_STATE = 42

# Ozellik isinma donemi: en uzun geriye bakan pencere kadar ilk satir NaN olur.
# shift(k) ilk k satiri, rolling(w, min_periods=w) ilk w-1 satiri bos birakir.
WARMUP_ROWS = max(max(LAG_DAYS), max(ROLLING_WINDOWS) - 1, VOLATILITY_WINDOW - 1,
                  COMPONENT_SHORT_WINDOW - 1, COMPONENT_MIN_PERIODS - 1)

TARGET_DEFINITION = "target_next_return[t] = log(P[t+1] / P[t]), P = theoretical_gram_try"
FORECAST_HORIZON = "1 gozlem: bir sonraki ortak islem gunu (takvim gunu degil)"

# Kodda acikca belirtilen model parametreleri (run_config.json'a aynen yazilir)
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=5,
                 random_state=RANDOM_STATE, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8,
                  objective="reg:squarederror", random_state=RANDOM_STATE,
                  n_jobs=-1, verbosity=0)


# ---------------------------------------------------------------------------
# Ozellik muhendisligi
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gecikmeli getiri, rolling istatistik ve takvim ozelliklerini ekler.

    Tum rolling pencereler geriye bakar (pandas varsayilani).
    min_periods, pencerenin tam dolmasini zorunlu kilar; ilk satirlar NaN kalir.

    Zamanlama:
      gold_return[t] = log(P[t]/P[t-1])  -> t kapanisinda bilinen
      return_lag_k[t] = gold_return[t-k]  -> shift(k), t'den k gun oncesi
      return_mean_w[t] = ortalama(gold_return[t-w+1 .. t])  -> geriye bakan pencere
    """
    out = df.copy()
    ret = out["gold_return"]

    for k in LAG_DAYS:
        out[f"return_lag_{k}"] = ret.shift(k)

    for w in ROLLING_WINDOWS:
        out[f"return_mean_{w}"] = ret.rolling(w, min_periods=w).mean()

    out[f"return_volatility_{VOLATILITY_WINDOW}"] = ret.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()

    out["day_of_week"] = out["Date"].dt.dayofweek

    # Faz B bilesen ozellikleri (yalnizca t ve oncesi; geriye bakan pencereler)
    for prefix, col in (("fx", "usdtry_return"), ("gold_usd", "gold_usd_return")):
        r = out[col]
        out[f"{prefix}_mean_{COMPONENT_SHORT_WINDOW}"] = r.rolling(
            COMPONENT_SHORT_WINDOW, min_periods=COMPONENT_SHORT_WINDOW).mean()
        out[f"{prefix}_mean_{COMPONENT_LONG_WINDOW}"] = r.rolling(
            COMPONENT_LONG_WINDOW, min_periods=COMPONENT_MIN_PERIODS).mean()
        out[f"{prefix}_vol_{COMPONENT_SHORT_WINDOW}"] = r.rolling(
            COMPONENT_SHORT_WINDOW, min_periods=COMPONENT_SHORT_WINDOW).std()
    return out


# ---------------------------------------------------------------------------
# Veri yukleme
# ---------------------------------------------------------------------------
def check_return_consistency(df: pd.DataFrame) -> dict:
    """Getiri sutunlarinin fiyatlarla ve birbirleriyle tutarliligini dogrular.

    Kontroller (satir 1..n-1; ilk satirin onceki fiyati dosyada yoktur):
      gold_return[t]     = log(P[t] / P[t-1])                 P = theoretical_gram_try
      gold_usd_return[t] = log(gold_usd_oz[t] / gold_usd_oz[t-1])
      usdtry_return[t]   = log(usdtry[t] / usdtry[t-1])
    Ozdeslik (tum satirlar):
      gold_return[t] = gold_usd_return[t] + usdtry_return[t]
    Sapma RETURN_TOL'u asarsa ValueError. Dondurur: kontrol bazinda max sapma.
    """
    out = {}
    if len(df) >= 2:
        for col, price in (("gold_return", PRICE_COL),
                           ("gold_usd_return", "gold_usd_oz"),
                           ("usdtry_return", "usdtry")):
            p = df[price].to_numpy(dtype=float)
            dev = np.abs(np.log(p[1:] / p[:-1]) - df[col].to_numpy(dtype=float)[1:])
            out[col] = float(dev.max())
            if out[col] > RETURN_TOL:
                worst = df["Date"].iat[int(np.argmax(dev)) + 1]
                raise ValueError(
                    f"{col}, {price} fiyatlarindan hesaplanan log getiriyle tutarsiz "
                    f"(max sapma {out[col]:.3e}, tarih {pd.Timestamp(worst).date()})."
                )
    ident = np.abs(df["gold_return"].to_numpy(dtype=float)
                   - df["gold_usd_return"].to_numpy(dtype=float)
                   - df["usdtry_return"].to_numpy(dtype=float))
    out["identity"] = float(ident.max()) if len(ident) else 0.0
    if out["identity"] > RETURN_TOL:
        worst = df["Date"].iat[int(np.argmax(ident))]
        raise ValueError(
            f"gold_return != gold_usd_return + usdtry_return "
            f"(max sapma {out['identity']:.3e}, tarih {pd.Timestamp(worst).date()})."
        )
    return out


def load_data(path=None) -> pd.DataFrame:
    """model_dataset.csv yukler, veri sozlesmesini dogrular, ozellikleri hesaplar.

    Veri sozlesmesi (ihlalde ValueError; hicbir satir sessizce dusmez):
      - Gerekli sutunlar mevcut, Date ayristirilabilir.
      - Tarihler dosyada ZATEN kesin artan sirada (veri siralanmaz; sirasiz
        dosya kaynakta bir sorun oldugunu gosterir).
      - Gerekli sutunlarda NaN yok; fiyatlar pozitif; sonsuz deger yok.
      - Getiri sutunlari fiyatlarla ve birbirleriyle tutarli.
      - Ozellik hesabindan sonra NaN yalnizca ilk WARMUP_ROWS satirda olabilir.

    path: varsayilan DATA_DIR / "model_dataset.csv" (testler icin degistirilebilir).
    Cikis DataFrame'i 0-tabanli ardisik indekse sahiptir; iloc ve pozisyon
    tabanli islemler bu varsayima dayanir.
    """
    path = Path(path) if path is not None else DATA_DIR / "model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    # Sutun kontrolu
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sutunlar: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date sutunu tarih olarak ayristirilamadi.")
    if len(df) == 0:
        raise ValueError("Veri dosyasi bos.")

    # Siralama ve mukerrer tarih: dosyanin kendi sirasi kontrol edilir
    if df["Date"].isna().any():
        raise ValueError(f"{int(df['Date'].isna().sum())} bos tarih var.")
    n_dup = int(df["Date"].duplicated().sum())
    if n_dup > 0:
        raise ValueError(f"{n_dup} mukerrer tarih bulundu.")
    if not df["Date"].is_monotonic_increasing:
        bad = int(np.argmax(df["Date"].diff().dt.total_seconds().to_numpy()[1:] < 0)) + 1
        raise ValueError(
            "Tarihler kronolojik sirali degil "
            f"(satir {bad}: {df['Date'].iat[bad].date()} < {df['Date'].iat[bad - 1].date()}). "
            "Veri sessizce siralanmaz; build_dataset.py ciktisini kontrol edin."
        )
    df = df.reset_index(drop=True)

    # Gerekli sutunlarda NaN
    nan_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLS[1:]}
    nan_counts = {c: n for c, n in nan_counts.items() if n}
    if nan_counts:
        raise ValueError(f"Gerekli sutunlarda NaN var: {nan_counts}")

    # Fiyat sutunlarinda gecersiz deger
    for col in ["gold_usd_oz", "usdtry", PRICE_COL]:
        n_bad = int((df[col] <= 0).sum() + df[col].isna().sum())
        if n_bad > 0:
            raise ValueError(f"{col} sutununda {n_bad} gecersiz deger (<=0 veya NaN).")

    # Sonsuz deger kontrolu (log getirilerden gelebilir)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[numeric_cols]).sum().sum())
    if n_inf > 0:
        raise ValueError(f"Veride {n_inf} sonsuz (inf/-inf) deger var.")

    # Getiri sutunlari tutarliligi
    check_return_consistency(df)

    df = build_features(df)

    # Isinma donemi disinda NaN olmamali
    nan_rows = df[ALL_FEATURES + [TARGET]].isna().any(axis=1).to_numpy()
    unexpected = np.flatnonzero(nan_rows[WARMUP_ROWS:]) + WARMUP_ROWS
    if len(unexpected):
        dates = [str(df["Date"].iat[i].date()) for i in unexpected[:5]]
        raise ValueError(
            f"Isinma donemi ({WARMUP_ROWS} satir) disinda {len(unexpected)} satirda NaN "
            f"ozellik/hedef var (ornek: {dates})."
        )

    before = len(df)
    df = df.loc[~nan_rows].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  {dropped} satir ozellik isinma donemi nedeniyle dustu "
              f"(en uzun pencere: {WARMUP_ROWS + 1} gozlem).")

    if len(df) == 0:
        raise ValueError("Isinma donemi sonrasi satir kalmadi; veri yetersiz.")

    return df


# ---------------------------------------------------------------------------
# Purging: egitim etiketlerinin test sinirina tasmasini onler
# ---------------------------------------------------------------------------
def _purge_train(df: pd.DataFrame, train_end_pos: int) -> pd.DataFrame:
    """Egitim kumesinin son satirini cikarir (1 gozlemlik muhafazakar ambargo).

    Son egitim satirinin etiketi log(P[t0] / P[t0-1]); t0 = train_end_pos,
    yani degerlendirme penceresinin ilk tahmin gunu. Model t0 kapanisinda,
    ilk tahminden hemen once egitiliyorsa P[t0] bilinir ve bu etiket sizinti
    degildir. Model t0 kapanisindan ONCE egitiliyorsa (ornegin gun icinde)
    P[t0] henuz bilinmez; bu durumda satirin cikarilmasi gerekir. Kod her iki
    senaryoyu da guvenli kilmak ve onceki sonuclarla uyumu korumak icin
    satiri her zaman cikarir. Davranis onceki surumle aynidir.

    Parametreler:
      df: tam veri seti (reset_index sonrasi, indeks = pozisyon)
      train_end_pos: egitim kumesinin bittigi pozisyon (exclusive),
                     yani df.iloc[:train_end_pos] egitim adayidir.

    Dondurulen deger:
      Purge edilmis egitim DataFrame'i.
    """
    if train_end_pos <= 0:
        return df.iloc[:0]

    train_df = df.iloc[:train_end_pos].copy()

    # Son egitim satirinin hedef tarihini kontrol et
    last_pos = train_end_pos - 1
    if last_pos + 1 < len(df):
        target_date = df["Date"].iat[last_pos + 1]
        boundary = df["Date"].iat[train_end_pos] if train_end_pos < len(df) else None
        # Hedef tarihi egitim sinirinin disindaysa bu satiri cikar
        if boundary is not None and target_date >= boundary:
            train_df = train_df.iloc[:-1]

    return train_df


# ---------------------------------------------------------------------------
# Hedef tarih hesaplama
# ---------------------------------------------------------------------------
def _get_target_dates(df: pd.DataFrame, row_positions: list) -> list:
    """Her tahmin satiri icin hedef tarihi (t+1) bulur.

    row_positions: DataFrame satirlarinin pozisyonlari.
    load_data() sonrasi reset_index yapildigi icin indeks = pozisyon.
    Sonraki satiri bulunmayan satirlar icin pd.NaT doner.
    """
    n = len(df)
    result = []
    for pos in row_positions:
        if pos + 1 < n:
            result.append(df["Date"].iat[pos + 1])
        else:
            result.append(pd.NaT)
    return result


# ---------------------------------------------------------------------------
# Model tanimlari
# ---------------------------------------------------------------------------
def get_models() -> dict:
    """Karsilastirilacak modeller.

    Baseline'lar:
      zero  - Getiri sifir: yarin bugunun fiyatina esit (naive persistence).
      mean  - Getiri egitim ortalamasina esit.
      last  - Getiri bugunkuyle ayni: gold_return[t] ile tahmin.

    ML modelleri:
      Her biri FEATURES vektorunden log getiri tahmin eder.
    """
    return {
        "Zero-Return": "zero",
        "Mean-Return": "mean",
        "Last-Return": "last",
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "XGBoost": XGBRegressor(**XGB_PARAMS),
    }


BASELINE_NAMES = {"Zero-Return", "Mean-Return", "Last-Return"}


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def direction_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    """Yon dogrulugu (%): sign(tahmin) == sign(gercek) olan gunlerin orani.

    Isaret: sign(x) > 0 yukselis, < 0 dusus, = 0 yonsuz.
    Payda: tahmin VE gercek getirinin ikisinin de sifirdan farkli oldugu
    gunler (n_directional). Sifir tahminler (ornegin Zero-Return'un tum
    tahminleri) ve sifir gerceklesen getiriler paydaya girmez.
    Payda 0 ise NaN doner; Zero-Return icin bu nedenle deger bostur.
    """
    a_sign = np.sign(actual)
    p_sign = np.sign(pred)
    mask = (p_sign != 0) & (a_sign != 0)
    n = mask.sum()
    if n == 0:
        return np.nan
    return float(np.mean(a_sign[mask] == p_sign[mask]) * 100)


def compute_metrics(
    actual_ret: np.ndarray,
    pred_ret: np.ndarray,
    prices_today: np.ndarray,
) -> dict:
    """Getiri ve teorik fiyat uzayinda metrikleri hesaplar.

    Fiyat metrikleri:
      actual_price[t+1] = P[t] * exp(actual_return[t])
      pred_price[t+1]   = P[t] * exp(pred_return[t])
      MAE_TL, RMSE_TL bu iki fiyat arasinda hesaplanir.

    Beceri skoru (skill_vs_zero):
      1 - MAE_model / MAE_zero_return
      Pozitifse model sifir-getiri tahminindan iyi.
      Referans: Zero-Return baseline (yarin = bugun).
    """
    n = len(actual_ret)
    if n == 0:
        return {
            "MAE_return": np.nan, "RMSE_return": np.nan,
            "direction_accuracy_pct": np.nan,
            "MAE_TL": np.nan, "RMSE_TL": np.nan,
            "skill_vs_zero": np.nan,
        }

    mae_ret = float(mean_absolute_error(actual_ret, pred_ret))
    rmse_ret = float(np.sqrt(mean_squared_error(actual_ret, pred_ret)))
    dir_acc = direction_accuracy(actual_ret, pred_ret)

    actual_price = prices_today * np.exp(actual_ret)
    pred_price = prices_today * np.exp(pred_ret)
    mae_tl = float(mean_absolute_error(actual_price, pred_price))
    rmse_tl = float(np.sqrt(mean_squared_error(actual_price, pred_price)))

    zero_mae = float(mean_absolute_error(actual_ret, np.zeros_like(actual_ret)))
    skill = 1.0 - mae_ret / zero_mae if zero_mae > 0 else 0.0

    return {
        "MAE_return": round(mae_ret, 6),
        "RMSE_return": round(rmse_ret, 6),
        "direction_accuracy_pct": round(dir_acc, 2) if not np.isnan(dir_acc) else None,
        "MAE_TL": round(mae_tl, 2),
        "RMSE_TL": round(rmse_tl, 2),
        "skill_vs_zero": round(skill, 4),
    }


def _predict(spec, test_df, train_mean, X_train, y_train, X_test):
    """Tek bir model veya baseline icin tahmin uretir."""
    if spec == "zero":
        return np.zeros(len(test_df))
    if spec == "mean":
        return np.full(len(test_df), train_mean)
    if spec == "last":
        return test_df["gold_return"].values.copy()
    m = clone(spec)
    m.fit(X_train, y_train)
    return m.predict(X_test)


def directional_count(actual: np.ndarray, pred: np.ndarray) -> int:
    """Yon dogrulugunun paydasi: tahmin ve gercegin ikisi de sifir degil."""
    return int(((np.sign(actual) != 0) & (np.sign(pred) != 0)).sum())


def up_fraction(actual: np.ndarray) -> float:
    """Sifirdan farkli gerceklesen getiriler icinde pozitif olanlarin orani (%).

    Her gun 'yukselis' diyen bir tahminin yon dogruluguna esittir; yon
    dogrulugu bu oranla birlikte okunmalidir.
    """
    nz = actual[actual != 0]
    return float(np.mean(nz > 0) * 100) if len(nz) else np.nan


def add_reference_metrics(metrics: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy metrik tablosunun SONUNA referans-goreli sutunlar ekler.

    skill_vs_mean   = 1 - MAE_model / MAE_Mean-Return
                      (Mean-Return'un ayni gozlemlerdeki tahminleriyle)
    n_obs           = degerlendirme gozlem sayisi
    n_directional   = yon dogrulugu paydasi
    up_fraction_pct = sifir olmayan gerceklesen getirilerde pozitif orani (%)
    Legacy sutunlar ve degerleri degismez.
    """
    actual = pred_df["actual_return"].to_numpy(dtype=float)
    mean_mae = float(mean_absolute_error(actual, pred_df["pred_Mean-Return"].to_numpy(dtype=float)))
    up = up_fraction(actual)
    out = metrics.copy()
    skill, n_dir = [], []
    for name in out["model"]:
        pred = pred_df[f"pred_{name}"].to_numpy(dtype=float)
        mae = float(mean_absolute_error(actual, pred))
        skill.append(round(1.0 - mae / mean_mae, 4) + 0.0 if mean_mae > 0 else np.nan)
        n_dir.append(directional_count(actual, pred))
    out["skill_vs_mean"] = skill
    out["n_obs"] = len(actual)
    out["n_directional"] = n_dir
    out["up_fraction_pct"] = round(up, 2) if not np.isnan(up) else None
    return out


def diebold_mariano(err_model, err_ref, lag=None) -> dict:
    """Mutlak hata kaybi icin Diebold-Mariano testi (1 adim ileri tahmin).

    Kayip farki:  d_t = |e_model,t| - |e_ref,t|        H0: E[d_t] = 0
    Istatistik:   DM = mean(d) / sqrt(Var_HAC(mean(d)))
    Varyans:      Newey-West (Bartlett cekirdegi) uzun donem varyansi / T.
                  lag varsayilani floor(4 * (T/100)^(2/9)), en fazla T-1.
    Dagilim:      Asimptotik standart normal; p iki yonlu, math.erfc ile.
    Isaret:       DM < 0 -> model referanstan dusuk mutlak hataya sahip.

    Varsayimlar ve sinirlar: d_t'nin kovaryans-duragan olmasi; kucuk
    orneklem duzeltmesi (Harvey-Leybourne-Newbold) ve t dagilimi
    UYGULANMAZ. Ic ice modellerde (ornegin Zero-Return ile Mean-Return)
    asimptotik dagilim yaklasiktir. Anlamlilik, pratik tahmin degeri
    oldugunu kanitlamaz; etki buyuklugu (mean_loss_diff) ile okunmalidir.

    Dejenere durumlar: tum d_t = 0 ise DM = 0, p = 1. d_t sabit ve sifirdan
    farkliysa varyans 0'dir; DM = +/-inf, p = 0 olarak raporlanir.
    """
    e1 = np.asarray(err_model, dtype=float)
    e2 = np.asarray(err_ref, dtype=float)
    if e1.shape != e2.shape or e1.ndim != 1:
        raise ValueError("Hata serileri ayni uzunlukta tek boyutlu olmali.")
    if not (np.isfinite(e1).all() and np.isfinite(e2).all()):
        raise ValueError("Hata serilerinde NaN veya sonsuz deger var.")
    d = np.abs(e1) - np.abs(e2)
    T = len(d)
    if T < 2:
        raise ValueError("DM testi icin en az 2 gozlem gerekir.")
    if lag is None:
        lag = int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    lag = int(max(0, min(lag, T - 1)))

    dbar = float(d.mean())
    dc = d - dbar
    long_run = float(dc @ dc) / T
    for k in range(1, lag + 1):
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * float(dc[k:] @ dc[:-k]) / T
    var_mean = long_run / T

    if var_mean <= 0.0 or not np.isfinite(var_mean):
        if dbar == 0.0:
            stat, p = 0.0, 1.0
        else:
            stat, p = math.copysign(math.inf, dbar), 0.0
    else:
        stat = dbar / math.sqrt(var_mean)
        p = math.erfc(abs(stat) / math.sqrt(2.0))
    return {"n": T, "mean_loss_diff": dbar, "dm_stat": stat, "p_value": p, "hac_lag": lag}


def significance_table(pred_df: pd.DataFrame, period: str,
                       references=("Mean-Return", "Zero-Return")) -> pd.DataFrame:
    """Her modeli referans baseline'lara karsi DM testiyle karsilastirir."""
    actual = pred_df["actual_return"].to_numpy(dtype=float)
    names = [c[len("pred_"):] for c in pred_df.columns if c.startswith("pred_")]
    rows = []
    for ref in references:
        e_ref = actual - pred_df[f"pred_{ref}"].to_numpy(dtype=float)
        for name in names:
            if name == ref:
                continue
            e_mod = actual - pred_df[f"pred_{name}"].to_numpy(dtype=float)
            r = diebold_mariano(e_mod, e_ref)
            rows.append({
                "period": period, "model": name, "reference": ref, "n": r["n"],
                "mean_abs_err_diff": r["mean_loss_diff"], "dm_stat": r["dm_stat"],
                "p_value": r["p_value"], "hac_lag": r["hac_lag"],
                "model_lower_loss": bool(r["mean_loss_diff"] < 0),
            })
    return pd.DataFrame(rows)


def annual_comparison(pred_df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Tum modeller icin yillik metrikler (yil = tahmin tarihinin yili).

    MIN_YEAR_OBS'tan az gozlemli yillar satir olarak kalir; metrikleri bos
    ve status = 'yetersiz_gozlem' olur.
    """
    pdf = pred_df.copy()
    pdf["year"] = pd.to_datetime(pdf["prediction_date"]).dt.year
    names = [c[len("pred_"):] for c in pdf.columns if c.startswith("pred_")]
    rows = []
    for year, grp in pdf.groupby("year"):
        actual = grp["actual_return"].to_numpy(dtype=float)
        n = len(grp)
        enough = n >= MIN_YEAR_OBS
        zero_mae = float(mean_absolute_error(actual, np.zeros_like(actual)))
        mean_mae = float(mean_absolute_error(actual, grp["pred_Mean-Return"].to_numpy(dtype=float)))
        up = up_fraction(actual)
        for name in names:
            pred = grp[f"pred_{name}"].to_numpy(dtype=float)
            row = {"period": period, "year": int(year), "model": name, "n_obs": n,
                   "status": "ok" if enough else "yetersiz_gozlem"}
            if enough:
                mae = float(mean_absolute_error(actual, pred))
                d_acc = direction_accuracy(actual, pred)
                row.update({
                    "MAE_return": round(mae, 6),
                    "RMSE_return": round(float(np.sqrt(mean_squared_error(actual, pred))), 6),
                    "direction_accuracy_pct": round(d_acc, 2) if not np.isnan(d_acc) else None,
                    "n_directional": directional_count(actual, pred),
                    "up_fraction_pct": round(up, 2) if not np.isnan(up) else None,
                    "skill_vs_zero": round(1 - mae / zero_mae, 4) + 0.0 if zero_mae > 0 else None,
                    "skill_vs_mean": round(1 - mae / mean_mae, 4) + 0.0 if mean_mae > 0 else None,
                })
            rows.append(row)
    cols = ["period", "year", "model", "n_obs", "status", "MAE_return", "RMSE_return",
            "direction_accuracy_pct", "n_directional", "up_fraction_pct",
            "skill_vs_zero", "skill_vs_mean"]
    return pd.DataFrame(rows).reindex(columns=cols)


# ---------------------------------------------------------------------------
# Donem ve pencere yardimcilari
# ---------------------------------------------------------------------------
def eval_positions(df: pd.DataFrame, start: str, end=None) -> list:
    """Degerlendirilecek satir pozisyonlari.

    start <= Date ve (end verilmisse) Date < end. end verilmisse ayrica
    hedef tarihi (bir sonraki satirin tarihi) end'den once olmali: aksi
    halde etiket sonraki donemin fiyatini kullanir. Bu, dogrulamanin son
    satirini test donemi fiyatindan ayirir. end=None ise (test) davranis
    onceki surumle aynidir.
    """
    dates = df["Date"]
    mask = dates >= pd.Timestamp(start)
    if end is not None:
        end_ts = pd.Timestamp(end)
        next_dates = dates.shift(-1)
        mask &= (dates < end_ts) & next_dates.notna() & (next_dates < end_ts)
    return df.index[mask].tolist()


def make_windows(positions: list, size: int = RETRAIN_EVERY) -> list:
    """Pozisyonlari sirayla en fazla `size` uzunlugunda pencerelere boler."""
    return [positions[i:i + size] for i in range(0, len(positions), size)]


# ---------------------------------------------------------------------------
# Walk-forward degerlendirme
# ---------------------------------------------------------------------------
def walk_forward_evaluate(df: pd.DataFrame, start: str = None, end: str = None,
                          label: str = "test", features: list = None) -> dict:
    """Genisleyen pencereli walk-forward degerlendirme.

    Degerlendirme satirlari (eval_positions) RETRAIN_EVERY gozlemlik
    pencerelere bolunur. Her pencerede modeller, pencerenin ilk satirindan
    ONCEKI tum veriyle (expanding window) yeniden egitilir; son egitim satiri
    _purge_train ile cikarilir. Pencere icindeki tahminler ayni modelle
    yapilir; tahmin aninda yalnizca o satirin ozellikleri kullanilir.

    start/end:
      Varsayilan (start=TEST_START, end=None): test donemi, onceki surumle ayni.
      Dogrulama: start=VALIDATION_START, end=TEST_START. Egitim verisi her
      zaman pencereden once biter; test donemi hicbir zaman kullanilmaz.

    Dondurur: metrics (legacy sutunlar + referans sutunlari), predictions,
    windows (pencere bazinda egitim/degerlendirme sinirlari), window_of_row
    (her tahmin satirinin pencere numarasi; predictions.csv'ye yazilmaz).

    features: model girdileri; varsayilan FEATURES ('base' seti, Faz A ile ayni).
    Baseline'lar ozellik kullanmaz; tahminleri ozellik setinden bagimsizdir.
    """
    start = TEST_START if start is None else start
    features = FEATURES if features is None else list(features)
    positions = eval_positions(df, start, end)
    if not positions:
        raise ValueError(f"{label} doneminde gozlem yok (start={start}, end={end}).")
    if end is not None and len(positions) < MIN_EVAL_ROWS:
        raise ValueError(f"{label} donemi yetersiz: {len(positions)} gozlem "
                         f"(en az {MIN_EVAL_ROWS}).")

    models_spec = get_models()
    all_preds = {name: [] for name in models_spec}
    all_actual, all_prices, all_dates, all_target_dates = [], [], [], []
    total_purged = 0
    window_info = []
    window_of_row = []

    windows = make_windows(positions, RETRAIN_EVERY)
    print(f"  [{label}] {len(windows)} pencere, ~{RETRAIN_EVERY} gun aralik, "
          f"toplam {len(positions)} gozlem")

    for w_idx, w_indices in enumerate(windows):
        # Purge uygulanmis egitim kumesi
        train_end_pos = w_indices[0]
        train_df = _purge_train(df, train_end_pos)
        n_purged = train_end_pos - len(train_df)
        total_purged += n_purged

        test_df = df.iloc[w_indices]

        if len(train_df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"{label} pencere {w_idx+1}: Purge sonrasi egitim verisi "
                f"yetersiz ({len(train_df)} satir)."
            )

        X_tr = train_df[features].values
        y_tr = train_df[TARGET].values
        X_te = test_df[features].values
        train_mean = float(y_tr.mean())

        window_of_row.extend([w_idx + 1] * len(test_df))
        all_actual.extend(test_df[TARGET].values)
        all_prices.extend(test_df[PRICE_COL].values)
        all_dates.extend(test_df["Date"].tolist())
        all_target_dates.extend(_get_target_dates(df, w_indices))

        for name, spec in models_spec.items():
            pred = _predict(spec, test_df, train_mean, X_tr, y_tr, X_te)
            all_preds[name].extend(pred)

        window_info.append({
            "window": w_idx + 1,
            "train_first_date": str(train_df["Date"].iloc[0].date()),
            "train_last_date": str(train_df["Date"].iloc[-1].date()),
            "n_train": len(train_df),
            "n_purged": n_purged,
            "eval_first_date": str(test_df["Date"].iloc[0].date()),
            "eval_last_date": str(test_df["Date"].iloc[-1].date()),
            "n_eval": len(test_df),
        })

        print(f"    [{w_idx+1}/{len(windows)}] egitim={len(train_df)}"
              f"{f' (purge={n_purged})' if n_purged else ''}, "
              f"degerlendirme={len(test_df)}, "
              f"{test_df['Date'].iloc[0].date()}"
              f" ~ {test_df['Date'].iloc[-1].date()}")

    if total_purged:
        print(f"  Toplam purge edilen egitim satiri: {total_purged}")

    actual_arr = np.array(all_actual)
    prices_arr = np.array(all_prices)

    rows = []
    pred_df = pd.DataFrame({
        "prediction_date": all_dates,
        "target_date": all_target_dates,
        "price_t": prices_arr,
        "actual_return": actual_arr,
    })
    for name in models_spec:
        p = np.array(all_preds[name])
        m = compute_metrics(actual_arr, p, prices_arr)
        m["model"] = name
        rows.append(m)
        pred_df[f"pred_{name}"] = p

    metrics = add_reference_metrics(pd.DataFrame(rows), pred_df)
    return {"metrics": metrics, "predictions": pred_df, "windows": window_info,
            "window_of_row": np.array(window_of_row)}


# ---------------------------------------------------------------------------
# Holdout degerlendirme
# ---------------------------------------------------------------------------
def holdout_evaluate(df: pd.DataFrame) -> dict:
    """Tek seferlik kronolojik bolme.

    Purging: Son egitim satirinin hedefi test doneminin ilk fiyatini
    iceriyorsa cikarilir.
    """
    test_mask = df["Date"] >= pd.Timestamp(TEST_START)
    test_df = df[test_mask]

    if len(test_df) == 0:
        raise ValueError(f"Test doneminde gozlem yok (TEST_START={TEST_START}).")

    train_end_pos = test_df.index[0]
    train_df = _purge_train(df, train_end_pos)
    n_purged = train_end_pos - len(train_df)

    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Purge sonrasi egitim verisi yetersiz ({len(train_df)}).")

    if n_purged:
        print(f"  Purge: {n_purged} egitim satiri cikarildi "
              f"(hedefi test doneminin ilk fiyatini iceriyor).")

    X_tr, y_tr = train_df[FEATURES].values, train_df[TARGET].values
    X_te = test_df[FEATURES].values
    y_te = test_df[TARGET].values
    prices = test_df[PRICE_COL].values
    train_mean = float(y_tr.mean())

    test_indices = test_df.index.tolist()
    target_dates = _get_target_dates(df, test_indices)

    models_spec = get_models()
    rows = []
    pred_df = pd.DataFrame({
        "prediction_date": test_df["Date"].values,
        "target_date": target_dates,
        "price_t": prices,
        "actual_return": y_te,
    })

    for name, spec in models_spec.items():
        pred = _predict(spec, test_df, train_mean, X_tr, y_tr, X_te)
        m = compute_metrics(y_te, pred, prices)
        m["model"] = name
        rows.append(m)
        pred_df[f"pred_{name}"] = pred

    metrics = add_reference_metrics(pd.DataFrame(rows), pred_df)
    return {"metrics": metrics, "predictions": pred_df}


# ---------------------------------------------------------------------------
# Alt donem analizi
# ---------------------------------------------------------------------------
def subperiod_analysis(pred_df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Yillik alt donem metrikleri."""
    pdf = pred_df.copy()
    pdf["year"] = pd.to_datetime(pdf["prediction_date"]).dt.year
    rows = []
    for year, grp in pdf.groupby("year"):
        if len(grp) < 10:
            continue
        actual = grp["actual_return"].values
        pred = grp[model_col].values
        zero_mae = float(mean_absolute_error(actual, np.zeros_like(actual)))
        mae = float(mean_absolute_error(actual, pred))
        d_acc = direction_accuracy(actual, pred)
        rows.append({
            "year": int(year),
            "n_obs": len(grp),
            "MAE_return": round(mae, 6),
            "RMSE_return": round(float(np.sqrt(mean_squared_error(actual, pred))), 6),
            "direction_pct": round(d_acc, 2) if not np.isnan(d_acc) else None,
            "skill_vs_zero": round(1 - mae / zero_mae, 4) if zero_mae > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ozellik onemi
# ---------------------------------------------------------------------------
def feature_importance_report(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost ozellik onemlerini raporlar (purge edilmis egitim verisi)."""
    test_mask = df["Date"] >= pd.Timestamp(TEST_START)
    test_indices = df.index[test_mask]
    if len(test_indices) == 0:
        return pd.DataFrame({"feature": FEATURES, "importance": 0.0})

    train_df = _purge_train(df, test_indices[0])
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(train_df[FEATURES].values, train_df[TARGET].values)
    fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Yapisal sizinti kontrolleri
# ---------------------------------------------------------------------------
def run_leakage_checks(df: pd.DataFrame):
    """Hedef ve ozellik zamanlamasini yapisal olarak dogrular.

    Korelasyon degil, veri hizalamasi kontrol edilir.
    """
    prices = df[PRICE_COL].values
    n = len(prices)

    # Kontrol 1: target[t] = log(P[t+1]/P[t]) gercekten saglanyor mu?
    expected = np.log(prices[1:] / prices[:-1])     # satirlar 0..n-2
    recorded = df[TARGET].values[:-1]                # satirlar 0..n-2
    diff = np.abs(expected - recorded)
    max_diff = float(diff.max()) if len(diff) > 0 else 0.0
    if max_diff > 1e-6:
        raise ValueError(
            f"SIZINTI RISKI: target_next_return fiyatlarla tutarsiz "
            f"(max sapma: {max_diff:.8f}). Hedef hesaplamasi hatali olabilir."
        )
    print(f"  [OK] target[t] = log(P[t+1]/P[t]) dogrulandi "
          f"({n-1} satir, max sapma: {max_diff:.2e})")

    # Kontrol 2: gold_return[t] = log(P[t]/P[t-1]) mi?
    expected_ret = np.log(prices[1:] / prices[:-1])  # satirlar 1..n-1 icin
    recorded_ret = df["gold_return"].values[1:]
    diff_ret = np.abs(expected_ret - recorded_ret)
    max_diff_ret = float(diff_ret.max()) if len(diff_ret) > 0 else 0.0
    if max_diff_ret > RETURN_TOL:
        raise ValueError(
            f"gold_return fiyatlarla tutarsiz (max sapma: {max_diff_ret:.2e})."
        )
    print(f"  [OK] gold_return[t] = log(P[t]/P[t-1]) dogrulandi")
    ident = float(np.abs(df["gold_return"] - df["gold_usd_return"]
                         - df["usdtry_return"]).max())
    if ident > RETURN_TOL:
        raise ValueError(f"gold_return != gold_usd_return + usdtry_return "
                         f"(max sapma: {ident:.2e}).")
    print(f"  [OK] gold_return = gold_usd_return + usdtry_return (max sapma: {ident:.1e})")

    # Kontrol 3: return_lag_1[t] = gold_return[t-1] mi?
    lag1 = df["return_lag_1"].values[1:]   # satirlar 1..n-1
    ret0 = df["gold_return"].values[:-1]   # satirlar 0..n-2
    diff_lag = np.abs(lag1 - ret0)
    valid_lag = ~np.isnan(diff_lag)
    if valid_lag.sum() > 0:
        max_lag_diff = float(diff_lag[valid_lag].max())
        if max_lag_diff > 1e-10:
            print(f"  [UYARI] return_lag_1 != gold_return.shift(1) "
                  f"(max sapma: {max_lag_diff:.2e})")
        else:
            print(f"  [OK] return_lag_1[t] = gold_return[t-1] dogrulandi")

    # Kontrol 4: Korelasyon (tanisal bilgi, karar degil)
    c1 = float(df[["gold_return", TARGET]].corr().iloc[0, 1])
    print(f"  [BILGI] gold_return <-> target korelasyonu: r = {c1:+.4f}")
    print(f"          (ayni degiskenin ardisik gunleri; sizinti degil)")


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------
def save_plots(pred_df: pd.DataFrame, best_ml_col: str, best_ml_name: str):
    """Walk-forward sonuc grafikleri.

    Ust panel:  Gercek ve tahmin edilen teorik gram altin fiyati (TL).
                Fiyatlar hedef tarihte (t+1) gosterilir.
    Alt panel:  Fiyat tahmin hatasi (TL), hedef tarihte.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib yuklu degil, grafik atlanacak.")
        return

    prices_t = pred_df["price_t"].values
    actual_ret = pred_df["actual_return"].values
    pred_ret = pred_df[best_ml_col].values

    actual_price_next = prices_t * np.exp(actual_ret)
    pred_price_next = prices_t * np.exp(pred_ret)

    target_dates = pd.to_datetime(pred_df["target_date"])
    valid = target_dates.notna()

    if valid.sum() == 0:
        print("  Gecerli hedef tarih yok, grafik atlanacak.")
        return

    t_dates = target_dates[valid]
    a_prices = actual_price_next[valid.values]
    p_prices = pred_price_next[valid.values]
    err_tl = p_prices - a_prices

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t_dates, a_prices, label="Gercek teorik fiyat",
                 lw=0.9, color="black")
    axes[0].plot(t_dates, p_prices, label=f"Tahmin ({best_ml_name})",
                 lw=0.9, alpha=0.8, color="tab:blue")
    axes[0].set_ylabel("Teorik gram altin (TL)")
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"Walk-Forward: {best_ml_name}\n"
        f"Hedef tarih araligi: {t_dates.iloc[0].date()} - {t_dates.iloc[-1].date()}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(t_dates, err_tl, width=1, alpha=0.4, color="gray")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("Fiyat hatasi (TL)")
    axes[1].set_xlabel("Hedef tarih (t+1)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "walk_forward_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grafik: {out_path}")


# ---------------------------------------------------------------------------
# Faz B deneyi: ozellik setlerinin dogrulama doneminde karsilastirilmasi
# ---------------------------------------------------------------------------
PHASE_B_SETS = ("base", "component", "compact")
PHASE_B_CANDIDATES = ("component", "compact")
PHASE_B_YEARS = (2022, 2023)
PHASE_B_MIN_ML_MODELS = 2       # kural: en az iki ML modelinde saglanmali
PHASE_B_EXPECTED_WINDOWS = 8
PHASE_B_WINS_REQUIRED = 5       # 8 pencerenin en az 5'i
PHASE_B_ALPHA = 0.05
PHASE_B_RULE = (
    "Bir aday set (component/compact), en az iki ML modelinde su uc kosulu "
    "saglarsa kabul edilir: (1) toplam dogrulama MAE'si ayni modelin base "
    "degerinden dusuk; (2) dogrulama pencerelerinin en az 5/8'inde base'den "
    "dusuk MAE; (3) 2022 ve 2023 yillarinin hicbirinde base'den yuksek MAE."
)


def required_window_wins(n_windows: int) -> int:
    """8 pencerede 5; farkli pencere sayisinda ayni oran (yukari yuvarlanir)."""
    if n_windows == PHASE_B_EXPECTED_WINDOWS:
        return PHASE_B_WINS_REQUIRED
    return int(math.ceil(PHASE_B_WINS_REQUIRED / PHASE_B_EXPECTED_WINDOWS * n_windows))


def holm_adjust(p_values) -> list:
    """Holm (adim adim asagi) duzeltmesi; duzeltilmis p-degerleri girdi sirasinda."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def window_metrics(res: dict, feature_set: str) -> pd.DataFrame:
    """Her degerlendirme penceresi ve model icin MAE (getiri) ve gozlem sayisi."""
    pred = res["predictions"]
    wid = np.asarray(res["window_of_row"])
    actual = pred["actual_return"].to_numpy(dtype=float)
    names = [c[len("pred_"):] for c in pred.columns if c.startswith("pred_")]
    rows = []
    for w in res["windows"]:
        mask = wid == w["window"]
        for name in names:
            p = pred[f"pred_{name}"].to_numpy(dtype=float)[mask]
            rows.append({
                "feature_set": feature_set, "window": w["window"],
                "eval_first_date": w["eval_first_date"], "eval_last_date": w["eval_last_date"],
                "n_obs": int(mask.sum()), "model": name,
                "MAE_return": float(np.mean(np.abs(actual[mask] - p))),
            })
    return pd.DataFrame(rows)


def evaluate_acceptance(results: dict, ml_models: list,
                        candidates=PHASE_B_CANDIDATES, years=PHASE_B_YEARS) -> pd.DataFrame:
    """Onceden tanimlanmis kabul kuralini yuvarlanmamis MAE'lerle uygular.

    results: {feature_set: walk_forward_evaluate sonucu}; 'base' zorunlu.
    Dondurur: aday set x ML modeli satirlari ve kosul sonuclari.
    Bir yil dogrulama verisinde yoksa o kosul saglanmamis sayilir.
    """
    base = results["base"]
    actual = base["predictions"]["actual_return"].to_numpy(dtype=float)
    wid = np.asarray(base["window_of_row"])
    year = pd.to_datetime(base["predictions"]["prediction_date"]).dt.year.to_numpy()
    windows = sorted(set(wid.tolist()))
    need = required_window_wins(len(windows))

    def abs_err(fs, model):
        return np.abs(actual - results[fs]["predictions"][f"pred_{model}"].to_numpy(dtype=float))

    rows = []
    for fs in candidates:
        for model in ml_models:
            e_c, e_b = abs_err(fs, model), abs_err("base", model)
            wins = sum(float(e_c[wid == w].mean()) < float(e_b[wid == w].mean()) for w in windows)
            row = {"feature_set": fs, "model": model,
                   "MAE_base": float(e_b.mean()), "MAE_set": float(e_c.mean())}
            row["mae_lower_than_base"] = bool(row["MAE_set"] < row["MAE_base"])
            row["windows_won"] = int(wins)
            row["windows_total"] = len(windows)
            row["windows_required"] = need
            row["windows_ok"] = bool(wins >= need)
            years_ok = True
            for y in years:
                m = year == y
                if not m.any():
                    row[f"year_{y}_not_worse"] = None
                    years_ok = False
                    continue
                ok = bool(float(e_c[m].mean()) <= float(e_b[m].mean()))
                row[f"year_{y}_not_worse"] = ok
                years_ok &= ok
            row["years_ok"] = bool(years_ok)
            row["passed"] = bool(row["mae_lower_than_base"] and row["windows_ok"] and years_ok)
            rows.append(row)
    return pd.DataFrame(rows)


def phase_b_significance(results: dict, ml_models: list) -> pd.DataFrame:
    """DM testleri: aday setin ML modeli vs ayni modelin base surumu ve vs Mean-Return.

    Holm duzeltmesi iki aile icin ayri uygulanir: (a) aday vs Mean-Return,
    (b) aday vs base. base setin ML modelleri vs Mean-Return baglam icin
    raporlanir (duzeltme ailesine girmez).
    """
    base_pred = results["base"]["predictions"]
    actual = base_pred["actual_return"].to_numpy(dtype=float)
    e_mean = actual - base_pred["pred_Mean-Return"].to_numpy(dtype=float)
    rows = []
    for fs in PHASE_B_SETS:
        pred = results[fs]["predictions"]
        for model in ml_models:
            e = actual - pred[f"pred_{model}"].to_numpy(dtype=float)
            r = diebold_mariano(e, e_mean)
            rows.append({"feature_set": fs, "model": model, "reference": "Mean-Return",
                         "family": "aday_vs_mean" if fs != "base" else "baglam", **r})
            if fs != "base":
                e_b = actual - base_pred[f"pred_{model}"].to_numpy(dtype=float)
                r = diebold_mariano(e, e_b)
                rows.append({"feature_set": fs, "model": model, "reference": f"base:{model}",
                             "family": "aday_vs_base", **r})
    out = pd.DataFrame(rows).rename(columns={"mean_loss_diff": "mean_abs_err_diff"})
    out["p_holm"] = np.nan
    for fam in ("aday_vs_mean", "aday_vs_base"):
        m = out["family"] == fam
        if m.any():
            out.loc[m, "p_holm"] = holm_adjust(out.loc[m, "p_value"])
    out["significant_lower_holm"] = (out["family"] != "baglam") & (out["dm_stat"] < 0) \
        & (out["p_holm"] < PHASE_B_ALPHA)
    return out


def run_phase_b(df: pd.DataFrame, out_dir: Path, data_path: Path = None) -> dict:
    """Faz B deneyini calistirir; yalnizca out_dir altina yazar.

    Uc set ayni dogrulama pencerelerinde degerlendirilir. Test donemi
    kullanilmaz. Satirlar, pencereler ve baseline tahminleri setler arasinda
    birebir ayni olmali; degilse deney durur (RuntimeError).
    """
    models_spec = get_models()
    ml_models = [n for n in models_spec if n not in BASELINE_NAMES]
    baselines = [n for n in models_spec if n in BASELINE_NAMES]

    results = {}
    for fs in PHASE_B_SETS:
        print(f"\n  Ozellik seti: {fs} ({len(FEATURE_SETS[fs])} ozellik)")
        results[fs] = walk_forward_evaluate(df, VALIDATION_START, TEST_START,
                                            label=f"dogrulama/{fs}", features=FEATURE_SETS[fs])

    # Butunluk: ayni satirlar, ayni pencereler, ayni baseline tahminleri
    ref = results["base"]
    for fs in PHASE_B_SETS[1:]:
        r = results[fs]
        if not ref["predictions"]["prediction_date"].equals(r["predictions"]["prediction_date"]):
            raise RuntimeError(f"{fs}: degerlendirme satirlari base ile ayni degil.")
        if [(w["n_train"], w["n_eval"]) for w in ref["windows"]] != \
                [(w["n_train"], w["n_eval"]) for w in r["windows"]]:
            raise RuntimeError(f"{fs}: pencere orneklemleri base ile ayni degil.")
        for b in baselines:
            if not np.array_equal(ref["predictions"][f"pred_{b}"].to_numpy(),
                                  r["predictions"][f"pred_{b}"].to_numpy()):
                raise RuntimeError(f"{fs}: {b} tahminleri base ile ayni degil.")

    metrics = pd.concat([r["metrics"].assign(feature_set=fs, n_features=len(FEATURE_SETS[fs]))
                         for fs, r in results.items()], ignore_index=True)
    metrics = metrics[["feature_set", "n_features"] + [c for c in metrics.columns
                                                       if c not in ("feature_set", "n_features")]]
    windows = pd.concat([window_metrics(r, fs) for fs, r in results.items()], ignore_index=True)
    annual = pd.concat([annual_comparison(r["predictions"], "validation").assign(feature_set=fs)
                        for fs, r in results.items()], ignore_index=True)
    annual = annual[["feature_set"] + [c for c in annual.columns if c != "feature_set"]]
    sig = phase_b_significance(results, ml_models)
    decision = evaluate_acceptance(results, ml_models)

    verdicts = {}
    for fs in PHASE_B_CANDIDATES:
        d = decision[decision["feature_set"] == fs]
        n_pass = int(d["passed"].sum())
        s_mean = sig[(sig["feature_set"] == fs) & (sig["family"] == "aday_vs_mean")]
        verdicts[fs] = {
            "ml_models_passed": n_pass,
            "ml_models_required": PHASE_B_MIN_ML_MODELS,
            "accepted": bool(n_pass >= PHASE_B_MIN_ML_MODELS),
            "models_significantly_better_than_mean_return_holm": s_mean.loc[
                s_mean["significant_lower_holm"], "model"].tolist(),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    windows.to_csv(out_dir / "window_metrics.csv", index=False)
    annual.to_csv(out_dir / "annual.csv", index=False)
    sig.to_csv(out_dir / "significance.csv", index=False)
    decision.to_csv(out_dir / "decision.csv", index=False)
    for fs, r in results.items():
        r["predictions"].to_csv(out_dir / f"validation_predictions_{fs}.csv", index=False)

    data_bytes = data_path.read_bytes() if data_path is not None and data_path.exists() else b""
    summary = {
        "experiment": "phase_b",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Bilesen ayristirmali suruklenme/oynaklik ozellikleri ML modellerine "
                    "dogrulama doneminde bilgi ekliyor mu?",
        "test_period_used": False,
        "target": TARGET_DEFINITION,
        "forecast_horizon": FORECAST_HORIZON,
        "periods": {"train_end": TRAIN_END, "validation_start": VALIDATION_START,
                    "validation_end_exclusive": TEST_START},
        "feature_sets": {fs: {"n_features": len(f), "features": f} for fs, f in FEATURE_SETS.items()},
        "component_feature_definitions": {
            "fx_mean_20": "usdtry_return.rolling(20, min_periods=20).mean()",
            "fx_mean_60": "usdtry_return.rolling(60, min_periods=20).mean()",
            "gold_usd_mean_20": "gold_usd_return.rolling(20, min_periods=20).mean()",
            "gold_usd_mean_60": "gold_usd_return.rolling(60, min_periods=20).mean()",
            "fx_vol_20": "usdtry_return.rolling(20, min_periods=20).std()  # ddof=1",
            "gold_usd_vol_20": "gold_usd_return.rolling(20, min_periods=20).std()  # ddof=1",
        },
        "warmup_rows": WARMUP_ROWS,
        "sample": {"rows_after_warmup": len(df),
                   "validation_rows": int(len(ref["predictions"])),
                   "identical_across_feature_sets": True,
                   "windows": ref["windows"]},
        "models": {"ml": ml_models, "baselines": baselines,
                   "explicit_params": {"Random Forest": {k: _json_safe(v) for k, v in RF_PARAMS.items()},
                                       "XGBoost": {k: _json_safe(v) for k, v in XGB_PARAMS.items()}},
                   "tuning": "yok (mevcut parametreler)"},
        "acceptance_rule": PHASE_B_RULE,
        "verdicts": verdicts,
        "dm_test": {
            "loss": "mutlak hata", "variance": "Newey-West HAC (Bartlett)",
            "lag_rule": "floor(4*(T/100)^(2/9))", "distribution": "asimptotik normal",
            "multiple_comparisons": "Holm; aday_vs_mean ve aday_vs_base aileleri ayri",
            "caveats": "Ic ice modellerde (base vs component) dagilim yaklasiktir; "
                       "kucuk orneklem duzeltmesi uygulanmaz; anlamlilik pratik "
                       "tahmin degeri kaniti degildir.",
        },
        "environment": {"python": sys.version.split()[0], "packages": _package_versions()},
        "dataset_md5": hashlib.md5(data_bytes).hexdigest() if data_bytes else "bulunamadi",
    }
    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_timestamp_utc": summary["run_timestamp_utc"],
                            "dataset_md5": summary["dataset_md5"],
                            "verdicts": verdicts}, ensure_ascii=False) + "\n")

    return {"results": results, "metrics": metrics, "windows": windows, "annual": annual,
            "significance": sig, "decision": decision, "verdicts": verdicts, "summary": summary}


def run_phase_b_main():
    """--experiment phase_b giris noktasi."""
    sep = "=" * 65
    print(f"\n{sep}\nFAZ B DENEYI - BILESEN OZELLIKLERI (YALNIZCA DOGRULAMA)\n{sep}")
    data_path = DATA_DIR / "model_dataset.csv"
    df = load_data(data_path)
    print(f"  {len(df)} gozlem | isinma satiri: {WARMUP_ROWS}")
    run_leakage_checks(df)
    out_dir = OUTPUT_DIR / "experiments" / "phase_b"
    res = run_phase_b(df, out_dir, data_path)

    show = res["metrics"][["feature_set", "n_features", "model", "MAE_return",
                           "skill_vs_mean", "direction_accuracy_pct", "up_fraction_pct"]]
    print("\n  DOGRULAMA METRIKLERI")
    print(show.to_string(index=False))
    cols = ["feature_set", "model", "MAE_base", "MAE_set", "windows_won",
            "windows_total", "years_ok", "passed"]
    print("\n  KABUL KURALI (aday set x ML modeli)")
    print(res["decision"][cols].to_string(index=False))
    for fs, v in res["verdicts"].items():
        status = "KABUL" if v["accepted"] else "RET"
        print(f"  {fs}: {status} ({v['ml_models_passed']}/{v['ml_models_required']} model); "
              f"Holm ile Mean-Return'u anlamli gecen: {v['models_significantly_better_than_mean_return_holm'] or 'yok'}")
    print("  Not: Test donemi kullanilmadi. Anlamlilik pratik tahmin degeri kaniti degildir.")
    print(f"\n  Ciktilar: {out_dir}\n{sep}\n")
    return res


# ---------------------------------------------------------------------------
# Deney kaydi
# ---------------------------------------------------------------------------
UNSET = "kodda belirtilmemis (kutuphane varsayilani)"


def _json_safe(value):
    if value is None:
        return UNSET
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _package_versions() -> dict:
    out = {}
    for pkg in ("pandas", "numpy", "scikit-learn", "xgboost", "matplotlib"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "bulunamadi"
    return out


def build_run_config(df: pd.DataFrame, data_path: Path, best_ml_name: str,
                     val_windows: list, test_windows: list) -> dict:
    """Calistirmayi yeniden uretmek icin gereken bilgileri toplar.

    Model parametreleri iki katmanda yazilir: 'explicit' kodda acikca
    verilenler; 'effective' kutuphanenin kullandigi tum parametreler (kodda
    verilmeyenler 'kodda belirtilmemis' olarak isaretlenir).
    """
    models = {}
    for name, spec in get_models().items():
        if isinstance(spec, str):
            rule = {"zero": "tahmin = 0",
                    "mean": "tahmin = egitim penceresindeki hedef ortalamasi",
                    "last": "tahmin = gold_return[t] (bugunun getirisi)"}[spec]
            models[name] = {"type": "baseline", "rule": rule}
        else:
            explicit = {"Random Forest": RF_PARAMS, "XGBoost": XGB_PARAMS}.get(name, {})
            models[name] = {
                "type": type(spec).__name__,
                "explicit_params": {k: _json_safe(v) for k, v in explicit.items()},
                "effective_params": {k: _json_safe(v) for k, v in sorted(spec.get_params().items())},
            }
    data_bytes = data_path.read_bytes() if data_path.exists() else b""
    test_rows = int((df["Date"] >= pd.Timestamp(TEST_START)).sum())
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": TARGET_DEFINITION,
        "forecast_horizon": FORECAST_HORIZON,
        "price_definition": "theoretical_gram_try = gold_usd_oz * usdtry / 31.1035 (teorik; fiziksel fiyat degil)",
        "periods": {
            "train": {"first_date": str(df["Date"].iloc[0].date()), "end": TRAIN_END,
                      "n_rows": int((df["Date"] <= pd.Timestamp(TRAIN_END)).sum())},
            "validation": {"start": VALIDATION_START, "end_exclusive": TEST_START,
                           "n_eval_rows": int(sum(w["n_eval"] for w in val_windows)),
                           "rule": "hedef tarihi TEST_START'tan once olan satirlar"},
            "test": {"start": TEST_START, "last_date": str(df["Date"].iloc[-1].date()),
                     "n_rows": test_rows},
        },
        "walk_forward": {
            "retrain_every": RETRAIN_EVERY, "window": "expanding",
            "purge": "her egitimde son satir cikarilir (1 gozlemlik muhafazakar ambargo)",
            "validation_windows": val_windows, "test_windows": test_windows,
        },
        "features": FEATURES,
        "warmup_rows": WARMUP_ROWS,
        "models": models,
        "random_seeds": {
            "Random Forest": RF_PARAMS["random_state"],
            "XGBoost": XGB_PARAMS["random_state"],
            "Linear Regression": "uygulanamaz (deterministik)",
            "baselines": "uygulanamaz (deterministik)",
            "numpy_global_seed": "ayarlanmadi (kod global rastgelelik kullanmiyor)",
            "feature_importance_xgboost": RANDOM_STATE,
        },
        "selection": {
            "best_model_for_plot_and_legacy_subperiod": best_ml_name,
            "selected_on": "dogrulama walk-forward MAE_return (test sonuclari kullanilmadi)",
        },
        "dm_test": {
            "loss": "mutlak hata", "variance": "Newey-West HAC (Bartlett)",
            "lag_rule": "floor(4*(T/100)^(2/9))", "distribution": "asimptotik normal",
            "references": ["Mean-Return", "Zero-Return"],
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "packages": _package_versions()},
        "dataset": {
            "file": "data/processed/model_dataset.csv",
            "md5": hashlib.md5(data_bytes).hexdigest() if data_bytes else "bulunamadi",
            "rows_after_warmup": len(df),
            "first_date": str(df["Date"].iloc[0].date()),
            "last_date": str(df["Date"].iloc[-1].date()),
        },
    }


# ---------------------------------------------------------------------------
# Ana
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Gram altin tahmini: egitim ve degerlendirme.")
    ap.add_argument("--experiment", choices=["phase_b"], default=None,
                    help="Izole deney calistir (varsayilan akisa ve results/ dosyalarina dokunmaz).")
    return ap.parse_args([] if argv is None else argv)


def main(argv=None):
    args = parse_args(argv)
    if args.experiment == "phase_b":
        run_phase_b_main()
        return

    sep = "=" * 65
    print(f"\n{sep}")
    print("GRAM ALTIN FIYAT TAHMINI - MODEL DEGERLENDIRME")
    print(sep)

    # 1. Veri
    print("\n[1/8] Veri yukleniyor...")
    data_path = DATA_DIR / "model_dataset.csv"
    df = load_data(data_path)
    n_train = int((df["Date"] < pd.Timestamp(TEST_START)).sum())
    n_test = int((df["Date"] >= pd.Timestamp(TEST_START)).sum())
    print(f"  {len(df)} gozlem | {df['Date'].iloc[0].date()}"
          f" - {df['Date'].iloc[-1].date()}")
    print(f"  Egitim+dogrulama: {n_train} | Test: {n_test}")

    # 2. Yapisal sizinti kontrolleri
    print("\n[2/8] Yapisal sizinti kontrolleri...")
    run_leakage_checks(df)

    cols = ["model", "MAE_return", "RMSE_return", "direction_accuracy_pct",
            "MAE_TL", "RMSE_TL", "skill_vs_zero", "skill_vs_mean", "up_fraction_pct"]

    # 3. Holdout (test, onceki surumle ayni)
    print("\n[3/8] Holdout degerlendirme (test)...")
    ho = holdout_evaluate(df)
    print(ho["metrics"][cols].to_string(index=False))

    # 4. Walk-forward test (onceki surumle ayni)
    print("\n[4/8] Walk-forward degerlendirme (test)...")
    wf = walk_forward_evaluate(df)
    print(wf["metrics"][cols].to_string(index=False))

    # 5. Walk-forward dogrulama
    print("\n[5/8] Walk-forward degerlendirme (dogrulama)...")
    val = walk_forward_evaluate(df, VALIDATION_START, TEST_START, label="dogrulama")
    print(val["metrics"][cols].to_string(index=False))

    # 6. Istatistiksel karsilastirma
    print("\n[6/8] Diebold-Mariano karsilastirmasi (mutlak hata)...")
    sig = pd.concat([significance_table(val["predictions"], "validation"),
                     significance_table(wf["predictions"], "test")], ignore_index=True)
    show = sig.copy()
    show["mean_abs_err_diff"] = show["mean_abs_err_diff"].map(lambda x: f"{x:+.2e}")
    show["dm_stat"] = show["dm_stat"].map(lambda x: f"{x:+.3f}")
    show["p_value"] = show["p_value"].map(lambda x: f"{x:.4f}")
    print(show.to_string(index=False))
    print("  Not: dm_stat < 0, modelin referanstan dusuk hataya sahip oldugunu gosterir.")
    print("  Istatistiksel anlamlilik tek basina pratik tahmin degeri kaniti degildir.")

    # 7. Yillik tablolar, legacy alt donem, ozellik onemi
    print("\n[7/8] Yillik karsilastirma ve ozellik onemi...")
    ml_rows = val["metrics"][~val["metrics"]["model"].isin(BASELINE_NAMES)]
    if not ml_rows.empty:
        best_ml_name = ml_rows.sort_values("MAE_return").iloc[0]["model"]
    else:
        best_ml_name = "XGBoost"
    best_ml_col = f"pred_{best_ml_name}"
    print(f"  Dogrulamada en dusuk MAE'li ML modeli: {best_ml_name}")

    annual = pd.concat([annual_comparison(val["predictions"], "validation"),
                        annual_comparison(wf["predictions"], "test")], ignore_index=True)
    sp = subperiod_analysis(wf["predictions"], best_ml_col)
    fi = feature_importance_report(df)
    print(f"  {best_ml_name} - test yillik performans (legacy tablo):")
    print(sp.to_string(index=False))

    # 8. Kaydet
    print("\n[8/8] Sonuclar kaydediliyor...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Legacy ciktilar (ayni dosya adlari; yeni sutunlar yalnizca sona eklendi)
    wf["metrics"].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    wf["predictions"].to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    ho["metrics"].to_csv(OUTPUT_DIR / "holdout_comparison.csv", index=False)
    sp.to_csv(OUTPUT_DIR / "subperiod_analysis.csv", index=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    # Faz A ciktilari
    val["metrics"].to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)
    val["predictions"].to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    sig.to_csv(OUTPUT_DIR / "significance.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "annual_comparison.csv", index=False)

    run_config = build_run_config(df, data_path, best_ml_name,
                                  val["windows"], wf["windows"])
    with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    summary = {
        "test_start": TEST_START,
        "test_end": str(df["Date"].iloc[-1].date()),
        "n_train": n_train,
        "n_test": n_test,
        "retrain_every": RETRAIN_EVERY,
        "evaluation": "walk-forward (expanding window, purged)",
        "target": "log(P[t+1] / P[t])",
        "features": FEATURES,
        "data_source": "Yahoo Finance (GC=F, USDTRY=X) -> teorik gram altin",
        "price_formula": "gold_usd_oz * usdtry / 31.1035",
        "note": "Teorik fiyat; fiziksel altin piyasa primini icermez.",
        "skill_reference": "Zero-Return baseline (yarin = bugun)",
        "purging": "Egitim kumesinin son satiri cikarilir (1 gozlemlik muhafazakar ambargo).",
        "walk_forward": wf["metrics"].to_dict(orient="records"),
        "holdout": ho["metrics"].to_dict(orient="records"),
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation": val["metrics"].to_dict(orient="records"),
        "skill_vs_mean_reference": "Mean-Return baseline (ayni gozlemlerdeki tahminleri)",
        "best_model_selected_on": "validation",
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    save_plots(wf["predictions"], best_ml_col, best_ml_name)

    print(f"\n  Tum dosyalar: {OUTPUT_DIR}")
    print(f"\n{sep}\nTAMAMLANDI\n{sep}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
