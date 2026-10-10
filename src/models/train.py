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

Purging (egitim etiketi temizleme)
----------------------------------
Egitim kumesinin son satirinin hedefi (target_next_return) test doneminin
ilk fiyat hareketini icerir: target[t_son] = log(P[t_test_ilk] / P[t_son]).
Bu satir egitimden cikarilir (purge). Ayni mantik walk-forward pencerelerinde
ve ozellik onemi raporunda da uygulanir. Purge edilen satir sayisi her
calistirmada raporlanir.
"""

from pathlib import Path
import json
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
OUTPUT_DIR = DATA_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = "2023-10-26"
RETRAIN_EVERY = 63

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

REQUIRED_COLS = [
    "Date", "gold_usd_oz", "usdtry", PRICE_COL,
    "gold_return", "gold_usd_return", "usdtry_return", TARGET,
]

RANDOM_STATE = 42


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
    return out


# ---------------------------------------------------------------------------
# Veri yukleme
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """model_dataset.csv yukler, dogrular, ozellikleri hesaplar.

    Cikis DataFrame'i reset_index(drop=True) ile 0-tabanli ardisik
    indekse sahiptir. Tum iloc ve indeks islemleri bu varsayima dayanir.
    """
    path = DATA_DIR / "model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path}")

    df = (
        pd.read_csv(path, parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Sutun kontrolu
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sutunlar: {missing}")

    # Siralama ve mukerrer tarih
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Tarihler kronolojik sirali degil.")
    if df["Date"].duplicated().sum() > 0:
        raise ValueError("Mukerrer tarih bulundu.")

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

    df = build_features(df)

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  {dropped} satir NaN nedeniyle dustu (rolling pencere dolumu).")

    if len(df) == 0:
        raise ValueError("Tum satirlar NaN nedeniyle dustu; veri yetersiz.")

    return df


# ---------------------------------------------------------------------------
# Purging: egitim etiketlerinin test sinirina tasmasini onler
# ---------------------------------------------------------------------------
def _purge_train(df: pd.DataFrame, train_end_pos: int) -> pd.DataFrame:
    """Egitim kumesinden, hedefi test donemiyle cakisan satirlari cikarir.

    target_next_return[t] = log(P[t+1]/P[t]). Eger t+1 satiri egitim
    kumesinin disindaysa, bu etiket egitim aninda bilinmeyen bir fiyata
    dayanir ve cikarilmalidir.

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
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


BASELINE_NAMES = {"Zero-Return", "Mean-Return", "Last-Return"}


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def direction_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    """Yon dogrulugu: tahmin ve gercek ayni yone mi isaret ediyor?

    Sifir tahminler (ornegin Zero-Return baseline) 'yonsuz' sayilir ve
    hesaba katilmaz. Tum gunler yonsuzse NaN doner.
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


# ---------------------------------------------------------------------------
# Walk-forward degerlendirme
# ---------------------------------------------------------------------------
def walk_forward_evaluate(df: pd.DataFrame) -> dict:
    """Genisleyen pencereli walk-forward degerlendirme.

    Test donemi RETRAIN_EVERY is gunluk pencerelere bolunur.
    Her pencerede model, o pencerenin baslangicina kadar bilinen
    TUM veri uzerinde yeniden egitilir (expanding window).

    Purging: Her pencerede egitim kumesinin son satirinin hedefi,
    test penceresinin ilk fiyat hareketini iceriyorsa cikarilir.
    """
    test_mask = df["Date"] >= pd.Timestamp(TEST_START)
    test_indices = df.index[test_mask].tolist()

    if not test_indices:
        raise ValueError(f"Test doneminde gozlem yok (TEST_START={TEST_START}).")

    models_spec = get_models()
    all_preds = {name: [] for name in models_spec}
    all_actual, all_prices, all_dates, all_target_dates = [], [], [], []
    total_purged = 0

    # Pencereleri olustur
    windows = []
    i = 0
    while i < len(test_indices):
        end = min(i + RETRAIN_EVERY, len(test_indices))
        windows.append(test_indices[i:end])
        i = end

    print(f"  {len(windows)} pencere, ~{RETRAIN_EVERY} gun aralik, "
          f"toplam {len(test_indices)} test gozlemi")

    for w_idx, w_indices in enumerate(windows):
        # Purge uygulanmis egitim kumesi
        train_end_pos = w_indices[0]
        train_df = _purge_train(df, train_end_pos)
        n_purged = train_end_pos - len(train_df)
        total_purged += n_purged

        test_df = df.iloc[w_indices]

        if len(train_df) < 50:
            raise ValueError(
                f"Pencere {w_idx+1}: Purge sonrasi egitim verisi "
                f"yetersiz ({len(train_df)} satir)."
            )

        X_tr = train_df[FEATURES].values
        y_tr = train_df[TARGET].values
        X_te = test_df[FEATURES].values
        train_mean = float(y_tr.mean())

        all_actual.extend(test_df[TARGET].values)
        all_prices.extend(test_df[PRICE_COL].values)
        all_dates.extend(test_df["Date"].tolist())
        all_target_dates.extend(_get_target_dates(df, w_indices))

        for name, spec in models_spec.items():
            pred = _predict(spec, test_df, train_mean, X_tr, y_tr, X_te)
            all_preds[name].extend(pred)

        print(f"    [{w_idx+1}/{len(windows)}] egitim={len(train_df)}"
              f"{f' (purge={n_purged})' if n_purged else ''}, "
              f"test={len(test_df)}, "
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

    return {"metrics": pd.DataFrame(rows), "predictions": pred_df}


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

    if len(train_df) < 50:
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

    return {"metrics": pd.DataFrame(rows), "predictions": pred_df}


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
    if max_diff_ret > 1e-6:
        print(f"  [UYARI] gold_return fiyatlarla tutarsiz "
              f"(max sapma: {max_diff_ret:.2e}). "
              f"Farkli kaynaklardan gelebilir (ons vs gram).")
    else:
        print(f"  [OK] gold_return[t] = log(P[t]/P[t-1]) dogrulandi")

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
# Ana
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 65
    print(f"\n{sep}")
    print("GRAM ALTIN FIYAT TAHMINI - MODEL DEGERLENDIRME")
    print(sep)

    # 1. Veri
    print("\n[1/6] Veri yukleniyor...")
    df = load_data()
    n_train = int((df["Date"] < pd.Timestamp(TEST_START)).sum())
    n_test = int((df["Date"] >= pd.Timestamp(TEST_START)).sum())
    print(f"  {len(df)} gozlem | {df['Date'].iloc[0].date()}"
          f" - {df['Date'].iloc[-1].date()}")
    print(f"  Egitim: {n_train} | Test: {n_test}")

    # 2. Yapisal sizinti kontrolleri
    print("\n[2/6] Yapisal sizinti kontrolleri...")
    run_leakage_checks(df)

    # 3. Holdout
    print("\n[3/6] Holdout degerlendirme...")
    ho = holdout_evaluate(df)
    cols = ["model", "MAE_return", "RMSE_return",
            "direction_accuracy_pct", "MAE_TL", "RMSE_TL", "skill_vs_zero"]
    print("\n  HOLDOUT:")
    print(ho["metrics"][cols].to_string(index=False))

    # 4. Walk-forward
    print("\n[4/6] Walk-forward degerlendirme...")
    wf = walk_forward_evaluate(df)
    print("\n  WALK-FORWARD:")
    print(wf["metrics"][cols].to_string(index=False))

    # 5. Analiz
    print("\n[5/6] Alt donem ve ozellik onemi...")
    ml_rows = wf["metrics"][~wf["metrics"]["model"].isin(BASELINE_NAMES)]
    if not ml_rows.empty:
        best_ml_name = ml_rows.sort_values("MAE_return").iloc[0]["model"]
    else:
        best_ml_name = "XGBoost"
    best_ml_col = f"pred_{best_ml_name}"

    sp = subperiod_analysis(wf["predictions"], best_ml_col)
    print(f"\n  {best_ml_name} — yillik performans:")
    print(sp.to_string(index=False))

    fi = feature_importance_report(df)
    print(f"\n  XGBoost ozellik onemleri:")
    print(fi.to_string(index=False))

    # 6. Kaydet
    print("\n[6/6] Sonuclar kaydediliyor...")
    wf["metrics"].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    wf["predictions"].to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    ho["metrics"].to_csv(OUTPUT_DIR / "holdout_comparison.csv", index=False)
    sp.to_csv(OUTPUT_DIR / "subperiod_analysis.csv", index=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

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
        "purging": "Egitim etiketleri test sinirina tasan satirlar cikarildi.",
        "walk_forward": wf["metrics"].to_dict(orient="records"),
        "holdout": ho["metrics"].to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    save_plots(wf["predictions"], best_ml_col, best_ml_name)

    print(f"\n  Tum dosyalar: {OUTPUT_DIR}")
    print(f"\n{sep}\nTAMAMLANDI\n{sep}\n")


if __name__ == "__main__":
    main()
