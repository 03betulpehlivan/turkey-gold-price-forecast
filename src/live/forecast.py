"""
Canli Tahmin Sistemi (LIVE_V1)
==============================
Adimlar AYRI calisir (scripts/gunluk_tahmin.ps1 bu sirayla cagirir):
  1) Arsiv   : python -m src.data.collect ; python -m src.data.collect_intraday
  2) Seriler : python -m src.data.build_panel --write --force --processed-dir data/processed/live
               python -m src.data.sync_dataset
  3) Puanlama: python -m src.live.forecast score
  4) Tahmin  : python -m src.live.forecast issue
  Karne      : python -m src.live.forecast card
Ana data/processed/market_panel.csv, FINAL_EXAM_V1 on kaydi ve sonuclari DEGISMEZ:
canli sistem GVZ'yi data/processed/live/market_panel.csv'den okur.

Hedef (tek tanim): ESANLI teorik gram altin P = ons (USD) x USD/TRY / 31.1035, ikisi de AYNI
saatlik cubuktan (NY 14:00 kurali, sync_dataset). Hedef = log(P[sonraki esanli gun] / P[bugun]).
Kuyumcu fiyati DEGILDIR. Eski gunluk seri (ons ~18:00 UTC x kur ~02-05 UTC) hicbir yerde kullanilmaz.

Nokta tahmini: Mean-Return, ESANLI serinin egitim penceresindeki hedeflerinin ortalamasi
  (genisleyen pencere, 63 gunde bir, 1 satir purge). Esanli gecmis ~2 yil ile sinirli oldugu icin
  suruklenme tahmini gurultuludur; standart hatasi kayda yazilir. "Yarin = bugun" de ayrica
  kaydedilir ve karnede karsilastirilir (2026 testinde ikisi pratikte esdegerdi).
Aralik: ayni merkez (mu) ve ayni hedef; sigma = bilesen oynakligi (GVZ[t-1] ons, GARCH kur,
  egitim korelasyonu); z kantilleri esanli serinin egitim artiklarindan; carpimsal ACI (eta 0.02),
  her gun ayni baslangictan deterministik yeniden hesaplanir.

Tahmin zamani kurali: tahmin yalnizca veri gununun NY takvim gununde ve o gunun 14:00 NY
  cubugu KAPANDIKTAN sonra kayda alinir (Turkiye saatiyle ABD yaz saatinde 21:00, kis saatinde
  22:00 sonrasi). Erken calistirma (cubuk kapanmamis) veya gec calistirma (ertesi NY gunu) reddedilir.

Kayit: research/live/forecast_log.csv (git ile izlenir). Her satirin tahmin alanlari icin
  row_hash tutulur; her yazmadan once butun satirlar dogrulanir, uyusmazlikta yazma reddedilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import math
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data import sync_dataset as sd
from src.models import adaptive as ad
from src.models import final_exam as fe
from src.models import implied_vol as ivx
from src.models import intervals as iv
from src.models import train as tr

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "research" / "live" / "forecast_log.csv"
SYNC_PATH = sd.OUT_DIR / "sync_daily.csv"
PANEL_PATH = ROOT / "data" / "processed" / "live" / "market_panel.csv"
SYSTEM_VERSION = "LIVE_V1"
NY = "America/New_York"
ETA = 0.02
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
MIN_TRAIN = 250
EXIT_ALREADY, EXIT_WINDOW = 3, 4
PRICE_BASIS = "esanli teorik gram: ons x USD/TRY ayni saatlik cubuk (NY 14:00), /31.1035; kuyumcu fiyati degil"
FORECAST_COLS = ["issued_at_utc", "system_version", "asof_date", "asof_bar_end_utc", "price_basis", "price_now",
                 "point_price", "point_return", "point_return_se", "point_price_rw", "q80_low", "q80_high",
                 "q95_low", "q95_high", "aci_k80", "aci_k95", "gvz_lag1", "train_days", "code_commit", "sync_md5"]
REALIZED_COLS = ["realized_date", "realized_price", "abs_err_TL", "rw_abs_err_TL", "in80", "in95"]
ALL_COLS = FORECAST_COLS + ["row_hash"] + REALIZED_COLS


class AlreadyIssued(Exception):
    """Bu veri tarihi icin tahmin zaten kayitli."""


class OutsideIssueWindow(Exception):
    """Tahmin zamani kurali saglanmadi (erken veya gec calistirma)."""


class LogTampered(RuntimeError):
    """Kayittaki tahmin alanlari row_hash ile uyusmuyor."""


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


TEXT_COLS = {"issued_at_utc", "system_version", "asof_date", "asof_bar_end_utc", "price_basis", "code_commit", "sync_md5"}


def row_hash(row) -> str:
    """Tahmin alanlarinin kanonik metninden md5. Tur SUTUNA gore belirlenir (CSV'de bos metin NaN
    olarak okunsa da ayni sonuc): metin -> '' veya str; sayi -> 10 anlamli basamak."""
    parts = []
    for c in FORECAST_COLS:
        v = row[c]
        if c in TEXT_COLS:
            parts.append("" if pd.isna(v) else str(v))
        else:
            parts.append("nan" if pd.isna(v) else f"{float(v):.10g}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Veri cercevesi ve tahmin
# ---------------------------------------------------------------------------
def build_frame(sync: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Esanli seri + GVZ[t-1] (panelde t'den KESIN once olan son gun). Panelin fiyatlari KULLANILMAZ."""
    df = sync.sort_values("Date").reset_index(drop=True).copy()
    p = df["theoretical_gram_try"].astype(float)
    df["r_gram"] = np.log(p / p.shift(1))
    df["r_gold"] = np.log(df["gold_usd_oz"] / df["gold_usd_oz"].shift(1))
    df["r_fx"] = np.log(df["usdtry"] / df["usdtry"].shift(1))
    df["y"] = df["r_gram"].shift(-1)                                   # hedef: SONRAKI esanli gunun getirisi
    g = panel[["Date", "gvz"]].dropna().sort_values("Date").rename(columns={"Date": "gvz_date", "gvz": "gvz_lag1"})
    df["Date"] = df["Date"].astype("datetime64[ns]")
    g["gvz_date"] = g["gvz_date"].astype("datetime64[ns]")
    df = pd.merge_asof(df, g, left_on="Date", right_on="gvz_date", direction="backward", allow_exact_matches=False)
    df = df.iloc[1:].reset_index(drop=True)
    if df["gvz_lag1"].isna().any():
        raise ValueError("GVZ[t-1] bulunamayan esanli gun var (canli panel guncel mi?).")
    return df


def compute(df: pd.DataFrame, eta: float = ETA) -> dict:
    """Son satir 'bugun' (hedefi bilinmiyor). Butun girdiler esanli seriden (+ GVZ[t-1])."""
    n = len(df)
    if n < MIN_TRAIN + 30:
        raise ValueError(f"Yetersiz esanli gecmis: {n} gun (en az {MIN_TRAIN + 30}).")
    y = df["y"].to_numpy(float)
    r_gold, r_fx = df["r_gold"].to_numpy(float), df["r_fx"].to_numpy(float)
    gvz_d = ivx.gvz_to_daily(df["gvz_lag1"])
    pos = list(range(MIN_TRAIN, n))
    mu_arr, sig_arr, z_rows = np.empty(len(pos)), np.empty(len(pos)), []
    k, last = 0, {}
    for w in tr.make_windows(pos, tr.RETRAIN_EVERY):
        ti = np.arange(w[0] - 1)                     # 1 satir purge: hedefi w[0] gunune ait satir egitime girmez
        y_tr = y[ti]
        mu = float(y_tr.mean())
        gf = iv.fit_garch(r_fx[ti])
        rho = float(np.corrcoef(r_gold[ti], r_fx[ti])[0, 1])
        sig = ivx.component_sigma(gvz_d, iv.garch_sigma(r_fx, gf), rho)
        z = (y_tr - mu) / sig[ti]
        z = z[np.isfinite(z)]
        for i in w:
            mu_arr[k], sig_arr[k] = mu, sig[i]
            z_rows.append(z)
            k += 1
        last = {"mu": mu, "mu_se": float(y_tr.std(ddof=1) / math.sqrt(len(y_tr))), "train_days": int(len(ti))}
    out = {**last, "sigma": float(sig_arr[-1])}
    yy = y[pos]
    for lvl in LEVELS:
        lo, hi, kpath, miss = ad.aci_sequence(yy, mu_arr, sig_arr, z_rows, 1 - lvl / 100, eta)
        out[f"lo{lvl}"], out[f"hi{lvl}"], out[f"k{lvl}"] = float(lo[-1]), float(hi[-1]), float(kpath[-1])
        out[f"backtest_cov{lvl}"] = float(1 - miss[:-1].mean())
    p_prev = df["theoretical_gram_try"].to_numpy(float)[pos[:-1]]
    out["backtest_mae_TL"] = float(np.mean(np.abs(p_prev * (np.exp(yy[:-1]) - np.exp(mu_arr[:-1])))))
    out["backtest_rw_mae_TL"] = float(np.mean(np.abs(p_prev * (np.exp(yy[:-1]) - 1))))
    out["backtest_days"] = len(pos) - 1
    out["backtest_period"] = [str(df["Date"].iat[pos[0]].date()), str(df["Date"].iat[pos[-2]].date())]
    return out


def check_issue_window(asof_date: pd.Timestamp, asof_bar_end_utc: pd.Timestamp, now_utc: pd.Timestamp):
    """Tahmin, veri gununun NY takvim gununde ve o gunun cubugu kapandiktan sonra yapilmali."""
    if now_utc < asof_bar_end_utc:
        raise OutsideIssueWindow(f"Erken: {asof_date.date()} cubugu {asof_bar_end_utc} UTC'de kapaniyor.")
    if now_utc.tz_convert(NY).date() != asof_date.date():
        raise OutsideIssueWindow(f"Gec: veri gunu {asof_date.date()}, simdi NY tarihi {now_utc.tz_convert(NY).date()}. "
                                 "Bu gun icin tahmin kaydedilemez (hedef gunun bilgisi kismen biliniyor olabilir).")


# ---------------------------------------------------------------------------
# Kayit
# ---------------------------------------------------------------------------
def read_log(path: Path = None) -> pd.DataFrame:
    path = Path(path or LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLS)
    return pd.read_csv(path, dtype={"asof_date": str, "realized_date": str, "code_commit": str,
                                    "issued_at_utc": str, "asof_bar_end_utc": str})


def verify_log(log: pd.DataFrame):
    for i, r in log.iterrows():
        if row_hash(r) != r["row_hash"]:
            raise LogTampered(f"Kayit satiri {i} ({r['asof_date']}) tahmin alanlari degistirilmis; yazma reddedildi.")


def _write_log(log: pd.DataFrame, path: Path):
    verify_log(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    log[ALL_COLS].to_csv(tmp, index=False)
    tmp.replace(path)


def issue(sync_path: Path = None, panel_path: Path = None, log_path: Path = None, now_utc=None) -> dict:
    sync_path, panel_path, log_path = Path(sync_path or SYNC_PATH), Path(panel_path or PANEL_PATH), Path(log_path or LOG_PATH)
    log = read_log(log_path)
    verify_log(log)
    sync = sd.load_sync_daily(sync_path)
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    df = build_frame(sync, panel)
    asof, bar_end = df["Date"].iat[-1], pd.Timestamp(df["bar_end_utc"].iat[-1])
    if (log["asof_date"].astype(str) == str(asof.date())).any():
        raise AlreadyIssued(f"{asof.date()} verisi icin tahmin zaten kayitli.")
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp(datetime.now(timezone.utc))
    check_issue_window(asof, bar_end, now)
    c = compute(df)
    p = float(df["theoretical_gram_try"].iat[-1])
    row = {"issued_at_utc": now.isoformat(), "system_version": SYSTEM_VERSION, "asof_date": str(asof.date()),
           "asof_bar_end_utc": str(bar_end), "price_basis": PRICE_BASIS, "price_now": p,
           "point_price": p * math.exp(c["mu"]), "point_return": c["mu"], "point_return_se": c["mu_se"],
           "point_price_rw": p, "q80_low": p * math.exp(c["lo80"]), "q80_high": p * math.exp(c["hi80"]),
           "q95_low": p * math.exp(c["lo95"]), "q95_high": p * math.exp(c["hi95"]),
           "aci_k80": c["k80"], "aci_k95": c["k95"], "gvz_lag1": float(df["gvz_lag1"].iat[-1]),
           "train_days": c["train_days"], "code_commit": _commit(), "sync_md5": _md5(sync_path),
           "realized_date": "", **{k: np.nan for k in REALIZED_COLS if k != "realized_date"}}
    row["row_hash"] = row_hash(pd.Series(row))
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    _write_log(log, log_path)
    return {"row": row, "backtest": c}


def score(sync_path: Path = None, log_path: Path = None) -> pd.DataFrame:
    """Gerceklesen fiyatlari doldurur; tahmin alanlari ve row_hash DEGISMEZ (dogrulanir)."""
    sync_path, log_path = Path(sync_path or SYNC_PATH), Path(log_path or LOG_PATH)
    log = read_log(log_path)
    if log.empty:
        return log
    verify_log(log)
    before = log[FORECAST_COLS + ["row_hash"]].astype(str).copy()
    sync = sd.load_sync_daily(sync_path)
    dates = sync["Date"].to_numpy()
    for i, r in log.iterrows():
        if isinstance(r["realized_date"], str) and r["realized_date"]:
            continue
        later = np.where(dates > np.datetime64(pd.Timestamp(r["asof_date"])))[0]
        if not len(later):
            continue
        j = later[0]
        real = float(sync["theoretical_gram_try"].iat[j])
        log.loc[i, "realized_date"] = str(pd.Timestamp(dates[j]).date())
        log.loc[i, "realized_price"] = real
        log.loc[i, "abs_err_TL"] = abs(real - float(r["point_price"]))
        log.loc[i, "rw_abs_err_TL"] = abs(real - float(r["point_price_rw"]))
        log.loc[i, "in80"] = float(r["q80_low"] <= real <= r["q80_high"])
        log.loc[i, "in95"] = float(r["q95_low"] <= real <= r["q95_high"])
    if not log[FORECAST_COLS + ["row_hash"]].astype(str).equals(before):
        raise LogTampered("Puanlama tahmin alanlarini degistirdi; kayit yazilmadi.")
    _write_log(log, log_path)
    return log


def scorecard(log: pd.DataFrame) -> dict:
    s = log[pd.to_numeric(log["realized_price"], errors="coerce").notna()]
    n = len(s)
    if n == 0:
        return {"n": 0}
    e, rw = s["abs_err_TL"].astype(float), s["rw_abs_err_TL"].astype(float)
    k80, k95 = int(s["in80"].astype(float).sum()), int(s["in95"].astype(float).sum())
    return {"n": n, "MAE_TL": float(e.mean()), "MedAE_TL": float(e.median()), "RW_MAE_TL": float(rw.mean()),
            "RW_MedAE_TL": float(rw.median()),
            **{f"share_le_{t:g}TL_pct": float(100 * (e <= t).mean()) for t in (1, 5, 10)},
            "cov80": k80 / n, "cov80_wilson95": fe.wilson(k80, n), "cov95": k95 / n, "cov95_wilson95": fe.wilson(k95, n)}


def print_card(card: dict):
    if not card["n"]:
        print("  CANLI KARNE: henuz puanlanmis tahmin yok.")
        return
    print(f"  CANLI KARNE ({card['n']} tahmin) | Mean-Return MAE {card['MAE_TL']:.2f} TL, medyan {card['MedAE_TL']:.2f} "
          f"| yarin=bugun MAE {card['RW_MAE_TL']:.2f}, medyan {card['RW_MedAE_TL']:.2f} | <=1TL %{card['share_le_1TL_pct']:.1f} "
          f"<=5TL %{card['share_le_5TL_pct']:.1f} <=10TL %{card['share_le_10TL_pct']:.1f}")
    print(f"  Kapsam %80 {card['cov80']:.3f} {[round(v, 3) for v in card['cov80_wilson95']]} | "
          f"%95 {card['cov95']:.3f} {[round(v, 3) for v in card['cov95_wilson95']]}")


def print_forecast(res: dict):
    r, b = res["row"], res["backtest"]
    print(f"  Veri: {r['asof_date']} (cubuk bitisi {r['asof_bar_end_utc']}) | {PRICE_BASIS}")
    print(f"  Bugunku esanli fiyat : {r['price_now']:.2f} TL")
    print(f"  Yarin (Mean-Return)  : {r['point_price']:.2f} TL  (suruklenme {1e4 * r['point_return']:.1f} bp, "
          f"standart hata {1e4 * r['point_return_se']:.1f} bp, {r['train_days']} gunluk esanli gecmis)")
    print(f"  Yarin (yarin=bugun)  : {r['point_price_rw']:.2f} TL")
    print(f"  %80 aralik           : {r['q80_low']:.2f} - {r['q80_high']:.2f} TL")
    print(f"  %95 aralik           : {r['q95_low']:.2f} - {r['q95_high']:.2f} TL")
    print(f"  Gecmis simulasyon (esanli seri, {b['backtest_period'][0]} .. {b['backtest_period'][1]}, {b['backtest_days']} gun; "
          f"BAGIMSIZ DEGIL): MAE {b['backtest_mae_TL']:.2f} TL (yarin=bugun {b['backtest_rw_mae_TL']:.2f}) | "
          f"kapsam %80 {b['backtest_cov80']:.3f}, %95 {b['backtest_cov95']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canli tahmin (LIVE_V1).")
    ap.add_argument("command", choices=["score", "issue", "card"])
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 72 + f"\nCANLI TAHMIN - {SYSTEM_VERSION} - {args.command.upper()}\n" + "=" * 72)
    try:
        if args.command == "score":
            log = score()
            print(f"  Puanlama tamam. Kayitli tahmin: {len(log)}")
            print_card(scorecard(log))
            return 0
        if args.command == "card":
            log = read_log()
            verify_log(log)
            print_card(scorecard(log))
            return 0
        print_forecast(issue())
        print(f"  Kayit: {LOG_PATH}")
        return 0
    except AlreadyIssued as e:
        print(f"  BILGI: {e}")
        return EXIT_ALREADY
    except OutsideIssueWindow as e:
        print(f"  REDDEDILDI: {e}")
        return EXIT_WINDOW
    except LogTampered as e:
        print(f"  KAYIT BUTUNLUGU HATASI: {e}")
        return 1
    except (FileNotFoundError, ValueError, sd.IntradayError) as e:
        print(f"  HATA: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
