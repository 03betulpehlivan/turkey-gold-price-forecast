"""
Sizinti Gosterimi: Literaturdeki Yuksek Basari Nereden Geliyor? (KESIF, betimleyici)
==================================================================================
Kullanim (proje kokunden):  python -m src.analysis.leakage_demo
Cikti: research/exploratory/leakage_demo/summary.json ve table.csv

Veri: fiziki gram altin kapanisi (TL/gr), 2006-01-01 .. 2022-12-31. 2023 ve sonrasi OKUNMADAN atilir
  (2023+ fiziki veri kilitli; bu calisma yalnizca gelistirme verisini kullanir).
Ayni modeller (Dogrusal, XGBoost, LSTM [src/models/lstm_np.py], yarin=bugun) uc kurulumda:
  K1 Literatur kurulumu: hedef YARININ FIYAT SEVIYESI; girdi son 20 fiyat; Min-Max olcekleme TUM veriyle;
     ornekler RASTGELE %80/%20 bolunur (ortusen pencereler test bilgisini egitime tasir). Olcut: R2, MAPE.
  K2 Kronolojik bolme: K1 ile ayni, ama ilk %80 egitim, son %20 test (olcekleme hala TUM veriyle).
  K3 Dogru kurulum: hedef ertesi gun LOG GETIRISI; girdi son 20 getiri; her test yili (2012..2022) icin
     yalnizca o yildan ONCEKI veriyle egitim ve olcekleme (genisleyen pencere). Olcut: TL mutlak hata,
     yarin=bugun'e gore iyilesme, DM (Newey-West + HLN), yon isabeti, getiri uzerinde OOS R2 (sifir tahmine gore).
     Ek taban: egitim donemi ortalama getirisi ("Ortalama").
Hiperparametreler ONCEDEN sabittir (asagida); arama yapilmaz. Tohum sabittir.
Bu bir hipotez testi degildir; sonuclar model secimi icin KULLANILMAZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import final_exam as fe
from src.models.lstm_np import LSTM

ROOT = Path(__file__).resolve().parents[2]
PHYS_PATH = ROOT / "data" / "processed" / "physical" / "physical_gold_daily.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "leakage_demo"
START, LOCK = "2006-01-01", "2023-01-01"
WIN, TEST_FRAC, SEED = 20, 0.2, 20260101
EVAL_YEARS = tuple(range(2012, 2023))
LSTM_CFG = {"hidden": 16, "epochs": 40, "batch": 128, "lr": 1e-3}
XGB_CFG = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": SEED,
           "n_jobs": 1, "verbosity": 0}
MODELS = ("Dogrusal", "XGBoost", "LSTM")


def load_prices(path: Path = None) -> pd.Series:
    df = pd.read_csv(Path(path or PHYS_PATH), usecols=["Date", "kapanis_tl_gr"], parse_dates=["Date"])
    df = df[(df["Date"] >= START) & (df["Date"] < LOCK) & (df["kapanis_tl_gr"] > 0)].dropna()
    s = df.sort_values("Date").drop_duplicates("Date").set_index("Date")["kapanis_tl_gr"].astype(float)
    if len(s) and s.index.max() >= pd.Timestamp(LOCK):
        raise RuntimeError("2023+ veri sizdi")
    return s


def windows(v: np.ndarray, win: int = WIN):
    X = np.lib.stride_tricks.sliding_window_view(v[:-1], win)
    return X.copy(), v[win:].copy()


def fit_predict(name: str, Xtr, ytr, Xte, lstm_cfg: dict):
    if name == "Dogrusal":
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
        return np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**XGB_CFG).fit(Xtr, ytr).predict(Xte).astype(float)
    if name == "LSTM":
        cfg = dict(lstm_cfg)
        m = LSTM(1, cfg.pop("hidden"), seed=SEED)
        m.fit(Xtr[..., None], ytr, seed=SEED, **cfg)
        return m.predict(Xte[..., None])
    raise ValueError(name)


def level_metrics(p, phat) -> dict:
    p, phat = np.asarray(p, float), np.asarray(phat, float)
    return {"r2_level": float(1 - np.sum((p - phat) ** 2) / np.sum((p - p.mean()) ** 2)),
            "mape_pct": float(100 * np.mean(np.abs(p - phat) / p)), "mae_tl": float(np.mean(np.abs(p - phat)))}


# ---------------------------------------------------------------------------
def setup_level(prices: pd.Series, chronological: bool, lstm_cfg: dict, models=MODELS) -> list:
    """K1 / K2: fiyat seviyesi, TUM veriyle Min-Max (sizintili, literaturdeki gibi)."""
    v = prices.to_numpy()
    lo, hi = v.min(), v.max()
    X, y = windows((v - lo) / (hi - lo))
    n = len(y)
    if chronological:
        cut = int(n * (1 - TEST_FRAC))
        tr, te = np.arange(cut), np.arange(cut, n)
    else:
        perm = np.random.default_rng(SEED).permutation(n)
        cut = int(n * (1 - TEST_FRAC))
        tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
    inv = lambda z: z * (hi - lo) + lo
    p_true = inv(y[te])
    rows = [{"model": "Yarin=bugun", **level_metrics(p_true, inv(X[te, -1]))}]
    for m in models:
        rows.append({"model": m, **level_metrics(p_true, inv(fit_predict(m, X[tr], y[tr], X[te], lstm_cfg)))})
    rw = rows[0]["mae_tl"]
    for r in rows:
        r["vs_rw_pct"] = float(100 * (1 - r["mae_tl"] / rw))
    return rows


def walk_forward_returns(prices: pd.Series, eval_years, lstm_cfg: dict, models=MODELS) -> pd.DataFrame:
    """K3: getiri; her yil icin yalnizca GECMISLE egitim ve olcekleme. Satir basina tahminler."""
    logp = np.log(prices.to_numpy())
    r = np.diff(logp)
    dates = prices.index[1:]
    X, y = windows(r)
    tgt_dates, p_prev = dates[WIN:], prices.to_numpy()[WIN:-1]
    out = []
    for Y in eval_years:
        tr = np.where(tgt_dates < pd.Timestamp(f"{Y}-01-01"))[0]
        te = np.where((tgt_dates >= pd.Timestamp(f"{Y}-01-01")) & (tgt_dates < pd.Timestamp(f"{Y + 1}-01-01")))[0]
        if len(te) == 0 or len(tr) < 250:
            continue
        mu, sd = y[tr].mean(), y[tr].std()                     # yalnizca egitim verisi
        z = lambda a: (a - mu) / sd
        part = pd.DataFrame({"date": tgt_dates[te], "year": Y, "p_prev": p_prev[te], "r": y[te],
                             "Yarin=bugun": 0.0, "Ortalama": mu})
        for m in models:
            part[m] = fit_predict(m, z(X[tr]), z(y[tr]), z(X[te]), lstm_cfg) * sd + mu
        out.append(part)
    return pd.concat(out, ignore_index=True)


def summarize_returns(pred: pd.DataFrame, models=MODELS) -> list:
    p_true = pred["p_prev"] * np.exp(pred["r"])
    err = lambda col: np.abs(p_true - pred["p_prev"] * np.exp(pred[col])).to_numpy()
    e_rw = err("Yarin=bugun")
    rows = []
    for m in ("Yarin=bugun", "Ortalama") + tuple(models):
        e = err(m)
        row = {"model": m, **level_metrics(p_true, pred["p_prev"] * np.exp(pred[m])),
               "vs_rw_pct": float(100 * (1 - e.mean() / e_rw.mean())),
               "oos_r2_return": float(1 - np.sum((pred["r"] - pred[m]) ** 2) / np.sum(pred["r"] ** 2))}
        if m != "Yarin=bugun":
            row["dm_p"] = fe.dm_hln(e, e_rw)["p_value"]
            nz = pred["r"] != 0
            row["direction_hit_pct"] = float(100 * np.mean(np.sign(pred.loc[nz, m]) == np.sign(pred.loc[nz, "r"])))
        rows.append(row)
    return rows


def run_all(prices: pd.Series, eval_years=EVAL_YEARS, lstm_cfg: dict = None, models=MODELS) -> dict:
    lstm_cfg = dict(lstm_cfg or LSTM_CFG)
    pred = walk_forward_returns(prices, eval_years, lstm_cfg, models)
    return {"nature": "KESIF (betimleyici); model secimi icin KULLANILMAZ",
            "data": {"first": str(prices.index.min().date()), "last": str(prices.index.max().date()), "n": int(len(prices))},
            "config": {"window": WIN, "test_frac": TEST_FRAC, "seed": SEED, "lstm": lstm_cfg, "xgboost": XGB_CFG,
                       "eval_years": [int(y) for y in sorted(pred["year"].unique())]},
            "K1_literatur_rastgele_bolme": setup_level(prices, False, lstm_cfg, models),
            "K2_kronolojik_seviye": setup_level(prices, True, lstm_cfg, models),
            "K3_dogru_walk_forward": summarize_returns(pred, models),
            "K3_n": int(len(pred))}


def table(res: dict) -> pd.DataFrame:
    rows = []
    for k in ("K1_literatur_rastgele_bolme", "K2_kronolojik_seviye", "K3_dogru_walk_forward"):
        for r in res[k]:
            rows.append({"kurulum": k, **r})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sizinti gosterimi (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    prices = load_prices()
    print("=" * 78 + "\nSIZINTI GOSTERIMI (KESIF) - fiziki gram altin, 2006-2022\n" + "=" * 78)
    res = run_all(prices)
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t = table(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    t.to_csv(out / "table.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["kurulum", "model", "r2_level", "mape_pct", "mae_tl", "vs_rw_pct", "oos_r2_return", "dm_p", "direction_hit_pct"]
        print(t[[c for c in cols if c in t.columns]].round(4).to_string(index=False))
    print(f"\n  Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
