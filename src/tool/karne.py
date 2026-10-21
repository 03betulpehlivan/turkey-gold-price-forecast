"""
Gram Altin Tahmin Karnesi (kullaniciya yonelik, basit)
=====================================================
Kullanim (proje kokunden):
  python -m src.tool.karne --egitim 2012-2019 --test 2020-2022
  python -m src.tool.karne --egitim 2012-2019 --test 2020-2022 --chronos      # Chronos kuruluysa
  python -m src.tool.karne yarin                                             # yarin icin tahmin karti

Soru: "Yarin gram altin hangi aralikta olacak ve gun sakin mi hareketli mi gececek?"
Seri: fiziki gram altin kapanisi (TL/gr, Borsa Istanbul; TCMB EVDS). Hedef: bir sonraki islem gunu kapanisi.
Kural: modeller YALNIZCA egitim yillariyla kurulur; test yillarinda parametreler sabit kalir, her gun yalnizca
  o gune kadar gerceklesen fiyatlar kullanilir (gelecek bilgisi yok).
Modeller:
  Yarin=bugun   tahmin = bugunku fiyat; aralik egitim donemindeki sabit oynaklikla (kiyas)
  GARCH         tahmin = bugunku fiyat; aralik GARCH(1,1) oynakligiyla (oynaklik kumelenmesini izler)
  ARIMA         ARIMA(1,0,1) getiri modeli; kendi (sabit genislikli) araligi
  XGBoost       son 10 gunun getirileri + 20 gunluk oynaklik; aralik egitim artiklarinin yuzdelikleriyle
  Chronos       (istege bagli) Amazon Chronos-Bolt; egitim gerektirmez, son 512 gunun fiyatindan aralik verir
Karne:
  Ortalama hata (TL ve %), "yarin = bugun"e gore fark;
  %80 ve %95 araligin gercekte tutma orani (hedef: 80 ve 95'e yakin), ortalama aralik genisligi;
  "Hareketli gun" = mutlak gunluk degisim > esik (varsayilan %1). Tahmin: P(hareketli) >= 0.5.
    Isabet, "her gun sakin" kuralinin isabeti, hareketli gunleri yakalama orani ve
    hava durumu tarzi guvenilirlik tablosu ("%30 dedigimiz gunlerin gercekte yuzde kaci hareketliydi?").
Yillara gore tablo, sakin ve kriz yillarini ayri gosterir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PATHS = [ROOT / "data" / "processed" / "live" / "physical_gold_daily.csv",
         ROOT / "data" / "processed" / "physical" / "physical_gold_daily.csv"]
OUT_DIR = ROOT / "research" / "tool"
Z80, Z95 = 1.2815515655446004, 1.959963984540054
ESIK = 1.0
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
LAGS, VOLWIN, CHRONOS_CTX = 10, 20, 512
MODELS = ("Yarin=bugun", "GARCH", "GARCH-t", "GJR-t", "EWMA", "GARCH-t+ACI", "ARIMA", "XGBoost")
PENCERELER = (250, 500, 750, 1000, 0)     # gun; 0 = genisleyen (egitim basindan bugune). Aylik yeniden egitim.
YENIDEN, ACI_GAMMA, EWMA_L = 21, 0.005, 0.94
CHRONOS_ID = "amazon/chronos-bolt-small"
ODAK = "GARCH-t+ACI"      # ayrintili tablolarda gosterilen model (onceden sabit)


def parse_years(s: str) -> tuple:
    a, _, b = s.partition("-")
    a, b = int(a), int(b or a)
    if a > b:
        raise ValueError(f"Yil araligi hatali: {s}")
    return a, b


def load_series(path: Path = None) -> pd.Series:
    p = Path(path) if path else next((q for q in PATHS if q.exists()), None)
    if p is None or not p.exists():
        raise FileNotFoundError("Fiziki altin verisi yok: " + ", ".join(str(q) for q in PATHS))
    df = pd.read_csv(p, usecols=["Date", "kapanis_tl_gr"], parse_dates=["Date"])
    df = df[df["kapanis_tl_gr"] > 0].dropna()
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")["kapanis_tl_gr"].astype(float)


def frame(prices: pd.Series) -> pd.DataFrame:
    """Satir: bilgi gunu t (bugun) ve hedef gun T (bir sonraki islem gunu)."""
    p = prices.to_numpy(float)
    return pd.DataFrame({"t": prices.index[:-1], "T": prices.index[1:], "P": p[:-1], "P1": p[1:],
                         "r": np.concatenate([[np.nan], np.diff(np.log(p))])[:-1],     # t'ye kadar bilinen getiri
                         "r1": np.diff(np.log(p))})                                      # hedef getiri


def split(df: pd.DataFrame, egitim: tuple, test: tuple) -> tuple:
    tr = df[(df["T"].dt.year >= egitim[0]) & (df["T"].dt.year <= egitim[1])]
    te = df[(df["T"].dt.year >= test[0]) & (df["T"].dt.year <= test[1])]
    if len(tr) < 250:
        raise ValueError(f"Egitim donemi cok kisa ({len(tr)} gun); en az 250 gun gerekli.")
    if len(te) == 0:
        raise ValueError("Test doneminde veri yok.")
    if te["T"].min() <= tr["T"].max():
        raise ValueError("Test donemi egitim doneminden SONRA olmali.")
    return tr, te


def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def _t_scale(nu):
    nu = np.asarray(nu, float)
    return np.where(np.isfinite(nu), np.sqrt(np.clip(nu - 2, 1e-9, None) / np.where(np.isfinite(nu), nu, 1.0)), 1.0)


def _cdf(x, nu):
    """Standart (birim varyansli) dagilimin CDF'i: nu NaN ise normal, degilse Student-t."""
    from scipy import stats
    x, nu = np.broadcast_arrays(np.asarray(x, float), np.asarray(nu, float))
    shape, x1, n1 = x.shape, x.ravel(), nu.ravel()
    out = stats.norm.cdf(x1)
    m = np.isfinite(n1)
    if m.any():
        out[m] = stats.t.cdf(x1[m] / _t_scale(n1[m]), n1[m])
    return out.reshape(shape) if shape else float(out[0])


def _ppf(p, nu):
    from scipy import stats
    p, nu = np.broadcast_arrays(np.asarray(p, float), np.asarray(nu, float))
    shape, p1, n1 = p.shape, p.ravel(), nu.ravel()
    out = stats.norm.ppf(p1)
    m = np.isfinite(n1)
    if m.any():
        out[m] = stats.t.ppf(p1[m], n1[m]) * _t_scale(n1[m])
    return out.reshape(shape) if shape else float(out[0])


def p_move(mu, sigma, esik, nu=np.nan):
    e = esik / 100.0
    return _cdf((-e - mu) / sigma, nu) + 1 - _cdf((e - mu) / sigma, nu)


def _out(te, mu, sigma=None, q=None, nu=np.nan):
    """Getiri ortalamasi + oynaklik (normal) VEYA dogrudan fiyat yuzdelikleri (q: 0.025,0.1,0.9,0.975)."""
    P = te["P"].to_numpy(float)
    mu = np.asarray(mu, float)
    o = pd.DataFrame({"T": te["T"].to_numpy(), "point": P * np.exp(mu)})
    if q is None:
        s = np.asarray(sigma, float)
        nu_a = np.broadcast_to(np.asarray(nu, float), s.shape)
        z80, z95 = _ppf(0.9, nu_a), _ppf(0.975, nu_a)
        o["lo80"], o["hi80"] = P * np.exp(mu - z80 * s), P * np.exp(mu + z80 * s)
        o["lo95"], o["hi95"] = P * np.exp(mu - z95 * s), P * np.exp(mu + z95 * s)
        o["mu"], o["sigma"], o["nu"] = mu, s, nu_a
    else:
        o["lo95"], o["lo80"], o["hi80"], o["hi95"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        o["mu"] = mu
        o["sigma"] = np.log(q[:, 2] / q[:, 1]) / (2 * Z80)
        o["nu"] = np.nan
    return o


def _pos(full, te):
    pos = full.index.get_indexer(te.index)
    if (pos < 1).any():
        raise ValueError("Test gunu icin onceki bilgi yok.")
    return pos


def m_rw(tr, te, full, prices=None):
    s = float(np.std(tr["r1"]))
    return _out(te, np.zeros(len(te)), np.full(len(te), s))


def m_garch(tr, te, full, prices=None):
    from src.models import intervals as iv
    gp = iv.fit_garch(tr["r1"].to_numpy(float))                   # parametreler yalnizca egitimle
    sig = iv.garch_sigma(full["r1"].to_numpy(float), gp)          # sig[i]: r1[0..i] ile, r1[i+1] icin
    return _out(te, np.zeros(len(te)), sig[_pos(full, te) - 1])


def m_arima(tr, te, full, prices=None):
    from statsmodels.tsa.arima.model import ARIMA
    res = ARIMA(tr["r1"].to_numpy(float), order=(1, 0, 1), trend="c").fit()
    pr = res.apply(full["r1"].to_numpy(float)).get_prediction()   # bir adim ileri; ayni parametreler
    pos = _pos(full, te)
    mu = np.asarray(pr.predicted_mean)[pos]
    se = np.sqrt(np.asarray(pr.var_pred_mean))[pos]
    return _out(te, mu, se)


def _xgb_features(df):
    r = df["r"].to_numpy(float)
    X = np.column_stack([np.roll(r, k) for k in range(LAGS)] +
                        [pd.Series(r).rolling(VOLWIN, min_periods=5).std().to_numpy()])
    X[:LAGS] = np.nan
    return X


def m_xgb(tr, te, full, prices=None):
    from xgboost import XGBRegressor
    X = _xgb_features(full)
    ptr = full.index.get_indexer(tr.index)
    Xtr, ytr = X[ptr], tr["r1"].to_numpy(float)
    ok = ~np.isnan(Xtr).any(axis=1)
    m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0, n_jobs=1)
    m.fit(Xtr[ok], ytr[ok])
    q = np.quantile(ytr[ok] - m.predict(Xtr[ok]), [0.025, 0.1, 0.9, 0.975])
    mu = m.predict(np.nan_to_num(X[_pos(full, te)]))
    P = te["P"].to_numpy(float)
    return _out(te, mu, q=np.column_stack([P * np.exp(mu + q[k]) for k in range(4)]))


def m_chronos(tr, te, full, prices, pipeline=None):
    try:
        import torch
        as_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    except ImportError:
        if pipeline is None:
            raise RuntimeError("Chronos icin: pip install chronos-forecasting (PyTorch ile birlikte kurulur)")
        as_tensor = np.asarray                                        # yalnizca testlerde (sahte pipeline)
    if pipeline is None:
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_ID, device_map="cpu", torch_dtype=torch.float32)
    pos = prices.index.get_indexer(pd.to_datetime(te["t"]))
    if (pos < 0).any():
        raise ValueError("Chronos: bilgi gunu fiyat serisinde yok.")
    v = prices.to_numpy(float)
    ctx = [as_tensor(v[max(0, i + 1 - CHRONOS_CTX): i + 1]) for i in pos]       # bilgi gunu dahil, sonrasi yok
    out = []
    for k in range(0, len(ctx), 256):
        q, _ = pipeline.predict_quantiles(ctx[k:k + 256], prediction_length=1, quantile_levels=[0.025, 0.1, 0.5, 0.9, 0.975])
        out.append(np.asarray(q)[:, 0, :])
    q = np.vstack(out)
    P = te["P"].to_numpy(float)
    return _out(te, np.log(q[:, 2] / P), q=q[:, [0, 1, 3, 4]])


def _arch(x, gjr, dist):
    from arch import arch_model
    return arch_model(x, mean="Zero", vol="GARCH", p=1, o=1 if gjr else 0, q=1, dist=dist, rescale=False)


def vol_family(tr, te, full, gjr=False, dist="t", pencere=None, yeniden=YENIDEN):
    """GARCH ailesi (yuzde getiri). pencere None: egitimle bir kez; 0: genisleyen; N: son N gun.
    Yeniden egitimde her blok icin parametreler YALNIZCA bloktan onceki gunlerle tahmin edilir."""
    x = 100 * full["r1"].to_numpy(float)
    pos = _pos(full, te)
    ptr = full.index.get_indexer(tr.index)
    s0 = int(ptr.min())
    sig, nu = np.empty(len(pos)), np.full(len(pos), np.nan)
    if pencere is None:
        blocks = [(int(pos[0]), int(pos[-1]) + 1, s0, int(ptr.max()) + 1)]
    else:
        blocks = []
        for b in range(int(pos[0]), int(pos[-1]) + 1, yeniden):
            e = min(b + yeniden, int(pos[-1]) + 1)
            fs = s0 if pencere == 0 else max(0, b - pencere)
            blocks.append((b, e, fs, b))
    k = 0
    for b, e, fs, fe in blocks:
        res = _arch(x[fs:fe], gjr, dist).fit(disp="off", show_warning=False)
        cv = np.asarray(_arch(x[fs:e], gjr, dist).fix(res.params).conditional_volatility)   # cv[i]: x[:i] ile
        n = e - b
        sig[k:k + n] = cv[b - fs: e - fs] / 100.0
        nu[k:k + n] = res.params.get("nu", np.nan)
        k += n
    return _out(te, np.zeros(len(te)), sig, nu=nu)


def m_garch_t(tr, te, full, prices=None, pencere=None):
    return vol_family(tr, te, full, gjr=False, dist="t", pencere=pencere)


def m_gjr_t(tr, te, full, prices=None, pencere=None):
    return vol_family(tr, te, full, gjr=True, dist="t", pencere=pencere)


def m_ewma(tr, te, full, prices=None):
    x = full["r1"].to_numpy(float)
    ptr = full.index.get_indexer(tr.index)
    v = float(np.var(x[ptr]))
    var = np.empty(len(x))
    for i in range(len(x)):
        var[i] = v                                    # var[i]: x[:i] ile
        v = EWMA_L * v + (1 - EWMA_L) * x[i] ** 2
    return _out(te, np.zeros(len(te)), np.sqrt(var[_pos(full, te)]))


def aci(te, o, gamma=ACI_GAMMA):
    """Uyarlamali kalibrasyon: aralik iskaladikca genisler, tuttukca daralir. Her gun yalnizca ONCEKI
    gunlerin iskalamalarini kullanir (o gunun gerceklesen fiyati, o gunun araligini etkilemez)."""
    y, P = te["P1"].to_numpy(float), te["P"].to_numpy(float)
    mu, s, nu = (o[c].to_numpy(float) for c in ("mu", "sigma", "nu"))
    o = o.copy()
    for lvl, lo, hi in ((0.20, "lo80", "hi80"), (0.05, "lo95", "hi95")):
        a = lvl
        L, H = np.empty(len(y)), np.empty(len(y))
        for i in range(len(y)):
            z = float(_ppf(1 - a / 2, nu[i]))
            L[i], H[i] = P[i] * np.exp(mu[i] - z * s[i]), P[i] * np.exp(mu[i] + z * s[i])
            miss = 0.0 if (np.isnan(y[i]) or L[i] <= y[i] <= H[i]) else 1.0
            a = float(np.clip(a + gamma * (lvl - miss), 1e-4, 0.999))
        o[lo], o[hi] = L, H
    return o


def m_garch_t_aci(tr, te, full, prices=None, pencere=None):
    return aci(te, m_garch_t(tr, te, full, prices, pencere))


FUNCS = {"Yarin=bugun": m_rw, "GARCH": m_garch, "GARCH-t": m_garch_t, "GJR-t": m_gjr_t, "EWMA": m_ewma,
         "GARCH-t+ACI": m_garch_t_aci, "ARIMA": m_arima, "XGBoost": m_xgb, "Chronos": m_chronos}
PENCERELI = {"GARCH-t", "GJR-t", "GARCH-t+ACI"}


def score(te: pd.DataFrame, o: pd.DataFrame, rw_mae: float, esik: float = ESIK) -> dict:
    y, P = te["P1"].to_numpy(float), te["P"].to_numpy(float)
    err = np.abs(o["point"].to_numpy() - y)
    cov80 = np.mean((y >= o["lo80"]) & (y <= o["hi80"]))
    cov95 = np.mean((y >= o["lo95"]) & (y <= o["hi95"]))
    move = np.abs(te["r1"].to_numpy(float)) > esik / 100.0
    nu = o["nu"].to_numpy(float) if "nu" in o else np.nan
    p = p_move(o["mu"].to_numpy(float), o["sigma"].to_numpy(float), esik, nu)
    pred = p >= 0.5
    rel = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            rel.append({"tahmin_araligi": f"%{int(a * 100)}-{min(int(b * 100), 100)}", "gun": int(m.sum()),
                        "ortalama_tahmin_pct": float(100 * p[m].mean()), "gerceklesen_pct": float(100 * move[m].mean())})
    return {"gun": int(len(te)), "ortalama_hata_tl": float(err.mean()), "ortalama_hata_pct": float(100 * np.mean(err / y)),
            "rw_gore_pct": float(100 * (1 - err.mean() / rw_mae)),
            "aralik80_tutma_pct": float(100 * cov80), "aralik95_tutma_pct": float(100 * cov95),
            "aralik80_genislik_pct": float(100 * np.mean((o["hi80"] - o["lo80"]) / P)),
            "hareketli_isabet_pct": float(100 * np.mean(pred == move)),
            "hep_sakin_isabet_pct": float(100 * np.mean(~move)),
            "hareketli_yakalama_pct": float(100 * pred[move].mean()) if move.any() else float("nan"),
            "hareketli_brier": float(np.mean((p - move) ** 2)), "guvenilirlik": rel}


def yearly(te: pd.DataFrame, o: pd.DataFrame) -> list:
    y = te["P1"].to_numpy(float)
    d = pd.DataFrame({"yil": pd.to_datetime(te["T"]).dt.year.to_numpy(), "hata": np.abs(o["point"].to_numpy() - y) / y,
                      "in80": (y >= o["lo80"].to_numpy()) & (y <= o["hi80"].to_numpy())})
    return [{"yil": int(k), "gun": int(len(g)), "ortalama_hata_pct": float(100 * g["hata"].mean()),
             "aralik80_tutma_pct": float(100 * g["in80"].mean())} for k, g in d.groupby("yil")]


def run(prices: pd.Series, egitim: tuple, test: tuple, models=MODELS, esik: float = ESIK, chronos_pipeline=None,
        pencere=None) -> dict:
    df = frame(prices)
    tr, te = split(df, egitim, test)
    full = df.loc[tr.index.min(): te.index.max()]                # egitim basindan test sonuna kesintisiz gecmis
    outs = {}
    for m in models:
        f = FUNCS[m]
        if m == "Chronos":
            outs[m] = f(tr, te, full, prices, chronos_pipeline)
        elif m in PENCERELI:
            outs[m] = f(tr, te, full, prices, pencere=pencere)
        else:
            outs[m] = f(tr, te, full, prices)
    rw_mae = float(np.mean(np.abs(te["P"].to_numpy(float) - te["P1"].to_numpy(float))))
    return {"seri": "fiziki gram altin (TL/gr)", "egitim": list(egitim), "test": list(test), "esik_pct": esik,
            "pencere": pencere,
            "egitim_gun": int(len(tr)), "test_gun": int(len(te)),
            "modeller": {m: score(te, o, rw_mae, esik) for m, o in outs.items()},
            "yillar": {m: yearly(te, o) for m, o in outs.items()}}


def gunluk(prices: pd.Series, egitim: tuple, test: tuple, model: str = ODAK, pencere=None) -> pd.DataFrame:
    """Test gunleri icin gunluk tahmin tablosu: hedef gun, gerceklesen fiyat, tahmin ve araliklar."""
    df = frame(prices)
    tr, te = split(df, egitim, test)
    full = df.loc[tr.index.min(): te.index.max()]
    f = FUNCS[model]
    o = f(tr, te, full, prices, pencere=pencere) if model in PENCERELI else f(tr, te, full, prices)
    out = o[["T", "point", "lo80", "hi80", "lo95", "hi95"]].copy()
    out.insert(1, "gercek", te["P1"].to_numpy(float))
    out["p_hareketli"] = p_move(o["mu"].to_numpy(float), o["sigma"].to_numpy(float), ESIK, o["nu"].to_numpy(float))
    return out.reset_index(drop=True)


def tomorrow(prices: pd.Series, egitim: tuple, model: str = ODAK, esik: float = ESIK, pencere=None) -> dict:
    """Son kapanisa gore bir sonraki islem gunu icin tahmin karti. Egitim yillarindan sonraki gunler de (hedefi
    bilinen) gecmis olarak islenir; ACI ve pencere boylece guncel kalir. Kart son (hedefi bilinmeyen) satirdir."""
    df = frame(prices)
    last = pd.DataFrame({"t": [prices.index[-1]], "T": [pd.NaT], "P": [float(prices.iloc[-1])], "P1": [np.nan],
                         "r": [float(np.log(prices.iloc[-1] / prices.iloc[-2]))], "r1": [0.0]}, index=[df.index.max() + 1])
    tr = df[(df["T"].dt.year >= egitim[0]) & (df["T"].dt.year <= egitim[1])]
    if len(tr) < 250:
        raise ValueError("Egitim donemi cok kisa.")
    full = pd.concat([df.loc[tr.index.min():], last])
    te = full[full.index > tr.index.max()]
    f = FUNCS[model]
    o = (f(tr, te, full, prices, pencere=pencere) if model in PENCERELI else f(tr, te, full, prices)).iloc[-1]
    return {"bilgi_gunu": str(prices.index[-1].date()), "son_kapanis": float(prices.iloc[-1]), "model": model,
            "egitim": list(egitim), "pencere": pencere, "tahmin": float(o["point"]),
            "aralik80": [float(o["lo80"]), float(o["hi80"])], "aralik95": [float(o["lo95"]), float(o["hi95"])],
            "hareketli_olasiligi_pct": float(100 * p_move(o["mu"], o["sigma"], esik, o.get("nu", np.nan))), "esik_pct": esik}


def pencere_secimi(prices: pd.Series, egitim: tuple, test: tuple, esik: float = ESIK, pencereler=PENCERELER) -> dict:
    """Pencere uzunlugu, test yillarina BAKILMADAN, egitimin son iki yilinda (dogrulama) secilir:
    olcut |%80 tutma - 80| + |%95 tutma - 95| (GARCH-t). Test sonuclari butun pencereler icin seffafca raporlanir."""
    ic, dog = (egitim[0], egitim[1] - 2), (egitim[1] - 1, egitim[1])
    rows = []
    for W in pencereler:
        v = run(prices, ic, dog, models=("GARCH-t",), esik=esik, pencere=W)["modeller"]["GARCH-t"]
        r = run(prices, egitim, test, models=("GARCH-t", "GARCH-t+ACI"), esik=esik, pencere=W)["modeller"]
        rows.append({"pencere_gun": W, "dogrulama_sapma": abs(v["aralik80_tutma_pct"] - 80) + abs(v["aralik95_tutma_pct"] - 95),
                     "test_80": r["GARCH-t"]["aralik80_tutma_pct"], "test_95": r["GARCH-t"]["aralik95_tutma_pct"],
                     "test_80_aci": r["GARCH-t+ACI"]["aralik80_tutma_pct"], "test_95_aci": r["GARCH-t+ACI"]["aralik95_tutma_pct"],
                     "hareketli_isabet": r["GARCH-t"]["hareketli_isabet_pct"], "hareketli_brier": r["GARCH-t"]["hareketli_brier"]})
    secilen = min(rows, key=lambda x: x["dogrulama_sapma"])["pencere_gun"]
    return {"egitim": list(egitim), "dogrulama": list(dog), "test": list(test), "secilen_pencere": secilen, "pencereler": rows}


def _fmt(v, d=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def print_report(res: dict):
    w = res.get("pencere")
    print(f"\n  Egitim {res['egitim'][0]}-{res['egitim'][1]} ({res['egitim_gun']} gun) | Test {res['test'][0]}-{res['test'][1]}"
          f" ({res['test_gun']} gun) | hareketli gun esigi %{_fmt(res['esik_pct'])} | pencere "
          f"{'yok (egitimle bir kez)' if w is None else ('genisleyen, aylik' if w == 0 else f'son {w} gun, aylik')}\n")
    rows = []
    for m, s in res["modeller"].items():
        rows.append([m, _fmt(s["ortalama_hata_tl"]), "%" + _fmt(s["ortalama_hata_pct"], 2), _fmt(s["rw_gore_pct"]),
                     "%" + _fmt(s["aralik80_tutma_pct"]), "%" + _fmt(s["aralik95_tutma_pct"]),
                     "%" + _fmt(s["hareketli_isabet_pct"]), "%" + _fmt(s["hep_sakin_isabet_pct"])])
    print(pd.DataFrame(rows, columns=["model", "hata TL", "hata %", "RW'ye gore %", "%80 aralik tuttu",
                                      "%95 aralik tuttu", "sakin/hareketli isabet", "'hep sakin' isabet"]).to_string(index=False))
    best = ODAK if ODAK in res["modeller"] else next(iter(res["modeller"]))     # onceden sabit; sonuca gore secilmez
    print(f"\n  Hava durumu tarzi guvenilirlik ({best}): 'yarin hareketli olma olasiligi' tahmini ve gerceklesen")
    print(pd.DataFrame(res["modeller"][best]["guvenilirlik"]).round(1).to_string(index=False))
    print(f"\n  Yillara gore ({best}):")
    print(pd.DataFrame(res["yillar"][best]).round(2).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gram altin tahmin karnesi.")
    ap.add_argument("komut", nargs="?", default="karne", choices=["karne", "yarin", "pencere"])
    ap.add_argument("--egitim", default="2012-2019")
    ap.add_argument("--test", default="2020-2022")
    ap.add_argument("--esik", type=float, default=ESIK, help="hareketli gun esigi, yuzde")
    ap.add_argument("--chronos", action="store_true", help="Chronos modelini de calistir (paket kurulu olmali)")
    ap.add_argument("--model", default=ODAK, choices=[m for m in FUNCS if m != "Chronos"])
    ap.add_argument("--pencere", type=int, default=None, help="gun; 0 = genisleyen; verilmezse egitimle bir kez")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    prices = load_series()
    print("=" * 78 + "\nGRAM ALTIN TAHMIN KARNESI - fiziki gram altin (TL/gr)\n" + "=" * 78)
    if args.komut == "pencere":
        r = pencere_secimi(prices, parse_years(args.egitim), parse_years(args.test), args.esik)
        print(f"\n  Pencere secimi: dogrulama {r['dogrulama'][0]}-{r['dogrulama'][1]} (test yillarina bakilmadan) | "
              f"test {r['test'][0]}-{r['test'][1]} | aylik yeniden egitim\n")
        tab = pd.DataFrame(r["pencereler"])
        tab["pencere"] = tab["pencere_gun"].map(lambda w: "genisleyen" if w == 0 else f"son {w} gun")
        tab["secim"] = np.where(tab["pencere_gun"] == r["secilen_pencere"], "<- secilen", "")
        print(tab[["pencere", "dogrulama_sapma", "test_80", "test_95", "test_80_aci", "test_95_aci", "hareketli_isabet", "secim"]]
              .round(1).to_string(index=False))
        out = Path(args.out or OUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        name = f"pencere_{r['egitim'][0]}-{r['egitim'][1]}_{r['test'][0]}-{r['test'][1]}.json"
        r["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out / name).write_text(json.dumps(r, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\n  Yazildi: {out / name}")
        return 0
    if args.komut == "yarin":
        last_y = prices.index[-1].year
        egitim = parse_years(args.egitim if args.egitim != "2012-2019" else f"2012-{last_y - 1}")
        c = tomorrow(prices, egitim, args.model, args.esik, args.pencere)
        print(f"  {c['bilgi_gunu']} kapanisi {_fmt(c['son_kapanis'], 2)} TL | model {c['model']}"
              f" | pencere {'yok' if c['pencere'] is None else ('genisleyen' if c['pencere'] == 0 else str(c['pencere']) + ' gun')}")
        print(f"  Bir sonraki islem gunu tahmini: {_fmt(c['tahmin'], 2)} TL")
        print(f"  %80 olasilikla {_fmt(c['aralik80'][0], 2)} - {_fmt(c['aralik80'][1], 2)} TL")
        print(f"  %95 olasilikla {_fmt(c['aralik95'][0], 2)} - {_fmt(c['aralik95'][1], 2)} TL")
        print(f"  Hareketli gun (degisim > %{_fmt(c['esik_pct'])}) olasiligi: %{_fmt(c['hareketli_olasiligi_pct'])}")
        return 0
    models = MODELS + (("Chronos",) if args.chronos else ())
    res = run(prices, parse_years(args.egitim), parse_years(args.test), models, args.esik, pencere=args.pencere)
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print_report(res)
    out = Path(args.out or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    name = f"karne_{res['egitim'][0]}-{res['egitim'][1]}_{res['test'][0]}-{res['test'][1]}"
    if args.pencere is not None:
        name += "_pencere" + ("genisleyen" if args.pencere == 0 else str(args.pencere))
    (out / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n  Yazildi: {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
