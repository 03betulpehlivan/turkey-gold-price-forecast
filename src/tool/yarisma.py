"""
Model Yarismasi: hangi model gram altinin 1, 3 ve 5 gun sonraki fiyat araligini en iyi tahmin ediyor?
==================================================================================================
Kullanim (proje kokunden):
  python -m src.tool.yarisma                       # 1, 3 ve 5 gun ufku, butun modeller (2-4 dakika)
  python -m src.tool.yarisma --ufuk 1 --chronos    # Chronos kuruluysa onu da ekler

Duzen (pencere pencere, gercek hayattaki gibi):
  Her ay basinda modeller YALNIZCA o gune kadar gerceklesmis fiyatlarla yeniden kurulur (son 750 islem gunu)
  ve o ayin her gunu icin h gun sonrasinin %80 ve %95 araligi tahmin edilir. Hicbir tahmin gelecegi gormez.
Iki donem:
  Secim   2012-2022: kazanan model burada belirlenir (hedefi gelen gunu 2023'ten once olan tahminler).
  Final   2023-2026: kazanan model ilk kez burada, hic gormedigi donemde sinanir.
Kazanma olcutu (onceden sabit): %80 ve %95 araliklarin tutma orani hedefe yakin (80 +-3, 95 +-2) olan modeller
  arasinda ARALIK PUANI en dusuk olan. Aralik puani = aralik genisligi + iskalanan gunlerde kacirilan mesafenin
  cezasi (Gneiting ve Raftery); hem dar hem isabetli olmayi odullendirir.
Modeller:
  Sabit aralik     son 750 gunun oynakligi, sqrt(h) ile olceklenir
  Gecmis benzer    son 750 gunde gerceklesen h gunluk degisimlerin yuzdelikleri (tarihsel simulasyon)
  EWMA             yakin gunlere agirlik veren oynaklik (lambda 0.94)
  GARCH-t          GARCH(1,1), Student-t; h gunluk varyans GARCH formuluyle
  GJR-t            asimetrik GARCH (dususler oynakligi daha cok artirir)
  EWMA+ACI, GARCH-t+ACI   araliklar, h gun once yapilan tahminlerin iskalamalarina gore kendini ayarlar
  XGBoost-kantil   makine ogrenmesi: son 10 gunluk getiriler ve 5/20/60 gunluk oynakliktan dogrudan yuzdelik
  ARIMA-GARCH      ortalama icin AR(1) (ARIMA(1,0,0)), oynaklik icin GARCH(1,1)-t
  HAR              cok olcekli oynaklik: gunluk, haftalik (5) ve aylik (22) kare getirilerden h gunluk varyans (EKK)
  Rejim (Markov)   iki rejimli (sakin / kriz) Markov modeli; rejim olasiliklari her gun yalnizca gecmisle guncellenir,
                   parametreler uc ayda bir yeniden tahmin edilir
  Birlesim         GARCH-t, EWMA ve Gecmis benzer modellerinin yuzdeliklerinin ortalamasi (onceden sabit uyeler)
  HAR-X            HAR + dis piyasa bilgisi: altin oynaklik endeksi (GVZ), korku endeksi (VIX), dolar/TL, Brent petrol
                   ve dolar endeksi (DXY) oynakliklari. Her bilgi gunu icin YALNIZCA bir onceki gunun degerleri
                   kullanilir (ABD kapanisi Turkiye kapanisindan sonra oldugu icin).
  Birlesim-2       (Chronos varsa) ARIMA-GARCH ve Chronos yuzdeliklerinin ortalamasi (onceden sabit uyeler)
Anlamlilik: kazanan modelin "Sabit aralik"a gore gunluk aralik puani farki icin DM testi (Newey-West + HLN)
  ve eslestirilmis blok bootstrap ile iyilesme yuzdesinin %95 guven araligi; secim ve final donemi ayri.
  Chronos          (istege bagli) Amazon Chronos-Bolt, egitimsiz; son 512 gunun fiyatindan %10/%50/%90 yuzdelik,
                   %95 araligi bunlardan normal varsayimiyla turetilir (Bolt %10-%90 disini egitmedi)
HAR ve Rejim modellerinde aralik sekli (kuyruk kalinligi), her pencerede modelin egitim donemindeki
  standartlastirilmis hatalarinin yuzdeliklerinden alinir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from src.models import final_exam as fe
from src.tool import karne as k

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "tool"
PANEL_PATH = ROOT / "data" / "processed" / "history" / "market_panel.csv"
DIS = ("gvz", "vix", "usdtry", "brent", "dxy")
DIS_YAS = 10                                  # gun; daha eski dis veri kullanilmaz
UFUKLAR = (1, 3, 5)
PENCERE, EWMA_L, ACI_GAMMA = 750, 0.94, 0.005
SECIM = ("2012-01-01", "2023-01-01")         # bilgi gunu >= ilk, hedef gunu < ikinci
FINAL = "2023-01-01"                          # bilgi gunu >= bu
HEDEF = {80: 3.0, 95: 2.0}                    # kabul edilen sapma (yuzde puani)
QS = (0.025, 0.10, 0.50, 0.90, 0.975)
MODELLER = ("Sabit aralik", "Gecmis benzer", "EWMA", "GARCH-t", "GJR-t", "EWMA+ACI", "GARCH-t+ACI", "XGBoost-kantil",
            "ARIMA-GARCH", "HAR", "Rejim (Markov)", "Birlesim", "HAR-X")
BIRLESIM = ("GARCH-t", "EWMA", "Gecmis benzer")


# ---------------------------------------------------------------------------
# Hazirlik
# ---------------------------------------------------------------------------
def hazirla(prices: pd.Series) -> dict:
    lp = np.log(prices.to_numpy(float))
    r = np.concatenate([[np.nan], np.diff(lp)])          # r[i]: i-1 -> i, i gununde bilinir
    return {"tarih": prices.index, "lp": lp, "r": r, "n": len(lp)}


def kokenler(d: dict, h: int, start: str = SECIM[0]) -> np.ndarray:
    """Tahmin yapilan gunler (bilgi gunu): pencere icin yeterli gecmis var, hedefi (i+h) veride."""
    i = np.arange(d["n"] - h)
    return i[(d["tarih"][i] >= pd.Timestamp(start)) & (i >= PENCERE + 61)]


def ay_bloklari(d: dict, idx: np.ndarray) -> list:
    """Ayni takvim ayindaki ardisik bilgi gunleri bir blok; model her blok basinda yeniden kurulur."""
    if len(idx) == 0:
        return []
    per = pd.DatetimeIndex(d["tarih"][idx]).to_period("M")
    cuts = np.flatnonzero(per[1:] != per[:-1]) + 1
    return np.split(idx, cuts)


# ---------------------------------------------------------------------------
# Modeller: her biri (mu, sig, nu) parametrik ya da Q (yuzdelikler) dondurur; h gunluk log getiri olcegi
# ---------------------------------------------------------------------------
def m_sabit(d, idx, h):
    r = d["r"]
    sig = np.array([np.nanstd(r[i - PENCERE + 1: i + 1]) for i in idx]) * math.sqrt(h)
    return {"mu": np.zeros(len(idx)), "sig": sig, "nu": np.full(len(idx), np.nan)}


def m_gecmis(d, idx, h):
    lp = d["lp"]
    Q = np.empty((len(idx), len(QS)))
    for j, i in enumerate(idx):
        a = np.arange(i - PENCERE, i - h + 1)               # hedefi i gunune kadar gerceklesmis degisimler
        Q[j] = np.quantile(lp[a + h] - lp[a], QS)
    return {"Q": Q}


def _ewma_var(r):
    v = np.empty(len(r))
    cur = np.nanvar(r[1:61])
    for i in range(len(r)):
        x = r[i]
        if np.isfinite(x):
            cur = EWMA_L * cur + (1 - EWMA_L) * x * x
        v[i] = cur                                          # v[i]: r[..i] ile, r[i+1] icin
    return v


def m_ewma(d, idx, h):
    v = _ewma_var(d["r"])
    return {"mu": np.zeros(len(idx)), "sig": np.sqrt(v[idx] * h), "nu": np.full(len(idx), np.nan)}


def _garch(d, idx, h, gjr, cache):
    key = ("gjr" if gjr else "garch")
    if key not in cache:
        cache[key] = _garch_daily(d, idx_all=cache["idx_all"], gjr=gjr)
    s2, persist, V, nu = cache[key]
    pos = np.searchsorted(cache["idx_all"], idx)
    a, p, v = s2[pos], persist[pos], V[pos]
    k_ = np.arange(h)
    tot = np.where(p < 0.999, (v[:, None] + p[:, None] ** k_ * (a - v)[:, None]).sum(axis=1), a * h)
    return {"mu": np.zeros(len(idx)), "sig": np.sqrt(tot), "nu": nu[pos]}


def _garch_daily(d, idx_all, gjr):
    """Her bilgi gunu i icin: sigma^2_{i+1}, kalicilik, uzun donem varyans ve nu (ay basi yeniden tahmin)."""
    from arch import arch_model
    x = 100 * np.nan_to_num(d["r"], nan=0.0)
    n = len(idx_all)
    s2, per, V, nu = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    k_ = 0
    for blok in ay_bloklari(d, idx_all):
        b, e = int(blok[0]), int(blok[-1])
        fs = b - PENCERE + 1
        am = arch_model(x[fs:b + 1], mean="Zero", vol="GARCH", p=1, o=1 if gjr else 0, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        pr = res.params
        seri = np.concatenate([x[fs:e + 2], np.zeros(max(0, e + 2 - len(x)))])   # son gun icin yer tutucu
        cv = np.asarray(arch_model(seri, mean="Zero", vol="GARCH", p=1, o=1 if gjr else 0, q=1, dist="t",
                                   rescale=False).fix(pr).conditional_volatility)
        p = float(pr["alpha[1]"] + pr["beta[1]"] + (0.5 * pr["gamma[1]"] if gjr else 0.0))
        lv = float(pr["omega"] / (1 - p)) if p < 0.999 else np.nan
        for i in blok:
            s2[k_] = (cv[i + 1 - fs] / 100.0) ** 2                       # cv[j]: seri[:j] ile
            per[k_], V[k_], nu[k_] = p, (lv / 1e4 if np.isfinite(lv) else np.nan), pr.get("nu", np.nan)
            k_ += 1
    return s2, per, V, nu


def m_garch_t(d, idx, h, cache):
    return _garch(d, idx, h, False, cache)


def m_gjr_t(d, idx, h, cache):
    return _garch(d, idx, h, True, cache)


def _ozellik(d):
    r = np.nan_to_num(d["r"], nan=0.0)
    s = pd.Series(r)
    cols = [np.roll(r, j) for j in range(10)] + [s.rolling(w).std().to_numpy() for w in (5, 20, 60)]
    X = np.column_stack(cols)
    X[:70] = np.nan
    return X


def m_xgb(d, idx, h):
    from xgboost import XGBRegressor
    X, lp = _ozellik(d), d["lp"]
    Q = np.empty((len(idx), len(QS)))
    yil = pd.DatetimeIndex(d["tarih"][idx]).year
    for y in np.unique(yil):
        blok = idx[yil == y]
        b = int(blok[0])
        tr = np.arange(max(70, b - 2500), b - h + 1)                  # hedefi b gunune kadar gerceklesmis
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), n_estimators=200, max_depth=3,
                         learning_rate=0.05, subsample=0.8, random_state=0, n_jobs=1)
        m.fit(X[tr], lp[tr + h] - lp[tr])
        Q[yil == y] = np.sort(m.predict(X[blok]), axis=1)
    return {"Q": Q}


def m_chronos(d, idx, h, pipeline=None):
    try:
        import torch
        tens = lambda a: torch.tensor(a, dtype=torch.float32)
    except ImportError:
        if pipeline is None:
            raise RuntimeError("Chronos icin: pip install chronos-forecasting")
        tens = np.asarray
    if pipeline is None:
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(k.CHRONOS_ID, device_map="cpu", torch_dtype=torch.float32)
    p = np.exp(d["lp"])
    Q = []
    for a in range(0, len(idx), 256):
        ctx = [tens(p[max(0, i + 1 - k.CHRONOS_CTX): i + 1]) for i in idx[a:a + 256]]
        q, _ = pipeline.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
        lq = np.log(np.asarray(q)[:, h - 1, :] / p[idx[a:a + 256], None])
        s = (lq[:, 2] - lq[:, 0]) / (2 * 1.2815515655446004)
        Q.append(np.column_stack([lq[:, 1] - 1.959963984540054 * s, lq[:, 0], lq[:, 1], lq[:, 2],
                                  lq[:, 1] + 1.959963984540054 * s]))
    return {"Q": np.vstack(Q)}


def _blok(d, idx, ceyrek=False):
    if not ceyrek:
        return ay_bloklari(d, idx)
    per = pd.DatetimeIndex(d["tarih"][idx]).to_period("Q")
    return np.split(idx, np.flatnonzero(per[1:] != per[:-1]) + 1) if len(idx) else []


def _z_yuzdelik(y, olcek):
    z = y / olcek
    z = z[np.isfinite(z)]
    return np.quantile(z, QS)


def m_arima_garch(d, idx, h):
    """AR(1) ortalama + GARCH(1,1)-t oynaklik; ay basi yeniden tahmin, sabit parametreyle gunluk filtre."""
    from arch import arch_model
    x = 100 * np.nan_to_num(d["r"], nan=0.0)
    mu, sig, nu = np.empty(len(idx)), np.empty(len(idx)), np.empty(len(idx))
    k_ = 0
    for blok in ay_bloklari(d, idx):
        b, e = int(blok[0]), int(blok[-1])
        fs = b - PENCERE + 1
        spec = dict(mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = arch_model(x[fs:b + 1], **spec).fit(disp="off", show_warning=False)
        pr = res.params
        seri = np.concatenate([x[fs:e + 2], np.zeros(max(0, e + 2 - len(x)))])
        cv = np.asarray(arch_model(seri, **spec).fix(pr).conditional_volatility)
        c, phi = float(pr["Const"]), float(pr["y[1]"])
        p = float(pr["alpha[1]"] + pr["beta[1]"])
        V = float(pr["omega"] / (1 - p)) if p < 0.999 else np.nan
        for i in blok:
            a2 = cv[i + 1 - fs] ** 2
            ks = np.arange(1, h + 1)
            m_ = np.sum(c * (1 - phi ** ks) / (1 - phi) + phi ** ks * x[i]) if abs(phi) < 0.999 else c * h
            var = np.sum(V + p ** (ks - 1) * (a2 - V)) if np.isfinite(V) else a2 * h
            mu[k_], sig[k_], nu[k_] = m_ / 100.0, math.sqrt(var) / 100.0, pr["nu"]
            k_ += 1
    return {"mu": mu, "sig": sig, "nu": nu}


def _har_x(r):
    r2 = np.nan_to_num(r, nan=0.0) ** 2
    s = pd.Series(r2)
    return np.column_stack([np.ones(len(r2)), r2, s.rolling(5).mean().to_numpy(), s.rolling(22).mean().to_numpy()])


def m_har(d, idx, h):
    """HAR: gelecek h gunun ortalama kare getirisi ~ dunku, haftalik ve aylik ortalama kare getiri (EKK)."""
    r, lp = d["r"], d["lp"]
    X = _har_x(r)
    r2 = np.nan_to_num(r, nan=0.0) ** 2
    hedef = np.full(len(r), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    j = np.arange(len(r) - h)
    hedef[j] = (cs[j + h + 1] - cs[j + 1]) / h                         # mean r2[j+1 .. j+h]
    Q = np.empty((len(idx), len(QS)))
    k_ = 0
    for blok in ay_bloklari(d, idx):
        b = int(blok[0])
        tr = np.arange(b - PENCERE, b - h + 1)                         # hedefi b gunune kadar gerceklesmis
        beta = np.linalg.lstsq(X[tr], hedef[tr], rcond=None)[0]
        alt = 0.1 * float(np.mean(hedef[tr]))
        olc = lambda rows: np.sqrt(h * np.maximum(X[rows] @ beta, alt))
        zq = _z_yuzdelik(lp[tr + h] - lp[tr], olc(tr))
        Q[k_:k_ + len(blok)] = olc(blok)[:, None] * zq[None, :]
        k_ += len(blok)
    return {"Q": Q}


def m_rejim(d, idx, h):
    """Iki rejimli Markov modeli (sabit + rejime gore varyans); ceyrek basi yeniden tahmin, gunluk filtre."""
    import warnings
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    x = 100 * np.nan_to_num(d["r"], nan=0.0)
    lp = d["lp"]
    Q = np.empty((len(idx), len(QS)))
    k_ = 0
    for blok in _blok(d, idx, ceyrek=True):
        b, e = int(blok[0]), int(blok[-1])
        fs = b - PENCERE + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MarkovRegression(x[fs:b + 1], k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
            fr = MarkovRegression(x[fs:e + 1], k_regimes=2, trend="c", switching_variance=True).filter(res.params)
        names = res.model.param_names
        s2 = np.array([res.params[names.index("sigma2[0]")], res.params[names.index("sigma2[1]")]])
        T = np.asarray(fr.regime_transition)[:, :, 0]                  # T[sonraki, onceki]
        P = np.asarray(fr.filtered_marginal_probabilities)              # P[t]: x[..t] ile
        Tk = [np.linalg.matrix_power(T, kk) for kk in range(1, h + 1)]

        def olcek(rows):
            pr = P[rows - fs]
            v = sum((pr @ M.T) @ s2 for M in Tk)
            return np.sqrt(v) / 100.0

        tr = np.arange(fs, b - h + 1)
        zq = _z_yuzdelik(lp[tr + h] - lp[tr], olcek(tr))
        Q[k_:k_ + len(blok)] = olcek(blok)[:, None] * zq[None, :]
        k_ += len(blok)
    return {"Q": Q}


def panel_yukle(path: Path = None):
    p = Path(path or PANEL_PATH)
    if not p.exists():
        return None
    cols = ["Date"] + [c for c in DIS if c in pd.read_csv(p, nrows=0).columns]
    return pd.read_csv(p, usecols=cols, parse_dates=["Date"]).sort_values("Date").drop_duplicates("Date")


def dis_ozellik(d: dict, panel: pd.DataFrame) -> np.ndarray:
    """Fiziki gun i icin, tarihi KESINLIKLE i'den once olan en son panel satirindan (en cok DIS_YAS gun) gunluk
    varyans olceginde ozellikler: (GVZ/100)^2/252, (VIX/100)^2/252 ve dolar/TL, Brent, DXY'nin 20 gunluk ortalama
    kare log degisimleri. Eksikse NaN."""
    pn = panel.set_index("Date")
    f = pd.DataFrame(index=pn.index)
    for c in ("gvz", "vix"):
        if c in pn:
            f[c] = (pn[c].where(pn[c] > 0) / 100.0) ** 2 / 252.0
    for c in ("usdtry", "brent", "dxy"):
        if c in pn:
            f[c] = (np.log(pn[c].where(pn[c] > 0)).diff() ** 2).rolling(20, min_periods=15).mean()
    f = f.reset_index().rename(columns={"Date": "pdate"}).sort_values("pdate")
    q = pd.DataFrame({"tarih": pd.DatetimeIndex(d["tarih"]) - pd.Timedelta(days=1)}).reset_index()
    m = pd.merge_asof(q.sort_values("tarih"), f, left_on="tarih", right_on="pdate", direction="backward")
    m = m.sort_values("index")
    eski = (m["tarih"] - m["pdate"]).dt.days > DIS_YAS
    X = m[[c for c in f.columns if c != "pdate"]].to_numpy(float)
    X[eski.to_numpy()] = np.nan
    return X


def m_harx(d, idx, h, panel):
    """HAR + dis oynaklik bilgisi (EKK, aylik). Dis verisi eksik gunlerde ayni bloktaki duz HAR kullanilir."""
    r, lp = d["r"], d["lp"]
    Xh = _har_x(r)
    E = dis_ozellik(d, panel)
    X = np.column_stack([Xh, E])
    r2 = np.nan_to_num(r, nan=0.0) ** 2
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    hedef = np.full(len(r), np.nan)
    j = np.arange(len(r) - h)
    hedef[j] = (cs[j + h + 1] - cs[j + 1]) / h
    Q = np.empty((len(idx), len(QS)))
    k_ = 0
    for blok in ay_bloklari(d, idx):
        b = int(blok[0])
        tr = np.arange(b - PENCERE, b - h + 1)
        ok = np.isfinite(X[tr]).all(axis=1)
        alt = 0.1 * float(np.mean(hedef[tr]))
        bh = np.linalg.lstsq(Xh[tr], hedef[tr], rcond=None)[0]
        if ok.sum() >= 250:
            bx = np.linalg.lstsq(X[tr][ok], hedef[tr][ok], rcond=None)[0]
        else:
            bx = None

        def olc(rows):
            v = Xh[rows] @ bh
            if bx is not None:
                fin = np.isfinite(X[rows]).all(axis=1)
                v = np.where(fin, np.nan_to_num(X[rows]) @ bx, v)
            return np.sqrt(h * np.maximum(v, alt))

        zq = _z_yuzdelik(lp[tr + h] - lp[tr], olc(tr))
        Q[k_:k_ + len(blok)] = olc(blok)[:, None] * zq[None, :]
        k_ += len(blok)
    return {"Q": Q}


def aci(d, idx, h, f, gamma=ACI_GAMMA):
    """Parametrik tahmin + uyarlamali kalibrasyon. i gunundeki ayar, yalnizca hedefi i'ye kadar gerceklesmis
    (i-h ve oncesi) tahminlerin iskalamalarini kullanir."""
    lp = d["lp"]
    y = np.full(len(idx), np.nan)
    ok = idx + h < d["n"]
    y[ok] = lp[idx[ok] + h] - lp[idx[ok]]
    Q = np.empty((len(idx), len(QS)))
    Q[:, 2] = f["mu"]
    pos = {int(i): j for j, i in enumerate(idx)}
    for lvl, lo_c, hi_c in ((0.20, 1, 3), (0.05, 0, 4)):
        a = np.full(len(idx) + 1, lvl)
        cur = lvl
        for j, i in enumerate(idx):
            src = pos.get(int(i) - h)                                   # h gun once yapilan tahmin, bugun sonuclandi
            if src is not None and np.isfinite(y[src]):
                miss = float(not (Q[src, lo_c] <= y[src] <= Q[src, hi_c]))
                cur = float(np.clip(cur + gamma * (lvl - miss), 1e-4, 0.999))
            z = float(k._ppf(1 - cur / 2, f["nu"][j]))
            Q[j, lo_c], Q[j, hi_c] = f["mu"][j] - z * f["sig"][j], f["mu"][j] + z * f["sig"][j]
    return {"Q": Q}


def yuzdelikler(f: dict) -> np.ndarray:
    if "Q" in f:
        return f["Q"]
    z = {q: k._ppf(np.full(len(f["sig"]), q), f["nu"]) for q in QS}
    return np.column_stack([f["mu"] + z[q] * f["sig"] for q in QS])


def tahminler(d, h, modeller=MODELLER, chronos_pipeline=None, panel=None) -> dict:
    idx = kokenler(d, h)
    cache = {"idx_all": idx}
    out = {}
    for m in modeller:
        if m == "Sabit aralik":
            f = m_sabit(d, idx, h)
        elif m == "Gecmis benzer":
            f = m_gecmis(d, idx, h)
        elif m == "EWMA":
            f = m_ewma(d, idx, h)
        elif m == "GARCH-t":
            f = m_garch_t(d, idx, h, cache)
        elif m == "GJR-t":
            f = m_gjr_t(d, idx, h, cache)
        elif m == "EWMA+ACI":
            f = aci(d, idx, h, m_ewma(d, idx, h))
        elif m == "GARCH-t+ACI":
            f = aci(d, idx, h, m_garch_t(d, idx, h, cache))
        elif m == "XGBoost-kantil":
            f = m_xgb(d, idx, h)
        elif m == "Chronos":
            f = m_chronos(d, idx, h, chronos_pipeline)
        elif m == "ARIMA-GARCH":
            f = m_arima_garch(d, idx, h)
        elif m == "HAR":
            f = m_har(d, idx, h)
        elif m == "Rejim (Markov)":
            f = m_rejim(d, idx, h)
        elif m == "HAR-X":
            if panel is None:
                continue                                              # dis veri yoksa yarismaya girmez
            f = m_harx(d, idx, h, panel)
        elif m in ("Birlesim", "Birlesim-2"):
            continue
        else:
            raise ValueError(m)
        out[m] = yuzdelikler(f)
    if "Birlesim" in modeller:
        eksik = [u for u in BIRLESIM if u not in out]
        for u in eksik:
            out[u] = yuzdelikler({"GARCH-t": lambda: m_garch_t(d, idx, h, cache), "EWMA": lambda: m_ewma(d, idx, h),
                                  "Gecmis benzer": lambda: m_gecmis(d, idx, h)}[u]())
        out["Birlesim"] = np.mean([out[u] for u in BIRLESIM], axis=0)
        for u in eksik:
            out.pop(u)
    if "Birlesim-2" in modeller and "Chronos" in out and "ARIMA-GARCH" in out:
        out["Birlesim-2"] = np.mean([out["ARIMA-GARCH"], out["Chronos"]], axis=0)
    return {"idx": idx, "Q": out}


# ---------------------------------------------------------------------------
# Puanlama
# ---------------------------------------------------------------------------
def aralik_puani(lo, hi, y, alpha):
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def puanla(Q: np.ndarray, y: np.ndarray) -> dict:
    yy = 100 * y
    Qp = 100 * Q
    in80 = (Qp[:, 1] <= yy) & (yy <= Qp[:, 3])
    in95 = (Qp[:, 0] <= yy) & (yy <= Qp[:, 4])
    return {"gun": int(len(y)), "tutma80": float(100 * in80.mean()), "tutma95": float(100 * in95.mean()),
            "genislik80": float(np.mean(Qp[:, 3] - Qp[:, 1])), "genislik95": float(np.mean(Qp[:, 4] - Qp[:, 0])),
            "puan80": float(np.mean(aralik_puani(Qp[:, 1], Qp[:, 3], yy, 0.20))),
            "puan95": float(np.mean(aralik_puani(Qp[:, 0], Qp[:, 4], yy, 0.05))),
            "nokta_hata": float(np.mean(np.abs(yy - Qp[:, 2]))), "rw_hata": float(np.mean(np.abs(yy)))}


def gunluk_puan(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    yy, Qp = 100 * y, 100 * Q
    return aralik_puani(Qp[:, 1], Qp[:, 3], yy, 0.20) + aralik_puani(Qp[:, 0], Qp[:, 4], yy, 0.05)


def anlamlilik(Q_model, Q_ref, y) -> dict:
    lm, lr = gunluk_puan(Q_model, y), gunluk_puan(Q_ref, y)
    dm = fe.dm_hln(lm, lr)
    bs = fe.paired_bootstrap(lm, lr)
    return {"iyilesme_pct": float(-bs["pct"]), "iyilesme_ga95": [float(-bs["pct_ci95"][1]), float(-bs["pct_ci95"][0])],
            "p_degeri": dm["p_value"], "gun": int(len(y))}


def kazanan(tablo: dict) -> str:
    uygun = [m for m, s in tablo.items() if abs(s["tutma80"] - 80) <= HEDEF[80] and abs(s["tutma95"] - 95) <= HEDEF[95]]
    aday = uygun or list(tablo)
    return min(aday, key=lambda m: tablo[m]["puan80"] + tablo[m]["puan95"])


def yarisma(prices: pd.Series, ufuklar=UFUKLAR, modeller=MODELLER, chronos_pipeline=None, panel=None) -> dict:
    d = hazirla(prices)
    sonuc = {}
    for h in ufuklar:
        t = tahminler(d, h, modeller, chronos_pipeline, panel)
        idx = t["idx"]
        y = d["lp"][idx + h] - d["lp"][idx]
        bilgi, hedef = d["tarih"][idx], d["tarih"][idx + h]
        secim = np.asarray((bilgi >= pd.Timestamp(SECIM[0])) & (hedef < pd.Timestamp(SECIM[1])))
        final = np.asarray(bilgi >= pd.Timestamp(FINAL))
        yillar = pd.DatetimeIndex(bilgi).year
        secim_t = {m: puanla(Q[secim], y[secim]) for m, Q in t["Q"].items()}
        final_t = {m: puanla(Q[final], y[final]) for m, Q in t["Q"].items()} if final.any() else {}
        yil_t = {m: [{"yil": int(Y), **{a: v for a, v in puanla(Q[yillar == Y], y[yillar == Y]).items()
                                        if a in ("gun", "tutma80", "tutma95", "genislik80")}}
                     for Y in np.unique(yillar)] for m, Q in t["Q"].items()}
        w = kazanan(secim_t)
        anl = {}
        if "Sabit aralik" in t["Q"] and w != "Sabit aralik":
            anl["secim"] = anlamlilik(t["Q"][w][secim], t["Q"]["Sabit aralik"][secim], y[secim])
            if final.any():
                anl["final"] = anlamlilik(t["Q"][w][final], t["Q"]["Sabit aralik"][final], y[final])
        sonuc[str(h)] = {"ufuk_gun": h, "secim": secim_t, "final": final_t, "yillar": yil_t, "kazanan": w,
                         "anlamlilik_sabit_araliga_gore": anl,
                         "donem": {"secim": [str(bilgi[secim].min().date()), str(hedef[secim].max().date())],
                                   "final": [str(bilgi[final].min().date()), str(hedef[final].max().date())] if final.any() else None}}
    return {"seri": "fiziki gram altin (TL/gr)", "pencere_gun": PENCERE, "yeniden_egitim": "aylik",
            "olcut": "tutma orani hedefe yakin modeller arasinda en dusuk aralik puani (%80 + %95)",
            "ufuklar": sonuc}


def _f(v, d=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}".replace(".", ",")


def yazdir(res: dict):
    for h, u in res["ufuklar"].items():
        print(f"\n  UFUK {h} GUN | secim {u['donem']['secim'][0]} .. {u['donem']['secim'][1]} | kazanan: {u['kazanan']}")
        rows = [[m, _f(s["tutma80"]), _f(s["tutma95"]), _f(s["genislik80"], 2), _f(s["puan80"] + s["puan95"], 2),
                 "<- kazanan" if m == u["kazanan"] else ""] for m, s in u["secim"].items()]
        print(pd.DataFrame(rows, columns=["model", "%80 tuttu", "%95 tuttu", "%80 genislik %", "aralik puani", ""]).to_string(index=False))
        if u["final"]:
            f = u["final"][u["kazanan"]]
            print(f"  FINAL ({u['donem']['final'][0]} .. {u['donem']['final'][1]}, hic gorulmemis, {f['gun']} gun): "
                  f"{u['kazanan']} %80 aralik %{_f(f['tutma80'])}, %95 aralik %{_f(f['tutma95'])} tuttu "
                  f"(ortalama %80 genislik %{_f(f['genislik80'], 2)})")
        for donem, a in u.get("anlamlilik_sabit_araliga_gore", {}).items():
            print(f"  Anlamlilik ({donem}): {u['kazanan']}, sabit araliga gore aralik puaninda %{_f(a['iyilesme_pct'])} iyi "
                  f"[%95 GA %{_f(a['iyilesme_ga95'][0])}, %{_f(a['iyilesme_ga95'][1])}], p = {_f(a['p_degeri'], 4)}")
        yy = pd.DataFrame(u["yillar"][u["kazanan"]])
        print("  Yillara gore (" + u["kazanan"] + "): " + ", ".join(f"{int(r.yil)} %{_f(r.tutma80)}" for r in yy.itertuples()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Model yarismasi (1/3/5 gun, aylik pencere).")
    ap.add_argument("--ufuk", default="1,3,5")
    ap.add_argument("--chronos", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    ufuklar = tuple(int(x) for x in args.ufuk.split(","))
    modeller = MODELLER + (("Chronos", "Birlesim-2") if args.chronos else ())
    print("=" * 78 + "\nMODEL YARISMASI - fiziki gram altin, aylik pencere, 1/3/5 gun ufuk\n" + "=" * 78)
    panel = panel_yukle()
    if panel is None:
        print(f"  Not: dis piyasa paneli yok ({PANEL_PATH}); HAR-X yarismaya girmiyor.")
    else:
        print(f"  Dis piyasa verisi: {', '.join(c for c in DIS if c in panel.columns)} | son tarih {panel['Date'].max().date()}"
              " (bundan sonraki gunlerde HAR-X duz HAR'a doner)")
    res = yarisma(k.load_series(), ufuklar, modeller, panel=panel)
    if panel is not None:
        res["dis_veri"] = {"dosya": str(PANEL_PATH.relative_to(ROOT)).replace("\\", "/"), "son_tarih": str(panel["Date"].max().date()),
                           "seriler": [c for c in DIS if c in panel.columns]}
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    yazdir(res)
    out = Path(args.out or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    name = "yarisma_" + "-".join(map(str, ufuklar)) + ("_chronos" if args.chronos else "") + ".json"
    (out / name).write_text(json.dumps(res, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n  Yazildi: {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
