"""
Canli Sabah Fiziki Altin Tahmini (LIVE_PHYS_V1)
===============================================
Kullanim (proje kokunden; scripts/sabah_fiziki.ps1 bu sirayla cagirir):
  python -m src.live.physical_forecast prepare     # BIR KEZ: protokolu kilitle, sonra git commit
  python -m src.data.collect                         # ons gunluk (dunku kapanis)
  python -m src.data.build_panel --write --processed-dir data/processed/live
  python -m src.data.collect_intraday                # USD/TRY saatlik (bugun 05:00 UTC cubugu)
  python -m src.live.physical_forecast refresh     # EVDS guncel fiziki seri (anahtar ortam degiskeninde)
  python -m src.live.physical_forecast score       # gerceklesenleri puanla
  python -m src.live.physical_forecast issue       # bugunun tahmini (08:00-14:00 TSI)
  python -m src.live.physical_forecast card        # karne
  python -m src.live.physical_forecast evaluate    # 120 puanlanmis tahminden sonra, TEK SEFER

Hedef: BUGUNUN fiziki altin kapanisi, TL/gr (EVDS; Borsa Istanbul toptan piyasa, kuyumcu degil).
Protokol (sonuclardan ONCE sabitlendi; degisiklik yeni surum gerektirir):
  Bilgi : P_t = EVDS'de yayimlanmis son fiziki kapanis (bugunden once, en fazla 4 takvim gunu);
          ons = panelde bugunden KESIN once olan son gunluk kapanis (>= t olmali);
          kur = saatlik USD/TRY, bitisi bugun 05:00 UTC olan cubuk (gecmisteki Yahoo gunluk kur saati
          02-05 UTC ile uyumlu; saat damgasi acik, sizinti tasarim geregi yok).
          M = ons x kur / 31.1035; bosluk = log M - log P_t.
  Modeller:
    RW            : P_t
    Bosluk        : log getiri = a + b*bosluk; OLS, 2011'den beri tum taze satirlar
    Bosluk-Kayan  : ayni, yalnizca son 250 taze satir
    Prim-Capa     : M x exp(prim_ewma); prim = log P - log M (gunluk), yari omur 20 gun. Model fiziki fiyatin
                    teorik fiyata yakinsamadigini, son primini korudugunu varsayar. FINAL_EXAM_PHYS_V1
                    sonucundan SONRA onerildi; bu yuzden yalnizca ileriye donuk test edilir.
  Egitim baslangici 2011-01-01: 2005-2010'da Yahoo kur cubugu BIST kapanisindan sonraydi (fx_timing_history).
  Tahmin ani: hedef gunun Istanbul takvim gununde, 05:00-11:00 UTC; hedef gunun kapanisi yayimlanmissa red.
  Degerlendirme: 120 puanlanmis tahmin (yalnizca on kayittaki kod md5'i ile uretilenler), TEK SEFER;
          her model vs RW: HLN-DM + Holm (3 karsilastirma) ve eslestirilmis blok bootstrap.
Kayit: research/live/physical_log.csv (git). Satir basina hash; tahmin alanlari degistirilemez.
Ana data/processed/physical/physical_gold_daily.csv DEGISMEZ (FINAL_EXAM_PHYS_V1 md5 guvencesi);
canli kopya data/processed/live/physical_gold_daily.csv.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from src.data import evds
from src.data import physical_gold as pg
from src.data import sync_dataset as sd
from src.models import final_exam as fe
from src.models import physical_eval as pv

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "LIVE_PHYS_V1"
LIVE_DIR = ROOT / "data" / "processed" / "live"
PHYS_LIVE = LIVE_DIR / "physical_gold_daily.csv"
PANEL_LIVE = LIVE_DIR / "market_panel.csv"
BASE_PHYS = pv.PHYS_PATH
LOG_PATH = ROOT / "research" / "live" / "physical_log.csv"
PREREG_PATH = ROOT / "research" / "preregistration" / "live_phys_v1.json"
REGISTRY_PATH = ROOT / "research" / "registry.jsonl"
MODELS = ["RW", "Bosluk", "Bosluk-Kayan", "Prim-Capa"]
TRAIN_START = "2011-01-01"
ROLL_N = 250
EWMA_HALFLIFE = 20
FX_BAR_END_UTC = 5
CUTOFF_UTC = 11
MAX_AGE_DAYS = 4
EVAL_N = 120
IST = "Europe/Istanbul"
FORECAST_COLS = ["forecast_id", "issued_at_utc", "protocol_id", "code_md5", "target_date", "last_phys_date",
                 "last_phys_price", "horizon_days", "gold_src_date", "gold_usd", "fx_bar_end_utc", "usdtry",
                 "M_morning", "gap", "prim_ewma", "beta_full", "beta_roll", "n_train"] + [f"pred_{m}" for m in MODELS]
REALIZED_COLS = ["realized_price"] + [f"abs_err_{m}" for m in MODELS]
ALL_COLS = FORECAST_COLS + ["row_hash"] + REALIZED_COLS
TEXT = {"forecast_id", "issued_at_utc", "protocol_id", "code_md5", "target_date", "last_phys_date",
        "gold_src_date", "fx_bar_end_utc"}


class Refused(Exception):
    """Kural geregi tahmin uretilmedi (veri eski/eksik veya zaman disi)."""


class LogTampered(RuntimeError):
    pass


def _md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def code_md5() -> str:
    return _md5(Path(__file__))


def row_hash(row) -> str:
    parts = []
    for c in FORECAST_COLS:
        v = row[c]
        parts.append(("" if pd.isna(v) else str(v)) if c in TEXT else ("nan" if pd.isna(v) else f"{float(v):.10g}"))
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def refresh(client, base_path: Path = None, out_path: Path = None, now=None) -> dict:
    """Guncel EVDS serisini (2018-07-02..bugun) indirir; arsiv kismini ana dosyadan alir; canli kopyaya yazar."""
    base_path, out_path = Path(base_path or BASE_PHYS), Path(out_path or PHYS_LIVE)
    current, _, snap = evds.run_download(client, list(pg.CURRENT), pg.CURRENT_START, "", now=now)
    empty = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"),
                          **{c: pd.Series(dtype=float) for c in pg.ARCHIVE}})   # sayisal tur: birlesimde metne donmesin
    new, _ = pg.build(empty, current)
    base = pd.read_csv(base_path, parse_dates=["Date"])
    out = pd.concat([base[base["kaynak"] == "IAB_arsiv"], new], ignore_index=True).sort_values("Date")
    if out["Date"].duplicated().any():
        raise pg.PhysicalError("Canli fiziki seride tekrar eden tarih.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return {"last_date": str(out["Date"].max().date()), "rows": int(len(out)), "snapshot": snap.name}


def training_frame(phys: pd.DataFrame, panel: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    df = pv.build_frame(phys, panel)
    td = pd.to_datetime(df["target_date"])
    fresh = (pd.to_datetime(df["fx_src_next"]) == td) & (pd.to_datetime(df["gold_src_next"]) >= pd.to_datetime(df["Date"]))
    keep = df["y"].notna() & fresh & (td >= pd.Timestamp(TRAIN_START)) & (td < before)
    return df[keep].reset_index(drop=True), df


def morning_inputs(phys: pd.DataFrame, panel: pd.DataFrame, fx_hourly: pd.DataFrame, now_utc: pd.Timestamp) -> dict:
    T = pd.Timestamp(now_utc.tz_convert(IST).date())
    if T.dayofweek >= 5:
        raise Refused("Hafta sonu: fiziki piyasa kapali.")
    bar_end = pd.Timestamp(T.date(), tz="UTC") + pd.Timedelta(hours=FX_BAR_END_UTC)
    cutoff = pd.Timestamp(T.date(), tz="UTC") + pd.Timedelta(hours=CUTOFF_UTC)
    if now_utc < bar_end:
        raise Refused(f"Erken: kur cubugu {bar_end} UTC'de tamamlaniyor.")
    if now_utc >= cutoff:
        raise Refused(f"Gec: tahmin {cutoff} UTC'den once yapilmali (fiziki kapanisa yakin).")
    if (phys["Date"] >= T).any():
        raise Refused("Hedef gunun fiziki kapanisi zaten yayimlanmis.")
    last = phys[phys["Date"] < T].dropna(subset=["kapanis_tl_gr"]).iloc[-1]
    age = (T - last["Date"]).days
    if age > MAX_AGE_DAYS:
        raise Refused(f"Son fiziki kapanis {last['Date'].date()} ({age} gun): EVDS guncel degil.")
    g = panel[(panel["Date"] < T)].dropna(subset=["gold_usd_oz"]).iloc[-1]
    if g["Date"] < last["Date"]:
        raise Refused(f"Ons verisi ({g['Date'].date()}) son fiziki kapanistan eski: panel guncel degil.")
    bar = fx_hourly[fx_hourly["ts_utc"] == bar_end - pd.Timedelta(hours=1)]
    if bar.empty:
        raise Refused(f"Kur cubugu (bitis {bar_end} UTC) saatlik arsivde yok.")
    fx = float(bar["close"].iat[0])
    M = float(g["gold_usd_oz"]) * fx / 31.1035
    return {"T": T, "last_date": last["Date"], "P": float(last["kapanis_tl_gr"]), "gold_date": g["Date"],
            "gold": float(g["gold_usd_oz"]), "fx": fx, "bar_end": bar_end, "M": M,
            "gap": math.log(M) - math.log(float(last["kapanis_tl_gr"])),
            "horizon": int(np.busday_count(last["Date"].date(), T.date()))}


def forecast(phys: pd.DataFrame, panel: pd.DataFrame, fx_hourly: pd.DataFrame, now_utc: pd.Timestamp) -> dict:
    x = morning_inputs(phys, panel, fx_hourly, now_utc)
    train, full = training_frame(phys, panel, x["T"])
    if len(train) < ROLL_N:
        raise Refused(f"Yetersiz egitim satiri: {len(train)}.")
    prim = full.loc[full["Date"] <= x["last_date"], ["Date", "prim"]].dropna()
    prim_ewma = float(prim["prim"].ewm(halflife=EWMA_HALFLIFE).mean().iat[-1])

    def ols(d):
        X = np.column_stack([np.ones(len(d)), d["gap"].to_numpy(float)])
        return np.linalg.lstsq(X, d["y"].to_numpy(float), rcond=None)[0]
    a1, b1 = ols(train)
    a2, b2 = ols(train.tail(ROLL_N))
    P = x["P"]
    preds = {"RW": P, "Bosluk": P * math.exp(a1 + b1 * x["gap"]), "Bosluk-Kayan": P * math.exp(a2 + b2 * x["gap"]),
             "Prim-Capa": x["M"] * math.exp(prim_ewma)}
    return {**x, "prim_ewma": prim_ewma, "beta_full": float(b1), "beta_roll": float(b2), "n_train": int(len(train)),
            "preds": preds}


# ---------------------------------------------------------------------------
# Kayit
# ---------------------------------------------------------------------------
def read_log(path: Path = None) -> pd.DataFrame:
    path = Path(path or LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLS)
    return pd.read_csv(path, dtype={c: str for c in TEXT})


def verify(log: pd.DataFrame):
    for i, r in log.iterrows():
        if row_hash(r) != r["row_hash"]:
            raise LogTampered(f"Kayit satiri {i} ({r['target_date']}) degistirilmis; islem reddedildi.")


def _write(log: pd.DataFrame, path: Path):
    verify(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    log[ALL_COLS].to_csv(tmp, index=False)
    tmp.replace(path)


def issue(phys: pd.DataFrame, panel: pd.DataFrame, fx_hourly: pd.DataFrame, now_utc, log_path: Path = None,
          prereg_path: Path = None) -> dict:
    log_path, prereg_path = Path(log_path or LOG_PATH), Path(prereg_path or PREREG_PATH)
    if not prereg_path.exists():
        raise Refused("On kayit yok: once 'prepare' calistirin ve commit edin.")
    log = read_log(log_path)
    verify(log)
    now = pd.Timestamp(now_utc)
    T = str(now.tz_convert(IST).date())
    if (log["target_date"].astype(str) == T).any():
        raise Refused(f"{T} icin tahmin zaten kayitli.")
    f = forecast(phys, panel, fx_hourly, now)
    row = {"forecast_id": f"{PROTOCOL_ID}-{T}", "issued_at_utc": now.isoformat(), "protocol_id": PROTOCOL_ID,
           "code_md5": code_md5(), "target_date": T, "last_phys_date": str(f["last_date"].date()),
           "last_phys_price": f["P"], "horizon_days": f["horizon"], "gold_src_date": str(f["gold_date"].date()),
           "gold_usd": f["gold"], "fx_bar_end_utc": str(f["bar_end"]), "usdtry": f["fx"], "M_morning": f["M"],
           "gap": f["gap"], "prim_ewma": f["prim_ewma"], "beta_full": f["beta_full"], "beta_roll": f["beta_roll"],
           "n_train": f["n_train"], **{f"pred_{m}": v for m, v in f["preds"].items()},
           **{c: np.nan for c in REALIZED_COLS}}
    row["row_hash"] = row_hash(pd.Series(row))
    _write(pd.concat([log, pd.DataFrame([row])], ignore_index=True), log_path)
    return row


def score(phys: pd.DataFrame, log_path: Path = None) -> pd.DataFrame:
    log_path = Path(log_path or LOG_PATH)
    log = read_log(log_path)
    if log.empty:
        return log
    verify(log)
    before = log[FORECAST_COLS + ["row_hash"]].astype(str).copy()
    px = phys.dropna(subset=["kapanis_tl_gr"]).set_index(phys["Date"].dt.strftime("%Y-%m-%d"))["kapanis_tl_gr"]
    for i, r in log.iterrows():
        if pd.notna(pd.to_numeric(r["realized_price"], errors="coerce")) or r["target_date"] not in px.index:
            continue
        real = float(px[r["target_date"]])
        log.loc[i, "realized_price"] = real
        for m in MODELS:
            log.loc[i, f"abs_err_{m}"] = abs(real - float(r[f"pred_{m}"]))
    if not log[FORECAST_COLS + ["row_hash"]].astype(str).equals(before):
        raise LogTampered("Puanlama tahmin alanlarini degistirdi.")
    _write(log, log_path)
    return log


def card(log: pd.DataFrame) -> dict:
    s = log[pd.to_numeric(log["realized_price"], errors="coerce").notna()]
    out = {"issued": int(len(log)), "scored": int(len(s)), "pending": int(len(log) - len(s)), "eval_n": EVAL_N}
    if len(s):
        real = s["realized_price"].astype(float)
        for m in MODELS:
            e = s[f"abs_err_{m}"].astype(float)
            out[m] = {"MAE_TL": float(e.mean()), "MAPE_pct": float(100 * (e / real).mean())}
        rw = out["RW"]["MAE_TL"]
        for m in MODELS[1:]:
            out[m]["vs_RW_pct"] = float(100 * (1 - out[m]["MAE_TL"] / rw)) if rw > 0 else None
    return out


# ---------------------------------------------------------------------------
# On kayit ve tek seferlik degerlendirme
# ---------------------------------------------------------------------------
def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "models": MODELS, "train_start": TRAIN_START, "roll_n": ROLL_N,
            "ewma_halflife": EWMA_HALFLIFE, "fx_bar_end_utc": FX_BAR_END_UTC, "cutoff_utc": CUTOFF_UTC,
            "max_age_days": MAX_AGE_DAYS, "eval_n": EVAL_N, "reference": "RW", "multiple": "Holm (3)",
            "alpha": fe.ALPHA, "bootstrap": {"block": fe.BOOT_BLOCK, "reps": fe.BOOT_REPS, "seed": fe.BOOT_SEED}}


def _log_event(event: str, **kw):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"protocol_id": PROTOCOL_ID, "event": event,
                            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw}) + "\n")


def prepare(prereg_path: Path = None) -> str:
    prereg_path = Path(prereg_path or PREREG_PATH)
    content = {"parameters": parameters(), "code_md5": code_md5()}
    if prereg_path.exists():
        old = json.loads(prereg_path.read_text(encoding="utf-8"))
        old.pop("created_at_utc", None)
        if old != content:
            raise Refused("On kayit var ve farkli: protokol degistirilemez (yeni surum kimligi gerekir).")
        return "zaten hazir"
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    prereg_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       **content}, indent=2), encoding="utf-8")
    _log_event("prepared")
    return "olusturuldu"


def evaluate(log: pd.DataFrame, prereg: dict) -> dict:
    s = log[pd.to_numeric(log["realized_price"], errors="coerce").notna()]
    s = s[s["code_md5"] == prereg["code_md5"]].sort_values("target_date")
    if len(s) < EVAL_N:
        raise Refused(f"Degerlendirme icin {EVAL_N} puanlanmis tahmin gerekli; su an {len(s)}.")
    s = s.head(EVAL_N)
    e_rw = s["abs_err_RW"].to_numpy(float)
    res, pvals = {}, {}
    for m in MODELS[1:]:
        e = s[f"abs_err_{m}"].to_numpy(float)
        res[m] = {"bootstrap": fe.paired_bootstrap(e, e_rw), "dm": fe.dm_hln(e, e_rw)}
        pvals[m] = res[m]["dm"]["p_value"]
    order = sorted(pvals, key=pvals.get)
    running = 0.0
    for k, m in enumerate(order):
        running = max(running, min(1.0, pvals[m] * (len(order) - k)))
        r = res[m]
        r["p_holm"] = running
        r["better"] = bool(running < fe.ALPHA and r["dm"]["dm_stat_hln"] < 0 and r["bootstrap"]["diff_ci95"][1] < 0)
    return {"n": int(len(s)), "first": s["target_date"].iat[0], "last": s["target_date"].iat[-1], "results": res}


# ---------------------------------------------------------------------------
def _fx_hourly() -> pd.DataFrame:
    merged, _ = sd.merge_snapshots()
    return merged["usdtry"][["ts_utc", "close"]]


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canli sabah fiziki altin tahmini (LIVE_PHYS_V1).")
    ap.add_argument("command", choices=["prepare", "refresh", "score", "issue", "card", "evaluate"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nCANLI SABAH FIZIKI ALTIN TAHMINI - {PROTOCOL_ID} - {args.command.upper()}\n{sep}")
    try:
        if args.command == "prepare":
            print(f"  On kayit: {prepare()} -> {PREREG_PATH}\n  SIMDI: git add {PREREG_PATH.relative_to(ROOT)}; git commit")
            return 0
        if args.command == "refresh":
            r = refresh(evds.Client(os.environ.get("EVDS_API_KEY", "")))
            print(f"  Canli fiziki seri: son tarih {r['last_date']} | {r['rows']} gun | EVDS goruntu {r['snapshot']}")
            return 0
        phys = pd.read_csv(PHYS_LIVE, parse_dates=["Date"])
        if args.command == "score":
            log = score(phys)
            print(f"  Puanlama tamam: {card(log)['scored']} puanlanmis / {len(log)} kayit")
            return 0
        if args.command == "issue":
            panel = pd.read_csv(PANEL_LIVE, parse_dates=["Date"])
            r = issue(phys, panel, _fx_hourly(), _now())
            print(f"  Hedef: {r['target_date']} fiziki kapanis (TL/gr; toptan piyasa, kuyumcu degil)")
            print(f"  Son bilinen: {r['last_phys_date']} {r['last_phys_price']:.2f} TL | ufuk {r['horizon_days']} gun")
            print(f"  Ons {r['gold_src_date']} {r['gold_usd']:.2f} $ | kur (bitis {r['fx_bar_end_utc']}) {r['usdtry']:.4f} "
                  f"| sabah teorik {r['M_morning']:.2f} TL | bosluk %{100 * r['gap']:+.2f} | prim (yari omur 20) %{100 * r['prim_ewma']:+.2f}")
            for m in MODELS:
                print(f"    {m:13s} {r[f'pred_{m}']:.2f} TL")
            return 0
        log = read_log()
        verify(log)
        if args.command == "card":
            c = card(log)
            print(f"  Kayit: {c['issued']} | puanlanmis: {c['scored']} | bekleyen (pending): {c['pending']} | "
                  f"degerlendirme esigi: {c['eval_n']}")
            for m in MODELS:
                if m in c:
                    extra = f" | RW'ye gore %{c[m]['vs_RW_pct']:+.1f}" if c[m].get("vs_RW_pct") is not None else ""
                    print(f"    {m:13s} MAE {c[m]['MAE_TL']:.2f} TL | MAPE %{c[m]['MAPE_pct']:.3f}{extra}")
            print("  Not: degerlendirme oncesi karne betimleyicidir; model secimi icin KULLANILMAZ.")
            return 0
        prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
        if any(json.loads(l).get("event") == "evaluated" for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l).get("protocol_id") == PROTOCOL_ID):
            print("  REDDEDILDI: degerlendirme zaten yapildi.")
            return 2
        ev = evaluate(log, prereg)
        _log_event("evaluated", n=ev["n"])
        for m, r in ev["results"].items():
            print(f"    {m:13s} iyilesme %{-r['bootstrap']['pct']:+.2f} | p_holm {r['p_holm']:.4f} | RW'den iyi: {r['better']}")
        return 0
    except Refused as e:
        print(f"  REDDEDILDI: {e}")
        return 3
    except (LogTampered, FileNotFoundError, evds.EVDSError, pg.PhysicalError, sd.IntradayError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
