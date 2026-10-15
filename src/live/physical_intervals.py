"""
Canli Tahmin Araliklari Eklentisi (LIVE_PHYS_INT_V1)
====================================================
Kullanim (proje kokunden; LIVE_PHYS_V1 'issue' adimindan SONRA, ayni sabah):
  python -m src.live.physical_intervals prepare    # BIR KEZ: protokolu kilitle, sonra git commit
  python -m src.live.physical_intervals score      # gerceklesenlerle kapsam
  python -m src.live.physical_intervals issue      # bugunun %80 / %95 araliklari
  python -m src.live.physical_intervals card
  python -m src.live.physical_intervals evaluate   # 120 puanlanmis gunden sonra, TEK SEFER

Ne yapar: LIVE_PHYS_V1'in bugunku dort tahmininin (RW, Bosluk, Bosluk-Kayan, Prim-Capa) etrafina
oynakliga duyarli %80 ve %95 tahmin araliklari ekler. LIVE_PHYS_V1 koduna ve kaydina DOKUNMAZ.

Protokol (sonuclardan once sabitlendi; yontem interval_study kesif sonuclarina bakilarak secildi, bu
yuzden yalnizca ileriye donuk kayitla dogrulanmis sayilir):
  Merkez      : bugunun LIVE_PHYS_V1 kaydindaki tahminler; ayni kodla yeniden hesaplanip esitligi dogrulanir.
  Kalibrasyon : son en fazla 500 is gunu D icin canli kod (physical_forecast.forecast) GERCEK saatlik kur
                arsiviyle (D 04:00 UTC'de baslayan cubuk) yeniden calistirilir; yalnizca kapanisi yayimlanmis
                gunler (D < bugun ve fiziki seride var). En az 300 gun gerekir.
  Yontem      : interval_study.intervals_for "Olcekli+ACI" (EWMA lambda 0.94, son 250 bilinen artik,
                carpimsal ACI eta 0.02).
  Zaman       : LIVE_PHYS_V1 ile ayni pencere (hedef gun, 05:00-11:00 UTC); bugunun V1 kaydi yoksa red.
  Degerlendirme: 120 puanlanmis gun (on kayittaki kod ozetleriyle), TEK SEFER; model x duzey basina kapsam
                (Wilson %95), Kupiec, Christoffersen, Winkler. "Kalibre": Kupiec p > 0.05 ve Christoffersen p > 0.05.
Kayit: research/live/physical_intervals_log.csv (git). Satir basina hash; aralik alanlari degistirilemez.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

from src.data import sync_dataset as sd
from src.live import physical_forecast as lp
from src.models import final_exam as fe
from src.models import interval_study as ist
from src.models import intervals as iv

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "LIVE_PHYS_INT_V1"
LOG_PATH = ROOT / "research" / "live" / "physical_intervals_log.csv"
PREREG_PATH = ROOT / "research" / "preregistration" / "live_phys_int_v1.json"
REGISTRY_PATH = ROOT / "research" / "registry.jsonl"
METHOD = "Olcekli+ACI"
N_CAL, MIN_CAL = 500, 300
LEVELS = (80, 95)
EVAL_N = 120
MODELS = lp.MODELS
BOUND_COLS = [f"{b}{lvl}_{m}" for m in MODELS for lvl in LEVELS for b in ("lo", "hi")]
FORECAST_COLS = ["forecast_id", "issued_at_utc", "protocol_id", "code_md5", "target_date", "v1_forecast_id",
                 "n_cal", "cal_first", "cal_last"] + [f"pred_{m}" for m in MODELS] + BOUND_COLS
REALIZED_COLS = ["realized_price"] + [f"in{lvl}_{m}" for m in MODELS for lvl in LEVELS]
ALL_COLS = FORECAST_COLS + ["row_hash"] + REALIZED_COLS
TEXT = {"forecast_id", "issued_at_utc", "protocol_id", "code_md5", "target_date", "v1_forecast_id", "cal_first", "cal_last"}


class Refused(Exception):
    pass


class LogTampered(RuntimeError):
    pass


def code_md5() -> str:
    """Bu protokolun dayandigi uc dosyanin birlesik ozeti (satir sonu bagimsiz)."""
    h = hashlib.md5()
    for mod in (sys.modules[__name__], lp, ist):
        h.update(Path(mod.__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def row_hash(row) -> str:
    parts = []
    for c in FORECAST_COLS:
        v = row[c]
        parts.append(("" if pd.isna(v) else str(v)) if c in TEXT else ("nan" if pd.isna(v) else f"{float(v):.10g}"))
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
def calibration(phys: pd.DataFrame, panel: pd.DataFrame, fx_hourly: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Canli kodu son N_CAL gun icin gercek saatlik kurla yeniden calistirir (yalnizca yayimlanmis gunler)."""
    bars = set(fx_hourly["ts_utc"])
    days = phys.loc[(phys["Date"] < T) & phys["kapanis_tl_gr"].notna(), ["Date", "kapanis_tl_gr"]]
    days = days[[pd.Timestamp(d.date(), tz="UTC") + pd.Timedelta(hours=lp.FX_BAR_END_UTC - 1) in bars for d in days["Date"]]]
    rows = []
    for D, actual in days.tail(N_CAL + 40).itertuples(index=False):
        try:
            f = lp.forecast(phys[phys["Date"] < D], panel[panel["Date"] < D], fx_hourly,
                            pd.Timestamp(D.date(), tz="UTC") + pd.Timedelta(hours=7))
        except lp.Refused:
            continue
        rows.append({"target_date": D, "last_date": f["last_date"], "actual": float(actual),
                     **{f"pred_{m}": v for m, v in f["preds"].items()}})
    return pd.DataFrame(rows).tail(N_CAL).reset_index(drop=True)


def bounds(cal: pd.DataFrame, today: dict) -> dict:
    """Kalibrasyon dizisi + bugun satiri -> bugunun Olcekli+ACI araliklari."""
    if len(cal) < MIN_CAL:
        raise Refused(f"Yetersiz kalibrasyon gunu: {len(cal)} (en az {MIN_CAL}).")
    row = {"target_date": today["T"], "last_date": today["last_date"], "actual": np.nan,
           **{f"pred_{m}": today["preds"][m] for m in MODELS}}
    df = pd.concat([cal, pd.DataFrame([row])], ignore_index=True)
    out = {}
    for m in MODELS:
        ivd = ist.intervals_for(df, m)
        for lvl in LEVELS:
            lo, hi = ivd[f"lo{lvl}_{METHOD}"].iat[-1], ivd[f"hi{lvl}_{METHOD}"].iat[-1]
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < today["preds"][m] < hi):
                raise Refused(f"Aralik hesaplanamadi ({m}, %{lvl}).")
            out[f"lo{lvl}_{m}"], out[f"hi{lvl}_{m}"] = float(lo), float(hi)
    return out


def read_log(path: Path = None) -> pd.DataFrame:
    path = Path(path or LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLS)
    return pd.read_csv(path, dtype={c: str for c in TEXT})


def verify(log: pd.DataFrame):
    for i, r in log.iterrows():
        if row_hash(r) != r["row_hash"]:
            raise LogTampered(f"Aralik kaydi satir {i} ({r['target_date']}) degistirilmis.")


def _write(log: pd.DataFrame, path: Path):
    verify(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    log[ALL_COLS].to_csv(tmp, index=False)
    tmp.replace(path)


def issue(phys, panel, fx_hourly, now_utc, v1_log: pd.DataFrame, log_path: Path = None, prereg_path: Path = None) -> dict:
    log_path, prereg_path = Path(log_path or LOG_PATH), Path(prereg_path or PREREG_PATH)
    if not prereg_path.exists():
        raise Refused("On kayit yok: once 'prepare' calistirin ve commit edin.")
    log = read_log(log_path)
    verify(log)
    now = pd.Timestamp(now_utc)
    T = str(now.tz_convert(lp.IST).date())
    if (log["target_date"].astype(str) == T).any():
        raise Refused(f"{T} icin aralik zaten kayitli.")
    v1 = v1_log[v1_log["target_date"].astype(str) == T]
    if v1.empty:
        raise Refused(f"{T} icin LIVE_PHYS_V1 tahmini yok (once V1 'issue').")
    v1 = v1.iloc[0]
    f = lp.forecast(phys, panel, fx_hourly, now)                 # zaman penceresi ve veri kurallari V1 ile ayni
    for m in MODELS:
        if not math.isclose(f["preds"][m], float(v1[f"pred_{m}"]), rel_tol=1e-9):
            raise Refused(f"Yeniden hesaplanan {m} tahmini V1 kaydiyla ayni degil (veri degismis olabilir).")
    cal = calibration(phys, panel, fx_hourly, f["T"])
    b = bounds(cal, f)
    row = {"forecast_id": f"{PROTOCOL_ID}-{T}", "issued_at_utc": now.isoformat(), "protocol_id": PROTOCOL_ID,
           "code_md5": code_md5(), "target_date": T, "v1_forecast_id": str(v1["forecast_id"]), "n_cal": int(len(cal)),
           "cal_first": str(pd.Timestamp(cal["target_date"].iat[0]).date()),
           "cal_last": str(pd.Timestamp(cal["target_date"].iat[-1]).date()),
           **{f"pred_{m}": float(v1[f"pred_{m}"]) for m in MODELS}, **b, **{c: np.nan for c in REALIZED_COLS}}
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
        a = float(px[r["target_date"]])
        log.loc[i, "realized_price"] = a
        for m in MODELS:
            for lvl in LEVELS:
                log.loc[i, f"in{lvl}_{m}"] = float(float(r[f"lo{lvl}_{m}"]) <= a <= float(r[f"hi{lvl}_{m}"]))
    if not log[FORECAST_COLS + ["row_hash"]].astype(str).equals(before):
        raise LogTampered("Puanlama aralik alanlarini degistirdi.")
    _write(log, log_path)
    return log


def card(log: pd.DataFrame) -> dict:
    s = log[pd.to_numeric(log["realized_price"], errors="coerce").notna()]
    out = {"issued": int(len(log)), "scored": int(len(s)), "pending": int(len(log) - len(s)), "eval_n": EVAL_N}
    for m in MODELS:
        for lvl in LEVELS:
            if len(s):
                k = int(s[f"in{lvl}_{m}"].astype(float).sum())
                out[f"{m}_{lvl}"] = {"coverage": k / len(s), "wilson95": fe.wilson(k, len(s))}
    return out


def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "method": METHOD, "n_cal": N_CAL, "min_cal": MIN_CAL, "levels": list(LEVELS),
            "ewma_lambda": ist.EWMA_LAMBDA, "window": ist.WINDOW, "eta": ist.ETA, "eval_n": EVAL_N,
            "calibrated_if": "Kupiec p > 0.05 ve Christoffersen p > 0.05", "fx_bar": "saatlik, D 04:00 UTC baslangic"}


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
    s = s[s["code_md5"] == prereg["code_md5"]].sort_values("target_date").head(EVAL_N)
    if len(s) < EVAL_N:
        raise Refused(f"Degerlendirme icin {EVAL_N} puanlanmis gun gerekli; su an {len(s)}.")
    a = s["realized_price"].astype(float).to_numpy()
    res = {}
    for m in MODELS:
        for lvl in LEVELS:
            alpha = 1 - lvl / 100
            L, H = s[f"lo{lvl}_{m}"].astype(float).to_numpy(), s[f"hi{lvl}_{m}"].astype(float).to_numpy()
            inside = (a >= L) & (a <= H)
            miss = (~inside).astype(int)
            wink = (H - L) + (2 / alpha) * ((L - a) * (a < L) + (a - H) * (a > H))
            kp, cp = iv.kupiec_p(miss, alpha), iv.christoffersen_p(miss)
            res[f"{m}_{lvl}"] = {"coverage": float(inside.mean()), "wilson95": fe.wilson(int(inside.sum()), len(a)),
                                 "kupiec_p": kp, "christoffersen_p": cp, "winkler_TL": float(wink.mean()),
                                 "calibrated": bool(kp > 0.05 and cp > 0.05)}
    return {"n": int(len(s)), "first": s["target_date"].iat[0], "last": s["target_date"].iat[-1], "results": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canli tahmin araliklari eklentisi (LIVE_PHYS_INT_V1).")
    ap.add_argument("command", choices=["prepare", "score", "issue", "card", "evaluate"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nCANLI TAHMIN ARALIKLARI - {PROTOCOL_ID} - {args.command.upper()}\n{sep}")
    try:
        if args.command == "prepare":
            print(f"  On kayit: {prepare()} -> {PREREG_PATH}\n  SIMDI: git add {PREREG_PATH.relative_to(ROOT)}; git commit")
            return 0
        phys = pd.read_csv(lp.PHYS_LIVE, parse_dates=["Date"])
        if args.command == "score":
            log = score(phys)
            print(f"  Puanlama tamam: {card(log)['scored']} puanlanmis / {len(log)} kayit")
            return 0
        if args.command == "issue":
            panel = pd.read_csv(lp.PANEL_LIVE, parse_dates=["Date"])
            merged, _ = sd.merge_snapshots()
            r = issue(phys, panel, merged["usdtry"][["ts_utc", "close"]], pd.Timestamp(datetime.now(timezone.utc)),
                      lp.read_log())
            print(f"  Hedef: {r['target_date']} | kalibrasyon: {r['n_cal']} gun ({r['cal_first']} .. {r['cal_last']}) | {METHOD}")
            for m in MODELS:
                print(f"    {m:13s} {r[f'pred_{m}']:.2f} TL | %80: {r[f'lo80_{m}']:.2f} - {r[f'hi80_{m}']:.2f} | "
                      f"%95: {r[f'lo95_{m}']:.2f} - {r[f'hi95_{m}']:.2f}")
            return 0
        log = read_log()
        verify(log)
        if args.command == "card":
            c = card(log)
            print(f"  Kayit: {c['issued']} | puanlanmis: {c['scored']} | bekleyen: {c['pending']} | esik: {c['eval_n']}")
            for m in MODELS:
                for lvl in LEVELS:
                    if f"{m}_{lvl}" in c:
                        x = c[f"{m}_{lvl}"]
                        print(f"    {m:13s} %{lvl}: kapsam {x['coverage']:.3f} {[round(v, 3) for v in x['wilson95']]}")
            print("  Not: degerlendirme oncesi karne betimleyicidir.")
            return 0
        prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
        if REGISTRY_PATH.exists() and any(json.loads(l).get("event") == "evaluated" and json.loads(l).get("protocol_id") == PROTOCOL_ID
                                          for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()):
            print("  REDDEDILDI: degerlendirme zaten yapildi.")
            return 2
        ev = evaluate(log, prereg)
        _log_event("evaluated", n=ev["n"])
        for k, x in ev["results"].items():
            print(f"    {k:18s} kapsam {x['coverage']:.3f} | Kupiec {x['kupiec_p']:.3f} | Christoffersen "
                  f"{x['christoffersen_p']:.3f} | kalibre: {x['calibrated']}")
        return 0
    except (Refused, lp.Refused) as e:
        print(f"  REDDEDILDI: {e}")
        return 3
    except (LogTampered, FileNotFoundError, sd.IntradayError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
