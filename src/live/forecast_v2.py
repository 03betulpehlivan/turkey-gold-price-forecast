"""
Canli Tahmin Sistemi, GLD Tabanli (LIVE_V2)
===========================================
Kullanim (proje kokunden; scripts/teorik_v2.ps1 aksam gorevinde, LIVE_V1 adimlarindan SONRA cagirir):
  python -m src.live.forecast_v2 fetch      # GLD saatlik anlik goruntu (data/raw/snapshots_gld/)
  python -m src.live.forecast_v2 prepare    # BIR KEZ: K ve protokol kilitlenir; sonra git commit
  python -m src.live.forecast_v2 score      # gerceklesenleri puanla
  python -m src.live.forecast_v2 issue      # bugunun tahmini (LIVE_V1 ile ayni zaman kurali)
  python -m src.live.forecast_v2 card       # karne
  python -m src.live.forecast_v2 status     # uygun tahmin sayisi (basari gostermez)
  python -m src.live.forecast_v2 evaluate   # 120 uygun puanlanmis tahminden sonra, TEK SEFER

Neden: GC=F (surekli vadeli) kontrat degisim gunlerinde sahte sicrama iceriyor. Onceden kayitli spot taramasi
(research/diagnostics/spot_probe_20261015.json) GLD'yi uygun buldu. LIVE_V2, LIVE_V1 ile AYNI modeli
(src/live/forecast.py: build_frame + compute; ice aktarilir, DEGISTIRILMEZ) GLD tabanli seri uzerinde calistirir.
Tek fark ons kaynagidir; boylece iki protokol ayni modelin iki veri kaynagiyla karsilastirmasidir.

Seri (sonuclardan ONCE sabitlendi):
  Kur: LIVE_V1 esanli serisindeki USD/TRY (bitisi 14:00 NY olan saatlik cubuk; data/processed/sync/sync_daily.csv).
  Ons: GLD saatlik cubugu, bitisi TAM 13:30 NY (ETF cubuklari :30'da baslar; 14:00'e en yakin tamamlanmis cubuk).
       Yoksa gun atilir. ons = GLD x K. K = son 60 ortak gunde (kontrat penceresi disi) GC=F / GLD medyani;
       on kayitta sayi olarak dondurulur. K yalnizca olcektir: RW'ye gore yuzde fark K'dan bagimsizdir.
  Teorik gram = ons x kur / 31.1035. Kuyumcu fiyati degildir.
  Not: ons ile kur arasinda 30 dakika fark vardir (GLD 13:30, kur 14:00).
Tahmin zamani: LIVE_V1 kurali (veri gununun NY takvim gununde, 14:00 NY cubugu kapandiktan sonra).
Degerlendirme: LIVE_V1 ile ayni kurallar (Mean-Return vs RW, DM-HLN, blok bootstrap; A1 Kupiec, A2
  Christoffersen); uygun tahmin = system_version LIVE_V2, puanlanmis, satirdaki kod ozeti on kayittakiyle ayni.
  Ikincil: kontrat penceresi disi satirlar (forecast_eval.in_roll_window).
Kayit: research/live/forecast_v2_log.csv (git). LIVE_V1 kaydina dokunulmaz.
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

from src.data import collect as co
from src.data import collect_intraday as ci
from src.data import sync_dataset as sd
from src.live import forecast as lf
from src.live import forecast_eval as le

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "LIVE_V2"
GLD_TICKER = "GLD"
GLD_ROOT = ROOT / "data" / "raw" / "snapshots_gld"
SYNC_PATH = ROOT / "data" / "processed" / "sync" / "sync_daily.csv"
PANEL_PATH = ROOT / "data" / "processed" / "live" / "market_panel.csv"
FRAME_PATH = ROOT / "data" / "processed" / "sync_v2" / "v2_daily.csv"
LOG_PATH = ROOT / "research" / "live" / "forecast_v2_log.csv"
PREREG_PATH = ROOT / "research" / "preregistration" / "live_v2.json"
REGISTRY_PATH = ROOT / "research" / "registry.jsonl"
RESULT_PATH = ROOT / "research" / "live" / "live_v2_result.json"
CODE_FILES = ["src/live/forecast_v2.py"] + le.FORECAST_FILES + ["src/live/forecast_eval.py", "src/models/final_exam.py"]
GLD_BAR_END_NY = (13, 30)
K_WIN = 60
EVAL_N = 120
PRICE_BASIS = "GLD tabanli teorik gram: GLD (13:30 NY cubugu) x K x USD/TRY (14:00 NY cubugu) / 31.1035; kuyumcu fiyati degil"
FORECAST_COLS = lf.FORECAST_COLS + ["gld_bar_end_utc", "K", "code_md5"]
TEXT_COLS = lf.TEXT_COLS | {"gld_bar_end_utc", "code_md5"}
ALL_COLS = FORECAST_COLS + ["row_hash"] + lf.REALIZED_COLS
EXIT_ALREADY, EXIT_WINDOW, EXIT_REFUSED = 3, 4, 5


class Refused(Exception):
    """Kural geregi reddedildi."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_md5s() -> dict:
    return {f: le.norm_md5((Path(ROOT) / f).read_bytes()) for f in CODE_FILES}


def combined_md5(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GLD saatlik arsiv
# ---------------------------------------------------------------------------
def fetch_gld(root: Path = None, fetch=None, now: datetime = None) -> dict:
    fetch = fetch or ci.fetch_hourly
    snap = co.new_snapshot_dir(Path(root or GLD_ROOT), now)
    m = {"retrieved_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "ticker": GLD_TICKER,
         "interval": "1h", "snapshot_dir": snap.name, "attempts": [], "status": "empty_or_error"}
    for period in ci.PERIOD_ATTEMPTS:
        try:
            df = fetch(GLD_TICKER, period)
        except Exception as e:  # ag veya kaynak hatasi
            m["attempts"].append({"period": period, "result": f"{type(e).__name__}: {e}"})
            continue
        m["attempts"].append({"period": period, "rows": 0 if df is None else int(len(df))})
        if df is not None and len(df):
            path = snap / "gld.csv"
            df.to_csv(path)
            try:
                parsed = ci.read_intraday_csv(path, GLD_TICKER)
                m.update(status="ok", file=path.name, rows=int(len(parsed)),
                         md5=hashlib.md5(path.read_bytes()).hexdigest())
            except ci.IntradayError as e:
                m.update(status="invalid", error=str(e))
            break
    (snap / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def merge_gld(root: Path = None) -> pd.DataFrame:
    """Butun basarili goruntuler; tamamlanmamis cubuk atilir; ayni zaman damgasinda en yeni goruntu."""
    root = Path(root or GLD_ROOT)
    snaps = sorted(p for p in root.iterdir() if (p / "manifest.json").exists()) if root.exists() else []
    parts = []
    for s in snaps:
        m = json.loads((s / "manifest.json").read_text(encoding="utf-8"))
        if m.get("status") != "ok":
            continue
        df = ci.read_intraday_csv(s / m["file"], GLD_TICKER)
        df = df[df["ts_utc"] + pd.Timedelta(hours=1) <= pd.Timestamp(m["retrieved_at_utc"])].copy()
        df["snapshot"] = s.name
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"GLD saatlik arsivi yok: {root} (once 'fetch').")
    allb = pd.concat(parts, ignore_index=True).sort_values(["ts_utc", "snapshot"])
    return allb.drop_duplicates("ts_utc", keep="last")[["ts_utc", "close"]].reset_index(drop=True)


def gld_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """NY gunu basina bitisi TAM 13:30 NY olan cubuk; yoksa gun yok."""
    end_utc = hourly["ts_utc"] + pd.Timedelta(hours=1)
    end = end_utc.dt.tz_convert(lf.NY)
    k = (end.dt.hour == GLD_BAR_END_NY[0]) & (end.dt.minute == GLD_BAR_END_NY[1])
    out = pd.DataFrame({"Date": pd.to_datetime(end[k].dt.date).astype("datetime64[ns]"),
                        "gld_close": hourly.loc[k, "close"].to_numpy(float), "gld_bar_end_utc": end_utc[k].to_numpy()})
    out["gld_bar_end_utc"] = pd.to_datetime(out["gld_bar_end_utc"], utc=True)
    return out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def estimate_k(sync: pd.DataFrame, gd: pd.DataFrame) -> dict:
    m = sync[["Date", "gold_usd_oz"]].assign(Date=lambda d: pd.to_datetime(d["Date"]).astype("datetime64[ns]")) \
        .merge(gd, on="Date", how="inner")
    m = m[~m["Date"].map(le.in_roll_window)].tail(K_WIN)
    if len(m) < K_WIN:
        raise Refused(f"K icin {K_WIN} ortak gun gerekli; {len(m)} var.")
    return {"K": float(np.median(m["gold_usd_oz"] / m["gld_close"])), "first": str(m["Date"].iat[0].date()),
            "last": str(m["Date"].iat[-1].date()), "n": int(len(m))}


def build_v2(sync: pd.DataFrame, gd: pd.DataFrame, K: float) -> pd.DataFrame:
    s = sync.copy()
    s["Date"] = pd.to_datetime(s["Date"]).astype("datetime64[ns]")
    m = s[["Date", "bar_start_utc", "bar_end_utc", "usdtry"]].merge(gd, on="Date", how="inner")
    m["gold_usd_oz"] = m["gld_close"] * K
    m["theoretical_gram_try"] = m["gold_usd_oz"] * m["usdtry"] / 31.1035
    return m.sort_values("Date").reset_index(drop=True)


def load_inputs(sync_path: Path = None, gld_root: Path = None) -> tuple:
    sync = sd.load_sync_daily(Path(sync_path or SYNC_PATH))
    return sync, gld_daily(merge_gld(gld_root))


# ---------------------------------------------------------------------------
# Kayit
# ---------------------------------------------------------------------------
def row_hash(row) -> str:
    parts = []
    for c in FORECAST_COLS:
        v = row[c]
        if c in TEXT_COLS:
            parts.append("" if pd.isna(v) else str(v))
        else:
            parts.append("nan" if pd.isna(v) else f"{float(v):.10g}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def read_log(path: Path = None) -> pd.DataFrame:
    path = Path(path or LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLS)
    return pd.read_csv(path, dtype={c: str for c in TEXT_COLS | {"asof_date", "realized_date"}},
                       float_precision="round_trip")                     # son basamak farki olusmaz


def verify_log(log: pd.DataFrame):
    for i, r in log.iterrows():
        if row_hash(r) != r["row_hash"]:
            raise lf.LogTampered(f"LIVE_V2 kayit satiri {i} ({r['asof_date']}) degistirilmis; yazma reddedildi.")


def _write_log(log: pd.DataFrame, path: Path):
    verify_log(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    log[ALL_COLS].to_csv(tmp, index=False)
    tmp.replace(path)


def _prereg(path: Path = None) -> dict:
    path = Path(path or PREREG_PATH)
    if not path.exists():
        raise Refused("On kayit yok: once 'prepare' ve commit.")
    return json.loads(path.read_text(encoding="utf-8"))


def issue(now_utc=None, sync_path: Path = None, gld_root: Path = None, panel_path: Path = None) -> dict:
    pr = _prereg()
    code = combined_md5(code_md5s())
    if code != pr["code_md5_combined"]:
        raise Refused("Kod on kayittan farkli: LIVE_V2 tahmini kaydedilmez (yeni protokol kimligi gerekir).")
    log_path = Path(LOG_PATH)
    log = read_log(log_path)
    verify_log(log)
    sync, gd = load_inputs(sync_path, gld_root)
    v2 = build_v2(sync, gd, pr["K"]["K"])
    frame_path = Path(FRAME_PATH)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    v2.to_csv(frame_path, index=False)
    panel = pd.read_csv(Path(panel_path or PANEL_PATH), parse_dates=["Date"])
    df = lf.build_frame(v2, panel)
    asof, bar_end = df["Date"].iat[-1], pd.Timestamp(df["bar_end_utc"].iat[-1])
    if (log["asof_date"].astype(str) == str(asof.date())).any():
        raise lf.AlreadyIssued(f"{asof.date()} verisi icin LIVE_V2 tahmini zaten kayitli.")
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp(datetime.now(timezone.utc))
    lf.check_issue_window(asof, bar_end, now)
    c = lf.compute(df)
    p = float(df["theoretical_gram_try"].iat[-1])
    row = {"issued_at_utc": now.isoformat(), "system_version": PROTOCOL_ID, "asof_date": str(asof.date()),
           "asof_bar_end_utc": str(bar_end), "price_basis": PRICE_BASIS, "price_now": p,
           "point_price": p * math.exp(c["mu"]), "point_return": c["mu"], "point_return_se": c["mu_se"],
           "point_price_rw": p, "q80_low": p * math.exp(c["lo80"]), "q80_high": p * math.exp(c["hi80"]),
           "q95_low": p * math.exp(c["lo95"]), "q95_high": p * math.exp(c["hi95"]),
           "aci_k80": c["k80"], "aci_k95": c["k95"], "gvz_lag1": float(df["gvz_lag1"].iat[-1]),
           "train_days": c["train_days"], "code_commit": lf._commit(),
           "sync_md5": hashlib.md5(frame_path.read_bytes()).hexdigest(),
           "gld_bar_end_utc": str(df["gld_bar_end_utc"].iat[-1]), "K": pr["K"]["K"], "code_md5": code,
           "realized_date": "", **{k: np.nan for k in lf.REALIZED_COLS if k != "realized_date"}}
    row["row_hash"] = row_hash(pd.Series(row))
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    _write_log(log, log_path)
    return {"row": row, "backtest": c}


def score(sync_path: Path = None, gld_root: Path = None) -> pd.DataFrame:
    log_path = Path(LOG_PATH)
    log = read_log(log_path)
    if log.empty:
        return log
    verify_log(log)
    pr = _prereg()
    before = log[FORECAST_COLS + ["row_hash"]].astype(str).copy()
    sync, gd = load_inputs(sync_path, gld_root)
    v2 = build_v2(sync, gd, pr["K"]["K"])
    dates = v2["Date"].to_numpy()
    for i, r in log.iterrows():
        if isinstance(r["realized_date"], str) and r["realized_date"]:
            continue
        later = np.where(dates > np.datetime64(pd.Timestamp(r["asof_date"])))[0]
        if not len(later):
            continue
        j = later[0]
        real = float(v2["theoretical_gram_try"].iat[j])
        log.loc[i, "realized_date"] = str(pd.Timestamp(dates[j]).date())
        log.loc[i, "realized_price"] = real
        log.loc[i, "abs_err_TL"] = abs(real - float(r["point_price"]))
        log.loc[i, "rw_abs_err_TL"] = abs(real - float(r["point_price_rw"]))
        log.loc[i, "in80"] = float(float(r["q80_low"]) <= real <= float(r["q80_high"]))
        log.loc[i, "in95"] = float(float(r["q95_low"]) <= real <= float(r["q95_high"]))
    if not log[FORECAST_COLS + ["row_hash"]].astype(str).equals(before):
        raise lf.LogTampered("Puanlama tahmin alanlarini degistirdi; kayit yazilmadi.")
    _write_log(log, log_path)
    return log


# ---------------------------------------------------------------------------
# On kayit ve degerlendirme
# ---------------------------------------------------------------------------
def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "model": "LIVE_V1 ile ayni (src/live/forecast.py build_frame + compute)",
            "gold_source": f"{GLD_TICKER} saatlik, bitisi {GLD_BAR_END_NY[0]:02d}:{GLD_BAR_END_NY[1]:02d} NY",
            "fx_source": "LIVE_V1 esanli serisi USD/TRY (14:00 NY)", "k_window": K_WIN,
            "eval_n": EVAL_N, "alpha": le.ALPHA,
            "point": {"primary": "Mean-Return", "reference": "yarin=bugun (RW)", "loss": "mutlak TL hata",
                      "test": "DM Newey-West + HLN, t(n-1), iki yonlu", "equivalence_margin_pct": le.EQUIV_MARGIN_PCT},
            "interval": {"criteria": ["A1 Kupiec p>0.05 (80, 95)", "A2 Christoffersen p>0.05 (80)"]},
            "bootstrap": dict(le.BOOT, type="eslestirilmis hareketli blok"),
            "eligibility": ["system_version == LIVE_V2", "puanlanmis", "issued_at > on kayit zamani",
                            "satirdaki code_md5 == code_md5_combined"],
            "secondary": "kontrat penceresi disi satirlar (forecast_eval.in_roll_window); hukmu degistirmez"}


def _log_event(event: str, **kw):
    path = Path(REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"protocol_id": PROTOCOL_ID, "event": event, "at_utc": _now(), **kw}) + "\n")


def _events(event: str) -> list:
    path = Path(REGISTRY_PATH)
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("protocol_id") == PROTOCOL_ID and r.get("event") == event]


def prepare(sync_path: Path = None, gld_root: Path = None) -> dict:
    path = Path(PREREG_PATH)
    if path.exists():
        raise Refused("On kayit zaten var; LIVE_V2 degistirilemez (yeni protokol kimligi gerekir).")
    sync, gd = load_inputs(sync_path, gld_root)
    k = estimate_k(sync, gd)
    v2 = build_v2(sync, gd, k["K"])
    md5s = code_md5s()
    content = {"created_at_utc": _now(), "parameters": parameters(), "K": k,
               "history": {"v2_days": int(len(v2)), "first": str(v2["Date"].iat[0].date()),
                           "last": str(v2["Date"].iat[-1].date())},
               "code_md5": md5s, "code_md5_combined": combined_md5(md5s)}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=2)
    path.write_text(text, encoding="utf-8")
    _log_event("prepared", prereg_md5=le.norm_md5(text.encode("utf-8")))
    return content


def eligible(log: pd.DataFrame, pr: dict) -> tuple:
    created = pd.Timestamp(pr["created_at_utc"])
    reasons = {"surum": 0, "puanlanmamis": 0, "on_kayittan_once": 0, "kod_farkli": 0}
    keep = []
    for i, r in log.iterrows():
        if str(r["system_version"]) != PROTOCOL_ID:
            reasons["surum"] += 1
        elif pd.isna(pd.to_numeric(r["realized_price"], errors="coerce")):
            reasons["puanlanmamis"] += 1
        elif pd.Timestamp(r["issued_at_utc"]) <= created:
            reasons["on_kayittan_once"] += 1
        elif str(r["code_md5"]) != pr["code_md5_combined"]:
            reasons["kod_farkli"] += 1
        else:
            keep.append(i)
    s = log.loc[keep].copy()
    s = s.assign(_d=s["asof_date"].astype(str)).sort_values("_d").drop(columns="_d")
    return s, reasons


def evaluate(log: pd.DataFrame, pr: dict) -> dict:
    verify_log(log)
    prm = pr["parameters"]
    s, reasons = eligible(log, pr)
    n = prm["eval_n"]
    if len(s) < n:
        raise Refused(f"Degerlendirme icin {n} uygun puanlanmis tahmin gerekli; su an {len(s)}.")
    s = s.head(n)
    res = {"protocol_id": PROTOCOL_ID, "n": n, "first_asof": str(s["asof_date"].iat[0]),
           "last_asof": str(s["asof_date"].iat[-1]), "excluded": reasons, "primary": le.block_stats(s, prm)}
    flag = (s["asof_date"].map(le.in_roll_window) | s["realized_date"].map(le.in_roll_window)).to_numpy(bool)
    rest = s[~flag]
    res["secondary_roll_excluded"] = {"dropped": int(flag.sum()),
                                      **(le.block_stats(rest, prm) if len(rest) >= le.MIN_SECONDARY else {"n": len(rest)})}
    return res


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canli tahmin, GLD tabanli (LIVE_V2).")
    ap.add_argument("command", choices=["fetch", "prepare", "score", "issue", "card", "status", "evaluate"])
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 76 + f"\nCANLI TAHMIN - {PROTOCOL_ID} (GLD) - {args.command.upper()}\n" + "=" * 76)
    try:
        if args.command == "fetch":
            m = fetch_gld()
            print(f"  GLD saatlik: {m['status']} | {m.get('rows', 0)} cubuk | goruntu {m['snapshot_dir']}")
            return 0 if m["status"] == "ok" else 1
        if args.command == "prepare":
            c = prepare()
            print(f"  K = {c['K']['K']:.6f} ({c['K']['n']} gun, {c['K']['first']} .. {c['K']['last']}) | "
                  f"GLD tabanli seri {c['history']['v2_days']} gun ({c['history']['first']} .. {c['history']['last']})")
            print(f"  On kayit: {Path(PREREG_PATH)}\n  SIMDI: on kaydi, kayit defterini ve kodu commit edin.")
            return 0
        if args.command == "score":
            log = score()
            print(f"  Puanlama tamam. Kayitli tahmin: {len(log)}")
            lf.print_card(lf.scorecard(log))
            return 0
        if args.command == "issue":
            r = issue()["row"]
            print(f"  Veri: {r['asof_date']} | GLD cubugu {r['gld_bar_end_utc']} | kur cubugu {r['asof_bar_end_utc']}")
            print(f"  Bugun {r['price_now']:.2f} TL | yarin Mean-Return {r['point_price']:.2f} | yarin=bugun {r['point_price_rw']:.2f}")
            print(f"  %80 {r['q80_low']:.2f} - {r['q80_high']:.2f} | %95 {r['q95_low']:.2f} - {r['q95_high']:.2f}")
            return 0
        log = read_log()
        verify_log(log)
        if args.command == "card":
            lf.print_card(lf.scorecard(log))
            return 0
        pr = _prereg()
        if args.command == "status":
            s, reasons = eligible(log, pr)
            print(f"  Kayit: {len(log)} | uygun puanlanmis: {len(s)} / {pr['parameters']['eval_n']}")
            print("  Uygun olmayan: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
            return 0
        if _events("evaluated"):
            print("  REDDEDILDI: degerlendirme zaten yapildi.")
            return 2
        res = evaluate(log, pr)
        out = Path(RESULT_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(res, indent=2, default=float)
        out.write_text(text, encoding="utf-8")
        _log_event("evaluated", n=res["n"], result_md5=le.norm_md5(text.encode("utf-8")))
        p = res["primary"]
        print(f"  n {res['n']} | nokta: {p['point_verdict']} | kapsam %80 {p['cov80']:.3f}, %95 {p['cov95']:.3f} | "
              f"A1 {p['A1_kupiec']} | A2 {p['A2_christoffersen_80']}")
        return 0
    except lf.AlreadyIssued as e:
        print(f"  BILGI: {e}")
        return EXIT_ALREADY
    except lf.OutsideIssueWindow as e:
        print(f"  REDDEDILDI: {e}")
        return EXIT_WINDOW
    except Refused as e:
        print(f"  REDDEDILDI: {e}")
        return EXIT_REFUSED
    except lf.LogTampered as e:
        print(f"  KAYIT BUTUNLUGU HATASI: {e}")
        return 1
    except (FileNotFoundError, ValueError, ci.IntradayError, sd.IntradayError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
