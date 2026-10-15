"""
LIVE_V1 Degerlendirme On Kaydi ve Tek Seferlik Degerlendirme
=============================================================
Kullanim (proje kokunden):
  python -m src.live.forecast_eval prepare     # BIR KEZ: degerlendirme planini kilitle, sonra git commit
  python -m src.live.forecast_eval status      # uygun tahmin sayisi ve uygunsuzluk nedenleri (BASARI GOSTERMEZ)
  python -m src.live.forecast_eval evaluate    # EVAL_N uygun puanlanmis tahminden sonra, TEK SEFER

Neden: LIVE_V1 (src/live/forecast.py, aksam esanli teorik tahmin) ilk tahminini 2026-10-15'te verdi; ancak
degerlendirme kurali dondurulmamisti. Bu modul kurali, ILK SONUC GORULMEDEN once sabitler. src/live/forecast.py
ve LIVE_V1 kaydi DEGISTIRILMEZ; bu modul kaydi yalnizca okur.

On kayit (sonuclardan ONCE):
  Uygun tahmin: system_version == LIVE_V1; asof_date >= 2026-10-15; puanlanmis; gerceklesen cubugun bitisi
    (realized_date 14:00 NY) on kayit zamanindan SONRA; satirin code_commit'inde FORECAST_FILES'in satir
    sonundan bagimsiz md5'leri on kayittakiyle AYNI. Siralama asof_date; ilk EVAL_N uygun satir.
  Birincil nokta: Mean-Return vs yarin=bugun (RW), mutlak TL hata; DM (Newey-West + HLN, t(n-1), iki yonlu)
    ve eslestirilmis hareketli blok bootstrap (blok 20, 5000, tohum 20260101).
    Hukum: DAHA_IYI (GA ust < 0 ve p < alfa), DAHA_KOTU (GA alt > 0 ve p < alfa),
    PRATIKTE_ESDEGER (yuzde fark GA'si -%2..+%2 icinde), aksi AYIRT_EDILEMEDI.
  Birincil aralik: A1 Kupiec p > 0.05 (%80 ve %95), A2 Christoffersen p > 0.05 (%80).
  Ikincil (yalnizca duyarlilik; hukmu degistirmez): vadeli kontrat penceresi disindaki satirlar. Pencere:
    ocak, mart, mayis, temmuz, eylul ve kasim aylarinin son 5 NY is gunu + izleyen ayin ilk
    2 is gunu; asof_date veya realized_date pencerede ise satir dislanir.
Kod kilidi: FORECAST_FILES degisirse sonraki tahminler UYGUN SAYILMAZ (yeni protokol kimligi gerekir).
  EVAL_FILES degisirse degerlendirme reddedilir.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.live import forecast as lf
from src.models import final_exam as fe
from src.models import intervals as iv

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "LIVE_V1"
PREREG_PATH = ROOT / "research" / "preregistration" / "live_v1.json"
REGISTRY_PATH = ROOT / "research" / "registry.jsonl"
RESULT_PATH = ROOT / "research" / "live" / "live_v1_result.json"
FORECAST_FILES = ["src/live/forecast.py", "src/data/sync_dataset.py", "src/models/adaptive.py",
                  "src/models/intervals.py", "src/models/implied_vol.py", "src/models/train.py"]
EVAL_FILES = ["src/live/forecast_eval.py", "src/models/final_exam.py"]
FIRST_ASOF = "2026-10-15"
EVAL_N = 120
ALPHA = 0.05
EQUIV_MARGIN_PCT = 2.0
BOOT = {"block": 20, "reps": 5000, "seed": 20260101}
ROLL_MONTHS = (1, 3, 5, 7, 9, 11)
ROLL_LAST_BDAYS, ROLL_NEXT_BDAYS = 5, 2
NY = "America/New_York"
BAR_NY = "14:00"
MIN_SECONDARY = 30


class Refused(Exception):
    """Kural geregi reddedildi."""


# ---------------------------------------------------------------------------
# Ozetler
# ---------------------------------------------------------------------------
def norm_md5(data: bytes) -> str:
    """Satir sonundan bagimsiz md5 (CRLF -> LF)."""
    return hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest()


def file_md5s(files: list, root: Path = None) -> dict:
    root = Path(root or ROOT)
    return {f: norm_md5((root / f).read_bytes()) for f in files}


def commit_md5s(commit: str, files: list = None, root: Path = None) -> dict:
    """Verilen commit'teki dosyalarin md5'leri; okunamayan dosya None."""
    files, root = files or FORECAST_FILES, Path(root or ROOT)
    out = {}
    for f in files:
        r = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{f}"], capture_output=True)
        out[f] = norm_md5(r.stdout) if r.returncode == 0 else None
    return out


# ---------------------------------------------------------------------------
# On kayit
# ---------------------------------------------------------------------------
def parameters() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "target": "esanli teorik gram (ons x USD/TRY ayni saatlik cubuk, NY 14:00), sonraki esanli gun",
        "first_asof_date": FIRST_ASOF,
        "eval_n": EVAL_N,
        "point": {"primary": "Mean-Return", "reference": "yarin=bugun (RW)", "loss": "mutlak TL hata",
                  "test": "DM Newey-West + HLN, t(n-1), iki yonlu", "equivalence_margin_pct": EQUIV_MARGIN_PCT},
        "interval": {"criteria": ["A1 Kupiec p>0.05 (80, 95)", "A2 Christoffersen p>0.05 (80)"]},
        "alpha": ALPHA,
        "bootstrap": dict(BOOT, type="eslestirilmis hareketli blok"),
        "eligibility": ["system_version == LIVE_V1", "asof_date >= first_asof_date", "puanlanmis",
                        "realized_date 14:00 NY > on kayit zamani", "code_commit'te forecast_code_md5 ayni"],
        "secondary_roll_window": {"months": list(ROLL_MONTHS), "last_bdays": ROLL_LAST_BDAYS,
                                  "next_month_first_bdays": ROLL_NEXT_BDAYS,
                                  "use": "yalnizca duyarlilik; birincil hukmu degistirmez"},
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def prepare(prereg_path: Path = None, root: Path = None) -> str:
    prereg_path = Path(prereg_path or PREREG_PATH)
    content = {"parameters": parameters(), "forecast_code_md5": file_md5s(FORECAST_FILES, root),
               "eval_code_md5": file_md5s(EVAL_FILES, root)}
    if prereg_path.exists():
        old = json.loads(prereg_path.read_text(encoding="utf-8"))
        old.pop("created_at_utc", None)
        if old != content:
            raise Refused("On kayit var ve farkli: protokol degistirilemez (yeni surum kimligi gerekir).")
        return "zaten hazir"
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"created_at_utc": _now(), **content}, indent=2)
    prereg_path.write_text(text, encoding="utf-8")
    _log_event("prepared", prereg_md5=norm_md5(text.encode("utf-8")))
    return "olusturuldu"


# ---------------------------------------------------------------------------
# Uygunluk ve kontrat penceresi
# ---------------------------------------------------------------------------
def in_roll_window(day) -> bool:
    d = pd.Timestamp(day).normalize()
    if d.month in ROLL_MONTHS:
        end = d + pd.offsets.MonthEnd(0)
        return d in set(pd.bdate_range(end - pd.Timedelta(days=20), end)[-ROLL_LAST_BDAYS:])
    if (d.month - 1 if d.month > 1 else 12) in ROLL_MONTHS:
        start = d - pd.offsets.MonthBegin(1) if d.day > 1 else d
        return d in set(pd.bdate_range(start, start + pd.Timedelta(days=10))[:ROLL_NEXT_BDAYS])
    return False


def realized_bar_utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {BAR_NY}").tz_localize(NY).tz_convert("UTC")


def eligible(log: pd.DataFrame, prereg: dict, md5_fn=None) -> tuple:
    """(uygun satirlar asof_date sirali, uygunsuzluk nedeni sayilari)."""
    md5_fn = md5_fn or commit_md5s
    created = pd.Timestamp(prereg["created_at_utc"])
    want = prereg["forecast_code_md5"]
    cache, keep = {}, []
    reasons = {"surum": 0, "erken_tarih": 0, "puanlanmamis": 0, "sonuc_on_kayittan_once": 0, "kod_farkli": 0}
    for i, r in log.iterrows():
        if str(r["system_version"]) != PROTOCOL_ID:
            reasons["surum"] += 1
        elif str(r["asof_date"]) < FIRST_ASOF:
            reasons["erken_tarih"] += 1
        elif pd.isna(pd.to_numeric(r["realized_price"], errors="coerce")):
            reasons["puanlanmamis"] += 1
        elif realized_bar_utc(str(r["realized_date"])) <= created:
            reasons["sonuc_on_kayittan_once"] += 1
        else:
            c = "" if pd.isna(r["code_commit"]) else str(r["code_commit"])
            if c not in cache:
                cache[c] = (md5_fn(c) == want) if c else False
            if cache[c]:
                keep.append(i)
            else:
                reasons["kod_farkli"] += 1
    s = log.loc[keep].copy()
    s = s.assign(_d=s["asof_date"].astype(str)).sort_values("_d").drop(columns="_d")
    return s, reasons


# ---------------------------------------------------------------------------
# Istatistik
# ---------------------------------------------------------------------------
def point_verdict(boot: dict, dm: dict, alpha: float, margin: float) -> str:
    lo, hi = boot["diff_ci95"]
    plo, phi = boot["pct_ci95"]
    if hi < 0 and dm["p_value"] < alpha:
        return "DAHA_IYI"
    if lo > 0 and dm["p_value"] < alpha:
        return "DAHA_KOTU"
    if -margin < plo and phi < margin:
        return "PRATIKTE_ESDEGER"
    return "AYIRT_EDILEMEDI"


def block_stats(s: pd.DataFrame, prm: dict) -> dict:
    e = s["abs_err_TL"].astype(float).to_numpy()
    rw = s["rw_abs_err_TL"].astype(float).to_numpy()
    b = prm["bootstrap"]
    boot = fe.paired_bootstrap(e, rw, block=b["block"], reps=b["reps"], seed=b["seed"])
    dm = fe.dm_hln(e, rw)
    out = {"n": int(len(s)), "MAE_TL": float(e.mean()), "RW_MAE_TL": float(rw.mean()), "bootstrap": boot, "dm": dm,
           "point_verdict": point_verdict(boot, dm, prm["alpha"], prm["point"]["equivalence_margin_pct"])}
    for lvl in (80, 95):
        miss = 1.0 - s[f"in{lvl}"].astype(float).to_numpy()
        k = int(len(miss) - miss.sum())
        out[f"cov{lvl}"] = k / len(miss)
        out[f"cov{lvl}_wilson95"] = fe.wilson(k, len(miss))
        out[f"kupiec_{lvl}_p"] = iv.kupiec_p(miss, 1 - lvl / 100)
        out[f"christoffersen_{lvl}_p"] = iv.christoffersen_p(miss)
    c80 = out["christoffersen_80_p"]
    out["A1_kupiec"] = bool(out["kupiec_80_p"] > prm["alpha"] and out["kupiec_95_p"] > prm["alpha"])
    out["A2_christoffersen_80"] = bool(np.isfinite(c80) and c80 > prm["alpha"])
    return out


def evaluate(log: pd.DataFrame, prereg: dict, md5_fn=None, root: Path = None) -> dict:
    if file_md5s(EVAL_FILES, root) != prereg["eval_code_md5"]:
        raise Refused("Degerlendirme kodu on kayittan farkli; degerlendirme yapilamaz.")
    prm = prereg["parameters"]
    lf.verify_log(log)
    s, reasons = eligible(log, prereg, md5_fn)
    n = prm["eval_n"]
    if len(s) < n:
        raise Refused(f"Degerlendirme icin {n} uygun puanlanmis tahmin gerekli; su an {len(s)}.")
    s = s.head(n)
    res = {"protocol_id": PROTOCOL_ID, "n": n, "first_asof": str(s["asof_date"].iat[0]),
           "last_asof": str(s["asof_date"].iat[-1]), "excluded": reasons, "primary": block_stats(s, prm)}
    flag = s["asof_date"].map(in_roll_window) | s["realized_date"].map(in_roll_window)
    rest = s[~flag.to_numpy(bool)]
    res["secondary_roll_excluded"] = {"dropped": int(flag.sum()),
                                      **(block_stats(rest, prm) if len(rest) >= MIN_SECONDARY else {"n": len(rest)})}
    return res


# ---------------------------------------------------------------------------
def _read_prereg(path: Path = None) -> dict:
    path = Path(path or PREREG_PATH)
    if not path.exists():
        raise Refused("On kayit yok: once 'prepare' calistirin ve commit edin.")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LIVE_V1 degerlendirme on kaydi.")
    ap.add_argument("command", choices=["prepare", "status", "evaluate"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nLIVE_V1 DEGERLENDIRME - {args.command.upper()}\n{sep}")
    try:
        if args.command == "prepare":
            path = Path(PREREG_PATH)
            print(f"  On kayit: {prepare()} -> {path}\n  SIMDI: git add {path.relative_to(ROOT)} "
                  f"research/registry.jsonl; git commit")
            return 0
        prereg = _read_prereg()
        log = lf.read_log()
        lf.verify_log(log)
        if args.command == "status":
            s, reasons = eligible(log, prereg)
            print(f"  Kayit: {len(log)} | uygun puanlanmis: {len(s)} / {prereg['parameters']['eval_n']}")
            print("  Uygun olmayan: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
            print("  Not: basari olcusu gosterilmez; ara sonuclar model secimi icin KULLANILMAZ.")
            return 0
        if _events("evaluated"):
            print("  REDDEDILDI: degerlendirme zaten yapildi.")
            return 2
        res = evaluate(log, prereg)
        out = Path(RESULT_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(res, indent=2, default=float)
        out.write_text(text, encoding="utf-8")
        _log_event("evaluated", n=res["n"], result_md5=norm_md5(text.encode("utf-8")))
        p = res["primary"]
        print(f"  n {res['n']} ({res['first_asof']} .. {res['last_asof']}) | nokta: {p['point_verdict']} "
              f"(fark %{p['bootstrap']['pct']:+.2f}, p {p['dm']['p_value']:.4f})")
        print(f"  Aralik: kapsam %80 {p['cov80']:.3f}, %95 {p['cov95']:.3f} | A1 {p['A1_kupiec']} | "
              f"A2 {p['A2_christoffersen_80']}\n  Sonuc: {out}")
        return 0
    except Refused as e:
        print(f"  REDDEDILDI: {e}")
        return 3
    except (lf.LogTampered, FileNotFoundError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
