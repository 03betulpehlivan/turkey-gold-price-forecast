"""
Fiziki Altin Bagimsiz Sinavi (FINAL_EXAM_PHYS_V1)
=================================================
Kullanim (proje kokunden), SIRAYLA:
  1) python -m src.models.physical_exam prepare      # on kayit (hesaplama YOK)
  2) git add src tests research/preregistration/final_exam_phys_v1.json && git commit -m "FINAL_EXAM_PHYS_V1 on kayit"
  3) python -m src.models.physical_exam run          # TEK SEFERLIK
  4) git add research && git commit -m "FINAL_EXAM_PHYS_V1 sonuc"

Protokol (kod sabitleri; sonuclardan ONCE):
  Hedef   : fiziki altin kapanisi TL/gr (EVDS; toptan piyasa, kuyumcu degil), t' gunu SABAHI tahmini.
  Kume    : hedef tarihi >= 2023-01-01 olan tum satirlar (gelistirmede KILITLIYDI; hic degerlendirilmedi).
  Tazelik : kur cubugu tarihi == hedef gun VE ons kaynagi >= onceki fiziki gun; saglamayan satir disarida
            (sayisi raporlanir). Gelistirme degerlendirmesinde bu kural yoktu; sinavda veri gecikmesine karsi eklendi.
  Egitim  : physical_eval.generate (63 gunluk genisleyen pencere, 1 satir purge), egitim tum onceki veri.
  Birincil: Bosluk vs Zero-Return (RW), mutlak TL hata. Kategoriler: DAHA_IYI | DAHA_KOTU |
            PRATIKTE_ESDEGER (+/-%2) | AYIRT_EDILEMEDI (HLN-DM t(n-1) iki yonlu + eslestirilmis blok bootstrap).
  Ikincil (karar disi): Bosluk-Ihtiyatli (guvenli alt sinir), Bosluk+Prim, XGBoost, Mean-Return; yillik kirilim.
  Tek sefer: on kayit git'e commit edilmeden calismaz; kod/veri md5 ve parametreler kilitli;
            research/registry.jsonl'de tamamlanmis sonuc varsa (cikti klasorunden bagimsiz) reddedilir.
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

from src.models import final_exam as fe
from src.models import physical_eval as pv
from src.models import point_eval as pe
from src.models import train as tr

PROTOCOL_ID = "FINAL_EXAM_PHYS_V1"
EXAM_START = "2023-01-01"
PRIMARY, REFERENCE = "Bosluk", "Zero-Return"
SECONDARY = ["Bosluk-Ihtiyatli", "Bosluk+Prim", "XGBoost", "Mean-Return"]

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
PREREG_PATH = RESEARCH / "preregistration" / "final_exam_phys_v1.json"
REGISTRY_PATH = RESEARCH / "registry.jsonl"
RESULT_DIR = RESEARCH / "final_exam_phys_v1"
PHYS_PATH = pv.PHYS_PATH
PANEL_PATH = pv.PANEL_PATH


class ExamError(Exception):
    pass


class ExamAlreadyRun(ExamError):
    pass


def _md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_files() -> dict:
    mods = [pv, pe, tr, fe]
    files = {Path(m.__file__).name: Path(m.__file__) for m in mods}
    files[Path(__file__).name] = Path(__file__)
    return files


def parameters() -> dict:
    return {"protocol_id": PROTOCOL_ID, "exam_start_target_date": EXAM_START, "primary": PRIMARY,
            "reference": REFERENCE, "secondary": SECONDARY, "equivalence_margin_pct": fe.EQUIV_MARGIN_PCT,
            "alpha": fe.ALPHA, "bootstrap": {"block": fe.BOOT_BLOCK, "reps": fe.BOOT_REPS, "seed": fe.BOOT_SEED},
            "freshness": "kur cubugu tarihi == hedef gun; ons kaynagi >= onceki fiziki gun",
            "retrain_every": tr.RETRAIN_EVERY, "purge": "1 satir"}


# ---------------------------------------------------------------------------
def registry() -> list:
    if not REGISTRY_PATH.exists():
        return []
    return [json.loads(l) for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def log(event: str, **kw):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"protocol_id": PROTOCOL_ID, "event": event, "at_utc": _now(), **kw}, ensure_ascii=False) + "\n")


def git_state(paths: list) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    head = g("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"ok": False, "reason": "git deposu yok"}
    rel = [str(Path(p).resolve().relative_to(ROOT.resolve())) for p in paths]
    tracked = g("ls-files", "--error-unmatch", str(PREREG_PATH.resolve().relative_to(ROOT.resolve()))).returncode == 0
    return {"ok": True, "commit": head.stdout.strip(), "dirty": g("status", "--porcelain", "--", *rel).stdout.strip(),
            "prereg_tracked": tracked}


GIT = git_state


def prepare() -> dict:
    for p in (PHYS_PATH, PANEL_PATH):
        if not Path(p).exists():
            raise ExamError(f"Veri yok: {p}")
    content = {"parameters": parameters(), "code_md5": {n: _md5(p) for n, p in code_files().items()},
               "data_md5": {Path(PHYS_PATH).name: _md5(PHYS_PATH), Path(PANEL_PATH).name: _md5(PANEL_PATH)}}
    if PREREG_PATH.exists():
        old = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
        old.pop("created_at_utc", None)
        if old != content:
            raise ExamError("On kayit zaten var ve icerik farkli: protokol degistirilemez (yeni kimlik gerekir).")
        return {"status": "zaten hazir"}
    PREREG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREREG_PATH.write_text(json.dumps({"created_at_utc": _now(), **content}, indent=2, ensure_ascii=False), encoding="utf-8")
    log("prepared", prereg_md5=_md5(PREREG_PATH))
    return {"status": "olusturuldu"}


def fresh_frame(df: pd.DataFrame) -> tuple:
    td = pd.to_datetime(df["target_date"])
    stale = (pd.to_datetime(df["fx_src_next"]) != td) | (pd.to_datetime(df["gold_src_next"]) < pd.to_datetime(df["Date"]))
    exam = (td >= pd.Timestamp(EXAM_START)) & df["y"].notna()
    dropped = df.loc[exam & stale, "target_date"].dt.strftime("%Y-%m-%d").tolist()
    keep = ~(exam & stale)
    return df[keep].reset_index(drop=True), dropped


def evaluate(pred: pd.DataFrame) -> dict:
    e_p, e_r = pe.abs_tl_errors(pred, PRIMARY), pe.abs_tl_errors(pred, REFERENCE)
    boot, dm = fe.paired_bootstrap(e_p, e_r), fe.dm_hln(e_p, e_r)
    metrics = pe.metrics_table(pred)
    secondary = {}
    for m in SECONDARY:
        e = pe.abs_tl_errors(pred, m)
        b = fe.paired_bootstrap(e, e_r)
        secondary[m] = {"improvement_pct": -b["pct"], "pct_ci95": b["pct_ci95"]}
    annual = pe.grouped_table(pred, pd.to_datetime(pred["target_date"]).dt.year.rename("year"), "year")
    return {"primary": {"bootstrap": boot, "dm": dm, "verdict": fe.point_verdict(boot, dm),
                        "improvement_pct": -boot["pct"]},
            "secondary": secondary, "metrics": metrics, "annual": annual}


def run() -> dict:
    if any(e.get("event") == "completed" for e in registry() if e.get("protocol_id") == PROTOCOL_ID):
        log("refused", reason="tamamlanmis sonuc var")
        raise ExamAlreadyRun(f"{PROTOCOL_ID} icin tamamlanmis sonuc zaten var.")
    if not PREREG_PATH.exists():
        log("refused", reason="on kayit yok")
        raise ExamError("On kayit yok. Once: python -m src.models.physical_exam prepare")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    g = GIT([PREREG_PATH] + list(code_files().values()))
    if not g.get("ok") or not g.get("prereg_tracked") or g.get("dirty"):
        log("refused", reason="git", git=g)
        raise ExamError(f"Git kosulu saglanmadi (on kayit ve kod commit edilmeli): {g}")
    now_code = {n: _md5(p) for n, p in code_files().items()}
    now_data = {Path(PHYS_PATH).name: _md5(PHYS_PATH), Path(PANEL_PATH).name: _md5(PANEL_PATH)}
    mism = [k for k, v in now_code.items() if prereg["code_md5"].get(k) != v] + \
           [k for k, v in now_data.items() if prereg["data_md5"].get(k) != v]
    if mism or prereg["parameters"] != parameters():
        log("refused", reason="md5/parametre", files=mism)
        raise ExamError(f"Kod/veri/parametreler on kayitla ayni degil: {mism or 'parametreler'}")
    log("started", commit=g["commit"])

    phys = pd.read_csv(PHYS_PATH, parse_dates=["Date"])
    panel = pd.read_csv(PANEL_PATH, parse_dates=["Date"])
    df = pv.build_frame(phys, panel)
    pv.audit_timing(df)
    df, dropped = fresh_frame(df)
    pred = pv.generate(df, start=EXAM_START, end=None)
    if pd.to_datetime(pred["target_date"]).min() < pd.Timestamp(EXAM_START):
        raise ExamError("Sinav kumesine 2023 oncesi hedef girdi.")
    res = evaluate(pred)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(RESULT_DIR / "predictions.csv", index=False)
    res["annual"].to_csv(RESULT_DIR / "annual.csv", index=False)
    res["metrics"].to_csv(RESULT_DIR / "metrics.csv", index=False)
    result = {"protocol_id": PROTOCOL_ID, "created_at_utc": _now(), "git_commit": g["commit"],
              "prereg_md5": _md5(PREREG_PATH), "n_forecasts": int(len(pred)),
              "first_target": str(pd.Timestamp(pred["target_date"].min()).date()),
              "last_target": str(pd.Timestamp(pred["target_date"].max()).date()),
              "stale_rows_excluded": dropped, "primary": res["primary"], "secondary": res["secondary"]}
    (RESULT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("completed", result_md5=_md5(RESULT_DIR / "result.json"), verdict=res["primary"]["verdict"])
    return {**result, "metrics": res["metrics"], "annual": res["annual"]}


def print_result(r: dict):
    pd.set_option("display.width", 200)
    p = r["primary"]
    print(f"  Sinav: hedef {r['first_target']} .. {r['last_target']} | {r['n_forecasts']} tahmin | "
          f"tazelik nedeniyle disarida: {len(r['stale_rows_excluded'])} {r['stale_rows_excluded'][:8]}")
    m = r["metrics"]
    print(m[["model", "MAE_TL", "MedAE_TL", "RMSE_TL", "MAPE_pct", "direction_accuracy_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    b, d = p["bootstrap"], p["dm"]
    print(f"\n  BIRINCIL (Bosluk vs RW): iyilesme %{p['improvement_pct']:.2f} "
          f"[GA %{-b['pct_ci95'][1]:.2f} .. %{-b['pct_ci95'][0]:.2f}] | DM-HLN {d['dm_stat_hln']:+.2f}, p={d['p_value']:.4f}")
    print(f"  SONUC: {p['verdict']}")
    print("\n  Ikincil (karar disi), RW'ye gore iyilesme:")
    for k, v in r["secondary"].items():
        print(f"    {k:18s} %{v['improvement_pct']:+.2f}  [GA %{-v['pct_ci95'][1]:.2f} .. %{-v['pct_ci95'][0]:.2f}]")
    print("\n  Yillara gore MAPE (%):")
    print(r["annual"].pivot(index="year", columns="model", values="MAPE_pct").to_string(float_format=lambda v: f"{v:.3f}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fiziki altin bagimsiz sinavi (FINAL_EXAM_PHYS_V1).")
    ap.add_argument("command", choices=["prepare", "run"])
    args = ap.parse_args([] if argv is None else argv)
    sep = "=" * 76
    print(f"\n{sep}\nFIZIKI ALTIN BAGIMSIZ SINAVI - {PROTOCOL_ID} - {args.command.upper()}\n{sep}")
    try:
        if args.command == "prepare":
            print(f"  On kayit: {prepare()['status']} -> {PREREG_PATH}")
            print("  SIMDI: git add src tests research/preregistration/final_exam_phys_v1.json; git commit")
            return 0
        r = run()
        print_result(r)
        print(f"\n  Ciktilar: {RESULT_DIR}\n  SIMDI: git add research; git commit")
        return 0
    except ExamAlreadyRun as e:
        print(f"  REDDEDILDI: {e}")
        return 2
    except (ExamError, pv.PhysicalEvalError, FileNotFoundError) as e:
        print(f"  HATA: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
