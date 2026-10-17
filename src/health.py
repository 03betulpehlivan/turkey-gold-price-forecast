"""
Sistem Sagligi
==============
Kullanim (proje kokunden):
  python -m src.health write --gorev sabah --kod 0 --log logs\\sabah_20261016_1230.log --durum <log>.durum
      # scripts/zamanli_calistir.ps1 her calismanin SONUNDA cagirir; durum kaydini yazar
  python -m src.health show
      # son durum + zaman cizelgesi kontrolu (beklenen calisma olmadiysa HATA)

Durum kaydi: data/processed/health/status.json (Git'e girmez). Her gorev icin SON calisma tutulur.
Kalp atisi: research/live/heartbeat.json (Git'e GIRER; yalnizca zaman, genel durum ve sorunlu denetim adlari).
  GitHub'daki bekci (.github/workflows/watchdog.yml, src/watchdog.py) bunu okuyup gerekirse e-posta ile uyarir.
Bir calismanin denetimleri:
  gorev        betigin cikis kodu (ilk push basarisiz ama esitleme gondermisse UYARI)
  gunluk       kayittaki "UYARI:" satirlari (LIVE_V2, veritabani, yedek ...)
  yedek        uc klasorun yedegi alindi mi
  github       esitleme sonucu ve GitHub'a gonderilmemis commit sayisi
  tahmin       bugunun tahmini kayitta mi (tatil veya kural geregi ret UYARI'dir, HATA degil)
  butunluk     src/audit.py sonucu
Zaman cizelgesi (okuma aninda): hafta ici 12:30 (sabah) ve 22:30 (aksam), Istanbul saati; calisma
  GRACE_MIN dakika icinde kaydedilmemisse HATA. Bilgisayar kapali veya uykudaysa bu boyle gorunur.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "processed" / "health" / "status.json"
HEARTBEAT_PATH = ROOT / "research" / "live" / "heartbeat.json"      # Git'e girer; GitHub'daki bekci okur
TZ = "Europe/Istanbul"
SCHEDULE = {"sabah": (12, 30), "aksam": (22, 30)}
GRACE_MIN = 45
OK, WARN, FAIL = "TAMAM", "UYARI", "HATA"
RANK = {OK: 0, WARN: 1, FAIL: 2}
LOGS = {"sabah": [("research/live/physical_log.csv", "target_date")],
        "aksam": [("research/live/forecast_log.csv", "asof_date"), ("research/live/forecast_v2_log.csv", "asof_date")]}


def worst(statuses) -> str:
    return max(statuses, key=lambda s: RANK[s], default=OK)


def now_tr(now=None) -> pd.Timestamp:
    t = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    return (t if t.tzinfo else t.tz_localize("UTC")).tz_convert(TZ)


def expected_last(task: str, now=None) -> pd.Timestamp:
    """Simdiye kadar kaydedilmis olmasi gereken en son calismanin son tarihi (calisma saati + GRACE_MIN)."""
    t = now_tr(now)
    hh, mm = SCHEDULE[task]
    day = t.normalize()
    for _ in range(8):
        due = day + pd.Timedelta(hours=hh, minutes=mm + GRACE_MIN)
        if day.dayofweek < 5 and due <= t:
            return due
        day -= pd.Timedelta(days=1)
    raise RuntimeError("zaman cizelgesi hesaplanamadi")


# ---------------------------------------------------------------------------
# Calisma sonunda yazilan denetimler
# ---------------------------------------------------------------------------
def parse_log(text: str) -> dict:
    return {"uyarilar": [l.strip() for l in text.splitlines() if l.strip().startswith("UYARI:")],
            "push_hatasi": "HATA: git push basarisiz" in text,
            "yedek_tamam": "YEDEK TAMAM" in text or text.count("Yedek tamam:") >= 3,
            "esitleme_tamam": "ESITLEME TAMAM" in text,
            "esitleme_hata": [l.strip() for l in text.splitlines() if l.strip().startswith("ESITLEME HATA")]}


def _git(root: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def check_forecast(root: Path, task: str, today: str) -> list:
    out = []
    for rel, col in LOGS[task]:
        p = root / rel
        if not p.exists():
            out.append((WARN, "tahmin", f"{rel} yok"))
            continue
        last = pd.read_csv(p, usecols=[col], dtype=str)[col].dropna().max()
        out.append((OK, "tahmin", f"{Path(rel).name}: bugunun tahmini var ({today})") if last == today else
                   (WARN, "tahmin", f"{Path(rel).name}: son tahmin {last}; bugun icin yok (nedeni gunlukte)"))
    return out


def collect(task: str, kod: int, log_text: str, root: Path = None, now=None, audit_fn=None) -> dict:
    root = Path(root or ROOT)
    t = now_tr(now)
    lg = parse_log(log_text)
    checks = []
    if kod == 0:
        checks.append((OK, "gorev", "cikis kodu 0"))
    elif lg["push_hatasi"] and lg["esitleme_tamam"]:
        checks.append((WARN, "gorev", "ilk push basarisizdi; esitleme kaydi GitHub'a gonderdi"))
    else:
        checks.append((FAIL, "gorev", f"cikis kodu {kod} (ayrinti gunlukte)"))
    checks += [(WARN, "gunluk", u) for u in lg["uyarilar"]]
    checks.append((OK, "yedek", "veri yedeklendi") if lg["yedek_tamam"] else (WARN, "yedek", "yedek alinmadi"))
    ahead = _git(root, "rev-list", "--count", "@{u}..HEAD")
    if lg["esitleme_hata"]:
        checks.append((FAIL, "github", lg["esitleme_hata"][0]))
    elif ahead not in ("", "0"):
        checks.append((FAIL, "github", f"GitHub'a gonderilmemis {ahead} commit"))
    else:
        checks.append((OK, "github", "GitHub ile esit"))
    if t.dayofweek < 5:
        checks += check_forecast(root, task, str(t.date()))
    try:
        res = (audit_fn or _audit)(root)
        n_fail = sum(s == "HATA" for s, _, _ in res)
        checks.append((FAIL, "butunluk", f"{n_fail} denetim hatali") if n_fail else
                      (OK, "butunluk", f"{len(res)} denetim, hata yok"))
    except Exception as e:  # denetim calismadiysa bu bir HATA'dir
        checks.append((FAIL, "butunluk", f"denetim calismadi: {type(e).__name__}"))
    return {"task": task, "at": t.isoformat(timespec="seconds"), "kod": int(kod),
            "checks": [{"status": s, "name": n, "detail": d} for s, n, d in checks],
            "overall": worst(s for s, _, _ in checks)}


def _audit(root: Path) -> list:
    from src import audit
    return audit.run_all(root)


def write(entry: dict, path: Path = None) -> Path:
    path = Path(path or STATUS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[entry["task"]] = entry
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def write_heartbeat(entry: dict, path: Path = None) -> Path:
    """Kalp atisi: her gorev icin son calisma zamani, genel durum ve sorunlu denetim adlari.
    Ayrinti ve gizli bilgi icermez. GitHub'daki bekci (src/watchdog.py) bunu okur."""
    path = Path(path or HEARTBEAT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[entry["task"]] = {"at": entry["at"], "overall": entry["overall"], "kod": entry["kod"],
                           "problems": sorted({c["name"] for c in entry["checks"] if c["status"] != OK})}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Okuma aninda: son durum + zaman cizelgesi
# ---------------------------------------------------------------------------
def system_status(now=None, path: Path = None) -> dict:
    path = Path(path or STATUS_PATH)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t = now_tr(now)
    tasks = {}
    for task in SCHEDULE:
        e = data.get(task)
        due = expected_last(task, t)
        if e is None:
            tasks[task] = {"overall": WARN, "at": None, "checks": [
                {"status": WARN, "name": "zaman", "detail": "henuz durum kaydi yok"}]}
            continue
        e = dict(e)
        at = pd.Timestamp(e["at"])
        due_start = due - pd.Timedelta(minutes=GRACE_MIN + 5)
        late = at < due_start
        zaman = {"status": FAIL if late else OK, "name": "zaman",
                 "detail": (f"beklenen calisma yok: {due_start.strftime('%d.%m %H:%M')} sonrasi kayit bekleniyordu"
                            if late else "zamaninda calisti")}
        e["checks"] = [zaman] + list(e["checks"])
        e["overall"] = worst(c["status"] for c in e["checks"])
        tasks[task] = e
    return {"generated_at": t.isoformat(timespec="seconds"), "overall": worst(v["overall"] for v in tasks.values()),
            "tasks": tasks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sistem sagligi.")
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--gorev", choices=list(SCHEDULE), required=True)
    w.add_argument("--kod", type=int, required=True)
    w.add_argument("--log", type=Path, required=True)
    w.add_argument("--durum", type=Path)
    sub.add_parser("show")
    args = ap.parse_args([] if argv is None else argv)
    if args.command == "write":
        text = "\n".join(p.read_text(encoding="utf-8-sig", errors="replace")
                         for p in (args.log, args.durum) if p is not None and p.exists())
        e = collect(args.gorev, args.kod, text)
        print(f"SISTEM DURUMU ({args.gorev}): {e['overall']} -> {write(e)}")
        print(f"  Kalp atisi: {write_heartbeat(e)}")
        for c in e["checks"]:
            print(f"  [{c['status']:5s}] {c['name']}: {c['detail']}")
        return 0
    s = system_status()
    print(f"SISTEM DURUMU: {s['overall']} ({s['generated_at']})")
    for task, e in s["tasks"].items():
        print(f"  {task}: {e['overall']} | son kayit {e.get('at')}")
        for c in e["checks"]:
            if c["status"] != OK:
                print(f"    [{c['status']}] {c['name']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
