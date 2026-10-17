"""
Bekci (GitHub uzerinde calisir; yalnizca standart kutuphane)
===========================================================
Kullanim:  python3 -m src.watchdog [--gorev sabah|aksam|ikisi] [--simdi 2026-10-19T11:20:00+00:00]
GitHub Actions (.github/workflows/watchdog.yml) hafta ici 14:20 ve 23:15 (Istanbul) calistirir. Bilgisayardan
BAGIMSIZDIR: bilgisayar kapali, uykuda veya kilitliyse kalp atisi gelmez ve bekci basarisiz olur; GitHub
basarisiz is akislari icin depo sahibine e-posta gonderir.
Okunan dosya: research/live/heartbeat.json (src/health.py her gorevin sonunda yazar, sarmalayici gonderir).
Kural: hafta ici, gorevin bugunku planli saatinden (12:30 / 22:30) sonra kalp atisi yoksa veya genel durum
  HATA ise basarisiz. UYARI (ornegin resmi tatilde tahmin uretilmemesi) basarisizlik DEGILDIR.
Cikis: 0 = sorun yok; 1 = sorun var (GitHub e-posta gonderir).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_PATH = ROOT / "research" / "live" / "heartbeat.json"
TR = timezone(timedelta(hours=3))                       # Turkiye: sabit UTC+3, yaz saati yok
SCHEDULE = {"sabah": (12, 30), "aksam": (22, 30)}
CHECK_AFTER_MIN = 40                                    # planli saatten bu kadar sonra denetlenir


def scheduled_today(task: str, now: datetime):
    """Bugunun planli calisma zamani (Istanbul); hafta sonu veya henuz zamani gelmediyse None."""
    t = now.astimezone(TR)
    if t.weekday() >= 5:
        return None
    hh, mm = SCHEDULE[task]
    start = t.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return start if t >= start + timedelta(minutes=CHECK_AFTER_MIN) else None


def auto_task(now: datetime) -> str:
    return "sabah" if now.astimezone(TR).hour < 18 else "aksam"


def check(heartbeat, task: str, now: datetime) -> list:
    start = scheduled_today(task, now)
    if start is None:
        return []
    if heartbeat is None:
        return [f"kalp atisi dosyasi yok ({HEARTBEAT_PATH.name})"]
    e = heartbeat.get(task)
    if not e:
        return [f"{task} gorevi icin hic kalp atisi yok"]
    at = datetime.fromisoformat(e["at"])
    problems = []
    if at < start:
        problems.append(f"{task} gorevi bugun calismadi: son kalp atisi {at.astimezone(TR):%d.%m %H:%M}, "
                        f"beklenen {start:%d.%m %H:%M} sonrasi (bilgisayar kapali, uykuda veya gorev cokmus olabilir)")
    elif e.get("overall") == "HATA":
        problems.append(f"{task} gorevi HATA ile bitti (cikis kodu {e.get('kod')}; sorunlu: {', '.join(e.get('problems', []))})")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bekci: kalp atisi denetimi.")
    ap.add_argument("--gorev", choices=["sabah", "aksam", "ikisi", "otomatik"], default="otomatik")
    ap.add_argument("--simdi", default=None, help="test icin ISO zaman")
    args = ap.parse_args([] if argv is None else argv)
    now = datetime.fromisoformat(args.simdi) if args.simdi else datetime.now(timezone.utc)
    path = Path(HEARTBEAT_PATH)
    hb = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    tasks = {"ikisi": ["sabah", "aksam"], "otomatik": [auto_task(now)]}.get(args.gorev, [args.gorev])
    problems = [p for t in tasks for p in check(hb, t, now)]
    print(f"BEKCI {now.astimezone(TR):%Y-%m-%d %H:%M} (Istanbul) | denetlenen: {', '.join(tasks)}")
    for t in tasks:
        e = (hb or {}).get(t)
        print(f"  {t}: son kalp atisi {e['at'] if e else 'yok'} | durum {e['overall'] if e else '-'}")
    if problems:
        for p in problems:
            print(f"  SORUN: {p}")
        return 1
    print("  Sorun yok.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
