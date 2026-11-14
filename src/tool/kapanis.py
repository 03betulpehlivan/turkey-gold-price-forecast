"""
Birinci asamanin dort canli deneyinin (LIVE_V1, LIVE_V2, LIVE_PHYS_V1, LIVE_PHYS_INT_V1) seffaf kapanisi.

  python -m src.tool.kapanis            # ne yazilacagini gosterir (hicbir sey yazmaz)
  python -m src.tool.kapanis --onay     # research/registry.jsonl sonuna "closed" olaylarini ekler

Neden: Bu deneyler "yarinin fiyati yarin = bugun tahmininden iyi tahmin edilebilir mi?" sorusunu canli olarak
sinamak icin acilmisti. Soru, ilk asamanin on kayitli final sinavlariyla yanitlandi (FINAL_EXAM_V1,
FINAL_EXAM_PHYS_V1, BACK_EXAM_ONS_V1: hicbir model yarin = bugunu anlamli bicimde gecmedi). Deneyler bilgisayardaki
birikmis saatlik veriye bagliydi; sistem tamamen buluta tasindigi icin kapatildi.

Kurallar:
  - Ön kayittaki degerlendirme (eval_n = 120 puanli tahmin) YAPILMAZ: kapanista kayit sayisi bunun cok altindadir.
    Sonuclara bakarak karar verilmedi; kapanis nedeni yalnizca altyapi ve arastirma sorusunun yanitlanmis olmasidir.
  - Kayitlar silinmez ve degistirilmez. Kapanis olayi, kayit dosyasinin o anki parmak izini (md5, LF) icerir;
    src/audit.py kapanistan sonra dosyada tek bayt degisirse HATA verir.
  - Kayit defteri yalnizca sona eklenir; kapanis geri alinamaz (deney yeniden gerekirse yeni protokol acilir).
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "research" / "registry.jsonl"
DENEYLER = {  # protokol: (kayit dosyasi, kayit zamani sutunu)
    "LIVE_V1": ("research/live/forecast_log.csv", "issued_at_utc"),
    "LIVE_V2": ("research/live/forecast_v2_log.csv", "issued_at_utc"),
    "LIVE_PHYS_V1": ("research/live/physical_log.csv", "issued_at_utc"),
    "LIVE_PHYS_INT_V1": ("research/live/physical_intervals_log.csv", "issued_at_utc"),
}
NEDEN = ("Bilgisayara bagli altyapi kaldirildi; sistem buluta tasindi. Arastirma sorusu (nokta tahmini yarin = bugunu "
         "gecer mi) on kayitli final sinavlariyla yanitlandi. On kayittaki degerlendirme (eval_n=120) yapilmadi; "
         "kapanis sonuclara bakilarak verilmedi. Kayitlar degistirilmeden korunur.")


def log_md5(data: bytes) -> str:
    return hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest()


def olaylar(root: Path = None) -> list:
    p = Path(root or ROOT) / "research" / "registry.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def plan(root: Path = None, now: datetime = None) -> list:
    root = Path(root or ROOT)
    now = now or datetime.now(timezone.utc)
    kapali = {e["protocol_id"] for e in olaylar(root) if e.get("event") == "closed"}
    out = []
    for pid, (rel, zaman) in DENEYLER.items():
        if pid in kapali:
            continue
        data = (root / rel).read_bytes()
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        out.append({"protocol_id": pid, "event": "closed", "at_utc": now.isoformat(timespec="seconds"),
                    "reason": NEDEN, "evaluated": False, "n_records": len(rows),
                    "last_issued_utc": max((r.get(zaman) or "" for r in rows), default=""),
                    "log": rel, "log_md5": log_md5(data)})
    return out


def yaz(events: list, root: Path = None) -> None:
    p = Path(root or ROOT) / "research" / "registry.jsonl"
    data = p.read_bytes()
    if data and not data.endswith(b"\n"):
        data += b"\n"
    p.write_bytes(data + "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events).encode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Birinci asama canli deneylerinin seffaf kapanisi.")
    ap.add_argument("--onay", action="store_true", help="olaylari kayit defterine yaz (geri alinamaz)")
    args = ap.parse_args([] if argv is None else argv)
    ev = plan()
    print("=" * 70 + "\nCANLI DENEY KAPANISI\n" + "=" * 70)
    if not ev:
        print("  Kapatilacak deney yok (hepsi zaten kapali).")
        return 0
    for e in ev:
        print(f"  {e['protocol_id']:17s} {e['n_records']:3d} kayit | son kayit {e['last_issued_utc'] or '-'} | "
              f"{e['log']} md5 {e['log_md5'][:12]}")
    if not args.onay:
        print("\n  Yalnizca gosterildi. Yazmak icin: python -m src.tool.kapanis --onay")
        return 0
    yaz(ev)
    print(f"\n  {len(ev)} kapanis olayi yazildi: {REGISTRY.relative_to(ROOT)}")
    print("  SIMDI: python -m src.audit ; git add research/registry.jsonl ; git commit ; git push")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
