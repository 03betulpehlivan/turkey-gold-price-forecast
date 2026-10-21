"""
Ileriye Donuk Karne (herkese acik): her sabah verilen 1/3/5 gunluk araliklar, gercek fiyat belli olmadan
once kaydedilir ve sonra gerceklesen fiyatla puanlanir.
==========================================================================================================
Kullanim (proje kokunden):
  python -m src.tool.canli_karne kaydet     # son kapanisa gore 1/3/5 gun araliklarini kaydeder (sabah gorevi cagirir)
  python -m src.tool.canli_karne karne      # simdiye kadar puanlanan tahminlerin tutma oranlari

Kayit: research/tool/canli_aralik_log.csv (yalnizca sonuna eklenir; her satirin SHA-256 ozeti var).
Zaman kurali: bir kayit, bilgi gununden sonraki ilk is gununun 14:00'unden (Istanbul) ONCE yapilmalidir;
  aksi halde kaydedilmez (fiziki kapanis o saatten sonra belli olur). Hafta sonu kaydi yapilmaz.
Puanlama kayit dosyasina YAZILMAZ: gerceklesen fiyat her okumada fiyat dosyasindan bulunur
  (bilgi gunundeki konumdan h islem gunu sonrasi). Ozeti tutmayan satirlar karneye girmez.
Modeller ana ekrandakiyle ayni: her ufukta model yarismasinin kazanani (src/tool/service.ufuklar).
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import io
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "research" / "tool" / "canli_aralik_log.csv"
TZ = "Europe/Istanbul"
SON_SAAT = 14
COLS = ["kayit_utc", "bilgi_gunu", "ufuk_gun", "model", "kullanilan", "son_kapanis", "lo80", "hi80", "lo95", "hi95", "satir_ozeti"]


def _ozet(row: dict) -> str:
    s = "|".join(str(row[c]) for c in COLS[:-1])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def oku(path: Path = None) -> pd.DataFrame:
    p = Path(path or LOG)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def dogrula(log: pd.DataFrame) -> np.ndarray:
    return np.array([_ozet(r) == r["satir_ozeti"] for r in log.to_dict("records")], dtype=bool)


def son_zaman(bilgi_gunu: str) -> pd.Timestamp:
    return (pd.Timestamp(bilgi_gunu) + pd.offsets.BDay(1)).tz_localize(TZ) + pd.Timedelta(hours=SON_SAAT)


def kaydet(ufuk: dict, now: datetime = None, path: Path = None) -> dict:
    """ufuk: service.ufuklar() ciktisi. Donus: {"eklenen": n, "neden": ...}."""
    p = Path(path or LOG)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now if now.tzinfo else now.tz_localize("UTC")
    bilgi = ufuk["bilgi_gunu"]
    if now.tz_convert(TZ) >= son_zaman(bilgi):
        return {"eklenen": 0, "neden": f"gec: {bilgi} kapanisina gore kayit {son_zaman(bilgi):%d.%m %H:%M} oncesinde yapilmaliydi"}
    if now.tz_convert(TZ).dayofweek >= 5:
        return {"eklenen": 0, "neden": "hafta sonu"}
    log = oku(p)
    var = set(zip(log["bilgi_gunu"], log["ufuk_gun"])) if len(log) else set()
    yeni = []
    for u in ufuk["ufuklar"]:
        if (bilgi, str(u["ufuk_gun"])) in var:
            continue
        r = {"kayit_utc": now.tz_convert("UTC").isoformat(timespec="seconds"), "bilgi_gunu": bilgi, "ufuk_gun": str(u["ufuk_gun"]),
             "model": u["model"], "kullanilan": u["kullanilan"], "son_kapanis": f"{ufuk['son_kapanis']:.4f}",
             "lo80": f"{u['aralik80'][0]:.4f}", "hi80": f"{u['aralik80'][1]:.4f}",
             "lo95": f"{u['aralik95'][0]:.4f}", "hi95": f"{u['aralik95'][1]:.4f}"}
        r["satir_ozeti"] = _ozet(r)
        yeni.append(r)
    if not yeni:
        return {"eklenen": 0, "neden": "bu gunun kaydi zaten var"}
    p.parent.mkdir(parents=True, exist_ok=True)
    bas = not p.exists()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n")
    if bas:
        w.writeheader()
    w.writerows(yeni)
    with open(p, "a", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    return {"eklenen": len(yeni), "neden": ""}


def karne(prices: pd.Series, path: Path = None, son: int = 15) -> dict:
    log = oku(path)
    if len(log) == 0:
        return {"kayit": 0, "ufuklar": [], "son_kayitlar": [], "baslangic": None, "bozuk": 0}
    ok = dogrula(log)
    log = log[ok].copy()
    dates = prices.index
    pos = {str(t.date()): i for i, t in enumerate(dates)}
    rows = []
    for r in log.to_dict("records"):
        h, i = int(r["ufuk_gun"]), pos.get(r["bilgi_gunu"])
        g = gun = None
        if i is not None and i + h < len(dates):
            g, gun = float(prices.iloc[i + h]), str(dates[i + h].date())
        f = {k: float(r[k]) for k in ("lo80", "hi80", "lo95", "hi95")}
        rows.append({"bilgi_gunu": r["bilgi_gunu"], "ufuk_gun": h, "model": r["kullanilan"], **f, "hedef_gun": gun, "gercek": g,
                     "in80": None if g is None else bool(f["lo80"] <= g <= f["hi80"]),
                     "in95": None if g is None else bool(f["lo95"] <= g <= f["hi95"])})
    df = pd.DataFrame(rows)
    ozet = []
    for h, gr in df.groupby("ufuk_gun"):
        sc = gr[gr["gercek"].notna()]
        ozet.append({"ufuk_gun": int(h), "kayit": int(len(gr)), "puanlanan": int(len(sc)),
                     "tuttu80": int(sc["in80"].sum()) if len(sc) else 0, "tuttu95": int(sc["in95"].sum()) if len(sc) else 0,
                     "tutma80": float(100 * sc["in80"].mean()) if len(sc) else None,
                     "tutma95": float(100 * sc["in95"].mean()) if len(sc) else None})
    sonlar = sorted(rows, key=lambda r: (r["bilgi_gunu"], r["ufuk_gun"]))[-son:]          # None korunur (NaN degil)
    return {"kayit": int(len(df)), "bozuk": int((~ok).sum()), "baslangic": min(df["bilgi_gunu"]), "ufuklar": ozet,
            "son_kayitlar": sonlar}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ileriye donuk karne (1/3/5 gun araliklari).")
    ap.add_argument("komut", choices=["kaydet", "karne"])
    args = ap.parse_args([] if argv is None else argv)
    from src.tool import karne as k
    from src.tool import service as ts
    print("=" * 70 + "\nILERIYE DONUK KARNE - " + args.komut.upper() + "\n" + "=" * 70)
    if args.komut == "kaydet":
        r = kaydet(ts.ufuklar())
        print(f"  Eklenen satir: {r['eklenen']}" + (f" ({r['neden']})" if r["neden"] else "") + f" | {LOG}")
        return 0 if r["eklenen"] or r["neden"] in ("bu gunun kaydi zaten var", "hafta sonu") else 1
    c = karne(k.load_series())
    print(f"  Kayit: {c['kayit']} satir, baslangic {c['baslangic']}, ozeti tutmayan {c['bozuk']}")
    for u in c["ufuklar"]:
        t80 = "-" if u["tutma80"] is None else f"%{u['tutma80']:.1f}"
        t95 = "-" if u["tutma95"] is None else f"%{u['tutma95']:.1f}"
        print(f"  {u['ufuk_gun']} gun: puanlanan {u['puanlanan']}/{u['kayit']} | %80 aralik {u['tuttu80']} kez ({t80}) | %95 {u['tuttu95']} kez ({t95})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
