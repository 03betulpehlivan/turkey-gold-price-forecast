"""
Olay Takvimi: Fed (FOMC) planli toplanti karar gunlerini resmi siteden indirir
=============================================================================
Kullanim (proje kokunden, internet gerekir):
  python -m src.data.olay_indir fed                 # 2012..bugun; research/tool/olay_takvimi.csv'ye ekler
  python -m src.data.olay_indir fed --kuru          # yazmadan once yalnizca gosterir

Kaynak: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (son yillar) ve
        https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm (eski yillar).
Kurallar:
  - Karar gunu = toplanti araliginin SON gunu (ornek "January 30-31" -> 31 Ocak; "Apr/May 30-1" -> 1 Mayis).
  - YALNIZCA planli toplantilar alinir. "unscheduled" ya da "conference call" olanlar alinmaz (onceden bilinmezler).
  - Mevcut takvimdeki diger turler (ABD_TUFE, TCMB_PPK) korunur; ayni tarih ve tur iki kez yazilmaz.
Sitenin sayfa yapisi degisebilir: betik yillara gore toplanti sayisini yazar (normalde yilda 8).
"""

from datetime import date
from pathlib import Path
import argparse
import re
import sys
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAKVIM = ROOT / "research" / "tool" / "olay_takvimi.csv"
GUNCEL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
TARIHSEL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{yil}.htm"
AYLAR = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                     "september", "october", "november", "december"], start=1)}
KISA = {k[:3]: v for k, v in AYLAR.items()}
BASLIK = ("# Olay takvimi: onceden ilan edilen, oynakligi artirabilecek olaylar. Her satir bir olay.\n"
          "# tarih: YYYY-AA-GG (olayin aciklandigi gun, Turkiye saatine gore)\n"
          "# tur: FED (FOMC faiz karari), ABD_TUFE (ABD enflasyon verisi), TCMB_PPK (TCMB faiz karari)\n"
          "# kaynak: tarihin alindigi resmi sayfa (federalreserve.gov, bls.gov, tcmb.gov.tr)\n")


def indir(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gram-altin bitirme projesi)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def _ay(s: str) -> int:
    s = s.strip().lower().rstrip(".")
    return AYLAR.get(s) or KISA.get(s[:3])


def karar_gunu(yil: int, ay_metni: str, gun_metni: str) -> date:
    """'January','30-31' -> 31 Ocak; 'Apr/May','30-1' -> 1 Mayis; 'March','15' -> 15 Mart."""
    aylar = [a for a in re.split(r"[/\-]", ay_metni) if a.strip()]
    gunler = [int(g) for g in re.findall(r"\d{1,2}", gun_metni)]
    if not aylar or not gunler:
        raise ValueError(f"Cozumlenemedi: {ay_metni!r} {gun_metni!r}")
    ay = _ay(aylar[-1])
    if ay is None:
        raise ValueError(f"Ay taninmadi: {ay_metni!r}")
    return date(yil, ay, gunler[-1])


def _temiz(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def ayristir_guncel(html: str) -> list:
    """fomccalendars.htm: her yil bolumu ("2024 FOMC Meetings") icinde ay ve gun hucreleri."""
    out = []
    parcalar = re.split(r"(\d{4})\s+FOMC\s+Meetings", html)
    for i in range(1, len(parcalar) - 1, 2):
        yil, govde = int(parcalar[i]), parcalar[i + 1]
        aylar = re.findall(r'fomc-meeting__month[^>]*>(.*?)</div>', govde, flags=re.S)
        gunler = re.findall(r'fomc-meeting__date[^>]*>(.*?)</div>', govde, flags=re.S)
        for a, g in zip(aylar, gunler):
            a_t, g_t = _temiz(a).strip(), _temiz(g).strip()
            if re.search(r"unscheduled|conference call|notation", g_t + " " + a_t, flags=re.I):
                continue
            try:
                out.append(karar_gunu(yil, a_t, g_t))
            except ValueError:
                continue
    return out


def ayristir_tarihsel(html: str, yil: int) -> list:
    """fomchistoricalYYYY.htm: basliklar "January 27-28 Meeting - 2015" bicimindedir."""
    out = []
    metin = _temiz(html)
    desen = re.compile(r"([A-Z][a-z]+(?:/[A-Z][a-z]+)?)\s+(\d{1,2}(?:-\d{1,2})?)\s+(\(unscheduled\)\s+)?"
                       r"(Meeting|Conference Call)\s*-\s*(\d{4})")
    for m in desen.finditer(metin):
        if int(m.group(5)) != yil or m.group(3) or m.group(4) != "Meeting":
            continue
        try:
            out.append(karar_gunu(yil, m.group(1), m.group(2)))
        except ValueError:
            continue
    return sorted(set(out))


def fed_tarihleri(bas: int, son: int, getir=indir) -> dict:
    sonuc, kaynak = {}, {}
    guncel = ayristir_guncel(getir(GUNCEL))
    for t in guncel:
        if bas <= t.year <= son:
            sonuc.setdefault(t.year, set()).add(t)
            kaynak[t] = GUNCEL
    for yil in range(bas, son + 1):
        if yil in sonuc:
            continue
        try:
            tt = ayristir_tarihsel(getir(TARIHSEL.format(yil=yil)), yil)
        except Exception:
            tt = []
        if tt:
            sonuc[yil] = set(tt)
            for t in tt:
                kaynak[t] = TARIHSEL.format(yil=yil)
    return {"tarihler": sorted(t for s in sonuc.values() for t in s), "kaynak": kaynak,
            "yillar": {y: len(sonuc.get(y, ())) for y in range(bas, son + 1)}}


def takvime_ekle(tarihler: list, kaynak: dict, tur: str = "FED", path: Path = None) -> int:
    p = Path(path or TAKVIM)
    mevcut = pd.read_csv(p, dtype=str, comment="#") if p.exists() else pd.DataFrame(columns=["tarih", "tur", "kaynak"])
    var = set(zip(mevcut["tarih"], mevcut["tur"]))
    yeni = [{"tarih": t.isoformat(), "tur": tur, "kaynak": kaynak.get(t, "")} for t in tarihler if (t.isoformat(), tur) not in var]
    hepsi = pd.concat([mevcut, pd.DataFrame(yeni, columns=["tarih", "tur", "kaynak"])], ignore_index=True)
    hepsi = hepsi.sort_values(["tarih", "tur"]).reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(BASLIK + hepsi.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="")
    return len(yeni)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Olay takvimi indirici.")
    ap.add_argument("tur", choices=["fed"])
    ap.add_argument("--bas", type=int, default=2012)
    ap.add_argument("--son", type=int, default=date.today().year)
    ap.add_argument("--kuru", action="store_true", help="takvime yazma, yalnizca goster")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 70 + "\nOLAY TAKVIMI - FED (FOMC planli toplantilar)\n" + "=" * 70)
    r = fed_tarihleri(args.bas, args.son)
    for y, n in r["yillar"].items():
        isaret = "" if 7 <= n <= 9 else "   <- KONTROL ET (normalde 8)"
        print(f"  {y}: {n} toplanti{isaret}")
    print(f"  Toplam {len(r['tarihler'])} karar gunu. Ilk uc: {[t.isoformat() for t in r['tarihler'][:3]]}")
    if args.kuru:
        return 0
    n = takvime_ekle(r["tarihler"], r["kaynak"])
    print(f"  Takvime eklenen: {n} | {TAKVIM}")
    return 0 if r["tarihler"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
