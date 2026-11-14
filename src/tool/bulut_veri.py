"""
Bulut gorevi icin guncel fiziki gram altin serisi (sabah_bulut.yml).

  python -m src.tool.bulut_veri          # TCMB EVDS'den guncel seriyi indirir, canli kopyaya yazar

Bilgisayarda canli kopya src.live.physical_forecast.refresh ile uretilir: 2018 sonrasi EVDS serisi + ana dosyadaki
2018 oncesi arsiv. Bulutta ana (arsiv) dosya yoktur (data/ depoda izlenmez). Bu modul:
  - ana dosya VARSA physical_forecast.refresh'i aynen cagirir (bilgisayardaki davranis),
  - YOKSA ayni EVDS serisini ayni kurucuyla (physical_gold.build) indirip arsivsiz yazar.
Aralik modelleri son 750 islem gununu, Chronos son 512 gunu kullandigi icin 2018 oncesi arsiv tahmini degistirmez
(tests/test_tool_yarisma.TestLive.test_live_is_unchanged_without_old_archive).

src/live/physical_forecast.py on kayitli protokollerin (LIVE_PHYS_V1, LIVE_PHYS_INT_V1) KILITLI kodudur; bu modul
onu degistirmez, yalnizca kullanir.
"""
import os
import sys
from pathlib import Path

import pandas as pd

from src.data import evds
from src.data import physical_gold as pg
from src.live import physical_forecast as lp


def guncelle(client, base_path: Path = None, out_path: Path = None, now=None) -> dict:
    base_path, out_path = Path(base_path or lp.BASE_PHYS), Path(out_path or lp.PHYS_LIVE)
    if base_path.exists():
        return {**lp.refresh(client, base_path, out_path, now=now), "arsiv": True}
    current, _, snap = evds.run_download(client, list(pg.CURRENT), pg.CURRENT_START, "", now=now)
    empty = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), **{c: pd.Series(dtype=float) for c in pg.ARCHIVE}})
    out, _ = pg.build(empty, current)
    out = out.sort_values("Date").reset_index(drop=True)
    if out["Date"].duplicated().any():
        raise pg.PhysicalError("Canli fiziki seride tekrar eden tarih.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return {"last_date": str(out["Date"].max().date()), "rows": int(len(out)), "snapshot": snap.name, "arsiv": False}


def main(argv=None) -> int:
    print("=" * 70 + "\nBULUT GUNCEL FIYAT (TCMB EVDS)\n" + "=" * 70)
    try:
        r = guncelle(evds.Client(os.environ.get("EVDS_API_KEY", "")))
    except (evds.EVDSError, pg.PhysicalError) as e:
        print(f"  HATA: {type(e).__name__}: {e}")
        return 1
    print(f"  Son tarih {r['last_date']} | {r['rows']} gun | EVDS goruntu {r['snapshot']}"
          + ("" if r["arsiv"] else " | arsivsiz (2018 sonrasi)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
