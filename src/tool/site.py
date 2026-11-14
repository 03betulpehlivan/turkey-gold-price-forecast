"""
Herkese acik site (GitHub Pages): ana arayuzun sabit (statik) kopyasi.

  python -m src.tool.site              # _site/ klasorunu uretir (bulut gorevi her sabah cagirir)
  python -m src.tool.site --cikti X    # baska klasore

Bilgisayardaki arayuz (scripts/pano.ps1) verileri FastAPI'den canli ister. GitHub Pages yalnizca sabit dosya
sunar; bu modul ayni API'yi (src/api/app.py) bir kez cagirip yanitlarini JSON dosyasi olarak yazar ve sayfanin
dosyalarini bu dosyalari okuyacak bicimde kopyalar. Kaynak dosyalar (src/web/) DEGISMEZ:
  - "/static/..." ve "/api/tahmin/X" mutlak yollari goreli yola cevrilir (site bir alt klasorde yayimlanir),
  - sunucu gerektiren baglantilar (ilk asama panosu, API belgeleri) GitHub deposu ve raporlarla degistirilir.
Veri yoksa ya da bir uc nokta hata verirse site URETILMEZ (yarim site yayimlanmaz).
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "src" / "web"
CIKTI = ROOT / "_site"
UCLAR = ("ufuklar", "bu-yil-yarisma", "canli-karne", "yarisma", "merkez")
DEPO = "https://github.com/03betulpehlivan/turkey-gold-price-forecast"
ESKI_BAGLANTI = re.compile(r'<p class="baglantilar">.*?</p>', re.S)
YENI_BAGLANTI = (f'<p class="baglantilar"><a href="{DEPO}">Kaynak kod ve kayıtlar (GitHub)</a> '
                 f'<a href="{DEPO}/blob/main/docs/Gram_Altin_Proje_Raporu.pdf">Proje raporu</a> '
                 f'<a href="{DEPO}/blob/main/docs/Literatur_Taramasi_Raporu.pdf">Literatür taraması</a></p>')


class SiteHatasi(RuntimeError):
    pass


def veriler(client=None) -> dict:
    if client is None:
        from fastapi.testclient import TestClient
        from src.api import app as api
        client = TestClient(api.app, base_url="http://127.0.0.1")
    out = {}
    for u in UCLAR:
        r = client.get(f"/api/tahmin/{u}")
        if r.status_code != 200:
            raise SiteHatasi(f"/api/tahmin/{u}: HTTP {r.status_code} {r.text[:200]}")
        out[u] = r.json()
    return out


def sayfa(html: str) -> str:
    html = html.replace('"/static/', '"static/')
    if not ESKI_BAGLANTI.search(html):
        raise SiteHatasi("index.html: baglantilar paragrafi bulunamadi")
    return ESKI_BAGLANTI.sub(YENI_BAGLANTI, html, count=1)


def betik(js: str) -> str:
    for u in UCLAR:
        js = js.replace(f'"/api/tahmin/{u}"', f'"api/tahmin/{u}.json"')
    kalan = re.findall(r'["\'`]/api/[^"\'`]*', js)
    if kalan:
        raise SiteHatasi(f"ana.js: sabit siteye cevrilmeyen API cagrisi: {kalan}")
    return js


def uret(cikti: Path = None, client=None, now: datetime = None) -> dict:
    cikti = Path(cikti or CIKTI)
    v = veriler(client)                                  # once veri: hata varsa eski site oldugu gibi kalir
    gecici = cikti.with_name(cikti.name + ".yeni")
    shutil.rmtree(gecici, ignore_errors=True)
    (gecici / "static").mkdir(parents=True)
    (gecici / "api" / "tahmin").mkdir(parents=True)
    (gecici / "index.html").write_text(sayfa((WEB / "index.html").read_text(encoding="utf-8")), encoding="utf-8")
    (gecici / "static" / "ana.js").write_text(betik((WEB / "ana.js").read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copyfile(WEB / "ana.css", gecici / "static" / "ana.css")
    for u, d in v.items():
        (gecici / "api" / "tahmin" / f"{u}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    durum = {"uretildi_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
             "bilgi_gunu": v["ufuklar"].get("bilgi_gunu"), "son_kapanis": v["ufuklar"].get("son_kapanis")}
    (gecici / "durum.json").write_text(json.dumps(durum, ensure_ascii=False), encoding="utf-8")
    (gecici / ".nojekyll").write_text("", encoding="utf-8")
    shutil.rmtree(cikti, ignore_errors=True)
    gecici.rename(cikti)
    return durum


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Herkese acik sabit siteyi uretir (GitHub Pages).")
    ap.add_argument("--cikti", default=str(CIKTI))
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 70 + "\nSITE (GitHub Pages)\n" + "=" * 70)
    try:
        d = uret(Path(args.cikti))
    except (SiteHatasi, FileNotFoundError) as e:
        print(f"  HATA: {e}")
        return 1
    print(f"  Uretildi: {args.cikti} | bilgi gunu {d['bilgi_gunu']} | son kapanis {d['son_kapanis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
