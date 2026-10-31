"""
Ekran Goruntuleri: web arayuzunden rapor icin bolum goruntuleri
===============================================================
Pano ACIKKEN (scripts\\pano.ps1) ikinci bir PowerShell penceresinde calistirilir:
  pip install playwright                 # bir kez; tarayici indirmez, kurulu Microsoft Edge kullanilir
  python -m src.ekran_goruntuleri        # docs/figures/ altina EKRANLAR'daki goruntuleri yazar

Secenekler:
  --url        pano adresi (varsayilan http://127.0.0.1:8000/)
  --out        cikti klasoru (varsayilan docs/figures)
  --tarayici   "msedge" (varsayilan), "chrome" ya da bir Chromium dosya yolu

Goruntuler gercek verilerle o anki ekrandan alinir; tahmin ve kayitlara dokunulmaz.
Sayfanin ust menusu goruntu alinmadan once gizlenir (bolumlerin ustune binmesin).
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

# dosya adi -> sayfadaki oge (index.html ile ayni kimlik ve siniflar)
EKRANLAR = {
    "ana_ekran": "#tahmin",                    # bugunku tahmin, aralik cetveli, yelpaze
    "ekran_noktalar": ".iki-aralik",           # %80 ve %95 aralik: 10 ve 100 nokta
    "ekran_adim": "#gosteri",                  # nasil calisiyor: adim adim (3. adim, iki aralik birlikte)
    "ekran_son15": "#tekrar",                  # son 15 islem gunu: aralik ve gerceklesen fiyat
    "ekran_yaris": "#yaris",                   # model yarismasi: kim kazandi?
}
HAZIR = ["#yaris-satirlar .yr-satir", "#tekrar-seritler button", "#gosteri-noktalar span", "#ufuk-sec button"]


def _tarayici(p, secim: str):
    yol = Path(secim)
    if yol.exists():
        return p.chromium.launch(executable_path=str(yol))
    return p.chromium.launch(channel=secim)


def cek(url: str, out: Path, secim: str = "msedge") -> list:
    from playwright.sync_api import sync_playwright        # istege bagli bagimlilik: yalnizca burada gerekir
    out.mkdir(parents=True, exist_ok=True)
    yazilan = []
    with sync_playwright() as p:
        b = _tarayici(p, secim)
        page = b.new_page(viewport={"width": 1200, "height": 900}, color_scheme="light", device_scale_factor=1.5)
        page.set_default_timeout(180000)                       # ilk acilista Chronos yuklenebilir
        page.goto(url, wait_until="networkidle")
        for sec in HAZIR:
            page.wait_for_selector(sec)
        h = page.evaluate("document.body.scrollHeight")
        for y in range(0, int(h), 600):                         # animasyonlar gorununce baslar: sayfayi bir kez gez
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(120)
        page.evaluate("window.scrollTo(0, 0)")
        page.evaluate("document.querySelector('.serit').style.display = 'none'")
        page.evaluate("gosteriDur(); gosteriGit(2)")           # adim adim: %80 ve %95 aralik birlikte
        page.wait_for_timeout(1500)
        for ad, sec in EKRANLAR.items():
            e = page.locator(sec).first
            e.scroll_into_view_if_needed()
            page.wait_for_timeout(1300)                         # cubuk ve nokta animasyonlari bitsin
            hedef = out / f"{ad}.png"
            e.screenshot(path=str(hedef))
            yazilan.append(hedef)
        b.close()
    return yazilan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Web arayuzunden rapor icin bolum goruntuleri.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tarayici", default="msedge")
    args = ap.parse_args([] if argv is None else argv)
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("HATA: playwright kurulu degil. Once: pip install playwright")
        return 1
    try:
        for f in cek(args.url, Path(args.out or OUT_DIR), args.tarayici):
            print(f"  yazildi: {f}")
    except Exception as e:                                      # pano kapali, tarayici yok vb.
        print(f"HATA: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        print("Pano acik mi? (scripts\\pano.ps1)  Edge yoksa: --tarayici chrome")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
