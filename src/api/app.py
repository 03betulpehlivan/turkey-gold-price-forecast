"""
REST API (FastAPI, salt okunur)
===============================
Calistirma (proje kokunden):
  python -m src.db.store load                       # once veritabanini guncelle
  python -m uvicorn src.api.app:app --port 8000     # sonra http://127.0.0.1:8000  (belgeler: /docs)

Ilke: API yalnizca OKUR (GET). Tahmin uretmez, kayit degistirmez; veriyi src/db/store.py aynasindan alir.
Guvenlik: yalnizca 127.0.0.1 / localhost Host basligi kabul edilir (DNS rebinding'e karsi; ALTIN_ALLOWED_HOSTS
  ile genisletilebilir); her yanitta guvenlik basliklari; pano icin siki Content-Security-Policy (yalnizca
  ayni kaynaktan betik/stil; satir ici betik yok). Sunucuyu scripts/pano.ps1 ile baslatin (127.0.0.1'e baglanir).
Veritabani baglantisi ALTIN_DB_URL ortam degiskeninden her istekte okunur (yoksa yerel SQLite).
Ara karneler BETIMLEYICIDIR; on kayitli degerlendirmenin yerine gecmez ve model secimi icin kullanilmaz.
Bu sistem arastirma ve egitim amaclidir; yatirim tavsiyesi degildir.
"""

from datetime import date
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import findings as fd
from src import health as hl
from src.db import store

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PROTOCOLS = ("LIVE_PHYS_V1", "LIVE_V1", "LIVE_V2")
SERIES = tuple(store.PRICE_SOURCES)
DISCLAIMER = "Arastirma ve egitim amaclidir; yatirim tavsiyesi degildir."

DEFAULT_HOSTS = "127.0.0.1,localhost"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                    "Cross-Origin-Resource-Policy": "same-origin", "Cache-Control": "no-store"}


def allowed_hosts() -> list:
    return [h.strip() for h in os.environ.get("ALTIN_ALLOWED_HOSTS", DEFAULT_HOSTS).split(",") if h.strip()]


app = FastAPI(title="Turkiye Gram Altin Tahmin Sistemi", version="1.0",
              description="Canli, on kayitli tahmin kayitlarinin salt okunur API'si. " + DISCLAIMER)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())
app.mount("/static", StaticFiles(directory=WEB_DIR, html=False), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if not request.url.path.startswith(DOC_PATHS):          # /docs, harici Swagger dosyalarini kullanir
        response.headers["Content-Security-Policy"] = CSP
    return response


def _conn():
    return store.get_engine().connect()


def _check(value, allowed, name):
    if value is not None and value not in allowed:
        raise HTTPException(status_code=404, detail=f"Bilinmeyen {name}: {value}. Gecerli: {list(allowed)}")


@app.get("/api/health")
def health():
    try:
        with _conn() as c:
            last = store.q_last_load(c)
    except Exception as e:  # veritabani yok veya ulasilamiyor
        return {"status": "veritabani_yok", "db": store.redact(store.db_url()), "detail": type(e).__name__}
    return {"status": "ok" if last else "bos", "db": store.redact(store.db_url()), "last_load": last}


@app.get("/api/system")
def system():
    """Zamanlanmis gorevlerin son durumu ve zaman cizelgesi (veritabani gerektirmez)."""
    return hl.system_status()


@app.get("/api/findings")
def findings():
    """Kesif calismalarinin ve geriye donuk sinavin anahtar sayilari (kayitli sonuc dosyalarindan)."""
    return {"items": fd.collect()}


@app.get("/api/protocols")
def protocols():
    with _conn() as c:
        return {"protocols": list(PROTOCOLS), "events": store.q_events(c)}


@app.get("/api/forecasts")
def forecasts(protocol: str = Query(None), model: str = Query(None), start: date = Query(None),
              end: date = Query(None), limit: int = Query(500, ge=1, le=5000)):
    _check(protocol, PROTOCOLS, "protokol")
    with _conn() as c:
        return {"forecasts": store.q_forecasts(c, protocol, model, start, end, limit)}


@app.get("/api/forecasts/latest")
def latest(protocol: str = Query(...)):
    _check(protocol, PROTOCOLS, "protokol")
    with _conn() as c:
        rows = store.q_latest(c, protocol)
    if not rows:
        raise HTTPException(status_code=404, detail=f"{protocol} icin tahmin yok.")
    return {"protocol_id": protocol, "asof_date": rows[0]["asof_date"], "forecasts": rows}


@app.get("/api/scorecard")
def scorecard(protocol: str = Query(...)):
    _check(protocol, PROTOCOLS, "protokol")
    with _conn() as c:
        return store.q_scorecard(c, protocol)


@app.get("/api/prices")
def prices(series: str = Query(...), start: date = Query(None), end: date = Query(None)):
    _check(series, SERIES, "seri")
    with _conn() as c:
        return {"series": series, "prices": store.q_prices(c, series, start, end)}


@app.get("/api/exams")
def exams():
    with _conn() as c:
        return {"exams": store.q_exams(c)}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")
