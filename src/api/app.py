"""
REST API (FastAPI, salt okunur)
===============================
Calistirma (proje kokunden):
  python -m src.db.store load                       # once veritabanini guncelle
  python -m uvicorn src.api.app:app --port 8000     # sonra http://127.0.0.1:8000  (belgeler: /docs)

Ilke: API yalnizca OKUR (GET). Tahmin uretmez, kayit degistirmez; veriyi src/db/store.py aynasindan alir.
Veritabani baglantisi ALTIN_DB_URL ortam degiskeninden her istekte okunur (yoksa yerel SQLite).
Ara karneler BETIMLEYICIDIR; on kayitli degerlendirmenin yerine gecmez ve model secimi icin kullanilmaz.
Bu sistem arastirma ve egitim amaclidir; yatirim tavsiyesi degildir.
"""

from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from src.db import store

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PROTOCOLS = ("LIVE_PHYS_V1", "LIVE_V1", "LIVE_V2")
SERIES = tuple(store.PRICE_SOURCES)
DISCLAIMER = "Arastirma ve egitim amaclidir; yatirim tavsiyesi degildir."

app = FastAPI(title="Turkiye Gram Altin Tahmin Sistemi", version="1.0",
              description="Canli, on kayitli tahmin kayitlarinin salt okunur API'si. " + DISCLAIMER)


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
