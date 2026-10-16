"""
Veritabani Katmani (Microsoft SQL Server / SQLite)
=================================================
Kullanim (proje kokunden):
  python -m src.db.store init      # veritabani ve tablolar (SQL Server'da veritabani yoksa olusturulur)
  python -m src.db.store load      # Git'teki kayitlari ve fiyat serilerini veritabanina yansit
  python -m src.db.store info      # tablo satir sayilari ve son yukleme

Ilke: TEK DOGRU KAYNAK Git'teki CSV kayitlaridir (research/live/*.csv, ozetli ve degistirilemez).
  Veritabani bunlarin SALT OKUNUR AYNASIDIR: her yuklemede satir ozetleri dogrulanir; bozuk kayit varsa
  HICBIR SEY yazilmaz. Yukleme tek islemde (transaction) eski aynayi silip yenisini yazar; tekrar
  calistirmak ayni sonucu verir. Kilitli kodlara ve kayit dosyalarina dokunulmaz.
Baglanti: ALTIN_DB_URL ortam degiskeni (calisma aninda okunur). Yoksa SQLite:
  data/processed/db/altin.sqlite. SQL Server ornegi (Windows, LocalDB):
  mssql+pyodbc://@(localdb)\\MSSQLLocalDB/AltinTahmin?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes
  Parola veya anahtar kodda ve Git'te tutulmaz.

Tablolar:
  forecast        protokol x bilgi gunu x model; nokta, %80/%95 aralik, gerceklesen, mutlak hata, kapsam
  price           seri x gun; fiziki kapanis, teorik (LIVE_V1 esanli), teorik (LIVE_V2 GLD)
  protocol_event  research/registry.jsonl olaylari
  exam_result     bagimsiz sinav sonuclari (JSON)
  load_run        yukleme denetim kaydi (zaman, git commit, satir sayilari)
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.live import forecast as lf
from src.live import forecast_v2 as v2
from src.live import physical_forecast as lp
from src.live import physical_intervals as pi

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE = ROOT / "data" / "processed" / "db" / "altin.sqlite"
EXAMS = {"FINAL_EXAM_V1": "research/final_exam_v1/result.json",
         "FINAL_EXAM_PHYS_V1": "research/final_exam_phys_v1/result.json",
         "BACK_EXAM_ONS_V1": "research/back_exam_ons_v1/result.json"}
PRICE_SOURCES = {"fiziki_kapanis": ("data/processed/live/physical_gold_daily.csv", "Date", "kapanis_tl_gr"),
                 "teorik_v1": ("data/processed/sync/sync_daily.csv", "Date", "theoretical_gram_try"),
                 "teorik_v2": ("data/processed/sync_v2/v2_daily.csv", "Date", "theoretical_gram_try")}

META = sa.MetaData()
FORECAST = sa.Table(
    "forecast", META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("protocol_id", sa.String(32), nullable=False),
    sa.Column("asof_date", sa.Date, nullable=False),
    sa.Column("target_date", sa.Date),
    sa.Column("model", sa.String(32), nullable=False),
    sa.Column("issued_at_utc", sa.String(40), nullable=False),
    sa.Column("point", sa.Float, nullable=False),
    sa.Column("lo80", sa.Float), sa.Column("hi80", sa.Float), sa.Column("lo95", sa.Float), sa.Column("hi95", sa.Float),
    sa.Column("realized", sa.Float), sa.Column("abs_err", sa.Float),
    sa.Column("in80", sa.Float), sa.Column("in95", sa.Float),
    sa.Column("row_hash", sa.String(32), nullable=False),
    sa.UniqueConstraint("protocol_id", "asof_date", "model", name="uq_forecast"),
)
PRICE = sa.Table(
    "price", META,
    sa.Column("series", sa.String(32), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("value", sa.Float, nullable=False),
)
EVENT = sa.Table(
    "protocol_event", META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("protocol_id", sa.String(32), nullable=False),
    sa.Column("event", sa.String(32), nullable=False),
    sa.Column("at_utc", sa.String(40), nullable=False),
    sa.Column("details", sa.Text),
)
EXAM = sa.Table(
    "exam_result", META,
    sa.Column("protocol_id", sa.String(32), primary_key=True),
    sa.Column("result_json", sa.Text, nullable=False),
)
LOAD_RUN = sa.Table(
    "load_run", META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("at_utc", sa.String(40), nullable=False),
    sa.Column("git_commit", sa.String(64)),
    sa.Column("counts", sa.Text, nullable=False),
)
MIRROR = [FORECAST, PRICE, EVENT, EXAM]


class StoreError(Exception):
    """Yukleme reddedildi."""


# ---------------------------------------------------------------------------
# Baglanti
# ---------------------------------------------------------------------------
def db_url() -> str:
    url = os.environ.get("ALTIN_DB_URL", "").strip()
    return url or f"sqlite:///{DEFAULT_SQLITE.as_posix()}"


def redact(url: str) -> str:
    u = sa.engine.make_url(url)
    return u.render_as_string(hide_password=True)


def get_engine(url: str = None) -> sa.Engine:
    url = url or db_url()
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    kw = {"fast_executemany": True} if url.startswith("mssql+pyodbc") else {}
    return sa.create_engine(url, future=True, **kw)


def init_db(url: str = None) -> str:
    url = url or db_url()
    u = sa.engine.make_url(url)
    if u.get_backend_name() == "mssql" and u.database:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", u.database):
            raise StoreError(f"Gecersiz veritabani adi: {u.database!r} (harf, rakam, alt cizgi).")
        master = sa.create_engine(u.set(database="master"), isolation_level="AUTOCOMMIT")
        with master.connect() as c:
            if c.execute(sa.text("SELECT DB_ID(:n)"), {"n": u.database}).scalar() is None:
                c.execute(sa.text(f"CREATE DATABASE [{u.database}]"))
        master.dispose()
    eng = get_engine(url)
    META.create_all(eng)
    eng.dispose()
    return redact(url)


# ---------------------------------------------------------------------------
# Kayitlardan satirlar (once BUTUN ozetler dogrulanir)
# ---------------------------------------------------------------------------
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _d(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan"):
        return None
    return pd.Timestamp(str(v)).date()


def physical_rows(root: Path) -> list:
    log = lp.read_log(root / "research" / "live" / "physical_log.csv")
    lp.verify(log)
    ivl = pi.read_log(root / "research" / "live" / "physical_intervals_log.csv")
    pi.verify(ivl)
    by_date = {str(r["target_date"]): r for _, r in ivl.iterrows()}
    rows = []
    for _, r in log.iterrows():
        iv = by_date.get(str(r["target_date"]))
        for m in lp.MODELS:
            rows.append({"protocol_id": "LIVE_PHYS_V1", "asof_date": _d(r["target_date"]), "target_date": _d(r["target_date"]),
                         "model": m, "issued_at_utc": str(r["issued_at_utc"]), "point": float(r[f"pred_{m}"]),
                         "lo80": _f(iv[f"lo80_{m}"]) if iv is not None else None,
                         "hi80": _f(iv[f"hi80_{m}"]) if iv is not None else None,
                         "lo95": _f(iv[f"lo95_{m}"]) if iv is not None else None,
                         "hi95": _f(iv[f"hi95_{m}"]) if iv is not None else None,
                         "realized": _f(r["realized_price"]), "abs_err": _f(r[f"abs_err_{m}"]),
                         "in80": _f(iv[f"in80_{m}"]) if iv is not None else None,
                         "in95": _f(iv[f"in95_{m}"]) if iv is not None else None,
                         "row_hash": str(r["row_hash"])})
    return rows


def theoretical_rows(root: Path) -> list:
    rows = []
    sources = [(lf.read_log, lf.verify_log, root / "research" / "live" / "forecast_log.csv"),
               (v2.read_log, v2.verify_log, root / "research" / "live" / "forecast_v2_log.csv")]
    for read, verify, path in sources:
        log = read(path)
        verify(log)
        for _, r in log.iterrows():
            base = {"protocol_id": str(r["system_version"]), "asof_date": _d(r["asof_date"]),
                    "target_date": _d(r["realized_date"]), "issued_at_utc": str(r["issued_at_utc"]),
                    "realized": _f(r["realized_price"]), "row_hash": str(r["row_hash"])}
            rows.append({**base, "model": "Mean-Return", "point": float(r["point_price"]),
                         "lo80": _f(r["q80_low"]), "hi80": _f(r["q80_high"]), "lo95": _f(r["q95_low"]),
                         "hi95": _f(r["q95_high"]), "abs_err": _f(r["abs_err_TL"]), "in80": _f(r["in80"]),
                         "in95": _f(r["in95"])})
            rows.append({**base, "model": "RW", "point": float(r["point_price_rw"]), "lo80": None, "hi80": None,
                         "lo95": None, "hi95": None, "abs_err": _f(r["rw_abs_err_TL"]), "in80": None, "in95": None})
    return rows


def price_rows(root: Path) -> tuple:
    rows, missing = [], []
    for series, (rel, dcol, vcol) in PRICE_SOURCES.items():
        p = root / rel
        if not p.exists():
            missing.append(series)
            continue
        df = pd.read_csv(p, usecols=[dcol, vcol]).dropna()
        df = df[df[vcol] > 0]
        rows += [{"series": series, "date": _d(d), "value": float(v)} for d, v in zip(df[dcol], df[vcol])]
    return rows, missing


def event_rows(root: Path) -> list:
    p = root / "research" / "registry.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        if line.strip():
            e = json.loads(line)
            out.append({"protocol_id": e["protocol_id"], "event": e["event"], "at_utc": e["at_utc"],
                        "details": json.dumps({k: v for k, v in e.items() if k not in ("protocol_id", "event", "at_utc")},
                                              sort_keys=True)})
    return out


def exam_rows(root: Path) -> list:
    out = []
    for pid, rel in EXAMS.items():
        p = root / rel
        if p.exists():
            res = json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda c: None)   # NaN/Infinity -> null
            out.append({"protocol_id": pid, "result_json": json.dumps(res, sort_keys=True, allow_nan=False)})
    return out


def _git_commit(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def load_all(url: str = None, root: Path = None) -> dict:
    """Butun kayitlari dogrular, tek islemde aynayi yeniler. Dogrulama hatasinda hicbir sey yazilmaz."""
    root = Path(root or ROOT)
    try:
        data = {FORECAST: physical_rows(root) + theoretical_rows(root), EVENT: event_rows(root), EXAM: exam_rows(root)}
    except (lf.LogTampered, lp.LogTampered, pi.LogTampered) as e:
        raise StoreError(f"Kayit butunlugu dogrulanamadi; veritabanina yazilmadi: {e}")
    data[PRICE], missing = price_rows(root)
    counts = {t.name: len(rows) for t, rows in data.items()}
    eng = get_engine(url)
    META.create_all(eng)
    with eng.begin() as c:
        for t in MIRROR:
            c.execute(t.delete())
        for t, rows in data.items():
            if rows:
                c.execute(t.insert(), rows)
        c.execute(LOAD_RUN.insert(), [{"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       "git_commit": _git_commit(root),
                                       "counts": json.dumps({**counts, "missing_price_series": missing})}])
    eng.dispose()
    return {"counts": counts, "missing_price_series": missing}


# ---------------------------------------------------------------------------
# Sorgular (API bunlari kullanir)
# ---------------------------------------------------------------------------
def _rows(c, stmt) -> list:
    out = []
    for r in c.execute(stmt).mappings():
        out.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()})
    return out


def q_forecasts(c, protocol: str = None, model: str = None, start=None, end=None, limit: int = 500) -> list:
    s = sa.select(FORECAST).order_by(FORECAST.c.asof_date.desc(), FORECAST.c.model)
    if protocol:
        s = s.where(FORECAST.c.protocol_id == protocol)
    if model:
        s = s.where(FORECAST.c.model == model)
    if start:
        s = s.where(FORECAST.c.asof_date >= start)
    if end:
        s = s.where(FORECAST.c.asof_date <= end)
    return _rows(c, s.limit(limit))


def q_latest(c, protocol: str) -> list:
    last = sa.select(sa.func.max(FORECAST.c.asof_date)).where(FORECAST.c.protocol_id == protocol).scalar_subquery()
    s = sa.select(FORECAST).where(FORECAST.c.protocol_id == protocol, FORECAST.c.asof_date == last) \
        .order_by(FORECAST.c.model)
    return _rows(c, s)


def q_scorecard(c, protocol: str) -> dict:
    s = sa.select(FORECAST.c.model, sa.func.count().label("n"), sa.func.avg(FORECAST.c.abs_err).label("mae"),
                  sa.func.avg(FORECAST.c.in80).label("cov80"), sa.func.avg(FORECAST.c.in95).label("cov95")) \
        .where(FORECAST.c.protocol_id == protocol, FORECAST.c.abs_err.is_not(None)).group_by(FORECAST.c.model)
    models = {r["model"]: dict(r) for r in c.execute(s).mappings()}
    rw = models.get("RW", {}).get("mae")
    for m in models.values():
        m["vs_rw_pct"] = (100 * (1 - m["mae"] / rw)) if rw and m["model"] != "RW" else None
    total = c.execute(sa.select(sa.func.count(sa.distinct(FORECAST.c.asof_date)))
                      .where(FORECAST.c.protocol_id == protocol)).scalar()
    return {"protocol_id": protocol, "issued_days": int(total or 0), "models": sorted(models.values(), key=lambda m: m["model"]),
            "note": "Betimleyici ara karne; on kayitli degerlendirme oncesi model secimi icin KULLANILMAZ."}


def q_prices(c, series: str, start=None, end=None) -> list:
    s = sa.select(PRICE.c.date, PRICE.c.value).where(PRICE.c.series == series).order_by(PRICE.c.date)
    if start:
        s = s.where(PRICE.c.date >= start)
    if end:
        s = s.where(PRICE.c.date <= end)
    return _rows(c, s)


def q_events(c) -> list:
    return _rows(c, sa.select(EVENT).order_by(EVENT.c.id))


def q_exams(c) -> list:
    return [{"protocol_id": r["protocol_id"], "result": json.loads(r["result_json"])}
            for r in c.execute(sa.select(EXAM).order_by(EXAM.c.protocol_id)).mappings()]


def q_last_load(c):
    r = c.execute(sa.select(LOAD_RUN).order_by(LOAD_RUN.c.id.desc()).limit(1)).mappings().first()
    return None if r is None else {**dict(r), "counts": json.loads(r["counts"])}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Veritabani katmani (SQL Server / SQLite).")
    ap.add_argument("command", choices=["init", "load", "info"])
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 72 + f"\nVERITABANI - {args.command.upper()} - {redact(db_url())}\n" + "=" * 72)
    try:
        if args.command == "init":
            print(f"  Hazir: {init_db()}")
            return 0
        if args.command == "load":
            r = load_all()
            print("  Yuklendi: " + ", ".join(f"{k} {v}" for k, v in r["counts"].items()))
            if r["missing_price_series"]:
                print(f"  Not: bulunamayan fiyat serileri: {r['missing_price_series']}")
            return 0
        eng = get_engine()
        with eng.connect() as c:
            for t in META.sorted_tables:
                print(f"  {t.name:15s} {c.execute(sa.select(sa.func.count()).select_from(t)).scalar()}")
            print(f"  Son yukleme: {q_last_load(c)}")
        return 0
    except StoreError as e:
        print(f"  REDDEDILDI: {e}")
        return 3
    except sa.exc.SQLAlchemyError as e:
        print(f"  VERITABANI HATASI: {type(e).__name__}: {str(e).splitlines()[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
