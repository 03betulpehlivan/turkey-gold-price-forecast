"""
TCMB EVDS Kesif ve Indirme Araci (fiziki altin fiyati arastirmasi)
==================================================================
Kullanim (proje kokunden, PowerShell; anahtar YALNIZCA ortam degiskeninde tutulur):
  $env:EVDS_API_KEY = "anahtariniz"
  python -m src.data.evds kesif                              # adinda "altin" gecen gruplar ve seriler
  python -m src.data.evds kesif --ara "kulce,gram"           # ek arama kelimeleri
  python -m src.data.evds indir --seri TP.X.Y,TP.A.B --baslangic 01-01-2015

Protokol (evds 0.4.0 istemcisinin kaynak kodundan dogrulandi, Subat 2026):
  Adres  : https://evds3.tcmb.gov.tr/igmevdsms-dis/
  Anahtar: HTTP basliginda 'key' (URL'de DEGIL)
  Uclar  : categories/type=json | datagroups/mode=0&code=&type=json (tum gruplar)
           serieList/type=json&code=<grup> | series=A-B&startDate=gg-aa-yyyy&endDate=...&type=json
  SSL    : sunucu eski TLS yeniden anlasma istiyor; OP_LEGACY_SERVER_CONNECT gerekli.
  Gercek calistirmada dogrulanan (14.10.2026): endDate BOS olursa 400 doner; her yanit EN FAZLA
  1000 satir icerir (fazlasi sessizce kesilir, son 1000 satir gelir). Bu nedenle indirme tarih
  araligini parcalara boler; 1000 satira ulasan parca otomatik olarak yeniden bolunur.

Kurallar:
  - Anahtar hicbir dosyaya, manifest'e veya ekrana yazilmaz.
  - Kesif yalnizca metadata indirir. Indirme yalnizca kullanicinin acikca verdigi seri kodlarini alir.
  - Ham yanitlar YENI bir anlik goruntuye yazilir: data/raw/evds/<zaman>/ (+ manifest, md5).
  - Bu modul modele veya mevcut veri setlerine BAGLANMAZ; hangi serinin fiziki gram altini
    temsil ettigine metadata incelenerek karar verilir. Teorik ve fiziki fiyatlar ayri seridir.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import ssl
import sys

import pandas as pd

from src.data import collect as co

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
OUT_ROOT = ROOT / "data" / "raw" / "evds"
DEFAULT_TERMS = ("altin", "gold")
DATE_KEYS = ("Tarih", "TARIH", "Date", "DATE")
ROW_LIMIT = 1000          # EVDS yanit basina satir siniri (gercek calistirmada gozlendi)
CHUNK_DAYS = 3 * 365      # ilk parca boyu (~750 is gunu < 1000)
DATE_FMT = "%d-%m-%Y"


class EVDSError(Exception):
    pass


def normalize(text: str) -> str:
    """Turkce buyuk/kucuk harf farkini gideren arama normali (ALTIN, Altin, altin -> altin)."""
    t = str(text).replace("I", "i").replace("\u0130", "i").casefold().replace("\u0131", "i")
    for a, b in (("\u00fc", "u"), ("\u00f6", "o"), ("\u00e7", "c"), ("\u015f", "s"), ("\u011f", "g")):
        t = t.replace(a, b)
    return t


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    import urllib3

    class LegacyAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kw):
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ctx.options |= 0x4                                   # OP_LEGACY_SERVER_CONNECT
            self.poolmanager = urllib3.poolmanager.PoolManager(num_pools=connections, maxsize=maxsize,
                                                               block=block, ssl_context=ctx)
    s = requests.Session()
    s.mount("https://", LegacyAdapter())
    return s


class Client:
    """En kucuk EVDS istemcisi. get(url, headers) -> (status_code, bytes) enjekte edilebilir (test)."""

    def __init__(self, key: str, get=None, base_url: str = BASE_URL):
        if not key:
            raise EVDSError("EVDS_API_KEY bos. PowerShell: $env:EVDS_API_KEY = \"anahtariniz\"")
        self._key = key
        self._get = get or self._default_get
        self.base = base_url
        self.calls = []                                           # anahtarsiz URL kaydi (denetim icin)

    def _default_get(self, url, headers):
        r = _session().get(url, headers=headers, timeout=60)
        return r.status_code, r.content

    def request(self, endpoint: str, params: dict):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = self.base + endpoint + query
        self.calls.append(url)
        status, content = self._get(url, {"key": self._key})
        if status != 200:
            detail = content[:300].decode("utf-8", errors="replace").strip() if content else ""
            raise EVDSError(f"EVDS {status}: {endpoint or 'veri'} {detail} (anahtar veya istek hatali olabilir)")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise EVDSError(f"EVDS yaniti JSON degil: {endpoint}") from e

    def categories(self):
        return self.request("categories/", {"type": "json"})

    def datagroups(self):
        return self.request("datagroups/", {"mode": 0, "code": "", "type": "json"})

    def series_list(self, datagroup_code: str):
        return self.request("serieList/", {"type": "json", "code": datagroup_code})

    def data(self, codes: list, start: str, end: str = ""):
        res = self.request("", {"series": "-".join(codes), "startDate": start, "endDate": end, "type": "json",
                                "formulas": "", "frequency": "", "aggregationTypes": ""})
        if "items" not in res:
            raise EVDSError("Veri yanitinda 'items' alani yok.")
        return res


# ---------------------------------------------------------------------------
def discover(client: Client, terms=DEFAULT_TERMS) -> pd.DataFrame:
    """Adinda arama kelimesi gecen veri gruplarinin TUM serileri + adinda kelime gecen seriler."""
    keys = [normalize(t) for t in terms]
    groups = client.datagroups()
    hit = [g for g in groups if any(k in normalize(g.get("DATAGROUP_NAME", "")) or
                                    k in normalize(g.get("DATAGROUP_NAME_ENG", "")) for k in keys)]
    rows = []
    for g in hit:
        for s in client.series_list(g["DATAGROUP_CODE"]):
            rows.append({"DATAGROUP_CODE": g["DATAGROUP_CODE"], "DATAGROUP_NAME": g.get("DATAGROUP_NAME", ""),
                         "SERIE_CODE": s.get("SERIE_CODE"), "SERIE_NAME": s.get("SERIE_NAME", ""),
                         "SERIE_NAME_ENG": s.get("SERIE_NAME_ENG", ""), "FREQUENCY": s.get("FREQUENCY_STR", ""),
                         "START_DATE": s.get("START_DATE", ""), "END_DATE": s.get("END_DATE", ""),
                         "DEFAULT_AGG": s.get("DEFAULT_AGG_METHOD_STR", s.get("DEFAULT_AGG_METHOD", ""))})
    return pd.DataFrame(rows, columns=["DATAGROUP_CODE", "DATAGROUP_NAME", "SERIE_CODE", "SERIE_NAME",
                                       "SERIE_NAME_ENG", "FREQUENCY", "START_DATE", "END_DATE", "DEFAULT_AGG"])


def _d(s: str) -> date:
    return datetime.strptime(s, DATE_FMT).date()


def fetch_chunked(client: Client, codes: list, start: str, end: str, chunk_days: int = CHUNK_DAYS) -> tuple:
    """Araligi parcalara bolerek indirir. 1000 satira ulasan parca ikiye bolunur (kesilme olmasin).

    Dondurur: (items listesi, parca raporu). Parcalar ortusmez; tarih tekrarinda deger farki hata verir.
    """
    a, b = _d(start), _d(end)
    if b < a:
        raise EVDSError("Bitis tarihi baslangictan once.")
    queue, chunks, items = [], [], []
    cur = a
    while cur <= b:
        nxt = min(cur + timedelta(days=chunk_days - 1), b)
        queue.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    while queue:
        s, e = queue.pop(0)
        res = client.data(codes, s.strftime(DATE_FMT), e.strftime(DATE_FMT))
        got = res["items"]
        if len(got) >= ROW_LIMIT:
            if (e - s).days < 1:
                raise EVDSError(f"Tek gunluk parca bile {ROW_LIMIT} satir dondurdu: {s}")
            mid = s + timedelta(days=(e - s).days // 2)
            queue[:0] = [(s, mid), (mid + timedelta(days=1), e)]
            chunks.append({"start": str(s), "end": str(e), "rows": len(got), "split": True})
            continue
        chunks.append({"start": str(s), "end": str(e), "rows": len(got), "split": False})
        items.extend(got)
    return items, chunks


def parse_items(items: list, codes: list) -> pd.DataFrame:
    """EVDS 'items' -> Date + seri sutunlari (kod: '.' -> '_'). Bos/'ND' degerler NaN."""
    if not items:
        return pd.DataFrame(columns=["Date"] + codes)
    date_key = next((k for k in DATE_KEYS if k in items[0]), None)
    if date_key is None:
        raise EVDSError(f"Tarih alani bulunamadi; alanlar: {sorted(items[0])}")
    df = pd.DataFrame(items)
    # KATI gunluk bicim (gg-aa-yyyy): ceyreklik/aylik tarihler ("2015-Q1", "2015-1") sessizce donusturulmesin
    out = pd.DataFrame({"Date": pd.to_datetime(df[date_key], format="%d-%m-%Y", errors="coerce")})
    if out["Date"].isna().any():
        bad = df.loc[out["Date"].isna(), date_key].head(3).tolist()
        raise EVDSError(f"Ayristirilamayan tarih: {bad} (gunluk disi frekans olabilir)")
    for c in codes:
        col = c.replace(".", "_")
        out[c] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else float("nan")
    out = out.sort_values("Date").reset_index(drop=True)
    dup = out["Date"].duplicated(keep=False)
    if dup.any():
        g = out[dup].groupby("Date")[codes].nunique(dropna=False)
        if (g > 1).any().any():
            raise EVDSError("Ayni tarih icin farkli degerler geldi (parca ortusmesi tutarsiz).")
        out = out.drop_duplicates("Date").reset_index(drop=True)
    return out


def coverage(df: pd.DataFrame, codes: list) -> dict:
    rep = {"rows": int(len(df)), "duplicate_dates": int(df["Date"].duplicated().sum()) if len(df) else 0}
    for c in codes:
        s = df.set_index("Date")[c] if len(df) else pd.Series(dtype=float)
        ok = s.dropna()
        rep[c] = {"observations": int(len(ok)), "missing_or_empty": int(s.isna().sum()),
                  "first_date": str(ok.index.min().date()) if len(ok) else None,
                  "last_date": str(ok.index.max().date()) if len(ok) else None,
                  "min": float(ok.min()) if len(ok) else None, "max": float(ok.max()) if len(ok) else None}
    return rep


def save_snapshot(kind: str, payload, extra: dict, root: Path = None, now: datetime = None) -> Path:
    snap = co.new_snapshot_dir(Path(root or OUT_ROOT), now)          # OUT_ROOT calisma aninda okunur
    raw = snap / f"{kind}_raw.json"
    raw.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {"retrieved_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                "source": "TCMB EVDS", "base_url": BASE_URL, "kind": kind,
                "raw_file": raw.name, "raw_md5": hashlib.md5(raw.read_bytes()).hexdigest(), **extra}
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap


def run_discover(client: Client, terms, root: Path = None, now=None) -> tuple:
    cat = discover(client, terms)
    snap = save_snapshot("catalog", cat.to_dict(orient="records"), {"terms": list(terms), "series": int(len(cat))},
                         root, now)
    cat.to_csv(snap / "catalog.csv", index=False, encoding="utf-8-sig")
    return cat, snap


def run_download(client: Client, codes: list, start: str, end: str = "", root: Path = None, now=None) -> tuple:
    end = end or (now or datetime.now(timezone.utc)).strftime(DATE_FMT)       # bos bitis -> bugun (EVDS zorunlu)
    items, chunks = fetch_chunked(client, codes, start, end)
    df = parse_items(items, codes)
    rep = coverage(df, codes)
    snap = save_snapshot("data", {"items": items}, {"series": codes, "start": start, "end": end, "chunks": chunks,
                                                    "coverage": rep}, root, now)
    df.to_csv(snap / "data.csv", index=False, date_format="%Y-%m-%d")
    return df, rep, snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TCMB EVDS kesif ve indirme (fiziki altin arastirmasi).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kesif")
    k.add_argument("--ara", default="", help="Ek arama kelimeleri (virgulle).")
    d = sub.add_parser("indir")
    d.add_argument("--seri", required=True, help="Virgulle ayrilmis EVDS seri kodlari.")
    d.add_argument("--baslangic", default="01-01-2015", help="gg-aa-yyyy")
    d.add_argument("--bitis", default="", help="gg-aa-yyyy (bos: bugun)")
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 66 + "\nTCMB EVDS - " + args.cmd.upper() + "\n" + "=" * 66)
    try:
        client = Client(os.environ.get("EVDS_API_KEY", ""))
        if args.cmd == "kesif":
            terms = list(DEFAULT_TERMS) + [t.strip() for t in args.ara.split(",") if t.strip()]
            cat, snap = run_discover(client, terms)
            if cat.empty:
                print(f"  Arama kelimeleri ({terms}) icin seri bulunamadi.")
            else:
                pd.set_option("display.width", 220)
                pd.set_option("display.max_colwidth", 70)
                print(cat[["SERIE_CODE", "SERIE_NAME", "FREQUENCY", "START_DATE", "END_DATE"]].to_string(index=False))
            print(f"\n  {len(cat)} seri | katalog: {snap / 'catalog.csv'}")
            return 0
        codes = [c.strip() for c in args.seri.split(",") if c.strip()]
        df, rep, snap = run_download(client, codes, args.baslangic, args.bitis)
        print(f"  Satir: {rep['rows']} | parca: {len(json.loads((snap / 'manifest.json').read_text(encoding='utf-8'))['chunks'])}")
        for c in codes:
            r = rep[c]
            print(f"  {c}: {r['observations']} gozlem, {r['first_date']} .. {r['last_date']}, bos {r['missing_or_empty']}, "
                  f"aralik {r['min']} .. {r['max']}")
        print(f"  Cikti: {snap}")
        return 0
    except EVDSError as e:
        print(f"  HATA: {e}")
        return 1
    except Exception as e:                                          # ag/SSL hatalari
        print(f"  BAGLANTI HATASI: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
