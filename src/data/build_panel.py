"""
Piyasa Paneli Derleme
=====================
Kullanim (proje kokunden):
  python -m src.data.build_panel                    # en son anlik goruntu; yalnizca rapor
  python -m src.data.build_panel --write            # data/processed/market_panel.csv yazar
  python -m src.data.build_panel --snapshot data/raw/snapshots/<klasor>

Girdi : collect.py'nin anlik goruntusu (manifest.json + seri CSV'leri). Ag gerekmez.
Cikti : data/processed/market_panel.csv ve market_panel_report.json
        Mevcut theoretical_gold.csv, model_dataset.csv ve ham dosyalar DEGISMEZ.

Panel semasi
------------
  Date                  temel takvim: altin (GC=F) ve USD/TRY'nin ortak gunleri
  gold_usd_oz, usdtry   kapanislar
  theoretical_gram_try  gold_usd_oz * usdtry / 31.1035 (build_dataset ile ayni hesap)
  <seri>                diger seriler: Date gunu veya oncesindeki EN SON kapanis
  <seri>_age_days       kullanilan degerin Date'ten kac takvim gunu once oldugu

Zaman varsayimi (as-of birlestirme)
-----------------------------------
Date gunune ait bir kapanisin, Date gunu icin yapilacak tahminden ONCE bilindigi
varsayilir: tahmin, panele giren tum piyasalar kapandiktan sonra (yaklasik
gece yarisi TSI) uretilir. Hicbir seri Date'ten sonraki bir degeri kullanmaz.
Degeri MAX_STALENESS_DAYS'ten eski olan gozlemler bos birakilir (bayram gibi
uzun tatillerde eski deger tasinmaz) ve raporlanir. Kapanis saatleri ve
Yahoo'nun gunluk cubuk tanimi DOGRULANMADI; ilgili tani adimi (D4) bunu sinayacak.
"""

from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.data import build_dataset as bd
from src.data.collect import SERIES, DEFAULT_SNAPSHOT_ROOT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
PANEL_FILE = "market_panel.csv"
REPORT_FILE = "market_panel_report.json"
MAX_STALENESS_DAYS = 7
EXTRA_SERIES = [n for n in SERIES if not SERIES[n]["required"]]
BASE_COLUMNS = ["Date", "gold_usd_oz", "usdtry", "theoretical_gram_try"]


class PanelError(Exception):
    """Girdi veya dogrulama hatasi; hicbir dosya yazilmaz."""


def _quiet(func, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args), buf.getvalue()


# ---------------------------------------------------------------------------
# Anlik goruntu okuma
# ---------------------------------------------------------------------------
def find_latest_snapshot(root: Path) -> Path:
    """Zorunlu serileri 'ok' olan en yeni anlik goruntu klasoru."""
    root = Path(root)
    candidates = sorted([p for p in root.iterdir() if (p / "manifest.json").exists()],
                        reverse=True) if root.exists() else []
    for snap in candidates:
        m = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        if all(m["series"].get(n, {}).get("status") == "ok" for n in SERIES if SERIES[n]["required"]):
            return snap
    raise PanelError(f"Zorunlu serileri basarili bir anlik goruntu bulunamadi: {root}")


def load_snapshot(snap: Path) -> tuple:
    """Manifest'i ve durumu 'ok' olan serileri (Date, close) olarak yukler.

    Dosyanin md5'i manifest'tekiyle eslesmezse hata verir (sonradan degismis dosya).
    """
    snap = Path(snap)
    mpath = snap / "manifest.json"
    if not mpath.exists():
        raise PanelError(f"manifest.json yok: {snap}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    series = {}
    for name, entry in manifest["series"].items():
        if entry.get("status") != "ok":
            continue
        path = snap / entry["file"]
        if not path.exists():
            raise PanelError(f"{name}: dosya yok ({path.name}).")
        if hashlib.md5(path.read_bytes()).hexdigest() != entry["md5"]:
            raise PanelError(f"{name}: dosya manifest'teki md5 ile eslesmiyor (sonradan degismis).")
        try:
            series[name], _ = _quiet(bd.read_yahoo_csv, path, entry["ticker"])
        except bd.DataError as e:
            raise PanelError(f"{name}: {e}") from e
    for n in SERIES:
        if SERIES[n]["required"] and n not in series:
            raise PanelError(f"Zorunlu seri eksik veya basarisiz: {n}")
    return manifest, series


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def build_panel(series: dict, max_staleness_days: int = MAX_STALENESS_DAYS) -> tuple:
    """Temel takvim + as-of hizalanmis ek seriler. Dondurur: (panel, kapsam raporu)."""
    (theo, _), _ = _quiet(bd.build, series["gold"], series["usdtry"])
    panel = theo[BASE_COLUMNS].copy().sort_values("Date").reset_index(drop=True)
    coverage = {}
    for name in EXTRA_SERIES:
        if name not in series or len(series[name]) == 0:
            panel[name] = np.nan
            panel[f"{name}_age_days"] = np.nan
            coverage[name] = {"status": "yok (anlik goruntude basarisiz, bos veya istenmedi)"}
            continue
        right = series[name].rename(columns={"Date": "obs_date", "close": name}).sort_values("obs_date")
        merged = pd.merge_asof(panel[["Date"]], right, left_on="Date", right_on="obs_date",
                               direction="backward", allow_exact_matches=True)
        age = (merged["Date"] - merged["obs_date"]).dt.days
        before_start = merged["obs_date"].isna()
        stale = age > max_staleness_days
        value = merged[name].where(~stale)
        age = age.where(~stale & ~before_start)
        panel[name] = value.to_numpy(dtype=float)
        panel[f"{name}_age_days"] = age.to_numpy(dtype=float)   # sema sabit: her zaman float
        coverage[name] = {
            "status": "ok",
            "ticker": SERIES[name]["ticker"],
            "raw_first_date": str(right["obs_date"].iloc[0].date()),
            "raw_last_date": str(right["obs_date"].iloc[-1].date()),
            "raw_rows": int(len(right)),
            "panel_rows_with_value": int(value.notna().sum()),
            "panel_coverage_pct": round(float(value.notna().mean() * 100), 2),
            "rows_before_series_start": int(before_start.sum()),
            "rows_dropped_stale": int((stale & ~before_start).sum()),
            "age_days_max_used": None if age.dropna().empty else int(age.max()),
            "age_days_share_same_day_pct": round(float((age == 0).sum() / max(value.notna().sum(), 1) * 100), 2),
        }
    validate_panel(panel, max_staleness_days)
    return panel, coverage


def validate_panel(panel: pd.DataFrame, max_staleness_days: int):
    """Panelin ic tutarliligi; ihlalde PanelError."""
    if not panel["Date"].is_monotonic_increasing or panel["Date"].duplicated().any():
        raise PanelError("Panel tarihleri kesin artan degil.")
    base = panel[BASE_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise PanelError("Temel sutunlarda bos, sonsuz veya pozitif olmayan deger var.")
    expected = panel["gold_usd_oz"] * panel["usdtry"] / bd.TROY_OUNCE_GRAM
    if (panel["theoretical_gram_try"] - expected).abs().max() > 1e-9:
        raise PanelError("theoretical_gram_try formulle tutarsiz.")
    for name in EXTRA_SERIES:
        age = panel[f"{name}_age_days"]
        val = panel[name]
        if (val.notna() != age.notna()).any():
            raise PanelError(f"{name}: deger ve yas sutunlari tutarsiz.")
        if (age < 0).any():
            raise PanelError(f"{name}: gelecekteki bir deger kullanilmis (negatif yas).")
        if (age > max_staleness_days).any():
            raise PanelError(f"{name}: bayatlik siniri asilmis.")


def compare_with_existing(panel: pd.DataFrame, path: Path) -> dict:
    """Paneli mevcut theoretical_gold.csv ile karsilastirir (bilgi; hata degil).

    Fark, Yahoo'nun gecmis degerleri duzeltmesinden veya veri araliginin
    uzamasindan kaynaklanabilir. Mevcut dosya degistirilmez.
    """
    if not Path(path).exists():
        return {"status": "mevcut dosya yok"}
    ref = pd.read_csv(path, parse_dates=["Date"])
    m = ref.merge(panel[BASE_COLUMNS], on="Date", suffixes=("_mevcut", "_yeni"))
    out = {
        "existing_rows": int(len(ref)), "panel_rows": int(len(panel)),
        "common_dates": int(len(m)),
        "dates_only_in_existing": int(len(set(ref["Date"]) - set(panel["Date"]))),
        "dates_only_in_panel_within_existing_range": int(len(
            set(panel.loc[panel["Date"] <= ref["Date"].max(), "Date"]) - set(ref["Date"]))),
        "panel_dates_after_existing_end": int((panel["Date"] > ref["Date"].max()).sum()),
    }
    for col in BASE_COLUMNS[1:]:
        d = (m[f"{col}_mevcut"] - m[f"{col}_yeni"]).abs()
        out[f"{col}_max_abs_diff"] = float(d.max()) if len(d) else None
        out[f"{col}_rows_diff_gt_1e-9"] = int((d > 1e-9).sum())
    out["status"] = "AYNI" if (out["dates_only_in_existing"] == 0
                               and out["dates_only_in_panel_within_existing_range"] == 0
                               and all(out[f"{c}_rows_diff_gt_1e-9"] == 0 for c in BASE_COLUMNS[1:])) \
        else "FARKLI"
    return out


def panels_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    if not (pd.to_datetime(a["Date"]).to_numpy() == pd.to_datetime(b["Date"]).to_numpy()).all():
        return False
    for c in a.columns[1:]:
        if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                           rtol=0, atol=1e-9, equal_nan=True):
            return False
    return True


# ---------------------------------------------------------------------------
# Komut satiri
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Anlik goruntuden as-of hizalanmis piyasa paneli.")
    ap.add_argument("--snapshot", type=Path, default=None, help="Anlik goruntu klasoru (varsayilan: en yeni).")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    ap.add_argument("--max-staleness", type=int, default=MAX_STALENESS_DAYS)
    ap.add_argument("--write", action="store_true", help="Paneli ve raporu yaz.")
    ap.add_argument("--force", action="store_true", help="--write ile: mevcut panel farkliysa yedekleyip yaz.")
    args = ap.parse_args(argv)
    if args.force and not args.write:
        ap.error("--force yalnizca --write ile kullanilabilir.")
    if args.max_staleness < 0:
        ap.error("--max-staleness negatif olamaz.")
    return args


def _write_json_with_backup(obj: dict, path: Path):
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print("PIYASA PANELI")
    print("=" * 60)
    try:
        snap = args.snapshot or find_latest_snapshot(args.snapshot_root)
        manifest, series = load_snapshot(snap)
        panel, coverage = build_panel(series, args.max_staleness)
    except (PanelError, bd.DataError) as e:
        print(f"  HATA: {e}\n  Hicbir dosya yazilmadi.")
        return 1

    print(f"  Anlik goruntu: {Path(snap).name} (indirme: {manifest['retrieved_at_utc']})")
    print(f"  Panel: {len(panel)} gun | {panel['Date'].iloc[0].date()} .. {panel['Date'].iloc[-1].date()}")
    print(f"\n  {'seri':8s} {'kapsam%':>8s} {'bayat':>6s} {'basla-once':>10s} {'max yas':>8s}  ham aralik")
    for name, c in coverage.items():
        if c["status"] != "ok":
            print(f"  {name:8s} {c['status']}")
            continue
        print(f"  {name:8s} {c['panel_coverage_pct']:8.2f} {c['rows_dropped_stale']:6d} "
              f"{c['rows_before_series_start']:10d} {str(c['age_days_max_used']):>8s}  "
              f"{c['raw_first_date']} .. {c['raw_last_date']}")

    cmp = compare_with_existing(panel, args.processed_dir / "theoretical_gold.csv")
    print(f"\n  Mevcut theoretical_gold.csv ile karsilastirma: {cmp['status']}")
    for k, v in cmp.items():
        if k != "status":
            print(f"    {k}: {v}")

    report = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_dir": Path(snap).name,
        "snapshot_retrieved_at_utc": manifest["retrieved_at_utc"],
        "snapshot_request": manifest["request"],
        "yfinance_version": manifest.get("yfinance_version"),
        "series_md5": {n: e.get("md5") for n, e in manifest["series"].items()},
        "rows": int(len(panel)),
        "date_range": [str(panel["Date"].iloc[0].date()), str(panel["Date"].iloc[-1].date())],
        "max_staleness_days": args.max_staleness,
        "timestamp_assumption": "Date gunune ait kapanislar, tum piyasalar kapandiktan sonra "
                                "uretilen tahminde bilinir; dogrulanmadi (D4).",
        "coverage": coverage,
        "comparison_with_theoretical_gold": cmp,
    }

    out_path = args.processed_dir / PANEL_FILE
    if not args.write:
        print("\n  Kontrol modu: hicbir dosya yazilmadi (yazmak icin --write).")
        return 0
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["Date"])
        if panels_equal(existing, panel):
            print(f"\n  {out_path.name} zaten ayni; dokunulmadi.")
            return 0
        if not args.force:
            print(f"\n  {out_path.name} mevcut ve farkli; --force olmadan yazilmaz.")
            return 2
    _, _ = _quiet(bd.write_csv, panel, out_path)
    _write_json_with_backup(report, args.processed_dir / REPORT_FILE)
    print(f"\n  Yazildi: {out_path}\n  Rapor: {args.processed_dir / REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
