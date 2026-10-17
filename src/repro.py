"""
Yeniden Uretim Denetimi (salt okunur)
====================================
Kullanim (proje kokunden):
  python -m src.repro                    # hepsi (sizinti gosterimi LSTM nedeniyle 2-3 dakika)
  python -m src.repro --sadece prim      # tek calisma: sizinti, oynaklik, trend, prim, bilesen, guc, geriye_donuk

Her calismayi yerel veriyle BASTAN hesaplar ve depodaki kayitli sonucla sayi sayi karsilastirir.
  Kayitli hicbir dosyanin uzerine YAZMAZ. Geriye donuk sinav (BACK_EXAM_ONS_V1) yeniden CALISTIRILMAZ:
  kayit defterine olay yazilmaz; on kayitli veri ozeti dogrulanir ve ayni hesap bellekte tekrarlanir.
Karsilastirma: anahtarlar ve liste uzunluklari ayni; sayilar goreli 1e-9 icinde; NaN == NaN;
  "created_at_utc" yok sayilir. Kullanilan veri dosyalarinin md5'i raporlanir.
Cikis: 0 = hepsi AYNI; 1 = en az bir FARKLI veya EKSIK.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {"created_at_utc"}
RTOL, ATOL = 1e-9, 1e-12


def _leakage():
    from src.analysis import leakage_demo as ld
    return ld.run_all(ld.load_prices())


def _volatility():
    from src.analysis import leakage_demo as ld
    from src.analysis import volatility_study as vs
    return vs.run_all(ld.load_prices(), vs.load_gvz())


def _trend():
    from src.analysis import leakage_demo as ld
    from src.analysis import trend_study as ts
    from src.data import macro as mc
    return ts.run_all(ld.load_prices(), ts.load_panel(), mc.load())


def _premium():
    from src.analysis import leakage_demo as ld
    from src.analysis import premium_study as ps
    return ps.run_all(ld.load_prices(), ps.load_theoretical())


def _components():
    from src.analysis import leakage_demo as ld
    from src.analysis import vol_components as vc
    from src.analysis import volatility_study as vs
    return vc.run_all(ld.load_prices(), vs.load_gvz(), vc.load_panel())


def _power():
    from src.analysis import power_study as pw
    return pw.run_all(pw.physical_dev_errors(), pw.theoretical_dev_errors())


def _back_exam():
    from src.models import back_exam as be
    pr = json.loads(Path(be.PREREG_PATH).read_text(encoding="utf-8"))
    ds = Path(be.DATA_DIR) / "dataset.csv"
    if be.raw_md5(ds) != pr["data_md5"]["dataset.csv"]:
        raise RuntimeError("Sinav verisi on kayittaki ozetle ayni degil.")
    df = pd.read_csv(ds, parse_dates=["t", "T"], float_precision="round_trip")
    return be.evaluate(be.walk_forward(df), pr["dev_coefficients"])


ARTIFACTS = {
    "sizinti": ("research/exploratory/leakage_demo/summary.json", _leakage),
    "oynaklik": ("research/exploratory/volatility_study/summary.json", _volatility),
    "trend": ("research/exploratory/trend_study/summary.json", _trend),
    "prim": ("research/exploratory/premium_study/summary.json", _premium),
    "bilesen": ("research/exploratory/vol_components/summary.json", _components),
    "guc": ("research/exploratory/power_study/summary.json", _power),
    "geriye_donuk": ("research/back_exam_ons_v1/result.json", _back_exam),
}
DATA_FILES = ["data/processed/physical/physical_gold_daily.csv", "data/processed/history/market_panel.csv",
              "data/processed/macro/macro_monthly.csv", "data/processed/back_exam/dataset.csv"]


def normalize(obj):
    """Hesaplanan sonucu JSON gidis-donusunden gecirir (int anahtarlar metne, numpy sayilari float'a)."""
    return json.loads(json.dumps(obj, default=float))


def compare(expected, actual, path: str = "") -> list:
    """Farklarin listesi (bos = ayni)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = []
        ke, ka = set(expected) - IGNORE, set(actual) - IGNORE
        for k in sorted(ke ^ ka):
            diffs.append(f"{path}/{k}: anahtar {'eksik' if k in ke else 'fazla'}")
        for k in sorted(ke & ka):
            diffs += compare(expected[k], actual[k], f"{path}/{k}")
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: uzunluk {len(expected)} != {len(actual)}"]
        out = []
        for i, (a, b) in enumerate(zip(expected, actual)):
            out += compare(a, b, f"{path}[{i}]")
        return out
    if isinstance(expected, bool) != isinstance(actual, bool):
        return [f"{path}: {expected!r} != {actual!r} (tur farkli)"]
    num = (int, float)
    if isinstance(expected, num) and isinstance(actual, num) and not isinstance(expected, bool) and not isinstance(actual, bool):
        a, b = float(expected), float(actual)
        if math.isnan(a) and math.isnan(b):
            return []
        if math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            return []
        return [f"{path}: {a!r} != {b!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def check(name: str, root: Path = None, artifacts: dict = None) -> dict:
    root, artifacts = Path(root or ROOT), artifacts or ARTIFACTS
    rel, fn = artifacts[name]
    p = root / rel
    if not p.exists():
        return {"name": name, "status": "EKSIK", "detail": f"kayitli sonuc yok: {rel}", "seconds": 0.0}
    expected = json.loads(p.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        actual = normalize(fn())
    except Exception as e:  # veri yok veya hesap coktu
        return {"name": name, "status": "EKSIK", "detail": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
    diffs = compare(expected, actual)
    return {"name": name, "status": "AYNI" if not diffs else "FARKLI", "detail": "; ".join(diffs[:5]),
            "n_diffs": len(diffs), "seconds": time.time() - t0}


def data_md5(root: Path = None) -> dict:
    root = Path(root or ROOT)
    return {f: hashlib.md5((root / f).read_bytes()).hexdigest() if (root / f).exists() else None for f in DATA_FILES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Yeniden uretim denetimi (salt okunur).")
    ap.add_argument("--sadece", choices=list(ARTIFACTS), default=None)
    args = ap.parse_args([] if argv is None else argv)
    print("=" * 78 + "\nYENIDEN URETIM DENETIMI\n" + "=" * 78)
    for f, h in data_md5().items():
        print(f"  veri {f}: {h or 'YOK'}")
    names = [args.sadece] if args.sadece else list(ARTIFACTS)
    results = []
    for n in names:
        r = check(n)
        results.append(r)
        print(f"  [{r['status']:6s}] {n:13s} {r['seconds']:6.1f} s  {r['detail']}")
    bad = [r for r in results if r["status"] != "AYNI"]
    print(f"\n  Sonuc: {len(results)} calisma, {len(results) - len(bad)} ayni, {len(bad)} sorunlu")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
