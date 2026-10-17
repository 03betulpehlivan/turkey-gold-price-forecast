"""
Guc Analizi: 120 Gunluk Canli Degerlendirme Neyi Gorebilir? (KESIF, betimleyici)
================================================================================
Kullanim (proje kokunden):  python -m src.analysis.power_study
Cikti: research/exploratory/power_study/summary.json

Soru: Canli protokoller 120 puanlanmis tahminden sonra BIR KEZ degerlendirilecek (on kayit). Bu orneklemle
  "yarin = bugun"e (RW) gore ne buyuklukte bir iyilesme guvenilir bicimde fark edilebilir? Aralik testleri
  (Kupiec, Christoffersen) ne buyuklukte bir sapmayi yakalar? Nisan'daki sonuc "ayirt edilemedi" cikarsa
  bunun "etki yok" mu yoksa "orneklem yetersiz" mi oldugunu ONCEDEN yorumlayabilmek icin.
Veri (yalnizca gelistirme, hedef < 2023-01-01; kilitli donem OKUNMAZ):
  Fiziki: physical_eval ile ayni gelistirme backtest'i (Bosluk ve RW, 2012..2022 hedefleri).
  Teorik (vekil): tarihsel panelin gunluk teorik gram fiyati; Ortalama getiri (gecmis ortalama) vs RW.
Yontem (nokta): hatalar fiyata bolunur (goreli; 2012-2022 fiyat duzeyi cok degisti). Goreli kayip farki
  d = |hata_model|/P - |hata_RW|/P serisi BIR KEZ, tum donemde ortalama iyilesme x olacak sekilde kaydirilir:
  d' = d - ort(d) - x * ort(|hata_RW|/P). Sonra 120 ardisik gunluk pencereler (adim 10) alinir; pencere
  ortalamalari dogal olarak dalgalanir (orneklem belirsizligi korunur). Guc = DM (Newey-West + HLN, t(n-1),
  iki yonlu) p < alfa VE ortalama fark < 0 olan pencere orani. x = 0'da iki yonlu red orani testin
  boyutunu (yanlis alarm oranini) verir; 0.05'e yakin olmali.
  Alfa: 0.05 ve Holm (3) icin en ihtiyatli 0.05/3 (LIVE_PHYS_V1 uc modeli birlikte sinar).
  En kucuk saptanabilir iyilesme (MDE80): gucun >= 0.80 oldugu en kucuk x.
  Ek: pencerelerde GOZLENEN iyilesme ve gozlenen guc (gelistirme donemindeki gercek etkiyle).
Yontem (aralik): bagimsiz Bernoulli ihlaller (Kupiec) ve Markov kumelenmesi (Christoffersen), n = 120,
  5000 tekrar, tohum 20260101.
Bu bir hipotez testi degildir; on kayitli canli degerlendirmeyi DEGISTIRMEZ.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models import final_exam as fe
from src.models import intervals as iv

ROOT = Path(__file__).resolve().parents[2]
HIST_PANEL = ROOT / "data" / "processed" / "history" / "market_panel.csv"
OUT_DIR = ROOT / "research" / "exploratory" / "power_study"
LOCK = "2023-01-01"
EVAL_N, STEP, ALPHA, HOLM_K = 120, 10, 0.05, 3
EFFECTS = (0, 2, 5, 10, 15, 20, 30, 40)
SEED, REPS = 20260101, 5000
KUPIEC_GRID = {80: (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40), 95: (0.01, 0.02, 0.05, 0.08, 0.10, 0.15)}
CLUSTER_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)      # P(ihlal | onceki gun ihlal); 0.2 = bagimsiz (%80 aralik)


# ---------------------------------------------------------------------------
# Veri (gelistirme)
# ---------------------------------------------------------------------------
def physical_dev_errors() -> pd.DataFrame:
    """Gelistirme backtest'i: Bosluk ve RW mutlak TL hatalari (hedef 2012..2022)."""
    from src.models import physical_eval as pv
    from src.models import physical_history as ph
    from src.models import point_eval as pe
    phys = pd.read_csv(pv.PHYS_PATH, parse_dates=["Date"])
    panel = pd.read_csv(HIST_PANEL, parse_dates=["Date"])
    phys, panel = phys[phys["Date"] < LOCK], panel[panel["Date"] < LOCK]
    df, _ = ph.fresh(pv.build_frame(phys, panel))
    pred = pv.generate(df, start="2012-01-01", end=LOCK)
    out = pd.DataFrame({"date": pd.to_datetime(pred["target_date"]), "model": pe.abs_tl_errors(pred, "Bosluk"),
                        "ref": pe.abs_tl_errors(pred, pv.REFERENCE), "price": pred["price_t"].to_numpy(float)})
    if (out["date"] >= pd.Timestamp(LOCK)).any():
        raise RuntimeError("Kilitli doneme tasma.")
    return out


def theoretical_dev_errors(path: Path = None) -> pd.DataFrame:
    """Vekil: gunluk teorik gram; Ortalama getiri (yalnizca gecmis) vs RW; hedef 2012..2022."""
    df = pd.read_csv(Path(path or HIST_PANEL), usecols=["Date", "theoretical_gram_try"], parse_dates=["Date"])
    df = df[(df["Date"] < LOCK) & (df["theoretical_gram_try"] > 0)].dropna().sort_values("Date").reset_index(drop=True)
    p = df["theoretical_gram_try"].to_numpy()
    r = np.diff(np.log(p))
    mu = np.concatenate([[np.nan], np.cumsum(r)[:-1] / np.arange(1, len(r))])   # mu[i]: r[0..i-1] ortalamasi
    out = pd.DataFrame({"date": df["Date"].iloc[1:].to_numpy(), "model": np.abs(p[:-1] * (np.exp(r) - np.exp(mu))),
                        "ref": np.abs(p[:-1] * (np.exp(r) - 1)), "price": p[:-1]})
    return out[(out["date"] >= "2012-01-01")].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Guc
# ---------------------------------------------------------------------------
def windows(n_total: int, n: int = EVAL_N, step: int = STEP) -> list:
    return [np.arange(s, s + n) for s in range(0, n_total - n + 1, step)]


def power_curve(e_model, e_ref, price, n: int = EVAL_N, step: int = STEP, effects=EFFECTS) -> dict:
    price = np.asarray(price, float)
    em, er = np.asarray(e_model, float) / price, np.asarray(e_ref, float) / price     # goreli hatalar
    d_all = em - er
    wins = windows(len(d_all), n, step)
    curve = []
    for x in effects:
        d_shift = d_all - d_all.mean() - (x / 100.0) * er.mean()                     # tum donem: iyilesme = x
        better = {ALPHA: 0, ALPHA / HOLM_K: 0}
        two_sided = 0
        for w in wins:
            d, base = d_shift[w], er[w]
            p = fe.dm_hln(base + d, base)["p_value"]
            two_sided += int(p < ALPHA)
            for a in better:
                better[a] += int(p < a and d.mean() < 0)
        curve.append({"effect_pct": x, "power": better[ALPHA] / len(wins),
                      "power_holm3": better[ALPHA / HOLM_K] / len(wins), "reject_two_sided": two_sided / len(wins)})
    imp, sig = [], []
    for w in wins:
        d, base = d_all[w], er[w]
        imp.append(float(-100 * d.mean() / base.mean()))
        sig.append(bool(fe.dm_hln(em[w], base)["p_value"] < ALPHA and d.mean() < 0))

    def mde(key):
        ok = [c["effect_pct"] for c in curve if c[key] >= 0.8 and c["effect_pct"] > 0]
        return min(ok) if ok else None

    return {"n_days": int(len(d_all)), "n_windows": len(wins), "curve": curve, "size_at_zero": curve[0]["reject_two_sided"],
            "mde80_pct": mde("power"), "mde80_holm3_pct": mde("power_holm3"),
            "observed": {"full_period_improvement_pct": float(-100 * d_all.mean() / er.mean()),
                         "window_improvement_p10_p50_p90": [float(np.percentile(imp, q)) for q in (10, 50, 90)],
                         "share_windows_significant": float(np.mean(sig))}}


def kupiec_power(n: int = EVAL_N, reps: int = None, seed: int = SEED) -> dict:
    reps = reps or REPS                                                    # calisma aninda okunur
    rng = np.random.default_rng(seed)
    out = {}
    for lvl, grid in KUPIEC_GRID.items():
        nominal = 1 - lvl / 100
        out[str(lvl)] = [{"true_miss_rate": q, "true_coverage_pct": 100 * (1 - q),
                          "power": float(np.mean([iv.kupiec_p((rng.random(n) < q).astype(float), nominal) < ALPHA
                                                  for _ in range(reps)]))} for q in grid]
    return out


def markov_hits(n: int, rate: float, p11: float, rng) -> np.ndarray:
    """Ortalama ihlal orani 'rate', ihlalden sonra ihlal olasiligi p11 olan iki durumlu Markov zinciri."""
    p01 = rate * (1 - p11) / (1 - rate)
    h = np.empty(n)
    h[0] = rng.random() < rate
    for i in range(1, n):
        h[i] = rng.random() < (p11 if h[i - 1] else p01)
    return h


def christoffersen_power(n: int = EVAL_N, reps: int = None, seed: int = SEED, rate: float = 0.2) -> list:
    reps = reps or REPS
    rng = np.random.default_rng(seed)
    out = []
    for p11 in CLUSTER_GRID:
        ps = [iv.christoffersen_p(markov_hits(n, rate, p11, rng)) for _ in range(reps)]
        ps = np.array([p for p in ps if np.isfinite(p)])
        out.append({"p_violation_after_violation": p11, "power": float(np.mean(ps < ALPHA)) if len(ps) else None})
    return out


def run_all(phys: pd.DataFrame, theo: pd.DataFrame) -> dict:
    return {"nature": "KESIF (betimleyici); on kayitli canli degerlendirmeyi DEGISTIRMEZ",
            "config": {"eval_n": EVAL_N, "step": STEP, "alpha": ALPHA, "holm_k": HOLM_K, "effects_pct": list(EFFECTS),
                       "reps": int(REPS), "seed": SEED},
            "physical_bosluk_vs_rw": power_curve(phys["model"], phys["ref"], phys["price"]),
            "theoretical_mean_vs_rw_proxy": power_curve(theo["model"], theo["ref"], theo["price"]),
            "kupiec": kupiec_power(), "christoffersen_80": christoffersen_power()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Guc analizi (kesif).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args([] if argv is None else argv)
    out = Path(args.out or OUT_DIR)
    print("=" * 78 + "\nGUC ANALIZI (KESIF) - 120 gunluk canli degerlendirme\n" + "=" * 78)
    res = run_all(physical_dev_errors(), theoretical_dev_errors())
    res["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for key, label in (("physical_bosluk_vs_rw", "FIZIKI: Bosluk vs RW"), ("theoretical_mean_vs_rw_proxy", "TEORIK (vekil): Ortalama vs RW")):
        r = res[key]
        o = r["observed"]
        print(f"\n  {label} | {r['n_days']} gun, {r['n_windows']} pencere")
        print(pd.DataFrame(r["curve"]).round(3).to_string(index=False))
        print(f"  Boyut (x=0, iki yonlu red): {r['size_at_zero']:.3f}")
        print(f"  MDE80: %{r['mde80_pct']} (alfa 0.05) | %{r['mde80_holm3_pct']} (Holm 3) | gozlenen tum donem %{o['full_period_improvement_pct']:.1f}"
              f" | pencere iyilesmesi p10/p50/p90 {[round(v, 1) for v in o['window_improvement_p10_p50_p90']]}"
              f" | anlamli pencere orani %{100 * o['share_windows_significant']:.0f}")
    print("\n  Kupiec gucu (n 120):")
    for lvl, rows in res["kupiec"].items():
        print(f"    %{lvl}: " + ", ".join(f"kapsam %{r['true_coverage_pct']:.0f} -> {r['power']:.2f}" for r in rows))
    print("  Christoffersen gucu (%80, ihlal orani 0.20): " +
          ", ".join(f"P(ihlal|ihlal) {r['p_violation_after_violation']} -> {r['power']:.2f}" for r in res["christoffersen_80"]))
    print(f"\n  Yazildi: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
